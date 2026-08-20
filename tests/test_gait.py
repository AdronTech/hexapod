"""
Gait invariants — the properties a walking robot has to keep, at any speed.

The free gait is event-driven, so most of these are statements about what must
never happen over a run rather than about a single call.
"""

import math

import pytest

from hexapod.body_ik import BodyPose, body_pose_ik, neutral_foot_body
from hexapod.control.state import (
    DT,
    FREE_STEP_EMERGENCY,
    FREE_STEP_THRESHOLD,
    REACH_MAX,
    REACH_MIN,
    STAND_HEIGHT,
)
from hexapod.gait import (
    _ADJACENT,
    _LANDING_LEAD,
    _NEUTRAL_REACH,
    _SUPPORT_MARGIN_MIN,
    FreeGait,
    RippleGait,
    TripodGait,
    WaveGait,
)
from hexapod.kinematics import IKError
from hexapod.robot.config import Leg
from hexapod.robot.soft_limits import SoftLimitError, SoftLimits
from hexapod.support import support_margin

# (vx, vy, omega) commands in the body frame, covering translation, rotation
# and the curving paths that combine them.
COMMANDS = [
    (5.0, 0.0, 0.0),
    (15.0, 0.0, 0.0),
    (30.0, 0.0, 0.0),
    (0.0, 15.0, 0.0),
    (-15.0, 0.0, 0.0),
    (0.0, 0.0, 60.0),
    (0.0, 0.0, -120.0),
    (15.0, 0.0, 60.0),
    (30.0, 0.0, 120.0),
    (10.0, 10.0, -30.0),
]


def neutral_feet() -> dict:
    return {
        leg: (neutral_foot_body(leg)[0], neutral_foot_body(leg)[1], 0.0) for leg in Leg
    }


def make_free(**kw) -> FreeGait:
    kw.setdefault("step_time", 0.40)
    kw.setdefault("step_threshold", FREE_STEP_THRESHOLD)
    return FreeGait(
        BodyPose(z=STAND_HEIGHT),
        neutral_feet(),
        neutral_reach=_NEUTRAL_REACH,
        step_height=4.0,
        step_emergency_threshold=FREE_STEP_EMERGENCY,
        step_reach_max=REACH_MAX,
        step_reach_min=REACH_MIN,
        **kw,
    )


def drive(gait, vx: float, vy: float, omega: float, secs: float = 12.0):
    """Run a gait with a body-frame command, yielding state each tick."""
    for _ in range(int(secs / DT)):
        yaw = math.radians(gait.body.yaw)
        wvx = vx * math.cos(yaw) - vy * math.sin(yaw)
        wvy = vx * math.sin(yaw) + vy * math.cos(yaw)
        pose, feet = gait.step(wvx, wvy, omega, DT)
        yield pose, feet


# ---------------------------------------------------------------------------
# Swing targeting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vx,vy,omega", COMMANDS)
def test_free_gait_feet_land_close_to_neutral(vx, vy, omega):
    """
    A foot must touch down near the neutral it has *at touchdown*.

    Aiming at the neutral the leg left behind lands it a full swing's travel
    short, which is what made the gait re-lift every leg the moment it landed.
    """
    gait = make_free()
    swinging = {leg: False for leg in Leg}
    errors = []
    for _ in drive(gait, vx, vy, omega):
        for leg in Leg:
            if swinging[leg] and not gait._swinging[leg]:
                errors.append(gait._foot_error(leg))
            swinging[leg] = gait._swinging[leg]
    assert errors
    # The lead is deliberate: the foot reaches out ahead of neutral so it can
    # drift back through it.  It must stay strictly inside the trigger.
    assert max(errors) <= gait.step_threshold * (1.0 + 1e-6)
    assert max(errors) == pytest.approx(_LANDING_LEAD * gait.step_threshold, abs=0.3)


@pytest.mark.parametrize("vx,vy,omega", COMMANDS)
def test_free_gait_never_relifts_without_a_stance(vx, vy, omega):
    """Every swing lasts one step_time — never two back to back."""
    gait = make_free()
    swinging = {leg: False for leg in Leg}
    since = {leg: 0 for leg in Leg}
    for i, _ in enumerate(drive(gait, vx, vy, omega)):
        for leg in Leg:
            if gait._swinging[leg] and not swinging[leg]:
                since[leg] = i
            elif swinging[leg] and not gait._swinging[leg]:
                duration = (i - since[leg]) * DT
                assert duration <= gait.step_time * 1.5, f"{leg.name} never landed"
            swinging[leg] = gait._swinging[leg]


def test_free_gait_stance_lengthens_as_the_robot_slows():
    """
    Event-driven means step rate follows speed.  The old targeting saturated at
    the maximum step rate for every speed above walking pace.
    """
    lifts = {}
    for speed in (4.0, 8.0, 14.0):
        gait = make_free()
        swinging = {leg: False for leg in Leg}
        n = 0
        for _ in drive(gait, speed, 0.0, 0.0, secs=20.0):
            for leg in Leg:
                if gait._swinging[leg] and not swinging[leg]:
                    n += 1
                swinging[leg] = gait._swinging[leg]
        lifts[speed] = n
    assert lifts[4.0] < lifts[8.0] < lifts[14.0]


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vx,vy,omega", COMMANDS)
def test_free_gait_keeps_at_least_three_feet_down(vx, vy, omega):
    gait = make_free()
    for _ in drive(gait, vx, vy, omega):
        assert sum(1 for s in gait._swinging.values() if s) <= 3


@pytest.mark.parametrize("vx,vy,omega", COMMANDS)
def test_free_gait_keeps_the_body_over_its_support(vx, vy, omega):
    """
    Three feet down is not enough if they are three feet of the same side.
    The body centre has to stay inside the polygon they span.
    """
    gait = make_free()
    worst = math.inf
    for pose, feet in drive(gait, vx, vy, omega):
        grounded = [
            (feet[leg][0] - pose.x, feet[leg][1] - pose.y)
            for leg in Leg
            if not gait._swinging[leg]
        ]
        worst = min(worst, support_margin(grounded))
    assert worst > 0.0, f"body centre left the support polygon ({worst:.2f} cm)"


@pytest.mark.parametrize("vx,vy,omega", COMMANDS)
def test_free_gait_only_breaks_adjacency_under_emergency(vx, vy, omega):
    gait = make_free()
    swinging = {leg: False for leg in Leg}
    for _ in drive(gait, vx, vy, omega):
        for leg in Leg:
            lifting = gait._swinging[leg] and not swinging[leg]
            if lifting and any(gait._swinging[adj] for adj in _ADJACENT[leg]):
                assert gait._foot_error(leg) > gait.step_threshold
            swinging[leg] = gait._swinging[leg]


@pytest.mark.parametrize("vx,vy,omega", COMMANDS)
def test_free_gait_stays_within_reach(vx, vy, omega):
    """Every commanded pose must be solvable and inside the soft limits."""
    limits = SoftLimits.load()
    assert limits is not None
    gait = make_free()
    for pose, feet in drive(gait, vx, vy, omega):
        try:
            angles = body_pose_ik(pose, feet)
        except IKError as e:  # pragma: no cover - failure path
            pytest.fail(f"IK failed at {vx},{vy},{omega}: {e}")
        for leg, (tc, tf, tt) in angles.items():
            try:
                limits.check(tc, tf, tt)
            except SoftLimitError as e:  # pragma: no cover - failure path
                pytest.fail(f"{leg.name} outside soft limits: {e}")


# ---------------------------------------------------------------------------
# Command limiting
# ---------------------------------------------------------------------------


def test_command_is_scaled_to_what_the_legs_can_service():
    gait = make_free()
    # Well inside the budget: untouched.
    gait.step(4.0, 0.0, 0.0, DT)
    assert gait.command_scale == 1.0
    # Far beyond it: scaled back.
    gait.step(0.0, 0.0, 120.0, DT)
    assert 0.0 < gait.command_scale < 1.0


def test_command_limit_preserves_direction():
    gait = make_free()
    before = gait.body
    gait.step(30.0, 30.0, 0.0, DT)
    moved = (gait.body.x - before.x, gait.body.y - before.y)
    assert moved[0] == pytest.approx(moved[1])
    assert gait.command_scale < 1.0


def test_raising_the_threshold_buys_speed():
    """
    The budget is (1 + lead) * threshold / step_time — the two knobs the web UI
    exposes are exactly the ones that trade stride length for step rate.
    """
    slow = make_free(step_threshold=3.0)
    fast = make_free(step_threshold=6.0)
    slow.step(30.0, 0.0, 0.0, DT)
    fast.step(30.0, 0.0, 0.0, DT)
    assert fast.command_scale > slow.command_scale

    quick = make_free(step_threshold=3.0, step_time=0.2)
    quick.step(30.0, 0.0, 0.0, DT)
    assert quick.command_scale > slow.command_scale


def test_standing_still_leaves_the_feet_alone():
    gait = make_free()
    for _ in drive(gait, 0.0, 0.0, 0.0, secs=5.0):
        assert not any(gait._swinging.values())
    assert gait.command_scale == 1.0


def test_support_margin_floor_is_respected_at_the_moment_of_lift():
    gait = make_free()
    for _ in drive(gait, 15.0, 0.0, 60.0):
        pass
    # The guard is predictive, so the realised margin can dip below the floor,
    # but never to the point of tipping.
    assert _SUPPORT_MARGIN_MIN > 0.0


# ---------------------------------------------------------------------------
# Phased gaits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [TripodGait, RippleGait, WaveGait])
def test_phased_gait_foot_excursion_is_centred_on_neutral(cls):
    """
    Mid-stance should coincide with neutral: the foot reaches as far ahead of
    neutral at touchdown as it trails behind it at lift-off.  Targeting the
    lift-off neutral biased the whole excursion backwards by a swing's travel.
    """
    gait = cls(BodyPose(z=STAND_HEIGHT), neutral_feet(), step_time=0.4)
    swinging = {leg: False for leg in Leg}
    at_land: list[float] = []
    at_lift: list[float] = []
    for _ in drive(gait, 12.0, 0.0, 0.0, secs=12.0):
        for leg in Leg:
            now = gait._leg_swinging[leg]
            if swinging[leg] and not now:
                at_land.append(gait._foot_error(leg))
            elif now and not swinging[leg]:
                at_lift.append(gait._foot_error(leg))
            swinging[leg] = now
    # Discard the first cycle, which starts from a standing pose.
    land = sum(at_land[6:]) / len(at_land[6:])
    lift = sum(at_lift[6:]) / len(at_lift[6:])
    assert land == pytest.approx(lift, rel=0.15)


def _arc(step_height: float = 4.0, n: int = 4000):
    """Sample the swing arc from (0,0,0) to (6,0,0), the tripod stride at 15 cm/s."""
    gait = TripodGait(
        BodyPose(z=STAND_HEIGHT), neutral_feet(), step_time=0.4, step_height=step_height
    )
    return [gait._swing_arc((0.0, 0.0, 0.0), (6.0, 0.0, 0.0), i / n) for i in range(n + 1)]


@pytest.mark.parametrize("step_height", [1.0, 4.0, 12.0])
def test_swing_arc_peaks_at_step_height(step_height):
    """The 1.6 control lift has to land the apex on step_height exactly."""
    pts = _arc(step_height)
    assert max(p[2] for p in pts) == pytest.approx(step_height, rel=1e-6)


def test_swing_arc_lands_and_lifts_with_no_speed():
    """
    The doubled end control points are the whole point.  The cubic arc this
    replaced left and met the ground at 3 × its control height per step_time —
    40 cm/s with the defaults, driven straight down into the floor.
    """
    step_time = 0.4
    pts = _arc()
    n = len(pts) - 1
    for i, name in ((1, "lift-off"), (n - 1, "touchdown")):
        a, b = pts[i - 1], pts[i + 1]
        speed = math.hypot(b[0] - a[0], b[2] - a[2]) * n / 2 / step_time
        assert speed < 0.5, f"{name} at {speed:.2f} cm/s"


def test_swing_arc_is_symmetric_and_stays_above_ground():
    pts = _arc()
    n = len(pts) - 1
    assert min(p[2] for p in pts) >= 0.0
    for i in range(0, n // 2, 37):
        mirror = pts[n - i]
        assert pts[i][2] == pytest.approx(mirror[2], abs=1e-9)
        assert pts[i][0] == pytest.approx(6.0 - mirror[0], abs=1e-9)
