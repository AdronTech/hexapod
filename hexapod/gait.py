"""
Gait engines: tripod, ripple, wave, and free, with Bezier swing trajectories.

Usage:
    gait = TripodGait(initial_pose, initial_feet)   # or RippleGait / WaveGait / FreeGait
    # each control tick:
    body_pose, foot_world = gait.step(vx, vy, omega_deg, dt)
    ticks = _compute_ticks(body_pose, foot_world, limits)

Gaits
-----
Tripod (2 groups of 3):  fastest, 50% duty factor, 3 legs always grounded.
Ripple (3 groups of 2):  medium,  67% duty factor, 4 legs always grounded.
Wave   (6 groups of 1):  slowest, 83% duty factor, 5 legs always grounded.
Free   (event-driven):   each leg steps when its foot drifts from neutral.

Bezier swing:
    Each swing foot travels along a cubic Bezier curve from its lift-off
    position to its landing target.  Two inner control points sit directly
    above the end-points at `step_height`, producing a smooth arch.

Turntable targets:
    The swing foot's landing target is placed half a stance-stride ahead of
    the leg's neutral so that mid-stance coincides with neutral (maximum
    stability window).
"""

import math
from dataclasses import replace

from hexapod.body_ik import BodyPose, corner_pos
from hexapod.kinematics import COXA_LEN, FEMUR_LEN
from hexapod.robot.config import Leg

# ---------------------------------------------------------------------------
# Phase tables
# ---------------------------------------------------------------------------

# Tripod: two groups of 3, 180° apart
_TRIPOD_PHASES: dict[Leg, float] = {
    Leg.FRONT_RIGHT: 0.0,
    Leg.REAR_RIGHT:  0.0,
    Leg.MID_LEFT:    0.0,
    Leg.MID_RIGHT:   0.5,
    Leg.REAR_LEFT:   0.5,
    Leg.FRONT_LEFT:  0.5,
}

# Ripple: 3 groups of 2 opposite legs, 120° apart
_RIPPLE_PHASES: dict[Leg, float] = {
    Leg.FRONT_RIGHT: 0 / 3,
    Leg.REAR_LEFT:   0 / 3,
    Leg.MID_RIGHT:   1 / 3,
    Leg.MID_LEFT:    1 / 3,
    Leg.REAR_RIGHT:  2 / 3,
    Leg.FRONT_LEFT:  2 / 3,
}

# Wave: 6 legs one at a time, alternating sides for continuous stability
_WAVE_PHASES: dict[Leg, float] = {
    Leg.FRONT_RIGHT: 0 / 6,
    Leg.MID_LEFT:    1 / 6,
    Leg.REAR_RIGHT:  2 / 6,
    Leg.FRONT_LEFT:  3 / 6,
    Leg.MID_RIGHT:   4 / 6,
    Leg.REAR_LEFT:   5 / 6,
}

# ---------------------------------------------------------------------------
# Free-gait stability structures
# ---------------------------------------------------------------------------

# Adjacency ring for the free gait stability guard
_LEG_RING: tuple[Leg, ...] = (
    Leg.FRONT_RIGHT, Leg.MID_RIGHT, Leg.REAR_RIGHT,
    Leg.REAR_LEFT,   Leg.MID_LEFT,  Leg.FRONT_LEFT,
)
_ADJACENT: dict[Leg, frozenset] = {
    leg: frozenset({_LEG_RING[(i - 1) % 6], _LEG_RING[(i + 1) % 6]})
    for i, leg in enumerate(_LEG_RING)
}

_NEUTRAL_REACH = COXA_LEN + FEMUR_LEN   # 17.4 cm from coxa pivot to neutral foot

Foot3D = tuple[float, float, float]
FootMap = dict[Leg, Foot3D]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _cubic_bezier(p0: Foot3D, p1: Foot3D, p2: Foot3D, p3: Foot3D, t: float) -> Foot3D:
    mt = 1.0 - t
    return (
        mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0],
        mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1],
        mt**3*p0[2] + 3*mt**2*t*p1[2] + 3*mt*t**2*p2[2] + t**3*p3[2],
    )


def _rotate2d(
    px: float, py: float,
    cx: float, cy: float,
    angle_rad: float,
) -> tuple[float, float]:
    """Rotate point (px, py) around centre (cx, cy) by angle_rad."""
    dx, dy = px - cx, py - cy
    co, so = math.cos(angle_rad), math.sin(angle_rad)
    return cx + dx * co - dy * so, cy + dx * so + dy * co


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _GaitBase:
    """Common state and helpers shared by all gait engines."""

    def __init__(
        self,
        initial_pose: BodyPose,
        initial_feet: FootMap,
        *,
        step_height: float,
        neutral_reach: float,
    ) -> None:
        self.step_height   = step_height
        self.neutral_reach = neutral_reach
        self._body       = initial_pose
        self._foot_world = dict(initial_feet)

    # --- body pose accessors ---

    @property
    def body(self) -> BodyPose:
        return self._body

    @body.setter
    def body(self, b: BodyPose) -> None:
        self._body = b

    @property
    def body_z(self) -> float:
        return self._body.z

    @body_z.setter
    def body_z(self, z: float) -> None:
        self._body = replace(self._body, z=z)

    @property
    def body_roll(self) -> float:
        return self._body.roll

    @body_roll.setter
    def body_roll(self, roll: float) -> None:
        self._body = replace(self._body, roll=roll)

    @property
    def body_pitch(self) -> float:
        return self._body.pitch

    @body_pitch.setter
    def body_pitch(self, pitch: float) -> None:
        self._body = replace(self._body, pitch=pitch)

    @property
    def feet(self) -> FootMap:
        return dict(self._foot_world)

    # --- shared gait helpers ---

    def _advance_body(self, vx: float, vy: float, omega_deg: float, dt: float) -> None:
        self._body = replace(
            self._body,
            x   = self._body.x   + vx        * dt,
            y   = self._body.y   + vy        * dt,
            yaw = self._body.yaw + omega_deg * dt,
        )

    def _neutral_foot_world(self, leg: Leg) -> Foot3D:
        """World-frame neutral foot position given the current body pose."""
        yaw = math.radians(self._body.yaw)
        cx, cy, _ = corner_pos(leg)
        corner_angle  = math.atan2(cy, cx)
        corner_radius = math.hypot(cx, cy)
        world_corner_angle = yaw + corner_angle
        wcx = self._body.x + corner_radius * math.cos(world_corner_angle)
        wcy = self._body.y + corner_radius * math.sin(world_corner_angle)
        return (
            wcx + self.neutral_reach * math.cos(world_corner_angle),
            wcy + self.neutral_reach * math.sin(world_corner_angle),
            0.0,
        )

    def _swing_arc(self, p0: Foot3D, p3: Foot3D, t: float) -> Foot3D:
        """
        Cubic Bezier from p0 to p3 with a smooth arch.

        Control points sit directly above the endpoints at step_height.
        The cubic Bezier peaks at 0.75 × control height at t = 0.5, so the
        control height is scaled by 4/3 to hit the exact step_height.
        """
        h  = self.step_height * (4.0 / 3.0)
        p1 = (p0[0], p0[1], p0[2] + h)
        p2 = (p3[0], p3[1], p3[2] + h)
        return _cubic_bezier(p0, p1, p2, p3, t)


# ---------------------------------------------------------------------------
# Phase-based gaits (tripod, ripple, wave)
# ---------------------------------------------------------------------------

class _PhasedGait(_GaitBase):
    """
    Generic phase-based gait engine.

    Each leg has a phase offset (fraction 0..1 of the full cycle at which its
    swing window opens).  All legs share the same swing_fraction of the cycle.
    """

    def __init__(
        self,
        initial_pose: BodyPose,
        initial_feet: FootMap,
        *,
        cycle_time: float,
        swing_fraction: float,
        phase_offsets: dict,
        step_height: float = 4.0,
        neutral_reach: float = _NEUTRAL_REACH,
    ) -> None:
        super().__init__(initial_pose, initial_feet,
                         step_height=step_height, neutral_reach=neutral_reach)
        self._cycle_time = cycle_time
        self._swing_frac = swing_fraction
        self._step_time  = cycle_time * swing_fraction
        self._offsets    = phase_offsets
        self._clock      = 0.0
        self._swing_start:   FootMap = {}
        self._swing_target:  FootMap = {}
        self._leg_swinging: dict[Leg, bool] = {leg: False for leg in Leg}

    @property
    def step_time(self) -> float:
        return self._step_time

    @step_time.setter
    def step_time(self, t: float) -> None:
        self._step_time  = t
        self._cycle_time = t / self._swing_frac

    def step(
        self,
        vx: float,
        vy: float,
        omega_deg: float,
        dt: float,
    ) -> tuple[BodyPose, FootMap]:
        """
        Advance gait by *dt* seconds.

        Parameters
        ----------
        vx, vy:
            Desired body velocity in the **world** frame (cm/s).
        omega_deg:
            Desired yaw rate in degrees/s.  Positive = CCW from above.
        dt:
            Time since last call (seconds).

        Returns
        -------
        (body_pose, foot_world_positions) — feed directly into body_pose_ik().
        """
        self._clock = (self._clock + dt) % self._cycle_time
        self._advance_body(vx, vy, omega_deg, dt)

        phase = self._clock / self._cycle_time
        for leg in Leg:
            rel          = (phase - self._offsets[leg]) % 1.0
            now_swinging = rel < self._swing_frac
            was_swinging = self._leg_swinging[leg]

            if now_swinging and not was_swinging:
                self._swing_start[leg]  = self._foot_world[leg]
                self._swing_target[leg] = self._swing_target_for(leg, vx, vy, omega_deg)

            self._leg_swinging[leg] = now_swinging

            if now_swinging:
                p0 = self._swing_start.get(leg)
                p3 = self._swing_target.get(leg)
                if p0 is not None and p3 is not None:
                    self._foot_world[leg] = self._swing_arc(p0, p3, rel / self._swing_frac)

        return self._body, dict(self._foot_world)

    def _swing_target_for(
        self, leg: Leg, vx: float, vy: float, omega_deg: float
    ) -> Foot3D:
        """
        Landing target for a swing foot — the turntable calculation.

        The foot is placed half a stance-stride *ahead* of the current neutral
        so that mid-stance coincides with neutral (maximum stability window).
        """
        nx, ny, nz = self._neutral_foot_world(leg)
        # half the stance duration = time the foot is on the ground / 2
        half_t     = (1.0 - self._swing_frac) * self._cycle_time * 0.5
        half_omega = math.radians(omega_deg * half_t)
        rx, ry     = _rotate2d(nx, ny, self._body.x, self._body.y, half_omega)
        return (rx + vx * half_t, ry + vy * half_t, nz)


class TripodGait(_PhasedGait):
    """
    Tripod gait: two groups of 3 diagonal legs swing alternately.
    Duty factor 1/2 — always three legs on the ground.  Fastest gait.
    """

    def __init__(
        self,
        initial_pose: BodyPose,
        initial_feet: FootMap,
        *,
        step_time: float = 0.40,
        step_height: float = 4.0,
        neutral_reach: float = _NEUTRAL_REACH,
    ) -> None:
        super().__init__(
            initial_pose, initial_feet,
            cycle_time     = step_time * 2,
            swing_fraction = 0.5,
            phase_offsets  = _TRIPOD_PHASES,
            step_height    = step_height,
            neutral_reach  = neutral_reach,
        )


class RippleGait(_PhasedGait):
    """
    Ripple gait: two opposite legs swing together in three alternating groups.
    Duty factor 2/3 — always four legs on the ground.
    """

    def __init__(
        self,
        initial_pose: BodyPose,
        initial_feet: FootMap,
        *,
        step_time: float = 0.40,
        step_height: float = 4.0,
        neutral_reach: float = _NEUTRAL_REACH,
    ) -> None:
        super().__init__(
            initial_pose, initial_feet,
            cycle_time     = step_time * 3,
            swing_fraction = 1.0 / 3.0,
            phase_offsets  = _RIPPLE_PHASES,
            step_height    = step_height,
            neutral_reach  = neutral_reach,
        )


class WaveGait(_PhasedGait):
    """
    Wave gait: one leg swings at a time in an alternating-sides sequence.
    Duty factor 5/6 — always five legs on the ground.  Slowest, most stable.
    """

    def __init__(
        self,
        initial_pose: BodyPose,
        initial_feet: FootMap,
        *,
        step_time: float = 0.40,
        step_height: float = 4.0,
        neutral_reach: float = _NEUTRAL_REACH,
    ) -> None:
        super().__init__(
            initial_pose, initial_feet,
            cycle_time     = step_time * 6,
            swing_fraction = 1.0 / 6.0,
            phase_offsets  = _WAVE_PHASES,
            step_height    = step_height,
            neutral_reach  = neutral_reach,
        )


# ---------------------------------------------------------------------------
# Free (event-driven) gait
# ---------------------------------------------------------------------------

class FreeGait(_GaitBase):
    """
    Free gait: each leg steps independently when its foot drifts more than
    `step_threshold` cm from its neutral position in the current body frame.

    At most three non-adjacent legs swing simultaneously, ensuring at least
    three legs remain grounded. Adjacent legs are never swung together unless
    a foot exceeds `step_emergency_threshold`.
    """

    def __init__(
        self,
        initial_pose: BodyPose,
        initial_feet: FootMap,
        *,
        step_time: float = 0.40,
        step_height: float = 4.0,
        neutral_reach: float = _NEUTRAL_REACH,
        step_threshold: float = 5.0,
        step_reach_max: float = COXA_LEN + FEMUR_LEN + 8.0,
        step_reach_min: float = COXA_LEN + 2.0,
        step_emergency_threshold: float = 6.0,
    ) -> None:
        super().__init__(initial_pose, initial_feet,
                         step_height=step_height, neutral_reach=neutral_reach)
        self.step_time                = step_time
        self.step_threshold           = step_threshold
        self.step_emergency_threshold = step_emergency_threshold
        self._step_reach_max          = step_reach_max
        self._step_reach_min          = step_reach_min
        self._swinging:     dict[Leg, bool]  = {leg: False for leg in Leg}
        self._swing_t:      dict[Leg, float] = {leg: 0.0   for leg in Leg}
        self._swing_start:  FootMap = {}
        self._swing_target: FootMap = {}

    def step(
        self,
        vx: float,
        vy: float,
        omega_deg: float,
        dt: float,
    ) -> tuple[BodyPose, FootMap]:
        """Advance gait by *dt* seconds.  See _PhasedGait.step() for parameter docs."""
        self._advance_body(vx, vy, omega_deg, dt)

        # Advance in-flight swings
        for leg in Leg:
            if not self._swinging[leg]:
                continue
            t = min(1.0, self._swing_t[leg] + dt / self.step_time)
            self._swing_t[leg] = t
            self._foot_world[leg] = self._swing_arc(
                self._swing_start[leg], self._swing_target[leg], t
            )
            if t >= 1.0:
                self._swinging[leg]   = False
                self._foot_world[leg] = self._swing_target[leg]

        # Collect grounded legs that need to step, largest error first
        candidates: list[tuple[float, Leg]] = [
            (self._foot_error(leg), leg)
            for leg in Leg
            if not self._swinging[leg] and self._foot_error(leg) > self.step_threshold
        ]
        candidates.sort(reverse=True)

        swing_count = sum(1 for s in self._swinging.values() if s)
        for err, leg in candidates:
            if swing_count >= 3:
                break
            emergency = err > self.step_emergency_threshold
            if not emergency and any(self._swinging[adj] for adj in _ADJACENT[leg]):
                continue
            self._swing_start[leg]  = self._foot_world[leg]
            self._swing_target[leg] = self._swing_target_for(leg, vx, vy, omega_deg)
            self._swing_t[leg]      = 0.0
            self._swinging[leg]     = True
            swing_count += 1

        return self._body, dict(self._foot_world)

    def _foot_error(self, leg: Leg) -> float:
        nx, ny, _ = self._neutral_foot_world(leg)
        fx, fy    = self._foot_world[leg][0], self._foot_world[leg][1]
        return math.hypot(fx - nx, fy - ny)

    def _swing_target_for(
        self, leg: Leg, vx: float, vy: float, omega_deg: float
    ) -> Foot3D:
        nx, ny, nz = self._neutral_foot_world(leg)
        half_t     = self.step_time * 0.5
        half_omega = math.radians(omega_deg * half_t)
        rx, ry     = _rotate2d(nx, ny, self._body.x, self._body.y, half_omega)
        tx, ty     = rx + vx * half_t, ry + vy * half_t

        # Clamp to reachable radius from the coxa pivot
        cx, cy, _ = corner_pos(leg)
        wa = math.radians(self._body.yaw) + math.atan2(cy, cx)
        cr = math.hypot(cx, cy)
        pivot_x = self._body.x + cr * math.cos(wa)
        pivot_y = self._body.y + cr * math.sin(wa)
        dx, dy = tx - pivot_x, ty - pivot_y
        dist = math.hypot(dx, dy)
        if dist > 1e-9:
            if dist > self._step_reach_max:
                s = self._step_reach_max / dist
                tx, ty = pivot_x + dx * s, pivot_y + dy * s
            elif dist < self._step_reach_min:
                s = self._step_reach_min / dist
                tx, ty = pivot_x + dx * s, pivot_y + dy * s
        return (tx, ty, nz)
