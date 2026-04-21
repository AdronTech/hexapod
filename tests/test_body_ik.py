import math
import pytest

from hexapod.kinematics import leg_fk, COXA_LEN, FEMUR_LEN, TIBIA_LEN
from hexapod.robot.config import Leg
from hexapod.body_ik import (
    BODY_RADIUS,
    corner_pos,
    neutral_foot_body,
    body_ik,
    body_pose_ik,
    BodyPose,
    _body_to_leg,
)

TOL = 1e-4


def assert_close(a, b, tol=TOL, msg=""):
    assert abs(a - b) < tol, f"{msg}: got {a:.6f}, expected {b:.6f}"


def foot_body_from_angles(leg: Leg, tc: float, tf: float, tt: float) -> tuple:
    """Forward-kinematics in leg frame → body frame (for round-trip tests)."""
    a = math.radians({
        Leg.FRONT_RIGHT: -30, Leg.MID_RIGHT: -90, Leg.REAR_RIGHT: -150,
        Leg.REAR_LEFT: 150,   Leg.MID_LEFT:  90,  Leg.FRONT_LEFT:  30,
    }[leg])
    cx = BODY_RADIUS * math.cos(a)
    cy = BODY_RADIUS * math.sin(a)
    xl, yl, zl = leg_fk(tc, tf, tt)
    # Rotate leg-frame point back to body frame
    x_body = cx + xl * math.cos(a) - yl * math.sin(a)
    y_body = cy + xl * math.sin(a) + yl * math.cos(a)
    return x_body, y_body, zl


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestCornerPos:
    def test_front_right_corner(self):
        x, y, z = corner_pos(Leg.FRONT_RIGHT)
        assert_close(x, BODY_RADIUS * math.cos(math.radians(-30)), msg="x")
        assert_close(y, BODY_RADIUS * math.sin(math.radians(-30)), msg="y")
        assert_close(z, 0.0, msg="z")

    def test_mid_left_corner(self):
        x, y, z = corner_pos(Leg.MID_LEFT)
        assert_close(x, 0.0, tol=1e-6, msg="x")
        assert_close(y, BODY_RADIUS, msg="y")

    def test_all_corners_at_body_radius(self):
        for leg in Leg:
            x, y, _ = corner_pos(leg)
            assert_close(math.hypot(x, y), BODY_RADIUS, msg=str(leg))

    def test_opposite_corners_are_antipodal(self):
        for a, b in [(Leg.FRONT_RIGHT, Leg.REAR_LEFT),
                     (Leg.MID_RIGHT, Leg.MID_LEFT),
                     (Leg.REAR_RIGHT, Leg.FRONT_LEFT)]:
            xa, ya, _ = corner_pos(a)
            xb, yb, _ = corner_pos(b)
            assert_close(xa, -xb, msg=f"{a} x vs {b}")
            assert_close(ya, -yb, msg=f"{a} y vs {b}")


class TestNeutralFootBody:
    def test_z_at_neutral(self):
        for leg in Leg:
            _, _, z = neutral_foot_body(leg)
            assert_close(z, -TIBIA_LEN, msg=str(leg))

    def test_neutral_in_leg_frame_is_canonical(self):
        for leg in Leg:
            foot_leg = _body_to_leg(leg, neutral_foot_body(leg))
            assert_close(foot_leg[0], COXA_LEN + FEMUR_LEN, msg=f"{leg} x_leg")
            assert_close(foot_leg[1], 0.0, msg=f"{leg} y_leg")
            assert_close(foot_leg[2], -TIBIA_LEN, msg=f"{leg} z_leg")


# ---------------------------------------------------------------------------
# Layer 1: body_ik
# ---------------------------------------------------------------------------

class TestBodyToLeg:
    def test_neutral_foot_gives_zero_angles(self):
        for leg in Leg:
            foot_body = neutral_foot_body(leg)
            tc, tf, tt = body_ik({leg: foot_body})[leg]
            assert_close(tc, 0.0, tol=1e-4, msg=f"{leg} coxa")
            assert_close(tf, 0.0, tol=1e-4, msg=f"{leg} femur")
            assert_close(tt, 0.0, tol=1e-4, msg=f"{leg} tibia")

    def test_all_legs_neutral_simultaneously(self):
        foot_positions = {leg: neutral_foot_body(leg) for leg in Leg}
        angles = body_ik(foot_positions)
        assert len(angles) == 6
        for leg, (tc, tf, tt) in angles.items():
            assert_close(tc, 0.0, tol=1e-4, msg=f"{leg} coxa")
            assert_close(tf, 0.0, tol=1e-4, msg=f"{leg} femur")
            assert_close(tt, 0.0, tol=1e-4, msg=f"{leg} tibia")

    def test_roundtrip_body_frame(self):
        """FK in body frame → body_ik → angles should match original angles."""
        test_cases = [(10, 0, 0), (20, 5, -10), (15, -3, -12)]
        for leg in Leg:
            for tc_in, tf_in, tt_in in test_cases:
                foot_body = foot_body_from_angles(leg, tc_in, tf_in, tt_in)
                tc_out, tf_out, tt_out = body_ik({leg: foot_body})[leg]
                assert_close(tc_out, tc_in, tol=1e-4, msg=f"{leg} coxa")
                assert_close(tf_out, tf_in, tol=1e-4, msg=f"{leg} femur")
                assert_close(tt_out, tt_in, tol=1e-4, msg=f"{leg} tibia")

    def test_partial_legs(self):
        """body_ik with a subset of legs only returns those legs."""
        subset = {Leg.FRONT_RIGHT: neutral_foot_body(Leg.FRONT_RIGHT),
                  Leg.REAR_LEFT:   neutral_foot_body(Leg.REAR_LEFT)}
        angles = body_ik(subset)
        assert set(angles.keys()) == {Leg.FRONT_RIGHT, Leg.REAR_LEFT}


# ---------------------------------------------------------------------------
# Layer 2: body_pose_ik
# ---------------------------------------------------------------------------

class TestBodyPoseIK:
    def _neutral_world_feet(self, height: float = 0.0) -> dict[Leg, tuple]:
        """Neutral feet planted on the ground (world z=0) with body at *height*."""
        pose = BodyPose(z=height)
        # neutral feet in body frame shifted to world by body z offset
        return {
            leg: (
                neutral_foot_body(leg)[0],
                neutral_foot_body(leg)[1],
                neutral_foot_body(leg)[2] + height,  # ground level
            )
            for leg in Leg
        }

    def test_zero_pose_matches_body_ik(self):
        """At the zero pose, body_pose_ik == body_ik with feet in body frame."""
        feet_world = {leg: neutral_foot_body(leg) for leg in Leg}
        pose = BodyPose()
        angles_pose = body_pose_ik(pose, feet_world)
        angles_body = body_ik(feet_world)
        for leg in Leg:
            for a, b in zip(angles_pose[leg], angles_body[leg]):
                assert_close(a, b, tol=1e-8, msg=str(leg))

    def test_pure_z_translation_neutral_angles(self):
        """Lifting body straight up; feet stay on ground → angles change uniformly."""
        height = 15.0
        feet_world = self._neutral_world_feet(height)
        pose = BodyPose(z=height)
        angles = body_pose_ik(pose, feet_world)
        # All legs see the same geometry: neutral (x, 0, -tibia) in leg frame
        for leg in Leg:
            tc, tf, tt = angles[leg]
            assert_close(tc, 0.0, tol=1e-4, msg=f"{leg} coxa at height")

    def test_yaw_corotation_is_identity(self):
        """
        If both the body AND all feet rotate by the same angle, the joint
        angles must be identical to the zero-yaw case.  This tests that yaw
        is handled consistently: rotating the whole robot changes nothing.
        """
        height = 15.0
        yaw = 37.0  # arbitrary non-trivial angle

        feet_world_0 = self._neutral_world_feet(height)

        # Rotate every foot by `yaw` degrees around the world Z axis
        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        feet_world_rot = {
            leg: (
                cy * fx - sy * fy,
                sy * fx + cy * fy,
                fz,
            )
            for leg, (fx, fy, fz) in feet_world_0.items()
        }

        pose_0   = BodyPose(z=height, yaw=0.0)
        pose_rot = BodyPose(z=height, yaw=yaw)

        angles_0   = body_pose_ik(pose_0,   feet_world_0)
        angles_rot = body_pose_ik(pose_rot, feet_world_rot)

        for leg in Leg:
            for a, b in zip(angles_0[leg], angles_rot[leg]):
                assert_close(a, b, tol=1e-6, msg=f"{leg}")

    def test_roll_raises_one_side(self):
        """
        Positive roll = left side of body rises.  Feet stay on the ground.
        From the body's perspective, the RIGHT feet appear HIGHER (body's right
        went down), so right legs need to angle the femur upward: tf increases.
        """
        height = 15.0
        feet_world = self._neutral_world_feet(height)

        pose_flat   = BodyPose(z=height)
        pose_rolled = BodyPose(z=height, roll=15.0)  # left side rises

        angles_flat   = body_pose_ik(pose_flat,   feet_world)
        angles_rolled = body_pose_ik(pose_rolled, feet_world)

        tf_flat   = angles_flat[Leg.MID_RIGHT][1]
        tf_rolled = angles_rolled[Leg.MID_RIGHT][1]
        assert tf_rolled > tf_flat, "right femur should angle up when body rolls left"

    def test_pitch_nose_up(self):
        """
        Positive pitch = nose rises.  Feet stay on the ground.
        From the body's perspective, FRONT feet appear HIGHER (body nose went
        up), so front femurs angle upward (tf increases).
        REAR feet appear LOWER, so rear femurs angle downward (tf decreases).
        """
        height = 15.0
        feet_world = self._neutral_world_feet(height)

        pose_flat    = BodyPose(z=height)
        pose_pitched = BodyPose(z=height, pitch=15.0)  # nose up

        angles_flat    = body_pose_ik(pose_flat,    feet_world)
        angles_pitched = body_pose_ik(pose_pitched, feet_world)

        tf_front_flat    = angles_flat[Leg.FRONT_RIGHT][1]
        tf_front_pitched = angles_pitched[Leg.FRONT_RIGHT][1]
        tf_rear_flat     = angles_flat[Leg.REAR_RIGHT][1]
        tf_rear_pitched  = angles_pitched[Leg.REAR_RIGHT][1]

        assert tf_front_pitched > tf_front_flat, "front femur should angle up when nose rises"
        assert tf_rear_pitched  < tf_rear_flat,  "rear femur should angle down when nose rises"
