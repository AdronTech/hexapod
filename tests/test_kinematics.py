import math
import pytest
from hexapod.kinematics import leg_fk, leg_ik, angle_to_tick, tick_to_angle, IKError

COXA, FEMUR, TIBIA = 6.4, 11.0, 15.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_close(a, b, tol=1e-6, msg=""):
    assert abs(a - b) < tol, f"{msg}: {a} != {b} (tol={tol})"


def fk_ik_roundtrip(x, y, z, tol=1e-6):
    tc, tf, tt = leg_ik(x, y, z)
    x2, y2, z2 = leg_fk(tc, tf, tt)
    assert_close(x2, x, tol, "x")
    assert_close(y2, y, tol, "y")
    assert_close(z2, z, tol, "z")


# ---------------------------------------------------------------------------
# FK: known positions
# ---------------------------------------------------------------------------

class TestFK:
    def test_neutral_all_zero(self):
        """All joints at 0° → documented neutral foot position."""
        x, y, z = leg_fk(0, 0, 0)
        assert_close(x, COXA + FEMUR, msg="x at neutral")  # 17.4
        assert_close(y, 0,             msg="y at neutral")
        assert_close(z, -TIBIA,        msg="z at neutral")  # -15.0

    def test_coxa_90(self):
        """Coxa +90° rotates foot to the Y direction."""
        x, y, z = leg_fk(90, 0, 0)
        assert_close(x, 0,             tol=1e-5, msg="x")
        assert_close(y, COXA + FEMUR,  tol=1e-5, msg="y")
        assert_close(z, -TIBIA,        msg="z")

    def test_femur_90_up(self):
        """Femur +90° → femur straight up; tibia joint-neutral is 90° from femur = pointing outward."""
        x, y, z = leg_fk(0, 90, 0)
        # tibia pivot: (coxa, 0, femur); neutral tibia is 90° CW from femur = pointing +r
        # foot: (coxa + tibia, 0, femur)
        assert_close(x, COXA + TIBIA, tol=1e-5, msg="x")
        assert_close(y, 0,            msg="y")
        assert_close(z, FEMUR,        tol=1e-5, msg="z")

    def test_non_neutral_femur_roundtrip(self):
        """FK→IK→FK with a non-zero femur angle exercises the joint-angle tibia formulation."""
        for tf, tt in [(30, 0), (45, -20), (-20, 30), (60, 10)]:
            x, y, z = leg_fk(0, tf, tt)
            fk_ik_roundtrip(x, y, z, tol=1e-5)

    def test_tibia_90_outward(self):
        """Tibia +90° → tibia points straight outward (parallel to ground)."""
        x, y, z = leg_fk(0, 0, 90)
        assert_close(x, COXA + FEMUR + TIBIA, tol=1e-5, msg="x")
        assert_close(y, 0, msg="y")
        assert_close(z, 0, tol=1e-5, msg="z")


# ---------------------------------------------------------------------------
# IK: round-trip FK→IK→FK
# ---------------------------------------------------------------------------

class TestIK:
    def test_neutral_position(self):
        """IK at neutral foot position returns all-zero angles."""
        tc, tf, tt = leg_ik(COXA + FEMUR, 0, -TIBIA)
        assert_close(tc, 0, tol=1e-4, msg="coxa")
        assert_close(tf, 0, tol=1e-4, msg="femur")
        assert_close(tt, 0, tol=1e-4, msg="tibia")

    def test_roundtrip_neutral(self):
        fk_ik_roundtrip(COXA + FEMUR, 0, -TIBIA)

    def test_roundtrip_foot_low(self):
        """Foot directly below coxa axis, lower than neutral."""
        fk_ik_roundtrip(COXA + 5, 0, -20)

    def test_roundtrip_foot_high(self):
        """Foot raised above neutral."""
        fk_ik_roundtrip(COXA + 10, 0, -8)

    def test_roundtrip_coxa_offset(self):
        """Foot displaced laterally (non-zero coxa angle)."""
        fk_ik_roundtrip(10, 5, -12)

    def test_roundtrip_various(self):
        targets = [
            (14, 0,  -10),
            (14, 0,  -18),
            (10, 8,  -14),
            (12, -4, -16),
        ]
        for pos in targets:
            fk_ik_roundtrip(*pos, tol=1e-5)

    def test_out_of_reach(self):
        far = COXA + FEMUR + TIBIA + 1
        with pytest.raises(IKError):
            leg_ik(far, 0, 0)

    def test_too_close(self):
        # Foot inside the minimum reachable sphere from femur pivot
        with pytest.raises(IKError):
            leg_ik(COXA, 0, 0)


# ---------------------------------------------------------------------------
# Tick ↔ angle
# ---------------------------------------------------------------------------

class TestTickAngle:
    def test_neutral_tick_is_zero_degrees(self):
        for joint in ("coxa", "femur", "tibia"):
            assert_close(tick_to_angle(joint, 2048), 0.0, msg=joint)

    def test_coxa_sign(self):
        # Positive angle = CCW → tick < 2048
        assert angle_to_tick("coxa", +45) < 2048
        assert angle_to_tick("coxa", -45) > 2048

    def test_femur_sign(self):
        # Positive angle = tip up → tick < 2048
        assert angle_to_tick("femur", +45) < 2048
        assert angle_to_tick("femur", -45) > 2048

    def test_tibia_sign(self):
        # Positive angle = tip outward → tick > 2048
        assert angle_to_tick("tibia", +45) > 2048
        assert angle_to_tick("tibia", -45) < 2048

    def test_roundtrip(self):
        for joint in ("coxa", "femur", "tibia"):
            for deg in (-90, -45, 0, 45, 90):
                tick = angle_to_tick(joint, deg)
                back = tick_to_angle(joint, tick)
                assert_close(back, deg, tol=0.01, msg=f"{joint} {deg}°")
