"""
Single-leg inverse and forward kinematics.

All angles use the IK-friendly sign convention from docs/kinematics.md:
  coxa  — positive = CCW from above
  femur — positive = tip rises above horizontal
  tibia — positive = tip swings outward (away from body)

Positions are in the leg frame (origin at coxa pivot, X radially outward,
Z up) in centimetres.

Neutral foot position (all joints at 0°): (17.4, 0, −15)
"""

import math

COXA_LEN  = 6.4   # cm
FEMUR_LEN = 11.0  # cm
TIBIA_LEN = 15.0  # cm

TICKS_PER_DEG = 4096 / 360


class IKError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def leg_fk(
    theta_coxa: float,
    theta_femur: float,
    theta_tibia: float,
    *,
    coxa: float = COXA_LEN,
    femur: float = FEMUR_LEN,
    tibia: float = TIBIA_LEN,
) -> tuple[float, float, float]:
    """Return foot (x, y, z) in the leg frame given joint angles in degrees."""
    tc = math.radians(theta_coxa)
    tf = math.radians(theta_femur)
    tt = math.radians(theta_tibia)

    # Tibia angle is a joint angle (relative to femur), so it's composed with tf.
    # At tt=0: tibia is 90° clockwise from femur = straight down when femur is horizontal.
    r = coxa + femur * math.cos(tf) + tibia * math.sin(tf + tt)
    z = femur * math.sin(tf) - tibia * math.cos(tf + tt)

    x = r * math.cos(tc)
    y = r * math.sin(tc)
    return x, y, z


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------

def leg_ik(
    x: float,
    y: float,
    z: float,
    *,
    coxa: float = COXA_LEN,
    femur: float = FEMUR_LEN,
    tibia: float = TIBIA_LEN,
) -> tuple[float, float, float]:
    """
    Return (theta_coxa, theta_femur, theta_tibia) in degrees for a target foot
    position (x, y, z) in the leg frame.

    Raises IKError if the target is outside the reachable workspace.
    """
    # --- Coxa: yaw to align with the foot in the horizontal plane ---
    theta_coxa = math.degrees(math.atan2(y, x))

    # --- Project into the vertical plane after coxa rotation ---
    r_total = math.hypot(x, y)          # distance from coxa axis to foot
    r = r_total - coxa                  # distance from femur pivot to foot (horizontal)

    # Distance from femur pivot to foot
    d = math.hypot(r, z)

    if d > femur + tibia:
        raise IKError(
            f"Target ({x:.1f}, {y:.1f}, {z:.1f}) is out of reach: "
            f"d={d:.2f} > femur+tibia={femur+tibia:.2f}"
        )
    if d < abs(femur - tibia):
        raise IKError(
            f"Target ({x:.1f}, {y:.1f}, {z:.1f}) is too close: "
            f"d={d:.2f} < |femur-tibia|={abs(femur - tibia):.2f}"
        )

    # --- Femur angle ---
    # Elevation of foot direction from femur pivot (negative when foot is below)
    phi = math.atan2(z, r)
    # Interior angle at femur pivot (law of cosines)
    cos_beta = (femur**2 + d**2 - tibia**2) / (2 * femur * d)
    cos_beta = max(-1.0, min(1.0, cos_beta))  # clamp for floating-point safety
    beta = math.acos(cos_beta)
    # Elbow-down configuration (knee points outward/downward — standard hexapod walk pose)
    theta_femur = math.degrees(phi + beta)

    # --- Tibia angle ---
    # Tibia is a joint angle relative to the femur. Compute the absolute direction
    # of the tibia from horizontal, then subtract the femur-relative neutral (tf - 90°).
    tf_rad = math.radians(theta_femur)
    dr = r - femur * math.cos(tf_rad)   # r = r_total - coxa (from femur pivot)
    dz = z - femur * math.sin(tf_rad)
    tibia_abs_angle = math.degrees(math.atan2(dz, dr))   # from +r horizontal
    theta_tibia = tibia_abs_angle - theta_femur + 90.0

    return theta_coxa, theta_femur, theta_tibia


# ---------------------------------------------------------------------------
# Tick ↔ angle conversion (per-joint sign convention from docs/kinematics.md)
# ---------------------------------------------------------------------------

def angle_to_tick(joint: str, degrees: float) -> int:
    """
    Convert an IK-friendly angle to a raw servo tick.
    joint: 'coxa', 'femur', or 'tibia'
    """
    if joint in ("coxa", "femur"):
        delta = -degrees * TICKS_PER_DEG
    elif joint == "tibia":
        delta = +degrees * TICKS_PER_DEG
    else:
        raise ValueError(f"Unknown joint: {joint!r}")
    return round(2048 + delta)


def tick_to_angle(joint: str, tick: int) -> float:
    """Convert a raw servo tick to an IK-friendly angle in degrees."""
    delta = tick - 2048
    if joint in ("coxa", "femur"):
        return -delta / TICKS_PER_DEG
    elif joint == "tibia":
        return +delta / TICKS_PER_DEG
    else:
        raise ValueError(f"Unknown joint: {joint!r}")
