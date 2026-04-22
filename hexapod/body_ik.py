"""
Full-body inverse kinematics.

Layer 1 — body_ik():
    Takes foot targets in the body frame (origin = body center, X = forward,
    Y = left, Z = up) and returns joint angles for all six legs.

Layer 2 — body_pose_ik():
    Takes foot targets in the world frame plus a body pose (translation +
    roll/pitch/yaw) and feeds the result into body_ik().  This is the entry
    point for gaits: feet stay planted in the world while the body moves.

Angles are in degrees throughout; positions in centimetres.
"""

import math
from dataclasses import dataclass

from hexapod.kinematics import leg_ik, IKError, COXA_LEN, FEMUR_LEN, TIBIA_LEN
from hexapod.robot.config import Leg


# ---------------------------------------------------------------------------
# Body geometry
# ---------------------------------------------------------------------------

BODY_RADIUS: float = 8.85  # cm, center to coxa pivot corner

# Angle of each coxa pivot corner measured from the forward (+X) axis,
# in the body frame (X = forward, Y = left).  From docs/kinematics.md.
_CORNER_ANGLE: dict[Leg, float] = {
    Leg.FRONT_RIGHT: -30.0,
    Leg.MID_RIGHT:   -90.0,
    Leg.REAR_RIGHT:  -150.0,
    Leg.REAR_LEFT:   +150.0,
    Leg.MID_LEFT:    +90.0,
    Leg.FRONT_LEFT:  +30.0,
}


def corner_pos(leg: Leg) -> tuple[float, float, float]:
    """Body-frame position of the coxa pivot for *leg*."""
    a = math.radians(_CORNER_ANGLE[leg])
    return BODY_RADIUS * math.cos(a), BODY_RADIUS * math.sin(a), 0.0


def neutral_foot_body(leg: Leg) -> tuple[float, float, float]:
    """
    Body-frame foot position when all joints are at 0° (neutral pose).

    The foot sits straight out from the corner at (COXA+FEMUR) and TIBIA below.
    """
    a = math.radians(_CORNER_ANGLE[leg])
    reach = COXA_LEN + FEMUR_LEN
    cx, cy, _ = corner_pos(leg)
    return (
        cx + reach * math.cos(a),
        cy + reach * math.sin(a),
        -TIBIA_LEN,
    )


# ---------------------------------------------------------------------------
# Frame transform helpers
# ---------------------------------------------------------------------------

def _body_to_leg(leg: Leg, foot_body: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Transform a point from the body frame to the leg frame.

    The leg frame has its origin at the coxa pivot with X pointing radially
    outward (along the corner angle) and Z pointing up.
    """
    a = math.radians(_CORNER_ANGLE[leg])
    cx, cy, _ = corner_pos(leg)

    # Translate to corner origin
    px = foot_body[0] - cx
    py = foot_body[1] - cy
    pz = foot_body[2]

    # Rotate by -angle around Z so that the outward direction becomes +X
    x_leg =  px * math.cos(a) + py * math.sin(a)
    y_leg = -px * math.sin(a) + py * math.cos(a)

    return x_leg, y_leg, pz


def _rotation_matrix_xyz(
    roll: float, pitch: float, yaw: float
) -> list[list[float]]:
    """
    Intrinsic X→Y→Z rotation matrix (roll, then pitch, then yaw), angles in degrees.
    Transforms vectors from body frame to world frame: v_world = R * v_body.
    """
    cr, sr = math.cos(math.radians(roll)),  math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cy, sy = math.cos(math.radians(yaw)),   math.sin(math.radians(yaw))

    return [
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,     cp*sr,             cp*cr            ],
    ]


def _mat_transpose_vec(
    m: list[list[float]], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Multiply M^T by vector v (i.e. apply the inverse rotation)."""
    return (
        m[0][0]*v[0] + m[1][0]*v[1] + m[2][0]*v[2],
        m[0][1]*v[0] + m[1][1]*v[1] + m[2][1]*v[2],
        m[0][2]*v[0] + m[1][2]*v[1] + m[2][2]*v[2],
    )


# ---------------------------------------------------------------------------
# Layer 1: body-frame IK
# ---------------------------------------------------------------------------

LegAngles = tuple[float, float, float]  # (theta_coxa, theta_femur, theta_tibia)


def body_ik(
    foot_positions: dict[Leg, tuple[float, float, float]],
) -> dict[Leg, LegAngles]:
    """
    Compute joint angles for every leg given foot targets in the body frame.

    Parameters
    ----------
    foot_positions:
        Mapping from Leg to (x, y, z) in the body frame (cm).
        Only legs present in the dict are solved; missing legs are ignored.

    Returns
    -------
    dict mapping each Leg to (theta_coxa, theta_femur, theta_tibia) in degrees.

    Raises
    ------
    IKError if any target is outside the leg's reachable workspace.
    """
    result: dict[Leg, LegAngles] = {}
    for leg, foot_body in foot_positions.items():
        foot_leg = _body_to_leg(leg, foot_body)
        try:
            result[leg] = leg_ik(*foot_leg)
        except IKError as e:
            raise IKError(f"{leg.name}: {e}") from e
    return result


# ---------------------------------------------------------------------------
# Layer 2: world-frame IK with body pose
# ---------------------------------------------------------------------------

@dataclass
class BodyPose:
    """
    Rigid-body pose of the hexapod chassis in the world frame.

    Attributes
    ----------
    x, y, z : float
        Position of the body center in the world frame (cm).
        Typical use: z = standing height above the ground.
    roll : float
        Rotation around the forward (X) axis, in degrees.
        Positive = left side rises.
    pitch : float
        Rotation around the lateral (Y) axis, in degrees.
        Positive = nose rises.
    yaw : float
        Rotation around the vertical (Z) axis, in degrees.
        Positive = turns left (CCW from above).
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll:  float = 0.0
    pitch: float = 0.0
    yaw:   float = 0.0


def body_pose_ik(
    pose: BodyPose,
    foot_positions_world: dict[Leg, tuple[float, float, float]],
) -> dict[Leg, LegAngles]:
    """
    Compute joint angles given foot targets in the world frame and a body pose.

    The body pose describes where/how the chassis sits in the world.  The feet
    are assumed to be stationary (resting on the ground); moving the pose while
    keeping feet fixed is how you produce body-sway and walking motion.

    Parameters
    ----------
    pose:
        Current body pose in the world frame.
    foot_positions_world:
        Mapping from Leg to (x, y, z) in the world frame (cm).

    Returns
    -------
    dict mapping each Leg to (theta_coxa, theta_femur, theta_tibia) in degrees.
    """
    R = _rotation_matrix_xyz(pose.roll, pose.pitch, pose.yaw)
    body_origin = (pose.x, pose.y, pose.z)

    foot_positions_body: dict[Leg, tuple[float, float, float]] = {}
    for leg, foot_world in foot_positions_world.items():
        # Translate into body-centered world coordinates, then rotate to body frame
        delta = (
            foot_world[0] - body_origin[0],
            foot_world[1] - body_origin[1],
            foot_world[2] - body_origin[2],
        )
        foot_positions_body[leg] = _mat_transpose_vec(R, delta)

    return body_ik(foot_positions_body)
