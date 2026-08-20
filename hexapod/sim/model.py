"""
Reconstruct the robot's shape from raw servo ticks.

The simulator only knows joint positions, so the world state is derived
kinematically: run forward kinematics per leg, settle the body onto the plane
its feet rest on — which is what tilts it when the body is rolled or pitched
over planted feet — and check whether the body center is still over the polygon
spanned by the feet that touch the ground.

Where the robot has walked to is recovered the same way a real robot does it:
feet that stay on the ground between two frames are fixed points in the world,
so the rigid transform they describe is the body's motion (see Odometry).

This is a kinematic model, not a physics engine — the body stays level; the
stability margin tells you whether the real robot would tip over.
"""

import math
from dataclasses import dataclass
from itertools import combinations

from hexapod.body_ik import corner_pos
from hexapod.kinematics import COXA_LEN, FEMUR_LEN, TIBIA_LEN, tick_to_angle
from hexapod.robot.config import Joint, Leg, servo_id
from hexapod.support import support_margin

CONTACT_EPS = 0.6  # cm — a foot this close to the ground counts as supporting
PLANE_EPS = 0.05  # cm — tolerance for "this foot is not below the plane"
MAX_TILT_DEG = 40.0  # steeper than this is a folded-up pose, not a stance

_JOINT_NAMES = {Joint.COXA: "coxa", Joint.FEMUR: "femur", Joint.TIBIA: "tibia"}

Point = tuple[float, float, float]


@dataclass
class LegState:
    leg: Leg
    ticks: dict[str, int]
    angles: dict[str, float]
    joints: list[Point]  # corner, femur pivot, tibia pivot, foot (world frame)
    grounded: bool

    @property
    def foot(self) -> Point:
        return self.joints[-1]


@dataclass
class WorldPose:
    """Where the body sits in the world, accumulated from foot contact."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # radians, CCW seen from above

    def apply(self, p: Point) -> Point:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.x + p[0] * c - p[1] * s, self.y + p[0] * s + p[1] * c, p[2])


@dataclass
class RobotState:
    legs: list[LegState]
    body_z: float
    roll: float  # degrees, positive = left side up
    pitch: float  # degrees, positive = nose up
    stable: bool
    margin: float  # cm from body center to the support polygon edge (<0 = outside)

    def to_dict(self, world: WorldPose | None = None) -> dict:
        """JSON view; with *world* the joints come out in world coordinates."""
        pose = world or WorldPose()
        return {
            "body_z": round(self.body_z, 2),
            "roll": round(self.roll, 1),
            "pitch": round(self.pitch, 1),
            "stable": self.stable,
            "margin": round(self.margin, 2),
            "world": {
                "x": round(pose.x, 2),
                "y": round(pose.y, 2),
                "heading": round(_wrap_degrees(math.degrees(pose.yaw)), 1),
            },
            "legs": [
                {
                    "leg": leg.leg.value,
                    "name": leg.leg.name,
                    "ticks": leg.ticks,
                    "angles": {k: round(v, 1) for k, v in leg.angles.items()},
                    "joints": [
                        [round(c, 2) for c in pose.apply(p)] for p in leg.joints
                    ],
                    "grounded": leg.grounded,
                }
                for leg in self.legs
            ],
        }


def _wrap_degrees(deg: float) -> float:
    """Fold an angle into [-180, 180) so the readout stays legible."""
    return (deg + 180.0) % 360.0 - 180.0


def leg_joint_positions(
    leg: Leg, coxa_deg: float, femur_deg: float, tibia_deg: float
) -> list[Point]:
    """Joint chain (corner → femur pivot → tibia pivot → foot) in the body frame."""
    tc, tf, tt = map(math.radians, (coxa_deg, femur_deg, tibia_deg))

    # Radial distance / height of each joint in the vertical plane of the leg,
    # matching kinematics.leg_fk.
    chain = [
        (0.0, 0.0),
        (COXA_LEN, 0.0),
        (COXA_LEN + FEMUR_LEN * math.cos(tf), FEMUR_LEN * math.sin(tf)),
        (
            COXA_LEN + FEMUR_LEN * math.cos(tf) + TIBIA_LEN * math.sin(tf + tt),
            FEMUR_LEN * math.sin(tf) - TIBIA_LEN * math.cos(tf + tt),
        ),
    ]

    cx, cy, _ = corner_pos(leg)
    a = math.atan2(cy, cx)  # corner direction in the body frame
    points: list[Point] = []
    for r, z in chain:
        # Leg frame: X radially outward, rotated by the coxa yaw
        x_leg, y_leg = r * math.cos(tc), r * math.sin(tc)
        points.append(
            (
                cx + x_leg * math.cos(a) - y_leg * math.sin(a),
                cy + x_leg * math.sin(a) + y_leg * math.cos(a),
                z,
            )
        )
    return points


def robot_state(positions: dict[int, int]) -> RobotState:
    """Build the world-frame robot state from servo_id → tick."""
    legs: list[LegState] = []
    for leg in Leg:
        ticks = {
            _JOINT_NAMES[joint]: positions.get(servo_id(leg, joint), 2048)
            for joint in Joint
        }
        angles = {name: tick_to_angle(name, tick) for name, tick in ticks.items()}
        joints = leg_joint_positions(
            leg, angles["coxa"], angles["femur"], angles["tibia"]
        )
        legs.append(
            LegState(leg=leg, ticks=ticks, angles=angles, joints=joints, grounded=False)
        )

    # Settle the body onto the ground: rotate it so the plane its feet rest on
    # becomes horizontal, then lower it until that plane is z = 0.
    rot, height = _settle([leg.foot for leg in legs])
    for leg in legs:
        leg.joints = [_shift(_rotate(rot, p), height) for p in leg.joints]
        leg.grounded = leg.foot[2] <= CONTACT_EPS

    roll, pitch = _roll_pitch(rot)
    support = [(leg.foot[0], leg.foot[1]) for leg in legs if leg.grounded]
    margin = support_margin(support)
    return RobotState(
        legs=legs,
        body_z=height,
        roll=roll,
        pitch=pitch,
        stable=margin > 0.0,
        margin=margin,
    )


# ---------------------------------------------------------------------------
# Settling the body on its feet
# ---------------------------------------------------------------------------


def _settle(feet: list[Point]) -> tuple[list[list[float]], float]:
    """Rotation that levels the ground under the robot, and the body's height.

    A rolled or pitched body keeps its feet on the floor and tilts itself, so
    the tilt has to be read back out of the foot layout: the robot rests on the
    plane that carries some feet and leaves none below it.  Without such a
    plane (legs folded up, fewer than three feet usable) the body is left level
    and simply lowered onto its lowest foot.
    """
    plane = _resting_plane(feet)
    if plane is None:
        return _IDENTITY, max(0.0, -min(f[2] for f in feet))
    normal, point = plane
    # The body plate cannot sink through the floor: with the legs folded above
    # it (storage pose) the robot simply lies on its belly.
    return _align_up(normal), max(0.0, -_dot(normal, point))


def _resting_plane(feet: list[Point]) -> tuple[Point, Point] | None:
    """The plane the body settles on: (unit normal pointing up, point on it).

    Every triple of feet spans a candidate plane; the ones with no foot below
    them are the faces of the feet's lower convex hull.  The body rests on
    whichever of those the vertical through its center meets highest up.
    """
    best: tuple[float, Point, Point] | None = None
    for p, q, r in combinations(feet, 3):
        normal = _cross(_sub(q, p), _sub(r, p))
        if normal[2] < 0:
            normal = (-normal[0], -normal[1], -normal[2])
        length = math.sqrt(_dot(normal, normal))
        if length < 1e-9 or normal[2] / length < math.cos(math.radians(MAX_TILT_DEG)):
            continue  # degenerate, or too steep to be a stance
        normal = (normal[0] / length, normal[1] / length, normal[2] / length)

        if any(_dot(normal, _sub(f, p)) < -PLANE_EPS for f in feet):
            continue  # not a lower-hull face: some foot is below it

        # Height at which the body's vertical crosses this plane
        z0 = _dot(normal, p) / normal[2]
        if best is None or z0 > best[0]:
            best = (z0, normal, p)

    return None if best is None else (best[1], best[2])


def _align_up(normal: Point) -> list[list[float]]:
    """Smallest rotation taking *normal* onto +Z (Rodrigues, no yaw component)."""
    axis = (normal[1], -normal[0], 0.0)  # normal × ẑ
    sin_a = math.hypot(axis[0], axis[1])
    if sin_a < 1e-9:
        return _IDENTITY
    ux, uy = axis[0] / sin_a, axis[1] / sin_a
    cos_a = max(-1.0, min(1.0, normal[2]))
    t = 1.0 - cos_a
    return [
        [cos_a + t * ux * ux, t * ux * uy, sin_a * uy],
        [t * ux * uy, cos_a + t * uy * uy, -sin_a * ux],
        [-sin_a * uy, sin_a * ux, cos_a],
    ]


def _roll_pitch(rot: list[list[float]]) -> tuple[float, float]:
    """Body roll and pitch from the body→world rotation (see body_ik)."""
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -rot[2][0]))))
    roll = math.degrees(math.atan2(rot[2][1], rot[2][2]))
    return roll, pitch


_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _rotate(rot: list[list[float]], p: Point) -> Point:
    return (
        rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2],
        rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2],
        rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2],
    )


def _shift(p: Point, dz: float) -> Point:
    return (p[0], p[1], p[2] + dz)


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


# ---------------------------------------------------------------------------
# Odometry
# ---------------------------------------------------------------------------

MAX_BODY_STEP = 5.0  # cm — a larger jump between frames is a stance change, not motion


class Odometry:
    """Accumulate the body's world pose from the feet that stay planted.

    A foot on the ground is a fixed point in the world, so if it drifts in the
    body frame the body itself must have moved.  Fitting a rigid transform to
    all feet that were down in both frames gives that motion; the fit needs two
    feet for rotation, one is enough for a straight translation.
    """

    def __init__(self) -> None:
        self.pose = WorldPose()
        self._prev: dict[Leg, tuple[float, float]] = {}

    def reset(self) -> None:
        self.pose = WorldPose()
        self._prev = {}

    def update(self, state: RobotState) -> WorldPose:
        current = {
            leg.leg: (leg.foot[0], leg.foot[1]) for leg in state.legs if leg.grounded
        }
        pairs = [
            (self._prev[leg], current[leg]) for leg in current if leg in self._prev
        ]
        self._prev = current
        if not pairs:
            return self.pose

        # Transform taking the previous foot layout onto the current one,
        # both expressed in the body frame.
        theta, tx, ty = _fit_rigid_2d(pairs)

        # Feet turning by +theta in the body frame means the body turned by -theta
        yaw = self.pose.yaw - theta
        c, s = math.cos(yaw), math.sin(yaw)
        x = self.pose.x - (tx * c - ty * s)
        y = self.pose.y - (tx * s + ty * c)

        if math.hypot(x - self.pose.x, y - self.pose.y) <= MAX_BODY_STEP:
            self.pose = WorldPose(x=x, y=y, yaw=yaw)
        return self.pose


def _fit_rigid_2d(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[float, float, float]:
    """Least-squares (rotation, translation) mapping every p onto its q."""
    n = len(pairs)
    px = sum(p[0] for p, _ in pairs) / n
    py = sum(p[1] for p, _ in pairs) / n
    qx = sum(q[0] for _, q in pairs) / n
    qy = sum(q[1] for _, q in pairs) / n

    num = sum((p[0] - px) * (q[1] - qy) - (p[1] - py) * (q[0] - qx) for p, q in pairs)
    den = sum((p[0] - px) * (q[0] - qx) + (p[1] - py) * (q[1] - qy) for p, q in pairs)
    theta = math.atan2(num, den) if (num or den) else 0.0

    c, s = math.cos(theta), math.sin(theta)
    return theta, qx - (px * c - py * s), qy - (px * s + py * c)


# ---------------------------------------------------------------------------
# Support polygon
# ---------------------------------------------------------------------------
