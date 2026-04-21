"""
Hexapod physical configuration.

Leg layout (top view, forward = up):

        FRONT
     6 ←── 1
    5         2
     4 ──→ 3
        BACK

Servo ID scheme: leg * 10 + joint
  joint 1 = coxa, 2 = femur, 3 = tibia
"""

from enum import IntEnum


class Joint(IntEnum):
    COXA  = 1
    FEMUR = 2
    TIBIA = 3


class Leg(IntEnum):
    FRONT_RIGHT = 1
    MID_RIGHT   = 2
    REAR_RIGHT  = 3
    REAR_LEFT   = 4
    MID_LEFT    = 5
    FRONT_LEFT  = 6


def servo_id(leg: Leg | int, joint: Joint | int) -> int:
    return int(leg) * 10 + int(joint)


LEGS: dict[Leg, dict[Joint, int]] = {
    leg: {joint: servo_id(leg, joint) for joint in Joint}
    for leg in Leg
}

ALL_SERVO_IDS: list[int] = [servo_id(leg, joint) for leg in Leg for joint in Joint]
