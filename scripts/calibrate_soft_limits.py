#!/usr/bin/env python3
"""
Interactive soft limit calibration.

Torque is disabled on all three joints at startup and stays off for the
entire session so the leg can be moved freely at any point.

For each joint (coxa → femur → tibia) you will be asked to move it to
its minimum and maximum safe angle. The recorded tick values are converted
to IK-friendly angles and saved to soft_limits.json.

Workflow per limit:
  Move joint to the limit position (all joints remain free throughout).
  SPACE = record this position.
  Q     = quit without saving.

The script operates on one reference leg (default: 1 = FRONT_RIGHT).
The resulting limits apply to all legs equally.
Use --leg <1-6> to choose a different reference leg.
"""

import argparse
import contextlib
import os
import select
import sys
import termios
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hexapod.servo.transport import SerialTransport, TransportError
from hexapod.servo.protocol import ProtocolError
from hexapod.servo.st3020 import ST3020Bus
from hexapod.robot.config import Leg, Joint, servo_id
from hexapod.robot.soft_limits import JointLimits, SoftLimits
from hexapod.kinematics import tick_to_angle

DEFAULT_PORT = "/dev/ttyACM0"

JOINT_NAMES = {Joint.COXA: "coxa", Joint.FEMUR: "femur", Joint.TIBIA: "tibia"}
JOINT_HINTS = {
    Joint.COXA:  "Rotate coxa left/right",
    Joint.FEMUR: "Lift/lower the femur",
    Joint.TIBIA: "Bend/extend the tibia",
}


@contextlib.contextmanager
def raw_tty():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key() -> str | None:
    fd = sys.stdin.fileno()
    if not select.select([fd], [], [], 0.08)[0]:
        return None
    data = os.read(fd, 32)
    if not data:
        return None
    return data[0:1].decode("utf-8", errors="replace")


def safe_read(bus: ST3020Bus, sid: int) -> int | None:
    try:
        return bus.read_position(sid)
    except (ProtocolError, TransportError):
        return None


def record_limit(bus: ST3020Bus, sid: int, joint: Joint, label: str) -> int | None:
    """
    Display live tick readout and wait for SPACE to record the current position.
    Torque remains off (caller's responsibility to have disabled it).
    Returns the recorded tick, or None on quit.
    """
    name = JOINT_NAMES[joint]
    print(f"\n  {label}  —  {JOINT_HINTS[joint]}")
    print("  SPACE = record   Q = quit")

    with raw_tty():
        while True:
            tick = safe_read(bus, sid)
            if tick is not None:
                angle = tick_to_angle(name, tick)
                pos_str = f"{tick:4d}  ({angle:+.1f}°)"
            else:
                pos_str = " ERR"
            sys.stdout.write(f"\r  [{name}] {pos_str}    ")
            sys.stdout.flush()

            key = read_key()
            if key == " ":
                sys.stdout.write("\n")
                sys.stdout.flush()
                return tick
            elif key in ("q", "Q", "\x03"):
                sys.stdout.write("\r  [quit]\n")
                sys.stdout.flush()
                return None


def calibrate_joint(bus: ST3020Bus, sid: int, joint: Joint) -> JointLimits | None:
    """Record min and max for one joint. Returns JointLimits or None on quit."""
    name = JOINT_NAMES[joint]
    print(f"\n{'─'*52}")
    print(f"  Joint: {name.upper()}  (servo ID {sid})")
    print(f"{'─'*52}")

    tick_a = record_limit(bus, sid, joint, "MINIMUM limit")
    if tick_a is None:
        return None

    tick_b = record_limit(bus, sid, joint, "MAXIMUM limit")
    if tick_b is None:
        return None

    angle_a = tick_to_angle(name, tick_a)
    angle_b = tick_to_angle(name, tick_b)
    lo, hi = min(angle_a, angle_b), max(angle_a, angle_b)

    print(f"  {name}: {lo:+.1f}° .. {hi:+.1f}°")
    return JointLimits(min_deg=lo, max_deg=hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate soft joint limits")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument(
        "--leg", type=int, default=1,
        choices=[l.value for l in Leg],
        help="Reference leg 1-6 (default: 1 = FRONT_RIGHT)",
    )
    args = parser.parse_args()

    try:
        leg = Leg(args.leg)
    except ValueError:
        print(f"Unknown leg {args.leg}")
        sys.exit(1)

    ids = {joint: servo_id(leg, joint) for joint in Joint}

    print("Hexapod soft limit calibration")
    print(f"Port: {args.port}   Leg: {leg.name} (leg {leg.value})")
    print("Limits apply to all legs equally.")
    print()
    print("Torque will be disabled on all three joints for the entire session.")
    print("Move any joint freely at any time — nothing will lock between steps.")

    limits: dict[Joint, JointLimits] = {}

    with SerialTransport(args.port) as transport:
        bus = ST3020Bus(transport)

        # Disable torque on all three joints once, up front.
        for sid in ids.values():
            bus.torque_enable(sid, False)

        for joint in [Joint.COXA, Joint.FEMUR, Joint.TIBIA]:
            result = calibrate_joint(bus, ids[joint], joint)
            if result is None:
                print("\nAborted — no limits saved.")
                sys.exit(0)
            limits[joint] = result

    soft = SoftLimits(
        coxa=limits[Joint.COXA],
        femur=limits[Joint.FEMUR],
        tibia=limits[Joint.TIBIA],
    )

    out_path = Path(__file__).parent.parent / "soft_limits.json"
    soft.save(out_path)

    print(f"\n{'='*52}")
    print(f"Saved: {out_path}")
    print(f"  coxa:  {soft.coxa.min_deg:+.1f}° .. {soft.coxa.max_deg:+.1f}°")
    print(f"  femur: {soft.femur.min_deg:+.1f}° .. {soft.femur.max_deg:+.1f}°")
    print(f"  tibia: {soft.tibia.min_deg:+.1f}° .. {soft.tibia.max_deg:+.1f}°")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
