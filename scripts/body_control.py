#!/usr/bin/env python3
"""
Interactive body IK controller.

Commands:
  stand [height]    raise body to <height> cm above ground with feet at neutral
                    positions (default: 15 cm).  Run this before jogging.
  sit               return to neutral ticks and disable torque.
  pose              print current body pose and all 18 joint angles.
  jog               enter keyboard jog mode to translate/rotate the body.
  speed <ticks/s>   servo speed for stand/sit motions (default: 300).
  step <cm> <deg>   jog step sizes for translation and rotation (default: 0.5 cm, 1.0°).
  off               disable torque on all 18 servos immediately.
  on                enable  torque on all 18 servos.
  q / quit          exit.

Jog key layout (ESC or Ctrl-C to exit jog mode):
  Translate mode (default):
    W / S   body +x / -x  (forward / backward)
    A / D   body +y / -y  (left / right)
    E / Q   body +z / -z  (up / down)

  Rotate mode  (Tab to toggle):
    W / S   +pitch / -pitch  (nose up / down)
    A / D   +yaw   / -yaw    (turn left / right)
    E / Q   +roll  / -roll   (left side up / down)
"""

import sys
import time
import tty
import termios
import argparse
import readline  # noqa: F401 — enables arrow-key history in input()
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hexapod.servo.transport import SerialTransport, TransportError
from hexapod.servo.protocol import ProtocolError
from hexapod.servo.st3020 import ST3020Bus, PositionCommand
from hexapod.servo.motion import MotionPlayer
from hexapod.robot.config import Leg, Joint, servo_id
from hexapod.robot.soft_limits import SoftLimits, SoftLimitError
from hexapod.kinematics import angle_to_tick, tick_to_angle, IKError
from hexapod.body_ik import (
    BodyPose,
    body_pose_ik,
    neutral_foot_body,
)

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_STAND_HEIGHT = 15.0   # cm
DEFAULT_SPEED = 300            # ticks/s for stand/sit
DEFAULT_STEP_CM = 0.5
DEFAULT_STEP_DEG = 1.0
DEFAULT_ACC = 0

JOINT_NAMES = {Joint.COXA: "coxa", Joint.FEMUR: "femur", Joint.TIBIA: "tibia"}

# Jog deltas: (dx, dy, dz) for translate mode, (droll, dpitch, dyaw) for rotate mode
_JOG_TRANSLATE: dict[str, tuple[float, float, float]] = {
    "w": (+1, 0, 0), "s": (-1, 0, 0),
    "a": (0, +1, 0), "d": (0, -1, 0),
    "e": (0, 0, +1), "q": (0, 0, -1),
}
_JOG_ROTATE: dict[str, tuple[float, float, float]] = {
    "w": (0, +1, 0), "s": (0, -1, 0),   # pitch: nose up / down
    "a": (0, 0, +1), "d": (0, 0, -1),   # yaw:   left / right
    "e": (+1, 0, 0), "q": (-1, 0, 0),   # roll:  left side up / down
}


# ---------------------------------------------------------------------------
# IK helpers
# ---------------------------------------------------------------------------

def neutral_world_feet() -> dict[Leg, tuple[float, float, float]]:
    """Neutral foot positions planted on the ground (world z = 0)."""
    return {
        leg: (neutral_foot_body(leg)[0], neutral_foot_body(leg)[1], 0.0)
        for leg in Leg
    }


def compute_all_ticks(
    pose: BodyPose,
    feet: dict[Leg, tuple[float, float, float]],
    limits: SoftLimits | None,
) -> dict[Leg, dict[Joint, int]]:
    """
    Run body_pose_ik for all legs, apply soft limits, return tick values.
    Raises IKError or SoftLimitError on any failure.
    """
    angles = body_pose_ik(pose, feet)
    ticks: dict[Leg, dict[Joint, int]] = {}
    for leg, (tc, tf, tt) in angles.items():
        if limits:
            limits.check(tc, tf, tt)
        ticks[leg] = {
            Joint.COXA:  angle_to_tick("coxa",  tc),
            Joint.FEMUR: angle_to_tick("femur", tf),
            Joint.TIBIA: angle_to_tick("tibia", tt),
        }
    return ticks


def apply_all_ticks(bus: ST3020Bus, ticks: dict[Leg, dict[Joint, int]]) -> None:
    cmds = [
        PositionCommand(servo_id(leg, joint), ticks[leg][joint], speed=0, acc=DEFAULT_ACC)
        for leg in Leg
        for joint in Joint
    ]
    bus.sync_write_position(cmds)


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def print_pose(
    pose: BodyPose,
    feet: dict[Leg, tuple[float, float, float]] | None,
    limits: SoftLimits | None,
) -> None:
    print()
    print(f"  Body pose:")
    print(f"    x={pose.x:+6.2f} cm   y={pose.y:+6.2f} cm   z={pose.z:+6.2f} cm")
    print(f"    roll={pose.roll:+6.2f}°   pitch={pose.pitch:+6.2f}°   yaw={pose.yaw:+6.2f}°")

    if feet is None:
        print("  (no feet planted — run 'stand' first)")
        print()
        return

    try:
        angles = body_pose_ik(pose, feet)
    except IKError as e:
        print(f"  IKError: {e}")
        print()
        return

    print()
    print(f"  {'Leg':<14}  {'Coxa':>8}  {'Femur':>8}  {'Tibia':>8}")
    print("  " + "-" * 46)
    for leg in Leg:
        tc, tf, tt = angles[leg]
        print(f"  {leg.name:<14}  {tc:>+7.2f}°  {tf:>+7.2f}°  {tt:>+7.2f}°")
    print()


# ---------------------------------------------------------------------------
# Stand / sit
# ---------------------------------------------------------------------------

def do_stand(
    bus: ST3020Bus,
    height: float,
    speed: int,
    limits: SoftLimits | None,
) -> tuple[BodyPose, dict[Leg, tuple[float, float, float]]] | None:
    """
    Move all 18 servos to the neutral standing pose at the given height.
    Returns (pose, feet) state on success, None on failure.
    """
    pose = BodyPose(z=height)
    feet = neutral_world_feet()

    try:
        ticks = compute_all_ticks(pose, feet, limits)
    except (IKError, SoftLimitError) as e:
        print(f"  {type(e).__name__}: {e}")
        return None

    targets = [
        (servo_id(leg, joint), ticks[leg][joint], speed)
        for leg in Leg
        for joint in Joint
    ]
    player = MotionPlayer(bus, acc=DEFAULT_ACC)
    try:
        player.move(targets)
    except (ProtocolError, TransportError) as e:
        print(f"  Bus error during stand: {e}")
        return None

    return pose, feet


def do_sit(bus: ST3020Bus, speed: int) -> None:
    """Drive all servos back to neutral ticks (2048) then disable torque."""
    targets = [(servo_id(leg, joint), 2048, speed) for leg in Leg for joint in Joint]
    player = MotionPlayer(bus, acc=DEFAULT_ACC)
    try:
        player.move(targets)
    except (ProtocolError, TransportError) as e:
        print(f"  Bus error during sit: {e}")
        return
    for leg in Leg:
        for joint in Joint:
            bus.torque_enable(servo_id(leg, joint), False)
    print("  Servos at neutral, torque disabled.")


# ---------------------------------------------------------------------------
# Jog mode
# ---------------------------------------------------------------------------

def jog_mode(
    bus: ST3020Bus,
    pose: BodyPose,
    feet: dict[Leg, tuple[float, float, float]],
    step_cm: float,
    step_deg: float,
    limits: SoftLimits | None,
) -> BodyPose:
    """
    Raw-terminal jog loop.  Returns the final body pose.
    """
    rotate_mode = False

    def status_line() -> str:
        mode = "ROTATE" if rotate_mode else "TRANSL"
        return (
            f"  [{mode}] Tab=toggle  "
            f"x={pose.x:+5.1f}  y={pose.y:+5.1f}  z={pose.z:+5.1f}  "
            f"roll={pose.roll:+5.1f}  pitch={pose.pitch:+5.1f}  yaw={pose.yaw:+5.1f}  "
            f"ESC=exit    "
        )

    print(f"\n  Jog mode  (step={step_cm} cm / {step_deg}°)")
    print(status_line(), end="", flush=True)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)

            if ch in ("\x1b", "\x03"):  # ESC or Ctrl-C
                break

            if ch == "\t":  # Tab — toggle translate / rotate
                rotate_mode = not rotate_mode
                sys.stdout.write(f"\r{status_line()}")
                sys.stdout.flush()
                continue

            key = ch.lower()
            if rotate_mode:
                delta = _JOG_ROTATE.get(key)
            else:
                delta = _JOG_TRANSLATE.get(key)

            if delta is None:
                continue

            if rotate_mode:
                droll, dpitch, dyaw = delta
                new_pose = replace(
                    pose,
                    roll=pose.roll   + droll  * step_deg,
                    pitch=pose.pitch + dpitch * step_deg,
                    yaw=pose.yaw     + dyaw   * step_deg,
                )
            else:
                dx, dy, dz = delta
                new_pose = replace(
                    pose,
                    x=pose.x + dx * step_cm,
                    y=pose.y + dy * step_cm,
                    z=pose.z + dz * step_cm,
                )

            try:
                ticks = compute_all_ticks(new_pose, feet, limits)
            except (IKError, SoftLimitError):
                continue  # silently skip unreachable / out-of-limits positions

            apply_all_ticks(bus, ticks)
            pose = new_pose
            sys.stdout.write(f"\r{status_line()}")
            sys.stdout.flush()

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()

    return pose


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive body IK controller")
    parser.add_argument("--port", default=DEFAULT_PORT)
    args = parser.parse_args()

    limits = SoftLimits.load()

    body_pose: BodyPose | None = None
    planted_feet: dict[Leg, tuple[float, float, float]] | None = None
    speed = DEFAULT_SPEED
    step_cm = DEFAULT_STEP_CM
    step_deg = DEFAULT_STEP_DEG

    print(f"Port: {args.port}")
    if limits:
        print(
            f"Soft limits: coxa [{limits.coxa.min_deg:+.1f}°..{limits.coxa.max_deg:+.1f}°]"
            f"  femur [{limits.femur.min_deg:+.1f}°..{limits.femur.max_deg:+.1f}°]"
            f"  tibia [{limits.tibia.min_deg:+.1f}°..{limits.tibia.max_deg:+.1f}°]"
        )
    else:
        print("Soft limits: none")
    print("Type 'stand' to raise the robot, or 'help' for commands.\n")

    with SerialTransport(args.port) as transport:
        bus = ST3020Bus(transport)

        while True:
            try:
                raw = input("body> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            parts = raw.split()
            lower = raw.lower()

            if not raw or lower in ("help", "?", "h"):
                print(__doc__)
                continue

            if lower in ("q", "quit", "exit"):
                break

            # ----------------------------------------------------------------
            # stand [height]
            # ----------------------------------------------------------------
            if parts and parts[0].lower() == "stand":
                height = DEFAULT_STAND_HEIGHT
                if len(parts) == 2:
                    try:
                        height = float(parts[1])
                        if height <= 0:
                            raise ValueError
                    except ValueError:
                        print("  Usage: stand [height]  (positive cm, default 15)")
                        continue
                result = do_stand(bus, height, speed, limits)
                if result is not None:
                    body_pose, planted_feet = result
                    print(f"  Standing at z={height} cm.")
                    print_pose(body_pose, planted_feet, limits)
                continue

            # ----------------------------------------------------------------
            # sit
            # ----------------------------------------------------------------
            if lower == "sit":
                do_sit(bus, speed)
                body_pose = None
                planted_feet = None
                continue

            # ----------------------------------------------------------------
            # pose
            # ----------------------------------------------------------------
            if lower == "pose":
                if body_pose is None:
                    print("  No active pose — run 'stand' first.")
                else:
                    print_pose(body_pose, planted_feet, limits)
                continue

            # ----------------------------------------------------------------
            # jog
            # ----------------------------------------------------------------
            if lower == "jog":
                if body_pose is None or planted_feet is None:
                    print("  Run 'stand' first to plant feet before jogging.")
                    continue
                body_pose = jog_mode(
                    bus, body_pose, planted_feet, step_cm, step_deg, limits
                )
                print_pose(body_pose, planted_feet, limits)
                continue

            # ----------------------------------------------------------------
            # speed <ticks/s>
            # ----------------------------------------------------------------
            if len(parts) == 2 and parts[0].lower() == "speed":
                try:
                    v = int(parts[1])
                    if v <= 0:
                        raise ValueError
                    speed = v
                    print(f"  Stand/sit speed set to {speed} ticks/s")
                except ValueError:
                    print("  Usage: speed <ticks/s>  (positive integer)")
                continue

            # ----------------------------------------------------------------
            # step <cm> <deg>
            # ----------------------------------------------------------------
            if len(parts) == 3 and parts[0].lower() == "step":
                try:
                    sc = float(parts[1])
                    sd = float(parts[2])
                    if sc <= 0 or sd <= 0:
                        raise ValueError
                    step_cm = sc
                    step_deg = sd
                    print(f"  Step set to {step_cm} cm / {step_deg}°")
                except ValueError:
                    print("  Usage: step <cm> <deg>  (both positive)")
                continue

            # ----------------------------------------------------------------
            # off / on
            # ----------------------------------------------------------------
            if lower == "off":
                for leg in Leg:
                    for joint in Joint:
                        bus.torque_enable(servo_id(leg, joint), False)
                body_pose = None
                planted_feet = None
                print("  Torque disabled on all 18 servos.")
                continue

            if lower == "on":
                for leg in Leg:
                    for joint in Joint:
                        bus.torque_enable(servo_id(leg, joint), True)
                print("  Torque enabled.")
                continue

            print(f"  Unknown command: {raw!r}  (type 'help')")


if __name__ == "__main__":
    main()
