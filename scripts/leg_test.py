#!/usr/bin/env python3
"""
Interactive single-leg test for empirical kinematics verification.

Commands:
  neutral              move all three joints to tick 2048
  ik <x> <y> <z>       move foot to absolute position in leg frame (cm)
  ik +<dx> +<dy> +<dz> translate foot relative to current position (signed)
  pos                  read current ticks and compute foot position via FK
  speed <cm/s>         set interpolation speed (default: 10 cm/s)
  jog                  enter keyboard jog mode (WASD = x/y, QE = z, ESC to exit)
  step <cm>            set jog step size (default: 0.5 cm)
  c <tick>             set coxa  to given tick (0-4095)
  f <tick>             set femur to given tick (0-4095)
  t <tick>             set tibia to given tick (0-4095)
  c+<deg>              offset coxa  by ±degrees from 2048  (e.g. c+45 or c-30)
  f+<deg>              offset femur by ±degrees from 2048
  t+<deg>              offset tibia by ±degrees from 2048
  read                 read and print current tick values + angles
  off                  disable torque on all three joints
  on                   enable  torque on all three joints
  q / quit             exit

Jog key layout (leg frame):
  W/S  = +x / -x  (forward / backward from body)
  A/D  = +y / -y  (left / right)
  E/Q  = +z / -z  (up / down)
  Hold a key for continuous movement. ESC or Ctrl-C to exit jog mode.

Angles use the IK-friendly sign convention from docs/kinematics.md:
  coxa  positive = CCW from above
  femur positive = tip rises above horizontal
  tibia positive = tip swings outward
"""

import sys
import re
import math
import time
import tty
import termios
import argparse
import readline  # noqa: F401 — enables arrow-key history in input()
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hexapod.servo.transport import SerialTransport, TransportError
from hexapod.servo.protocol import ProtocolError
from hexapod.servo.st3020 import ST3020Bus, PositionCommand
from hexapod.robot.config import Leg, Joint, servo_id
from hexapod.robot.soft_limits import SoftLimits, SoftLimitError
from hexapod.kinematics import leg_ik, leg_fk, angle_to_tick, tick_to_angle, IKError

DEFAULT_ACC = 0
DEFAULT_PORT = "/dev/ttyACM0"
INTERP_STEP_CM = 0.5  # Cartesian step size per interpolation step
INTERP_HZ = 50  # target command rate

JOINT_NAMES = {Joint.COXA: "coxa", Joint.FEMUR: "femur", Joint.TIBIA: "tibia"}


def read_angles(bus: ST3020Bus, ids: dict[Joint, int]) -> dict[Joint, float] | None:
    angles = {}
    for joint in [Joint.COXA, Joint.FEMUR, Joint.TIBIA]:
        try:
            tick = bus.read_position(ids[joint])
            angles[joint] = tick_to_angle(JOINT_NAMES[joint], tick)
        except (ProtocolError, TransportError) as e:
            print(f"  {joint.name}: read failed ({e})")
            return None
    return angles


def read_foot_position(
    bus: ST3020Bus, ids: dict[Joint, int]
) -> tuple[float, float, float] | None:
    angles = read_angles(bus, ids)
    if angles is None:
        return None
    return leg_fk(angles[Joint.COXA], angles[Joint.FEMUR], angles[Joint.TIBIA])


def print_status(bus: ST3020Bus, ids: dict[Joint, int]) -> None:
    print()
    print(f"  {'Joint':<8}  {'Tick':>6}  {'Angle':>8}")
    print("  " + "-" * 28)
    for joint in [Joint.COXA, Joint.FEMUR, Joint.TIBIA]:
        try:
            tick = bus.read_position(ids[joint])
            angle = tick_to_angle(JOINT_NAMES[joint], tick)
            print(f"  {joint.name:<8}  {tick:>6d}  {angle:>+8.2f}°")
        except (ProtocolError, TransportError) as e:
            print(f"  {joint.name:<8}   FAIL  ({e})")
    print()


def print_position(
    bus: ST3020Bus, ids: dict[Joint, int]
) -> tuple[float, float, float] | None:
    pos = read_foot_position(bus, ids)
    if pos is None:
        return None
    x, y, z = pos
    print()
    print(f"  Foot position (leg frame):  x={x:+.2f}  y={y:+.2f}  z={z:+.2f}  cm")
    print()
    return pos


def move(bus: ST3020Bus, sid: int, tick: int) -> None:
    bus.write_position(sid, max(0, min(4095, tick)), speed=0, acc=DEFAULT_ACC)


def move_smooth(
    bus: ST3020Bus,
    ids: dict[Joint, int],
    target: tuple[float, float, float],
    speed_cm_s: float,
    limits: SoftLimits | None = None,
) -> bool:
    """
    Interpolate in Cartesian space from the current foot position to target.
    Returns True on success, False if current position could not be read.
    Raises IKError if any step along the path is unreachable.
    Raises SoftLimitError if any step would violate soft limits — partial
    motion will have occurred up to that point.
    """
    current = read_foot_position(bus, ids)
    if current is None:
        return False

    dx = target[0] - current[0]
    dy = target[1] - current[1]
    dz = target[2] - current[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    if dist < 1e-4:
        return True

    dt = INTERP_STEP_CM / speed_cm_s  # time per step
    n = max(1, math.ceil(dist / INTERP_STEP_CM))

    t_next = time.monotonic()
    for i in range(1, n + 1):
        t = i / n
        x = current[0] + t * dx
        y = current[1] + t * dy
        z = current[2] + t * dz
        tc, tf, tt = leg_ik(x, y, z)  # raises IKError if unreachable
        if limits:
            limits.check(tc, tf, tt)   # raises SoftLimitError if violated
        bus.sync_write_position(
            [
                PositionCommand(
                    ids[Joint.COXA], angle_to_tick("coxa", tc), speed=0, acc=DEFAULT_ACC
                ),
                PositionCommand(
                    ids[Joint.FEMUR],
                    angle_to_tick("femur", tf),
                    speed=0,
                    acc=DEFAULT_ACC,
                ),
                PositionCommand(
                    ids[Joint.TIBIA],
                    angle_to_tick("tibia", tt),
                    speed=0,
                    acc=DEFAULT_ACC,
                ),
            ]
        )
        t_next += dt
        wait = t_next - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    return True


_JOG_KEYS: dict[str, tuple[float, float, float]] = {
    "w": (-1, 0, 0),
    "s": (+1, 0, 0),
    "a": (0, -1, 0),
    "d": (0, +1, 0),
    "e": (0, 0, +1),
    "q": (0, 0, -1),
}


def jog_mode(
    bus: ST3020Bus,
    ids: dict[Joint, int],
    step_cm: float,
    limits: SoftLimits | None = None,
) -> None:
    pos = read_foot_position(bus, ids)
    if pos is None:
        print("  Could not read current position.")
        return

    x, y, z = pos
    print(f"  Jog mode  (step={step_cm} cm)  —  WASD=x/y  QE=z  ESC/Ctrl-C to exit")
    print(f"  x={x:+.2f}  y={y:+.2f}  z={z:+.2f}", end="", flush=True)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\x1b", "\x03"):
                break
            delta = _JOG_KEYS.get(ch.lower())
            if delta is None:
                continue
            nx = x + delta[0] * step_cm
            ny = y + delta[1] * step_cm
            nz = z + delta[2] * step_cm
            try:
                tc, tf, tt = leg_ik(nx, ny, nz)
                if limits:
                    limits.check(tc, tf, tt)
            except (IKError, SoftLimitError):
                continue
            x, y, z = nx, ny, nz
            bus.sync_write_position(
                [
                    PositionCommand(
                        ids[Joint.COXA],
                        angle_to_tick("coxa", tc),
                        speed=0,
                        acc=DEFAULT_ACC,
                    ),
                    PositionCommand(
                        ids[Joint.FEMUR],
                        angle_to_tick("femur", tf),
                        speed=0,
                        acc=DEFAULT_ACC,
                    ),
                    PositionCommand(
                        ids[Joint.TIBIA],
                        angle_to_tick("tibia", tt),
                        speed=0,
                        acc=DEFAULT_ACC,
                    ),
                ]
            )
            sys.stdout.write(f"\r  x={x:+.2f}  y={y:+.2f}  z={z:+.2f}    ")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()


def parse_ik_args(parts: list[str]) -> tuple[list[float], bool] | None:
    if len(parts) != 3:
        return None
    relative = all(re.match(r"^[+-]", p) for p in parts)
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    return values, relative


def parse_offset_cmd(token: str) -> tuple[str, float] | None:
    m = re.fullmatch(r"([cft])([+-]\d+(?:\.\d+)?)", token)
    if m:
        return m.group(1), float(m.group(2))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive single-leg kinematics test"
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument(
        "--leg",
        type=int,
        default=1,
        choices=[l.value for l in Leg],
        help="Leg index 1-6 (default: 1 = FRONT_RIGHT)",
    )
    args = parser.parse_args()

    try:
        leg = Leg(args.leg)
    except ValueError:
        print(f"Unknown leg {args.leg}")
        sys.exit(1)

    ids = {joint: servo_id(leg, joint) for joint in Joint}
    joint_map = {"c": Joint.COXA, "f": Joint.FEMUR, "t": Joint.TIBIA}
    speed_cm_s = 10.0
    jog_step_cm = 0.5

    limits = SoftLimits.load()

    print(
        f"Leg: {leg.name}  (coxa={ids[Joint.COXA]}, femur={ids[Joint.FEMUR]}, tibia={ids[Joint.TIBIA]})"
    )
    print(f"Port: {args.port}")
    if limits:
        print(
            f"Soft limits: coxa [{limits.coxa.min_deg:+.1f}°..{limits.coxa.max_deg:+.1f}°]"
            f"  femur [{limits.femur.min_deg:+.1f}°..{limits.femur.max_deg:+.1f}°]"
            f"  tibia [{limits.tibia.min_deg:+.1f}°..{limits.tibia.max_deg:+.1f}°]"
        )
    else:
        print("Soft limits: none (run calibrate_soft_limits.py to configure)")
    print("Type 'neutral' to go to all-2048, or 'help' for commands.\n")

    with SerialTransport(args.port) as transport:
        bus = ST3020Bus(transport)

        while True:
            try:
                raw = input("leg> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            lower = raw.lower()

            if not raw or lower in ("help", "?", "h"):
                print(__doc__)
                continue

            if lower in ("q", "quit", "exit"):
                break

            if lower == "neutral":
                try:
                    move_smooth(bus, ids, (17.4, 0.0, -15.0), speed_cm_s, limits)
                except (IKError, SoftLimitError) as e:
                    print(f"  {type(e).__name__}: {e}")
                time.sleep(0.05)
                print_status(bus, ids)
                continue

            if lower == "read":
                print_status(bus, ids)
                continue

            if lower == "pos":
                print_position(bus, ids)
                continue

            if lower == "off":
                for sid in ids.values():
                    bus.torque_enable(sid, False)
                print("Torque disabled — you can move joints by hand.")
                continue

            if lower == "on":
                for sid in ids.values():
                    bus.torque_enable(sid, True)
                print("Torque enabled.")
                continue

            if lower == "jog":
                jog_mode(bus, ids, jog_step_cm, limits)
                continue

            parts = raw.split()

            if len(parts) == 2 and parts[0].lower() == "step":
                try:
                    v = float(parts[1])
                    if v <= 0:
                        raise ValueError
                    jog_step_cm = v
                    print(f"  Jog step set to {jog_step_cm} cm")
                except ValueError:
                    print("  Usage: step <cm>  (positive number)")
                continue

            if len(parts) == 2 and parts[0].lower() == "speed":
                try:
                    v = float(parts[1])
                    if v <= 0:
                        raise ValueError
                    speed_cm_s = v
                    print(f"  Speed set to {speed_cm_s} cm/s")
                except ValueError:
                    print("  Usage: speed <cm/s>  (positive number)")
                continue

            # ik <x> <y> <z>  or  ik +dx +dy +dz
            if parts and parts[0].lower() == "ik":
                result = parse_ik_args(parts[1:])
                if result is None:
                    print("Usage:  ik <x> <y> <z>   or   ik +dx +dy +dz")
                    continue
                values, relative = result
                if relative:
                    current = read_foot_position(bus, ids)
                    if current is None:
                        continue
                    target = (
                        current[0] + values[0],
                        current[1] + values[1],
                        current[2] + values[2],
                    )
                else:
                    target = (values[0], values[1], values[2])
                try:
                    move_smooth(bus, ids, target, speed_cm_s, limits)
                except (IKError, SoftLimitError) as e:
                    print(f"  {type(e).__name__}: {e}")
                    continue
                time.sleep(0.05)
                print_status(bus, ids)
                print(
                    f"  → foot target:  x={target[0]:+.2f}  y={target[1]:+.2f}  z={target[2]:+.2f}  cm"
                )
                print()
                continue

            # c/f/t <tick>
            if len(parts) == 2 and parts[0].lower() in joint_map:
                try:
                    tick = int(parts[1])
                except ValueError:
                    print("Tick must be an integer.")
                    continue
                joint = joint_map[parts[0].lower()]
                if limits:
                    angle = tick_to_angle(JOINT_NAMES[joint], tick)
                    jlim = getattr(limits, JOINT_NAMES[joint])
                    if not jlim.contains(angle):
                        print(f"  SoftLimitError: {JOINT_NAMES[joint]} {angle:+.1f}° outside [{jlim.min_deg:+.1f}°, {jlim.max_deg:+.1f}°]")
                        continue
                move(bus, ids[joint], tick)
                time.sleep(0.1)
                print_status(bus, ids)
                continue

            # c+deg / f+deg / t+deg
            parsed = parse_offset_cmd(lower)
            if parsed:
                key, deg = parsed
                joint = joint_map[key]
                if limits:
                    jlim = getattr(limits, JOINT_NAMES[joint])
                    if not jlim.contains(deg):
                        print(f"  SoftLimitError: {JOINT_NAMES[joint]} {deg:+.1f}° outside [{jlim.min_deg:+.1f}°, {jlim.max_deg:+.1f}°]")
                        continue
                tick = angle_to_tick(JOINT_NAMES[joint], deg)
                move(bus, ids[joint], tick)
                time.sleep(0.1)
                print_status(bus, ids)
                continue

            print(f"Unknown command: {raw!r}  (type 'help' for command list)")


if __name__ == "__main__":
    main()
