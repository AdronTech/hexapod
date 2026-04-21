import time

from .st3020 import ST3020Bus, PositionCommand


class MotionPlayer:
    """Smooth motion via dense position streaming at a fixed tick rate.

    Each tick sends the next interpolated position with speed=0 (unlimited),
    so the servo moves as fast as possible to each nearby target. At 50 Hz
    with small steps the motion appears continuous.

    Current positions are remembered between move() calls so consecutive
    moves start from where the previous one ended without re-reading the bus.
    """

    def __init__(self, bus: ST3020Bus, tick_hz: float = 50, acc: int = 0):
        self._bus = bus
        self._interval = 1.0 / tick_hz
        self._acc = acc
        self._current: dict[int, float] = {}

    def move(self, targets: list[tuple[int, int, int]]) -> None:
        """Block until all servos reach their targets.

        targets: list of (servo_id, position, speed_ticks_per_s)
        """
        for servo_id, _, _ in targets:
            if servo_id not in self._current:
                self._current[servo_id] = float(self._bus.read_position(servo_id))

        goal  = {sid: float(pos) for sid, pos, _   in targets}
        speed = {sid: float(spd) for sid, _,   spd in targets}

        t_next = time.monotonic()
        while True:
            commands = []
            all_done = True
            for sid, target in goal.items():
                cur  = self._current[sid]
                step = speed[sid] * self._interval
                remaining = target - cur
                if abs(remaining) <= step:
                    nxt = target
                else:
                    nxt = cur + step if remaining > 0 else cur - step
                    all_done = False
                self._current[sid] = nxt
                commands.append(PositionCommand(sid, round(nxt), speed=0, acc=self._acc))

            self._bus.sync_write_position(commands)

            if all_done:
                break

            t_next += self._interval
            wait = t_next - time.monotonic()
            if wait > 0:
                time.sleep(wait)
