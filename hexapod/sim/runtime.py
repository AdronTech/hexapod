"""
Simulator runtime: pumps the virtual serial port and advances the servo model.

The loop waits on the pty with a short timeout, so incoming packets are
answered within a fraction of a millisecond while the motion model keeps
stepping at a steady rate even when the bus is idle.
"""

import select
import threading
import time
from pathlib import Path

from .bus import VirtualBus
from .model import Odometry, RobotState, WorldPose, robot_state
from .port import VirtualSerialPort

TICK_TIMEOUT = 0.002  # s — max wait for incoming bytes per loop iteration
MAX_STEP = 0.05  # s — clamp the integration step after a stall
STATE_HZ = 50  # rate at which the world state is reconstructed


class SimulatorThread(threading.Thread):
    """Runs the virtual bus on a pty until stop() is called."""

    def __init__(
        self,
        link: Path | None = None,
        bus: VirtualBus | None = None,
    ) -> None:
        super().__init__(daemon=True, name="simulator")
        self.bus = bus or VirtualBus()
        self.port = VirtualSerialPort(link)
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._error: str | None = None
        self._last_rx = 0.0
        self._odometry = Odometry()
        self._state = robot_state(self.positions())
        self._world = self._odometry.update(self._state)

    # --- lifecycle ---

    def open(self) -> str:
        """Allocate the pty up front so the caller can print the device name."""
        device = self.port.open()
        self._ready.set()
        return device

    def stop(self) -> None:
        self._stop_event.set()

    def shutdown(self, timeout: float = 1.0) -> None:
        """Stop the loop and release the pty (and its symlink)."""
        self.stop()
        if self.is_alive():
            self.join(timeout)
        self.port.close()

    def run(self) -> None:
        if not self._ready.is_set():
            self.open()
        fd = self.port.fileno()
        previous = time.monotonic()
        next_state = previous
        try:
            while not self._stop_event.is_set():
                ready, _, _ = select.select([fd], [], [], TICK_TIMEOUT)
                now = time.monotonic()
                self.bus.step(min(now - previous, MAX_STEP))
                previous = now

                if now >= next_state:
                    self._track_world()
                    next_state = now + 1.0 / STATE_HZ

                if ready:
                    data = self.port.read()
                    if data:
                        self._last_rx = now
                        for response in self.bus.feed(data):
                            self.port.write(response)
        except OSError as e:
            with self._lock:
                self._error = str(e)
        finally:
            self.port.close()

    # --- observation ---

    @property
    def device(self) -> str:
        self._ready.wait(timeout=5.0)
        return self.port.device

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def positions(self) -> dict[int, int]:
        return {sid: round(servo.pos) for sid, servo in sorted(self.bus.servos.items())}

    def _track_world(self) -> None:
        state = robot_state(self.positions())
        world = self._odometry.update(state)
        with self._lock:
            self._state = state
            self._world = world

    def state(self) -> tuple[RobotState, WorldPose]:
        """Latest reconstructed robot state and its pose in the world."""
        with self._lock:
            return self._state, self._world

    def reset_world_pose(self) -> None:
        """Put the robot back at the world origin."""
        with self._lock:
            self._odometry.reset()
            self._world = self._odometry.pose

    def snapshot(self) -> dict:
        """JSON-ready view of the whole simulator, for the viewer UI."""
        now = time.monotonic()
        stats = self.bus.stats
        state, world = self.state()
        servos = [
            {
                "id": sid,
                "pos": round(s.pos),
                "goal": int(s.goal),
                "speed": round(s.vel),
                "torque": s.torque_on,
                "moving": s.moving,
                "temp": round(s.temperature, 1),
            }
            for sid, s in sorted(self.bus.servos.items())
        ]
        return {
            "device": self.port.device,
            "connected": self._last_rx > 0.0 and (now - self._last_rx) < 1.0,
            "servos": servos,
            "robot": state.to_dict(world),
            "bus": {
                "packets": stats.packets,
                "responses": stats.responses,
                "bad_checksum": stats.bad_checksum,
                "unknown_id": stats.unknown_id,
                "last": stats.last_instruction,
            },
        }
