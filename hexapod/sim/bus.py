"""
Virtual SCS bus: parses instruction packets and answers as the servos would.

Supported instructions: PING, READ, WRITE, REG_WRITE + ACTION, SYNC_WRITE and
SYNC_READ.  Broadcasts (ID 0xFE) are applied to every servo and answered with
silence, exactly like the real bus.
"""

from dataclasses import dataclass

from hexapod.robot.config import ALL_SERVO_IDS
from hexapod.servo import registers as R
from hexapod.servo.protocol import build_packet

from .servo import VirtualServo


def _checksum_ok(packet: bytes) -> bool:
    length = packet[3]
    body = packet[2 : 3 + length]  # id, len, instruction, params
    return ~sum(body) & 0xFF == packet[3 + length]


@dataclass
class BusStats:
    packets: int = 0
    responses: int = 0
    bad_checksum: int = 0
    unknown_id: int = 0
    unknown_instruction: int = 0
    last_instruction: str = "—"


_INSTRUCTION_NAMES = {
    R.INST_PING: "PING",
    R.INST_READ: "READ",
    R.INST_WRITE: "WRITE",
    R.INST_REG_WRITE: "REG_WRITE",
    R.INST_REG_ACTION: "ACTION",
    R.INST_SYNC_READ: "SYNC_READ",
    R.INST_SYNC_WRITE: "SYNC_WRITE",
}


class VirtualBus:
    """Byte-stream front end for a set of VirtualServo instances."""

    def __init__(
        self,
        servo_ids: list[int] | None = None,
        *,
        missing: set[int] | None = None,
    ) -> None:
        ids = ALL_SERVO_IDS if servo_ids is None else servo_ids
        absent = missing or set()
        self.servos: dict[int, VirtualServo] = {
            sid: VirtualServo(sid) for sid in ids if sid not in absent
        }
        self.stats = BusStats()
        self._buf = bytearray()
        self._pending: dict[int, tuple[int, bytes]] = {}  # REG_WRITE staging

    # --- simulation ---

    def step(self, dt: float) -> None:
        for servo in self.servos.values():
            servo.step(dt)

    # --- byte stream ---

    def feed(self, data: bytes) -> list[bytes]:
        """Consume received bytes, return the status packets to send back."""
        self._buf.extend(data)
        responses: list[bytes] = []

        while True:
            start = self._buf.find(b"\xff\xff")
            if start < 0:
                # Keep a trailing 0xFF: it may be the first header byte
                del self._buf[: max(0, len(self._buf) - 1)]
                break
            if start:
                del self._buf[:start]
            if len(self._buf) < 4:
                break
            if self._buf[2] == 0xFF:
                # Run of 0xFF bytes — no servo has ID 0xFF, so the real
                # header starts one byte later.
                del self._buf[:1]
                continue

            total = self._buf[3] + 4
            if len(self._buf) < total:
                break

            packet = bytes(self._buf[:total])
            if not _checksum_ok(packet):
                self.stats.bad_checksum += 1
                del self._buf[:2]  # resync: this was not a real header
                continue

            del self._buf[:total]
            responses.extend(self._handle(packet))

        self.stats.responses += len(responses)
        return responses

    def _handle(self, packet: bytes) -> list[bytes]:
        servo_id = packet[2]
        length = packet[3]
        instruction = packet[4]
        params = packet[5 : 3 + length]

        self.stats.packets += 1
        self.stats.last_instruction = _INSTRUCTION_NAMES.get(
            instruction, f"0x{instruction:02X}"
        )

        if instruction == R.INST_SYNC_WRITE:
            self._sync_write(params)
            return []
        if instruction == R.INST_SYNC_READ:
            return self._sync_read(params)

        if servo_id == R.BROADCAST_ID:
            self._broadcast(instruction, params)
            return []

        servo = self.servos.get(servo_id)
        if servo is None:
            self.stats.unknown_id += 1
            return []

        if instruction == R.INST_PING:
            return [self._status(servo_id, b"")]
        if instruction == R.INST_READ:
            addr, count = params[0], params[1]
            return [self._status(servo_id, servo.read(addr, count))]
        if instruction == R.INST_WRITE:
            servo.write(params[0], params[1:])
            return [self._status(servo_id, b"")]
        if instruction == R.INST_REG_WRITE:
            self._pending[servo_id] = (params[0], params[1:])
            return [self._status(servo_id, b"")]
        if instruction == R.INST_REG_ACTION:
            self._action(servo_id)
            return [self._status(servo_id, b"")]

        self.stats.unknown_instruction += 1
        return []

    # --- instruction handlers ---

    def _broadcast(self, instruction: int, params: bytes) -> None:
        if instruction == R.INST_WRITE:
            for servo in self.servos.values():
                servo.write(params[0], params[1:])
        elif instruction == R.INST_REG_WRITE:
            for sid in self.servos:
                self._pending[sid] = (params[0], params[1:])
        elif instruction == R.INST_REG_ACTION:
            for sid in list(self.servos):
                self._action(sid)

    def _action(self, servo_id: int) -> None:
        staged = self._pending.pop(servo_id, None)
        servo = self.servos.get(servo_id)
        if staged and servo:
            servo.write(staged[0], staged[1])

    def _sync_write(self, params: bytes) -> None:
        if len(params) < 2:
            return
        addr, per_servo = params[0], params[1]
        body = params[2:]
        stride = per_servo + 1
        for i in range(0, len(body) - per_servo, stride):
            servo = self.servos.get(body[i])
            if servo:
                servo.write(addr, body[i + 1 : i + 1 + per_servo])

    def _sync_read(self, params: bytes) -> list[bytes]:
        if len(params) < 2:
            return []
        addr, count = params[0], params[1]
        out = []
        for sid in params[2:]:
            servo = self.servos.get(sid)
            if servo:
                out.append(self._status(sid, servo.read(addr, count)))
        return out

    @staticmethod
    def _status(servo_id: int, data: bytes, error: int = 0) -> bytes:
        # A status packet has the same layout as an instruction packet,
        # with the instruction byte replaced by the error byte.
        return build_packet(servo_id, error, list(data))
