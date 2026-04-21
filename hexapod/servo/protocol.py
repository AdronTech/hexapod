"""
SCS protocol packet building and parsing.

Packet format:
  TX:  FF FF ID LEN INSTR [PARAMS...] CHECKSUM
  RX:  FF FF ID LEN ERROR [DATA...]   CHECKSUM

  LEN      = len(PARAMS or DATA) + 2  (accounts for INSTR/ERROR + CHECKSUM)
  CHECKSUM = ~(ID + LEN + INSTR + sum(PARAMS)) & 0xFF
"""

from dataclasses import dataclass
from .registers import BROADCAST_ID, INST_SYNC_WRITE


class ProtocolError(Exception):
    pass


@dataclass
class StatusPacket:
    servo_id: int
    error: int
    data: bytes


def build_packet(servo_id: int, instruction: int, params: list[int]) -> bytes:
    length = len(params) + 2
    checksum = ~(servo_id + length + instruction + sum(params)) & 0xFF
    return bytes([0xFF, 0xFF, servo_id, length, instruction] + params + [checksum])


def build_sync_write(
    start_addr: int,
    data_per_servo: int,
    servo_data: list[tuple[int, list[int]]],
) -> bytes:
    params: list[int] = [start_addr, data_per_servo]
    for sid, data in servo_data:
        params.append(sid)
        params.extend(data)
    return build_packet(BROADCAST_ID, INST_SYNC_WRITE, params)


def parse_status_packet(raw: bytes) -> StatusPacket:
    idx = raw.find(b"\xff\xff")
    if idx < 0:
        raise ProtocolError("No 0xFF 0xFF header found")
    raw = raw[idx:]

    if len(raw) < 6:
        raise ProtocolError(f"Packet too short: {len(raw)} bytes")

    servo_id = raw[2]
    length = raw[3]   # = len(data) + 2  (ERROR byte + CHECKSUM byte)
    error = raw[4]

    if len(raw) < length + 4:
        raise ProtocolError(f"Incomplete packet: need {length + 4} bytes, have {len(raw)}")

    data = bytes(raw[5 : 3 + length])      # length - 2 payload bytes
    checksum_received = raw[3 + length]

    checksum_calc = ~(servo_id + length + error + sum(data)) & 0xFF
    if checksum_calc != checksum_received:
        raise ProtocolError(
            f"Checksum mismatch: calculated {checksum_calc:#04x}, received {checksum_received:#04x}"
        )

    return StatusPacket(servo_id=servo_id, error=error, data=data)


def encode_word(value: int) -> tuple[int, int]:
    """Encode a 16-bit value as (low_byte, high_byte)."""
    return (value & 0xFF, (value >> 8) & 0xFF)


def decode_word(low: int, high: int) -> int:
    """Decode two bytes (little-endian) into a 16-bit unsigned value."""
    return (low & 0xFF) | ((high & 0xFF) << 8)


def decode_signed(value: int) -> int:
    """Decode a value where bit 15 is the sign (magnitude, not two's complement)."""
    if value & (1 << 15):
        return -(value & 0x7FFF)
    return value


def encode_signed(value: int) -> int:
    """Encode a signed int into SCS sign-magnitude 16-bit word (mirror of decode_signed)."""
    if value < 0:
        return ((-value) & 0x7FFF) | 0x8000
    return value & 0x7FFF
