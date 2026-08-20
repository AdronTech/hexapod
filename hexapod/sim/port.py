"""
Virtual serial port backed by a pseudo-terminal.

open() allocates a pty pair; the slave side looks like an ordinary serial
device (/dev/pts/N), so pyserial — and therefore every script in this repo —
can open it with no changes.  An optional symlink gives it a stable name.
"""

import os
import termios
import tty
from pathlib import Path
from typing import Self


class VirtualSerialPort:
    def __init__(self, link: Path | None = None) -> None:
        self._link = link
        self._master: int | None = None
        self._slave: int | None = None
        self._device: str | None = None

    def open(self) -> str:
        master, slave = os.openpty()
        # Raw mode: no echo, no CR/LF translation, no line buffering —
        # anything else would corrupt the binary protocol.
        tty.setraw(slave)
        termios.tcflush(slave, termios.TCIOFLUSH)
        os.set_blocking(master, False)

        self._master, self._slave = master, slave
        self._device = os.ttyname(slave)

        if self._link is not None:
            self._make_link(self._device)
        return self._device

    def _make_link(self, device: str) -> None:
        link = self._link
        assert link is not None
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise FileExistsError(f"{link} exists and is not a symlink")
        link.symlink_to(device)

    def close(self) -> None:
        if self._link is not None and self._link.is_symlink():
            try:
                self._link.unlink()
            except OSError:
                pass
        for fd in (self._master, self._slave):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._master = self._slave = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- I/O ---

    def fileno(self) -> int:
        if self._master is None:
            raise RuntimeError("Port is not open")
        return self._master

    def read(self, size: int = 4096) -> bytes:
        try:
            return os.read(self.fileno(), size)
        except BlockingIOError:
            return b""

    def write(self, data: bytes) -> None:
        fd = self.fileno()
        view = memoryview(data)
        while view:
            try:
                written = os.write(fd, view)
            except BlockingIOError:
                return  # client is not draining the port; drop the rest
            view = view[written:]

    @property
    def device(self) -> str:
        if self._device is None:
            raise RuntimeError("Port is not open")
        return self._device

    @property
    def link(self) -> Path | None:
        return self._link
