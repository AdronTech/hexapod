#!/usr/bin/env python3
"""
Hexapod simulator — a virtual servo bus on a virtual serial port.

Usage:
    uv run scripts/simulator.py [--link /tmp/hexapod-sim] [--http-port 8090]

Emulates all 18 ST3020 servos behind a pseudo-terminal, so any script in this
repo can be pointed at it instead of the real robot:

    uv run scripts/simulator.py                       # terminal 1
    uv run scripts/web_control.py --port /tmp/hexapod-sim   # terminal 2

A 3D viewer at http://<bind>:<http-port> shows what the simulated robot is
doing: joint angles reconstructed from the servo ticks, ground contact and
the support polygon.
"""

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from hexapod.robot.config import ALL_SERVO_IDS
from hexapod.sim import SimulatorThread, VirtualBus

DEFAULT_LINK = "/tmp/hexapod-sim"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8090
STREAM_HZ = 30

WEB_DIR = Path(__file__).parent / "sim_web"


# ---------------------------------------------------------------------------
# Viewer app
# ---------------------------------------------------------------------------


def build_app(sim: SimulatorThread) -> Starlette:

    @asynccontextmanager
    async def lifespan(app: Starlette):
        yield

    async def index(request: Request) -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        async def send_loop() -> None:
            try:
                while True:
                    await ws.send_text(json.dumps(sim.snapshot()))
                    await asyncio.sleep(1.0 / STREAM_HZ)
            except (WebSocketDisconnect, ConnectionError, RuntimeError):
                pass

        send_task = asyncio.create_task(send_loop())
        try:
            async for raw in ws.iter_text():
                try:
                    if json.loads(raw).get("type") == "reset_world":
                        sim.reset_world_pose()
                except (json.JSONDecodeError, AttributeError):
                    pass
        except (WebSocketDisconnect, ConnectionError, RuntimeError):
            pass
        finally:
            send_task.cancel()

    return Starlette(
        lifespan=lifespan,
        routes=[
            Route("/", index),
            WebSocketRoute("/ws", ws_endpoint),
            Mount("/static", StaticFiles(directory=WEB_DIR), name="static"),
        ],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_missing(value: str) -> set[int]:
    if not value:
        return set()
    return {int(part) for part in value.replace(",", " ").split()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hexapod servo bus simulator")
    parser.add_argument(
        "--link",
        default=DEFAULT_LINK,
        help=f"Symlink to the virtual port (default {DEFAULT_LINK})",
    )
    parser.add_argument(
        "--no-link", action="store_true", help="Use the raw /dev/pts path only"
    )
    parser.add_argument("--bind", default=DEFAULT_HTTP_HOST, help="Viewer bind address")
    parser.add_argument(
        "--http-port", default=DEFAULT_HTTP_PORT, type=int, help="Viewer HTTP port"
    )
    parser.add_argument("--no-viewer", action="store_true", help="Serial bus only")
    parser.add_argument(
        "--missing",
        default="",
        help="Servo IDs that should not respond, e.g. --missing 23,31",
    )
    args = parser.parse_args()

    missing = parse_missing(args.missing)
    unknown = missing - set(ALL_SERVO_IDS)
    if unknown:
        parser.error(f"Unknown servo IDs: {sorted(unknown)}")

    link = None if args.no_link else Path(args.link)
    sim = SimulatorThread(link=link, bus=VirtualBus(missing=missing))
    device = sim.open()
    sim.start()

    port_name = str(link) if link else device
    print(f"Virtual servo bus ready: {device}")
    if link:
        print(f"  symlink: {link}")
    if missing:
        print(f"  not responding: {sorted(missing)}")
    print("\nConnect the control software with:")
    print(f"  uv run scripts/web_control.py --port {port_name}")

    if args.no_viewer:
        print("\nCtrl-C to stop.")
        try:
            sim.join()
        except KeyboardInterrupt:
            pass
        finally:
            sim.shutdown()
        return

    print(f"\nViewer: http://{args.bind}:{args.http_port}")
    app = build_app(sim)
    try:
        uvicorn.run(app, host=args.bind, port=args.http_port, log_level="warning")
    finally:
        sim.shutdown()


if __name__ == "__main__":
    main()
