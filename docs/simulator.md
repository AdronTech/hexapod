# Hexapod Simulator

A virtual robot: all 18 ST3020 servos emulated behind a virtual serial port, so
the control software runs unchanged with no hardware attached.

```
scripts/web_control.py ──serial──▶ /tmp/hexapod-sim ──pty──▶ virtual servo bus
                                                                    │
                                            3D viewer ◀── websocket ─┘
```

## Running it

```bash
uv run scripts/simulator.py                              # terminal 1
uv run scripts/web_control.py --port /tmp/hexapod-sim    # terminal 2
```

Then open the controller at http://localhost:8080 and the simulator viewer at
http://localhost:8090. Everything works as with the real robot: stand, sit,
pose, the gaits, storage mode.

Options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--link PATH` | `/tmp/hexapod-sim` | Symlink pointing at the allocated pty |
| `--no-link` | — | Use the raw `/dev/pts/N` path only (printed at startup) |
| `--bind` / `--http-port` | `127.0.0.1` / `8090` | Viewer HTTP endpoint |
| `--no-viewer` | — | Serial bus only, no web viewer |
| `--missing 23,31` | — | Servo IDs that stay silent, to test error handling |

Other scripts take `--port` too, e.g.
`uv run scripts/leg_test.py --port /tmp/hexapod-sim`.
`ping_all.py`, `monitor_positions.py` and `calibrate.py` still have the port
hardcoded at the top of the file.

## What is simulated

**Protocol** (`hexapod/sim/bus.py`) — PING, READ, WRITE, REG_WRITE + ACTION,
SYNC_WRITE and SYNC_READ, with the same packet framing, checksums and status
replies as the real bus. Broadcasts stay silent, unknown IDs time out, and
bad checksums are counted and dropped. The parser re-syncs on garbage, so a
partially written packet does not wedge the stream.

**Servos** (`hexapod/sim/servo.py`) — a 256-byte register file per servo plus a
trapezoidal motion model: `GOAL_SPEED` caps velocity (0 = unlimited),
`ACC` caps acceleration (unit = 100 ticks/s²), and the joint decelerates to
land exactly on the goal. Position feedback, speed, load, voltage, a slow
temperature model and the moving flag are all reported back. With torque
disabled the joint holds still; writing 128 to `TORQUE_ENABLE` performs the
same centre calibration as the firmware.

**Robot state** (`hexapod/sim/model.py`) — forward kinematics from the raw
ticks gives every joint position, and the body then settles onto its feet: of
all the planes spanned by three feet, the robot rests on the one that leaves no
foot below it and that the body's vertical meets highest up (a face of the
feet's lower convex hull). Levelling that plane is what tilts the body, so
rolling or pitching over planted feet shows up as a tilted body on flat ground
rather than as feet lifting off — the recovered roll and pitch match what the
controller commanded to within a fraction of a degree. With the legs folded
above the body (storage pose) there is no such plane and the robot lies on its
belly. The body centre is then tested against the convex hull of the feet that
are down, which is what the viewer's *stable / tipping* badge and the
support-margin readout show.

**Where it walks to** — the robot moves through the world, it does not walk on
the spot. A foot that stays on the ground between two frames is a fixed point in
the world, so the rigid transform those feet describe *is* the body's motion;
fitting it every frame accumulates a world pose (`Odometry`). This is the same
trick a real robot uses for leg odometry, and it stays accurate here because the
gait plants its stance feet in the world frame: walking forward for 4 s at
15 cm/s puts the body 59.5 cm ahead, a 3 s turn at 60 °/s comes out as 179°.

## What is not simulated

There is no physics: the ground is always flat, and nothing falls over —
tipping is reported, not enacted, and the body tilt is inferred from the feet
rather than integrated from forces. Nothing sags either, so a limp robot keeps
its shape; sitting looks like standing in the viewer because sit commands the
neutral pose and then cuts torque, and only gravity would fold the legs on the
real machine. Servo torque, momentum, foot slip and ground compliance are
ignored, and the bus has no baud-rate delay, so timing is optimistic compared to
the real 1 Mbaud bus.

## The viewer

`http://localhost:8090` renders the reconstructed robot together with the live
servo table (tick, goal, temperature, torque state) and bus counters — useful on
its own for watching what the control software actually puts on the wire.

The overlay reads out body height, roll/pitch, support margin, feet on the
ground, world position and heading. The camera follows the robot, so the ground
grid slides past as it walks, with the world origin marked and a fading trail
behind it. Drag to orbit, wheel to
zoom, double-click to recentre the camera; *Reset position* puts the robot back
at the origin and clears the trail.
