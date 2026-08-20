# Control Recordings

`web_control.py --record` writes one JSON object per control tick to a JSONL
file. A recording captures the whole decision chain — gamepad input, resolved
command velocities, body pose, per-leg gait state and the servo goals that went
on the wire — so a misbehaving run can be picked apart offline, with or without
the robot attached.

## Recording

```bash
uv run scripts/web_control.py --port /tmp/hexapod-sim --record
```

`--record` with no argument writes `recordings/<timestamp>.jsonl`; pass a path
to choose your own. Recording starts with the process and runs continuously,
including while idle — glitches are not polite enough to announce themselves.

When you see something go wrong, hit **📌 Mark** in the web UI header. That
stamps the moment into the recording; the analyzer prints every marked moment
tick by tick. The button flashes blue to confirm.

The file is flushed once a second, so killing the process loses at most the last
second of data.

## Analysing

```bash
uv run scripts/analyze_recording.py recordings/20260820-233307.jsonl
uv run scripts/analyze_recording.py rec.jsonl --legs          # per-leg breakdown
uv run scripts/analyze_recording.py rec.jsonl --window 15 17  # raw ticks in a range
```

The summary covers:

| Section | What it answers |
|---------|-----------------|
| Session | duration, mode split, loop timing and overruns, config that changed mid-run |
| Stepping | lifts, swing/stance durations, foot error at lift and at landing, how often the stability guard was bypassed |
| IK failures | how many ticks sent nothing to the servos, and the longest frozen run |
| Discontinuities | biggest per-frame servo goal jumps, and grounded feet that teleported |
| Marked moments | a tick-by-tick dump either side of every 📌 |

The tick dump renders the swing state as one column per leg in ring order
(`FR MR RR RL ML FL`), `^` for swinging and `.` for grounded, which makes gait
patterns and stability violations visible at a glance:

```
       t mode  swing    pose x,y,yaw              max err  note
   16.34 free  .^.^.^     106.9,  14.7,   93.0       18.68  EMERGENCY
   16.39 free  ^^^...     106.9,  14.7,   93.0       18.68  IK-FAIL REAR_RIGHT  EMERGENCY
```

`^^^...` means the whole right side is in the air at once.

## Numbers worth knowing

- **foot err at land** — how far a foot is from its neutral the moment it
  touches down. If this reaches `step_threshold`, the free gait re-lifts the
  leg immediately and stops being event-driven.
- **emergency lifts** — lifts taken with the foot past `FREE_STEP_EMERGENCY`
  (6 cm), which bypasses the adjacency guard. A high rate means the stability
  constraint is not being honoured.
- **legs swinging** — the histogram should sit at 1–2 for a calm free gait;
  parked at 3 means the gait is saturated and every leg is behind.
- **servo goal jumps** — ticks of change per 50 ms frame. Smooth motion is tens
  of ticks; hundreds is a visible snap (1 tick ≈ 0.088°).

## File format

```
{"type":"header","v":1,"created":…,"control_hz":20,"dt":0.05,"servo_ids":{…}}
{"type":"tick","t":12.35,"i":247,"mode":"free","ax":[…],"btn":[4],"trig":[0,0],
 "cfg":{…},"gait":"free","cmd":[vx,vy,omega],"pose":[x,y,z,roll,pitch,yaw],
 "legs":{"FRONT_RIGHT":{"foot":[x,y,z],"neutral":[x,y],"err":…,"swing":false,
 "t":0.0,"target":[…],"due":false,"emergency":false}, …},
 "ticks":{"11":2048,…},"err":"…"}
{"type":"mark","t":13.2,"n":1,"label":"","tick":264}
{"type":"footer","t":31.7,"ticks":634,"marks":2}
```

`mode` is one of `idle`, `pose`, `walk`, `free`. `cmd` is the world-frame
velocity fed to the gait. `legs` is absent outside walk/free mode, `ticks` is
absent on a tick that sent nothing (an IK failure, or pose mode with no input),
and `due`/`emergency` are free-gait only. Positions are cm, angles degrees.

Everything is plain JSONL — `jq` works fine if you want to slice it yourself:

```bash
jq -c 'select(.type=="tick" and .err) | {t, err}' rec.jsonl
```
