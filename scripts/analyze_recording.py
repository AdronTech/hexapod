#!/usr/bin/env python3
"""
Analyse a control recording produced by `web_control.py --record`.

    uv run scripts/analyze_recording.py recordings/20260820-101500.jsonl

Reads the JSONL tick log and reports what the control loop actually did:
loop timing, gait step statistics, stability-guard violations, IK failures and
the largest discontinuities in the servo commands.  Anything the operator
flagged with the Mark button is printed tick by tick.

Options:
    --window T0 T1   dump every tick in a time range instead of a summary
    --legs           per-leg breakdown of the step statistics
    --top N          how many discontinuities to list (default 10)
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Adjacent legs must not swing together — mirrors _ADJACENT in hexapod/gait.py.
_RING = [
    "FRONT_RIGHT",
    "MID_RIGHT",
    "REAR_RIGHT",
    "REAR_LEFT",
    "MID_LEFT",
    "FRONT_LEFT",
]
_ADJ = {leg: {_RING[(i - 1) % 6], _RING[(i + 1) % 6]} for i, leg in enumerate(_RING)}

MIN_GROUNDED = 3  # the free gait promises at least this many feet down


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(path: Path) -> tuple[dict, list[dict], list[dict]]:
    header: dict = {}
    ticks: list[dict] = []
    marks: list[dict] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A recording cut short mid-write loses only its last line.
                print(f"warning: skipping malformed line {lineno}", file=sys.stderr)
                continue
            kind = rec.get("type")
            if kind == "header":
                header = rec
            elif kind == "tick":
                ticks.append(rec)
            elif kind == "mark":
                marks.append(rec)
    return header, ticks, marks


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:5.1f}%" if whole else "    —"


def _stats(vals: list[float]) -> str:
    if not vals:
        return "—"
    vals = sorted(vals)
    p95 = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
    return (
        f"n={len(vals):<5d} mean={statistics.fmean(vals):7.3f} "
        f"min={vals[0]:7.3f} p95={p95:7.3f} max={vals[-1]:7.3f}"
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def report_session(header: dict, ticks: list[dict], marks: list[dict]) -> None:
    print("=" * 78)
    print("SESSION")
    print("=" * 78)
    print(
        f"  recorded    {header.get('created', '?')}   schema v{header.get('v', '?')}"
    )
    if not ticks:
        print("  no ticks recorded")
        return
    dur = ticks[-1]["t"] - ticks[0]["t"]
    print(f"  duration    {dur:.1f} s over {len(ticks)} ticks   marks: {len(marks)}")

    modes = Counter(t["mode"] for t in ticks)
    print("  modes       " + "  ".join(f"{m}={n}" for m, n in modes.most_common()))

    dts = [b["t"] - a["t"] for a, b in pairwise(ticks)]
    nominal = header.get("dt", 0.05)
    late = sum(1 for d in dts if d > nominal * 1.5)
    print(f"  loop dt     {_stats(dts)}   (nominal {nominal:.3f})")
    print(f"  overruns    {late} ticks > 1.5x nominal  {_pct(late, len(dts))}")

    # Configuration that changed during the session matters for interpretation.
    for key in (
        "speed_cm",
        "speed_deg",
        "reach",
        "step_height",
        "step_time",
        "step_threshold",
    ):
        vals = {t["cfg"][key] for t in ticks if "cfg" in t}
        if len(vals) == 1:
            print(f"  {key:<14}{next(iter(vals))}")
        elif vals:
            print(f"  {key:<14}{min(vals)} … {max(vals)}  (changed mid-session)")


def report_steps(ticks: list[dict], per_leg: bool) -> None:
    """Lift/land bookkeeping — the core of the free-gait behaviour."""
    gaited = [t for t in ticks if "legs" in t]
    if not gaited:
        return
    print()
    print("=" * 78)
    print("STEPPING")
    print("=" * 78)

    swinging_prev: dict[str, bool] = {}
    lift_t: dict[str, float] = {}
    land_t: dict[str, float] = {}
    lifts: Counter = Counter()
    stance: dict[str, list[float]] = defaultdict(list)
    swing: dict[str, list[float]] = defaultdict(list)
    err_at_lift: list[float] = []
    err_at_land: list[float] = []
    emergency_lifts = 0
    adjacent_lifts = 0
    concurrent = Counter()
    understaffed = 0
    due_ticks = 0

    for t in gaited:
        legs = t["legs"]
        n_swing = sum(1 for d in legs.values() if d["swing"])
        concurrent[n_swing] += 1
        if 6 - n_swing < MIN_GROUNDED:
            understaffed += 1
        if any(d.get("due") for d in legs.values()):
            due_ticks += 1
        for name, d in legs.items():
            was = swinging_prev.get(name, False)
            now = d["swing"]
            if now and not was:
                lifts[name] += 1
                err_at_lift.append(d["err"])
                lift_t[name] = t["t"]
                if name in land_t:
                    stance[name].append(t["t"] - land_t[name])
                if d.get("emergency"):
                    emergency_lifts += 1
                if any(legs[a]["swing"] for a in _ADJ.get(name, ()) if a in legs):
                    adjacent_lifts += 1
            elif was and not now:
                err_at_land.append(d["err"])
                land_t[name] = t["t"]
                if name in lift_t:
                    swing[name].append(t["t"] - lift_t[name])
            swinging_prev[name] = now

    total_lifts = sum(lifts.values())
    all_stance = [v for vals in stance.values() for v in vals]
    all_swing = [v for vals in swing.values() for v in vals]
    print(f"  lifts             {total_lifts}")
    print(f"  swing duration    {_stats(all_swing)}  s")
    print(f"  stance duration   {_stats(all_stance)}  s")
    print(f"  foot err at lift  {_stats(err_at_lift)}  cm")
    print(f"  foot err at land  {_stats(err_at_land)}  cm")
    print()
    print(
        "  legs swinging     "
        + "  ".join(
            f"{k}:{v} ({_pct(v, len(gaited)).strip()})"
            for k, v in sorted(concurrent.items())
        )
    )
    print(
        f"  <{MIN_GROUNDED} feet grounded  {understaffed} ticks  {_pct(understaffed, len(gaited))}"
    )
    print(
        f"  emergency lifts   {emergency_lifts} / {total_lifts}  {_pct(emergency_lifts, total_lifts)}"
        "   (stability guard bypassed)"
    )
    print(
        f"  adjacent lifts    {adjacent_lifts} / {total_lifts}  {_pct(adjacent_lifts, total_lifts)}"
        "   (adjacent legs swinging together)"
    )
    print(
        f"  step-due ticks    {due_ticks}  {_pct(due_ticks, len(gaited))}"
        "   (a leg past threshold, held back)"
    )

    # A foot that lands already past the trigger threshold re-lifts on the same
    # tick: the leg never gets a stance phase and the gait stops being
    # event-driven.  Back-to-back swings show up as one swing of 2x step_time.
    step_times = {
        t["cfg"]["step_threshold"]: t["cfg"]["step_time"] for t in gaited if "cfg" in t
    }
    nominal_swing = max(step_times.values(), default=0.0)
    if err_at_land and nominal_swing:
        thr = max(step_times, default=float("nan"))
        bad = sum(1 for e in err_at_land if e >= thr)
        no_stance = sum(1 for v in all_swing if v > nominal_swing * 1.5)
        print()
        print(
            "  landing quality   how far from neutral a foot is when it touches down;"
        )
        print("                    at or past step_threshold the leg re-lifts at once")
        print(
            f"                    {bad} / {len(err_at_land)} landings past "
            f"threshold {thr}  {_pct(bad, len(err_at_land))}"
        )
        print(
            f"                    {no_stance} / {len(all_swing)} swings ran past "
            f"{nominal_swing * 1.5:.2f}s — landed and re-lifted with no stance"
        )

    if per_leg:
        print()
        print(f"  {'leg':<12}{'lifts':>6}{'stance mean':>13}{'swing mean':>12}")
        for name in _RING:
            if name not in lifts:
                continue
            sm = statistics.fmean(stance[name]) if stance[name] else float("nan")
            wm = statistics.fmean(swing[name]) if swing[name] else float("nan")
            print(f"  {name:<12}{lifts[name]:>6}{sm:>13.3f}{wm:>12.3f}")


def report_errors(ticks: list[dict]) -> None:
    errs = [t for t in ticks if t.get("err")]
    print()
    print("=" * 78)
    print("IK / SOFT-LIMIT FAILURES")
    print("=" * 78)
    print(
        f"  failed ticks      {len(errs)} / {len(ticks)}  {_pct(len(errs), len(ticks))}"
    )
    if not errs:
        return
    # A failed tick sends nothing to the servos: the robot freezes for a frame
    # while the gait state keeps advancing.
    msgs = Counter(t["err"].split(":")[0] for t in errs)
    for msg, n in msgs.most_common(8):
        print(f"    {n:5d}  {msg}")
    runs, run = [], 1
    for a, b in pairwise(errs):
        if b["i"] == a["i"] + 1:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    print(
        f"  longest run       {max(runs)} consecutive failed ticks "
        f"({max(runs) * 0.05:.2f} s of frozen output)"
    )
    print(f"  example           {errs[0]['err']}")


def report_discontinuities(ticks: list[dict], top: int) -> None:
    """Biggest instantaneous jumps — the visible 'glitch' signature."""
    print()
    print("=" * 78)
    print(f"LARGEST DISCONTINUITIES (top {top})")
    print("=" * 78)

    tick_jumps: list[tuple[int, float, str, int]] = []
    foot_jumps: list[tuple[float, float, str, int]] = []
    prev = None
    for t in ticks:
        if prev is not None:
            if "ticks" in t and "ticks" in prev:
                for sid, v in t["ticks"].items():
                    if sid in prev["ticks"]:
                        d = abs(v - prev["ticks"][sid])
                        tick_jumps.append((d, t["t"], sid, t["i"]))
            if "legs" in t and "legs" in prev:
                for name, d in t["legs"].items():
                    p = prev["legs"].get(name)
                    # Only grounded feet: a swinging foot is supposed to move.
                    if p is None or d["swing"] or p["swing"]:
                        continue
                    jump = math.dist(d["foot"], p["foot"])
                    foot_jumps.append((jump, t["t"], name, t["i"]))
        prev = t

    if tick_jumps:
        tick_jumps.sort(reverse=True)
        print("  servo goal jumps (ticks per control frame)")
        for d, t, sid, i in tick_jumps[:top]:
            print(
                f"    t={t:8.2f}s  tick#{i:<6d} servo {sid:<4s} Δ{d:5.0f} ticks"
                f"  ({d * 0.08789:6.1f}°)"
            )
    if foot_jumps:
        foot_jumps.sort(reverse=True)
        big = [f for f in foot_jumps if f[0] > 0.5]
        print()
        print(f"  grounded-foot teleports (should be ~0): {len(big)} over 0.5 cm")
        for d, t, name, i in foot_jumps[:top]:
            if d <= 0.01:
                break
            print(f"    t={t:8.2f}s  tick#{i:<6d} {name:<12s} {d:6.2f} cm")


def report_marks(ticks: list[dict], marks: list[dict], span: float = 1.0) -> None:
    if not marks:
        return
    print()
    print("=" * 78)
    print("MARKED MOMENTS")
    print("=" * 78)
    for m in marks:
        print(
            f"\n  --- mark #{m['n']} at t={m['t']:.2f}s "
            f"{m.get('label', '')} (±{span}s) ---"
        )
        _dump(
            [t for t in ticks if abs(t["t"] - m["t"]) <= span],
            highlight=m["t"],
        )


def _dump(window: list[dict], highlight: float | None = None) -> None:
    if not window:
        print("    (no ticks in range)")
        return
    print(
        f"    {'t':>8} {'mode':<5} {'swing':<8} {'pose x,y,yaw':<24} "
        f"{'max err':>8}  note"
    )
    for t in window:
        legs = t.get("legs", {})
        swing = "".join("^" if legs.get(n, {}).get("swing") else "." for n in _RING)
        maxerr = max((d["err"] for d in legs.values()), default=float("nan"))
        p = t.get("pose") or [float("nan")] * 6
        note = []
        if t.get("err"):
            note.append("IK-FAIL " + t["err"][:40])
        if any(d.get("emergency") for d in legs.values()):
            note.append("EMERGENCY")
        if sum(1 for d in legs.values() if d["swing"]) > 3:
            note.append("<3 GROUNDED")
        flag = (
            ">>" if highlight is not None and abs(t["t"] - highlight) < 0.03 else "  "
        )
        print(
            f" {flag} {t['t']:8.2f} {t['mode']:<5} {swing:<8} "
            f"{p[0]:7.1f},{p[1]:6.1f},{p[5]:7.1f}    {maxerr:8.2f}  "
            f"{' '.join(note)}"
        )
    print(
        "    swing columns: "
        + " ".join("".join(w[0] for w in n.split("_")) for n in _RING)
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("recording", type=Path)
    ap.add_argument(
        "--window",
        nargs=2,
        type=float,
        metavar=("T0", "T1"),
        help="dump every tick between T0 and T1 seconds",
    )
    ap.add_argument("--legs", action="store_true", help="per-leg step breakdown")
    ap.add_argument("--top", type=int, default=10, help="discontinuities to list")
    args = ap.parse_args()

    if not args.recording.exists():
        sys.exit(f"no such recording: {args.recording}")

    header, ticks, marks = load(args.recording)

    if args.window:
        t0, t1 = args.window
        _dump([t for t in ticks if t0 <= t["t"] <= t1])
        return

    report_session(header, ticks, marks)
    if ticks:
        report_steps(ticks, args.legs)
        report_errors(ticks)
        report_discontinuities(ticks, args.top)
        report_marks(ticks, marks)


if __name__ == "__main__":
    main()
