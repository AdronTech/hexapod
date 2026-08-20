import json

from hexapod.body_ik import BodyPose, neutral_foot_body
from hexapod.control.recorder import SCHEMA_VERSION, Recorder
from hexapod.control.state import DT, FREE_STEP_EMERGENCY, STAND_HEIGHT
from hexapod.gait import _NEUTRAL_REACH, FreeGait, TripodGait
from hexapod.robot.config import Joint, Leg, servo_id


def neutral_feet() -> dict:
    return {
        leg: (neutral_foot_body(leg)[0], neutral_foot_body(leg)[1], 0.0) for leg in Leg
    }


def make_free(**kw) -> FreeGait:
    return FreeGait(
        BodyPose(z=STAND_HEIGHT),
        neutral_feet(),
        neutral_reach=_NEUTRAL_REACH,
        step_emergency_threshold=FREE_STEP_EMERGENCY,
        **kw,
    )


def read(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# --- gait diagnostics -------------------------------------------------------


def test_diagnostics_cover_every_leg():
    gait = make_free()
    d = gait.diagnostics()
    assert set(d) == {leg.name for leg in Leg}
    for entry in d.values():
        assert set(entry) >= {"foot", "neutral", "err", "swing", "t"}
        assert len(entry["foot"]) == 3


def test_diagnostics_err_matches_internal_foot_error():
    gait = make_free(step_time=0.4, step_threshold=3.0)
    for _ in range(40):
        gait.step(12.0, 0.0, 0.0, DT)
    d = gait.diagnostics()
    for leg in Leg:
        assert abs(d[leg.name]["err"] - gait._foot_error(leg)) < 1e-9


def test_diagnostics_track_swing_state():
    gait = make_free(step_time=0.4, step_threshold=3.0)
    seen_swing = False
    for _ in range(60):
        gait.step(12.0, 0.0, 0.0, DT)
        d = gait.diagnostics()
        for leg in Leg:
            assert d[leg.name]["swing"] == gait._swinging[leg]
            if d[leg.name]["swing"]:
                seen_swing = True
                assert 0.0 <= d[leg.name]["t"] <= 1.0
    assert seen_swing


def test_free_diagnostics_flag_due_and_emergency():
    # A tiny threshold with a fast body makes both flags fire.
    gait = make_free(step_time=0.4, step_threshold=0.5)
    due = emergency = False
    for _ in range(80):
        gait.step(30.0, 0.0, 0.0, DT)
        for entry in gait.diagnostics().values():
            due |= entry["due"]
            emergency |= entry["emergency"]
    assert due and emergency


def test_phased_diagnostics_report_swing():
    gait = TripodGait(BodyPose(z=STAND_HEIGHT), neutral_feet(), step_time=0.4)
    for _ in range(20):
        gait.step(10.0, 0.0, 0.0, DT)
        d = gait.diagnostics()
        assert sum(1 for e in d.values() if e["swing"]) == 3


# --- recorder ---------------------------------------------------------------


def test_header_and_footer_bracket_the_ticks(tmp_path):
    path = tmp_path / "rec.jsonl"
    with Recorder(path, meta={"port": "/dev/null"}) as rec:
        rec.tick(
            mode="idle",
            axes=[0.0] * 8,
            buttons=[0.0] * 17,
            speeds=(15.0, 60.0),
            step_params=(4.0, 0.4, 3.0),
            reach=17.4,
        )
    records = read(path)
    assert records[0]["type"] == "header"
    assert records[0]["v"] == SCHEMA_VERSION
    assert records[0]["port"] == "/dev/null"
    assert records[-1]["type"] == "footer"
    assert records[-1]["ticks"] == 1
    assert [r["type"] for r in records[1:-1]] == ["tick"]


def test_tick_captures_pose_gait_and_servo_ticks(tmp_path):
    path = tmp_path / "rec.jsonl"
    gait = make_free()
    pose, _ = gait.step(10.0, 0.0, 0.0, DT)
    ticks = {leg: {j: 2048 + servo_id(leg, j) for j in Joint} for leg in Leg}
    with Recorder(path) as rec:
        rec.tick(
            mode="free",
            axes=[0.1, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            buttons=[0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 12,
            speeds=(15.0, 60.0),
            step_params=(4.0, 0.4, 3.0),
            reach=17.4,
            cmd=(10.0, 0.0, 0.0),
            pose=pose,
            gait=gait,
            gait_type="free",
            ticks=ticks,
        )
    rec_tick = read(path)[1]
    assert rec_tick["mode"] == "free"
    assert rec_tick["btn"] == [4]  # only the pressed button index is stored
    assert rec_tick["ax"] == [0.1, -1.0, 0.0, 0.0]
    assert rec_tick["cmd"] == [10.0, 0.0, 0.0]
    assert len(rec_tick["pose"]) == 6
    assert set(rec_tick["legs"]) == {leg.name for leg in Leg}
    assert rec_tick["ticks"]["11"] == 2048 + 11
    assert len(rec_tick["ticks"]) == 18
    assert "err" not in rec_tick


def test_failed_tick_records_the_error_and_no_servo_ticks(tmp_path):
    path = tmp_path / "rec.jsonl"
    with Recorder(path) as rec:
        rec.tick(
            mode="free",
            axes=[0.0] * 8,
            buttons=[0.0] * 17,
            speeds=(15.0, 60.0),
            step_params=(4.0, 0.4, 3.0),
            reach=17.4,
            error="REAR_RIGHT: out of reach",
        )
    rec_tick = read(path)[1]
    assert rec_tick["err"] == "REAR_RIGHT: out of reach"
    assert "ticks" not in rec_tick


def test_marks_are_numbered_and_timestamped(tmp_path):
    path = tmp_path / "rec.jsonl"
    with Recorder(path) as rec:
        rec.mark("first")
        rec.tick(
            mode="idle",
            axes=[0.0] * 8,
            buttons=[0.0] * 17,
            speeds=(15.0, 60.0),
            step_params=(4.0, 0.4, 3.0),
            reach=17.4,
        )
        rec.mark("second")
    marks = [r for r in read(path) if r["type"] == "mark"]
    assert [m["n"] for m in marks] == [1, 2]
    assert [m["label"] for m in marks] == ["first", "second"]
    # The second mark points at the tick recorded before it.
    assert marks[1]["tick"] == 1
    assert marks[0]["t"] <= marks[1]["t"]


def test_close_is_idempotent(tmp_path):
    path = tmp_path / "rec.jsonl"
    rec = Recorder(path)
    rec.close()
    rec.close()
    assert [r["type"] for r in read(path)].count("footer") == 1


def test_every_line_is_valid_json_after_a_gait_run(tmp_path):
    path = tmp_path / "rec.jsonl"
    gait = make_free(step_time=0.4, step_threshold=3.0)
    with Recorder(path) as rec:
        for _ in range(100):
            pose, _ = gait.step(20.0, 0.0, 30.0, DT)
            rec.tick(
                mode="free",
                axes=[0.0, -1.0, 0.0, 0.0],
                buttons=[0.0] * 17,
                speeds=(20.0, 30.0),
                step_params=(4.0, 0.4, 3.0),
                reach=17.4,
                cmd=(20.0, 0.0, 30.0),
                pose=pose,
                gait=gait,
                gait_type="free",
            )
    records = read(path)
    assert len(records) == 102
    assert sum(1 for r in records if r["type"] == "tick") == 100
