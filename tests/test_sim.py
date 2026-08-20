import math
import re
from pathlib import Path

from hexapod.body_ik import BodyPose, body_pose_ik, neutral_foot_body
from hexapod.kinematics import angle_to_tick, leg_fk
from hexapod.robot.config import ALL_SERVO_IDS, Joint, Leg, servo_id
from hexapod.servo import registers as R
from hexapod.servo.protocol import build_packet, build_sync_write, parse_status_packet
from hexapod.sim.bus import VirtualBus
from hexapod.sim.model import (
    MAX_BODY_STEP,
    Odometry,
    WorldPose,
    leg_joint_positions,
    robot_state,
)
from hexapod.sim.servo import VirtualServo

TOL = 1e-4


def step_for(target, seconds: float, dt: float = 0.005) -> None:
    for _ in range(int(seconds / dt)):
        target.step(dt)


def ticks_for(pose: BodyPose, feet: dict) -> dict[int, int]:
    out: dict[int, int] = {}
    for leg, (tc, tf, tt) in body_pose_ik(pose, feet).items():
        out[servo_id(leg, Joint.COXA)] = angle_to_tick("coxa", tc)
        out[servo_id(leg, Joint.FEMUR)] = angle_to_tick("femur", tf)
        out[servo_id(leg, Joint.TIBIA)] = angle_to_tick("tibia", tt)
    return out


def planted_feet() -> dict:
    return {
        leg: (neutral_foot_body(leg)[0], neutral_foot_body(leg)[1], 0.0) for leg in Leg
    }


# ---------------------------------------------------------------------------
# Servo model
# ---------------------------------------------------------------------------


def test_servo_reaches_goal_and_stops():
    servo = VirtualServo(11)
    servo.write(R.ACC, bytes([0, 0xE8, 0x0B, 0, 0, 0, 0]))  # goal 3048, speed 0
    step_for(servo, 2.0)
    assert servo.pos == 3048
    assert servo.vel == 0
    assert not servo.moving


def test_servo_respects_speed_limit():
    servo = VirtualServo(11)
    servo.write(R.GOAL_SPEED_L, bytes([200, 0]))  # 200 ticks/s
    servo.write(R.GOAL_POS_L, bytes([0xFF, 0x0F]))  # 4095, far away
    step_for(servo, 1.0)
    assert 150 < servo.pos - 2048 < 250  # ≈ 200 ticks in one second


def test_servo_holds_position_without_torque():
    servo = VirtualServo(11)
    servo.write(R.TORQUE_ENABLE, bytes([0]))
    servo.write(R.GOAL_POS_L, bytes([0xFF, 0x0F]))
    step_for(servo, 1.0)
    assert servo.pos == 2048


def test_servo_position_is_clamped_to_encoder_range():
    servo = VirtualServo(11, position=100)
    servo.write(R.GOAL_POS_L, bytes([0, 0]))
    step_for(servo, 1.0)
    assert servo.pos == 0


def test_calibrate_middle_shifts_reported_position():
    servo = VirtualServo(11)
    servo.write(R.GOAL_POS_L, bytes([0xE8, 0x0B]))  # 3048
    step_for(servo, 2.0)
    servo.write(R.TORQUE_ENABLE, bytes([128]))
    assert servo.pos == 2048
    assert servo.ofs == 1000
    assert servo.torque_on


def test_feedback_registers_track_the_model():
    servo = VirtualServo(11)
    servo.write(R.GOAL_POS_L, bytes([0xFF, 0x0F]))
    step_for(servo, 0.2)
    data = servo.read(R.PRESENT_POS_L, 11)
    assert data[0] | (data[1] << 8) == round(servo.pos)
    assert data[6] > 0  # voltage
    assert data[10] == 1  # moving


# ---------------------------------------------------------------------------
# Bus protocol
# ---------------------------------------------------------------------------


def test_ping_answers_for_known_ids_only():
    bus = VirtualBus()
    replies = bus.feed(build_packet(11, R.INST_PING, []))
    assert len(replies) == 1
    assert parse_status_packet(replies[0]).servo_id == 11

    assert bus.feed(build_packet(99, R.INST_PING, [])) == []
    assert bus.stats.unknown_id == 1


def test_missing_servos_do_not_answer():
    bus = VirtualBus(missing={23})
    assert bus.feed(build_packet(23, R.INST_PING, [])) == []
    assert bus.feed(build_packet(22, R.INST_PING, [])) != []


def test_read_returns_present_position():
    bus = VirtualBus()
    replies = bus.feed(build_packet(11, R.INST_READ, [R.PRESENT_POS_L, 2]))
    data = parse_status_packet(replies[0]).data
    assert data[0] | (data[1] << 8) == 2048


def test_write_is_acknowledged_and_applied():
    bus = VirtualBus()
    replies = bus.feed(build_packet(11, R.INST_WRITE, [R.GOAL_POS_L, 0xE8, 0x0B]))
    assert parse_status_packet(replies[0]).error == 0
    assert bus.servos[11].goal == 3048


def test_sync_write_reaches_every_servo_without_replies():
    bus = VirtualBus()
    packet = build_sync_write(
        R.ACC, 7, [(sid, [0, 0xE8, 0x03, 0, 0, 0, 0]) for sid in ALL_SERVO_IDS]
    )
    assert bus.feed(packet) == []
    assert all(servo.goal == 1000 for servo in bus.servos.values())


def test_reg_write_is_deferred_until_action():
    bus = VirtualBus()
    bus.feed(build_packet(11, R.INST_REG_WRITE, [R.GOAL_POS_L, 0xE8, 0x0B]))
    assert bus.servos[11].goal == 2048
    bus.feed(build_packet(R.BROADCAST_ID, R.INST_REG_ACTION, []))
    assert bus.servos[11].goal == 3048


def test_broadcast_write_is_silent():
    bus = VirtualBus()
    assert (
        bus.feed(build_packet(R.BROADCAST_ID, R.INST_WRITE, [R.TORQUE_ENABLE, 0])) == []
    )
    assert all(not servo.torque_on for servo in bus.servos.values())


def test_corrupt_packets_are_dropped():
    bus = VirtualBus()
    packet = bytearray(build_packet(11, R.INST_PING, []))
    packet[-1] ^= 0xFF
    assert bus.feed(bytes(packet)) == []
    assert bus.stats.bad_checksum == 1


def test_packets_split_across_reads_are_reassembled():
    bus = VirtualBus()
    packet = build_packet(11, R.INST_READ, [R.PRESENT_POS_L, 2])
    assert bus.feed(packet[:3]) == []
    assert bus.feed(packet[3:]) != []


def test_garbage_before_a_packet_is_skipped():
    bus = VirtualBus()
    packet = build_packet(11, R.INST_PING, [])
    assert bus.feed(b"\x00\x17\xff" + packet) != []


def test_sync_read_answers_for_each_requested_servo():
    bus = VirtualBus()
    replies = bus.feed(
        build_packet(R.BROADCAST_ID, R.INST_SYNC_READ, [R.PRESENT_POS_L, 2, 11, 12, 99])
    )
    assert [parse_status_packet(r).servo_id for r in replies] == [11, 12]


# ---------------------------------------------------------------------------
# Kinematic state model
# ---------------------------------------------------------------------------


def test_joint_chain_ends_at_the_leg_fk_foot():
    tc, tf, tt = 12.0, -20.0, 35.0
    for leg in Leg:
        joints = leg_joint_positions(leg, tc, tf, tt)
        x, y, z = leg_fk(tc, tf, tt)
        # Foot in the body frame = corner + leg-frame foot rotated by the corner angle
        corner = joints[0]
        a = math.atan2(corner[1], corner[0])
        expected = (
            corner[0] + x * math.cos(a) - y * math.sin(a),
            corner[1] + x * math.sin(a) + y * math.cos(a),
            z,
        )
        for got, want in zip(joints[3], expected):
            assert abs(got - want) < TOL


def test_state_matches_the_pose_the_ik_was_solved_for():
    pose = BodyPose(z=15.0)
    feet = {
        leg: (neutral_foot_body(leg)[0], neutral_foot_body(leg)[1], 0.0) for leg in Leg
    }
    ticks = {}
    for leg, (tc, tf, tt) in body_pose_ik(pose, feet).items():
        ticks[servo_id(leg, Joint.COXA)] = angle_to_tick("coxa", tc)
        ticks[servo_id(leg, Joint.FEMUR)] = angle_to_tick("femur", tf)
        ticks[servo_id(leg, Joint.TIBIA)] = angle_to_tick("tibia", tt)

    state = robot_state(ticks)
    assert abs(state.body_z - 15.0) < 0.01
    assert all(leg.grounded for leg in state.legs)
    assert state.stable
    for leg in state.legs:
        want = feet[leg.leg]
        assert abs(leg.foot[0] - want[0]) < 0.01
        assert abs(leg.foot[1] - want[1]) < 0.01
        assert abs(leg.foot[2]) < 0.01


def test_roll_and_pitch_are_recovered_from_the_ticks():
    """A body tilted over planted feet has to come back out of the FK."""
    for roll, pitch in [(15.0, 0.0), (0.0, 15.0), (0.0, -20.0), (10.0, 10.0)]:
        pose = BodyPose(z=15.0, roll=roll, pitch=pitch)
        state = robot_state(ticks_for(pose, planted_feet()))
        assert abs(state.roll - roll) < 0.2
        assert abs(state.pitch - pitch) < 0.2
        assert abs(state.body_z - 15.0) < 0.1
        assert all(leg.grounded for leg in state.legs)
        assert state.stable


def test_body_tilts_onto_the_legs_that_carry_it():
    ticks = {sid: 2048 for sid in ALL_SERVO_IDS}
    for leg in (Leg.FRONT_RIGHT, Leg.MID_RIGHT, Leg.REAR_RIGHT):
        ticks[servo_id(leg, Joint.FEMUR)] = angle_to_tick("femur", 40.0)
    state = robot_state(ticks)
    assert state.roll > 10.0  # right feet lifted, so the left side carries it
    assert sum(leg.grounded for leg in state.legs) == 3
    assert state.stable


def test_folded_legs_leave_the_body_on_the_ground():
    ticks = {sid: 2048 for sid in ALL_SERVO_IDS}
    for leg in Leg:
        ticks[servo_id(leg, Joint.FEMUR)] = angle_to_tick("femur", 140.0)
        ticks[servo_id(leg, Joint.TIBIA)] = angle_to_tick("tibia", -70.0)
    state = robot_state(ticks)
    assert state.body_z == 0.0  # lying on its belly, never below the floor
    assert not any(leg.grounded for leg in state.legs)
    assert not state.stable


# ---------------------------------------------------------------------------
# Odometry
# ---------------------------------------------------------------------------


def test_odometry_starts_at_the_origin():
    odo = Odometry()
    pose = odo.update(robot_state(ticks_for(BodyPose(z=15.0), planted_feet())))
    assert (pose.x, pose.y, pose.yaw) == (0.0, 0.0, 0.0)


def test_odometry_follows_the_body_over_planted_feet():
    odo = Odometry()
    feet = planted_feet()
    for step in range(11):  # body creeps forward over feet that stay put
        pose = BodyPose(x=step * 0.2, z=15.0)
        world = odo.update(robot_state(ticks_for(pose, feet)))
    assert abs(world.x - 2.0) < 0.05
    assert abs(world.y) < 0.05
    assert abs(world.yaw) < 1e-3


def test_odometry_follows_body_rotation():
    odo = Odometry()
    feet = planted_feet()
    for step in range(11):
        pose = BodyPose(z=15.0, yaw=step * 1.0)
        world = odo.update(robot_state(ticks_for(pose, feet)))
    assert abs(math.degrees(world.yaw) - 10.0) < 0.3


def test_odometry_ignores_implausible_jumps():
    odo = Odometry()
    odo.update(robot_state(ticks_for(BodyPose(z=15.0), planted_feet())))
    folded = {sid: 2048 for sid in ALL_SERVO_IDS}
    for leg in Leg:
        folded[servo_id(leg, Joint.FEMUR)] = angle_to_tick("femur", 60.0)
    world = odo.update(robot_state(folded))
    assert math.hypot(world.x, world.y) <= MAX_BODY_STEP


def test_world_pose_places_joints_in_world_coordinates():
    feet = planted_feet()
    state = robot_state(ticks_for(BodyPose(z=15.0), feet))
    moved = state.to_dict(WorldPose(x=10.0, y=-5.0, yaw=math.pi / 2))
    assert moved["world"] == {"x": 10.0, "y": -5.0, "heading": 90.0}
    front_right = moved["legs"][0]["joints"][3]
    want = feet[Leg.FRONT_RIGHT]
    assert abs(front_right[0] - (10.0 - want[1])) < 0.01  # rotated 90° CCW
    assert abs(front_right[1] - (-5.0 + want[0])) < 0.01


def test_heading_is_reported_wrapped():
    state = robot_state(ticks_for(BodyPose(z=15.0), planted_feet()))
    assert state.to_dict(WorldPose(yaw=math.radians(-260)))["world"]["heading"] == 100.0
    assert state.to_dict(WorldPose(yaw=math.radians(540)))["world"]["heading"] == -180.0


# ---------------------------------------------------------------------------
# Viewer page
# ---------------------------------------------------------------------------


def test_viewer_html_defines_every_element_the_script_uses():
    """A missing id throws at load time and leaves the whole page blank."""
    web = Path(__file__).parent.parent / "scripts" / "sim_web"
    ids = set(re.findall(r'id="([^"]+)"', (web / "index.html").read_text()))
    used = set(re.findall(r'getElementById\("([^"]+)"\)', (web / "app.js").read_text()))
    assert not used - ids, f"missing from index.html: {sorted(used - ids)}"
