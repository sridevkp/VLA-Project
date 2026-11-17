import msgpack
import numpy as np
import pytest

from vlarobot.core.types import (
    CameraInfo,
    EEDelta,
    Observation,
    Pose2D,
    Pose3D,
    RobotState,
    Vec3,
)
from vlarobot.core.wire import (
    PROTOCOL_VERSION,
    ProtocolError,
    ProtocolVersionError,
    decode_message,
    encode_message,
    make_message,
)


def _sample_observation() -> Observation:
    return Observation(
        t=1.25,
        rgb=np.arange(18, dtype=np.uint8).reshape((2, 3, 3)),
        depth=np.arange(6, dtype=np.float32).reshape((2, 3)),
        cam_info=CameraInfo(
            K=((1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
            T_base_cam=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 1.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            width=3,
            height=2,
        ),
        robot=RobotState(
            joint_pos={"arm_1_joint": 0.25},
            joint_vel={"arm_1_joint": 0.0},
            base_pose=Pose2D(0.0, 0.0, 0.0),
            ee_pose=Pose3D(Vec3(0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
            gripper_opening=0.04,
            base_pose_world=Pose3D(Vec3(1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)),
            diagnostics={"torso_lift_joint_sensor_rad": 0.35, "contact_nodes": []},
        ),
    )


def test_observation_round_trip_preserves_dataclasses_and_array_data() -> None:
    observation = _sample_observation()
    message = make_message("step", 8, {"observation": observation}, sim_time=1.25)

    decoded = decode_message(encode_message(message))

    result = decoded["payload"]["observation"]
    assert isinstance(result, Observation)
    assert result.t == observation.t
    assert result.cam_info == observation.cam_info
    assert result.robot == observation.robot
    np.testing.assert_array_equal(result.rgb, observation.rgb)
    np.testing.assert_array_equal(result.depth, observation.depth)
    assert decoded["seq"] == 8
    assert decoded["sim_time"] == 1.25


def test_action_round_trip_preserves_nested_tuple_types() -> None:
    action = EEDelta(Vec3(0.1, 0.2, 0.3), Vec3(0.0, 0.0, 0.1), gripper=0.04)

    decoded = decode_message(encode_message(make_message("apply", 1, {"action": action})))

    assert decoded["payload"]["action"] == action


def test_protocol_version_mismatch_is_rejected_on_encode_and_decode() -> None:
    message = make_message("ping", 0)
    message["protocol_version"] = PROTOCOL_VERSION - 1

    with pytest.raises(ProtocolVersionError, match="expected"):
        encode_message(message)

    incompatible_wire = msgpack.packb(message, use_bin_type=True)
    with pytest.raises(ProtocolVersionError, match="expected"):
        decode_message(incompatible_wire)


def test_boolean_protocol_version_and_integer_mapping_keys_are_rejected() -> None:
    message = make_message("ping", 0)
    message["protocol_version"] = True
    with pytest.raises(ProtocolVersionError, match="expected"):
        encode_message(message)
    with pytest.raises(ProtocolVersionError, match="expected"):
        decode_message(msgpack.packb(message, use_bin_type=True))

    message = make_message("ping", 0, {"invalid": {1: "integer key"}})
    with pytest.raises(ProtocolError, match="string keys"):
        encode_message(message)


def test_wire_rejects_malformed_envelopes() -> None:
    message = make_message("ping", 0)
    del message["seq"]

    with pytest.raises(ProtocolError, match="fields"):
        decode_message(msgpack.packb(message, use_bin_type=True))


def test_wire_rejects_object_arrays() -> None:
    message = make_message("step", 1, {"array": np.asarray([object()], dtype=object)})

    with pytest.raises(ProtocolError, match="dtype"):
        encode_message(message)


pytestmark = pytest.mark.safe_ci
