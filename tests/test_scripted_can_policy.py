from vlarobot.backends.fake import FakeBackend
from vlarobot.execution.scripted_can import ScriptedCanPolicy


def test_scripted_can_policy_runs_without_rejections_on_fake_backend() -> None:
    backend = FakeBackend()
    policy = ScriptedCanPolicy()

    result = policy.execute(backend, seed=0)

    assert result.seed == 0
    assert result.rejections == 0
    assert result.final_state in {
        "pregrasp",
        "grasp",
        "lift",
        "transit",
        "place_descend",
        "release",
        "retreat",
        "home",
    }
