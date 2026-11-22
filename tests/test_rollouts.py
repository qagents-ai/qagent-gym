import asyncio
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from qagent_gym.records import Context, Generation, Task
from qagent_gym.rollouts import EnvironmentResult, GeneratorPolicy, Rollout, run_episode

TASK = Task("test", "Resolve the task.")
CONTEXT = Context("test")


class CounterEnvironment:
    def __init__(self, *, end_at=2, truncated=False):
        self.end_at = end_at
        self.truncated = truncated
        self.closed = 0
        self.actions = []

    async def reset(self, *, seed):
        self.seed = seed
        self.actions = []
        return "0"

    async def step(self, action):
        self.actions.append(action)
        ended = len(self.actions) == self.end_at
        return EnvironmentResult(
            str(len(self.actions)),
            float(ended),
            terminated=ended and not self.truncated,
            truncated=ended and self.truncated,
            info_json='{"private_verifier": "never show this"}',
        )

    async def close(self):
        self.closed += 1


class ConstantPolicy:
    async def act(self, observation, history):
        return Generation("advance")


def run(env, policy=None, **kwargs):
    return asyncio.run(run_episode(env, policy or ConstantPolicy(), TASK, CONTEXT, **kwargs))


def test_terminal_rollout_chain_and_json_roundtrip(tmp_path):
    env = CounterEnvironment()
    result = run(env, seed=42)
    assert env.seed == 42 and env.closed == 1
    assert result.terminated and not result.truncated
    assert result.total_reward == 1
    assert result.transitions[1].observation == "1"
    assert Rollout.from_json(result.to_json()) == result
    with pytest.raises(FrozenInstanceError):
        result.seed = 0
    path = tmp_path / "trajectory.json"
    result.save(path)
    with pytest.raises(FileExistsError):
        result.save(path)
    assert Rollout.from_json(path.read_text()) == result


def test_native_and_runner_truncation_do_not_become_termination():
    native = run(CounterEnvironment(truncated=True))
    assert native.truncated and not native.terminated
    assert native.stop_reason == "truncated"
    budget = run(CounterEnvironment(end_at=100), max_steps=3)
    assert budget.truncated and not budget.terminated
    assert budget.stop_reason == "max_steps"
    assert len(budget.transitions) == budget.max_steps == 3
    assert not budget.transitions[-1].result.truncated  # original environment evidence


def test_incomplete_policy_stops_without_stepping():
    class Stopped:
        async def act(self, observation, history):
            return Generation("", status="paused", stop_reason="approval required")

    env = CounterEnvironment()
    result = run(env, Stopped())
    assert not env.actions and env.closed == 1
    assert result.stop_reason == "policy_stopped"
    assert result.policy_stop.status == "paused"
    assert not result.terminated and not result.truncated


def test_timeout_failure_and_cancellation_all_close():
    class Slow:
        async def act(self, observation, history):
            await asyncio.sleep(60)

    env = CounterEnvironment()
    with pytest.raises(TimeoutError):
        run(env, Slow(), timeout_s=0.01)
    assert env.closed == 1

    class Broken:
        async def act(self, observation, history):
            raise RuntimeError("provider failed")

    env = CounterEnvironment()
    with pytest.raises(RuntimeError, match="provider failed"):
        run(env, Broken())
    assert env.closed == 1

    async def cancel():
        env = CounterEnvironment()
        started = asyncio.Event()

        class Waiting:
            async def act(self, observation, history):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(run_episode(env, Waiting(), TASK, CONTEXT))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert env.closed == 1

    asyncio.run(cancel())


@pytest.mark.parametrize(
    "options",
    [
        {"max_steps": True},
        {"max_steps": 0},
        {"timeout_s": float("nan")},
        {"timeout_s": 0},
        {"timeout_s": True},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_invalid_limits_close_without_starting(options):
    env = CounterEnvironment()
    with pytest.raises(ValueError):
        run(env, **options)
    assert env.closed == 1 and not env.actions


@pytest.mark.parametrize(
    "changes",
    [
        {"reward": float("inf")},
        {"reward": 10**400},
        {"reward": True},
        {"terminated": "true"},
        {"info_json": '{"secret": NaN}'},
        {"info_json": '{"a":1,"a":2}'},
    ],
)
def test_invalid_environment_results_fail(changes):
    data = {"observation": "state", "reward": 0, **changes}
    with pytest.raises(ValueError):
        EnvironmentResult(**data)


def test_malformed_and_inconsistent_wire_records_are_rejected():
    result = run(CounterEnvironment())
    payload = json.loads(result.to_json())
    payload["transitions"][0]["extra"] = True
    with pytest.raises(ValueError, match="invalid rollout"):
        Rollout.from_json(json.dumps(payload))
    with pytest.raises(ValueError, match="continuous"):
        replace(result, initial_observation="different")
    with pytest.raises(ValueError, match="stop_reason"):
        replace(result, stop_reason="max_steps", max_steps=2)
    with pytest.raises(ValueError, match="invalid rollout"):
        Rollout.from_json('{"format":"qagent-gym-rollout","format":"other"}')


def test_complete_wire_record_rejects_identical_duplicate_fields():
    encoded = run(CounterEnvironment()).to_json()
    assert '"schema_version": 1' in encoded
    ambiguous = encoded.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        Rollout.from_json(ambiguous)


def test_generator_policy_hides_private_info_and_reward():
    seen = []

    class Generator:
        async def generate(self, task, context):
            seen.append(task.prompt)
            return Generation("advance")

    policy = GeneratorPolicy(Generator(), TASK, CONTEXT)
    run(CounterEnvironment(), policy)
    assert len(seen) == 2 and "advance" in seen[1]
    assert all("private_verifier" not in text and "never show this" not in text for text in seen)


@pytest.mark.parametrize(
    "end_at,max_steps,truncated,expected",
    [
        (2, 3, False, 1),
        (100, 2, False, 0),
        (2, 3, True, 0),
    ],
)
def test_gym_generation_and_environment_reward_contract(end_at, max_steps, truncated, expected):
    from qagent_gym.rollouts import EnvironmentReturnEvaluator, RolloutGenerator

    async def check():
        generator = RolloutGenerator(
            lambda task: CounterEnvironment(end_at=end_at, truncated=truncated),
            lambda task, context: ConstantPolicy(),
            max_steps=max_steps,
        )
        generation = await generator.generate(TASK, CONTEXT)
        reward = await EnvironmentReturnEvaluator().evaluate(TASK, generation)
        assert reward.value == expected
        rollout = Rollout.from_json(generation.steps[0].content)
        if expected:
            assert generation.status == "completed"
            assert generation.output == rollout.transitions[-1].result.observation
            assert "private_verifier" not in generation.output
        else:
            assert generation.status == "limit_reached"
            assert generation.output == ""
        with pytest.raises(ValueError, match="does not match"):
            await EnvironmentReturnEvaluator().evaluate(Task("other", "x"), generation)
        with pytest.raises(ValueError, match="exactly one"):
            await EnvironmentReturnEvaluator().evaluate(TASK, Generation('{"reward":999}'))

    asyncio.run(check())


def test_factory_failure_closes_the_constructed_environment():
    from qagent_gym.rollouts import RolloutGenerator

    env = CounterEnvironment()

    def broken(task, context):
        raise RuntimeError("policy factory failed")

    with pytest.raises(RuntimeError, match="factory failed"):
        asyncio.run(RolloutGenerator(lambda task: env, broken).generate(TASK, CONTEXT))
    assert env.closed == 1


def test_learning_example_freezes_heldout_context(tmp_path):
    from examples.environment_learning import run

    result = asyncio.run(run(tmp_path / "learning.sqlite3"))
    assert result["training_reward"] == 0
    assert result["heldout_rewards"] == [1, 1]
    assert result["heldout_actions"] == [["review", "revise", "submit"]] * 2
    assert result["frozen_evaluation"] is True
