import asyncio
import threading

import pytest

gymnasium = pytest.importorskip("gymnasium")

from qagent_gym.integrations.gymnasium import GymnasiumEnvironment  # noqa: E402
from qagent_gym.records import Context, Generation, Task  # noqa: E402
from qagent_gym.rollouts import run_episode  # noqa: E402

ROUTE = [1, 1, 2, 1, 2, 2]


class RoutePolicy:
    async def act(self, observation, history):
        return Generation(str(ROUTE[len(history)]))


def frozen_lake(limit=100):
    return GymnasiumEnvironment(
        gymnasium.make("FrozenLake-v1", is_slippery=False, max_episode_steps=limit)
    )


def run(limit=100):
    return asyncio.run(
        run_episode(
            frozen_lake(limit),
            RoutePolicy(),
            Task("lake", "Reach the goal."),
            Context("lake"),
            seed=42,
        )
    )


def test_native_frozen_lake_seeded_replay_and_terminal_reward():
    first, second = run(), run()
    assert first.total_reward == 1
    assert first.terminated and not first.truncated
    assert len(first.transitions) == 6
    assert first.transitions == second.transitions


def test_native_time_limit_and_simultaneous_terminal_flags():
    truncated = run(limit=2)
    assert truncated.truncated and not truncated.terminated
    assert truncated.stop_reason == "truncated"
    both = run(limit=6)
    assert both.terminated and both.truncated
    assert both.transitions[-1].result.terminated
    assert both.transitions[-1].result.truncated


@pytest.mark.parametrize("action", ["99", "right", "true", "2.0", '"2"', "NaN"])
def test_invalid_discrete_actions_do_not_change_environment(action):
    async def check():
        env = frozen_lake()
        await env.reset(seed=1)
        with pytest.raises(ValueError):
            await env.step(action)
        assert (await env.step("1")).observation == "4"
        await env.close()
        await env.close()
        with pytest.raises(ValueError, match="closed"):
            await env.reset(seed=1)

    asyncio.run(check())


def test_reset_and_close_are_owned_and_serial_on_one_worker_thread():
    thread_ids = []

    class Recorded(gymnasium.Env):
        action_space = gymnasium.spaces.Discrete(1)
        observation_space = gymnasium.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            thread_ids.append(threading.get_ident())
            return 0, {}

        def step(self, action):
            thread_ids.append(threading.get_ident())
            return 0, 1, True, False, {}

        def close(self):
            thread_ids.append(threading.get_ident())

    async def check():
        env = GymnasiumEnvironment(Recorded())
        with pytest.raises(ValueError, match="reset is required"):
            await env.step("0")
        await env.reset(seed=1)
        assert (await env.step("0")).terminated
        with pytest.raises(ValueError, match="reset is required"):
            await env.step("0")
        await env.close()

    asyncio.run(check())
    assert len(thread_ids) == 3 and len(set(thread_ids)) == 1
    assert thread_ids[0] != threading.get_ident()
