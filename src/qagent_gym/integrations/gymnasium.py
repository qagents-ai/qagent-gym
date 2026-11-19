"""Gymnasium 1.4 single-agent environments behind the text rollout protocol.

The SDK is loaded only when constructing an adapter. Calls run serially on one
owned worker thread. Native calls cannot be force-cancelled; close waits for
in-flight work before closing the environment.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, SupportsFloat

from ..rollouts import EnvironmentResult, _loads


class GymnasiumEnvironment:
    """Own a Gymnasium Env and validate JSON actions against its action space.

    Observations use observation_space.to_jsonable. Actions are single JSON values,
    decoded through action_space.from_jsonable before contains validation. Vector
    environments and old four-value 'done' step APIs are not supported.
    """

    def __init__(self, environment: Any) -> None:
        try:
            import gymnasium
            import numpy
        except ImportError as error:
            raise ImportError("Install qagent-gym[gymnasium] for GymnasiumEnvironment") from error
        if not isinstance(environment, gymnasium.Env):
            raise ValueError("environment must be a Gymnasium Env, not a vector environment")
        self.environment = environment
        self._gymnasium = gymnasium
        self._numpy = numpy
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qagent-gymnasium")
        self._closed = False
        self._ended = True

    def _json(self, value: Any) -> str:
        def convert(item):
            if isinstance(item, (self._numpy.ndarray, self._numpy.generic)):
                return item.tolist()
            raise TypeError(f"unsupported environment JSON value: {type(item).__name__}")

        return json.dumps(
            value, default=convert, ensure_ascii=False, sort_keys=True, allow_nan=False
        )

    async def _call(self, function, *args):
        if self._closed:
            raise ValueError("environment is closed")
        return await asyncio.get_running_loop().run_in_executor(self._executor, function, *args)

    def _reset(self, seed: int) -> str:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        observation, info = self.environment.reset(seed=seed)
        self.environment.action_space.seed(seed)
        if not isinstance(info, dict) or not self.environment.observation_space.contains(
            observation
        ):
            raise ValueError("reset returned invalid observation or info")
        self._ended = False
        return self._json(self.environment.observation_space.to_jsonable([observation])[0])

    async def reset(self, *, seed: int) -> str:
        return await self._call(self._reset, seed)

    def _step(self, action: str) -> EnvironmentResult:
        if self._ended:
            raise ValueError("reset is required before stepping an ended environment")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a JSON value encoded as text")
        try:
            decoded = _loads(action)
            if isinstance(self.environment.action_space, self._gymnasium.spaces.Discrete):
                if type(decoded) is not int:
                    raise ValueError("Discrete actions must be JSON integers")
            value = self.environment.action_space.from_jsonable([decoded])[0]
        except (ValueError, TypeError, IndexError, KeyError) as error:
            raise ValueError("action is not valid JSON for the action space") from error
        if not self.environment.action_space.contains(value):
            raise ValueError("action is outside the environment action space")
        output = self.environment.step(value)
        if not isinstance(output, tuple) or len(output) != 5:
            raise ValueError("Gymnasium step must return five values")
        observation, reward, terminated, truncated, info = output
        if (
            not isinstance(terminated, (bool, self._numpy.bool_))
            or not isinstance(truncated, (bool, self._numpy.bool_))
            or not isinstance(info, dict)
            or not self.environment.observation_space.contains(observation)
            or isinstance(reward, (bool, self._numpy.bool_))
            or not isinstance(reward, SupportsFloat)
        ):
            raise ValueError("step returned invalid observation, reward, flags or info")
        result = EnvironmentResult(
            self._json(self.environment.observation_space.to_jsonable([observation])[0]),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._json(info),
        )
        self._ended = result.terminated or result.truncated
        return result

    async def step(self, action: str) -> EnvironmentResult:
        return await self._call(self._step, action)

    async def close(self) -> None:
        """Close once, after queued native work; no future operations are accepted."""
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.get_running_loop().run_in_executor(self._executor, self.environment.close)
        finally:
            self._executor.shutdown(wait=False)
