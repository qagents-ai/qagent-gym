"""Small, dependency-free boundary for Agent Lightning v1 reward events."""

from collections.abc import Callable
from math import isfinite
from typing import Any
from urllib.parse import urlsplit

from qagent_gym.records import Reward


def reward_payload(reward: Reward) -> dict[str, Any]:
    """Build a v1 reward event, recording the evaluator as the reward source."""
    value = reward.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("reward value must be a finite number")
    try:
        numeric_value = float(value)
    except OverflowError as exc:
        raise ValueError("reward value must be a finite number") from exc
    if not isfinite(numeric_value):
        raise ValueError("reward value must be a finite number")
    return {
        "event_type": "reward",
        "data": {
            "value": numeric_value,
            "source": reward.evaluator,
            "reason": reward.feedback,
        },
    }


def send_reward(
    reward: Reward,
    *,
    event_url: str,
    api_key: str,
    transport: Callable[..., None],
) -> None:
    """Send through an injected synchronous transport that raises on failure.

    The transport is called once as ``transport(url, json=payload, headers=headers)``.
    It must check the remote response and raise on failure; no retry is performed.
    This module provides no default network implementation.
    """
    if not isinstance(event_url, str) or not event_url or any(c.isspace() for c in event_url):
        raise ValueError("event_url must be an absolute HTTP or HTTPS URL")
    try:
        parsed = urlsplit(event_url)
        valid_url = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
    except ValueError as exc:
        raise ValueError("event_url must be an absolute HTTP or HTTPS URL") from exc
    if not valid_url or parsed.username is not None or parsed.password is not None:
        raise ValueError("event_url must be an absolute HTTP or HTTPS URL without credentials")
    if not isinstance(api_key, str) or not api_key.strip() or "\r" in api_key or "\n" in api_key:
        raise ValueError("api_key must be a nonempty single-line string")
    payload = reward_payload(reward)
    transport(event_url, json=payload, headers={"Authorization": f"Bearer {api_key}"})
