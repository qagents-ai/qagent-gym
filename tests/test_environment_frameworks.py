"""Native cross-framework rollout agreement; no model endpoint is contacted."""

import asyncio

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("langgraph")


def test_native_qagents_and_langgraph_trajectories_agree(tmp_path):
    from examples.environment_benchmark import run

    result = asyncio.run(run(seeds=(0, 1, 2), output_dir=tmp_path / "rollouts"))
    assert result["rollouts"] == 12
    assert result["trajectory_agreement"] is True
    assert result["paired"]["mean_delta"] == 0
    assert result["paired"]["lower"] == result["paired"]["upper"] == 0
    assert len(list((tmp_path / "rollouts").glob("qagents-*.json"))) == 6
    with pytest.raises(FileExistsError):
        asyncio.run(run(seeds=(0,), output_dir=tmp_path / "rollouts"))
