"""The base package remains importable when every optional SDK is unavailable."""

import subprocess
import sys


def test_integration_exports_do_not_import_optional_sdks():
    source = r"""
import importlib.abc
import sys

class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'dspy', 'elevenlabs', 'langgraph', 'langchain_core',
            'mem0', 'opentelemetry', 'twilio', 'gymnasium', 'numpy',
        }:
            raise AssertionError('optional SDK imported: ' + fullname)

sys.meta_path.insert(0, NoSDK())
from qagent_gym.integrations import (
    DSPyGenerator, ElevenLabsSynthesizer, FrozenMem0Memory,
    LangGraphGenerator, Mem0Memory, TracingEvaluator, TracingGenerator,
    TwilioVoiceWebhook, GymnasiumEnvironment,
)
assert all(isinstance(item, type) for item in (
    DSPyGenerator, ElevenLabsSynthesizer, FrozenMem0Memory,
    LangGraphGenerator, Mem0Memory, TracingEvaluator, TracingGenerator,
    TwilioVoiceWebhook, GymnasiumEnvironment,
))
"""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr
