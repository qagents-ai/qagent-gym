"""Run after installing qagent-gym, or with PYTHONPATH=src."""

import asyncio
import json

from qagent_gym.demo import run_demo

if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_demo(".gym/example.sqlite3")), indent=2))
