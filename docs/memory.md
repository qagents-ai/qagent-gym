# Memory integrations

QAgent Gym stores explicit text with a namespace, a stable memory ID, and its
source episode ID. The default SQLite backend is local lexical retrieval and
needs no embedding model or credentials. Playbook changes and retrieved memories
adapt the agent's input context; they do not update model weights.

## Optional Mem0 adapter

`Mem0Memory` accepts an already configured synchronous client implementing the
current v2 `top_k` contract below (OSS `Memory` or hosted `MemoryClient`). This
module does not import the Mem0 SDK,
create a provider, read credentials, or send data during import.
Calling this adapter's methods invokes the supplied client, which may contact
your configured remote service or models. Install and configure that client
separately if you want to use it.

```python
from qagent_gym.integrations import Mem0Memory
from qagent_gym.records import MemoryRecord

memory = Mem0Memory(preconfigured_client)
memory.add(MemoryRecord(
    memory_id="lesson-1",
    namespace="training:project-a",
    text="Verify the final answer before submitting.",
    source_episode_id="episode-1",
))
matches = memory.search("verify answer", namespace="training:project-a", limit=5)
```

The boundary targets Mem0's v2 `results` envelope:

- `add(text, user_id=namespace, metadata=..., infer=False)` stores the supplied
  text without asking Mem0 to extract inferred memories. Metadata carries
  `memory_id` and `source_episode_id`. The response must acknowledge exactly one
  `ADD` with a provider ID and the supplied text.
- `search(query, filters={"user_id": namespace}, top_k=limit)` returns a tuple
  of Gym `MemoryRecord` values. Results must include the original Gym metadata.
  Every result must also include a top-level `user_id` exactly matching the
  requested namespace; omitted or mismatched IDs reject the entire response.
- Pending acknowledgements, bare legacy lists, malformed envelopes, and missing
  provenance raise `ValueError`. Empty search results become `()`.

These shapes follow the [Mem0 Python client guide](https://github.com/mem0ai/mem0/blob/main/skills/mem0/client/python.md)
and [OSS memory implementation](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py).
The adapter requires the current v2 result shape with its promoted top-level
`user_id`, rather than trusting only the outgoing filter or relabeling unscoped
results. It does not guess a client version or invent provenance. An older client
exposing `limit` instead
of `top_k` needs an explicit caller-supplied wrapper; there is no automatic
fallback that could repeat a request. The provider ID is validated but the
Gym memory ID remains the value in metadata. Existing Mem0 memories without Gym
metadata need an explicit migration before this adapter can retrieve them.

`infer=False` does not promise zero model work: an underlying client may still
compute embeddings. Metadata is not a deduplication constraint, and adding a
memory is not idempotent. The adapter never retries. A timeout or rejected
acknowledgement can occur after the remote write succeeded; reconcile the remote
state before retrying. Transport/provider exceptions propagate unchanged.

Fake-client tests cover mapping, normalization, scope checks, malformed results,
and pending writes. No live Mem0 service or embedding backend was exercised.
Mem0 is [Apache-2.0 licensed](https://github.com/mem0ai/mem0/blob/main/LICENSE);
model/provider terms are separate.
