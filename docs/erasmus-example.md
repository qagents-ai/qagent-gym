# A complete fictional Erasmus+ KA220 proposal

This recipe drafts an adult-education cooperation proposal and checks it against
the supplied brief. Reviewer feedback becomes a stored lesson that the writer
uses to revise the document. The agents run offline with local `FunctionModel`
policies, so you can inspect the workflow before connecting a model service.

The result is a fictional planning draft for testing proposal workflows. A real
application still needs partner commitments, evidence of need and review against
the relevant call. The example does not establish eligibility or predict an award.

## Run it

From a repository checkout with the package and qagents installed:

```console
python -m examples.erasmus_proposal --output-dir .gym/erasmus-proposal
```

The output directory must be new or empty. After validating the brief, the run atomically claims the directory with an exclusively created `.run-reservation` file before starting any agent. Competing runs fail before planning. SQLite and each output artifact are also created exclusively, so a file created during execution is not overwritten. This is a repository example. The `examples` directory is not included in the package wheel.

The reservation remains after success, failure or cancellation. It is an ownership marker, not a completion signal. An interrupted or failed run can leave partial artifacts. Use a fresh destination to retry. The recipe does not remove reservations or partially written output automatically, and never deletes another writer's files. A successful CLI exit and the final workflow/review files identify a completed run.

```console
python -m examples.erasmus_proposal --brief examples/data/erasmus/bridge_brief.json --output-dir .gym/erasmus-adapted
python -m examples.erasmus_proposal --clean-first-draft --output-dir .gym/erasmus-clean
```

`--clean-first-draft` disables the deliberate initial faults. The clean path produces no corrective playbook entry, while retaining evidence warnings.

## The scenario and its output

BRIDGE is a fictional 24-month KA220-ADU partnership between adult-learning organisations in Greece, Italy and Spain. It plans accessible digital learning for 180 adults and prepares 36 facilitators. Its requested grant is EUR 250,000. Estimated overall resources are EUR 275,000. The EUR 25,000 difference is a fictional own-resource planning assumption, not an assertion of a mandatory fixed contribution rate or confirmed financing.

The proposal includes twelve substantive sections: summary. Evidence and assumptions. Rationale and European added value. Objectives and target groups. Partnership and governance. Work packages and activities. Timeline and dependencies. Budget and value for money. Impact and indicators. Risks and safeguards. Dissemination and sustainability and sources and review limits. Each activity has a deliverable and acceptance condition. The document also includes work-package budgets, a partner allocation matrix, cost assumptions, measurable targets, evaluation methods, risk owners and maintenance responsibilities.

| Artifact | Purpose |
| --- | --- |
| `proposal.md` | Complete corrected planning draft |
| `draft-initial.md` | Original draft with deliberately injected faults |
| `brief.json`, `plan.json` | Input snapshot and derived counts/allocations |
| `review-before.json`, `review-after.json` | Errors, evidence warnings and number of checks |
| `workflow.json` | Agent call counts, correction result and frozen-state check |
| `episodes.jsonl` | Training and final-review episodes with messages and feedback |
| `gym.sqlite3` | Persistent playbook, provenance, memory and episode records |
| `.run-reservation` | Retained ownership marker preventing concurrent or accidental reuse |

The JSON brief separates supplied fictional inputs from assumptions and missing evidence. Partner legal identities, OIDs, capacity, needs evidence, cost feasibility, financing commitments and the actual call remain unresolved. Unknown baseline competence is not replaced with an invented baseline of zero. Target improvements are ambitions, not achieved impact. The proposed before/after assessment does not justify causal attribution.

## What the agents execute

1. The planner validates the brief and calculates participant counts, completion
   targets and partner funding totals. Contradictory briefs fail before agents
   or output files are created.
2. The writer drafts the narrative from the brief and plan. The default fixture
   deliberately introduces a budget mismatch and omits the risk section.
3. The reviewer checks the Markdown against the brief, including allocations,
   timelines, deliverables, risk entries and evidence disclosures.
4. The reflector turns the failed review into a playbook lesson. Gym stores the
   lesson with its source episode.
5. The writer uses the lesson to revise the draft. The reviewer checks the
   revision with frozen context, without adding lessons or memories.

The default run corrects the deliberately introduced errors. Evidence warnings
remain visible, and `submission_ready` stays false. A reward of one means the
implemented document checks passed. It is not an Erasmus+ evaluation score.

The final review uses Gym's `eval` split to freeze learning. It reviews the same
project brief, so this run demonstrates correction within a workflow. An
independent benchmark would need separate project briefs and an evaluation of
model quality. This example keeps model weights fixed.

## Verified rule boundary

Sources were checked on 13 November 2026. The [Programme Guide page](https://erasmus-plus.ec.europa.eu/erasmus-programme-guide) identifies the 2026 call and version 1 dated 12 November 2025. This recipe fixes that snapshot for a historical exercise and makes no claim that a funding round is currently open.

The [Cooperation Partnerships action](https://erasmus-plus.ec.europa.eu/programme-guide/part-b/key-action-2/cooperation-partnerships) supports the checks for at least three organisations from three qualifying countries, coordinator establishment of at least two years, a duration of 12 to 36 months, the EUR 120,000/250,000/400,000 lump-sum choices, estimated costs above the grant, and a management allocation no greater than 20%. The programme recommends at most four implementation work packages. This recipe chooses exactly four plus management. That recipe choice is not a claim of a universal eligibility maximum. The code conservatively checks management against both grant allocation and overall estimated cost.

The fixture selects the [inclusion and diversity priority](https://erasmus-plus.ec.europa.eu/programme-guide/part-b/key-action-2/partnerships-cooperation). Full partners are restricted to the listed [EU and associated countries](https://erasmus-plus.ec.europa.eu/programme-guide/part-a/eligible-countries) for this example. Other participation scenarios need separate assessment. An unsupported-country error here is not a legal finding of ineligibility.

The code does not check every action condition, legal registration, application-count limits, national priorities, exclusion criteria, financial/operational capacity, special organisation requirements, the real deadline or the official application form. Consult the complete guide, relevant call and National Agency before developing an actual application.

## Adapt the recipe

Copy the brief and change the fictional title, rationale, participant targets, objectives, activities and resource assumptions. Counts are computed from per-partner inputs. For example, 40 learners per partner with a 75% completion target produces 120 learners and 90 completions throughout the summary, activities and indicator table. Budget edits must reconcile at work-package, cost-item and partner levels. A changed funding tier must also be accompanied by a coherent resource plan.

This is a structured BRIDGE planning template, not a universal proposal generator. WP1 to WP5, the facilitator-preparation activity A3.2 and pilot activity A4.1 define its workflow. Narrative role assignments and some milestone references are specific to this example. If you change those roles, dates or delivery design, adapt and review the narrative policy and tests as well as the JSON. Consistency checks do not understand every claim in free prose. Keep the fictional marker when using this recipe.

For a real-model experiment, replace one local policy with a configured model through the existing `QAgentsGenerator` seam, keep deterministic validation and reviewer checks, and add an independent human quality review. Paid model execution and sensitive real application data are outside this example. New prose must not turn assumptions into claims of verified evidence. The source prose and fixture are original project material. Linked programme material is not copied into the repository.

## Checks

```console
python -m pytest tests/test_erasmus_proposal.py -q -p no:asyncio
python -m ruff check examples/erasmus_proposal.py tests/test_erasmus_proposal.py
python -m ruff format --check examples/erasmus_proposal.py tests/test_erasmus_proposal.py
```

Tests run the real agents and CLI. Inspect generated proposal content, persisted provenance and frozen learning state. Corrupt budget, allocation, timeline, KPI, risk and citation content to test rejection. Exercise a changed brief and the clean-draft path. Reject incomplete/contradictory inputs before agent creation and confirm existing output is preserved. Contention tests pause the first planner and start a competing run against both a new and an initially empty directory. Further tests insert Markdown, JSON, JSONL and SQLite files during planning and check their exact bytes remain unchanged. A failed planner retains its reservation.

Validation on 13 November 2026: 38 tests passed, Ruff lint and format checks passed, and the installed qagents API checker reported no problems. The concurrency regressions first reproduced six failures before the reservation fix. The executed default CLI produced a proposal that passed all 75 implemented consistency checks. The six unresolved evidence warnings remained visible.

## Rehearse worker restart and backup restore

The BRIDGE mailbox pilot runs three agents in separate Python processes. The coordinator
assigns the draft. The writer sends the full Markdown to the reviewer.
The reviewer checks it and returns feedback. The resumed writer sends
a corrected draft, and the reviewer sends the final decision to the
coordinator. Model responses are scripted locally using the fictional
KA220-ADU brief above.

Use a QAgents checkout at commit `dd74a3d` or later and install both
packages in the same virtual environment. From QAgent Gym, choose a new
output directory.

```console
python -m examples.erasmus_mailbox_pilot .gym/bridge-recovery
```

The controller terminates the writer after SQLite commits its first
outgoing message and before QAgents records the tool result. It verifies
the SQLite inbox contains that draft and waits for the writer's lease
to expire before resuming. The two recovery paths use the same interruption
point.

1. The restart uses the original mailbox and agent journals.
2. The restore copies a backup to a fresh directory and continues with
   fresh processes and restored sessions.

The controller takes the backup while all workers are stopped. It uses
SQLite's backup API for the mailbox and each of the three agent journals,
then checks each copy with `PRAGMA integrity_check`. Keep the four
databases together. Each file has its own SQLite transaction, so stopping
writers is what gives this exercise a consistent recovery point across
the set. See the [SQLite Backup API](https://www.sqlite.org/backup.html).

Both paths check the delivered draft against the original sequence and
text, confirm the coordinator receives the final review and verify that
every inbox is drained. The controller compares the SHA-256 hashes of
`live/proposal.md` and `restored/proposal.md`. The output also contains
`pilot-report.json` with timings, review counts and result hashes. The recovery phase
time starts when the test launches the workers, includes any remaining
lease wait and ends when the verified proposal is written.
The backup represents the point where the first draft was waiting for
review. Work committed after that snapshot requires a newer backup to
recover.

In the Windows pilot, each path completed 75 internal checks, resolved
all seven deliberately injected errors and preserved six evidence
warnings. The two drafts had the same SHA-256 hash. The documents are
fictional planning drafts. For an actual application, use the relevant
call and verify partner commitments, needs evidence and budget inputs
with the people responsible for the proposal.
