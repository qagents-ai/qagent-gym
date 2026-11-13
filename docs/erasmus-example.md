# A complete fictional Erasmus+ KA220 proposal

This recipe generates a full adult-education cooperation proposal, checks the document, learns from reviewer feedback, and writes a corrected version. It uses four real `qagents.Agent` instances with local `FunctionModel` policies. No model service, credentials, network calls, application portal or paid inference is used when running it.

The result is an original **fictional worked draft, not a submission-ready application**. It is useful for testing proposal workflows and adapting a coherent planning structure. It does not predict an award, certify eligibility, or invent evidence of need or partner commitments.

## Run it

From a repository checkout with the package and qagents installed:

```console
python -m examples.erasmus_proposal --output-dir .gym/erasmus-proposal
```

The output directory must be new or empty. After validating the brief, the run atomically claims the directory with an exclusively created `.run-reservation` file before starting any agent. Competing runs fail before planning. SQLite and each output artifact are also created exclusively, so a file created during execution is not overwritten. This is a repository example; the `examples` directory is not included in the package wheel.

The reservation remains after success, failure or cancellation; it is an ownership marker, not a completion signal. An interrupted or failed run can leave partial artifacts. Use a fresh destination to retry. The recipe does not remove reservations or partially written output automatically, and never deletes another writer's files. A successful CLI exit and the final workflow/review files identify a completed run.

```console
python -m examples.erasmus_proposal --brief examples/data/erasmus/bridge_brief.json --output-dir .gym/erasmus-adapted
python -m examples.erasmus_proposal --clean-first-draft --output-dir .gym/erasmus-clean
```

`--clean-first-draft` disables the deliberate initial faults. The clean path produces no corrective playbook entry, while retaining evidence warnings.

## The scenario and its output

BRIDGE is a fictional 24-month KA220-ADU partnership between adult-learning organisations in Greece, Italy and Spain. It plans accessible digital learning for 180 adults and prepares 36 facilitators. Its requested grant is EUR 250,000; estimated overall resources are EUR 275,000. The EUR 25,000 difference is a fictional own-resource planning assumption, not an assertion of a mandatory fixed contribution rate or confirmed financing.

The proposal includes twelve substantive sections: summary; evidence and assumptions; rationale and European added value; objectives and target groups; partnership and governance; work packages and activities; timeline and dependencies; budget and value for money; impact and indicators; risks and safeguards; dissemination and sustainability; and sources and review limits. Each activity has a deliverable and acceptance condition. The document also includes work-package budgets, a partner allocation matrix, cost assumptions, measurable targets, evaluation methods, risk owners and maintenance responsibilities.

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

The JSON brief separates supplied fictional inputs from assumptions and missing evidence. Partner legal identities, OIDs, capacity, needs evidence, cost feasibility, financing commitments and the actual call remain unresolved. Unknown baseline competence is not replaced with an invented baseline of zero. Target improvements are ambitions, not achieved impact; the proposed before/after assessment does not justify causal attribution.

## What the agents execute

1. **Planner:** validates the complete brief and derives participant counts, completion targets and partner funding totals. Incomplete or contradictory briefs fail before agents or output files are created.
2. **Writer:** builds the full narrative from the brief and derived plan. By default the initial draft deliberately claims EUR 260,000 in its summary despite a EUR 250,000 plan, and omits the risk section.
3. **Reviewer:** inspects the actual Markdown. It checks required sections, numerical claims, each work-package budget, partner allocation rows, timelines, deliverables, objectives, indicator targets, risks, evidence disclosures and official citations. Its first review detects seven errors, including the missing risk entries.
4. **Reflector:** turns the failed review into a validated playbook lesson with provenance. Gym persists the lesson and the training episode.
5. **Writer and reviewer again:** the writer sees the stored lesson and fixes the output. The final review uses frozen context and does not add lessons or memories.

The default run executes six model-policy calls: planner once, writer twice, reviewer twice and reflector once. It moves from seven internal errors to zero, with playbook revision one. Six evidence warnings remain, and `submission_ready` stays false. A reward of one means the implemented document checks passed; it is not an Erasmus+ evaluation score.

The final task uses Gym's `eval` split to freeze learning while reviewing the revision. **It is the same project brief, not an independent heldout benchmark.** The deterministic correction demonstrates orchestration, validation and persistent context use; it does not measure the reasoning quality of an LLM or perform weight training.

## Verified rule boundary

Sources were checked on 13 November 2026. The [Programme Guide page](https://erasmus-plus.ec.europa.eu/erasmus-programme-guide) identifies the 2026 call and version 1 dated 12 November 2025. This recipe fixes that snapshot for a historical exercise and makes no claim that a funding round is currently open.

The [Cooperation Partnerships action](https://erasmus-plus.ec.europa.eu/programme-guide/part-b/key-action-2/cooperation-partnerships) supports the checks for at least three organisations from three qualifying countries, coordinator establishment of at least two years, 12–36-month duration, the EUR 120,000/250,000/400,000 lump-sum choices, estimated costs above the grant, and a management allocation no greater than 20%. The programme recommends at most four implementation work packages; this recipe chooses exactly four plus management. That recipe choice is not a claim of a universal eligibility maximum. The code conservatively checks management against both grant allocation and overall estimated cost.

The fixture selects the [inclusion and diversity priority](https://erasmus-plus.ec.europa.eu/programme-guide/part-b/key-action-2/partnerships-cooperation). Full partners are restricted to the listed [EU and associated countries](https://erasmus-plus.ec.europa.eu/programme-guide/part-a/eligible-countries) for this example. Other participation scenarios need separate assessment; an unsupported-country error here is not a legal finding of ineligibility.

The code does not check every action condition, legal registration, application-count limits, national priorities, exclusion criteria, financial/operational capacity, special organisation requirements, the real deadline or the official application form. Consult the complete guide, relevant call and National Agency before developing an actual application.

## Adapt the recipe

Copy the brief and change the fictional title, rationale, participant targets, objectives, activities and resource assumptions. Counts are computed from per-partner inputs. For example, 40 learners per partner with a 75% completion target produces 120 learners and 90 completions throughout the summary, activities and indicator table. Budget edits must reconcile at work-package, cost-item and partner levels. A changed funding tier must also be accompanied by a coherent resource plan.

This is a structured BRIDGE planning template, not a universal proposal generator. WP1–WP5, the facilitator-preparation activity A3.2 and pilot activity A4.1 define its workflow. Narrative role assignments and some milestone references are specific to this example. If you change those roles, dates or delivery design, adapt and review the narrative policy and tests as well as the JSON; consistency checks do not understand every claim in free prose. Keep the fictional marker when using this recipe.

For a real-model experiment, replace one local policy with a configured model through the existing `QAgentsGenerator` seam, keep deterministic validation and reviewer checks, and add an independent human quality review. Paid model execution and sensitive real application data are outside this example. New prose must not turn assumptions into claims of verified evidence. The source prose and fixture are original project material; linked programme material is not copied into the repository.

## Checks

```console
python -m pytest tests/test_erasmus_proposal.py -q -p no:asyncio
python -m ruff check examples/erasmus_proposal.py tests/test_erasmus_proposal.py
python -m ruff format --check examples/erasmus_proposal.py tests/test_erasmus_proposal.py
```

Tests run the real agents and CLI; inspect generated proposal content, persisted provenance and frozen learning state; corrupt budget, allocation, timeline, KPI, risk and citation content to test rejection; exercise a changed brief and the clean-draft path; reject incomplete/contradictory inputs before agent creation; and confirm existing output is preserved. Contention tests pause the first planner and start a competing run against both a new and an initially empty directory. Further tests insert Markdown, JSON, JSONL and SQLite files during planning and check their exact bytes remain unchanged; a failed planner retains its reservation.

Validation on 13 November 2026: 38 tests passed, Ruff lint and format checks passed, and the installed qagents API checker reported no problems. The concurrency regressions first reproduced six failures before the reservation fix. The executed default CLI produced a 3,201-word proposal and passed all 75 implemented consistency checks; the six unresolved evidence warnings remained visible.
