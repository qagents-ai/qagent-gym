"""Generate, review and correct a fictional KA220 adult-education proposal offline.

python -m examples.erasmus_proposal --output-dir .gym/erasmus-proposal
Policies are deterministic Python executed by real qagents Agents. Their output
is a worked planning draft, not a real application or an eligibility decision.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from qagents import Agent, Limits
from qagents.models import ModelRequest
from qagents.models.testing import FunctionModel

from qagent_gym.export import export_jsonl
from qagent_gym.gym import Gym
from qagent_gym.qagents_adapter import QAgentsGenerator, QAgentsReflector
from qagent_gym.records import Context, Generation, Reward, Task
from qagent_gym.storage import SQLiteStore

from ._support import has_rule, task_payload, user_prompt

DEFAULT_BRIEF = Path(__file__).parent / "data" / "erasmus" / "bridge_brief.json"
CHECKED_ON = "2026-11-13"
SOURCES = {
    "guide": "https://erasmus-plus.ec.europa.eu/erasmus-programme-guide",
    "action": "https://erasmus-plus.ec.europa.eu/programme-guide/part-b/key-action-2/cooperation-partnerships",
    "countries": "https://erasmus-plus.ec.europa.eu/programme-guide/part-a/eligible-countries",
    "priorities": "https://erasmus-plus.ec.europa.eu/programme-guide/part-b/key-action-2/partnerships-cooperation",
}
ELIGIBLE_COUNTRIES = frozenset(
    "Belgium Bulgaria Czechia Denmark Germany Estonia Ireland Greece Spain France Croatia Italy "
    "Cyprus Latvia Lithuania Luxembourg Hungary Malta Netherlands Austria Poland Portugal Romania "
    "Slovenia Slovakia Finland Sweden Serbia Iceland Liechtenstein Norway Türkiye".split()
) | {"North Macedonia"}
SECTIONS = (
    "Executive summary",
    "Evidence, assumptions and open questions",
    "Rationale and European added value",
    "Objectives and target groups",
    "Partnership and governance",
    "Work packages and activities",
    "Timeline and dependencies",
    "Budget and value for money",
    "Impact, indicators and evaluation",
    "Risks and safeguards",
    "Dissemination and sustainability",
    "Programme references and review boundary",
)
CORRECTION_RULE = (
    "Before releasing a proposal, include every required section and reconcile all funding "
    "claims and allocations against the validated project plan. Keep evidence gaps visible."
)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: a nonempty string is required")
    return value


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: integer >= {minimum} required")
    return value


def _objects(value: Any, label: str) -> list[dict]:
    if not isinstance(value, list) or not value or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"{label}: a nonempty list of objects is required")
    return value


def validate_brief(raw: dict) -> dict:
    """Validate supported scenario inputs before any agent runs or artifact is written.

    This recipe handles a fictional five-WP KA220-ADU design with full partners
    in EU/associated countries. Other action/country/partner cases need a separate
    assessment; rejection here is not a legal finding that they are ineligible.
    """
    if not isinstance(raw, dict):
        raise ValueError("brief must be a JSON object")
    brief = json.loads(json.dumps(raw, allow_nan=False))
    if type(brief.get("schema_version")) is not int or brief["schema_version"] != 1:
        raise ValueError("schema_version must be the integer 1")
    if brief.get("fictional") is not True:
        raise ValueError("this recipe requires schema_version=1 and fictional=true")
    if brief.get("action") != "KA220-ADU" or brief.get("guide_year") != 2026:
        raise ValueError("this recipe supports KA220-ADU using the 2026 guide snapshot")
    for field in (
        "acronym",
        "title",
        "priority",
        "rationale",
        "european_added_value",
        "target_group",
        "coordinator_id",
    ):
        _string(brief.get(field), field)
    if brief["priority"] != "inclusion and diversity":
        raise ValueError("this particular scenario implements the inclusion and diversity priority")
    months = _integer(brief.get("duration_months"), "duration_months")
    if not 12 <= months <= 36:
        raise ValueError("2026 KA220 duration must be between 12 and 36 months")
    grant = _integer(brief.get("grant_eur"), "grant_eur")
    if grant not in (120000, 250000, 400000):
        raise ValueError("grant_eur must use a verified 2026 KA220 lump-sum tier")
    total = _integer(brief.get("estimated_cost_eur"), "estimated_cost_eur")
    if total <= grant:
        raise ValueError("estimated overall cost must exceed the requested grant")
    for field in (
        "learners_per_partner",
        "facilitators_per_partner",
        "interviews_per_partner",
        "codesign_per_partner",
        "pilot_hours",
        "external_adopters",
        "resource_visits",
    ):
        _integer(brief.get(field), field)
    for field in ("completion_percent", "fewer_opportunities_percent", "skill_gain_pp"):
        if not 1 <= _integer(brief.get(field), field) <= 100:
            raise ValueError(f"{field}: expected 1..100")
    if (
        max(brief["interviews_per_partner"], brief["codesign_per_partner"])
        > brief["learners_per_partner"]
    ):
        raise ValueError("interviews/co-design must be subsets of the learner pool")
    partners = _objects(brief.get("partners"), "partners")
    for partner in partners:
        for field in ("id", "name", "country", "role", "contribution"):
            _string(partner.get(field), f"partner.{field}")
        _integer(
            partner.get("years_established_at_deadline"), "years_established_at_deadline", minimum=0
        )
        if partner["country"] not in ELIGIBLE_COUNTRIES:
            raise ValueError("recipe supports only the listed EU/associated-country full partners")
    partner_ids = {partner["id"] for partner in partners}
    if len(partner_ids) != len(partners):
        raise ValueError("partner IDs must be unique")
    if len(partners) < 3 or len({p["country"] for p in partners}) < 3:
        raise ValueError(
            "KA220 requires at least three organisations from three qualifying countries"
        )
    coordinator = next((p for p in partners if p["id"] == brief["coordinator_id"]), None)
    if coordinator is None or coordinator["years_established_at_deadline"] < 2:
        raise ValueError(
            "coordinator must be identified and report at least two years at the deadline"
        )
    objectives = _objects(brief.get("objectives"), "objectives")
    for objective in objectives:
        _string(objective.get("id"), "objective.id")
        _string(objective.get("text"), "objective.text")
    objective_ids = {objective["id"] for objective in objectives}
    if len(objective_ids) != len(objectives):
        raise ValueError("objective IDs must be unique")
    packages = _objects(brief.get("work_packages"), "work_packages")
    if len(packages) != 5 or {wp.get("id") for wp in packages} != {f"WP{i}" for i in range(1, 6)}:
        raise ValueError("this recipe expects WP1 management plus WP2-WP5 implementation")
    activity_ids = set()
    activities = {}
    for wp in packages:
        _string(wp.get("title"), "work_package.title")
        if wp.get("lead") not in partner_ids:
            raise ValueError("work-package lead must be a known partner")
        start, end = _integer(wp.get("start"), "WP.start"), _integer(wp.get("end"), "WP.end")
        if not 1 <= start <= end <= months:
            raise ValueError("work-package dates must lie inside the project")
        amount = _integer(wp.get("grant_eur"), "WP.grant_eur")
        estimate = _integer(wp.get("estimated_cost_eur"), "WP.estimated_cost_eur")
        if estimate < amount:
            raise ValueError("work-package estimated cost cannot be below its grant allocation")
        allocation = wp.get("partner_grants")
        if not isinstance(allocation, dict) or set(allocation) != partner_ids:
            raise ValueError("each WP must allocate grant to exactly the identified partners")
        if (
            sum(_integer(value, "partner_grant", minimum=0) for value in allocation.values())
            != amount
        ):
            raise ValueError("partner allocations must reconcile with the work-package grant")
        if not isinstance(wp.get("objectives"), list) or not wp["objectives"]:
            raise ValueError("each WP needs objective references")
        if not set(wp["objectives"]).issubset(objective_ids):
            raise ValueError("unknown work-package objective reference")
        cost_items = _objects(wp.get("cost_items"), "cost_items")
        if (
            sum(_integer(item.get("amount_eur"), "cost_item.amount_eur") for item in cost_items)
            != estimate
        ):
            raise ValueError("cost assumptions must reconcile with the WP estimated cost")
        for item in cost_items:
            _string(item.get("label"), "cost_item.label")
        for activity in _objects(wp.get("activities"), "activities"):
            for field in ("id", "title", "description", "deliverable", "acceptance"):
                _string(activity.get(field), f"activity.{field}")
            a_start = _integer(activity.get("start"), "activity.start")
            a_end = _integer(activity.get("end"), "activity.end")
            if not start <= a_start <= a_end <= end:
                raise ValueError("activity dates must lie inside their work package")
            if activity["id"] in activity_ids:
                raise ValueError("activity IDs must be unique")
            activity_ids.add(activity["id"])
            activities[activity["id"]] = activity
    if sum(wp["grant_eur"] for wp in packages) != grant:
        raise ValueError("work-package grants must sum to the requested grant")
    if sum(wp["estimated_cost_eur"] for wp in packages) != total:
        raise ValueError("work-package estimates must sum to overall estimated cost")
    management = next(wp for wp in packages if wp["id"] == "WP1")
    if management["start"] != 1 or management["end"] != months:
        raise ValueError("management must cover the full project duration")
    if management["grant_eur"] * 5 > grant or management["estimated_cost_eur"] * 5 > total:
        raise ValueError("management allocation exceeds the 20% check")
    if "A3.2" not in activities or "A4.1" not in activities:
        raise ValueError("recipe requires facilitator preparation A3.2 and pilot A4.1")
    if activities["A3.2"]["end"] >= activities["A4.1"]["start"]:
        raise ValueError("facilitator preparation must finish before pilot delivery starts")
    for risk in _objects(brief.get("risks"), "risks"):
        for field in ("id", "risk", "likelihood", "impact", "trigger", "mitigation"):
            _string(risk.get(field), f"risk.{field}")
        if risk.get("owner") not in partner_ids:
            raise ValueError("risk owner must be a known partner")
    if not isinstance(brief.get("assumptions"), list) or not brief["assumptions"]:
        raise ValueError("explicit assumptions are required")
    for assumption in brief["assumptions"]:
        _string(assumption, "assumption")
    for evidence in _objects(brief.get("missing_evidence"), "missing_evidence"):
        for field in ("item", "due", "reason"):
            _string(evidence.get(field), f"missing_evidence.{field}")
        if evidence.get("owner") not in partner_ids:
            raise ValueError("evidence owner must be a known partner")
    return brief


def compute_plan(brief: dict) -> dict:
    count = len(brief["partners"])
    learners = count * brief["learners_per_partner"]
    values = {
        "partners": count,
        "learners": learners,
        "facilitators": count * brief["facilitators_per_partner"],
        "interviews": count * brief["interviews_per_partner"],
        "codesign": count * brief["codesign_per_partner"],
        "completions": (learners * brief["completion_percent"] + 99) // 100,
        "fewer_opportunities": (learners * brief["fewer_opportunities_percent"] + 99) // 100,
        "cofinancing": brief["estimated_cost_eur"] - brief["grant_eur"],
        **{
            key: brief[key]
            for key in ("pilot_hours", "skill_gain_pp", "external_adopters", "resource_visits")
        },
    }
    values["partner_grants"] = {
        partner["id"]: sum(wp["partner_grants"][partner["id"]] for wp in brief["work_packages"])
        for partner in brief["partners"]
    }
    # Reject misspelled expansion keys before producing a partial document.
    for text in (
        brief["rationale"],
        brief["target_group"],
        *brief["assumptions"],
        *(obj["text"] for obj in brief["objectives"]),
        *(a["description"] for wp in brief["work_packages"] for a in wp["activities"]),
    ):
        try:
            text.format_map(values)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown or malformed planning placeholder: {exc}") from exc
    return values


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows),
        ]
    )


def render_proposal(brief: dict, plan: dict, *, inject_faults: bool = False) -> str:
    """Render a complete planning narrative from inputs and derived, reconciled figures."""
    sections = {}
    grant = brief["grant_eur"] + (10000 if inject_faults else 0)
    sections[SECTIONS[0]] = (
        f"**{brief['acronym']}: {brief['title']}**\n\n"
        f"Action: {brief['action']} (fictional adult-education cooperation partnership)\n\n"
        f"Requested grant: EUR {grant:,}\n\n"
        f"Estimated total project cost: EUR {brief['estimated_cost_eur']:,}\n\n"
        f"Duration: {brief['duration_months']} months\n\n"
        f"Planned unique learners: {plan['learners']}\n\n"
        f"Planned facilitators: {plan['facilitators']}\n\n"
        f"Planned completions: {plan['completions']}\n\n"
        "The partnership will turn tested learner needs into accessible teaching units, prepare "
        "facilitators, run supported community pilots and help other providers adopt the results. "
        "A common assessment rubric and cross-partner moderation will support comparison. "
        "All targets are proposed commitments for this exercise, not achieved results."
    )
    sections[SECTIONS[1]] = (
        "### Supplied scenario inputs\n\n"
        "The JSON brief supplies fictional partner profiles, objectives, activities, targets "
        "and resource estimates. The workflow checks consistency; it has not verified these "
        "claims against registries, participant evidence or partner agreements. Official programme "
        "references are listed separately at the end.\n\n### Planning assumptions\n\n"
        + "\n".join(f"- {item.format_map(plan)}" for item in brief["assumptions"])
        + "\n\n### Missing evidence and decisions\n\n"
        + _table(
            ["Evidence / decision", "Owner", "Due", "Why still open"],
            [
                [item["item"], item["owner"], item["due"], item["reason"]]
                for item in brief["missing_evidence"]
            ],
        )
    )
    sections[SECTIONS[2]] = (
        brief["rationale"].format_map(plan)
        + "\n\n"
        + brief["european_added_value"]
        + "\n\nThe selected horizontal priority is inclusion and diversity. "
        "Digital practice serves "
        "that inclusion objective: it is not a claim that providing technology alone removes "
        "barriers. The programme's priority descriptions provide the policy context. "
        f"[Priorities]({SOURCES['priorities']})"
    )
    sections[SECTIONS[3]] = (
        "\n\n".join(
            f"**{obj['id']}** — {obj['text'].format_map(plan)}" for obj in brief["objectives"]
        )
        + "\n\n"
        + brief["target_group"].format_map(plan)
        + "\n\nRecruitment will use community referral routes and plain-language information. "
        "Applicants can request accessible formats and reasonable practical support without "
        "publicly disclosing sensitive details. Participants will help review materials and can "
        "withdraw from optional research without losing learning support."
    )
    sections[SECTIONS[4]] = _table(
        ["ID", "Fictional organisation", "Country", "Role", "Contribution"],
        [
            [p["id"], p["name"], p["country"], p["role"], p["contribution"]]
            for p in brief["partners"]
        ],
    ) + (
        "\n\nThe coordinator consolidates delivery evidence and the budget; work-package leads "
        "remain responsible for their outputs. A steering group with one representative per "
        "partner meets quarterly and records decisions. Operational leads meet monthly. Changes "
        "affecting agreed activities, responsibilities or resources require a documented decision "
        "and any relevant agency approval. Disagreements first go to the work-package lead, then "
        "to the steering group. Adult participant representatives review accessibility and "
        "learning materials without receiving access to identifiable evaluation records."
    )
    packages = []
    for wp in brief["work_packages"]:
        activities = "\n\n".join(
            f"**{a['id']} — {a['title']} (M{a['start']}–M{a['end']})**\n\n"
            f"{a['description'].format_map(plan)}\n\n"
            f"Deliverable: {a['deliverable']}. Acceptance: {a['acceptance']}"
            for a in wp["activities"]
        )
        packages.append(
            f"### {wp['id']} — {wp['title']}\n\nLead: {wp['lead']}; "
            f"months {wp['start']}–{wp['end']}; objectives: {', '.join(wp['objectives'])}."
            f"\n\n{activities}"
        )
    sections[SECTIONS[5]] = "\n\n".join(packages)
    sections[SECTIONS[6]] = _table(
        ["Work package", "Start month", "End month", "Gate / output"],
        [
            [wp["id"], wp["start"], wp["end"], wp["activities"][-1]["deliverable"]]
            for wp in brief["work_packages"]
        ],
    ) + (
        "\n\nDependencies: the M6 needs gate controls the curriculum brief; early prototyping "
        "does not bypass it. Facilitator preparation (A3.2) must finish before pilot delivery "
        "(A4.1) begins. Assessment methods are agreed before baseline collection. Pilot findings "
        "inform the final transfer materials. The final steering review checks deliverable "
        "acceptance, unresolved evidence and named post-project owners. Months are relative: "
        "a real start date and call deadline still need to be selected and checked."
    )
    wp_rows = [
        [
            wp["id"],
            wp["title"],
            f"{wp['grant_eur']:,}",
            f"{wp['estimated_cost_eur']:,}",
            f"{wp['estimated_cost_eur'] - wp['grant_eur']:,}",
        ]
        for wp in brief["work_packages"]
    ]
    partner_ids = [p["id"] for p in brief["partners"]]
    allocation_rows = [
        [
            wp["id"],
            *(f"{wp['partner_grants'][pid]:,}" for pid in partner_ids),
            f"{wp['grant_eur']:,}",
        ]
        for wp in brief["work_packages"]
    ]
    allocation_rows.append(
        [
            "Total",
            *(f"{plan['partner_grants'][pid]:,}" for pid in partner_ids),
            f"{brief['grant_eur']:,}",
        ]
    )
    management = next(wp for wp in brief["work_packages"] if wp["id"] == "WP1")
    sections[SECTIONS[7]] = (
        _table(["WP", "Purpose", "Grant EUR", "Estimated cost EUR", "Own resources EUR"], wp_rows)
        + "\n\nWork-package allocation total: EUR "
        + f"{sum(wp['grant_eur'] for wp in brief['work_packages']):,}"
        + f"\n\nEstimated own resources: EUR {plan['cofinancing']:,}. "
        "Written commitments remain open."
        + "\n\n### Partner grant allocation\n\n"
        + _table(["WP", *partner_ids, "Total EUR"], allocation_rows)
        + f"\n\nPartner allocation total: EUR {sum(plan['partner_grants'].values()):,}"
        + f"\n\nManagement represents {management['grant_eur'] / brief['grant_eur']:.1%} of grant "
        f"allocation and {management['estimated_cost_eur'] / brief['estimated_cost_eur']:.2%} "
        "of estimated total cost.\n\n### Cost assumptions and value for money\n\n"
        + "\n".join(
            f"- **{wp['id']}**: "
            + "; ".join(f"{item['label']} EUR {item['amount_eur']:,}" for item in wp["cost_items"])
            + "."
            for wp in brief["work_packages"]
        )
        + "\n\nThese are fictional planning estimates, not reimbursement rates or verified quotes. "
        "The grant is the selected lump sum; the larger cost estimate shows the planned resource "
        "envelope. Partners propose to cover the gap from their own unrestricted cash budgets. "
        "Shared materials, blended coordination and reuse of existing accessible tools avoid "
        "unnecessary duplication. Core design and delivery remain with the partners. Financial "
        "feasibility, commitments and applicable grant terms need human review. "
        f"[Funding model]({SOURCES['action']})"
    )
    sections[SECTIONS[8]] = _table(
        ["Indicator", "Planned target", "Evidence and timing", "Owner"],
        [
            [
                "Unique adult learners",
                plan["learners"],
                "De-duplicated enrolment register, M11–M18",
                "P1",
            ],
            [
                "Facilitators prepared",
                plan["facilitators"],
                "Practice and moderation records, M10",
                "P3",
            ],
            [
                "Adult completions",
                plan["completions"],
                "Attendance >=80% plus final-task attempt, M20",
                "P2",
            ],
            [
                "Learners facing defined access barriers",
                plan["fewer_opportunities"],
                "Voluntary, minimised access-needs records; report aggregates, M18",
                "P1",
            ],
            [
                "Mean paired assessment gain",
                f"{plan['skill_gain_pp']} percentage points",
                "Same practical rubric before/after; report missing pairs, M20",
                "P2",
            ],
            [
                "External adoption plans",
                plan["external_adopters"],
                "Named owner and delivery plan, M24",
                "P3",
            ],
            [
                "Unique resource-page visits",
                plan["resource_visits"],
                "Privacy-aware analytics, M24",
                "P3",
            ],
        ],
    ) + (
        "\n\nBaseline competence and current provider practice are not known yet and will be "
        "measured rather than set to a fictional zero. The evaluation will report recruitment "
        "and completion denominators, withdrawals and missing assessments. Participant feedback "
        "will explain usability and relevance alongside numerical results. Without a suitable "
        "comparison design, before/after gains will be described as observed changes, not "
        "effects caused solely by the project. Provider adoption is distinct from downloads, "
        "and resource visits are not counted as trained people."
    )
    if not inject_faults:
        sections[SECTIONS[9]] = _table(
            ["ID", "Risk", "Likelihood / impact", "Owner", "Trigger", "Response"],
            [
                [
                    r["id"],
                    r["risk"],
                    f"{r['likelihood']} / {r['impact']}",
                    r["owner"],
                    r["trigger"],
                    r["mitigation"],
                ]
                for r in brief["risks"]
            ],
        ) + (
            "\n\nSafeguards include accessible alternatives, clear participant information, "
            "limited data collection, role-based access and an agreed retention/deletion schedule. "
            "Local data-handling and participation arrangements need competent review before "
            "recruitment. Prefer virtual coordination where effective and compare lower-impact "
            "travel options for necessary meetings; no unmeasured carbon-reduction claim is made."
        )
    sections[SECTIONS[10]] = (
        "Dissemination begins with identifying adult-learning networks and intended adopters. "
        "At pilot stage, share progress and practical lessons without presenting provisional "
        "findings as proven impact. At M18–M24, publish the accessible resource collection and "
        "evaluation limitations, run demonstrations and support adoption planning. Use partner "
        "channels and EPALE where appropriate; acknowledge any actual EU support only under "
        "the applicable grant and communication requirements. This fictional draft must never "
        "be described as funded.\n\n"
        f"The target is {plan['external_adopters']} external adoption plans "
        f"and {plan['resource_visits']} "
        "unique resource-page visits. Each provider plan records intended learners, resources, "
        "delivery dates and an owner. P3 proposes to maintain the resource repository for twelve "
        "months after project end; P2 maintains the assessment guidance and P1 convenes a "
        "six-month post-project review. Partners must confirm this staff time and hosting budget "
        "in writing. Planned open licensing applies only to material the consortium has the "
        "rights to release; third-party content will be checked separately. No continued funding "
        "or adoption is assumed to have been secured."
    )
    sections[SECTIONS[11]] = (
        f"Rule snapshot checked {CHECKED_ON}; the guide page identifies the 2026 call, version 1 "
        "dated 12 November 2025. This is a historical planning exercise, not a claim that a call "
        "is currently open.\n\n"
        f"- [Programme Guide and version]({SOURCES['guide']})\n"
        f"- [Cooperation Partnerships: action, funding and work packages]({SOURCES['action']})\n"
        f"- [EU and associated countries]({SOURCES['countries']})\n"
        f"- [Partnership priorities]({SOURCES['priorities']})\n\n"
        "The automated checks cover only selected scenario constraints and document consistency. "
        "They do not verify legal identities, registration, financial/operational capacity, "
        "application limits, exclusion criteria, national priorities, the applicable form or "
        "all other action conditions. A human must select the real call, gather the missing "
        "evidence and complete the official application. No award score is predicted."
    )
    title = f"# {brief['acronym']} — {brief['title']}\n\n"
    notice = (
        "> **FICTIONAL WORKED DRAFT — NOT SUBMISSION-READY.** "
        "No real consortium or award is claimed.\n\n"
    )
    return (
        title
        + notice
        + "\n\n".join(
            f"## {heading}\n\n{sections[heading]}" for heading in SECTIONS if heading in sections
        )
        + "\n"
    )


def _section(markdown: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", markdown, re.M | re.S)
    return match.group(1) if match else ""


def review_proposal(markdown: str, brief: dict) -> dict:
    """Inspect the actual Markdown independently of the writer's claimed success."""
    plan = compute_plan(validate_brief(brief))
    errors = []
    checks = 0

    def check(condition: bool, code: str, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append({"code": code, "message": message})

    check(
        "FICTIONAL WORKED DRAFT" in markdown and "NOT SUBMISSION-READY" in markdown,
        "missing_disclaimer",
        "Retain the fictional/non-submission status.",
    )
    for heading in SECTIONS:
        check(
            len(re.findall(rf"^## {re.escape(heading)}$", markdown, re.M)) == 1
            and len(_section(markdown, heading).strip()) >= 60,
            "missing_section",
            f"Include one substantive '{heading}' section.",
        )
    claims = {
        "Requested grant: EUR": brief["grant_eur"],
        "Estimated total project cost: EUR": brief["estimated_cost_eur"],
        "Duration:": brief["duration_months"],
        "Planned unique learners:": plan["learners"],
        "Planned facilitators:": plan["facilitators"],
        "Planned completions:": plan["completions"],
        "Work-package allocation total: EUR": brief["grant_eur"],
        "Partner allocation total: EUR": brief["grant_eur"],
    }
    for label, expected in claims.items():
        values = re.findall(rf"^{re.escape(label)} ([\d,]+)", markdown, re.M)
        check(
            len(values) == 1 and int(values[0].replace(",", "")) == expected,
            "claim_mismatch",
            f"Reconcile '{label}' with the validated value {expected}.",
        )
    budget = _section(markdown, SECTIONS[7]).split("### Partner grant allocation", 1)[0]
    rows = [line.split("|")[1:-1] for line in budget.splitlines() if re.match(r"\| WP\d+ \|", line)]
    check(
        len(rows) == len(brief["work_packages"]),
        "budget_rows",
        "Include every work-package budget row.",
    )
    for wp in brief["work_packages"]:
        match = next((row for row in rows if row[0].strip() == wp["id"]), None)
        try:
            numbers = [int(value.strip().replace(",", "")) for value in match[2:5]] if match else []
        except ValueError:
            numbers = []
        check(
            numbers
            == [
                wp["grant_eur"],
                wp["estimated_cost_eur"],
                wp["estimated_cost_eur"] - wp["grant_eur"],
            ],
            "budget_mismatch",
            f"Reconcile {wp['id']} budget and own-resource figures.",
        )
        timeline = _section(markdown, SECTIONS[6])
        check(
            f"| {wp['id']} | {wp['start']} | {wp['end']} |" in timeline,
            "timeline_mismatch",
            f"Reconcile {wp['id']} timeline.",
        )
        for activity in wp["activities"]:
            check(
                activity["deliverable"] in _section(markdown, SECTIONS[5]),
                "missing_deliverable",
                f"Describe {activity['deliverable']}.",
            )
    for objective in brief["objectives"]:
        check(
            f"**{objective['id']}**" in _section(markdown, SECTIONS[3]),
            "missing_objective",
            f"Include objective {objective['id']}.",
        )
    allocation = _section(markdown, SECTIONS[7]).partition("### Partner grant allocation")[2]
    allocation = allocation.partition("### Cost assumptions")[0]
    allocation_rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in allocation.splitlines()
        if re.match(r"\| (WP\d+|Total) \|", line)
    ]
    partner_ids = [p["id"] for p in brief["partners"]]
    expected_rows = {
        wp["id"]: [wp["partner_grants"][pid] for pid in partner_ids] + [wp["grant_eur"]]
        for wp in brief["work_packages"]
    }
    expected_rows["Total"] = [plan["partner_grants"][pid] for pid in partner_ids] + [
        brief["grant_eur"]
    ]
    check(
        len(allocation_rows) == len(expected_rows),
        "allocation_rows",
        "Include every partner allocation row and one total row.",
    )
    for label, expected in expected_rows.items():
        matches = [row for row in allocation_rows if row[0] == label]
        try:
            actual = [int(value.replace(",", "")) for value in matches[0][1:]]
        except (ValueError, IndexError):
            actual = []
        check(
            len(matches) == 1 and actual == expected,
            "allocation_mismatch",
            f"Reconcile the {label} partner allocation row.",
        )
    indicators = {
        "Unique adult learners": str(plan["learners"]),
        "Facilitators prepared": str(plan["facilitators"]),
        "Adult completions": str(plan["completions"]),
        "Learners facing defined access barriers": str(plan["fewer_opportunities"]),
        "Mean paired assessment gain": f"{plan['skill_gain_pp']} percentage points",
        "External adoption plans": str(plan["external_adopters"]),
        "Unique resource-page visits": str(plan["resource_visits"]),
    }
    impact = _section(markdown, SECTIONS[8])
    for label, target in indicators.items():
        values = re.findall(rf"^\| {re.escape(label)} \| ([^|]+) \|", impact, re.M)
        check(
            len(values) == 1 and values[0].strip() == target,
            "indicator_mismatch",
            f"Reconcile the '{label}' target.",
        )
    risk_section = _section(markdown, SECTIONS[9])
    for risk in brief["risks"]:
        check(
            all(
                value in risk_section
                for value in (
                    f"| {risk['id']} |",
                    f"| {risk['owner']} |",
                    risk["trigger"],
                    risk["mitigation"],
                )
            ),
            "missing_risk",
            f"Include {risk['id']} with its trigger and response owner.",
        )
    for url in SOURCES.values():
        check(url in markdown, "missing_citation", f"Keep the official source {url}.")
    evidence_section = _section(markdown, SECTIONS[1])
    for item in brief["missing_evidence"]:
        check(
            item["item"] in evidence_section and item["reason"] in evidence_section,
            "missing_evidence_notice",
            f"Disclose unresolved evidence: {item['item']}.",
        )
    warnings = [
        {"code": "missing_evidence", "message": item["item"], "owner": item["owner"]}
        for item in brief["missing_evidence"]
    ]
    warnings.append(
        {
            "code": "limited_rule_snapshot",
            "message": (
                "Check the real call and full eligibility; this is a fictional 2026 exercise."
            ),
        }
    )
    return {
        "checks_run": checks,
        "internal_checks_passed": not errors,
        "submission_ready": False,
        "errors": errors,
        "warnings": warnings,
    }


class ProposalEvaluator:
    def __init__(self, reviewer: QAgentsGenerator, brief: dict) -> None:
        self.reviewer, self.brief = reviewer, brief

    async def evaluate(self, task: Task, generation: Generation) -> Reward:
        if generation.status != "completed":
            return Reward(0, "proposal-integrity-v1", json.dumps({"errors": ["incomplete run"]}))
        result = await self.reviewer.generate(
            Task(
                f"review:{task.task_id}",
                json.dumps({"brief": self.brief, "markdown": generation.output}),
            ),
            Context("proposal-review"),
        )
        if result.status != "completed":
            raise ValueError("proposal reviewer did not complete")
        report = json.loads(result.output)
        return Reward(
            float(report["internal_checks_passed"]), "proposal-integrity-v1", result.output
        )


def _reserve_output(output: Path) -> None:
    """Claim a fresh/empty directory; retain the marker on success, error or cancellation."""
    refusal = "output directory must be new or empty; existing artifacts are not overwritten"
    if output.exists() and any(output.iterdir()):
        raise ValueError(refusal)
    output.mkdir(parents=True, exist_ok=True)
    reservation = output / ".run-reservation"
    try:
        with reservation.open("x", encoding="utf-8") as handle:
            handle.write(
                "Reserved by an Erasmus example run. Retained after success or failure.\n"
                "Use a fresh destination to retry; this marker is not a completion signal.\n"
            )
    except FileExistsError as exc:
        raise ValueError(refusal) from exc
    if any(path != reservation for path in output.iterdir()):
        raise ValueError(refusal)


def _write_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


async def run(
    output_dir: str | Path, *, brief: dict | None = None, inject_faults: bool = True
) -> dict:
    """Write auditable initial/final documents only after validating a complete brief."""
    brief = validate_brief(
        brief if brief is not None else json.loads(DEFAULT_BRIEF.read_text(encoding="utf-8"))
    )
    compute_plan(brief)
    output = Path(output_dir)
    _reserve_output(output)

    def planner_policy(request: ModelRequest) -> str:
        return json.dumps(compute_plan(validate_brief(task_payload(request))))

    def writer_policy(request: ModelRequest) -> str:
        payload = task_payload(request)
        faults = inject_faults and not has_rule(request, CORRECTION_RULE)
        return render_proposal(payload["brief"], payload["plan"], inject_faults=faults)

    def reviewer_policy(request: ModelRequest) -> str:
        payload = task_payload(request)
        return json.dumps(review_proposal(payload["markdown"], payload["brief"]))

    def reflector_policy(request: ModelRequest) -> str:
        episode = json.loads(user_prompt(request).split("EPISODE:\n", 1)[1])
        feedback = json.loads(episode["reward"]["feedback"])
        return json.dumps(
            [{"bullet_id": "proposal-integrity", "text": CORRECTION_RULE}]
            if feedback["errors"]
            else []
        )

    models = {
        name: FunctionModel(policy, name=f"quentin-{name}")
        for name, policy in (
            ("planner", planner_policy),
            ("writer", writer_policy),
            ("reviewer", reviewer_policy),
            ("reflector", reflector_policy),
        )
    }
    agents = {
        name: QAgentsGenerator(
            Agent(model, name=f"Quentin-{name}"),
            limits=Limits(max_turns=2, timeout_s=30, max_total_tokens=48000),
            max_context_chars=100000,
        )
        for name, model in models.items()
    }
    planned = await agents["planner"].generate(
        Task("proposal-plan", json.dumps(brief)), Context("planner")
    )
    if planned.status != "completed":
        raise ValueError("planning did not complete")
    plan = json.loads(planned.output)
    if plan != compute_plan(brief):
        raise ValueError("planner output does not reconcile with the validated brief")
    payload = json.dumps({"brief": brief, "plan": plan})
    # SQLite normally opens existing files; establish ownership before it can do so.
    with (output / "gym.sqlite3").open("xb"):
        pass
    with SQLiteStore(output / "gym.sqlite3") as store:
        gym = Gym(
            agents["writer"],
            ProposalEvaluator(agents["reviewer"], brief),
            QAgentsReflector(agents["reflector"]),
            store,
        )
        initial = await gym.train(Task("proposal-initial", payload), namespace="erasmus-proposal")
        frozen = (store.snapshot("erasmus-proposal"), store.memories("erasmus-proposal"))
        (corrected,) = await gym.evaluate(
            [Task("proposal-corrected", payload, split="eval")], namespace="erasmus-proposal"
        )
        unchanged = frozen == (
            store.snapshot("erasmus-proposal"),
            store.memories("erasmus-proposal"),
        )
        before, after = json.loads(initial.reward.feedback), json.loads(corrected.reward.feedback)
        report = {
            "mode": "fictional KA220-ADU; deterministic policies; real qagents execution",
            "action": brief["action"],
            "initial_error_count": len(before["errors"]),
            "final_error_count": len(after["errors"]),
            "remaining_evidence_warnings": len(after["warnings"]),
            "internal_checks_passed": after["internal_checks_passed"],
            "submission_ready": False,
            "learning_revision": store.snapshot("erasmus-proposal").revision,
            "frozen_final_review": unchanged,
            "agent_calls": {name: len(model.requests) for name, model in models.items()},
            "grant_eur": brief["grant_eur"],
            "estimated_cost_eur": brief["estimated_cost_eur"],
            "learners": plan["learners"],
            "facilitators": plan["facilitators"],
            "source_check_date": CHECKED_ON,
            "sources": SOURCES,
        }
        _write_new(output / "draft-initial.md", initial.generation.output)
        _write_new(output / "proposal.md", corrected.generation.output)
        for name, value in (
            ("brief.json", brief),
            ("plan.json", plan),
            ("review-before.json", before),
            ("review-after.json", after),
            ("workflow.json", report),
        ):
            _write_new(output / name, json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        export_jsonl((initial, corrected), output / "episodes.jsonl")
    return {**report, "proposal_path": str(output / "proposal.md")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF)
    parser.add_argument("--output-dir", type=Path, default=Path(".gym/erasmus-proposal"))
    parser.add_argument(
        "--clean-first-draft", action="store_true", help="Skip the deliberate QA fault injection"
    )
    args = parser.parse_args(argv)
    try:
        brief = json.loads(args.brief.read_text(encoding="utf-8"))
        report = asyncio.run(
            run(args.output_dir, brief=brief, inject_faults=not args.clean_first_draft)
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["internal_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
