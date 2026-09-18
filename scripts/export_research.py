#!/usr/bin/env python3
"""Build the public research snapshot from the complete, local evidence packet.

This is a read-only historical export. It never submits jobs, executes scorers,
modifies the research repository, or substitutes new statistical estimates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import register_model  # noqa: E402  (four-outcome model and pinned partition-audit import)


PORTAL = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PORTAL.parents[1]
ACCOUNT_LABELS = {"salehkaleybars": "Account1", "s5014158": "Account2"}
PRIVATE_RE = re.compile(
    r"/Users/|/home/|/data1/|/scratch/|/zfsstore/user/|/private/var/|teshnizi|salehkaleybars|s5014158|hmkhd2|"
    r"100\.120\.248\.20|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I
)
HOST_RE = re.compile(r"\b(?:p-cfer-\d+|node\d{3}|nodelogin\d+|login\d+|login\.[A-Za-z0-9_.…-]+)\b", re.I)
SECRET_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b")
PRIVATE_HINT_RE = re.compile(r"/Users/|/home/|/data1/|/scratch/|/zfsstore/|/private/|teshnizi|salehkaleybars|s5014158|hmkhd2|100\.120\.248\.20|@|\bsk-|\bgh[pousr]_|api[_-]?key|access[_-]?token|auth[_-]?token|password|secret", re.I)
ID_RE = re.compile(r"\b(?:MT\d{3}|CVK2)\b")
OUTCOMES = set(register_model.OUTCOMES)  # "Corrected" is a badge, never an outcome.
KINDS = set(register_model.KINDS)
PARTITION_IDS = [register_model.partition_id(line) for line in range(register_model.FIRST_ROW, register_model.LAST_ROW + 1)]
APPENDED_IDS = [register_model.partition_id(line) for line in range(register_model.APPENDED_FIRST_ROW, register_model.APPENDED_LAST_ROW + 1)]
LANDED_IDS = [register_model.partition_id(line) for line in range(register_model.LANDED_FIRST_ROW, register_model.LANDED_LAST_ROW + 1)]
CGN3_IDS = [register_model.partition_id(line) for line in range(register_model.CGN3_FIRST_ROW, register_model.CGN3_LAST_ROW + 1)]
CVT23_IDS = [register_model.partition_id(line) for line in range(register_model.CVT23_FIRST_ROW, register_model.CVT23_LAST_ROW + 1)]
CVT45_IDS = [register_model.partition_id(line) for line in range(register_model.CVT45_FIRST_ROW, register_model.CVT45_LAST_ROW + 1)]
CVT67_IDS = [register_model.partition_id(line) for line in range(register_model.CVT67_FIRST_ROW, register_model.CVT67_LAST_ROW + 1)]
CVT89_IDS = [register_model.partition_id(line) for line in range(register_model.CVT89_FIRST_ROW, register_model.CVT89_LAST_ROW + 1)]
MUST_IDS = [register_model.partition_id(line) for line in range(register_model.MUST_FIRST_ROW, register_model.MUST_LAST_ROW + 1)]
REGISTER_CSV = "complete_experiment_register.csv"
RUN_INVENTORY_CSV = "complete_run_inventory.csv"
# Summary tables derived from the register are regenerated from it, like the register table.
OUTCOMES_BY_AREA_CSV = "outcomes_by_area.csv"
GOAL_OUTCOMES_CSV = "goal_outcomes.csv"
OUTCOME_COUNT_COLUMNS = ["research_questions", "success", "fail", "mixed", "unresolved", "method_checks", "corrected"]
OUTCOMES_BY_AREA_COLUMNS = ["area", "records", *OUTCOME_COUNT_COLUMNS, "interpretation"]
GOAL_OUTCOMES_COLUMNS = ["page", "title", "goal", "comparison", "why_test", "status", "scope", "sources", "file", "kind", "entry_ids", *OUTCOME_COUNT_COLUMNS]
OUTCOME_INTERPRETATION = "Outcomes judge each stated goal; these are not method win rates or independent hypotheses. Method checks carry no research outcome, and Corrected is a badge beside the outcome."
# The published register lists every record of the notebook register, not the 111-row campaign export.
REGISTER_COLUMNS = ["id", "kind", "outcome", "outcome_label", "corrected", "correction_note", "section", "area", "goal", "comparison", "why", "result", "reason", "scope", "batches", "sources", "original_question", "register_page", "mapping_rule", "master_table_line"]
PORTFOLIO_LINKS = {
    "alpha0_robustness": ["MT047", "MT048", "MT049", "MT051"],
    "alpha0_300_extremes": ["MT050"],
    "schedule_controls": ["MT014", "MT015", "MT016"],
    "baseline_model_transfer": ["MT018"],
    "baseline_budget": ["MT017"],
    "granularity_model_transfer": ["MT057", "MT058"],
    "granularity_cifar100_historical": ["MT059", "MT060", "MT149"],
    "granularity_cifar100_tuning": ["MT059", "MT149", "MT157", "MT158"],
    "granularity_dataset_network_extensions": ["MT136", "MT164", "MT166"],
    "full_granularity_ladder": ["MT026", "MT027", "MT028", "MT033", "MT034"],
    "weightwise_tuning": ["MT029", "MT033"],
    "horizon_reversal": ["MT039", "MT040", "MT041", "MT063"],
    "long_trajectory_pooling": ["MT062", "MT063"],
    "initialization_startup": ["MT061"],
    "abandoned_fixes": ["MT069", "MT071", "MT072", "MT074", "MT075", "MT079"],
    "additive_transfer_curves": ["MT058", "MT078", "MT080", "MT135"],
    "original_proposal_and_joint_tuning": ["MT073", "MT074", "MT076", "MT132", "MT133", "MT134"],
    "cau1": ["MT019", "MT020"],
    "cdn1": ["MT155"],
    "count_matched_four_submissions": ["MT098"],
}
TOP_TABLE_LINKS = {
    "campaign_selected_cells.csv": ["MT047", "MT048", "MT049", "MT050", "MT051", "MT057", "MT058", "MT073", "MT098", "MT132", "MT133", "MT134", "MT136", "MT157"],
    "campaign_selected_seeds.csv": ["MT047", "MT048", "MT049", "MT050", "MT051", "MT057", "MT058", "MT073", "MT098", "MT132", "MT133", "MT134", "MT136", "MT157"],
    "isolation_endpoints.csv": ["MT162", "MT163", "MT164", "MT165", "MT166"],
    "isolation_seed_values.csv": ["MT162", "MT163", "MT164", "MT165", "MT166"],
    "ciso2_epoch_curves.csv": ["MT165"],
    "cut_position_cells.csv": ["MT141", "MT142", "MT145", "MT146", "MT147", "MT148", "MT150"],
}
PHASE_LINKS = [
    ["MT069", "MT071", "MT072", "MT073", "MT074", "MT075"],
    ["MT026", "MT086", "MT087", "MT088", "MT089", "MT093", "MT098"],
    ["MT098", *PARTITION_IDS], ["MT132", "MT133", "MT134"],  # phase-03 Partition tests: MASTER-TABLE section 10
    ["MT141", "MT142", "MT145", "MT146", "MT147", "MT149", "MT157"],
    ["MT162", "MT163", "MT165"], ["MT019", "MT020", "MT164", "MT166"], ["CVK2"],
]
PHASE_STARTS = ["2026-08-18", "2026-08-20", "2026-08-24", "2026-09-03", "2026-09-04", "2026-09-08", "2026-09-09", "2026-09-14"]
EXPECTED_STATS = {"experiments": 168, "researchQuestions": 150, "methodChecks": 18, "runs": 3181, "figures": 54, "areas": 10}
CVK2_RUNS = 27


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_id(prefix, value):
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:16]


def ids_in(value):
    return list(dict.fromkeys(ID_RE.findall(str(value))))


@lru_cache(maxsize=8192)
def sanitize_text(text, workspace=None):
    """Replace private provenance prefixes while retaining scientific content."""
    text = str(text)
    if not PRIVATE_HINT_RE.search(text) and not HOST_RE.search(text):
        return text
    # Match the source catalog's stable host aliases without redacting scientific
    # group identifiers such as node14420 or labels such as PEAK-AT-22.
    text = HOST_RE.sub(lambda m: "[HOST_" + hashlib.sha256(m[0].lower().encode()).hexdigest()[:10] + "]", text)
    if workspace:
        text = text.replace(str(Path(workspace).resolve()), "publication-workspace")
    # Longest known prefixes first: filenames and JSON pointers remain meaningful.
    replacements = [
        (str(Path.home() / "Saber Optimization/alice-backup/hierarchical-metaoptimize"), "repository"),
        (str(Path.home() / "Saber Optimization/alice-backup/runs_alice2"), "archive/Account2/runs"),
        (str(Path.home() / "Saber Optimization/alice-backup/runs"), "archive/Account1/runs"),
        (str(Path.home() / "Saber Optimization/alice-backup"), "archive"),
    ]
    for private, public in replacements:
        text = text.replace(private, public)
    for account, label in ACCOUNT_LABELS.items():
        for root in ["/home/", "/data1/", "/zfsstore/user/", "/scratch/"]:
            text = text.replace(root + account, "cluster/" + label)
    text = re.sub(r"/Users/[^/\s\"'<>]+", "local-user", text)
    text = re.sub(r"/(?:home|data1|scratch)/[^/\s\"'<>]+", "cluster-user", text)
    text = re.sub(r"/zfsstore/user/[^/\s\"'<>]+", "cluster-user", text)
    text = text.replace("/private/var/", "temporary/")
    text = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "[email-redacted]", text, flags=re.I)
    for account, label in ACCOUNT_LABELS.items():
        text = text.replace(account, label)
    text = re.sub(r"teshnizi|hmkhd2", "Researcher", text, flags=re.I)
    text = text.replace("100.120.248.20", "[private-host]")
    text = SECRET_RE.sub("[credential-redacted]", text)
    text = re.sub(r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)[ \t]*([=:])[ \t]*([^\s,;\"']+)", r"\1\2[redacted]", text)
    return text


def sanitize(value, workspace):
    if isinstance(value, str):
        return sanitize_text(value, workspace)
    if isinstance(value, list):
        return [sanitize(item, workspace) for item in value]
    if isinstance(value, dict):
        return {sanitize_text(key, workspace): sanitize(item, workspace) for key, item in value.items()}
    return value


def require_public(text, context):
    match = PRIVATE_RE.search(text) or SECRET_RE.search(text) or HOST_RE.search(text)
    if match:
        raise ValueError(f"Private content remains in {context}: {match.group(0)!r}")


def load_source_catalog(public_dir):
    public_dir = Path(public_dir)
    paths = [public_dir / "data/source-index.json", public_dir / "data/source-links.json"]
    if not all(path.is_file() for path in paths):
        raise ValueError("The final source catalog is required: source-index.json and source-links.json")
    sources, links = [json.loads(path.read_text()) for path in paths]
    if not isinstance(sources, list) or not isinstance(links, dict) or not sources:
        raise ValueError("Invalid source catalog schema")
    required = {"id", "path", "language", "role", "href", "originalSha256", "publicSha256", "redacted", "lines", "experimentIds", "referenceType"}
    for source in sources:
        if not required <= set(source):
            raise ValueError(f"Incomplete source-index record: {source.get('id')}")
        href = source["href"]
        if not href.startswith("/source/") or ".." in Path(href).parts:
            raise ValueError("Unsafe public source href")
        public_file = public_dir / href.lstrip("/")
        if not public_file.is_file() or digest(public_file) != source["publicSha256"]:
            raise ValueError(f"Public source file missing or hash mismatch: {source['id']}")
        require_public(public_file.read_text(), "public source " + source["id"])
    for path in paths:
        require_public(path.read_text(), path.name)
    return sources, links


def read_evidence(workspace):
    work = workspace / "work"
    return {name: json.loads((work / f"{name}.json").read_text()) for name in ["full_mechanism_evidence", "full_measurement_evidence", "full_portfolio_evidence", "cvk2_evidence"]}


def make_figures(rows):
    return [{
        "id": f"page-{row['page']}", "page": int(row["page"]), "title": row["title"],
        "goal": row["goal"], "comparison": row["comparison"], "why": row["why_test"],
        "outcome": row["status"], "scope": row["scope"],
        "href": f"/assets/figures/page-{row['page']}.webp", "experimentIds": ids_in(row["entry_ids"]), "kind": row["kind"],
    } for row in rows]


def warning_for(experiment):
    """Verdict warning for the current outcome, plus the retained correction history."""
    outcome = experiment["outcome"]
    if experiment["kind"] == "method-check":
        severity, title, status = ("correction", "A method check corrected the campaign's own process", "resolved")
    else:
        severity, title, status = {
            "success": ("limitation", "Scope of the supported result", "documented"),
            "fail": ("caution", "The tested goal was not supported", "documented"),
            "mixed": ("limitation", "The result depends on scope", "documented"),
            "unresolved": ("open", "The question remains unresolved", "open"),
        }[outcome]
    detail = " ".join(dict.fromkeys([experiment["result"], experiment["reason"], experiment["scope"]]))
    if outcome == "fail":
        detail += " This is a scientific verdict about the tested goal, not a runtime-failure classification."
    warnings = [{"id": f"warning-{experiment['id']}-verdict", "experimentId": experiment["id"], "severity": severity, "title": title, "detail": detail, "status": status}]
    amendment = experiment.get("amendment")
    if amendment:
        labels = register_model.OUTCOME_LABELS
        moved = amendment["previousOutcome"] != amendment["outcome"]
        detail = amendment["reason"]
        if moved:
            detail += f" Outcome before the amendment: {labels[amendment['previousOutcome']]}."
            detail += "".join(f" Previous {key}: {value}" for key, value in (amendment.get("previous") or {}).items())
        warnings.append({"id": f"warning-{experiment['id']}-amendment", "experimentId": experiment["id"], "severity": "limitation",
                         "title": "Outcome moved by a later result" if moved else "Wording amended by a later result", "status": "documented",
                         "detail": detail + " Amendment record: " + amendment["source"] + "."})
    later = experiment.get("later_amendment")
    if later:
        # A second in-place amendment (CORRECTIONS 231) keeps its own record beside the first.
        warnings.append({"id": f"warning-{experiment['id']}-amendment-231", "experimentId": experiment["id"], "severity": "limitation",
                         "title": "Wording amended by a later result", "status": "documented",
                         "detail": later["reason"] + " Amendment record: " + later["source"] + "."})
    for further in experiment.get("further_amendments") or []:
        # Later in-place amendments (CORRECTIONS 234, 236, 244; 253.15 keyed as 253) each keep their own record, keyed on the entry number.
        warnings.append({"id": f"warning-{experiment['id']}-amendment-{further['number']}", "experimentId": experiment["id"], "severity": "limitation",
                         "title": "Wording amended by a later result", "status": "documented",
                         "detail": further["reason"] + " Amendment record: " + further["source"] + "."})
    for corrected in experiment.get("registration_corrections") or []:
        # The landing corrected its own registration text in place (CORRECTIONS 246.9, 247.12, 253.9): documented, not a Corrected badge.
        warnings.append({"id": f"warning-{experiment['id']}-registration-{corrected['number']}", "experimentId": experiment["id"], "severity": "limitation",
                         "title": "Registration wording corrected in place", "status": "documented",
                         "detail": corrected["reason"] + " Correction record: " + corrected["source"] + "."})
    intervention = experiment.get("intervention")
    if intervention:
        warnings.append({"id": f"warning-{experiment['id']}-intervention", "experimentId": experiment["id"], "severity": "caution",
                         "title": intervention["title"], "status": "documented",
                         "detail": intervention["note"] + " Intervention record: " + intervention["source"] + "."})
    deviations = experiment.get("args_deviations")
    if deviations:
        # No patch ran: these arms deviate from the standard cell in a base-optimiser CLI flag alone (CORRECTIONS 263).
        warnings.append({"id": f"warning-{experiment['id']}-args-deviation", "experimentId": experiment["id"], "severity": "caution",
                         "title": deviations["title"], "status": "documented",
                         "detail": deviations["note"] + " Exclusion record: " + deviations["source"] + "."})
    if experiment["corrected"] and experiment["kind"] == "research":
        warnings.append({"id": f"warning-{experiment['id']}-correction", "experimentId": experiment["id"], "severity": "correction",
                         "title": "A historical claim was corrected", "status": "resolved",
                         "detail": experiment["correction"]["note"] + " Correction record: " + experiment["correction"]["source"] + "."})
    return warnings


def build_experiments(register, source_links):
    experiments, warnings = [], []
    for row in register:
        eid = row["id"]
        if eid not in source_links:
            raise ValueError(f"Source links missing for {eid}")
        linkage = source_links[eid]
        if not linkage.get("sourceRefs"):
            raise ValueError(f"No linked source reference for {eid}")
        experiment = {
            "id": eid, "section": int(row["section"]), "area": row["area"],
            "title": row["original_question"] or row["goal"],
            **{key: row[key] for key in ["goal", "comparison", "why", "result", "reason", "scope"]},
            "kind": row["kind"], "outcome": row["outcome"] or None,
            "corrected": row["corrected"] == "1", "correction": json.loads(row["correction"]) if row["correction"] else None,
            "batches": json.loads(row["batches"]), "figureIds": [], "codeIds": linkage["codeIds"],
            "tableIds": [], "warningIds": [], "eventIds": [], "runIds": [], "sourceRefs": linkage["sourceRefs"],
        }
        amendment = json.loads(row["amendment"]) if row.get("amendment") else None
        intervention = json.loads(row["intervention"]) if row.get("intervention") else None
        deviations = json.loads(row["args_deviations"]) if row.get("args_deviations") else None
        later = json.loads(row["later_amendment"]) if row.get("later_amendment") else None
        further = json.loads(row["further_amendments"]) if row.get("further_amendments") else []
        registration = json.loads(row["registration_corrections"]) if row.get("registration_corrections") else []
        for warning in warning_for({**experiment, "amendment": amendment, "intervention": intervention, "args_deviations": deviations,
                                    "later_amendment": later, "further_amendments": further,
                                    "registration_corrections": registration}):
            experiment["warningIds"].append(warning["id"])
            warnings.append(warning)
        for index, missing in enumerate(linkage.get("missing", []), 1):
            warning = {"id": f"warning-{eid}-source-{index}", "experimentId": eid, "severity": "limitation", "title": "An archived source is unavailable", "detail": str(missing), "status": "open"}
            warnings.append(warning)
            experiment["warningIds"].append(warning["id"])
        experiments.append(experiment)
    return experiments, warnings


def attach_latest_warnings(experiments, warnings, evidence):
    experiment = next(row for row in experiments if row["id"] == "CVK2")
    details = [evidence["registered_scorer_passed_means"] + " " + evidence["uncertainty"], *evidence["limitations"]]
    for index, detail in enumerate(details, 1):
        row = {"id": f"warning-CVK2-scope-{index}", "experimentId": "CVK2", "severity": "limitation", "title": "CVK2 validity and scope" if index == 1 else "CVK2 scope limitation", "detail": detail, "status": "documented"}
        warnings.append(row)
        experiment["warningIds"].append(row["id"])


def register_csv_rows(register):
    """The published register table: one row per record, four-way outcome, kind and Corrected badge as columns."""
    rows = [REGISTER_COLUMNS]
    for row in register:
        correction = json.loads(row["correction"]) if row["correction"] else {}
        sources = [{key: value for key, value in source.items() if key != "rowText"} if isinstance(source, dict) else source for source in json.loads(row.get("sources") or "[]")]
        values = {**row, "kind": row["kind"], "outcome": row["outcome"],
                  "outcome_label": register_model.OUTCOME_LABELS.get(row["outcome"], "Method check (no research outcome)"),
                  "corrected": "Corrected" if row["corrected"] == "1" else "", "correction_note": correction.get("note") or "",
                  "sources": json.dumps(sources, ensure_ascii=False), "master_table_line": row.get("master_table_line", "")}
        rows.append([str(values.get(column) or "") for column in REGISTER_COLUMNS])
    return rows


def outcome_counts(records):
    research = [row for row in records if row["kind"] == "research"]
    return {"research_questions": len(research), **{outcome: sum(row["outcome"] == outcome for row in research) for outcome in register_model.OUTCOMES},
            "method_checks": sum(row["kind"] == "method-check" for row in records), "corrected": sum(bool(row["corrected"]) for row in records)}


def outcomes_by_area_rows(experiments, areas):
    """One row per research area, counted exactly as the homepage area table counts them."""
    rows = [OUTCOMES_BY_AREA_COLUMNS]
    for area in areas:
        records = [e for e in experiments if e["area"] == area["label"]]
        values = {"area": area["label"], "records": len(records), **outcome_counts(records), "interpretation": OUTCOME_INTERPRETATION}
        rows.append([str(values[column]) for column in OUTCOMES_BY_AREA_COLUMNS])
    return rows


def goal_outcomes_rows(original_rows, figures, experiments):
    """The report pages' goal table with each page's current record links and outcome counts."""
    header, body = original_rows[0], original_rows[1:]
    if header != GOAL_OUTCOMES_COLUMNS[:len(header)]:
        raise ValueError("goal_outcomes.csv columns changed")
    by_page = {f"page-{row[0]}": row for row in body}
    by_id = {e["id"]: e for e in experiments}
    rows = [GOAL_OUTCOMES_COLUMNS]
    for figure in figures:
        if figure["id"] not in by_page:
            continue
        values = dict(zip(header, by_page.pop(figure["id"])))
        records = [by_id[eid] for eid in figure["experimentIds"]]
        values.update(entry_ids="; ".join(figure["experimentIds"]), **outcome_counts(records))
        rows.append([str(values[column]) for column in GOAL_OUTCOMES_COLUMNS])
    if by_page:
        raise ValueError(f"goal_outcomes.csv names pages without a figure: {sorted(by_page)}")
    return rows


def table_contexts(table_root, experiments):
    ids = {e["id"] for e in experiments}
    contexts = {}
    for row in read_csv(table_root / "measurement_and_mechanism/panel_index.csv"):
        contexts["measurement_and_mechanism/" + row["filename"]] = {
            "title": row["panel_title"] + " — " + row["role"].replace("_", " "),
            "experimentIds": ids_in(row["entry_ids"]),
            "scope": row["metric"] + ". " + row["scope"] + " " + row["notes"],
        }
    portfolio_union = sorted(set(eid for linked in PORTFOLIO_LINKS.values() for eid in linked) & ids)
    for row in read_csv(table_root / "portfolio/panel_index.csv"):
        for column in ["primary_cells_file", "legacy_cells_file", "contrasts_file"]:
            if row[column]:
                contexts["portfolio/" + row[column]] = {
                    "title": row["panel_id"].replace("_", " ").capitalize() + " — " + column.replace("_file", "").replace("_", " "),
                    "experimentIds": PORTFOLIO_LINKS[row["panel_id"]],
                    "scope": row["scope"] + " " + row["metric"] + " " + " ".join(json.loads(row["notes"])),
                }
    for path in table_root.rglob("*.csv"):
        rel = path.relative_to(table_root).as_posix()
        if rel in contexts:
            continue
        linked = None
        scope = "Catalog or supporting provenance for the complete historical evidence packet; individual rows retain their own metric, source, and scope."
        if path.name.startswith("cvk2_"):
            linked = ["CVK2"]
            scope = "Completed CVK2 / cvk1: VGG11_bn, CIFAR-100, 100 epochs, nine arms × three seeds 55–57. Validity checks passed; the carrier-cut prediction remains unresolved."
        elif rel == OUTCOMES_BY_AREA_CSV:
            linked = sorted(ids)
            scope = f"Outcome counts by research area, regenerated from the notebook register (scripts/register_model.py): research questions and their four outcomes (success = Goal met, fail = Goal missed, mixed, unresolved = Open), method checks listed separately, and the Corrected badge counted on its own. The campaign's earlier table with a 'correction' outcome column is an input, not this table."
        elif rel == GOAL_OUTCOMES_CSV:
            linked = sorted(ids)
            scope = "Report pages 2-23 with their stated goals, as in the report, plus each page's current linked records (entry_ids) and those records' four-way outcome, method-check and Corrected counts, regenerated from the notebook register. The page status column is the report's own page label, not a register outcome."
        elif rel == REGISTER_CSV:
            linked = sorted(ids)
            scope = f"Complete notebook register, regenerated from scripts/register_model.py: all {len(experiments)} records, each with its kind (research question or method check), its four-way outcome (Goal met, Goal missed, Mixed, Open; blank for method checks) and the Corrected badge in its own column. The campaign's 111-row export with the retired 'correction' outcome is an input, not this table."
        elif path.name in TOP_TABLE_LINKS:
            linked = TOP_TABLE_LINKS[path.name]
            scope = "Supporting numeric table for the linked experiments; each row's batch, horizon, metric, and uncertainty must be retained."
        elif rel.startswith("portfolio/"):
            linked = portfolio_union
            scope = "Portfolio data/provenance catalog. Use panel_id, occurrence_id, and source pointers to locate each experiment's actual subset; this catalog does not make every row evidence for every linked question."
        elif rel.startswith("measurement_and_mechanism/"):
            linked = [e["id"] for e in experiments if 7 <= e["section"] <= 9 and e["id"] != "CVK2"]
            scope = "Measurement and mechanism catalog. Dataset IDs and the panel index identify the relevant subset; this is a catalog-level link."
        else:
            linked = sorted(ids)
        if path.name in ["packet_file_manifest.csv", "file_manifest.csv", "table_index.csv", "source_fingerprints.csv"]:
            scope += " Historical manifest: its recorded hashes identify original archive/packet files, not these sanitized public copies. Public source hash pairs are available in the source catalog."
        contexts[rel] = {"title": path.stem.replace("_", " ").capitalize(), "experimentIds": sorted(set(linked) & ids), "scope": scope}
    # Panel indexes copy notes with their Markdown (`lsm1`, **bold**); the notebook shows them as plain text.
    return {rel: {**context, "title": register_model.clean(context["title"]), "scope": register_model.clean(context["scope"])} for rel, context in contexts.items()}


def publish_tables(workspace, public, experiments, register, figures, areas, run_inventory):
    root = workspace / "outputs/tables"
    contexts = table_contexts(root, experiments)
    tables, receipts = [], []
    regenerated = {REGISTER_CSV: "scripts/register_model.py register", OUTCOMES_BY_AREA_CSV: "scripts/register_model.py register, by area",
                   GOAL_OUTCOMES_CSV: "report goal table with register links and outcome counts", RUN_INVENTORY_CSV: "extended run inventory"}
    for original in sorted(root.rglob("*.csv")):
        rel = original.relative_to(root)
        destination = public / "assets/tables" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if rel.as_posix() == RUN_INVENTORY_CSV:
            original = Path(run_inventory)
        if rel.as_posix() == REGISTER_CSV:
            rows = register_csv_rows(register)
        elif rel.as_posix() == OUTCOMES_BY_AREA_CSV:
            rows = outcomes_by_area_rows(experiments, areas)
        elif rel.as_posix() == GOAL_OUTCOMES_CSV:
            with original.open(newline="", encoding="utf-8") as handle:
                rows = goal_outcomes_rows(list(csv.reader(handle)), figures, experiments)
        else:
            with original.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows([[sanitize_text(cell, workspace) for cell in row] for row in rows])
        require_public(destination.read_text(), "table " + rel.as_posix())
        with destination.open(newline="", encoding="utf-8") as handle:
            restored = list(csv.reader(handle))
        if len(rows) != len(restored) or any(len(a) != len(b) for a, b in zip(rows, restored)):
            raise ValueError(f"CSV table truncated: {rel}")
        # Redaction changes provenance text only; all plain numeric cells are exact strings.
        numeric_cells = 0
        for left, right in zip(rows, restored):
            for a, b in zip(left, right):
                if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", a):
                    numeric_cells += 1
                    if a != b:
                        raise ValueError(f"Numeric value changed during sanitization: {rel}")
        context = contexts[rel.as_posix()]
        tables.append({"id": stable_id("table-", rel.as_posix()), **context, "href": "/assets/tables/" + rel.as_posix(), "rows": len(rows) - 1, "columns": len(rows[0]) if rows else 0})
        receipts.append({"file": rel.as_posix(), **({"regenerated": regenerated[rel.as_posix()]} if rel.as_posix() in regenerated else {}), "rows": len(rows) - 1, "columns": len(rows[0]) if rows else 0, "numeric_cells_preserved": numeric_cells, "original_sha256": digest(original), "public_sha256": digest(destination)})
    return tables, receipts


def archive_log_index(evidence):
    """Resolve recorded run IDs to existing archived files without executing them."""
    repository = Path(evidence["full_portfolio_evidence"]["repository"])
    archive = repository.parent
    found = defaultdict(list)
    for directory in [archive / "runs", archive / "runs_alice2"]:
        if directory.is_dir():
            for path in directory.rglob("*.out"):
                match = re.search(r"[-_](\d{6,})\.out$", path.name)
                if match:
                    found[match.group(1)].append(path)
    # Some historical names have nonstandard prefixes; exact recorded paths fill gaps.
    for batch in evidence["full_mechanism_evidence"]["batches"].values():
        for run in batch["runs"]:
            path = Path(run["raw_source"])
            if path.is_file() and path not in found[str(run["job_id"])]:
                found[str(run["job_id"])].append(path)
    return found


def infer_batch(run_name, candidates):
    for batch in sorted(candidates, key=len, reverse=True):
        if re.fullmatch(r"[A-Za-z0-9_-]+", batch) and re.match(re.escape(batch) + r"(?:[-_]|$)", run_name):
            return batch
    return re.split(r"[-_]", run_name, maxsplit=1)[0]


def publish_raw_log(source, job_id, href, public, workspace, expected_hash=None):
    """Publish an existing log with a verifiable provenance trail and no new data."""
    source = Path(source)
    original = source.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    if expected_hash and original_hash != expected_hash:
        raise ValueError(f"Archived raw log hash changed: {job_id}")
    original_text = original.decode("utf-8")
    text = sanitize_text(original_text, workspace)
    require_public(text, "raw log " + job_id)
    if re.findall(r"\r\n|\r|\n", original_text) != re.findall(r"\r\n|\r|\n", text):
        raise ValueError(f"Raw log line breaks changed: {job_id}")
    original_lines, public_lines = original_text.splitlines(), text.splitlines()
    if len(original_lines) != len(public_lines):
        raise ValueError(f"Raw log line count changed: {job_id}")
    accuracy_lines = 0
    for raw_line, public_line in zip(original_lines, public_lines):
        if re.match(r"^\s*Epoch\s+\d+,", raw_line):
            accuracy_lines += 1
            if raw_line != public_line:
                raise ValueError(f"Epoch accuracy line changed: {job_id}")
    exported = text.encode("utf-8")
    destination = public / href.lstrip("/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(exported)
    return {
        "job_id": job_id, "href": href,
        "source": sanitize_text(str(source), workspace),
        "original_sha256": original_hash, "public_sha256": hashlib.sha256(exported).hexdigest(),
        "original_lines": len(original_lines), "public_lines": len(public_lines),
        "original_bytes": len(original), "public_bytes": len(exported),
        "redacted": original != exported, "line_breaks_preserved": True,
        "epoch_accuracy_lines_preserved": accuracy_lines,
    }


def make_runs(inventory, experiments, evidence, public, workspace, intervened=None):
    intervened = intervened or {}
    by_batch = defaultdict(set)
    for experiment in experiments:
        for batch in experiment["batches"]:
            by_batch[batch].add(experiment["id"])
    primary_batches = {}
    for batch, info in evidence["full_mechanism_evidence"]["batches"].items():
        for run in info["runs"]:
            primary_batches[str(run["job_id"])] = batch
    logs = archive_log_index(evidence)
    cvk_runs = {str(run["job_id"]): run for run in evidence["cvk2_evidence"]["runs"]}
    runs, log_receipts = [], []
    for row in inventory:
        job_id = row["job_id"]
        batch = "cvk1" if job_id in cvk_runs else primary_batches.get(job_id) or infer_batch(row["run"], by_batch)
        linked = sorted(by_batch.get(batch, set()))
        params = {key: value for key, value in row.items() if key not in ["job_id", "account", "network", "dataset", "seed", "epochs_done", "complete", "node"]}
        params["runLabel"] = row["run"]
        params["accuracyMetric"] = "plateau5: mean test accuracy over the final five completed epochs; no final/best fallback"
        params["experimentLinkBasis"] = "Explicit batch family in the experiment register; the run is part of the batch, not necessarily every reported contrast."
        done = row.get("complete") == "1"
        status = "completed" if done else "incomplete-archive"
        if row.get("superseded") == "1":
            status = "superseded"
        params["runCompletion"] = "complete" if done else "incomplete or unknown in archived inventory"
        if job_id in intervened and intervened[job_id]["argsDeviation"]:
            # No patch ran: this run deviates from the standard cell in a base-optimiser CLI flag alone, which no column
            # of the inventory carries, so the row keeps the plain arm's cell key (CORRECTIONS 255, 263).
            listed = intervened[job_id]
            if listed["run"] != row["run"]:
                raise ValueError(f"Exclusion list names another run for job {job_id}")
            kind, flag, value, standard = register_model.args_deviation(listed["intervention"], listed["witness"])
            params["argsDeviation"] = f"{listed['arm']}: {listed['intervention']}"
            params["argsDeviationKind"] = kind
            params["argsDeviationWitness"] = listed["witness"]
            params["argsDeviationNote"] = (f"Not a plain {listed['looks_like']} measurement at the standard cell: the inventory carries that arm's cell key "
                                           f"because no column records the {register_model.args_phrase(kind)}, set to {value} here against the standard "
                                           f"{standard}. Listed in {register_model.INTERVENTIONS_TSV} ({listed['registered_at']}); drop before pooling runs by cell.")
        elif job_id in intervened:
            # The inventory's cell key hides this run's harness intervention (results/CORPUS-EXCLUSIONS.tsv).
            listed = intervened[job_id]
            if listed["run"] != row["run"]:
                raise ValueError(f"Intervention list names another run for job {job_id}")
            params["intervention"] = f"{listed['arm']}: {listed['intervention']}"
            params["interventionWitness"] = listed["witness"]
            # One kind per PATCH=value in the list's intervention cell; cvt6's and cvt8's forced arms carry two, cvt9's EARLY / LATE three
            # (CORRECTIONS 245, 251).
            params["interventionNote"] = (f"Not a plain {listed['looks_like']} measurement: the inventory carries that arm's cell key because no column records the "
                                          f"{register_model.intervention_phrase(listed['intervention'])}. Listed in {register_model.INTERVENTIONS_TSV} ({listed['registered_at']}); drop before pooling runs by cell.")
        if row.get("collapsed") == "1":
            params["accuracyOutcome"] = "collapse flagged in inventory; separate from process completion"
        run = {
            "id": "run-" + job_id, "jobId": job_id, "account": ACCOUNT_LABELS[row["account"]], "batch": batch,
            "seed": row["seed"], "architecture": row["network"], "dataset": row["dataset"],
            "epochs": int(float(row["epochs_done"])) if row["epochs_done"] else 0, "status": status,
            "testAccuracy": float(row["plateau5"]) if row.get("plateau5") and row.get("window_ok") == "1" else None,
            "experimentIds": linked, "parameters": params,
        }
        if job_id in cvk_runs:
            recorded = cvk_runs[job_id]
            source = Path(recorded["raw_path"])
            expected_hash = recorded["sha256"]
        elif logs.get(job_id):
            candidates = sorted(logs[job_id])
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous archived raw log: {job_id}")
            source, expected_hash = candidates[0], None
            params["archivedLogReferences"] = " | ".join(sanitize_text(str(path), workspace) for path in logs[job_id])
        else:
            raise ValueError(f"Required raw log is absent from the available local archive: {job_id}")
        listed = intervened.get(job_id)
        if listed and listed["argsDeviation"]:
            # The witness is the run's OWN ARGS line, read with argparse last-wins semantics; the published witness quotes
            # the checked flag alone, because the raw line also carries the run's private save directory.
            kind, flag, value, standard = register_model.args_deviation(listed["intervention"], listed["witness"])
            log_lines = Path(source).read_text(encoding="utf-8", errors="replace").splitlines()
            params["argsDeviationArgsWitness"] = register_model.args_witness_line(log_lines, flag, value, standard)
        extra_holds = register_model.intervention_kinds(listed["intervention"])[1:] if listed and not listed["argsDeviation"] else []
        if extra_holds:
            # The list carries the first hold's witness only; every further hold must print exactly one matching ON line in the run's own log,
            # joined in the listed order (cvt9's EARLY / LATE: COMP_HOLD | WINDOW_HOLD).
            log_lines = Path(source).read_text(encoding="utf-8", errors="replace").splitlines()
            params["interventionAdditionalWitness"] = " | ".join(register_model.additional_witness(log_lines, patch, value) for patch, value in extra_holds)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["run"]):
            raise ValueError(f"Unsafe raw log filename for job {job_id}")
        href = f"/assets/logs/{row['run']}--{job_id}.txt"
        receipt = publish_raw_log(source, job_id, href, public, workspace, expected_hash)
        run["logHref"] = href
        params["logAvailability"] = "published: sanitized original raw log"
        params["originalLogSha256"] = receipt["original_sha256"]
        params["publicLogSha256"] = receipt["public_sha256"]
        params["logSourceReference"] = receipt["source"]
        params["logScope"] = "Archived stdout as recorded, including any incomplete or superseded run; publication does not certify a scientific result."
        log_receipts.append(receipt)
        runs.append(run)
    return runs, log_receipts


def make_activity(timeline, snapshot_date, experiments, run_count):
    activity = []
    for index, row in enumerate(timeline):
        phase, _, label = row["phase"].partition("\n")
        activity.append({
            "id": f"phase-{index + 1:02d}", "date": PHASE_STARTS[index], "kind": "research-phase",
            "title": label or phase,
            "detail": f"Documented phase: {phase} 2026. Test: {row['test']}. Result: {row['observed_result']}. Next question: {row['next_question']}. Date marks the start of the documented phase window, not individual run launch dates.",
            "experimentIds": PHASE_LINKS[index],
        })
    for phase in register_model.ADDED_PHASES:
        activity.append({
            "id": phase["id"], "date": phase["date"], "kind": "research-phase", "title": phase["title"],
            "detail": f"Documented phase: {phase['period']} 2026. Test: {phase['test']}. Result: {phase['observed_result']}. Next question: {phase['next_question']}. Date marks the start of the documented phase window, not individual run launch dates.",
            "experimentIds": list(phase["experimentIds"]),
        })
    activity.extend([
        {"id": "cvk2-completed-20260914", "date": "2026-09-14", "kind": "completed-experiment", "title": "CVK2 completed and independently checked", "detail": "All 27 registered runs completed. Validity gates and independent verification passed. Observed best cut 19; cuts 16, 19, 22 share the registered peak set. The carrier-cut prediction remains unresolved.", "experimentIds": ["CVK2"]},
        {"id": "publication-snapshot-" + snapshot_date.replace("-", ""), "date": snapshot_date, "kind": "publication-snapshot", "title": "Linked research publication snapshot", "detail": f"Public snapshot assembled from the {len(experiments)}-record register ({sum(e['kind'] == 'research' for e in experiments)} research questions and {sum(e['kind'] == 'method-check' for e in experiments)} method checks, including the {len(PARTITION_IDS)}-row count-matched partition audit from MASTER-TABLE section 10 at {register_model.PARTITION_AUDIT_COMMIT[:7]} and the {len(APPENDED_IDS)} rows appended at MASTER-TABLE lines {register_model.APPENDED_FIRST_ROW}-{register_model.APPENDED_LAST_ROW} at {register_model.APPENDED_COMMIT[:7]}, with the in-place row amendments of CORRECTIONS 229 at {register_model.AMENDMENT_COMMIT[:7]} the {len(LANDED_IDS)} row appended at MASTER-TABLE line {register_model.LANDED_FIRST_ROW} at {register_model.LANDED_COMMIT[:7]}, the {len(CGN3_IDS)} row appended at MASTER-TABLE line {register_model.CGN3_FIRST_ROW} with the in-place row amendments of CORRECTIONS 231 at {register_model.CGN3_COMMIT[:7]}, the {len(CVT23_IDS)} rows appended at MASTER-TABLE lines {register_model.CVT23_FIRST_ROW}-{register_model.CVT23_LAST_ROW} with the in-place row amendments of CORRECTIONS 234 and 236 at {register_model.CVT23_COMMIT[:7]}, the {len(CVT45_IDS)} rows appended at MASTER-TABLE lines {register_model.CVT45_FIRST_ROW}-{register_model.CVT45_LAST_ROW} at {register_model.CVT45_COMMIT[:7]} with the in-place row amendments of CORRECTIONS 244 at {register_model.C244_COMMIT[:7]}, the {len(CVT67_IDS)} rows appended at MASTER-TABLE lines {register_model.CVT67_FIRST_ROW}-{register_model.CVT67_LAST_ROW} at {register_model.CVT67_COMMIT[:7]}, the {len(CVT89_IDS)} rows appended at MASTER-TABLE lines {register_model.CVT89_FIRST_ROW}-{register_model.CVT89_LAST_ROW} at {register_model.CVT89_COMMIT[:7]} with the in-place row amendment of CORRECTIONS 253.15 at {register_model.CVT89_AUDIT_COMMIT[:7]}, and the {len(MUST_IDS)} rows appended at MASTER-TABLE lines {register_model.MUST_FIRST_ROW}-{register_model.MUST_LAST_ROW} at {register_model.MUST_COMMIT[:7]}, which amended no earlier row), the {run_count:,}-run inventory, 54 report pages, and complete numeric tables. This is a publication event, not a new experiment or inferred run date.", "experimentIds": []},
    ])
    return activity


def attach_relationships(experiments, figures, tables, runs, activity):
    by_id = {row["id"]: row for row in experiments}
    for items, field in [(figures, "figureIds"), (tables, "tableIds"), (runs, "runIds"), (activity, "eventIds")]:
        for row in items:
            for eid in row["experimentIds"]:
                if eid not in by_id:
                    raise ValueError(f"Unknown experiment relationship: {eid}")
                by_id[eid][field].append(row["id"])


def attach_snapshot_import(experiments, activity, snapshot_date):
    """Date the public import without fabricating historical training dates."""
    missing = [experiment for experiment in experiments if not experiment["eventIds"]]
    if not missing:
        return
    event_id = "snapshot-import-" + snapshot_date.replace("-", "")
    activity.append({
        "id": event_id,
        "date": snapshot_date,
        "kind": "snapshot-import",
        "title": "Historical records imported into the public snapshot",
        "detail": f"{len(missing)} historical questions and audits have no individually linked dated research phase. This event records their public snapshot import, not training dates or new experimental results. The original register and sources retain each record's scientific scope.",
        "experimentIds": [experiment["id"] for experiment in missing],
    })
    for experiment in missing:
        experiment["eventIds"].append(event_id)


def publish_report_and_figures(workspace, public, figures):
    from PIL import Image
    from pypdf import PdfReader

    original = workspace / "outputs/MetaOptimize_Visual_Summary.pdf"
    reader = PdfReader(original)
    if len(reader.pages) != 54 or reader.attachments:
        raise ValueError("Report PDF must have exactly 54 pages and no embedded files")
    require_public(str(reader.metadata), "report PDF metadata")
    for page_number, page in enumerate(reader.pages, 1):
        require_public(page.extract_text() or "", f"report PDF page {page_number}")
        if page.get("/Annots"):
            require_public(str([item.get_object() for item in page["/Annots"]]), f"report PDF annotations {page_number}")
    destination = public / "assets/report.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, destination)
    receipts = []
    for figure in figures:
        source = workspace / f"outputs/figures/page-{figure['page']}.png"
        target = public / figure["href"].lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            # Lossless conversion removes source metadata and preserves readable glyphs.
            image = image.convert("RGB")
            image.save(target, "WEBP", lossless=True, method=4)
            with Image.open(target) as restored:
                if restored.size != image.size or restored.convert("RGB").tobytes() != image.tobytes():
                    raise ValueError(f"Figure changed in lossless conversion: {figure['id']}")
            receipts.append({"id": figure["id"], "page": figure["page"], "width": image.width, "height": image.height, "original_sha256": digest(source), "public_sha256": digest(target), "bytes": target.stat().st_size})
    return receipts


def publish_bundle(public, tables, figures, log_receipts):
    historical = len(log_receipts) - CVK2_RUNS
    destination = public / "assets/charts-and-tables.zip"
    paths = [(public / "assets/report.pdf", "report.pdf")]
    paths.extend((public / row["href"].lstrip("/"), row["href"].removeprefix("/assets/")) for row in [*tables, *figures])
    paths.extend((public / row["href"].lstrip("/"), row["href"].removeprefix("/assets/")) for row in log_receipts)
    readme = f"""MetaOptimize public research snapshot

Includes all 54 report pages, the report PDF, all 201 numeric CSVs, and all {len(log_receipts):,} sanitized raw logs ({historical:,} historical logs and {CVK2_RUNS} CVK2 logs). Tables keep their original rows, columns, and numeric cell strings. Raw logs retain their original line breaks and epoch accuracy lines, including incomplete and superseded runs. Private account names, local filesystem prefixes, and machine identifiers were replaced; source hashes labeled as original still identify the original private archive, not these sanitized copies. The portal run inventory records original/public SHA-256 pairs for every raw log. Use the public source catalog for original/public code and document hash pairs.

CSV references such as repository/, archive/Account1/, archive/Account2/, and publication-workspace/ are provenance labels, not downloadable URLs. Every run inventory record has a logHref for its sanitized raw-log download. Publishing an archived log does not certify the associated scientific goal, and goal failures remain distinct from process failures.

Accuracy is percent; differences and seed SD/SE are percentage points. Plateau5 is the five registered terminal epochs. Historical last20 values remain labeled. Kernel permutation-null Monte Carlo SE is not training-seed SE. CVK2 completed with clean gates, but its scientific prediction remains unresolved: cuts 16, 19, 22 share the registered peak set.
"""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in paths:
            archive.write(path, name)
        archive.writestr("README.txt", readme)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise ValueError("Public charts/table ZIP failed CRC verification")
    return {"href": "/assets/charts-and-tables.zip", "sha256": digest(destination), "bytes": destination.stat().st_size, "members": len(paths) + 1}


def validate(data, runs, public):
    errors = []
    def verify(value, message):
        if not value:
            errors.append(message)

    expected = EXPECTED_STATS
    verify(data["meta"]["stats"] == expected, "Incorrect snapshot totals")
    verify(len(data["experiments"]) == expected["experiments"], "Experiment count")
    verify(len(runs) == expected["runs"] and len({r["id"] for r in runs}) == expected["runs"], "Run count or duplicate identifiers")
    verify({f["id"] for f in data["figures"]} == {f"page-{p}" for p in range(1, 55)}, "Figure page coverage")
    verify(len(data["tables"]) == 201, "CSV table coverage")
    verify(sum(a["count"] for a in data["areas"]) == expected["experiments"], "Area totals")
    by_id = {e["id"]: e for e in data["experiments"]}
    for eid, (kind, outcome, _) in register_model.CORRECTION_REMAP.items():
        row = by_id.get(eid, {})
        verify(row.get("kind") == kind and row.get("outcome") == outcome and row.get("corrected") is True, f"Approved correction mapping not applied: {eid}")
    phase = next((e for e in data["activity"] if e["id"] == register_model.PHASE_ID), {})
    for eid in PARTITION_IDS:
        row = by_id.get(eid, {})
        verify(row.get("section") == register_model.SECTION and row.get("area") == register_model.AREA, f"Partition-audit record missing: {eid}")
        verify(eid in phase.get("experimentIds", []), f"Partition-audit record not linked to the Partition tests phase: {eid}")
    for phase in register_model.ADDED_PHASES:
        event = next((e for e in data["activity"] if e["id"] == phase["id"]), {})
        verify(event.get("kind") == "research-phase" and event.get("experimentIds") == phase["experimentIds"], f"Added research phase missing: {phase['id']}")
    for line, (rule, section, batches, figures, _, _) in register_model.APPENDED_ROWS.items():
        row = by_id.get(register_model.partition_id(line), {})
        verify(row.get("section") == section and row.get("area") == register_model.AREAS[section] and row.get("outcome") == register_model.RULE_OUTCOME[rule], f"Appended MASTER-TABLE row missing or remapped: line {line}")
        verify(row.get("batches") == batches and set(figures) <= set(row.get("figureIds", [])), f"Appended row links: line {line}")
        verify(any(e["kind"] == "research-phase" and row.get("id") in e["experimentIds"] for e in data["activity"]), f"Appended row has no research phase: line {line}")
    for line, (rule, section, batches, figures, _, _) in register_model.LANDED_ROWS.items():
        eid = register_model.partition_id(line)
        row = by_id.get(eid, {})
        verify(row.get("section") == section and row.get("area") == register_model.AREAS[section] and row.get("outcome") == register_model.RULE_OUTCOME[rule], f"Landed MASTER-TABLE row missing or remapped: line {line}")
        verify(row.get("batches") == batches and set(figures) <= set(row.get("figureIds", [])), f"Landed row links: line {line}")
        verify(any(e["kind"] == "research-phase" and eid in e["experimentIds"] for e in data["activity"]), f"Landed row has no research phase: line {line}")
        linked = [run for run in runs if eid in run["experimentIds"]]
        verify(bool(linked) and {run["batch"] for run in linked} == set(batches), f"Landed row runs not linked: {eid}")
    for line, (rule, section, batches, figures, _, _) in register_model.CGN3_ROWS.items():
        eid = register_model.partition_id(line)
        row = by_id.get(eid, {})
        verify(row.get("section") == section and row.get("area") == register_model.AREAS[section] and row.get("outcome") == register_model.RULE_OUTCOME[rule], f"cgn3 MASTER-TABLE row missing or remapped: line {line}")
        verify(row.get("batches") == batches and set(figures) <= set(row.get("figureIds", [])), f"cgn3 row links: line {line}")
        verify(any(e["kind"] == "research-phase" and eid in e["experimentIds"] for e in data["activity"]), f"cgn3 row has no research phase: line {line}")
        linked = [run for run in runs if eid in run["experimentIds"]]
        verify(bool(linked) and {run["batch"] for run in linked} == set(batches), f"cgn3 row runs not linked: {eid}")
    for line, (rule, section, batches, figures, _, _) in register_model.CVT23_ROWS.items():
        eid = register_model.partition_id(line)
        row = by_id.get(eid, {})
        verify(row.get("section") == section and row.get("area") == register_model.AREAS[section] and row.get("outcome") == register_model.RULE_OUTCOME[rule], f"cvt3/cvt2 MASTER-TABLE row missing or remapped: line {line}")
        verify(row.get("batches") == batches and set(figures) <= set(row.get("figureIds", [])), f"cvt3/cvt2 row links: line {line}")
        verify(any(e["kind"] == "research-phase" and eid in e["experimentIds"] for e in data["activity"]), f"cvt3/cvt2 row has no research phase: line {line}")
        linked = [run for run in runs if eid in run["experimentIds"]]
        verify(bool(linked) and {run["batch"] for run in linked} == set(batches), f"cvt3/cvt2 row runs not linked: {eid}")
    for line, (rule, section, batches, figures, _, _) in register_model.CVT45_ROWS.items():
        eid = register_model.partition_id(line)
        row = by_id.get(eid, {})
        verify(row.get("section") == section and row.get("area") == register_model.AREAS[section] and row.get("outcome") == register_model.RULE_OUTCOME[rule], f"cvt4/cvt5 MASTER-TABLE row missing or remapped: line {line}")
        verify(row.get("batches") == batches and set(figures) <= set(row.get("figureIds", [])), f"cvt4/cvt5 row links: line {line}")
        verify(any(e["kind"] == "research-phase" and eid in e["experimentIds"] for e in data["activity"]), f"cvt4/cvt5 row has no research phase: line {line}")
        linked = [run for run in runs if eid in run["experimentIds"]]
        verify(bool(linked) and {run["batch"] for run in linked} == set(batches), f"cvt4/cvt5 row runs not linked: {eid}")
    for line, (rule, section, batches, figures, _, _) in register_model.CVT67_ROWS.items():
        eid = register_model.partition_id(line)
        row = by_id.get(eid, {})
        verify(row.get("section") == section and row.get("area") == register_model.AREAS[section] and row.get("outcome") == register_model.RULE_OUTCOME[rule], f"cvt6/cvt7 MASTER-TABLE row missing or remapped: line {line}")
        verify(row.get("batches") == batches and set(figures) <= set(row.get("figureIds", [])), f"cvt6/cvt7 row links: line {line}")
        verify(any(e["kind"] == "research-phase" and eid in e["experimentIds"] for e in data["activity"]), f"cvt6/cvt7 row has no research phase: line {line}")
        linked = [run for run in runs if eid in run["experimentIds"]]
        verify(bool(linked) and {run["batch"] for run in linked} == set(batches), f"cvt6/cvt7 row runs not linked: {eid}")
    for line, (rule, section, batches, figures, _, _) in register_model.CVT89_ROWS.items():
        eid = register_model.partition_id(line)
        row = by_id.get(eid, {})
        verify(row.get("section") == section and row.get("area") == register_model.AREAS[section] and row.get("outcome") == register_model.RULE_OUTCOME[rule], f"cvt8/cvt9 MASTER-TABLE row missing or remapped: line {line}")
        verify(row.get("batches") == batches and set(figures) <= set(row.get("figureIds", [])), f"cvt8/cvt9 row links: line {line}")
        verify(any(e["kind"] == "research-phase" and eid in e["experimentIds"] for e in data["activity"]), f"cvt8/cvt9 row has no research phase: line {line}")
        linked = [run for run in runs if eid in run["experimentIds"]]
        verify(bool(linked) and {run["batch"] for run in linked} == set(batches), f"cvt8/cvt9 row runs not linked: {eid}")
    for line, (rule, section, batches, figures, _, _) in register_model.MUST_ROWS.items():
        eid = register_model.partition_id(line)
        row = by_id.get(eid, {})
        verify(row.get("section") == section and row.get("area") == register_model.AREAS[section] and row.get("outcome") == register_model.RULE_OUTCOME[rule], f"MUST-tier MASTER-TABLE row missing or remapped: line {line}")
        verify(row.get("batches") == batches and set(figures) <= set(row.get("figureIds", [])), f"MUST-tier row links: line {line}")
        verify(any(e["kind"] == "research-phase" and eid in e["experimentIds"] for e in data["activity"]), f"MUST-tier row has no research phase: line {line}")
        verify(not row.get("corrected"), f"MUST-tier row must carry no Corrected badge: line {line}")
        linked = [run for run in runs if eid in run["experimentIds"]]
        verify(bool(linked) and {run["batch"] for run in linked} == set(batches), f"MUST-tier row runs not linked: {eid}")
    verify(not any(w["id"].startswith(tuple(f"warning-{eid}-registration" for eid in MUST_IDS)) for w in data["warnings"]), "The MUST-tier landings corrected no registration text")
    verify(not any(w["id"].startswith(tuple(f"warning-{eid}-amendment" for eid in MUST_IDS)) for w in data["warnings"]), "The MUST-tier landings amended no row")
    for eid, spec in {**register_model.CVT67_REGISTRATION_CORRECTIONS, **register_model.CVT89_REGISTRATION_CORRECTIONS}.items():
        warning = next((w for w in data["warnings"] if w["id"] == f"warning-{eid}-registration-{spec['number']}"), {})
        verify(warning.get("experimentId") == eid and warning.get("id") in by_id.get(eid, {}).get("warningIds", []) and f"CORRECTIONS {spec['entry']}" in warning.get("detail", ""), f"Registration correction warning missing: {eid}")
    for run in runs:
        if run["parameters"].get("intervention"):
            holds = register_model.intervention_kinds(run["parameters"]["intervention"].split(": ", 1)[1])
            witnesses = run["parameters"].get("interventionAdditionalWitness", "")
            # Every hold after the first is witnessed by its own ON line, in the listed order; a one-hold run carries none.
            verify([w.split(": on ", 1)[0] for w in witnesses.split(" | ")] == [patch for patch, _ in holds[1:]] if len(holds) > 1 else not witnesses, f"Further hold not witnessed: {run['id']}")
    for amendments, commit in [(register_model.CVT23_AMENDMENTS, register_model.CVT23_COMMIT), (register_model.C244_AMENDMENTS, register_model.C244_COMMIT)]:
        for eid, spec in amendments.items():
            row = by_id.get(eid, {})
            warning = next((w for w in data["warnings"] if w["id"] == f"warning-{eid}-amendment-{spec['number']}"), {})
            verify(row.get("outcome") == spec["outcome"], f"CORRECTIONS {spec['number']} amendment moved an outcome: {eid}")
            verify(warning.get("experimentId") == eid and f"line {spec['line']} at {commit[:7]}" in warning.get("detail", "") and warning.get("id") in row.get("warningIds", []), f"CORRECTIONS {spec['number']} amendment warning missing: {eid}")
    for eid, spec in register_model.CVT89_AMENDMENTS.items():
        # CORRECTIONS 253.15 (the cvt8 / cvt9 final audit): the corrected cell is in the scope, its bracket keeps the old words, no outcome moves.
        row = by_id.get(eid, {})
        warning = next((w for w in data["warnings"] if w["id"] == f"warning-{eid}-amendment-{spec['number']}"), {})
        verify(row.get("outcome") == spec["outcome"] and not row.get("corrected"), f"CORRECTIONS {spec['entry']} amendment moved an outcome or badge: {eid}")
        verify(warning.get("experimentId") == eid and f"line {spec['line']} at {register_model.CVT89_AUDIT_COMMIT[:7]}; CORRECTIONS {spec['entry']}" in warning.get("detail", "") and warning.get("id") in row.get("warningIds", []), f"CORRECTIONS {spec['entry']} amendment warning missing: {eid}")
        verify(f"{spec['amended']} [CORRECTED IN PLACE" in row.get("scope", "") and f"SUPERSEDED wording, kept verbatim: '{spec['superseded']}']" in row.get("scope", ""), f"CORRECTIONS {spec['entry']} correction not in the scope: {eid}")
    for eid, spec in register_model.CGN3_AMENDMENTS.items():
        row = by_id.get(eid, {})
        warning = next((w for w in data["warnings"] if w["id"] == f"warning-{eid}-amendment-231"), {})
        verify(row.get("outcome") == spec["outcome"], f"CORRECTIONS 231 amendment moved an outcome: {eid}")
        verify(warning.get("experimentId") == eid and f"line {spec['line']} at {register_model.CGN3_COMMIT[:7]}" in warning.get("detail", "") and warning.get("id") in row.get("warningIds", []), f"CORRECTIONS 231 amendment warning missing: {eid}")
        if spec["retire"]:
            retired = spec["retire"][0]
            verify(all(retired not in w["detail"] for w in data["warnings"] if w["experimentId"] == eid) and retired not in row.get("scope", ""), f"Retired wording still published: {eid}")
    for eid, spec in {**register_model.LANDED_INTERVENTIONS, **register_model.CVT23_INTERVENTIONS, **register_model.CVT45_INTERVENTIONS, **register_model.CVT67_INTERVENTIONS, **register_model.CVT89_INTERVENTIONS}.items():
        warning = next((w for w in data["warnings"] if w["id"] == f"warning-{eid}-intervention"), {})
        verify(warning.get("experimentId") == eid and warning.get("id") in by_id.get(eid, {}).get("warningIds", []), f"Intervention warning missing: {eid}")
        marked = [run for run in runs if run["batch"] == spec["batch"] and "intervention" in run["parameters"]]
        verify(sorted(run["parameters"]["intervention"].split(":")[0] for run in marked) == sorted(arm for arm, n in spec["arms"].items() for _ in range(n)), f"Intervened runs not marked: {eid}")
    for eid, spec in register_model.MUST_ARGS_DEVIATIONS.items():
        # The ARGS-value rows carry their own mark, so the patch-intervention count above is unchanged by this landing.
        warning = next((w for w in data["warnings"] if w["id"] == f"warning-{eid}-args-deviation"), {})
        verify(warning.get("experimentId") == eid and warning.get("id") in by_id.get(eid, {}).get("warningIds", []), f"ARGS-value deviation warning missing: {eid}")
        marked = [run for run in runs if run["batch"] == spec["batch"] and "argsDeviation" in run["parameters"]]
        verify(sorted(run["parameters"]["argsDeviation"].split(":")[0] for run in marked) == sorted(arm for arm, n in spec["arms"].items() for _ in range(n)), f"ARGS-value rows not marked: {eid}")
        for run in marked:
            kind = run["parameters"]["argsDeviationKind"]
            flag, standard, _name = register_model.ARGS_KINDS[kind]
            verify(run["parameters"]["argsDeviationWitness"].startswith(f"{kind}: {flag}="), f"ARGS-value witness malformed: {run['id']}")
            verify(run["parameters"]["argsDeviationArgsWitness"].startswith(f"{register_model.ARGS_LINE_PREFIX} --{flag} "), f"ARGS line witness missing: {run['id']}")
            verify("intervention" not in run["parameters"], f"An ARGS-value row must not be marked as a patch intervention: {run['id']}")
    for eid, (line, previous, outcome, _, batches, _, _) in register_model.ROW_AMENDMENTS.items():
        row = by_id.get(eid, {})
        warning = next((w for w in data["warnings"] if w["id"] == f"warning-{eid}-amendment"), {})
        verify(row.get("outcome") == outcome and set(batches) <= set(row.get("batches", [])), f"Row amendment not applied: {eid}")
        verify(warning.get("experimentId") == eid and f"line {line} at {register_model.AMENDMENT_COMMIT[:7]}" in warning.get("detail", ""), f"Row amendment warning missing: {eid}")
        verify(outcome == previous or register_model.OUTCOME_LABELS[previous] in warning.get("detail", ""), f"Row amendment must keep the previous outcome: {eid}")
        for batch in batches:
            verify(any(run["batch"] == batch and eid in run["experimentIds"] for run in runs), f"Amendment batch has no linked runs: {eid}/{batch}")
    for line, (_, _, batches, _, _, _) in register_model.APPENDED_ROWS.items():
        eid = register_model.partition_id(line)
        linked = [run for run in runs if eid in run["experimentIds"]]
        verify(bool(linked) and {run["batch"] for run in linked} == set(batches), f"Appended row runs not linked: {eid}")
    for name, columns in [(OUTCOMES_BY_AREA_CSV, OUTCOMES_BY_AREA_COLUMNS), (GOAL_OUTCOMES_CSV, GOAL_OUTCOMES_COLUMNS)]:
        table = next((t for t in data["tables"] if t["href"] == "/assets/tables/" + name), None)
        verify(table is not None and table["columns"] == len(columns), f"Regenerated outcome table missing: {name}")
    register_table = next((t for t in data["tables"] if t["href"] == "/assets/tables/" + REGISTER_CSV), None)
    verify(register_table is not None and register_table["rows"] == expected["experiments"] and register_table["columns"] == len(REGISTER_COLUMNS), "Published register table must list every record")
    catalogs = {"figureIds": {r["id"]: r for r in data["figures"]}, "tableIds": {r["id"]: r for r in data["tables"]}, "codeIds": {r["id"]: r for r in data["sources"]}, "runIds": {r["id"]: r for r in runs}, "warningIds": {r["id"]: r for r in data["warnings"]}, "eventIds": {r["id"]: r for r in data["activity"]}}
    for experiment in data["experiments"]:
        eid = experiment["id"]
        verify(experiment["kind"] in KINDS, f"Unknown record kind: {eid}")
        if experiment["kind"] == "research":
            verify(experiment["outcome"] in OUTCOMES, f"Research question needs one of the four outcomes: {eid}")
        else:
            verify(experiment["outcome"] is None, f"Method checks carry no research outcome: {eid}")
        verify(isinstance(experiment["corrected"], bool) and bool(experiment["corrected"]) == bool(experiment["correction"] and experiment["correction"].get("note")), f"Corrected badge needs its correction history: {eid}")
        verify(bool(experiment["figureIds"] and experiment["tableIds"] and experiment["sourceRefs"] and experiment["eventIds"]), f"Missing experiment evidence/provenance links: {eid}")
        for field, catalog in catalogs.items():
            for ident in experiment[field]:
                verify(ident in catalog, f"Broken {field}: {eid}/{ident}")
                if ident in catalog and field in ["figureIds", "tableIds", "runIds", "eventIds"]:
                    verify(eid in catalog[ident]["experimentIds"], f"Missing reverse relationship: {eid}/{ident}")
        for ref in experiment["sourceRefs"]:
            source = catalogs["codeIds"].get(ref["sourceId"])
            verify(source is not None, f"Missing source reference: {eid}")
            if source and "line" in ref:
                verify(1 <= ref["line"] <= source["lines"], f"Source line out of range: {eid}/{ref['sourceId']}/{ref['line']}")
    for row in [*data["figures"], *data["tables"], *data["sources"]]:
        verify((public / row["href"].lstrip("/")).is_file(), f"Missing public asset: {row['id']}")
    for run in runs:
        verify(run["account"] in {"Account1", "Account2"}, "Private account label")
        verify(run["parameters"].get("logAvailability") == "published: sanitized original raw log", f"Missing log availability: {run['id']}")
        verify(bool(run.get("logHref")), f"Missing raw-log link: {run['id']}")
        for field in ["originalLogSha256", "publicLogSha256"]:
            verify(bool(re.fullmatch(r"[0-9a-f]{64}", run["parameters"].get(field, ""))), f"Missing {field}: {run['id']}")
        if run.get("logHref"):
            raw_path = public / run["logHref"].lstrip("/")
            verify(raw_path.is_file(), f"Fake raw-log link: {run['id']}")
            if raw_path.is_file():
                verify(digest(raw_path) == run["parameters"].get("publicLogSha256"), f"Public raw-log hash mismatch: {run['id']}")
    cvk = next(row for row in data["experiments"] if row["id"] == "CVK2")
    verify(cvk["outcome"] == "unresolved" and len(cvk["runIds"]) == CVK2_RUNS, "CVK2 scope/verdict")
    verify(data["latest"]["peakSet"] == [16, 19, 22] and data["latest"]["bestCut"] == 19 and data["latest"]["predictedCut"] == 22, "CVK2 registered peak set")
    verify(len(data["latest"]["cells"]) == 9 and all(c["n"] == 3 for c in data["latest"]["cells"]), "CVK2 sample counts")
    require_public(json.dumps(data, ensure_ascii=False), "research.json")
    require_public(json.dumps(runs, ensure_ascii=False), "runs.json")
    if errors:
        raise ValueError("Public evidence validation failed:\n" + "\n".join(errors))
    return {"status": "PASS", "coverage": expected, "tables": len(data["tables"]), "sources": len(data["sources"]), "warnings": len(data["warnings"]), "activity_events": len(data["activity"]), "published_raw_logs": sum(bool(r.get("logHref")) for r in runs), "log_availability": dict(Counter(r["parameters"]["logAvailability"] for r in runs)), "experiment_links": {field: sum(len(e[field]) for e in data["experiments"]) for field in catalogs}}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def export(workspace, public, snapshot_date, audit_path=None, run_inventory=None):
    workspace, public = Path(workspace).resolve(), Path(public).resolve()
    sources, source_links = load_source_catalog(public)  # Fail before publishing partial data.
    evidence = read_evidence(workspace)
    table_root = workspace / "outputs/tables"
    repository_path = Path(evidence["full_portfolio_evidence"]["repository"])
    register = register_model.load_register(workspace, repository_path)
    # The run inventory can be an extended copy outside the (read-only) campaign workspace.
    run_inventory = Path(run_inventory).resolve() if run_inventory else table_root / RUN_INVENTORY_CSV
    inventory = read_csv(run_inventory)
    figures = make_figures(read_csv(table_root / "complete_figure_index.csv"))
    for figure in figures:
        if figure["id"] in register_model.FIGURE_IDS:
            figure["experimentIds"] = list(dict.fromkeys([*figure["experimentIds"], *PARTITION_IDS]))
        appended = [row["id"] for row in register if figure["id"] in json.loads(row.get("figure_ids") or "[]")]
        if appended:
            figure["experimentIds"] = list(dict.fromkeys([*figure["experimentIds"], *appended]))
    experiments, warnings = build_experiments(register, source_links)
    cvk = evidence["cvk2_evidence"]
    attach_latest_warnings(experiments, warnings, cvk)
    areas = [{"id": section, "label": next(e["area"] for e in experiments if e["section"] == section), "count": sum(e["section"] == section for e in experiments)} for section in sorted({e["section"] for e in experiments})]
    tables, table_receipts = publish_tables(workspace, public, experiments, register, figures, areas, run_inventory)
    intervened = register_model.intervened_runs(repository_path)
    runs, log_receipts = make_runs(inventory, experiments, evidence, public, workspace, intervened)
    missing = set(intervened) - {run["jobId"] for run in runs if "intervention" in run["parameters"] or "argsDeviation" in run["parameters"]}
    if missing:
        raise ValueError(f"Intervened runs absent from the run inventory: {sorted(missing)}")
    activity = make_activity(read_csv(table_root / "research_timeline.csv"), snapshot_date, experiments, len(runs))
    attach_relationships(experiments, figures, tables, runs, activity)
    attach_snapshot_import(experiments, activity, snapshot_date)
    repository = Path(evidence["full_portfolio_evidence"]["repository"])
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    input_paths = [table_root / REGISTER_CSV, run_inventory, table_root / "complete_figure_index.csv"]
    # The pinned MASTER-TABLE bytes and the mapping module are inputs too.
    fingerprint = hashlib.sha256(("".join(digest(path) for path in input_paths) + register_model.MASTER_TABLE_SHA256 + register_model.APPENDED_MASTER_TABLE_SHA256 + register_model.AMENDMENT_MASTER_TABLE_SHA256 + register_model.LANDED_MASTER_TABLE_SHA256 + register_model.INTERVENTIONS_TSV_SHA256 + register_model.CGN3_MASTER_TABLE_SHA256 + register_model.CVT23_MASTER_TABLE_SHA256 + register_model.CVT23_INTERVENTIONS_TSV_SHA256 + register_model.CVT45_MASTER_TABLE_SHA256 + register_model.CVT45_INTERVENTIONS_TSV_SHA256 + register_model.C244_MASTER_TABLE_SHA256 + register_model.CVT67_MASTER_TABLE_SHA256 + register_model.CVT67_INTERVENTIONS_TSV_SHA256 + register_model.CVT67_CORRECTIONS_SHA256 + register_model.CVT89_MASTER_TABLE_SHA256 + register_model.CVT89_INTERVENTIONS_TSV_SHA256 + register_model.CVT89_CORRECTIONS_SHA256 + register_model.CVT89_AUDIT_MASTER_TABLE_SHA256 + register_model.CVT89_AUDIT_CORRECTIONS_SHA256 + register_model.MUST_MASTER_TABLE_SHA256 + register_model.MUST_CORRECTIONS_SHA256 + register_model.MUST_INTERVENTIONS_TSV_SHA256 + digest(Path(register_model.__file__))).encode()).hexdigest()[:16]
    data = {
        "meta": {"title": "MetaOptimize Research Notebook", "asOf": snapshot_date, "generatedAt": datetime.now(timezone.utc).isoformat(), "snapshotId": snapshot_date + "-" + fingerprint, "repositoryCommit": commit, "stats": {"experiments": len(experiments), "researchQuestions": sum(e["kind"] == "research" for e in experiments), "methodChecks": sum(e["kind"] == "method-check" for e in experiments), "runs": len(runs), "figures": len(figures), "areas": len(areas)}, "downloads": {"pdf": "/assets/report.pdf", "bundle": "/assets/charts-and-tables.zip"},
                 "outcomeModel": {"outcomes": register_model.OUTCOME_LABELS, "badge": "corrected", "kinds": {"research": "Research question", "method-check": "Method check"}, "mapping": "scripts/register_model.py", "partitionAuditCommit": register_model.PARTITION_AUDIT_COMMIT, "masterTableSha256": register_model.MASTER_TABLE_SHA256, "appendedRowsCommit": register_model.APPENDED_COMMIT, "appendedMasterTableSha256": register_model.APPENDED_MASTER_TABLE_SHA256, "amendmentCommit": register_model.AMENDMENT_COMMIT, "amendmentMasterTableSha256": register_model.AMENDMENT_MASTER_TABLE_SHA256, "landedRowsCommit": register_model.LANDED_COMMIT, "landedMasterTableSha256": register_model.LANDED_MASTER_TABLE_SHA256, "cgn3LandingCommit": register_model.CGN3_COMMIT, "cgn3LandingMasterTableSha256": register_model.CGN3_MASTER_TABLE_SHA256, "cvt23LandingCommit": register_model.CVT23_COMMIT, "cvt23LandingMasterTableSha256": register_model.CVT23_MASTER_TABLE_SHA256, "cvt45LandingCommit": register_model.CVT45_COMMIT, "cvt45LandingMasterTableSha256": register_model.CVT45_MASTER_TABLE_SHA256, "c244AmendmentCommit": register_model.C244_COMMIT, "c244AmendmentMasterTableSha256": register_model.C244_MASTER_TABLE_SHA256, "cvt67LandingCommit": register_model.CVT67_COMMIT, "cvt67LandingMasterTableSha256": register_model.CVT67_MASTER_TABLE_SHA256, "cvt89LandingCommit": register_model.CVT89_COMMIT, "cvt89LandingMasterTableSha256": register_model.CVT89_MASTER_TABLE_SHA256, "cvt89AuditCommit": register_model.CVT89_AUDIT_COMMIT, "cvt89AuditMasterTableSha256": register_model.CVT89_AUDIT_MASTER_TABLE_SHA256, "mustTierLandingCommit": register_model.MUST_COMMIT, "mustTierMasterTableSha256": register_model.MUST_MASTER_TABLE_SHA256, "mustTierIngestCommit": register_model.MUST_INGEST_COMMIT}},
        "areas": areas, "experiments": experiments, "figures": figures, "sources": sources,
        "tables": tables, "warnings": warnings, "activity": activity,
        "latest": {"experimentId": "CVK2", "peakSet": cvk["peak_cuts"], "predictedCut": cvk["predicted_peak_cut"], "bestCut": cvk["best_mean_cut"], "bar": cvk["peak_bar_pp"], "cells": [{key: row[key] for key in ["arm", "cut", "mean", "sem", "n"]} for row in cvk["cells"]]},
    }
    data, runs = sanitize(data, workspace), sanitize(runs, workspace)
    figure_receipts = publish_report_and_figures(workspace, public, figures)
    bundle_receipt = publish_bundle(public, tables, figures, log_receipts)
    audit = validate(data, runs, public)
    for path in (public / "assets").rglob("*"):
        if path.suffix in [".txt", ".csv", ".json"]:
            require_public(path.read_text(), "public asset " + path.relative_to(public).as_posix())
    audit.update({
        "generated_at": data["meta"]["generatedAt"], "snapshot_id": data["meta"]["snapshotId"],
        "inputs": [{"file": str(path.relative_to(workspace)) if path.is_relative_to(workspace) else sanitize_text(str(path), workspace), "sha256": digest(path)} for path in input_paths],
        "numeric_cells_preserved": sum(row["numeric_cells_preserved"] for row in table_receipts),
        "table_receipts": table_receipts, "figure_receipts": figure_receipts, "raw_log_receipts": log_receipts,
        "raw_logs": {"original_bytes": sum(row["original_bytes"] for row in log_receipts), "public_bytes": sum(row["public_bytes"] for row in log_receipts), "redacted_files": sum(row["redacted"] for row in log_receipts), "line_breaks_preserved": len(log_receipts), "epoch_accuracy_lines_preserved": sum(row["epoch_accuracy_lines_preserved"] for row in log_receipts)},
        "bundle": bundle_receipt, "privacy": {"text_scan": "PASS", "pdf_text_metadata_annotations": "PASS", "embedded_pdf_files": 0, "lossless_figure_pixel_checks": 54, "account_labels": ["Account1", "Account2"]},
        "scope": "No raw data or scientific statistics were changed. Dates identify documented research phases and actual publication/completion events, not inferred per-run dates.",
    })
    atomic_json(public / "data/research.json", data)
    atomic_json(public / "data/runs.json", runs)
    atomic_json(Path(audit_path) if audit_path else workspace / "work/portal_data_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--public-dir", type=Path, default=PORTAL / "public")
    parser.add_argument("--as-of", default="2026-09-17", help="Documented publication-snapshot date, YYYY-MM-DD")
    parser.add_argument("--check", action="store_true", help="Validate existing public JSON and links without regenerating assets")
    parser.add_argument("--audit", type=Path, help="Where to write the export audit receipt (default: <workspace>/work/portal_data_audit.json)")
    parser.add_argument("--run-inventory", type=Path, help="Run inventory CSV to publish instead of <workspace>/outputs/tables/complete_run_inventory.csv (an extended copy kept outside a read-only workspace)")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.as_of):
        parser.error("--as-of must be YYYY-MM-DD")
    if args.check:
        load_source_catalog(args.public_dir)
        data = json.loads((args.public_dir / "data/research.json").read_text())
        runs = json.loads((args.public_dir / "data/runs.json").read_text())
        audit = validate(data, runs, args.public_dir)
    else:
        if any(importlib.util.find_spec(name) is None for name in ["PIL", "pypdf"]):
            bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
            if bundled.is_file() and Path(sys.executable).resolve() != bundled.resolve():
                os.execv(str(bundled), [str(bundled), str(Path(__file__).resolve()), *sys.argv[1:]])
            raise RuntimeError("Pillow and pypdf are required to verify lossless public figure assets and PDF privacy")
        audit = export(args.workspace, args.public_dir, args.as_of, args.audit, args.run_inventory)
    print(json.dumps({key: audit[key] for key in ["status", "coverage", "tables", "sources", "warnings", "activity_events", "published_raw_logs"]}, indent=2))


if __name__ == "__main__":
    main()
