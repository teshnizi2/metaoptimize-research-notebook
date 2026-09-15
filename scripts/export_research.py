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
OUTCOMES = {"success", "fail", "mixed", "unresolved", "correction"}
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
    ["MT098"], ["MT132", "MT133", "MT134"],
    ["MT141", "MT142", "MT145", "MT146", "MT147", "MT149", "MT157"],
    ["MT162", "MT163", "MT165"], ["MT019", "MT020", "MT164", "MT166"], ["CVK2"],
]
PHASE_STARTS = ["2026-08-18", "2026-08-20", "2026-08-24", "2026-09-03", "2026-09-04", "2026-09-08", "2026-09-09", "2026-09-14"]


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
    outcome = experiment["outcome"]
    severity, title, status = {
        "success": ("limitation", "Scope of the supported result", "documented"),
        "fail": ("caution", "The tested goal was not supported", "documented"),
        "mixed": ("limitation", "The result depends on scope", "documented"),
        "unresolved": ("open", "The question remains unresolved", "open"),
        "correction": ("correction", "A historical claim was corrected", "resolved"),
    }[outcome]
    detail = " ".join(dict.fromkeys([experiment["result"], experiment["reason"], experiment["scope"]]))
    if outcome == "fail":
        detail += " This is a scientific verdict about the tested goal, not a runtime-failure classification."
    return {"id": f"warning-{experiment['id']}-verdict", "experimentId": experiment["id"], "severity": severity, "title": title, "detail": detail, "status": status}


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
            **{key: row[key] for key in ["goal", "comparison", "why", "result", "outcome", "reason", "scope"]},
            "batches": json.loads(row["batches"]), "figureIds": [], "codeIds": linkage["codeIds"],
            "tableIds": [], "warningIds": [], "eventIds": [], "runIds": [], "sourceRefs": linkage["sourceRefs"],
        }
        warning = warning_for(experiment)
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
        elif path.name in TOP_TABLE_LINKS:
            linked = TOP_TABLE_LINKS[path.name]
            scope = "Supporting numeric table for the linked experiments; each row's batch, horizon, metric, and uncertainty must be retained."
        elif rel.startswith("portfolio/"):
            linked = portfolio_union
            scope = "Portfolio data/provenance catalog. Use panel_id, occurrence_id, and source pointers to locate each experiment's actual subset; this catalog does not make every row evidence for every linked question."
        elif rel.startswith("measurement_and_mechanism/"):
            linked = [e["id"] for e in experiments if e["section"] >= 7 and e["id"] != "CVK2"]
            scope = "Measurement and mechanism catalog. Dataset IDs and the panel index identify the relevant subset; this is a catalog-level link."
        else:
            linked = sorted(ids)
        if path.name in ["packet_file_manifest.csv", "file_manifest.csv", "table_index.csv", "source_fingerprints.csv"]:
            scope += " Historical manifest: its recorded hashes identify original archive/packet files, not these sanitized public copies. Public source hash pairs are available in the source catalog."
        contexts[rel] = {"title": path.stem.replace("_", " ").capitalize(), "experimentIds": sorted(set(linked) & ids), "scope": scope}
    return contexts


def publish_tables(workspace, public, experiments):
    root = workspace / "outputs/tables"
    contexts = table_contexts(root, experiments)
    tables, receipts = [], []
    for original in sorted(root.rglob("*.csv")):
        rel = original.relative_to(root)
        destination = public / "assets/tables" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
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
        receipts.append({"file": rel.as_posix(), "rows": len(rows) - 1, "columns": len(rows[0]) if rows else 0, "numeric_cells_preserved": numeric_cells, "original_sha256": digest(original), "public_sha256": digest(destination)})
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


def make_runs(inventory, experiments, evidence, public, workspace):
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


def make_activity(timeline, snapshot_date):
    activity = []
    for index, row in enumerate(timeline):
        phase, _, label = row["phase"].partition("\n")
        activity.append({
            "id": f"phase-{index + 1:02d}", "date": PHASE_STARTS[index], "kind": "research-phase",
            "title": label or phase,
            "detail": f"Documented phase: {phase} 2026. Test: {row['test']}. Result: {row['observed_result']}. Next question: {row['next_question']}. Date marks the start of the documented phase window, not individual run launch dates.",
            "experimentIds": PHASE_LINKS[index],
        })
    activity.extend([
        {"id": "cvk2-completed-20260914", "date": "2026-09-14", "kind": "completed-experiment", "title": "CVK2 completed and independently checked", "detail": "All 27 registered runs completed. Validity gates and independent verification passed. Observed best cut 19; cuts 16, 19, 22 share the registered peak set. The carrier-cut prediction remains unresolved.", "experimentIds": ["CVK2"]},
        {"id": "publication-snapshot-" + snapshot_date.replace("-", ""), "date": snapshot_date, "kind": "publication-snapshot", "title": "Linked research publication snapshot", "detail": "Public snapshot assembled from the 111-question register, 2,863-run inventory, 54 report pages, and complete numeric tables. This is a publication event, not a new experiment or inferred run date.", "experimentIds": []},
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
    destination = public / "assets/charts-and-tables.zip"
    paths = [(public / "assets/report.pdf", "report.pdf")]
    paths.extend((public / row["href"].lstrip("/"), row["href"].removeprefix("/assets/")) for row in [*tables, *figures])
    paths.extend((public / row["href"].lstrip("/"), row["href"].removeprefix("/assets/")) for row in log_receipts)
    readme = """MetaOptimize public research snapshot

Includes all 54 report pages, the report PDF, all 201 numeric CSVs, and all 2,863 sanitized raw logs (2,836 historical logs and 27 CVK2 logs). Tables keep their original rows, columns, and numeric cell strings. Raw logs retain their original line breaks and epoch accuracy lines, including incomplete and superseded runs. Private account names, local filesystem prefixes, and machine identifiers were replaced; source hashes labeled as original still identify the original private archive, not these sanitized copies. The portal run inventory records original/public SHA-256 pairs for every raw log. Use the public source catalog for original/public code and document hash pairs.

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

    expected = {"experiments": 111, "runs": 2863, "figures": 54, "areas": 9}
    verify(data["meta"]["stats"] == expected, "Incorrect snapshot totals")
    verify(len(data["experiments"]) == 111, "Experiment count")
    verify(len(runs) == 2863 and len({r["id"] for r in runs}) == 2863, "Run count or duplicate identifiers")
    verify({f["id"] for f in data["figures"]} == {f"page-{p}" for p in range(1, 55)}, "Figure page coverage")
    verify(len(data["tables"]) == 201, "CSV table coverage")
    verify(sum(a["count"] for a in data["areas"]) == 111, "Area totals")
    catalogs = {"figureIds": {r["id"]: r for r in data["figures"]}, "tableIds": {r["id"]: r for r in data["tables"]}, "codeIds": {r["id"]: r for r in data["sources"]}, "runIds": {r["id"]: r for r in runs}, "warningIds": {r["id"]: r for r in data["warnings"]}, "eventIds": {r["id"]: r for r in data["activity"]}}
    for experiment in data["experiments"]:
        eid = experiment["id"]
        verify(experiment["outcome"] in OUTCOMES, f"Unknown outcome: {eid}")
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
    verify(cvk["outcome"] == "unresolved" and len(cvk["runIds"]) == 27, "CVK2 scope/verdict")
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


def export(workspace, public, snapshot_date):
    workspace, public = Path(workspace).resolve(), Path(public).resolve()
    sources, source_links = load_source_catalog(public)  # Fail before publishing partial data.
    evidence = read_evidence(workspace)
    table_root = workspace / "outputs/tables"
    register = read_csv(table_root / "complete_experiment_register.csv")
    inventory = read_csv(table_root / "complete_run_inventory.csv")
    figures = make_figures(read_csv(table_root / "complete_figure_index.csv"))
    experiments, warnings = build_experiments(register, source_links)
    cvk = evidence["cvk2_evidence"]
    attach_latest_warnings(experiments, warnings, cvk)
    tables, table_receipts = publish_tables(workspace, public, experiments)
    runs, log_receipts = make_runs(inventory, experiments, evidence, public, workspace)
    activity = make_activity(read_csv(table_root / "research_timeline.csv"), snapshot_date)
    attach_relationships(experiments, figures, tables, runs, activity)
    attach_snapshot_import(experiments, activity, snapshot_date)
    areas = [{"id": section, "label": next(e["area"] for e in experiments if e["section"] == section), "count": sum(e["section"] == section for e in experiments)} for section in sorted({e["section"] for e in experiments})]
    repository = Path(evidence["full_portfolio_evidence"]["repository"])
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    input_paths = [table_root / name for name in ["complete_experiment_register.csv", "complete_run_inventory.csv", "complete_figure_index.csv"]]
    fingerprint = hashlib.sha256("".join(digest(path) for path in input_paths).encode()).hexdigest()[:16]
    data = {
        "meta": {"title": "MetaOptimize Research Notebook", "asOf": snapshot_date, "generatedAt": datetime.now(timezone.utc).isoformat(), "snapshotId": snapshot_date + "-" + fingerprint, "repositoryCommit": commit, "stats": {"experiments": len(experiments), "runs": len(runs), "figures": len(figures), "areas": len(areas)}, "downloads": {"pdf": "/assets/report.pdf", "bundle": "/assets/charts-and-tables.zip"}},
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
        "inputs": [{"file": str(path.relative_to(workspace)), "sha256": digest(path)} for path in input_paths],
        "numeric_cells_preserved": sum(row["numeric_cells_preserved"] for row in table_receipts),
        "table_receipts": table_receipts, "figure_receipts": figure_receipts, "raw_log_receipts": log_receipts,
        "raw_logs": {"original_bytes": sum(row["original_bytes"] for row in log_receipts), "public_bytes": sum(row["public_bytes"] for row in log_receipts), "redacted_files": sum(row["redacted"] for row in log_receipts), "line_breaks_preserved": len(log_receipts), "epoch_accuracy_lines_preserved": sum(row["epoch_accuracy_lines_preserved"] for row in log_receipts)},
        "bundle": bundle_receipt, "privacy": {"text_scan": "PASS", "pdf_text_metadata_annotations": "PASS", "embedded_pdf_files": 0, "lossless_figure_pixel_checks": 54, "account_labels": ["Account1", "Account2"]},
        "scope": "No raw data or scientific statistics were changed. Dates identify documented research phases and actual publication/completion events, not inferred per-run dates.",
    })
    atomic_json(public / "data/research.json", data)
    atomic_json(public / "data/runs.json", runs)
    atomic_json(workspace / "work/portal_data_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--public-dir", type=Path, default=PORTAL / "public")
    parser.add_argument("--as-of", default="2026-09-15", help="Documented publication-snapshot date, YYYY-MM-DD")
    parser.add_argument("--check", action="store_true", help="Validate existing public JSON and links without regenerating assets")
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
        audit = export(args.workspace, args.public_dir, args.as_of)
    print(json.dumps({key: audit[key] for key in ["status", "coverage", "tables", "sources", "warnings", "activity_events", "published_raw_logs"]}, indent=2))


if __name__ == "__main__":
    main()
