"""Public research-data contract, evidence coverage, and privacy regressions."""

import csv
import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
import zipfile
from pathlib import Path

from external_inputs import inputs, needs


PORTAL = Path(__file__).resolve().parents[1]
# A verified campaign workspace and research repository are read only. Both can be
# supplied explicitly when the portal is checked out somewhere else; tests that need
# them are skipped, with the variable named, when they are neither configured nor present
# (tests/external_inputs.py). Everything else reads only the published snapshot.
_WORKSPACE, _REPO, _AUDIT, _INVENTORY = inputs()
WORKSPACE, REPO, DATA_AUDIT = _WORKSPACE.path, _REPO.path, _AUDIT.path
# The published run inventory can be an extended copy kept outside the read-only workspace
# (export_research.py --run-inventory): the campaign inventory plus the corpus rows of landed batches.
RUN_INVENTORY = _INVENTORY.path
CAMPAIGN_RUN_INVENTORY = WORKSPACE / "outputs/tables/complete_run_inventory.csv"
RUNS = 3049
LANDED_BATCH_RUNS = {"cgn1": 6, "cpl1": 15, "cvh1": 12, "cuc1": 30, "cgn2": 15, "cpl2": 15, "cvt1": 15, "cgn3": 12, "cvt3": 15, "cvt2": 21, "cvt4": 18, "cvt5": 12}
PUBLIC = PORTAL / "public"
PARTITION_IDS = [f"MT{line}" for line in range(175, 212)]
APPENDED_IDS = [f"MT{line}" for line in range(212, 218)]
LANDED_IDS = ["MT218"]
CGN3_IDS = ["MT219"]
CVT23_IDS = ["MT220", "MT221"]
CVT45_IDS = ["MT222", "MT223"]
PRIVATE = re.compile(r"/Users/|/home/|/scratch/|/data1/|teshnizi|salehkaleybars|s5014158|hmkhd2|100\.120\.248\.20|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|\b(?:p-cfer-\d+|node\d{3}|nodelogin\d+|login\d+|login\.[A-Za-z0-9_.…-]+)\b", re.I)


PUBLIC_REPOSITORY_PREFIX = re.compile(
    r'https://github\.com/teshnizi2/(?:hierarchical-metaoptimize|metaoptimize-research-notebook)(?=$|[/#?\s"\'])'
)


def public_metadata_private_match(text):
    # Exempt only the explicitly public repository prefix, never the remaining
    # path, query, or surrounding content that could still contain private data.
    return PRIVATE.search(PUBLIC_REPOSITORY_PREFIX.sub('PUBLIC_REPOSITORY', text))


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ResearchExportTests(unittest.TestCase):
    def research(self):
        path = PUBLIC / "data/research.json"
        self.assertTrue(path.is_file(), "Public research export has not been generated")
        return json.loads(path.read_text())

    def runs(self):
        path = PUBLIC / "data/runs.json"
        self.assertTrue(path.is_file(), "Public run export has not been generated")
        return json.loads(path.read_text())

    def exporter(self):
        path = PORTAL / "scripts/export_research.py"
        self.assertTrue(path.is_file(), "Source-catalog-aware exporter is not implemented")
        spec = importlib.util.spec_from_file_location("research_exporter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def model(self):
        self.exporter()  # puts scripts/ on sys.path
        import register_model
        return register_model

    def test_clean_turns_markdown_escapes_into_literal_characters(self):
        # MASTER-TABLE row 215 writes the selected cells as S\* and M\*. The escape must not leak into the notebook text,
        # and an escaped star must not be read as an emphasis mark (that used to eat the stars between two S\* marks).
        clean = self.model().clean
        self.assertEqual(clean(r"PRIMARY `GAP_END` = plateau5(S\*) - plateau5(M\*) = 65.0573"), "PRIMARY GAP_END = plateau5(S*) - plateau5(M*) = 65.0573")
        self.assertEqual(clean(r"18-19 for M\* against 80 for S\*). Selection: S\* and M\* are both max-of-5"),
                         "18-19 for M* against 80 for S*). Selection: S* and M* are both max-of-5")
        self.assertEqual(clean(r"AUC rung but **-2.3489 pp at S\***, hence"), "AUC rung but -2.3489 pp at S*, hence")
        self.assertEqual(clean(r"CEIL-\* / a\_b / `\| LOWER-BOUND`"), "CEIL-* / a_b / | LOWER-BOUND")
        self.assertEqual(clean("an *emphasised* word and 2 * 3"), "an emphasised word and 2 * 3")

    def test_published_text_carries_no_markdown_escapes(self):
        def strings(value, path):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from strings(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    yield from strings(item, f"{path}[{index}]")
            elif isinstance(value, str):
                yield path, value
        for name, value in [("research.json", self.research()), ("runs.json", self.runs())]:
            hits = [(path, text[max(0, m.start() - 30):m.end() + 10]) for path, text in strings(value, name) for m in re.finditer(r"\\[*_|`]", text)]
            self.assertEqual(hits, [], name)

    def test_complete_register_and_area_counts(self):
        data = self.research()
        self.assertEqual(len(data["experiments"]), 160)
        self.assertEqual(len({e["id"] for e in data["experiments"]}), 160)
        self.assertEqual(len(data["areas"]), 10)
        self.assertEqual(sum(a["count"] for a in data["areas"]), 160)
        self.assertEqual(data["meta"]["stats"], {"experiments": 160, "researchQuestions": 142, "methodChecks": 18, "runs": RUNS, "figures": 54, "areas": 10})

    @needs(_WORKSPACE)
    def test_register_ids_are_the_campaign_register_plus_the_imported_master_table_rows(self):
        data = self.research()
        original = csv_rows(WORKSPACE / "outputs/tables/complete_experiment_register.csv")
        self.assertEqual({e["id"] for e in data["experiments"]}, {e["id"] for e in original} | set(PARTITION_IDS) | set(APPENDED_IDS) | set(LANDED_IDS) | set(CGN3_IDS) | set(CVT23_IDS) | set(CVT45_IDS))

    def test_published_runs_have_unique_identifiers_and_logs(self):
        runs = self.runs()
        self.assertEqual(len(runs), RUNS)
        self.assertEqual(len({r["id"] for r in runs}), RUNS)
        self.assertEqual({r["account"] for r in runs}, {"Account1", "Account2"})
        self.assertTrue(all(r["parameters"].get("logAvailability") for r in runs))
        self.assertEqual({batch: sum(r["batch"] == batch for r in runs) for batch in LANDED_BATCH_RUNS}, LANDED_BATCH_RUNS)

    @needs(_WORKSPACE, _INVENTORY)
    def test_run_inventory_keeps_all_jobs_with_unique_identifiers(self):
        runs = self.runs()
        self.assertEqual(len(runs), RUNS)
        self.assertEqual(len({r["id"] for r in runs}), RUNS)
        original = csv_rows(RUN_INVENTORY)
        self.assertEqual({r["jobId"] for r in runs}, {r["job_id"] for r in original})
        # The extension only appends: every campaign row is kept, in order and unchanged, ahead of the new batches.
        campaign = csv_rows(CAMPAIGN_RUN_INVENTORY)
        self.assertEqual(original[:len(campaign)], campaign)
        added = original[len(campaign):]
        self.assertEqual({batch: sum(r["run"].startswith(batch + "-") for r in added) for batch in LANDED_BATCH_RUNS}, LANDED_BATCH_RUNS)
        self.assertEqual(len(added), sum(LANDED_BATCH_RUNS.values()))
        self.assertEqual({r["account"] for r in runs}, {"Account1", "Account2"})
        self.assertTrue(all(r["parameters"].get("logAvailability") for r in runs))

    def test_all_54_figures_preserve_register_pages(self):
        figures = self.research()["figures"]
        self.assertEqual({f["id"] for f in figures}, {f"page-{n}" for n in range(1, 55)})
        self.assertEqual(sum(f["kind"] == "register" for f in figures), 30)
        for figure in figures:
            self.assertTrue((PUBLIC / figure["href"].lstrip("/")).is_file(), figure["id"])

    def test_every_published_raw_log_matches_its_public_hash(self):
        runs = self.runs()
        self.assertEqual(sum(bool(run.get("logHref")) for run in runs), RUNS)
        for run in runs:
            public = (PUBLIC / run["logHref"].lstrip("/")).read_bytes()
            self.assertEqual(run["parameters"]["publicLogSha256"], hashlib.sha256(public).hexdigest(), run["jobId"])
            self.assertRegex(run["parameters"]["originalLogSha256"], r"^[0-9a-f]{64}$", run["jobId"])

    @needs(_WORKSPACE, _AUDIT)
    def test_all_raw_logs_are_exact_sanitized_archives_with_hashes(self):
        runs = self.runs()
        self.assertEqual(sum(bool(run.get("logHref")) for run in runs), RUNS)
        exporter = self.exporter()
        evidence = exporter.read_evidence(WORKSPACE)
        archived = exporter.archive_log_index(evidence)
        latest = {str(run["job_id"]): run for run in evidence["cvk2_evidence"]["runs"]}
        audit = json.loads(DATA_AUDIT.read_text())
        receipts = {row["job_id"]: row for row in audit["raw_log_receipts"]}
        self.assertEqual(set(receipts), {run["jobId"] for run in runs})
        measurement_rows = 0
        for run in runs:
            job = run["jobId"]
            if job in latest:
                source = Path(latest[job]["raw_path"])
            else:
                self.assertEqual(len(archived[job]), 1, job)
                source = archived[job][0]
            original = source.read_bytes()
            public = (PUBLIC / run["logHref"].lstrip("/")).read_bytes()
            original_text, public_text = original.decode("utf-8"), public.decode("utf-8")
            self.assertEqual(public_text, exporter.sanitize_text(original_text, WORKSPACE), job)
            self.assertEqual(re.findall(r"\r\n|\r|\n", original_text), re.findall(r"\r\n|\r|\n", public_text), job)
            self.assertEqual(run["parameters"]["originalLogSha256"], hashlib.sha256(original).hexdigest(), job)
            self.assertEqual(run["parameters"]["publicLogSha256"], hashlib.sha256(public).hexdigest(), job)
            self.assertEqual(receipts[job]["original_sha256"], run["parameters"]["originalLogSha256"], job)
            self.assertEqual(receipts[job]["public_sha256"], run["parameters"]["publicLogSha256"], job)
            self.assertEqual(run["parameters"]["logAvailability"], "published: sanitized original raw log", job)
            self.assertIsNone(PRIVATE.search(public_text), job)
            for raw_line, public_line in zip(original_text.splitlines(), public_text.splitlines()):
                # Epoch accuracy lines and numeric-only rows must be verbatim.
                if re.match(r"^\s*Epoch\s+\d+,", raw_line) or re.fullmatch(r"\s*(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s+){2,}[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*", raw_line):
                    measurement_rows += 1
                    self.assertEqual(raw_line, public_line, job)
        self.assertGreater(measurement_rows, 1000)

    def test_relationships_resolve_in_both_directions(self):
        data, runs = self.research(), self.runs()
        experiments = {r["id"]: r for r in data["experiments"]}
        catalogs = {"figureIds": {r["id"]: r for r in data["figures"]}, "tableIds": {r["id"]: r for r in data["tables"]}, "codeIds": {r["id"]: r for r in data["sources"]}, "runIds": {r["id"]: r for r in runs}, "warningIds": {r["id"]: r for r in data["warnings"]}, "eventIds": {r["id"]: r for r in data["activity"]}}
        for eid, experiment in experiments.items():
            self.assertTrue(experiment["figureIds"], eid)
            self.assertTrue(experiment["sourceRefs"], eid)
            self.assertTrue(experiment["tableIds"], eid)
            for field, catalog in catalogs.items():
                for ident in experiment[field]:
                    self.assertIn(ident, catalog, f"{eid}.{field}")
                    if field in ["figureIds", "tableIds", "runIds", "eventIds"]:
                        self.assertIn(eid, catalog[ident]["experimentIds"], f"Reverse link {eid}/{ident}")
            for ref in experiment["sourceRefs"]:
                self.assertIn(ref["sourceId"], catalogs["codeIds"])
                if "line" in ref:
                    self.assertGreaterEqual(ref["line"], 1)
                    self.assertLessEqual(ref["line"], catalogs["codeIds"][ref["sourceId"]]["lines"])
        for items in [data["figures"], data["tables"], data["activity"], runs]:
            for row in items:
                self.assertTrue(set(row["experimentIds"]) <= set(experiments))

    def test_cvk2_clean_gates_do_not_become_scientific_success(self):
        data = self.research()
        row = next(e for e in data["experiments"] if e["id"] == "CVK2")
        self.assertEqual(row["outcome"], "unresolved")
        self.assertEqual(set(row["figureIds"]), {"page-22", "page-53"})
        self.assertEqual(len(row["runIds"]), 27)
        self.assertEqual(data["latest"]["peakSet"], [16, 19, 22])
        self.assertEqual(data["latest"]["predictedCut"], 22)
        self.assertEqual(data["latest"]["bestCut"], 19)
        self.assertAlmostEqual(data["latest"]["bar"], 0.9573130207199779)
        self.assertEqual(len(data["latest"]["cells"]), 9)
        self.assertTrue(all(c["n"] == 3 for c in data["latest"]["cells"]))
        latest_warnings = " ".join(w["detail"] for w in data["warnings"] if w["experimentId"] == "CVK2")
        self.assertIn("PROBE=0", latest_warnings)
        self.assertIn("two groups", latest_warnings)
        for run in self.runs():
            if "CVK2" in run["experimentIds"]:
                self.assertEqual(run["status"], "completed")
                self.assertTrue((PUBLIC / run["logHref"].lstrip("/")).is_file())

    @needs(_WORKSPACE, _INVENTORY)
    def test_every_csv_is_published_without_row_or_column_truncation(self):
        data = self.research()
        originals = list((WORKSPACE / "outputs/tables").rglob("*.csv"))
        self.assertEqual(len(originals), 201)
        tables_by_href = {row["href"]: row for row in data["tables"]}
        for original in originals:
            rel = original.relative_to(WORKSPACE / "outputs/tables")
            href = "/assets/tables/" + rel.as_posix()
            self.assertIn(href, tables_by_href, str(rel))
            if rel.as_posix() in {"complete_experiment_register.csv", "outcomes_by_area.csv", "goal_outcomes.csv"}:
                continue  # regenerated from the notebook register; see the regenerated-table tests
            if rel.as_posix() == "complete_run_inventory.csv":
                original = RUN_INVENTORY
            with original.open(newline="") as handle:
                rows = list(csv.reader(handle))
            with (PUBLIC / href.lstrip("/")).open(newline="") as handle:
                saved = list(csv.reader(handle))
            self.assertEqual(len(saved), len(rows), str(rel))
            self.assertEqual(len(saved[0]), len(rows[0]), str(rel))
            self.assertEqual(tables_by_href[href]["rows"], len(rows) - 1)
        kernel = PUBLIC / "assets/tables/measurement_and_mechanism/measurement__kernel_field_curves__all_curve_points.csv"
        with kernel.open() as handle:
            self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 561)
        hdis = PUBLIC / "assets/tables/measurement_and_mechanism/mechanism__sign_disagreement__full_61_cut_curve.csv"
        with hdis.open() as handle:
            self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 122)

    def test_privacy_scan_allows_only_the_two_public_repository_prefixes(self):
        for repository in ["hierarchical-metaoptimize", "metaoptimize-research-notebook"]:
            url = "https://github.com/teshnizi2/" + repository
            self.assertIsNone(public_metadata_private_match(url))
            self.assertIsNone(public_metadata_private_match(url + "/commit/" + "a" * 40))
        for text in [
            "teshnizi", "account=teshnizi2",
            "https://github.com/teshnizi2/unrelated",
            "https://github.com/teshnizi2/hierarchical-metaoptimize-copy",
            "https://github.com/teshnizi2/hierarchical-metaoptimize/blob/main/home/teshnizi/key",
            "https://github.com/teshnizi2/hierarchical-metaoptimize?account=s5014158",
        ]:
            self.assertIsNotNone(public_metadata_private_match(text))

    def test_public_text_has_no_private_paths_or_accounts(self):
        self.research()
        for path in list((PUBLIC / "data").glob("*.json")) + list((PUBLIC / "assets").rglob("*.csv")) + list((PUBLIC / "assets").rglob("*.txt")):
            self.assertIsNone(public_metadata_private_match(path.read_text()), str(path.relative_to(PUBLIC)))

    def test_zip_contains_all_charts_and_tables_and_sanitized_text(self):
        data = self.research()
        path = PUBLIC / data["meta"]["downloads"]["bundle"].lstrip("/")
        self.assertTrue(path.is_file())
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            self.assertEqual(len([n for n in names if n.startswith("figures/")]), 54)
            self.assertEqual(len([n for n in names if n.startswith("tables/") and n.endswith(".csv")]), 201)
            self.assertEqual(len([n for n in names if n.startswith("logs/")]), RUNS)
            self.assertIn("report.pdf", names)
            for name in names:
                self.assertFalse(name.startswith("/") or ".." in Path(name).parts)
                if name.endswith((".csv", ".txt", ".json")):
                    self.assertIsNone(PRIVATE.search(archive.read(name).decode()), name)

    def test_source_catalog_is_mandatory(self):
        exporter = self.exporter()
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(ValueError, "source-index|source-links|source catalog"):
                exporter.load_source_catalog(Path(empty))

    def test_source_catalog_rejects_private_source_bodies(self):
        exporter = self.exporter()
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary)
            (public / "data").mkdir()
            (public / "source").mkdir()
            content = b"archive = '/zfsstore/user/synthetic-private-account/data'\n"
            (public / "source/example.txt").write_bytes(content)
            source = {"id": "src-example", "path": "analysis/example.py", "language": "python", "role": "Scorer", "href": "/source/example.txt", "originalSha256": hashlib.sha256(content).hexdigest(), "publicSha256": hashlib.sha256(content).hexdigest(), "redacted": False, "lines": 1, "experimentIds": [], "referenceType": "explicit"}
            (public / "data/source-index.json").write_text(json.dumps([source]))
            (public / "data/source-links.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Private content"):
                exporter.load_source_catalog(public)

    def test_embedded_sources_equal_the_current_source_catalog(self):
        catalog = json.loads((PUBLIC / "data/source-index.json").read_text())
        self.assertEqual(self.research()["sources"], catalog)

    def test_sanitizer_keeps_measurements_and_removes_private_payloads(self):
        exporter = self.exporter()
        example = 'account=s5014158 /home/s5014158/metaopt/runs/run.out /Users/teshnizi/My Folder/table.csv email=person@example.org alpha0=1e-6 test=69.24 token=sk-abc12345678901234567890123456789'
        public = exporter.sanitize_text(example, WORKSPACE)
        self.assertIsNone(PRIVATE.search(public))
        self.assertIn("Account2", public)
        self.assertIn("alpha0=1e-6 test=69.24", public)
        self.assertNotIn("sk-abc123", public)

    def test_log_sanitizer_preserves_line_breaks_and_scientific_identifiers(self):
        exporter = self.exporter()
        original = "host=node851 login=login.alice.example.nl\r\npath=/data1/salehkaleybars/run.out\nnode14420 CIFAR-100 ResNet-18 PEAK-AT-22\r\n0 1.2345 69.2400 1e-6\n"
        public = exporter.sanitize_text(original, WORKSPACE)
        self.assertIsNone(PRIVATE.search(public))
        self.assertEqual(re.findall(r"\r\n|\r|\n", original), re.findall(r"\r\n|\r|\n", public))
        self.assertIn("node14420 CIFAR-100 ResNet-18 PEAK-AT-22", public)
        self.assertIn("0 1.2345 69.2400 1e-6", public)
        self.assertEqual(public.count("[HOST_"), 2)

    def test_historical_phase_dates_are_not_run_start_dates(self):
        data = self.research()
        phases = [event for event in data["activity"] if event["kind"] == "research-phase"]
        self.assertEqual(len(phases), 9)
        self.assertTrue(all("Documented phase" in event["detail"] for event in phases))
        self.assertTrue(any(e["kind"] == "completed-experiment" and e["date"] == "2026-09-14" for e in data["activity"]))
        self.assertTrue(any(e["kind"] == "publication-snapshot" and e["date"] == data["meta"]["asOf"] == "2026-09-17" for e in data["activity"]))
        self.assertTrue(all("date" not in run for run in self.runs()))

    def test_records_without_dated_phases_have_snapshot_import_provenance(self):
        data = self.research()
        imported = [event for event in data["activity"] if event["kind"] == "snapshot-import"]
        self.assertEqual(len(imported), 1, "Undated historical records need an explicit snapshot-import event")
        event = imported[0]
        self.assertEqual(event["date"], data["meta"]["asOf"])
        self.assertIn("not training dates", event["detail"])
        phase_ids = {eid for item in data["activity"] if item["kind"] != "snapshot-import" for eid in item["experimentIds"]}
        expected = {e["id"] for e in data["experiments"]} - phase_ids
        self.assertEqual(set(event["experimentIds"]), expected)
        self.assertEqual(len(expected), 80)
        for experiment in data["experiments"]:
            self.assertTrue(experiment["eventIds"], experiment["id"])
            if experiment["id"] in expected:
                self.assertIn(event["id"], experiment["eventIds"])

    # ---- Four-outcome model and the count-matched partition audit ----
    @needs(_WORKSPACE)
    def test_approved_correction_split_is_explicit_and_complete(self):
        model = self.model()
        original = csv_rows(WORKSPACE / "outputs/tables/complete_experiment_register.csv")
        self.assertEqual({r["id"] for r in original if r["outcome"] == "correction"}, set(model.CORRECTION_REMAP))
        kinds = [kind for kind, _, _ in model.CORRECTION_REMAP.values()]
        self.assertEqual((kinds.count("research"), kinds.count("method-check")), (8, 15))
        self.assertTrue(all(reason.strip() for _, _, reason in model.CORRECTION_REMAP.values()))
        published = {e["id"]: e for e in self.research()["experiments"]}
        for eid, (kind, outcome, reason) in model.CORRECTION_REMAP.items():
            row = published[eid]
            self.assertEqual((row["kind"], row["outcome"], row["corrected"]), (kind, outcome, True), eid)
            self.assertEqual(row["correction"]["note"], reason, eid)
            self.assertEqual(row["correction"]["previousOutcome"], "correction", eid)
        with self.assertRaisesRegex(ValueError, "no approved mapping"):
            model.remap_base_row({"id": "MT999", "outcome": "correction"})

    def test_published_outcome_model_has_four_outcomes_and_a_badge(self):
        data = self.research()
        self.assertEqual(list(data["meta"]["outcomeModel"]["outcomes"].values()), ["Goal met", "Goal missed", "Mixed", "Open"])
        for e in data["experiments"]:
            self.assertNotEqual(e["outcome"], "correction", e["id"])
            if e["kind"] == "method-check":
                self.assertIsNone(e["outcome"], e["id"])
            else:
                self.assertIn(e["outcome"], {"success", "fail", "mixed", "unresolved"}, e["id"])
            self.assertEqual(e["corrected"], bool(e["correction"]), e["id"])
        warnings = {w["id"]: w for w in data["warnings"]}
        self.assertEqual(warnings["warning-MT014-verdict"]["severity"], "limitation")
        self.assertEqual(warnings["warning-MT014-correction"]["severity"], "correction")
        self.assertEqual(warnings["warning-MT113-verdict"]["severity"], "correction")
        self.assertNotIn("warning-MT113-correction", warnings, "a method check's verdict warning is its correction record")

    @needs(_REPO)
    def test_partition_audit_rows_come_from_the_pinned_master_table(self):
        model = self.model()
        lines = model.master_table_at_commit(REPO)
        rows = model.partition_rows(lines)
        self.assertEqual([r["id"] for r in rows], PARTITION_IDS)
        self.assertTrue(all(r["section"] == "10" and r["area"] == "Count-matched partition audit" for r in rows))
        outcomes = [r["outcome"] or r["kind"] for r in rows]
        self.assertEqual({o: outcomes.count(o) for o in set(outcomes)}, {"success": 12, "fail": 5, "mixed": 4, "unresolved": 13, "method-check": 3})
        headline = rows[0]
        self.assertEqual(headline["outcome"], "success")
        self.assertIn("20 of 20 cells", headline["result"])
        self.assertIn("+0.5556 ± 0.0448", headline["result"])
        with self.assertRaisesRegex(ValueError, "heading moved"):
            model.partition_rows(["", *lines])
        with self.assertRaisesRegex(ValueError, "pinned section-10 bytes"):
            model.master_table_at_commit(REPO, "627d69ffd624d768178719b4c5b52b0e3b0e9ed5")

    def test_partition_audit_links_runs_phase_and_figure(self):
        data, runs = self.research(), self.runs()
        by_id = {e["id"]: e for e in data["experiments"]}
        phase = next(e for e in data["activity"] if e["id"] == "phase-03")
        self.assertIn("Partition tests", phase["title"])
        self.assertEqual(set(phase["experimentIds"]), {"MT098", *PARTITION_IDS})
        page8 = next(f for f in data["figures"] if f["id"] == "page-8")
        self.assertTrue(set(PARTITION_IDS) <= set(page8["experimentIds"]))
        expected_runs = {"MT176": 9, "MT178": 15, "MT181": 6, "MT186": 12, "MT190": 36, "MT211": 4, "MT175": 0, "MT192": 0}
        for eid, count in expected_runs.items():
            self.assertEqual(len(by_id[eid]["runIds"]), count, eid)
        baseline = {r["id"]: r for r in runs}
        self.assertTrue(all(r["batch"] == baseline[r["id"]]["batch"] for r in runs))

    # ---- MASTER-TABLE lines 212-217 (campaign commit 64e4f47) ----
    @needs(_REPO)
    def test_appended_rows_come_from_the_pinned_master_table(self):
        model = self.model()
        rows = model.appended_rows(model.appended_master_table(REPO))
        self.assertEqual([r["id"] for r in rows], APPENDED_IDS)
        self.assertEqual([(r["section"], r["outcome"], json.loads(r["batches"])) for r in rows], [
            ("9", "success", ["cgn1"]), ("9", "mixed", ["cpl1"]), ("9", "success", ["cvh1"]),
            ("1", "success", ["cuc1"]), ("9", "success", ["cgn2"]), ("9", "success", ["cpl2"])])
        self.assertTrue(all(r["kind"] == "research" and not r["corrected"] for r in rows))
        self.assertTrue(all(r["reason"].startswith("Verdict: ") and model.APPENDED_ROWS[int(r["master_table_line"])][5] in r["reason"] for r in rows))
        self.assertIn("+41.5167 pp = +67.23 SE", rows[4]["result"])
        self.assertIn("+52.1367 pp = +91.95 SE", rows[5]["result"])
        with self.assertRaisesRegex(ValueError, "pinned appended-row bytes"):
            original = model.APPENDED_COMMIT
            try:
                model.APPENDED_COMMIT = model.PARTITION_AUDIT_COMMIT
                model.appended_master_table(REPO)
            finally:
                model.APPENDED_COMMIT = original
        with self.assertRaisesRegex(ValueError, "exactly lines 212-217"):
            model.appended_rows(model.appended_master_table(REPO)[:-1])
        self.assertEqual(model.table_cells(r"| a \| b | c |"), ["a | b", "c"])

    def test_appended_rows_link_phase_figures_and_their_batch_runs(self):
        data, runs = self.research(), self.runs()
        by_id = {e["id"]: e for e in data["experiments"]}
        phase = next(e for e in data["activity"] if e["id"] == "phase-09")
        self.assertEqual((phase["date"], phase["kind"], phase["experimentIds"]), ("2026-09-15", "research-phase", APPENDED_IDS + LANDED_IDS + CGN3_IDS + CVT23_IDS + CVT45_IDS))
        self.assertTrue(phase["detail"].startswith("Documented phase: 15-16 Sep 2026. Test: "))
        figures = {"MT212": "page-19", "MT213": "page-19", "MT214": "page-21", "MT215": "page-23", "MT216": "page-19", "MT217": "page-19"}
        batches = dict(zip(APPENDED_IDS, LANDED_BATCH_RUNS))
        for eid, page in figures.items():
            self.assertEqual(by_id[eid]["figureIds"], [page])
            linked = [r for r in runs if eid in r["experimentIds"]]
            self.assertEqual(len(linked), LANDED_BATCH_RUNS[batches[eid]], eid)
            self.assertEqual({r["batch"] for r in linked}, {batches[eid]}, eid)
            self.assertEqual(sorted(by_id[eid]["runIds"]), sorted(r["id"] for r in linked), eid)
        for run in runs:
            if run["batch"] in LANDED_BATCH_RUNS:
                self.assertEqual((run["account"], run["status"], run["dataset"]), ("Account2", "completed", "CIFAR100"), run["id"])
                self.assertTrue(run["parameters"]["logSourceReference"].startswith("archive/Account2/runs/"), run["id"])
                self.assertTrue(run["logHref"].startswith("/assets/logs/" + run["batch"] + "-"), run["id"])
        # cuc1 closes MT019's remaining counts, so its runs link to MT019 as well as MT215, like cau1 to MT019 and MT020.
        self.assertEqual({tuple(r["experimentIds"]) for r in runs if r["batch"] == "cuc1"}, {("MT019", "MT215")})

    @needs(_WORKSPACE, _REPO)
    def test_row_amendments_come_from_the_pinned_bookkeeping_commit(self):
        model = self.model()
        lines, before = model.amended_master_table(REPO), model.appended_master_table(REPO)
        self.assertEqual({n for n in range(1, len(lines) + 1) if lines[n - 1] != before[n - 1]}, {3, 19, 162, 163, 166, 213})
        changes = {eid: (previous, outcome) for eid, (_, previous, outcome, *_rest) in model.ROW_AMENDMENTS.items() if previous != outcome}
        self.assertEqual(changes, {"MT019": ("unresolved", "success")}, "only row 19's verdict column moved")
        self.assertTrue(all(reason.strip() and len(reason) < 500 for *_, reason, _ in model.ROW_AMENDMENTS.values()))
        data = self.research()
        by_id = {e["id"]: e for e in data["experiments"]}
        warnings = {w["id"]: w for w in data["warnings"]}
        for eid, (line, previous, outcome, token, batches, reason, replacement) in model.ROW_AMENDMENTS.items():
            record = by_id[eid]
            self.assertEqual((record["outcome"], record["corrected"]), (outcome, eid in model.CORRECTION_REMAP), eid)
            self.assertTrue(set(batches) <= set(record["batches"]), eid)
            warning = warnings[f"warning-{eid}-amendment"]
            # CORRECTIONS 231 retired the closing pending-cvt1 clause of MT213's 229 note.
            if eid in model.CGN3_AMENDMENTS and model.CGN3_AMENDMENTS[eid]["retire"]:
                reason = reason.replace(*model.CGN3_AMENDMENTS[eid]["retire"])
            self.assertIn(reason, warning["detail"])
            self.assertIn(f"line {line} at 6e33fd8", warning["detail"])
            if replacement:
                self.assertEqual({key: record[key] for key in replacement}, replacement, eid)
                self.assertIn("Outcome before the amendment: Open.", warning["detail"])
            else:
                self.assertIn("Amended at CORRECTIONS 229: " + reason, record["scope"], eid)
            self.assertTrue(any(ref["label"].endswith("CORRECTIONS.md section 229") for ref in record["sourceRefs"]), eid)
        # A drifted outcome or a verdict-less outcome move stops the export instead of being applied.
        unamended = model.load_unamended_register(WORKSPACE, REPO)
        self.assertEqual(next(r for r in unamended if r["id"] == "MT019")["outcome"], "unresolved")
        with self.assertRaisesRegex(ValueError, "outcome drifted"):
            model.apply_row_amendments([dict(r, outcome="fail") if r["id"] == "MT019" else r for r in unamended], REPO)
        original = model.ROW_AMENDMENTS["MT163"]
        try:
            model.ROW_AMENDMENTS["MT163"] = (original[0], "success", "fail", *original[3:])
            with self.assertRaisesRegex(ValueError, "unchanged verdict cannot change the outcome"):
                model.apply_row_amendments(unamended, REPO)
        finally:
            model.ROW_AMENDMENTS["MT163"] = original

    # ---- MASTER-TABLE line 218 (campaign commit 82867bb, CORRECTIONS 230: cvt1) ----
    @needs(_REPO)
    def test_landed_row_comes_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.landed_master_table(REPO), model.amended_master_table(REPO)
        self.assertEqual(len(lines), 218)
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3, 19})
        rows = model.landed_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"]) for r in rows],
                         [("MT218", "9", "mixed", ["cvt1"], "mixed")])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: STEP-SIZE-NEEDED-VOTE-SUFFICES + HARNESS-CLEAN + "))
        self.assertIn("+34.6540 pp = +61.12 SE", rows[0]["result"])
        self.assertEqual(json.loads(rows[0]["intervention"])["arms"], {"MUTE": 3, "DOSE": 3, "INJECT": 3})
        intervened = model.intervened_runs(REPO)
        # The exclusion list has since grown (cvt3, cvt2: test_cvt23_rows_come_from_the_pinned_landing_commit); cvt1's rows are unchanged.
        self.assertEqual(sorted(r["arm"] for r in intervened.values() if r["batch"] == "cvt1"), sorted(["MUTE", "DOSE", "INJECT"] * 3))
        with self.assertRaisesRegex(ValueError, "does not match the pinned landed-row bytes"):
            pinned = model.LANDED_COMMIT
            try:
                model.LANDED_COMMIT = model.AMENDMENT_COMMIT
                model.landed_master_table(REPO)
            finally:
                model.LANDED_COMMIT = pinned
        with self.assertRaisesRegex(ValueError, "exactly lines 218-218"):
            model.appended_rows(lines[:-1], model.LANDED_ROWS, 218, 218, model.LANDED_COMMIT)
        data, runs = self.research(), self.runs()
        record = next(e for e in data["experiments"] if e["id"] == "MT218")
        self.assertEqual(sorted(record["runIds"]), sorted(r["id"] for r in runs if r["batch"] == "cvt1"))
        marked = {r["jobId"] for r in runs if "intervention" in r["parameters"] and r["batch"] == "cvt1"}
        self.assertEqual(marked, {job for job, r in intervened.items() if r["batch"] == "cvt1"})
        self.assertIn("warning-MT218-intervention", record["warningIds"])

    def test_outcome_summary_tables_are_regenerated_from_the_register(self):
        data = self.research()
        experiments = data["experiments"]
        with (PUBLIC / "assets/tables/outcomes_by_area.csv").open(newline="") as handle:
            areas = list(csv.DictReader(handle))
        self.assertEqual([r["area"] for r in areas], [a["label"] for a in data["areas"]])
        self.assertNotIn("correction", areas[0])
        for row in areas:
            records = [e for e in experiments if e["area"] == row["area"]]
            research = [e for e in records if e["kind"] == "research"]
            self.assertEqual(int(row["records"]), len(records))
            self.assertEqual(int(row["research_questions"]), len(research))
            for outcome in ["success", "fail", "mixed", "unresolved"]:
                self.assertEqual(int(row[outcome]), sum(e["outcome"] == outcome for e in research), (row["area"], outcome))
            self.assertEqual(int(row["method_checks"]), len(records) - len(research))
            self.assertEqual(int(row["corrected"]), sum(e["corrected"] for e in records))
        self.assertEqual(sum(int(r["records"]) for r in areas), 160)
        self.assertEqual([sum(int(r[o]) for r in areas) for o in ["success", "fail", "mixed", "unresolved", "method_checks"]], [49, 41, 29, 23, 18])

    @needs(_WORKSPACE, _AUDIT)
    def test_goal_outcome_table_keeps_the_report_goals_with_current_counts(self):
        data = self.research()
        experiments = data["experiments"]
        with (PUBLIC / "assets/tables/goal_outcomes.csv").open(newline="") as handle:
            goals = list(csv.DictReader(handle))
        original = csv_rows(WORKSPACE / "outputs/tables/goal_outcomes.csv")
        figures = {f["id"]: f for f in data["figures"]}
        self.assertEqual([g["page"] for g in goals], [o["page"] for o in original])
        by_id = {e["id"]: e for e in experiments}
        for goal, old in zip(goals, original):
            for column in ["title", "goal", "comparison", "why_test", "status", "file", "kind"]:
                self.assertEqual(goal[column], old[column], (goal["page"], column))
            linked = figures["page-" + goal["page"]]["experimentIds"]
            self.assertEqual(goal["entry_ids"], "; ".join(linked))
            research = [by_id[i] for i in linked if by_id[i]["kind"] == "research"]
            self.assertEqual(int(goal["research_questions"]), len(research))
            self.assertEqual(int(goal["unresolved"]), sum(e["outcome"] == "unresolved" for e in research))
        page23 = next(g for g in goals if g["page"] == "23")
        self.assertIn("MT215", page23["entry_ids"].split("; "))
        audit = json.loads(DATA_AUDIT.read_text())
        receipts = {r["file"]: r.get("regenerated") for r in audit["table_receipts"]}
        self.assertTrue(receipts["outcomes_by_area.csv"] and receipts["goal_outcomes.csv"] and receipts["complete_run_inventory.csv"])

    def test_register_table_lists_every_record(self):
        data = self.research()
        with (PUBLIC / "assets/tables/complete_experiment_register.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        exporter = self.exporter()
        self.assertEqual(list(rows[0].keys()), exporter.REGISTER_COLUMNS)
        self.assertEqual([r["id"] for r in rows], [e["id"] for e in data["experiments"]])
        self.assertNotIn("correction", {r["outcome"] for r in rows})
        self.assertEqual({(r["id"], r["kind"], r["outcome"] or None, r["corrected"] == "Corrected") for r in rows},
                         {(e["id"], e["kind"], e["outcome"], e["corrected"]) for e in data["experiments"]})

    @needs(_AUDIT)
    def test_register_table_receipt_records_the_regeneration(self):
        audit = json.loads(DATA_AUDIT.read_text())
        receipt = next(r for r in audit["table_receipts"] if r["file"] == "complete_experiment_register.csv")
        self.assertEqual((receipt["rows"], receipt["columns"], receipt.get("regenerated")), (160, 20, "scripts/register_model.py register"))

    # ---- MASTER-TABLE line 219 and the in-place amendments of rows 213 and 218 (campaign commit e3a43da, CORRECTIONS 231: cgn3) ----
    @needs(_REPO)
    def test_cgn3_row_comes_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.cgn3_master_table(REPO), model.landed_master_table(REPO)
        self.assertEqual(len(lines), 219)
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3, 213, 218})
        rows = model.cgn3_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"])) for r in rows],
                         [("MT219", "9", "success", ["cgn3"], "met", ["page-19"])])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: RESCUE-SURVIVES + RHO:1.2235 + TRAIN-AGREES + "))
        self.assertIn("+ ONE-CELL-ONLY Goal met: RESCUE-SURVIVES: ", rows[0]["reason"])
        self.assertIn("= +48.5193 / +39.6573 = 1.2235", rows[0]["result"])
        with self.assertRaisesRegex(ValueError, "does not match the pinned cgn3-landing bytes"):
            pinned = model.CGN3_COMMIT
            try:
                model.CGN3_COMMIT = model.LANDED_COMMIT
                model.cgn3_master_table(REPO)
            finally:
                model.CGN3_COMMIT = pinned
        with self.assertRaisesRegex(ValueError, "exactly lines 219-219"):
            model.appended_rows(lines[:-1], model.CGN3_ROWS, 219, 219, model.CGN3_COMMIT)
        data, runs = self.research(), self.runs()
        record = next(e for e in data["experiments"] if e["id"] == "MT219")
        self.assertEqual(sorted(record["runIds"]), sorted(r["id"] for r in runs if r["batch"] == "cgn3"))
        self.assertEqual({r["epochs"] for r in runs if r["batch"] == "cgn3"}, {430})

    @needs(_WORKSPACE, _REPO)
    def test_cgn3_amendments_change_wording_only(self):
        model = self.model()
        self.assertEqual(set(model.CGN3_AMENDMENTS), {"MT213", "MT218"})
        before = {r["id"]: r for r in model.apply_row_amendments(model.load_unamended_register(WORKSPACE, REPO), REPO)}
        after = {r["id"]: r for r in model.apply_cgn3_amendments(list(before.values()), REPO)}
        self.assertEqual(set(before), set(after))
        for eid, row in after.items():
            if eid not in model.CGN3_AMENDMENTS:
                self.assertEqual(row, before[eid], eid)
                continue
            changed = {key for key in row if row[key] != before[eid].get(key)}
            self.assertEqual(changed, {"scope", "sources", "later_amendment"} | ({"amendment"} if eid == "MT213" else set()), eid)
            self.assertEqual(row["outcome"], before[eid]["outcome"], eid)
        self.assertNotIn("cvt1 has not landed", after["MT213"]["scope"] + after["MT213"]["amendment"])
        self.assertIn("cvt1 has not landed", before["MT213"]["scope"])
        self.assertIn("[SUPERSEDED, not true when written: every cell-pooling reader drops them.]", after["MT218"]["scope"])
        # A drifted outcome stops the export instead of being applied.
        drifted = [dict(r, outcome="success") if r["id"] == "MT218" else r for r in before.values()]
        with self.assertRaisesRegex(ValueError, "outcome drifted before its CORRECTIONS 231 amendment"):
            model.apply_cgn3_amendments(drifted, REPO)

    # ---- MASTER-TABLE lines 220-221 and the in-place amendments of rows 213 and 216 (campaign commit 9c5d72a, CORRECTIONS 234-236: cvt3, cvt2) ----
    @needs(_REPO)
    def test_cvt23_rows_come_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.cvt23_master_table(REPO), model.cgn3_master_table(REPO)
        self.assertEqual(len(lines), 221)
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3, 5, 213, 216})
        rows = model.cvt23_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"])) for r in rows],
                         [("MT220", "9", "mixed", ["cvt3"], "mixed", ["page-19"]), ("MT221", "9", "mixed", ["cvt2"], "mixed", ["page-19"])])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: OWN-STEP-NECESSARY + HARNESS-CLEAN + PATCH-BITES + "))
        self.assertIn("+ TRAIN-AGREES Mixed: OWN-STEP-NECESSARY: ", rows[0]["reason"])
        self.assertTrue(rows[1]["reason"].startswith("Verdict: GRADED + TOP-ATTENUATED + HARNESS-CLEAN + "))
        self.assertIn("+ TRAIN-AGREES Mixed: GRADED + TOP-ATTENUATED: ", rows[1]["reason"])
        self.assertEqual([json.loads(r["intervention"])["arms"] for r in rows],
                         [{"MUTE50": 3, "MUTEDOWN": 3, "MUTECTL": 3}, {"K13": 3, "K33": 3, "K152": 3, "K691": 3, "K2000": 3}])
        intervened = model.intervened_runs(REPO)
        # The list has since grown by cvt4 and cvt5 (test_cvt45_rows_come_from_the_pinned_landing_commit).
        self.assertEqual(sorted(r["batch"] for r in intervened.values() if r["batch"] in {"cvt1", "cvt2", "cvt3"}), sorted(["cvt1"] * 9 + ["cvt3"] * 9 + ["cvt2"] * 15))
        with self.assertRaisesRegex(ValueError, "does not match the pinned cvt3/cvt2-landing bytes"):
            pinned = model.CVT23_COMMIT
            try:
                model.CVT23_COMMIT = model.CGN3_COMMIT
                model.cvt23_master_table(REPO)
            finally:
                model.CVT23_COMMIT = pinned
        with self.assertRaisesRegex(ValueError, "exactly lines 220-221"):
            model.appended_rows(lines[:-1], model.CVT23_ROWS, 220, 221, model.CVT23_COMMIT)
        self.assertEqual(model.changed_span("abcXdef", "abcYYdef"), ("X", "YY"))
        data, runs = self.research(), self.runs()
        by_id = {e["id"]: e for e in data["experiments"]}
        marked = {r["jobId"] for r in runs if "intervention" in r["parameters"]}
        self.assertEqual(marked, set(intervened))
        for eid, batch in [("MT220", "cvt3"), ("MT221", "cvt2")]:
            self.assertEqual(sorted(by_id[eid]["runIds"]), sorted(r["id"] for r in runs if r["batch"] == batch))
            self.assertIn(f"warning-{eid}-intervention", by_id[eid]["warningIds"])

    @needs(_WORKSPACE, _REPO)
    def test_cvt23_amendments_change_wording_only(self):
        model = self.model()
        self.assertEqual({eid: spec["number"] for eid, spec in model.CVT23_AMENDMENTS.items()}, {"MT216": 234, "MT213": 236})
        before = {r["id"]: r for r in model.apply_cgn3_amendments(model.apply_row_amendments(model.load_unamended_register(WORKSPACE, REPO), REPO), REPO)}
        after = {r["id"]: r for r in model.load_register(WORKSPACE, REPO)}
        self.assertEqual(set(before), set(after))
        for eid, row in after.items():
            if eid not in model.CVT23_AMENDMENTS:
                self.assertEqual(row, before[eid], eid)
                continue
            changed = {key for key in row if row[key] != before[eid].get(key)}
            self.assertEqual(changed, {"result" if eid == "MT216" else "scope", "sources", "further_amendments"}, eid)
            self.assertEqual(row["outcome"], before[eid]["outcome"], eid)
        self.assertIn("ISO sits +1.81 pp above kL", after["MT216"]["result"])
        self.assertIn("[RESCOPED at cycle 152, CORRECTIONS 234:", after["MT216"]["result"])
        self.assertTrue(after["MT213"]["scope"].startswith(before["MT213"]["scope"] + " Amended at CORRECTIONS 236: "))
        drifted = [dict(r, outcome="fail") if r["id"] == "MT216" else r for r in before.values()]
        with self.assertRaisesRegex(ValueError, "outcome drifted before its CORRECTIONS 234 amendment"):
            model.apply_cvt23_amendments(drifted, REPO)


    # ---- MASTER-TABLE lines 222-223 (campaign commit 643264c, CORRECTIONS 240-241: cvt4, cvt5); no row amended ----
    @needs(_REPO)
    def test_cvt45_rows_come_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.cvt45_master_table(REPO), model.cvt23_master_table(REPO)
        self.assertEqual(len(lines), 223)
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3, 5})
        rows = model.cvt45_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"]), r["corrected"]) for r in rows],
                         [("MT222", "9", "success", ["cvt4"], "met", ["page-19"], ""), ("MT223", "9", "success", ["cvt5"], "met", ["page-19"], "")])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: OWN-STEP-MAGNITUDE + HARNESS-CLEAN + PATCH-BITES + HOLD-FROM-INIT + "))
        self.assertIn("+ TRAIN-AGREES Goal met: OWN-STEP-MAGNITUDE: ", rows[0]["reason"])
        self.assertTrue(rows[1]["reason"].startswith("Verdict: K-DEPENDENT-EQUILIBRIUM + RESCUE-SURVIVES + K33-HOLDS-PINNED + HARNESS-CLEAN + "))
        self.assertIn("+ K01-PINNED + FLOOR-HOLDS Goal met: K-DEPENDENT-EQUILIBRIUM + RESCUE-SURVIVES + K33-HOLDS-PINNED: ", rows[1]["reason"])
        self.assertEqual([json.loads(r["intervention"])["arms"] for r in rows],
                         [{"HOLDLOW": 3, "HOLDSHARED": 3, "HOLDHIGH": 3, "HOLDHEAD": 3}, {"K13": 3, "K33": 3}])
        intervened = model.intervened_runs(REPO)
        self.assertEqual(sorted(r["batch"] for r in intervened.values()), sorted(["cvt1"] * 9 + ["cvt3"] * 9 + ["cvt2"] * 15 + ["cvt4"] * 12 + ["cvt5"] * 6))
        self.assertTrue(all(r["intervention"].startswith("BETA_HOLD=") for r in intervened.values() if r["batch"] == "cvt4"))
        with self.assertRaisesRegex(ValueError, "does not match the pinned cvt4/cvt5-landing bytes"):
            pinned = model.CVT45_COMMIT
            try:
                model.CVT45_COMMIT = model.CVT23_COMMIT
                model.cvt45_master_table(REPO)
            finally:
                model.CVT45_COMMIT = pinned
        with self.assertRaisesRegex(ValueError, "exactly lines 222-223"):
            model.appended_rows(lines[:-1], model.CVT45_ROWS, 222, 223, model.CVT45_COMMIT)
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.CVT45_INTERVENTIONS_TSV_SHA256
            try:
                model.CVT45_INTERVENTIONS_TSV_SHA256 = model.CVT23_INTERVENTIONS_TSV_SHA256
                model.intervened_runs(REPO)
            finally:
                model.CVT45_INTERVENTIONS_TSV_SHA256 = pinned
        data, runs = self.research(), self.runs()
        by_id = {e["id"]: e for e in data["experiments"]}
        marked = {r["jobId"] for r in runs if "intervention" in r["parameters"]}
        self.assertEqual(marked, set(intervened))
        for eid, batch in [("MT222", "cvt4"), ("MT223", "cvt5")]:
            self.assertEqual(sorted(by_id[eid]["runIds"]), sorted(r["id"] for r in runs if r["batch"] == batch))
            self.assertIn(f"warning-{eid}-intervention", by_id[eid]["warningIds"])

    @needs(_WORKSPACE, _REPO)
    def test_cvt45_import_leaves_every_earlier_record_unchanged(self):
        model = self.model()
        register = model.load_register(WORKSPACE, REPO)
        self.assertEqual([r["id"] for r in register[-2:]], ["MT222", "MT223"])
        earlier = [r for r in register if r["id"] not in {"MT222", "MT223"}]
        self.assertFalse(any(r.get("batches") and set(json.loads(r["batches"])) & {"cvt4", "cvt5"} for r in earlier))
        self.assertFalse(any("CORRECTIONS 240" in r.get("sources", "") or "CORRECTIONS 241" in r.get("sources", "") for r in earlier))

if __name__ == "__main__":
    unittest.main()
