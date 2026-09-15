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


PORTAL = Path(__file__).resolve().parents[1]
WORKSPACE = PORTAL.parents[1]
PUBLIC = PORTAL / "public"
PRIVATE = re.compile(r"/Users/|/home/|/scratch/|/data1/|teshnizi|salehkaleybars|s5014158|hmkhd2|100\.120\.248\.20|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|\b(?:p-cfer-\d+|node\d{3}|nodelogin\d+|login\d+|login\.[A-Za-z0-9_.…-]+)\b", re.I)


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

    def test_complete_register_and_area_counts(self):
        data = self.research()
        original = csv_rows(WORKSPACE / "outputs/tables/complete_experiment_register.csv")
        self.assertEqual(len(data["experiments"]), 111)
        self.assertEqual({e["id"] for e in data["experiments"]}, {e["id"] for e in original})
        self.assertEqual(len(data["areas"]), 9)
        self.assertEqual(sum(a["count"] for a in data["areas"]), 111)
        self.assertEqual(data["meta"]["stats"], {"experiments": 111, "runs": 2863, "figures": 54, "areas": 9})

    def test_run_inventory_keeps_all_jobs_with_unique_identifiers(self):
        runs = self.runs()
        self.assertEqual(len(runs), 2863)
        self.assertEqual(len({r["id"] for r in runs}), 2863)
        original = csv_rows(WORKSPACE / "outputs/tables/complete_run_inventory.csv")
        self.assertEqual({r["jobId"] for r in runs}, {r["job_id"] for r in original})
        self.assertEqual({r["account"] for r in runs}, {"Account1", "Account2"})
        self.assertTrue(all(r["parameters"].get("logAvailability") for r in runs))

    def test_all_54_figures_preserve_register_pages(self):
        figures = self.research()["figures"]
        self.assertEqual({f["id"] for f in figures}, {f"page-{n}" for n in range(1, 55)})
        self.assertEqual(sum(f["kind"] == "register" for f in figures), 30)
        for figure in figures:
            self.assertTrue((PUBLIC / figure["href"].lstrip("/")).is_file(), figure["id"])

    def test_all_raw_logs_are_exact_sanitized_archives_with_hashes(self):
        runs = self.runs()
        self.assertEqual(sum(bool(run.get("logHref")) for run in runs), 2863)
        exporter = self.exporter()
        evidence = exporter.read_evidence(WORKSPACE)
        archived = exporter.archive_log_index(evidence)
        latest = {str(run["job_id"]): run for run in evidence["cvk2_evidence"]["runs"]}
        audit = json.loads((WORKSPACE / "work/portal_data_audit.json").read_text())
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

    def test_every_csv_is_published_without_row_or_column_truncation(self):
        data = self.research()
        originals = list((WORKSPACE / "outputs/tables").rglob("*.csv"))
        self.assertEqual(len(originals), 201)
        tables_by_href = {row["href"]: row for row in data["tables"]}
        for original in originals:
            rel = original.relative_to(WORKSPACE / "outputs/tables")
            href = "/assets/tables/" + rel.as_posix()
            self.assertIn(href, tables_by_href, str(rel))
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

    def test_public_text_has_no_private_paths_or_accounts(self):
        self.research()
        for path in list((PUBLIC / "data").glob("*.json")) + list((PUBLIC / "assets").rglob("*.csv")) + list((PUBLIC / "assets").rglob("*.txt")):
            self.assertIsNone(PRIVATE.search(path.read_text()), str(path.relative_to(PUBLIC)))

    def test_zip_contains_all_charts_and_tables_and_sanitized_text(self):
        data = self.research()
        path = PUBLIC / data["meta"]["downloads"]["bundle"].lstrip("/")
        self.assertTrue(path.is_file())
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            self.assertEqual(len([n for n in names if n.startswith("figures/")]), 54)
            self.assertEqual(len([n for n in names if n.startswith("tables/") and n.endswith(".csv")]), 201)
            self.assertEqual(len([n for n in names if n.startswith("logs/")]), 2863)
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
        self.assertEqual(len(phases), 8)
        self.assertTrue(all("Documented phase" in event["detail"] for event in phases))
        self.assertTrue(any(e["kind"] == "completed-experiment" and e["date"] == "2026-09-14" for e in data["activity"]))
        self.assertTrue(any(e["kind"] == "publication-snapshot" and e["date"] == "2026-09-15" for e in data["activity"]))
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


if __name__ == "__main__":
    unittest.main()
