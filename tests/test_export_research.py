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
RUNS = 3437
LANDED_BATCH_RUNS = {"cgn1": 6, "cpl1": 15, "cvh1": 12, "cuc1": 30, "cgn2": 15, "cpl2": 15, "cvt1": 15, "cgn3": 12, "cvt3": 15, "cvt2": 21, "cvt4": 18, "cvt5": 12, "cvt6": 21, "cvt7": 15, "cvt8": 21, "cvt9": 21, "cmo1": 27, "cst1": 9, "cct1": 6, "cmg1": 12, "cvt10": 30, "cwd1": 9, "csv1": 18, "cwd2": 15, "cwd3": 15, "cwd4": 21, "cwd5": 27, "caw2": 27, "cgw1": 40, "crt1": 36, "csh1": 18}
PUBLIC = PORTAL / "public"
PARTITION_IDS = [f"MT{line}" for line in range(175, 212)]
APPENDED_IDS = [f"MT{line}" for line in range(212, 218)]
LANDED_IDS = ["MT218"]
CGN3_IDS = ["MT219"]
CVT23_IDS = ["MT220", "MT221"]
CVT45_IDS = ["MT222", "MT223"]
CVT67_IDS = ["MT224", "MT225"]
CVT89_IDS = ["MT226", "MT227"]
MUST_IDS = ["MT228", "MT229", "MT230", "MT231"]
MECH4_IDS = ["MT232", "MT233", "MT234", "MT235"]
MECH5_IDS = ["MT236"]
MECH6_IDS = ["MT237", "MT238"]
MECH7_IDS = ["MT239", "MT240"]
MECH8_IDS = ["MT241", "MT242"]
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

    def test_ref_cells_split_into_citations_on_semicolons_outside_brackets_only(self):
        # MASTER-TABLE rows 224-225 (campaign commit dae2a49) put a ';' inside the parentheses of their ref cell. Splitting there
        # published 'Cited research record: CORRECTIONS 242 (registration and launch' and dropped the rest of the cell.
        split = self.model().split_references
        row224 = "CORRECTIONS 242 (registration and launch; RULE H's '12 runs', 242.14 and 242.3(2) corrected in place at 246), 246 (landing)"
        self.assertEqual(split(row224), [row224])
        self.assertEqual(split("CORRECTIONS 212 (registration), 216 (launch); `analysis/x.py` [a; b]; FINDINGS 3"),
                         ["CORRECTIONS 212 (registration), 216 (launch)", "`analysis/x.py` [a; b]", "FINDINGS 3"])
        self.assertEqual(split(" a ; ; b "), ["a", "b"])
        # An unbalanced cell is not guessed at: it stays one citation rather than being cut inside a bracket.
        self.assertEqual(split("CORRECTIONS 9 (open; never closed"), ["CORRECTIONS 9 (open; never closed"])

    def test_published_source_labels_have_balanced_brackets(self):
        def unbalanced(text):
            pairs, stack = {")": "(", "]": "[", "}": "{"}, []
            for char in text:
                if char in "([{":
                    stack.append(char)
                elif char in pairs and (not stack or stack.pop() != pairs[char]):
                    return True
            return bool(stack)
        research = self.research()
        links = json.loads((PUBLIC / "data/source-links.json").read_text())
        labels = [(e["id"], ref["label"]) for e in research["experiments"] for ref in e["sourceRefs"]]
        labels += [(eid, ref["label"]) for eid, entry in links.items() for ref in entry["sourceRefs"]]
        self.assertGreater(len(labels), 2000)
        self.assertEqual([item for item in labels if unbalanced(item[1])], [])

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

    def test_published_text_carries_no_markdown_bold_or_code_marks(self):
        # The campaign's register export keeps MASTER-TABLE cells as Markdown, so 39 of its 111 rows reached the notebook
        # with literal ** and backticks in their titles and scopes (and in the verdict warnings built from them), and the
        # workspace's panel indexes did the same to 80 table scopes. Hrefs are file paths (a__b.csv), not text.
        def strings(value, path):
            if isinstance(value, dict):
                for key, item in value.items():
                    if not key.lower().endswith("href"):
                        yield from strings(item, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    yield from strings(item, f"{path}[{index}]")
            elif isinstance(value, str):
                yield path, value
        for name, value in [("research.json", self.research()), ("runs.json", self.runs())]:
            hits = [(path, text[max(0, m.start() - 30):m.end() + 12]) for path, text in strings(value, name) for m in re.finditer(r"\*\*|__|`", text)]
            self.assertEqual(hits, [], name)

    @needs(_WORKSPACE, _REPO)
    def test_base_register_rows_are_cleaned_like_master_table_rows(self):
        model = self.model()
        raw = {r["id"]: r for r in csv_rows(WORKSPACE / "outputs/tables/complete_experiment_register.csv")}
        loaded = {r["id"]: r for r in model.load_unamended_register(WORKSPACE, REPO) if r["id"] in raw}
        fields = ["goal", "comparison", "why", "result", "reason", "scope", "original_question"]
        changed = sorted(eid for eid, row in loaded.items() if any(row[f] != raw[eid][f] for f in fields))
        self.assertEqual(len(changed), 39)
        self.assertIn("MT033", changed)
        for eid, row in loaded.items():
            for field in fields:
                self.assertEqual(row[field], model.clean(raw[eid][field]), f"{eid}.{field}")
                self.assertIsNone(re.search(r"\*\*|`", row[field]), f"{eid}.{field}")
            self.assertEqual((row["id"], row["outcome"], row["kind"], row["corrected"]), (eid, model.remap_base_row(raw[eid])["outcome"], model.remap_base_row(raw[eid])["kind"], model.remap_base_row(raw[eid])["corrected"]))

    def test_complete_register_and_area_counts(self):
        data = self.research()
        self.assertEqual(len(data["experiments"]), 179)
        self.assertEqual(len({e["id"] for e in data["experiments"]}), 179)
        self.assertEqual(len(data["areas"]), 10)
        self.assertEqual(sum(a["count"] for a in data["areas"]), 179)
        self.assertEqual(data["meta"]["stats"], {"experiments": 179, "researchQuestions": 161, "methodChecks": 18, "runs": RUNS, "figures": 54, "areas": 10})

    @needs(_WORKSPACE)
    def test_register_ids_are_the_campaign_register_plus_the_imported_master_table_rows(self):
        data = self.research()
        original = csv_rows(WORKSPACE / "outputs/tables/complete_experiment_register.csv")
        self.assertEqual({e["id"] for e in data["experiments"]}, {e["id"] for e in original} | set(PARTITION_IDS) | set(APPENDED_IDS) | set(LANDED_IDS) | set(CGN3_IDS) | set(CVT23_IDS) | set(CVT45_IDS) | set(CVT67_IDS) | set(CVT89_IDS) | set(MUST_IDS) | set(MECH4_IDS) | set(MECH5_IDS) | set(MECH6_IDS) | set(MECH7_IDS) | set(MECH8_IDS))

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
        self.assertEqual(len(phases), 18)
        self.assertTrue(all("Documented phase" in event["detail"] for event in phases))
        self.assertTrue(any(e["kind"] == "completed-experiment" and e["date"] == "2026-09-14" for e in data["activity"]))
        self.assertTrue(any(e["kind"] == "publication-snapshot" and e["date"] == data["meta"]["asOf"] == "2026-09-22" for e in data["activity"]))
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
        self.assertEqual((phase["date"], phase["kind"], phase["experimentIds"]), ("2026-09-15", "research-phase", APPENDED_IDS + LANDED_IDS + CGN3_IDS + CVT23_IDS + CVT45_IDS + CVT67_IDS + CVT89_IDS))
        self.assertTrue(phase["detail"].startswith("Documented phase: 15-17 Sep 2026. Test: "))
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
                # cct1, cgw1 and crt1 are the landed batches on CIFAR-10 (MASTER-TABLE lines 230, 240 and 241); every other landed run is CIFAR-100.
                expected_dataset = "CIFAR10" if run["batch"] in ("cct1", "cgw1", "crt1") else "CIFAR100"
                self.assertEqual((run["account"], run["status"], run["dataset"]), ("Account2", "completed", expected_dataset), run["id"])
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
        self.assertEqual(sum(int(r["records"]) for r in areas), 179)
        self.assertEqual([sum(int(r[o]) for r in areas) for o in ["success", "fail", "mixed", "unresolved", "method_checks"]], [56, 41, 40, 24, 18])

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
        self.assertEqual((receipt["rows"], receipt["columns"], receipt.get("regenerated")), (179, 20, "scripts/register_model.py register"))

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
        # cmo1's 18 ARGS-value rows carry their own mark (argsDeviation), not the patch-intervention one.
        marked = {r["jobId"] for r in runs if "intervention" in r["parameters"]}
        deviating = {r["jobId"] for r in runs if "argsDeviation" in r["parameters"]}
        self.assertEqual(marked | deviating, set(intervened))
        self.assertEqual(marked & deviating, set())
        for eid, batch in [("MT220", "cvt3"), ("MT221", "cvt2")]:
            self.assertEqual(sorted(by_id[eid]["runIds"]), sorted(r["id"] for r in runs if r["batch"] == batch))
            self.assertIn(f"warning-{eid}-intervention", by_id[eid]["warningIds"])

    @needs(_WORKSPACE, _REPO)
    def test_cvt23_amendments_change_wording_only(self):
        model = self.model()
        self.assertEqual({eid: spec["number"] for eid, spec in model.CVT23_AMENDMENTS.items()}, {"MT216": 234, "MT213": 236})
        before = {r["id"]: r for r in model.apply_cgn3_amendments(model.apply_row_amendments(model.load_unamended_register(WORKSPACE, REPO), REPO), REPO)}
        # load_register applies CORRECTIONS 244 after these; test_c244_amendments_change_wording_only covers that step.
        after = {r["id"]: r for r in model.apply_cvt23_amendments(list(before.values()), REPO)}
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
        # The list has since grown by cvt6 and cvt7 (test_cvt67_rows_come_from_the_pinned_landing_commit), then cvt8 and cvt9.
        self.assertEqual(sorted(r["batch"] for r in intervened.values() if r["batch"] not in {"cvt6", "cvt7", "cvt8", "cvt9", "cmo1", "cvt10", "cwd1", "csv1", "cwd2", "cwd3", "cwd4", "cwd5", "caw2", "cgw1", "crt1", "csh1"}), sorted(["cvt1"] * 9 + ["cvt3"] * 9 + ["cvt2"] * 15 + ["cvt4"] * 12 + ["cvt5"] * 6))
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
        # cmo1's 18 ARGS-value rows carry their own mark (argsDeviation), not the patch-intervention one.
        marked = {r["jobId"] for r in runs if "intervention" in r["parameters"]}
        deviating = {r["jobId"] for r in runs if "argsDeviation" in r["parameters"]}
        self.assertEqual(marked | deviating, set(intervened))
        self.assertEqual(marked & deviating, set())
        for eid, batch in [("MT222", "cvt4"), ("MT223", "cvt5")]:
            self.assertEqual(sorted(by_id[eid]["runIds"]), sorted(r["id"] for r in runs if r["batch"] == batch))
            self.assertIn(f"warning-{eid}-intervention", by_id[eid]["warningIds"])

    @needs(_WORKSPACE, _REPO)
    def test_cvt45_import_leaves_every_earlier_record_unchanged(self):
        model = self.model()
        register = model.load_register(WORKSPACE, REPO)
        # MT224-MT227 were appended after them.
        self.assertEqual([r["id"] for r in register[-21:-19]], ["MT222", "MT223"])
        earlier = [r for r in register if r["id"] not in {"MT222", "MT223"}]
        self.assertFalse(any(r.get("batches") and set(json.loads(r["batches"])) & {"cvt4", "cvt5"} for r in earlier))
        self.assertFalse(any("CORRECTIONS 240" in r.get("sources", "") or "CORRECTIONS 241" in r.get("sources", "") for r in earlier))

    # ---- CORRECTIONS 244 (campaign commit 0ade9cc): in-place bracketed amendments of rows 222 and 223, and line 5 ----
    @needs(_REPO)
    def test_c244_master_table_differs_from_the_cvt45_pin_only_by_inserted_brackets(self):
        model = self.model()
        lines, before = model.c244_master_table(REPO), model.cvt45_master_table(REPO)
        self.assertEqual(len(lines), 223)
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {5, 222, 223})
        for line in (222, 223):
            old, new = model.table_cells(before[line - 1]), model.table_cells(lines[line - 1])
            self.assertEqual([i for i in range(7) if old[i] != new[i]], [5], line)
            self.assertEqual(model.strip_inserted_brackets(new[5], "CORRECTIONS 244"), old[5], line)
        with self.assertRaisesRegex(ValueError, "does not match the pinned CORRECTIONS 244 bytes"):
            pinned = model.C244_COMMIT
            try:
                model.C244_COMMIT = model.CVT45_COMMIT
                model.c244_master_table(REPO)
            finally:
                model.C244_COMMIT = pinned

    @needs(_WORKSPACE, _REPO)
    def test_c244_amendments_change_wording_only(self):
        model = self.model()
        self.assertEqual({eid: spec["line"] for eid, spec in model.C244_AMENDMENTS.items()}, {"MT222": 222, "MT223": 223})
        before = {r["id"]: r for r in model.apply_cvt23_amendments(model.apply_cgn3_amendments(model.apply_row_amendments(model.load_unamended_register(WORKSPACE, REPO), REPO), REPO), REPO)}
        # load_register applies CORRECTIONS 253.15 (row 227) after this step; test_cvt89_amendment_changes_mt227_wording_only covers it.
        after = {r["id"]: r for r in model.apply_c244_amendments(list(before.values()), REPO)}
        self.assertEqual(set(before), set(after))
        for eid, row in after.items():
            if eid not in model.C244_AMENDMENTS:
                self.assertEqual(row, before[eid], eid)
                continue
            changed = {key for key in row if row[key] != before[eid].get(key)}
            self.assertEqual(changed, {"scope", "sources", "further_amendments"}, eid)
            self.assertEqual(row["outcome"], before[eid]["outcome"], eid)
            self.assertEqual(model.strip_inserted_brackets(row["scope"], "CORRECTIONS 244"), before[eid]["scope"], eid)
        self.assertIn("read as SUFFICIENCY at this cell", after["MT222"]["scope"])
        self.assertIn("not a measured absence of coupling", after["MT222"]["scope"])
        self.assertIn("equilibrium of the LEVEL only", after["MT223"]["scope"])
        drifted = [dict(r, outcome="mixed") if r["id"] == "MT222" else r for r in before.values()]
        with self.assertRaisesRegex(ValueError, "outcome drifted before its CORRECTIONS 244 amendment"):
            model.apply_c244_amendments(drifted, REPO)


    # ---- MASTER-TABLE lines 224-225 (campaign commit dae2a49, CORRECTIONS 246-247: cvt6, cvt7); no row amended ----
    def test_intervention_kinds_name_every_hold_in_the_exclusion_list(self):
        # The exclusion list's intervention column is free text: one PATCH=value per hold, space-separated. cvt6's forced
        # arms carry two (BETA_HOLD and COMP_HOLD, CORRECTIONS 245's MULTI_KIND); cvt7 holds a whole group (GROUP_HOLD).
        model = self.model()
        kinds = model.intervention_kinds
        self.assertEqual(kinds("VOTE_W=layer4.1.bn2.weight:0"), [("VOTE_W", "layer4.1.bn2.weight:0")])
        self.assertEqual(kinds("BETA_HOLD=layer4.1.bn2.weight:floor COMP_HOLD=rec:cvt6_headpath"),
                         [("BETA_HOLD", "layer4.1.bn2.weight:floor"), ("COMP_HOLD", "rec:cvt6_headpath")])
        self.assertEqual(kinds("GROUP_HOLD=layer4.0.bn2.weight+layer4.0.shortcut.1.weight+layer4.1.bn2.weight:tri:8609"),
                         [("GROUP_HOLD", "layer4.0.bn2.weight+layer4.0.shortcut.1.weight+layer4.1.bn2.weight:tri:8609")])
        phrase = model.intervention_phrase
        # Single-kind wording is the wording the 51 earlier runs were published with.
        self.assertEqual(phrase("VOTE_W=x:0"), "vote-weight intervention")
        self.assertEqual(phrase("BETA_HOLD=x:floor"), "step-size hold intervention")
        self.assertEqual(phrase("GROUP_HOLD=a+b:floor"), "group step-size hold intervention")
        self.assertEqual(phrase("BETA_HOLD=x:tri:9428 COMP_HOLD=rec:cvt6_headpath"), "step-size hold and complement step-size hold interventions")
        for bad in ["", "SOMETHING=x", "BETA_HOLD=x BETA_HOLD=y", "BETA_HOLD x"]:
            with self.assertRaises(ValueError, msg=bad):
                kinds(bad)

    def test_a_second_hold_is_witnessed_by_exactly_one_matching_line_of_the_run_log(self):
        model = self.model()
        rec = "COMP_HOLD: on type=blockwise group=0 groupsize=52 mode=rec id=cvt6_headpath sha256=74be71fa knots=500 n0=2"
        tri = "COMP_HOLD: on type=blockwise group=0 groupsize=52 mode=tri P=9428 b0=-13.8 ms=0.001"
        lines = ["VOTE_W: off", "BETA_HOLD: on type=blockwise group=1 groupsize=1 name=layer4.1.bn2.weight mode=floor value=-15.0", rec]
        self.assertEqual(model.additional_witness(lines, "COMP_HOLD", "rec:cvt6_headpath"), rec)
        self.assertEqual(model.additional_witness(["x", tri], "COMP_HOLD", "tri:9428"), tri)
        with self.assertRaisesRegex(ValueError, "does not match"):
            model.additional_witness(lines, "COMP_HOLD", "tri:9428")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.additional_witness(["COMP_HOLD: off"], "COMP_HOLD", "tri:9428")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.additional_witness([tri, tri], "COMP_HOLD", "tri:9428")

    @needs(_REPO)
    def test_cvt67_rows_come_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.cvt67_master_table(REPO), model.c244_master_table(REPO)
        self.assertEqual((len(before), len(lines)), (223, 225))
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3, 5})
        # Line 5 feeds no record; it may only have gained the CORRECTIONS 246 and 247 brackets.
        self.assertEqual(model.strip_inserted_brackets(model.strip_inserted_brackets(lines[4], "CORRECTIONS 247"), "CORRECTIONS 246"), before[4])
        rows = model.cvt67_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"]), r["corrected"]) for r in rows],
                         [("MT224", "9", "mixed", ["cvt6"], "mixed", ["page-19"], ""), ("MT225", "9", "success", ["cvt7"], "met", ["page-19"], "")])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: GRADED + HARNESS-CLEAN + PATCH-BITES + HOLD-FROM-INIT + MUTEPATH-MAX-RATE-TRIANGLE + "))
        self.assertIn("+ FLOOR-READINGS-ARE-BOUNDS + TRAIN-AGREES Mixed: GRADED: ", rows[0]["reason"])
        self.assertTrue(rows[1]["reason"].startswith("Verdict: TRANSFERS-GRADED + HARNESS-CLEAN + "))
        self.assertIn("+ HOLDISO-AT-ISO + TRAIN-AGREES Goal met: TRANSFERS-GRADED: ", rows[1]["reason"])
        self.assertEqual([json.loads(r["intervention"])["arms"] for r in rows],
                         [{"HOLDLOW": 3, "HOLDHIGH": 3, "HIGHHEADPATH": 3, "LOWMUTEPATH": 3, "LOWHEADPATH": 3}, {"HOLDLOW": 3, "HOLDHIGH": 3, "HOLDISO": 3}])
        self.assertEqual([[c["number"] for c in json.loads(r["registration_corrections"])] for r in rows], [[246], [247]])
        intervened = model.intervened_runs(REPO)
        # The list has since grown by cvt8 and cvt9 (test_cvt89_rows_come_from_the_pinned_landing_commit).
        self.assertEqual(sorted(r["batch"] for r in intervened.values() if r["batch"] not in {"cvt8", "cvt9", "cmo1", "cvt10", "cwd1", "csv1", "cwd2", "cwd3", "cwd4", "cwd5", "caw2", "cgw1", "crt1", "csh1"}),
                         sorted(["cvt1"] * 9 + ["cvt3"] * 9 + ["cvt2"] * 15 + ["cvt4"] * 12 + ["cvt5"] * 6 + ["cvt6"] * 15 + ["cvt7"] * 9))
        two_kind = sorted(r["arm"] for r in intervened.values() if r["batch"] == "cvt6" and len(model.intervention_kinds(r["intervention"])) == 2)
        self.assertEqual(two_kind, sorted(["HIGHHEADPATH", "LOWMUTEPATH", "LOWHEADPATH"] * 3))
        self.assertTrue(all(r["intervention"].startswith("GROUP_HOLD=") for r in intervened.values() if r["batch"] == "cvt7"))
        with self.assertRaisesRegex(ValueError, "does not match the pinned cvt6/cvt7-landing bytes"):
            pinned = model.CVT67_COMMIT
            try:
                model.CVT67_COMMIT = model.C244_COMMIT
                model.cvt67_master_table(REPO)
            finally:
                model.CVT67_COMMIT = pinned
        with self.assertRaisesRegex(ValueError, "exactly lines 224-225"):
            model.appended_rows(lines[:-1], model.CVT67_ROWS, 224, 225, model.CVT67_COMMIT)
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.CVT67_INTERVENTIONS_TSV_SHA256
            try:
                model.CVT67_INTERVENTIONS_TSV_SHA256 = model.CVT45_INTERVENTIONS_TSV_SHA256
                model.intervened_runs(REPO)
            finally:
                model.CVT67_INTERVENTIONS_TSV_SHA256 = pinned
        data, runs = self.research(), self.runs()
        by_id = {e["id"]: e for e in data["experiments"]}
        # cmo1's 18 ARGS-value rows carry their own mark (argsDeviation), not the patch-intervention one.
        marked = {r["jobId"] for r in runs if "intervention" in r["parameters"]}
        deviating = {r["jobId"] for r in runs if "argsDeviation" in r["parameters"]}
        self.assertEqual(marked | deviating, set(intervened))
        self.assertEqual(marked & deviating, set())
        for eid, batch in [("MT224", "cvt6"), ("MT225", "cvt7")]:
            self.assertEqual(sorted(by_id[eid]["runIds"]), sorted(r["id"] for r in runs if r["batch"] == batch))
            self.assertIn(f"warning-{eid}-intervention", by_id[eid]["warningIds"])

    @needs(_REPO)
    def test_cvt67_registration_corrections_are_bracket_insertions_only(self):
        model = self.model()
        lines, before = model.cvt67_corrections(REPO)
        # CORRECTIONS 246 and 247 were appended; entries 242 and 243 were corrected in place by inserted brackets only.
        self.assertEqual((len(before), len(lines)), (33059, 33417))
        changed = {n: model.strip_inserted_brackets(lines[n - 1], tag) for n, tag in model.CVT67_CORRECTED_LINES.items()}
        self.assertEqual(sorted(changed), [32557, 32588, 32589, 32592, 32715, 32775, 32834])
        self.assertTrue(all(changed[n] == before[n - 1] for n in changed))
        self.assertEqual([n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]], sorted(changed))
        with self.assertRaisesRegex(ValueError, "does not match the pinned"):
            pinned = model.CVT67_CORRECTIONS_SHA256
            try:
                model.CVT67_CORRECTIONS_SHA256 = model.CVT67_PRE_CORRECTIONS_SHA256
                model.cvt67_corrections(REPO)
            finally:
                model.CVT67_CORRECTIONS_SHA256 = pinned

    @needs(_WORKSPACE, _REPO)
    def test_cvt67_import_leaves_every_earlier_record_unchanged(self):
        model = self.model()
        register = model.load_register(WORKSPACE, REPO)
        # MT226 and MT227 (cvt8, cvt9) were appended after them (test_cvt89_import_leaves_every_earlier_record_unchanged).
        self.assertEqual([r["id"] for r in register[-19:-17]], ["MT224", "MT225"])
        earlier = [r for r in register if r["id"] not in {"MT224", "MT225", "MT226", "MT227"}]
        self.assertFalse(any(r.get("batches") and set(json.loads(r["batches"])) & {"cvt6", "cvt7"} for r in earlier))
        self.assertFalse(any("CORRECTIONS 246" in r.get("sources", "") or "CORRECTIONS 247" in r.get("sources", "") for r in earlier))
        self.assertFalse(any(r.get("registration_corrections") for r in earlier))


    # ---- MASTER-TABLE lines 226-227 (campaign commit ba01f54, CORRECTIONS 252-253: cvt8, cvt9); no row amended ----
    def test_rest_and_window_holds_are_known_kinds_and_three_kind_runs_name_every_hold(self):
        # cvt8's forced arms carry GROUP_HOLD + REST_HOLD (the 59-tensor rest forced onto ISOPATH); cvt9's held arms carry
        # BETA_HOLD + COMP_HOLD, and EARLY / LATE add WINDOW_HOLD, a third hold (CORRECTIONS 251's MULTI_KIND).
        model = self.model()
        kinds = model.intervention_kinds
        self.assertEqual(kinds("GROUP_HOLD=a+b+c:tri:8609 REST_HOLD=rec:cvt8_isopath"), [("GROUP_HOLD", "a+b+c:tri:8609"), ("REST_HOLD", "rec:cvt8_isopath")])
        self.assertEqual(kinds("BETA_HOLD=layer4.1.bn2.weight:tri:9428 COMP_HOLD=rec:cvt6_headpath WINDOW_HOLD=0:9429"),
                         [("BETA_HOLD", "layer4.1.bn2.weight:tri:9428"), ("COMP_HOLD", "rec:cvt6_headpath"), ("WINDOW_HOLD", "0:9429")])
        phrase = model.intervention_phrase
        self.assertEqual(phrase("GROUP_HOLD=a+b:tri:9428 REST_HOLD=rec:cvt8_isopath"), "group step-size hold and rest-group step-size hold interventions")
        self.assertEqual(phrase("BETA_HOLD=x:tri:9428 COMP_HOLD=rec:cvt6_headpath WINDOW_HOLD=9429:end"),
                         "step-size hold, complement step-size hold and update-window hold interventions")
        # The two-kind wording published for cvt6 does not move.
        self.assertEqual(phrase("BETA_HOLD=x:tri:9428 COMP_HOLD=rec:cvt6_headpath"), "step-size hold and complement step-size hold interventions")
        for bad in ["WINDOW_HOLD", "BETA_HOLD=x:floor WINDOW_HOLD=0:1 WINDOW_HOLD=0:2"]:
            with self.assertRaises(ValueError, msg=bad):
                kinds(bad)

    def test_rest_and_window_holds_are_witnessed_by_their_own_log_lines(self):
        model = self.model()
        rest = ("REST_HOLD: on type=blockwise group=0 groupsize=59 mode=rec id=cvt8_isopath sha256=08ab25f3a329cb260bb39fb72f3299c021e7169bf612fa27d539166296e28e70 "
                "knots=500 n0=2 n1=49902 b0=-13.815510749816895 lo=-15.0 hi=-2.3026 vmax=-4.985378742218018 vlast=-14.924964427947998")
        early = "WINDOW_HOLD: on type=blockwise group=1 name=layer4.1.bn2.weight base=tri P=9428 n0=0 n1=9429 outside=floor value=-15.0"
        late = "WINDOW_HOLD: on type=blockwise group=1 name=layer4.1.bn2.weight base=tri P=9428 n0=9429 n1=end outside=floor value=-15.0"
        self.assertEqual(model.additional_witness(["GROUP_HOLD: on x", rest], "REST_HOLD", "rec:cvt8_isopath"), rest)
        self.assertEqual(model.additional_witness(["COMP_HOLD: on y", early], "WINDOW_HOLD", "0:9429"), early)
        self.assertEqual(model.additional_witness([late], "WINDOW_HOLD", "9429:end"), late)
        with self.assertRaisesRegex(ValueError, "does not match"):
            model.additional_witness([early], "WINDOW_HOLD", "9429:end")
        with self.assertRaisesRegex(ValueError, "does not match"):
            model.additional_witness([early], "WINDOW_HOLD", "0:942")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.additional_witness(["WINDOW_HOLD: off"], "WINDOW_HOLD", "0:9429")
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            model.additional_witness([early], "WINDOW_HOLD", "tri:9428")
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            model.additional_witness([rest], "REST_HOLD", "0:9429")

    @needs(_REPO)
    def test_cvt89_rows_come_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.cvt89_master_table(REPO), model.cvt67_master_table(REPO)
        self.assertEqual((len(before), len(lines)), (225, 227))
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3, 5})
        # Line 5 feeds no record; it may only have gained the CORRECTIONS 252 and 253 brackets.
        self.assertEqual(model.strip_inserted_brackets(model.strip_inserted_brackets(lines[4], "CORRECTIONS 253"), "CORRECTIONS 252"), before[4])
        rows = model.cvt89_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"]), r["corrected"]) for r in rows],
                         [("MT226", "9", "mixed", ["cvt8"], "mixed", ["page-19"], ""), ("MT227", "9", "mixed", ["cvt9"], "mixed", ["page-19"], "")])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: DOSE-FULL+ROUTE-PARTIAL+BIGROUTE-DIRECT + HARNESS-CLEAN + PATCH-BITES + "))
        self.assertIn("+ BIGISOPATH-BELOW-K01 + TRAIN-AGREES Mixed: DOSE-FULL+ROUTE-PARTIAL+BIGROUTE-DIRECT: ", rows[0]["reason"])
        self.assertTrue(rows[1]["reason"].startswith("Verdict: DOSE-GRADED + WINDOW-GRADED + HARNESS-CLEAN + "))
        self.assertIn("+ FLOOR-READINGS-ARE-BOUNDS + TRAIN-AGREES Mixed: DOSE-GRADED + WINDOW-GRADED: ", rows[1]["reason"])
        self.assertEqual([json.loads(r["intervention"])["arms"] for r in rows],
                         [{"HOLDHIGH": 3, "HOLDBIG": 3, "HIGHISOPATH": 3, "BIGISOPATH": 3, "LOWISOPATH": 3},
                          {"LOWHEADPATH": 3, "HIGHHEADPATH": 3, "MIDDOSE": 3, "RESDOSE": 3, "EARLY": 3, "LATE": 3}])
        # cvt8's landing corrected no registration text; cvt9's corrected 249.3 in place (253.9).
        self.assertEqual([[c["number"] for c in json.loads(r.get("registration_corrections") or "[]")] for r in rows], [[], [253]])
        intervened = model.intervened_runs(REPO)
        self.assertEqual(sorted(r["batch"] for r in intervened.values()),
                         sorted(["cvt1"] * 9 + ["cvt3"] * 9 + ["cvt2"] * 15 + ["cvt4"] * 12 + ["cvt5"] * 6 + ["cvt6"] * 15 + ["cvt7"] * 9 + ["cvt8"] * 15 + ["cvt9"] * 18 + ["cmo1"] * 18
                                + ["cvt10"] * 15 + ["cwd1"] * 6 + ["csv1"] * 12 + ["cwd2"] * 12 + ["cwd3"] * 12 + ["cwd4"] * 18 + ["cwd5"] * 21 + ["caw2"] * 18 + ["cgw1"] * 28 + ["crt1"] * 36 + ["csh1"] * 18))
        holds = {r["arm"] + "/" + r["batch"]: [patch for patch, _ in model.intervention_kinds(r["intervention"])] for r in intervened.values() if r["batch"] in {"cvt8", "cvt9"}}
        self.assertEqual(holds, {"HOLDHIGH/cvt8": ["GROUP_HOLD"], "HOLDBIG/cvt8": ["GROUP_HOLD"], "HIGHISOPATH/cvt8": ["GROUP_HOLD", "REST_HOLD"],
                                 "BIGISOPATH/cvt8": ["GROUP_HOLD", "REST_HOLD"], "LOWISOPATH/cvt8": ["GROUP_HOLD", "REST_HOLD"],
                                 "LOWHEADPATH/cvt9": ["BETA_HOLD", "COMP_HOLD"], "HIGHHEADPATH/cvt9": ["BETA_HOLD", "COMP_HOLD"],
                                 "MIDDOSE/cvt9": ["BETA_HOLD", "COMP_HOLD"], "RESDOSE/cvt9": ["BETA_HOLD", "COMP_HOLD"],
                                 "EARLY/cvt9": ["BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD"], "LATE/cvt9": ["BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD"]})
        with self.assertRaisesRegex(ValueError, "does not match the pinned cvt8/cvt9-landing bytes"):
            pinned = model.CVT89_COMMIT
            try:
                model.CVT89_COMMIT = model.CVT67_COMMIT
                model.cvt89_master_table(REPO)
            finally:
                model.CVT89_COMMIT = pinned
        with self.assertRaisesRegex(ValueError, "exactly lines 226-227"):
            model.appended_rows(lines[:-1], model.CVT89_ROWS, 226, 227, model.CVT89_COMMIT)
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.CVT89_INTERVENTIONS_TSV_SHA256
            try:
                model.CVT89_INTERVENTIONS_TSV_SHA256 = model.CVT67_INTERVENTIONS_TSV_SHA256
                model.intervened_runs(REPO)
            finally:
                model.CVT89_INTERVENTIONS_TSV_SHA256 = pinned
        data, runs = self.research(), self.runs()
        by_id = {e["id"]: e for e in data["experiments"]}
        # cmo1's 18 ARGS-value rows carry their own mark (argsDeviation), not the patch-intervention one.
        marked = {r["jobId"] for r in runs if "intervention" in r["parameters"]}
        deviating = {r["jobId"] for r in runs if "argsDeviation" in r["parameters"]}
        self.assertEqual(marked | deviating, set(intervened))
        self.assertEqual(marked & deviating, set())
        for eid, batch in [("MT226", "cvt8"), ("MT227", "cvt9")]:
            self.assertEqual(sorted(by_id[eid]["runIds"]), sorted(r["id"] for r in runs if r["batch"] == batch))
            self.assertIn(f"warning-{eid}-intervention", by_id[eid]["warningIds"])
        # Every further hold of a multi-kind run is witnessed by its own log line, in the listed order.
        for run in runs:
            if run["batch"] in {"cvt8", "cvt9"} and "intervention" in run["parameters"]:
                extra = model.intervention_kinds(run["parameters"]["intervention"].split(": ", 1)[1])[1:]
                witnesses = run["parameters"].get("interventionAdditionalWitness", "")
                self.assertEqual([w.split(":", 1)[0] for w in witnesses.split(" | ")] if witnesses else [], [patch for patch, _ in extra], run["id"])

    @needs(_REPO)
    def test_cvt89_registration_correction_is_one_bracket_insertion(self):
        model = self.model()
        lines, before = model.cvt89_corrections(REPO)
        # CORRECTIONS 252 and 253 were appended at the landing; 249.3 (LATE's integral) was corrected in place by one bracket.
        self.assertEqual((len(before), len(lines)), (33850, 34249))
        self.assertEqual([n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]], [33610])
        self.assertEqual(model.strip_inserted_brackets(lines[33609], "CORRECTIONS 253"), before[33609])
        self.assertIn("4.06480e7", lines[33609])
        # Every entry up to 247 is byte-identical to the cvt6/cvt7 pin.
        earlier, _ = model.cvt67_corrections(REPO)
        self.assertEqual(lines[:len(earlier)], earlier)
        with self.assertRaisesRegex(ValueError, "does not match the pinned"):
            pinned = model.CVT89_CORRECTIONS_SHA256
            try:
                model.CVT89_CORRECTIONS_SHA256 = model.CVT89_PRE_CORRECTIONS_SHA256
                model.cvt89_corrections(REPO)
            finally:
                model.CVT89_CORRECTIONS_SHA256 = pinned

    @needs(_WORKSPACE, _REPO)
    def test_cvt89_import_leaves_every_earlier_record_unchanged(self):
        model = self.model()
        register = model.load_register(WORKSPACE, REPO)
        self.assertEqual([r["id"] for r in register[-17:-15]], ["MT226", "MT227"])
        earlier = [r for r in register if r["id"] not in {"MT226", "MT227"}]
        self.assertFalse(any(r.get("batches") and set(json.loads(r["batches"])) & {"cvt8", "cvt9"} for r in earlier))
        self.assertFalse(any("CORRECTIONS 252" in r.get("sources", "") or "CORRECTIONS 253" in r.get("sources", "") for r in earlier))
        self.assertFalse(any("CORRECTIONS 253" in r.get("registration_corrections", "") for r in earlier))


    # ---- The final audit of the cvt8 / cvt9 landing (CORRECTIONS 253.15, campaign commit 3cf4201): wording only ----
    def test_mt226_reason_matches_the_bigroute_bound_and_counts_the_dose_accounts_exactly(self):
        model = self.model()
        reason = model.CVT89_ROWS[226][5]
        # CORRECTIONS 252.8 item 3: at PlainNet's dose a recovery under 5 pp cannot be seen, and that is not evidence that the
        # free complement plays no role; "none at PlainNet's dose" overstated it.
        self.assertNotIn("none at PlainNet's dose", reason)
        self.assertIn("at PlainNet's dose, with the complement forced, the carriers alone reproduce HOLDBIG's stall (BIGISOPATH 19.42, "
                      "P_ROUTE_BIG +0.05 pp), but both arms sit below k01, so a recovery under 5 pp could not be seen there and this is not "
                      "evidence that the free complement's collapse plays no role in HOLDBIG", reason)
        # CORRECTIONS 252.3: four DOSE-family accounts; three hit 6 of 7 and DOSE x VIA hits 5.
        self.assertNotIn("three DOSE accounts", reason)
        self.assertIn("of the four DOSE-family accounts, three hit 6 of 7 (DOSE x DIRECT misses HIGHISOPATH by 2.19 pp) and DOSE x VIA hits 5", reason)

    @needs(_REPO)
    def test_cvt89_audit_pin_allows_only_row_227s_correction_and_the_253_15_section(self):
        model = self.model()
        lines, landing = model.cvt89_audit_master_table(REPO), model.cvt89_master_table(REPO)
        self.assertEqual((len(landing), len(lines)), (227, 227))
        self.assertEqual([n for n in range(1, 228) if lines[n - 1] != landing[n - 1]], [227])
        spec = model.CVT89_AMENDMENTS["MT227"]
        old, new = model.table_cells(landing[226]), model.table_cells(lines[226])
        self.assertEqual([i for i in range(7) if old[i] != new[i]], [5])
        self.assertIn(f"SUPERSEDED wording, kept verbatim: '{spec['superseded']}']", new[5])
        self.assertEqual(model.strip_inserted_brackets(new[5], "CORRECTIONS 253.15").replace(spec["amended"], spec["superseded"], 1), old[5])
        # Anything beyond that one correction is refused: a further edit, a bracket that does not keep the old words, no bracket.
        for tampered in [lines[226].replace("(2) dose is the triangle family", "(2) dose is a triangle family"),
                         lines[226].replace("SUPERSEDED wording, kept verbatim: 'and DOSE-GRADED", "SUPERSEDED wording: 'and DOSE-GRADED"),
                         landing[226].replace(spec["superseded"], spec["amended"])]:
            with self.assertRaisesRegex(ValueError, "CORRECTIONS 253.15"):
                model.check_in_place_correction(landing[226], tampered, spec)
        with self.assertRaisesRegex(ValueError, "does not match the pinned"):
            pinned = model.CVT89_AUDIT_COMMIT
            try:
                model.CVT89_AUDIT_COMMIT = "6d09d1fa01d83039dbcf6f6bede32804a32f3c29"  # 253.15 appended, row 227 not yet corrected
                model.cvt89_audit_master_table(REPO)
            finally:
                model.CVT89_AUDIT_COMMIT = pinned
        corrections, earlier = model.cvt89_audit_corrections(REPO)
        self.assertEqual(earlier, model.cvt89_corrections(REPO)[0])
        # Every line of the landing pin is unchanged; only the 253.15 section, with its note, sits before the closing line.
        self.assertEqual(corrections[:len(earlier) - 1], earlier[:-1])
        self.assertEqual(corrections[-1], earlier[-1])
        inserted = corrections[len(earlier) - 1:-1]
        self.assertTrue(inserted[0].startswith("### 253.15 Addendum (final audit, zero GPU): "))
        self.assertFalse(any(line.startswith("#") for line in inserted[1:]))
        self.assertEqual(sum(line.startswith("*Note, added after this entry: MASTER-TABLE row 227 has since been amended in place.") for line in inserted), 1)
        with self.assertRaisesRegex(ValueError, "does not match the pinned"):
            pinned = model.CVT89_AUDIT_CORRECTIONS_SHA256
            try:
                model.CVT89_AUDIT_CORRECTIONS_SHA256 = model.CVT89_CORRECTIONS_SHA256
                model.cvt89_audit_corrections(REPO)
            finally:
                model.CVT89_AUDIT_CORRECTIONS_SHA256 = pinned

    @needs(_WORKSPACE, _REPO)
    def test_cvt89_amendment_changes_mt227_wording_only(self):
        model = self.model()
        self.assertEqual({eid: (spec["line"], spec["entry"]) for eid, spec in model.CVT89_AMENDMENTS.items()}, {"MT227": (227, "253.15")})
        before = {r["id"]: r for r in model.apply_c244_amendments(model.apply_cvt23_amendments(model.apply_cgn3_amendments(model.apply_row_amendments(model.load_unamended_register(WORKSPACE, REPO), REPO), REPO), REPO), REPO)}
        # Compared before the NEXT landing's amendment: CORRECTIONS 273 amends row 229 as well (test_mech4_amendment_...).
        after = {r["id"]: r for r in model.apply_cvt89_amendments(list(before.values()), REPO)}
        self.assertEqual(list(before), list(after))
        for eid, row in after.items():
            if eid != "MT227":
                self.assertEqual(row, before[eid], eid)
                continue
            self.assertEqual({key for key in row if row[key] != before[eid].get(key)}, {"scope", "sources", "further_amendments"})
            self.assertEqual((row["outcome"], row["corrected"], row["kind"]), (before[eid]["outcome"], before[eid]["corrected"], before[eid]["kind"]))
        spec = model.CVT89_AMENDMENTS["MT227"]
        # MT227's scope carries both single-arm flip routes; removing the bracket and restoring the old words gives the landing scope.
        self.assertIn("DOSE-GRADED changes on one arm only if RESDOSE rises, or MIDDOSE falls, by ~39 pp (DOSE-NONMONOTONE) "
                      "[CORRECTED IN PLACE at cycle 153, CORRECTIONS 253.15:", after["MT227"]["scope"])
        self.assertEqual(model.strip_inserted_brackets(after["MT227"]["scope"], "CORRECTIONS 253.15").replace(spec["amended"], spec["superseded"], 1),
                         before["MT227"]["scope"])
        further = json.loads(after["MT227"]["further_amendments"])
        self.assertEqual([(f["number"], f["line"], f["commit"], f["previousOutcome"], f["outcome"]) for f in further],
                         [(253, 227, model.CVT89_AUDIT_COMMIT, "mixed", "mixed")])
        drifted = [dict(r, outcome="success") if r["id"] == "MT227" else r for r in before.values()]
        with self.assertRaisesRegex(ValueError, "outcome drifted before its CORRECTIONS 253.15 amendment"):
            model.apply_cvt89_amendments(drifted, REPO)

    # ---- The four MUST-tier landings (CORRECTIONS 264-266, campaign commit 40d29cf): cmo1, cst1, cct1 and cmg1 ----
    def test_args_value_kinds_are_read_from_their_flag_and_witness(self):
        # CORRECTIONS 263: cmo1's M9 / W0 arms deviate from the standard cell in a base-optimiser CLI FLAG alone, which no
        # patch announces and no CSV column carries, so the exclusion row carries the flag and an ARGS-value witness.
        model = self.model()
        self.assertEqual(model.args_deviation("--momentum-param-base 0.9", "ARGS_MOMENTUM_BASE: momentum-param-base=0.9"),
                         ("ARGS_MOMENTUM_BASE", "momentum-param-base", "0.9", "0.99"))
        self.assertEqual(model.args_deviation("--weight-decay-base 0", "ARGS_WD_BASE: weight-decay-base=0"),
                         ("ARGS_WD_BASE", "weight-decay-base", "0", "0.1"))
        self.assertTrue(model.is_args_deviation("ARGS_WD_BASE: weight-decay-base=0"))
        self.assertFalse(model.is_args_deviation("VOTE_W: on type=scalar items=50:layer4.1.bn2.weight:w=0.0:group=0:groupsize=53"))
        self.assertEqual(model.args_phrase("ARGS_MOMENTUM_BASE"), "base momentum flag --momentum-param-base")
        self.assertEqual(model.args_phrase("ARGS_WD_BASE"), "base weight decay flag --weight-decay-base")
        # A patch witness is never read as an ARGS deviation, and vice versa.
        with self.assertRaisesRegex(ValueError, "Unknown ARGS-value witness"):
            model.args_deviation("--momentum-param-base 0.9", "GROUP_HOLD: on type=blockwise")
        with self.assertRaisesRegex(ValueError, "Unreadable ARGS-value witness"):
            model.args_deviation("--momentum-param-base 0.9", "ARGS_MOMENTUM_BASE: weight-decay-base=0.9")
        with self.assertRaisesRegex(ValueError, "Unreadable ARGS-value witness"):
            model.args_deviation("--momentum-param-base 0.9", "ARGS_MOMENTUM_BASE: momentum-param-base")
        # The intervention cell must be the flag exactly as it was written on the command line.
        with self.assertRaisesRegex(ValueError, "as written on the command line"):
            model.args_deviation("momentum-param-base=0.9", "ARGS_MOMENTUM_BASE: momentum-param-base=0.9")
        with self.assertRaisesRegex(ValueError, "as written on the command line"):
            model.args_deviation("--momentum-param-base 0.90", "ARGS_MOMENTUM_BASE: momentum-param-base=0.9")
        # The standard value is not a deviation, numerically compared, so 0.99 and 0.990 are both refused.
        for standard in ["0.99", "0.990"]:
            with self.assertRaisesRegex(ValueError, "lists the standard value"):
                model.args_deviation(f"--momentum-param-base {standard}", f"ARGS_MOMENTUM_BASE: momentum-param-base={standard}")
        with self.assertRaises(ValueError):
            model.intervention_kinds("--momentum-param-base 0.9")

    def test_args_value_rows_are_witnessed_by_the_runs_own_args_line(self):
        # The evidence is the run's OWN ARGS line, read with argparse last-wins semantics (analysis/argsline_guard.py's
        # rule, re-typed here rather than imported). The published witness quotes the checked flag alone, because the raw
        # line also carries the run's private save directory.
        model = self.model()
        args = ("ARGS: --optimizer HF --alg-base SGDm --momentum-param-base 0.9 --weight-decay-base 0.1 --alg-meta Lion "
                "--dataset CIFAR100 --NN-name ResNet18_c100 --seed 108 --run-name cmo1-M9k01-s108")
        self.assertEqual(model.args_effective(args[len("ARGS:"):])["--momentum-param-base"], "0.9")
        # A repeated flag takes its LAST value; --flag=value and a quoted value are read the same way.
        self.assertEqual(model.args_effective(" --weight-decay-base 0.1 --weight-decay-base 0")["--weight-decay-base"], "0")
        self.assertEqual(model.args_effective(" --weight-decay-base=0")["--weight-decay-base"], "0")
        self.assertEqual(model.args_effective(" --run-name 'cmo1 M9k01' --weight-decay-base 0")["--run-name"], "cmo1 M9k01")
        self.assertEqual(model.args_witness_line(["Loading", args], "momentum-param-base", "0.9", "0.99"),
                         "ARGS: --momentum-param-base 0.9")
        # 0.90 and 0.9 are one value; 0.8 is not the listed one.
        self.assertEqual(model.args_witness_line([args.replace("0.9 ", "0.90 ", 1)], "momentum-param-base", "0.9", "0.99"),
                         "ARGS: --momentum-param-base 0.90")
        with self.assertRaisesRegex(ValueError, "not the listed"):
            model.args_witness_line([args], "momentum-param-base", "0.8", "0.99")
        with self.assertRaisesRegex(ValueError, "not the listed"):
            model.args_witness_line([args], "weight-decay-base", "0", "0.1")
        with self.assertRaisesRegex(ValueError, "does not deviate"):
            model.args_witness_line([args], "momentum-param-base", "0.9", "0.9")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.args_witness_line(["Loading"], "momentum-param-base", "0.9", "0.99")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.args_witness_line([args, args], "momentum-param-base", "0.9", "0.99")

    @needs(_REPO)
    def test_must_rows_come_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.must_master_table(REPO), model.cvt89_audit_master_table(REPO)
        self.assertEqual((len(before), len(lines)), (227, 231))
        # Header line 3 alone: CORRECTIONS 266.12 deliberately left the bottom-line paragraph (line 5) unamended.
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3})
        rows = model.must_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"]), r["corrected"]) for r in rows],
                         [("MT228", "9", "mixed", ["cmo1"], "mixed", ["page-19"], ""), ("MT229", "9", "unresolved", ["cst1"], "open", ["page-19"], ""),
                          ("MT230", "9", "success", ["cct1"], "met", ["page-19"], ""), ("MT231", "9", "mixed", ["cmg1"], "mixed", ["page-19"], "")])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: M9:COLLAPSE-PERSISTS/ISO-RESCUES+W0:COLLAPSE-IS-CONFIG/ISO-UNREADABLE + HARNESS-CLEAN + "))
        self.assertIn("+ FLOOR-READINGS-ARE-BOUNDS + TRAIN-AGREES Mixed: M9:COLLAPSE-PERSISTS/ISO-RESCUES+W0:COLLAPSE-IS-CONFIG/ISO-UNREADABLE: ", rows[0]["reason"])
        self.assertIn("Open: UNRESOLVED-DECOMPOSITION: ", rows[1]["reason"])
        self.assertIn("Goal met: NOT-COLLAPSED+CARRIERS-DO-NOT-DOMINATE: ", rows[2]["reason"])
        self.assertIn("Mixed: NO-MERGE-HARMS: ", rows[3]["reason"])
        # Only cmo1 owes exclusion rows, and they are ARGS-value rows, not patch interventions.
        self.assertEqual([bool(r.get("intervention")) for r in rows], [False, False, False, False])
        self.assertEqual([json.loads(r["args_deviations"])["arms"] if r.get("args_deviations") else None for r in rows],
                         [{"M9k01": 3, "M9kL": 3, "M9ISO": 3, "W0k01": 3, "W0kL": 3, "W0ISO": 3}, None, None, None])
        # No registration text was corrected in place this cycle.
        self.assertEqual([json.loads(r.get("registration_corrections") or "[]") for r in rows], [[], [], [], []])
        with self.assertRaisesRegex(ValueError, "does not match the pinned MUST-tier-landing bytes"):
            pinned = model.MUST_COMMIT
            try:
                model.MUST_COMMIT = model.CVT89_AUDIT_COMMIT
                model.must_master_table(REPO)
            finally:
                model.MUST_COMMIT = pinned
        with self.assertRaisesRegex(ValueError, "exactly lines 228-231"):
            model.appended_rows(lines[:-1], model.MUST_ROWS, 228, 231, model.MUST_COMMIT)

    @needs(_REPO)
    def test_must_landing_appends_entries_264_to_266_below_the_ingest_trailer(self):
        model = self.model()
        lines, before = model.must_corrections(REPO)
        # docs/CORRECTIONS.md is append-only from the ingest: its closing "Next free number" line is the only one that goes.
        self.assertEqual((len(before), len(lines)), (35028, 35429))
        self.assertEqual(lines[:len(before) - 1], before[:-1])
        self.assertEqual(before[-1], "Next free number: **264**.")
        self.assertEqual(lines[-1], "Next free number: **267**.")
        self.assertEqual([line.split(".")[0] for line in lines[len(before) - 1:] if line.startswith("## ")], ["## 264", "## 265", "## 266"])
        earlier, _ = model.cvt89_audit_corrections(REPO)
        self.assertEqual(lines[:len(earlier)], earlier)
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MUST_CORRECTIONS_SHA256
            try:
                model.MUST_CORRECTIONS_SHA256 = model.MUST_INGEST_CORRECTIONS_SHA256
                model.must_corrections(REPO)
            finally:
                model.MUST_CORRECTIONS_SHA256 = pinned

    @needs(_REPO)
    def test_only_cmo1_owes_exclusion_rows_and_they_carry_the_new_args_kinds(self):
        model = self.model()
        intervened = model.intervened_runs(REPO)
        self.assertEqual(len(intervened), 322)
        # cwd5, caw2 and cgw1 later added 21, 18 and 28 ARGS-value rows of their own (CORRECTIONS 285, 296); of the MUST tier,
        # cmo1 still owns all 18.
        args_rows = [row for row in intervened.values() if row["argsDeviation"] and row["batch"] not in ("cwd5", "caw2", "cgw1", "crt1", "csh1")]
        self.assertEqual(len(args_rows), 18)
        self.assertEqual({row["batch"] for row in args_rows}, {"cmo1"})
        self.assertEqual({row["batch"] for row in intervened.values() if row["argsDeviation"]}, {"cmo1", "cwd5", "caw2", "cgw1", "crt1", "csh1"})
        # cst1, cct1 and cmg1 own no exclusion row at all (CORRECTIONS 265.1, 266.1).
        self.assertFalse({row["batch"] for row in intervened.values()} & {"cst1", "cct1", "cmg1"})
        kinds = {row["arm"]: model.args_deviation(row["intervention"], row["witness"])[:3] for row in args_rows}
        self.assertEqual(kinds, {"M9k01": ("ARGS_MOMENTUM_BASE", "momentum-param-base", "0.9"),
                                 "M9kL": ("ARGS_MOMENTUM_BASE", "momentum-param-base", "0.9"),
                                 "M9ISO": ("ARGS_MOMENTUM_BASE", "momentum-param-base", "0.9"),
                                 "W0k01": ("ARGS_WD_BASE", "weight-decay-base", "0"),
                                 "W0kL": ("ARGS_WD_BASE", "weight-decay-base", "0"),
                                 "W0ISO": ("ARGS_WD_BASE", "weight-decay-base", "0")})
        # The patch interventions keep their own mark and are untouched by the new kind: the 108 landed before this batch,
        # 45 more with the cvt10 / cwd1 / csv1 / cwd2 landing, 12 with cwd3 and 18 with cwd4.
        self.assertEqual(sum(not row["argsDeviation"] for row in intervened.values()), 183)
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MUST_INTERVENTIONS_TSV_SHA256
            try:
                model.MUST_INTERVENTIONS_TSV_SHA256 = model.CVT89_INTERVENTIONS_TSV_SHA256
                model.intervened_runs(REPO)
            finally:
                model.MUST_INTERVENTIONS_TSV_SHA256 = pinned

    @needs(_WORKSPACE, _REPO)
    def test_must_import_leaves_every_earlier_record_unchanged(self):
        model = self.model()
        register = model.load_register(WORKSPACE, REPO)
        self.assertEqual([r["id"] for r in register[-15:-11]], MUST_IDS)
        earlier = [r for r in register if r["id"] not in set(MUST_IDS)]
        self.assertFalse(any(r.get("batches") and set(json.loads(r["batches"])) & {"cmo1", "cst1", "cct1", "cmg1"} for r in earlier))
        for number in ["CORRECTIONS 264", "CORRECTIONS 265", "CORRECTIONS 266"]:
            self.assertFalse(any(number in r.get("sources", "") for r in earlier), number)
        # cwd5, caw2 and cgw1 later added ARGS-value rows of their own (MT238-MT240); no record of or before the MUST tier gains any.
        self.assertFalse(any(r.get("args_deviations") for r in earlier if r["id"] not in set(MECH6_IDS) | set(MECH7_IDS) | set(MECH8_IDS)))

    # ---- MASTER-TABLE lines 232-235 (campaign commit 2972d48, CORRECTIONS 270-273: cvt10, cwd1, csv1, cwd2); row 229 amended ----
    def test_decay_mask_and_shadow_vote_are_known_kinds_and_a_three_kind_row_names_every_hold(self):
        # CORRECTIONS 269 added two witness kinds: DECAY_MASK (cwd1's 20 BatchNorm scales, cwd2's single carrier scale) and
        # SHADOW_VOTE (csv1's applied-step / vote separation). cwd2's HIGHWD0 and LOWWD0 carry THREE kinds at once.
        model = self.model()
        kinds = model.intervention_kinds
        self.assertEqual(kinds("DECAY_MASK=normscale"), [("DECAY_MASK", "normscale")])
        self.assertEqual(kinds("SHADOW_VOTE=shadow:floor:layer4.1.bn2.weight"), [("SHADOW_VOTE", "shadow:floor:layer4.1.bn2.weight")])
        self.assertEqual(kinds("BETA_HOLD=layer4.1.bn2.weight:tri:9428 COMP_HOLD=rec:cvt6_headpath DECAY_MASK=layer4.1.bn2.weight"),
                         [("BETA_HOLD", "layer4.1.bn2.weight:tri:9428"), ("COMP_HOLD", "rec:cvt6_headpath"), ("DECAY_MASK", "layer4.1.bn2.weight")])
        phrase = model.intervention_phrase
        self.assertEqual(phrase("DECAY_MASK=normscale"), "coupled weight-decay mask intervention")
        self.assertEqual(phrase("SHADOW_VOTE=shadow:shared:layer4.1.bn2.weight"), "shadow-vote intervention")
        self.assertEqual(phrase("BETA_HOLD=x:tri:9428 COMP_HOLD=rec:cvt6_headpath DECAY_MASK=x"),
                         "step-size hold, complement step-size hold and coupled weight-decay mask interventions")
        # Every wording published before this landing is unchanged.
        self.assertEqual(phrase("VOTE_W=x:0"), "vote-weight intervention")
        self.assertEqual(phrase("BETA_HOLD=x:tri:9428 COMP_HOLD=rec:cvt6_headpath"), "step-size hold and complement step-size hold interventions")
        for bad in ["DECAY_MASK", "DECAY_MASK=a DECAY_MASK=b", "SHADOW_VOTE"]:
            with self.assertRaises(ValueError, msg=bad):
                kinds(bad)

    def test_decay_mask_and_shadow_vote_holds_are_witnessed_by_their_own_log_lines(self):
        # The exclusion list carries the FIRST hold's witness only; a further hold must print exactly one matching ON line
        # in the run's own log. cwd2's three-kind arms are witnessed by their BETA_HOLD line, so DECAY_MASK is read here.
        model = self.model()
        mask = ("DECAY_MASK: on base=SGDm wd=0.1 spec=layer4.1.bn2.weight masked=1 of=53 numel=512 idx=50 names=layer4.1.bn2.weight")
        norm = ("DECAY_MASK: on base=SGDm wd=0.1 spec=normscale masked=20 of=62 numel=4800 idx=2,5 names=bn1.weight,layer1.0.bn1.weight")
        shadow = ("SHADOW_VOTE: on type=scalar base=SGDm vote=shadow applied=floor floor=-15.0 items=50:layer4.1.bn2.weight:numel=512")
        inert = ("SHADOW_VOTE: on type=scalar base=SGDm vote=shadow applied=shared floor=na items=50:layer4.1.bn2.weight:numel=512")
        self.assertEqual(model.additional_witness(["BETA_HOLD: on x", "COMP_HOLD: on y", mask], "DECAY_MASK", "layer4.1.bn2.weight"), mask)
        self.assertEqual(model.additional_witness([norm], "DECAY_MASK", "normscale"), norm)
        self.assertEqual(model.additional_witness([shadow], "SHADOW_VOTE", "shadow:floor:layer4.1.bn2.weight"), shadow)
        self.assertEqual(model.additional_witness([inert], "SHADOW_VOTE", "shadow:shared:layer4.1.bn2.weight"), inert)
        # A mask on another tensor set, another vote mode, another applied mode or another tensor is not this row's witness.
        with self.assertRaisesRegex(ValueError, "does not match"):
            model.additional_witness([mask], "DECAY_MASK", "normscale")
        with self.assertRaisesRegex(ValueError, "does not match"):
            model.additional_witness([shadow], "SHADOW_VOTE", "natural:floor:layer4.1.bn2.weight")
        with self.assertRaisesRegex(ValueError, "does not match"):
            model.additional_witness([shadow], "SHADOW_VOTE", "shadow:shared:layer4.1.bn2.weight")
        with self.assertRaisesRegex(ValueError, "does not match"):
            model.additional_witness([shadow], "SHADOW_VOTE", "shadow:floor:layer4.1.bn1.weight")
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            model.additional_witness([shadow], "SHADOW_VOTE", "shadow:floor")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model.additional_witness(["DECAY_MASK: off"], "DECAY_MASK", "normscale")
        # The kinds published before this landing keep their own value grammar: a DECAY_MASK or SHADOW_VOTE value
        # written against one of them is refused rather than guessed.
        comp = "COMP_HOLD: on type=blockwise group=0 groupsize=52 mode=rec id=cvt6_headpath knots=500"
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            model.additional_witness([comp], "COMP_HOLD", "normscale")
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            model.additional_witness([comp], "COMP_HOLD", "shadow:floor:layer4.1.bn2.weight")
        self.assertEqual(model.additional_witness([comp], "COMP_HOLD", "rec:cvt6_headpath"), comp)

    @needs(_REPO)
    def test_mech4_rows_come_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.mech4_master_table(REPO), model.must_master_table(REPO)
        self.assertEqual((len(before), len(lines)), (231, 235))
        # Header line 3 and row 229, which CORRECTIONS 273 amended in place; no line moved.
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3, 229})
        rows = model.mech4_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"]), r["corrected"]) for r in rows],
                         [("MT232", "9", "success", ["cvt10"], "met", ["page-19"], ""), ("MT233", "9", "success", ["cwd1"], "met", ["page-19"], ""),
                          ("MT234", "9", "success", ["csv1"], "met", ["page-19"], ""), ("MT235", "9", "success", ["cwd2"], "met", ["page-19"], "")])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: ONE-SUFFICES+ONE50-STALLS+ONE59-STALLS+ONE53-STALLS+SPLIT-NO-EFFECT + HARNESS-CLEAN + "))
        # The campaign writes its field count inside the last verdict cell, so it travels with the last token.
        self.assertIn("+ FLOOR-READINGS-ARE-BOUNDS + TRAIN-AGREES (one branch token of five words + 22 stamps) Goal met: ONE-SUFFICES+", rows[0]["reason"])
        self.assertIn("+ TRAIN-AGREES (one branch token + 12 stamps) Goal met: COLLAPSE-VANISHES: ", rows[1]["reason"])
        self.assertIn("+ TRAIN-AGREES (one branch token + 21 stamps) Goal met: BOTH-ROUTES: ", rows[2]["reason"])
        # cwd2 is the one row of the cycle whose FINAL carries two branch tokens.
        self.assertTrue(rows[3]["reason"].startswith("Verdict: WD-ROUTE + SCALAR-NEEDS-CARRIER-WD + HARNESS-CLEAN + "))
        self.assertIn("+ TRAIN-AGREES (two branch tokens + 14 stamps) Goal met: WD-ROUTE + SCALAR-NEEDS-CARRIER-WD: ", rows[3]["reason"])
        # All four own exclusion rows, and none of them is an ARGS-value row.
        self.assertEqual([bool(r.get("intervention")) for r in rows], [True, True, True, True])
        self.assertEqual([bool(r.get("args_deviations")) for r in rows], [False, False, False, False])
        self.assertEqual([json.loads(r.get("registration_corrections") or "[]") for r in rows], [[], [], [], []])
        with self.assertRaisesRegex(ValueError, "does not match the pinned cvt10/cwd1/csv1/cwd2-landing bytes"):
            pinned = model.MECH4_COMMIT
            try:
                model.MECH4_COMMIT = model.MUST_COMMIT
                model.mech4_master_table(REPO)
            finally:
                model.MECH4_COMMIT = pinned

    @needs(_REPO)
    def test_mech4_landing_appends_entries_270_to_273_below_the_ingest_trailer(self):
        model = self.model()
        lines, before = model.mech4_corrections(REPO)
        # docs/CORRECTIONS.md is append-only from the MUST-tier pin through the ingest to this landing; only the closing
        # "Next free number" line is ever replaced.
        self.assertEqual((len(before), len(lines)), (35785, 36229))
        self.assertEqual(lines[:len(before) - 1], before[:-1])
        self.assertEqual(before[-1], "Next free number: **270**.")
        self.assertEqual(lines[-1], "Next free number: **274**.")
        self.assertEqual([line.split(".")[0] for line in lines[len(before) - 1:] if line.startswith("## ")], ["## 270", "## 271", "## 272", "## 273"])
        # The ingest itself added 267, 268 and 269 below the MUST-tier landing's trailer.
        earlier, _ = model.must_corrections(REPO)
        self.assertEqual(before[:len(earlier) - 1], earlier[:-1])
        self.assertEqual([line.split(".")[0] for line in before[len(earlier) - 1:] if line.startswith("## ")], ["## 267", "## 268", "## 269"])
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MECH4_CORRECTIONS_SHA256
            try:
                model.MECH4_CORRECTIONS_SHA256 = model.MECH4_INGEST_CORRECTIONS_SHA256
                model.mech4_corrections(REPO)
            finally:
                model.MECH4_CORRECTIONS_SHA256 = pinned

    @needs(_REPO)
    def test_the_four_batches_own_45_exclusion_rows_under_the_new_kinds(self):
        model = self.model()
        intervened = model.intervened_runs(REPO)
        self.assertEqual(len(intervened), 322)
        mine = [row for row in intervened.values() if row["batch"] in {"cvt10", "cwd1", "csv1", "cwd2"}]
        self.assertEqual(len(mine), 45)
        self.assertFalse(any(row["argsDeviation"] for row in mine))
        counts = {}
        for row in mine:
            counts.setdefault(row["batch"], {}).setdefault(row["arm"], 0)
            counts[row["batch"]][row["arm"]] += 1
        self.assertEqual(counts["cvt10"], {"HOLDBIG3": 3, "ONE50BIG": 3, "ONE59BIG": 3, "ONE53BIG": 3, "ISOSPLIT": 3})
        self.assertEqual(counts["cwd1"], {"k01NWD": 3, "kLNWD": 3})
        self.assertEqual(counts["csv1"], {"INERT": 1, "SHADOWLOW": 4, "NAIVELOW": 4, "MUTE": 3})
        self.assertEqual(counts["cwd2"], {"k01WD0": 3, "HIGHHEADPATH": 3, "HIGHWD0": 3, "LOWWD0": 3})
        # Nine cwd2 rows are multi-kind and six of those carry three kinds; csv1's MUTE rows keep the existing VOTE_W kind.
        multi = [row for row in mine if len(model.intervention_kinds(row["intervention"])) > 1]
        self.assertEqual(len(multi), 9)
        self.assertEqual(sorted(row["arm"] for row in multi if len(model.intervention_kinds(row["intervention"])) == 3),
                         ["HIGHWD0"] * 3 + ["LOWWD0"] * 3)
        self.assertEqual({row["arm"] for row in mine if row["intervention"].startswith("VOTE_W=")}, {"MUTE"})
        self.assertEqual({row["arm"] for row in mine if row["intervention"].startswith("SHADOW_VOTE=")}, {"INERT", "SHADOWLOW", "NAIVELOW"})
        # The 126 rows of the earlier landings keep their marks and their count; cwd3 later added 12 of its own, cwd4 18
        # and cwd5 21. Of the ARGS-value rows, the 18 that existed at this landing are still cmo1's alone.
        self.assertEqual(sum(row["argsDeviation"] and row["batch"] not in ("cwd5", "caw2", "cgw1", "crt1", "csh1") for row in intervened.values()), 18)
        self.assertEqual(sum(row["batch"] not in {"cvt10", "cwd1", "csv1", "cwd2", "cwd3", "cwd4", "cwd5", "caw2", "cgw1", "crt1", "csh1"} for row in intervened.values()), 126)
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MECH4_INTERVENTIONS_TSV_SHA256
            try:
                model.MECH4_INTERVENTIONS_TSV_SHA256 = model.MUST_INTERVENTIONS_TSV_SHA256
                model.intervened_runs(REPO)
            finally:
                model.MECH4_INTERVENTIONS_TSV_SHA256 = pinned

    @needs(_WORKSPACE, _REPO)
    def test_mech4_amendment_moves_mt229_from_open_to_mixed_without_a_corrected_badge(self):
        model = self.model()
        before = {r["id"]: r for r in model.apply_cvt89_amendments(model.apply_c244_amendments(model.apply_cvt23_amendments(
            model.apply_cgn3_amendments(model.apply_row_amendments(model.load_unamended_register(WORKSPACE, REPO), REPO), REPO), REPO), REPO), REPO)}
        self.assertEqual(before["MT229"]["outcome"], "unresolved")
        after = {r["id"]: r for r in model.load_register(WORKSPACE, REPO)}
        row = after["MT229"]
        # The registered predecessor is UNEDITED and still frozen; a frozen SUCCESSOR, registered before it was read,
        # reaches a branch, so the clauses now split between met and missed -- Mixed, not Goal met, and no Corrected badge.
        self.assertEqual((row["outcome"], row["corrected"]), ("mixed", ""))
        self.assertTrue(row["reason"].startswith("Verdict: NOMINATION-PARTIAL+ISO-RESCUES+CTL-NULL + FLOOR-READINGS-ARE-BOUNDS + "))
        self.assertIn("+ SIGMA-PRIOR-FROZEN Mixed: NOMINATION-PARTIAL+ISO-RESCUES+CTL-NULL: ", row["reason"])
        self.assertIn("MISSING BY 41", row["reason"])
        further = json.loads(row["further_amendments"])
        self.assertEqual([(a["number"], a["previousOutcome"], a["outcome"]) for a in further], [(273, "unresolved", "mixed")])
        self.assertIn("docs/MASTER-TABLE.md line 229 at 2972d48; CORRECTIONS 273", further[0]["source"])
        # The superseded predecessor verdict is kept verbatim in the record, and the amended cells are the row's own.
        self.assertIn("UNRESOLVED-DECOMPOSITION", row["reason"])
        self.assertIn("a GATE, not a branch. The batch has NO verdict and NO licence", row["reason"])
        self.assertIn("THE ISOLATION RESCUE TRANSFERS TO A SECOND META STEP SIZE", row["scope"])
        self.assertIn("[AMENDED at cycle 155, CORRECTIONS 273", row["result"])
        # No other record moves, and MT229 is the only row this landing amended.
        moved = [i for i, r in after.items() if before.get(i) and before[i]["outcome"] != r["outcome"]]
        self.assertEqual(moved, ["MT229"])

    @needs(_WORKSPACE, _REPO)
    def test_mech4_import_leaves_every_earlier_record_unchanged(self):
        model = self.model()
        register = model.load_register(WORKSPACE, REPO)
        self.assertEqual([r["id"] for r in register[-11:-7]], MECH4_IDS)
        earlier = [r for r in register if r["id"] not in set(MECH4_IDS) and r["id"] != "MT229"]
        self.assertFalse(any(r.get("batches") and set(json.loads(r["batches"])) & {"cvt10", "cwd1", "csv1", "cwd2"} for r in earlier))
        for number in ["CORRECTIONS 270", "CORRECTIONS 271", "CORRECTIONS 272", "CORRECTIONS 273"]:
            self.assertFalse(any(number in r.get("sources", "") for r in earlier), number)

    # ---- CORRECTIONS 278 (campaign commit 97eb049): MASTER-TABLE line 236, cwd3 ----

    @needs(_REPO)
    def test_mech5_rows_come_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.mech5_master_table(REPO), model.mech4_master_table(REPO)
        self.assertEqual((len(before), len(lines)), (235, 236))
        # Header line 3 only: this landing amended NO earlier row, and did not touch line 5 either.
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3})
        rows = model.mech5_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"]), r["corrected"]) for r in rows],
                         [("MT236", "9", "success", ["cwd3"], "met", ["page-19"], "")])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: CARRIER-DECAY-SUFFICES + CAR-REC+CTL-NULL+CTL2-NULL + HARNESS-CLEAN + "))
        # cwd3's FINAL carries TWO branch tokens, so both are repeated after the rule; the field count travels with the last token.
        self.assertIn("+ HW-UNIFORM-NVIDIA_L4 + TRAIN-AGREES (two branch tokens + 15 stamps) Goal met: CARRIER-DECAY-SUFFICES + CAR-REC+CTL-NULL+CTL2-NULL: ", rows[0]["reason"])
        # It owns exclusion rows, none of them an ARGS-value row, and it corrected no registration text.
        self.assertEqual([bool(r.get("intervention")) for r in rows], [True])
        self.assertEqual([bool(r.get("args_deviations")) for r in rows], [False])
        self.assertEqual([json.loads(r.get("registration_corrections") or "[]") for r in rows], [[]])
        with self.assertRaisesRegex(ValueError, "does not match the pinned cwd3-landing bytes"):
            pinned = model.MECH5_COMMIT
            try:
                model.MECH5_COMMIT = model.MECH4_COMMIT
                model.mech5_master_table(REPO)
            finally:
                model.MECH5_COMMIT = pinned

    @needs(_REPO)
    def test_mech5_landing_appends_entry_278_below_the_previous_trailer(self):
        model = self.model()
        lines, before = model.mech5_corrections(REPO)
        # docs/CORRECTIONS.md is append-only from the cvt10/cwd1/csv1/cwd2 pin through the write-up revisions to this
        # landing; only the closing "Next free number" line is ever replaced.
        self.assertEqual((len(before), len(lines)), (36489, 36632))
        self.assertEqual(lines[:len(before) - 1], before[:-1])
        self.assertEqual(before[-1], "Next free number: **278**.")
        self.assertEqual(lines[-1], "Next free number: **279**.")
        self.assertEqual([line.split(".")[0] for line in lines[len(before) - 1:] if line.startswith("## ")], ["## 278"])
        # 274-277 were appended between the two landings, below the cvt10/cwd1/csv1/cwd2 landing's own trailer.
        earlier, _ = model.mech4_corrections(REPO)
        self.assertEqual(before[:len(earlier) - 1], earlier[:-1])
        self.assertEqual([line.split(".")[0] for line in before[len(earlier) - 1:] if line.startswith("## ")], ["## 274", "## 275", "## 276", "## 277"])
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MECH5_CORRECTIONS_SHA256
            try:
                model.MECH5_CORRECTIONS_SHA256 = model.MECH5_PREVIOUS_CORRECTIONS_SHA256
                model.mech5_corrections(REPO)
            finally:
                model.MECH5_CORRECTIONS_SHA256 = pinned

    @needs(_REPO)
    def test_cwd3_owns_12_decay_mask_rows_under_the_existing_kind_and_adds_no_multi_kind_row(self):
        model = self.model()
        intervened = model.intervened_runs(REPO)
        self.assertEqual(len(intervened), 322)
        mine = [row for row in intervened.values() if row["batch"] == "cwd3"]
        self.assertEqual(len(mine), 12)
        self.assertFalse(any(row["argsDeviation"] for row in mine))
        counts = {}
        for row in mine:
            counts[row["arm"]] = counts.get(row["arm"], 0) + 1
        self.assertEqual(counts, {"CARWD0": 3, "CTLWD0": 3, "CTL2WD0": 3, "NWD": 3})
        # ONE kind per run: the batch adds no multi-kind row and no new kind at all.
        self.assertTrue(all(row["intervention"].startswith("DECAY_MASK=") for row in mine))
        self.assertTrue(all(len(model.intervention_kinds(row["intervention"])) == 1 for row in mine))
        # Each arm's registered tensor set, read from the listed witness rather than from the arm name.
        masked = {row["arm"]: row["witness"].split(" masked=")[1].split(" ")[0] for row in mine}
        numels = {row["arm"]: row["witness"].split(" numel=")[1].split(" ")[0] for row in mine}
        self.assertEqual(masked, {"CARWD0": "3", "CTLWD0": "3", "CTL2WD0": "2", "NWD": "20"})
        self.assertEqual(numels, {"CARWD0": "1536", "CTLWD0": "1536", "CTL2WD0": "1024", "NWD": "4800"})
        # The 210 rows of the other landings keep their count; the three k01 runs of cwd3 own no row.
        self.assertEqual(sum(row["batch"] != "cwd3" for row in intervened.values()), 310)
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MECH5_INTERVENTIONS_TSV_SHA256
            try:
                model.MECH5_INTERVENTIONS_TSV_SHA256 = model.MECH4_INTERVENTIONS_TSV_SHA256
                model.intervened_runs(REPO)
            finally:
                model.MECH5_INTERVENTIONS_TSV_SHA256 = pinned

    @needs(_WORKSPACE, _REPO)
    def test_mech5_import_leaves_every_earlier_record_unchanged_and_moves_no_outcome(self):
        model = self.model()
        register = model.load_register(WORKSPACE, REPO)
        self.assertEqual([r["id"] for r in register[-7:-6]], MECH5_IDS)
        earlier = [r for r in register if r["id"] not in set(MECH5_IDS)]
        self.assertFalse(any(r.get("batches") and "cwd3" in set(json.loads(r["batches"])) for r in earlier))
        self.assertFalse(any("CORRECTIONS 278" in r.get("sources", "") for r in earlier))
        # The landing amends no row, so no earlier record gains a further amendment and no outcome moves.
        self.assertFalse(any(278 in [a["number"] for a in json.loads(r.get("further_amendments") or "[]")] for r in register))


    @needs(_REPO)
    def test_mech6_rows_come_from_the_pinned_landing_commits(self):
        model = self.model()
        lines, before = model.mech6_master_table(REPO), model.mech5_master_table(REPO)
        self.assertEqual((len(before), len(lines)), (236, 238))
        step = model.mech5_master_table(REPO)  # the cwd4 step is checked inside mech6_master_table; re-derive its shape here
        self.assertEqual(len(step), 236)
        # Two steps, each editing header line 3 alone and appending ONE row. Neither touched line 5 and no line moved.
        self.assertEqual(lines[:2], before[:2])
        self.assertNotEqual(lines[2], before[2])
        self.assertEqual(lines[3:236], before[3:236])
        rows = model.mech6_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"]), r["corrected"]) for r in rows],
                         [("MT237", "9", "mixed", ["cwd4"], "mixed", ["page-19"], ""),
                          ("MT238", "9", "mixed", ["cwd5"], "mixed", ["page-19"], "")])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: ONE-SUFFICES-PARTIAL + TWO-REC+CTL2-NULL+ONE50-REC+ONE53-PART+ONE59-REC + "))
        self.assertTrue(rows[1]["reason"].startswith("Verdict: THRESHOLD-W1-W2 + W1-COLLAPSE+W2-NOGAP+W3-NOGAP+W4-NOGAP+CAR-UNREADABLE + "))
        # Both FINALs carry TWO branch tokens, so both are repeated after the rule, and both rows are Mixed.
        self.assertIn("(two branch tokens + 19 stamps) Mixed: ONE-SUFFICES-PARTIAL + TWO-REC+CTL2-NULL+ONE50-REC+ONE53-PART+ONE59-REC: ", rows[0]["reason"])
        self.assertIn("(two branch tokens + 19 stamps) Mixed: THRESHOLD-W1-W2 + W1-COLLAPSE+W2-NOGAP+W3-NOGAP+W4-NOGAP+CAR-UNREADABLE: ", rows[1]["reason"])
        # cwd4 owns patch-intervention rows and no ARGS-value row; cwd5 is the other way round.
        self.assertEqual([bool(r.get("intervention")) for r in rows], [True, False])
        self.assertEqual([bool(r.get("args_deviations")) for r in rows], [False, True])
        self.assertEqual([json.loads(r.get("registration_corrections") or "[]") for r in rows], [[], []])
        with self.assertRaisesRegex(ValueError, "does not match the pinned cwd4/cwd5-landing bytes"):
            pinned = model.MECH6_COMMIT
            try:
                model.MECH6_COMMIT = model.MECH6_STEP_COMMIT
                model.mech6_master_table(REPO)
            finally:
                model.MECH6_COMMIT = pinned

    @needs(_REPO)
    def test_mech6_landings_append_entries_283_to_285_below_the_previous_trailer(self):
        model = self.model()
        lines, before = model.mech6_corrections(REPO)
        # docs/CORRECTIONS.md is append-only from the cwd3 landing through all four steps of this cycle; only the
        # closing "Next free number" line is ever replaced, and the two ingest commits touched no entry at all.
        self.assertEqual((len(before), len(lines)), (37509, 37985))
        self.assertEqual(lines[:len(before) - 1], before[:-1])
        self.assertEqual(before[-1], "Next free number: **285**.")
        self.assertEqual(lines[-1], "Next free number: **286**.")
        self.assertEqual([line.split(".")[0] for line in lines[len(before) - 1:] if line.startswith("## ")], ["## 285"])
        earlier, _ = model.mech5_corrections(REPO)
        self.assertEqual(len(earlier), 36632)
        self.assertEqual(lines[:len(earlier) - 1], earlier[:-1])
        # 279-282 between the two landings, then 283 (the cwd4 landing), 284 (the two-axis code gap) and 285 (cwd5),
        # in that textual order and in numeric order.
        appended = [line.split(".")[0] for line in lines[len(earlier) - 1:] if line.startswith("## ")]
        self.assertEqual(appended, ["## 279", "## 280", "## 281", "## 282", "## 283", "## 284", "## 285"])
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MECH6_CORRECTIONS_SHA256
            try:
                model.MECH6_CORRECTIONS_SHA256 = model.MECH6_STEP_CORRECTIONS_SHA256
                model.mech6_corrections(REPO)
            finally:
                model.MECH6_CORRECTIONS_SHA256 = pinned

    @needs(_REPO)
    def test_cwd4_owns_18_decay_mask_rows_and_cwd5_owns_21_args_rows(self):
        model = self.model()
        intervened = model.intervened_runs(REPO)
        self.assertEqual(len(intervened), 322)
        cwd4 = [row for row in intervened.values() if row["batch"] == "cwd4"]
        cwd5 = [row for row in intervened.values() if row["batch"] == "cwd5"]
        self.assertEqual((len(cwd4), len(cwd5)), (18, 21))
        # cwd4: one DECAY_MASK kind per run, no ARGS deviation, no multi-kind row.
        self.assertFalse(any(row["argsDeviation"] for row in cwd4))
        self.assertTrue(all(row["intervention"].startswith("DECAY_MASK=") for row in cwd4))
        self.assertTrue(all(len(model.intervention_kinds(row["intervention"])) == 1 for row in cwd4))
        counts = {}
        for row in cwd4:
            counts[row["arm"]] = counts.get(row["arm"], 0) + 1
        self.assertEqual(counts, {"CARWD0": 3, "TWOWD0": 3, "CTL2WD0": 3, "ONE50": 3, "ONE53": 3, "ONE59": 3})
        masked = {row["arm"]: row["witness"].split(" masked=")[1].split(" ")[0] for row in cwd4}
        numels = {row["arm"]: row["witness"].split(" numel=")[1].split(" ")[0] for row in cwd4}
        self.assertEqual(masked, {"CARWD0": "3", "TWOWD0": "2", "CTL2WD0": "2", "ONE50": "1", "ONE53": "1", "ONE59": "1"})
        self.assertEqual(numels, {"CARWD0": "1536", "TWOWD0": "1024", "CTL2WD0": "1024", "ONE50": "512", "ONE53": "512", "ONE59": "512"})
        # cwd5: every listed row is an ARGS-value deviation on the ONE varied flag, and the six anchor runs own none.
        self.assertTrue(all(row["argsDeviation"] for row in cwd5))
        self.assertTrue(all(row["witness"].startswith("ARGS_WD_BASE: weight-decay-base=") for row in cwd5))
        values = sorted(row["witness"].split("=")[1] for row in cwd5)
        self.assertEqual(values, ["1e-2"] * 9 + ["1e-3"] * 6 + ["5e-4"] * 6)
        self.assertFalse(any(row["arm"].endswith("W1") for row in cwd5))
        # Exactly three TWO-AXIS rows (CORRECTIONS 284), and they are CARW2's.
        two_axis = [row for row in cwd5 if model.args_extra_kinds(row["intervention"])]
        self.assertEqual(sorted(row["arm"] for row in two_axis), ["CARW2"] * 3)
        self.assertTrue(all(model.args_extra_kinds(row["intervention"]) ==
                            [("DECAY_MASK", "layer4.0.bn2.weight+layer4.0.shortcut.1.weight+layer4.1.bn2.weight")] for row in two_axis))
        # The 183 rows of the earlier landings keep their count.
        self.assertEqual(sum(row["batch"] not in ("cwd4", "cwd5", "caw2", "cgw1", "crt1", "csh1") for row in intervened.values()), 183)
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MECH6_INTERVENTIONS_TSV_SHA256
            try:
                model.MECH6_INTERVENTIONS_TSV_SHA256 = model.MECH6_CWD4_INTERVENTIONS_TSV_SHA256
                model.intervened_runs(REPO)
            finally:
                model.MECH6_INTERVENTIONS_TSV_SHA256 = pinned

    def test_a_two_axis_exclusion_row_is_read_on_both_axes_and_a_one_axis_row_is_unchanged(self):
        model = self.model()
        one_axis = "--weight-decay-base 1e-3"
        two_axis = "--weight-decay-base 1e-2 + DECAY_MASK=layer4.0.bn2.weight+layer4.1.bn2.weight"
        witness = "ARGS_WD_BASE: weight-decay-base=1e-2"
        # A one-axis row keeps exactly the behaviour CORRECTIONS 263 gave it.
        self.assertEqual(model.args_axes(one_axis), ("--weight-decay-base 1e-3", []))
        self.assertEqual(model.args_extra_kinds(one_axis), [])
        self.assertEqual(model.args_deviation(one_axis, "ARGS_WD_BASE: weight-decay-base=1e-3"),
                         ("ARGS_WD_BASE", "weight-decay-base", "1e-3", "0.1"))
        # A two-axis row keeps its ARGS reading AND names its patch axis; the patch spec's own "+" is not a separator.
        self.assertEqual(model.args_deviation(two_axis, witness), ("ARGS_WD_BASE", "weight-decay-base", "1e-2", "0.1"))
        self.assertEqual(model.args_extra_kinds(two_axis), [("DECAY_MASK", "layer4.0.bn2.weight+layer4.1.bn2.weight")])
        # An unknown patch axis, an empty axis and a flag clause that is not the leading axis all stop the export.
        with self.assertRaisesRegex(ValueError, "Unknown intervention"):
            model.args_extra_kinds("--weight-decay-base 1e-2 + NOT_A_PATCH=x")
        with self.assertRaisesRegex(ValueError, "Unreadable two-axis intervention"):
            model.args_axes("--weight-decay-base 1e-2 + ")
        with self.assertRaisesRegex(ValueError, "must open with the flag as written on the command line"):
            model.args_deviation("DECAY_MASK=a + --weight-decay-base 1e-2", witness)
        # The standard value is still refused as a deviation, on either axis count.
        with self.assertRaisesRegex(ValueError, "lists the standard value"):
            model.args_deviation("--weight-decay-base 0.1 + DECAY_MASK=a", "ARGS_WD_BASE: weight-decay-base=0.1")

    @needs(_WORKSPACE, _REPO)
    def test_mech6_import_leaves_every_earlier_record_unchanged_and_moves_no_outcome(self):
        model = self.model()
        register = model.load_register(WORKSPACE, REPO)
        self.assertEqual([r["id"] for r in register[-6:-4]], MECH6_IDS)
        earlier = [r for r in register if r["id"] not in set(MECH6_IDS)]
        self.assertFalse(any(r.get("batches") and {"cwd4", "cwd5"} & set(json.loads(r["batches"])) for r in earlier))
        self.assertFalse(any("CORRECTIONS 283" in r.get("sources", "") or "CORRECTIONS 285" in r.get("sources", "") for r in earlier))
        # Neither landing amends a row, so no earlier record gains a further amendment and no outcome moves.
        self.assertFalse(any({283, 285} & {a["number"] for a in json.loads(r.get("further_amendments") or "[]")} for r in register))

    # ---- CORRECTIONS 295-297 (campaign commit 8a99001): MASTER-TABLE lines 239-240, caw2 and cgw1 ----

    @needs(_REPO)
    def test_mech7_rows_come_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.mech7_master_table(REPO), model.mech6_master_table(REPO)
        self.assertEqual((len(before), len(lines)), (238, 240))
        # ONE step: header line 3 recounted and two rows appended; no earlier row amended, line 5 untouched, no line moved.
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3})
        rows = model.mech7_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["area"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"]), r["corrected"]) for r in rows],
                         [("MT239", "9", "Mechanism and isolation", "mixed", ["caw2"], "mixed", ["page-19"], ""),
                          ("MT240", "10", "Count-matched partition audit", "unresolved", ["cgw1"], "open", ["page-8"], "")])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: NO-COLLAPSE-CONTROL-DOSE-NOT-REACHED + M-COLLAPSE+L-NOGAP+A-NOGAP+X-NOGAP+K-COLLAPSED + ATTR-BASE-PROTECTS + GATE-DOES-NOT-FIRE + "))
        self.assertIn("(four branch words + 26 stamps) Mixed: NO-COLLAPSE-CONTROL-DOSE-NOT-REACHED: ", rows[0]["reason"])
        self.assertTrue(rows[1]["reason"].startswith("Verdict: AUDIT-UNDECIDED + SCALAR-BEATS-BEST + W1-SURVIVES+W2-SURVIVES+W4-UNDECIDED + "))
        self.assertIn("(three branch words + 7 stamps) Open: AUDIT-UNDECIDED: ", rows[1]["reason"])
        # Both own ARGS-value exclusion rows and no patch-intervention row, and neither corrected any registration text.
        self.assertEqual([bool(r.get("intervention")) for r in rows], [False, False])
        self.assertEqual([bool(r.get("args_deviations")) for r in rows], [True, True])
        self.assertEqual([json.loads(r.get("registration_corrections") or "[]") for r in rows], [[], []])
        with self.assertRaisesRegex(ValueError, "does not match the pinned caw2/cgw1-landing bytes"):
            pinned = model.MECH7_COMMIT
            try:
                model.MECH7_COMMIT = model.MECH6_COMMIT
                model.mech7_master_table(REPO)
            finally:
                model.MECH7_COMMIT = pinned

    @needs(_REPO)
    def test_mech7_landing_appends_entries_286_to_297_below_the_cwd5_trailer(self):
        model = self.model()
        lines, before = model.mech7_corrections(REPO)
        # docs/CORRECTIONS.md is append-only from the cwd5 landing's pin: every line above its closing trailer is unchanged,
        # and entries 286-297 follow it in numeric order (289-291 were inserted before 292 while those tracks ran, which the
        # final file shows in order). The file closes on a fresh "Next free number" trailer.
        self.assertEqual((len(before), len(lines)), (37985, 39485))
        self.assertEqual(lines[:len(before) - 1], before[:-1])
        self.assertEqual(before[-1], "Next free number: **286**.")
        self.assertEqual(lines[-1], "Next free number: **298**.")
        self.assertEqual([line.split(".")[0] for line in lines[len(before) - 1:] if line.startswith("## ")],
                         [f"## {n}" for n in range(286, 298)])
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MECH7_CORRECTIONS_SHA256
            try:
                model.MECH7_CORRECTIONS_SHA256 = model.MECH6_CORRECTIONS_SHA256
                model.mech7_corrections(REPO)
            finally:
                model.MECH7_CORRECTIONS_SHA256 = pinned

    @needs(_REPO)
    def test_caw2_owns_18_args_rows_six_of_them_two_args_and_cgw1_owns_28(self):
        model = self.model()
        intervened = model.intervened_runs(REPO)
        self.assertEqual(len(intervened), 322)
        caw2 = [row for row in intervened.values() if row["batch"] == "caw2"]
        cgw1 = [row for row in intervened.values() if row["batch"] == "cgw1"]
        self.assertEqual((len(caw2), len(cgw1)), (18, 28))
        self.assertTrue(all(row["argsDeviation"] for row in caw2 + cgw1))
        # caw2: the AdamW base's own beta1 0.9 against the per-flag standard 0.99, and at the dose arm ALSO wd 1.0.
        # The control and the meta swap (K01, MS, ML) run the standard 0.99 / 0.1 and own no row.
        counts = {}
        for row in caw2:
            counts[row["arm"]] = counts.get(row["arm"], 0) + 1
        self.assertEqual(counts, {"LS": 3, "LL": 3, "AS": 3, "AL": 3, "XS": 3, "XL": 3})
        one = [row for row in caw2 if row["arm"][0] != "X"]
        two = [row for row in caw2 if row["arm"][0] == "X"]
        self.assertTrue(all(row["witness"] == "ARGS_MOMENTUM_BASE: momentum-param-base=0.9" and not model.args_extra_args(row["intervention"], row["witness"]) for row in one))
        # The six TWO-ARGS rows (CORRECTIONS 294) are listed ONCE, by the dose arm's weight decay, with the momentum kind
        # read from the intervention cell and registered for exactly these two arms.
        self.assertTrue(all(row["witness"] == "ARGS_WD_BASE: weight-decay-base=1.0" for row in two))
        self.assertTrue(all(row["intervention"] == "--momentum-param-base 0.9 + --weight-decay-base 1.0" for row in two))
        self.assertTrue(all(model.args_extra_args(row["intervention"], row["witness"]) == [("ARGS_MOMENTUM_BASE", "momentum-param-base", "0.9")] for row in two))
        self.assertTrue(all(model.args_extra_kinds(row["intervention"]) == [] for row in caw2))
        self.assertEqual(sorted(model.MECH7_MULTI_ARGS), [("caw2", "XL"), ("caw2", "XS")])
        # cgw1: one ARGS kind per run, the rung's weight decay; the twelve W1 runs at the standard 0.1 own none.
        self.assertTrue(all(row["witness"].startswith("ARGS_WD_BASE: weight-decay-base=") for row in cgw1))
        self.assertEqual(sorted(row["witness"].split("=")[1] for row in cgw1), ["1e-2"] * 12 + ["5e-4"] * 16)
        self.assertFalse(any(row["arm"].endswith("W1") for row in cgw1))
        # The 222 rows of the earlier landings keep their count.
        self.assertEqual(sum(row["batch"] not in ("caw2", "cgw1", "crt1", "csh1") for row in intervened.values()), 222)
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MECH7_INTERVENTIONS_TSV_SHA256
            try:
                model.MECH7_INTERVENTIONS_TSV_SHA256 = model.MECH6_INTERVENTIONS_TSV_SHA256
                model.intervened_runs(REPO)
            finally:
                model.MECH7_INTERVENTIONS_TSV_SHA256 = pinned
        # A two-ARGS row outside the registry stops the export, as CORRECTIONS 294's per-arm escape does on the campaign side.
        registry = dict(model.MECH7_MULTI_ARGS)
        try:
            del model.MECH7_MULTI_ARGS[("caw2", "XL")]
            with self.assertRaisesRegex(ValueError, "unregistered TWO-ARGS row"):
                model.intervened_runs(REPO)
        finally:
            model.MECH7_MULTI_ARGS.clear()
            model.MECH7_MULTI_ARGS.update(registry)

    def test_a_two_args_row_is_read_on_both_args_axes_and_every_earlier_row_shape_is_unchanged(self):
        model = self.model()
        two_args = "--momentum-param-base 0.9 + --weight-decay-base 1.0"
        by_wd = "ARGS_WD_BASE: weight-decay-base=1.0"
        by_mom = "ARGS_MOMENTUM_BASE: momentum-param-base=0.9"
        # Listed by either kind, the row reads its witness's kind and names the OTHER ARGS kind as a further axis.
        self.assertEqual(model.args_deviation(two_args, by_wd), ("ARGS_WD_BASE", "weight-decay-base", "1.0", "0.1"))
        self.assertEqual(model.args_extra_args(two_args, by_wd), [("ARGS_MOMENTUM_BASE", "momentum-param-base", "0.9")])
        self.assertEqual(model.args_deviation(two_args, by_mom), ("ARGS_MOMENTUM_BASE", "momentum-param-base", "0.9", "0.99"))
        self.assertEqual(model.args_extra_args(two_args, by_mom), [("ARGS_WD_BASE", "weight-decay-base", "1.0")])
        self.assertEqual(model.args_extra_kinds(two_args), [])
        # One-axis rows and CORRECTIONS 284's ARGS + patch rows keep exactly their earlier reading.
        self.assertEqual(model.args_deviation("--weight-decay-base 1e-3", "ARGS_WD_BASE: weight-decay-base=1e-3"),
                         ("ARGS_WD_BASE", "weight-decay-base", "1e-3", "0.1"))
        self.assertEqual(model.args_extra_args("--weight-decay-base 1e-3", "ARGS_WD_BASE: weight-decay-base=1e-3"), [])
        patch_row = "--weight-decay-base 1e-2 + DECAY_MASK=layer4.0.bn2.weight+layer4.1.bn2.weight"
        self.assertEqual(model.args_extra_kinds(patch_row), [("DECAY_MASK", "layer4.0.bn2.weight+layer4.1.bn2.weight")])
        self.assertEqual(model.args_extra_args(patch_row, "ARGS_WD_BASE: weight-decay-base=1e-2"), [])
        # Malformed two-ARGS cells stop the export rather than being guessed.
        with self.assertRaisesRegex(ValueError, "does not carry the witness's flag"):
            model.args_deviation("--momentum-param-base 0.9 + --weight-decay-base 2.0", by_wd)
        with self.assertRaisesRegex(ValueError, "further ARGS axis lists the standard value"):
            model.args_deviation("--momentum-param-base 0.99 + --weight-decay-base 1.0", by_wd)
        with self.assertRaisesRegex(ValueError, "names one ARGS flag twice"):
            model.args_deviation("--weight-decay-base 0.5 + --weight-decay-base 1.0", by_wd)
        with self.assertRaisesRegex(ValueError, "is not a registered ARGS-value flag"):
            model.args_deviation("--momentum-param-meta 0.5 + --weight-decay-base 1.0", by_wd)
        with self.assertRaisesRegex(ValueError, "must come before every patch axis"):
            model.args_deviation("--weight-decay-base 1.0 + DECAY_MASK=a + --momentum-param-base 0.9", by_wd)

    @needs(_WORKSPACE, _REPO)
    def test_mech7_import_leaves_every_earlier_record_unchanged_and_moves_no_outcome(self):
        model = self.model()
        register = model.load_register(WORKSPACE, REPO)
        self.assertEqual([r["id"] for r in register[-4:-2]], MECH7_IDS)
        earlier = [r for r in register if r["id"] not in set(MECH7_IDS)]
        self.assertFalse(any(r.get("batches") and {"caw2", "cgw1"} & set(json.loads(r["batches"])) for r in earlier))
        self.assertFalse(any("CORRECTIONS 295" in r.get("sources", "") or "CORRECTIONS 296" in r.get("sources", "") for r in earlier))
        # Neither landing amends a row, so no earlier record gains a further amendment and no outcome moves.
        self.assertFalse(any({295, 296, 297} & {a["number"] for a in json.loads(r.get("further_amendments") or "[]")} for r in register))

    # ---- CORRECTIONS 309-310 (campaign commit a53bce1): MASTER-TABLE lines 241-242, crt1 and csh1 ----

    @needs(_REPO)
    def test_mech8_rows_come_from_the_pinned_landing_commit(self):
        model = self.model()
        lines, before = model.mech8_master_table(REPO), model.mech7_master_table(REPO)
        self.assertEqual((len(before), len(lines)), (240, 242))
        # ONE step: header line 3 recounted and two rows appended; no earlier row amended, line 5 untouched, no line moved.
        self.assertEqual({n for n in range(1, len(before) + 1) if lines[n - 1] != before[n - 1]}, {3})
        rows = model.mech8_rows(lines)
        self.assertEqual([(r["id"], r["section"], r["area"], r["outcome"], json.loads(r["batches"]), r["mapping_rule"], json.loads(r["figure_ids"]), r["corrected"]) for r in rows],
                         [("MT241", "10", "Count-matched partition audit", "mixed", ["crt1"], "mixed", ["page-8"], ""),
                          ("MT242", "9", "Mechanism and isolation", "mixed", ["csh1"], "mixed", ["page-19"], "")])
        self.assertTrue(rows[0]["reason"].startswith("Verdict: WEAKENED-TO-TIE + TUNED-SCALAR-TIES-BEST + ORACLE-SCALAR-TIES-BEST + M2-BEATS+A2-BEATS+M3-BEATS+M1-BEATS + "))
        self.assertIn("(branch + two readings + every-config + selections + 12 stamps) Mixed: WEAKENED-TO-TIE: ", rows[0]["reason"])
        self.assertTrue(rows[1]["reason"].startswith("Verdict: HORIZON-DOES-NOT-REPRODUCE + A-NOGAP+M-NOGAP+P-NOGAP + "))
        self.assertIn("(branch + cell states + 17 stamps; the 13 registered bounds in the next column) Mixed: HORIZON-DOES-NOT-REPRODUCE: ", rows[1]["reason"])
        # Both own ARGS-value exclusion rows and no patch-intervention row, and neither corrected any registration text.
        self.assertEqual([bool(r.get("intervention")) for r in rows], [False, False])
        self.assertEqual([bool(r.get("args_deviations")) for r in rows], [True, True])
        self.assertEqual([json.loads(r.get("registration_corrections") or "[]") for r in rows], [[], []])
        with self.assertRaisesRegex(ValueError, "does not match the pinned crt1/csh1-landing bytes"):
            pinned = model.MECH8_COMMIT
            try:
                model.MECH8_COMMIT = model.MECH7_COMMIT
                model.mech8_master_table(REPO)
            finally:
                model.MECH8_COMMIT = pinned

    @needs(_REPO)
    def test_mech8_landing_appends_entries_298_to_310_below_the_cgw1_trailer(self):
        model = self.model()
        lines, before = model.mech8_corrections(REPO)
        # docs/CORRECTIONS.md is append-only from the caw2 / cgw1 landing's pin: every line above its closing trailer is
        # unchanged, and entries 298-310 follow it in numeric order, closed by a fresh "Next free number" trailer.
        self.assertEqual((len(before), len(lines)), (39485, 41924))
        self.assertEqual(lines[:len(before) - 1], before[:-1])
        self.assertEqual(before[-1], "Next free number: **298**.")
        self.assertTrue(lines[-1].startswith("Next free number: **311**."))
        self.assertEqual([line.split(".")[0] for line in lines[len(before) - 1:] if line.startswith("## ")],
                         [f"## {n}" for n in range(298, 311)])
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MECH8_CORRECTIONS_SHA256
            try:
                model.MECH8_CORRECTIONS_SHA256 = model.MECH7_CORRECTIONS_SHA256
                model.mech8_corrections(REPO)
            finally:
                model.MECH8_CORRECTIONS_SHA256 = pinned

    @needs(_REPO)
    def test_crt1_owns_36_args_rows_and_csh1_owns_18(self):
        model = self.model()
        intervened = model.intervened_runs(REPO)
        self.assertEqual(len(intervened), 322)
        crt1 = [row for row in intervened.values() if row["batch"] == "crt1"]
        csh1 = [row for row in intervened.values() if row["batch"] == "csh1"]
        self.assertEqual((len(crt1), len(csh1)), (36, 18))
        self.assertTrue(all(row["argsDeviation"] for row in crt1 + csh1))
        # One ARGS kind per run: the base weight decay 5e-4 against the standard 0.1; meta step size, alpha0 and gamma are
        # CSV columns, so no run is two-ARGS or two-axis.
        self.assertTrue(all(row["witness"] == "ARGS_WD_BASE: weight-decay-base=5e-4" and row["intervention"] == "--weight-decay-base 5e-4" for row in crt1 + csh1))
        self.assertFalse(any(model.args_extra_args(row["intervention"], row["witness"]) or model.args_extra_kinds(row["intervention"]) for row in crt1 + csh1))
        counts = {}
        for row in crt1 + csh1:
            counts[row["arm"]] = counts.get(row["arm"], 0) + 1
        self.assertEqual(counts, {**{f"{g}{c}": 3 for c in ("M2", "A2", "M3", "M1") for g in ("ch", "nd", "k01")},
                                  **{arm: 3 for arm in ("G1S", "G1L", "GMS", "GML", "GPS", "GPL")}})
        # csh1's lookalike is the corpus base arm (the refute fix), registered at the landing entry.
        self.assertEqual({row["looks_like"] for row in csh1}, {"k01 (granularity scalar)", "kL (granularity layerwise)"})
        self.assertEqual({row["registered_at"] for row in crt1}, {"CORRECTIONS 309"})
        self.assertEqual({row["registered_at"] for row in csh1}, {"CORRECTIONS 310"})
        # The 268 rows of the earlier landings keep their count.
        self.assertEqual(sum(row["batch"] not in ("crt1", "csh1") for row in intervened.values()), 268)
        with self.assertRaisesRegex(ValueError, "does not match the pinned bytes"):
            pinned = model.MECH8_INTERVENTIONS_TSV_SHA256
            try:
                model.MECH8_INTERVENTIONS_TSV_SHA256 = model.MECH7_INTERVENTIONS_TSV_SHA256
                model.intervened_runs(REPO)
            finally:
                model.MECH8_INTERVENTIONS_TSV_SHA256 = pinned

    @needs(_WORKSPACE, _REPO)
    def test_mech8_import_leaves_every_earlier_record_unchanged_and_moves_no_outcome(self):
        model = self.model()
        register = model.load_register(WORKSPACE, REPO)
        self.assertEqual([r["id"] for r in register[-2:]], MECH8_IDS)
        earlier = [r for r in register if r["id"] not in set(MECH8_IDS)]
        self.assertFalse(any(r.get("batches") and {"crt1", "csh1"} & set(json.loads(r["batches"])) for r in earlier))
        self.assertFalse(any("CORRECTIONS 309" in r.get("sources", "") or "CORRECTIONS 310" in r.get("sources", "") for r in earlier))
        # Neither landing amends a row, so no earlier record gains a further amendment and no outcome moves; MT240 keeps Open.
        self.assertFalse(any(set(range(298, 311)) & {a["number"] for a in json.loads(r.get("further_amendments") or "[]")} for r in register))
        self.assertEqual(next(r for r in register if r["id"] == "MT240")["outcome"], "unresolved")


if __name__ == "__main__":
    unittest.main()
