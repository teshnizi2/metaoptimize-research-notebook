"""The data tests stay portable: inputs outside the repository skip their tests, never error."""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from external_inputs import external_input, inputs

TESTS = Path(__file__).resolve().parent
PORTAL = TESTS.parent
# The tests that read the maintainer's workspace, audit receipt, extended run inventory or
# research repository. Every other data test reads only the published snapshot.
WORKSPACE_OR_REPO_TESTS = {
    "test_export_research.ResearchExportTests.test_all_raw_logs_are_exact_sanitized_archives_with_hashes",
    "test_export_research.ResearchExportTests.test_appended_rows_come_from_the_pinned_master_table",
    "test_export_research.ResearchExportTests.test_approved_correction_split_is_explicit_and_complete",
    "test_export_research.ResearchExportTests.test_cgn3_amendments_change_wording_only",
    "test_export_research.ResearchExportTests.test_cgn3_row_comes_from_the_pinned_landing_commit",
    "test_export_research.ResearchExportTests.test_cvt23_amendments_change_wording_only",
    "test_export_research.ResearchExportTests.test_cvt23_rows_come_from_the_pinned_landing_commit",
    "test_export_research.ResearchExportTests.test_cvt45_import_leaves_every_earlier_record_unchanged",
    "test_export_research.ResearchExportTests.test_cvt67_import_leaves_every_earlier_record_unchanged",
    "test_export_research.ResearchExportTests.test_cvt67_registration_corrections_are_bracket_insertions_only",
    "test_export_research.ResearchExportTests.test_cvt67_rows_come_from_the_pinned_landing_commit",
    "test_export_research.ResearchExportTests.test_cvt89_import_leaves_every_earlier_record_unchanged",
    "test_export_research.ResearchExportTests.test_cvt89_registration_correction_is_one_bracket_insertion",
    "test_export_research.ResearchExportTests.test_cvt89_rows_come_from_the_pinned_landing_commit",
    "test_export_research.ResearchExportTests.test_cvt89_audit_pin_allows_only_row_227s_correction_and_the_253_15_section",
    "test_export_research.ResearchExportTests.test_cvt89_amendment_changes_mt227_wording_only",
    "test_export_research.ResearchExportTests.test_must_rows_come_from_the_pinned_landing_commit",
    "test_export_research.ResearchExportTests.test_mech4_rows_come_from_the_pinned_landing_commit",
    "test_export_research.ResearchExportTests.test_mech4_landing_appends_entries_270_to_273_below_the_ingest_trailer",
    "test_export_research.ResearchExportTests.test_the_four_batches_own_45_exclusion_rows_under_the_new_kinds",
    "test_export_research.ResearchExportTests.test_mech4_amendment_moves_mt229_from_open_to_mixed_without_a_corrected_badge",
    "test_export_research.ResearchExportTests.test_mech4_import_leaves_every_earlier_record_unchanged",
    "test_export_research.ResearchExportTests.test_must_landing_appends_entries_264_to_266_below_the_ingest_trailer",
    "test_export_research.ResearchExportTests.test_only_cmo1_owes_exclusion_rows_and_they_carry_the_new_args_kinds",
    "test_export_research.ResearchExportTests.test_must_import_leaves_every_earlier_record_unchanged",
    "test_export_research.ResearchExportTests.test_base_register_rows_are_cleaned_like_master_table_rows",
    "test_export_research.ResearchExportTests.test_c244_master_table_differs_from_the_cvt45_pin_only_by_inserted_brackets",
    "test_export_research.ResearchExportTests.test_c244_amendments_change_wording_only",
    "test_export_research.ResearchExportTests.test_cvt45_rows_come_from_the_pinned_landing_commit",
    "test_export_research.ResearchExportTests.test_every_csv_is_published_without_row_or_column_truncation",
    "test_export_research.ResearchExportTests.test_goal_outcome_table_keeps_the_report_goals_with_current_counts",
    "test_export_research.ResearchExportTests.test_landed_row_comes_from_the_pinned_landing_commit",
    "test_export_research.ResearchExportTests.test_partition_audit_rows_come_from_the_pinned_master_table",
    "test_export_research.ResearchExportTests.test_register_ids_are_the_campaign_register_plus_the_imported_master_table_rows",
    "test_export_research.ResearchExportTests.test_register_table_receipt_records_the_regeneration",
    "test_export_research.ResearchExportTests.test_row_amendments_come_from_the_pinned_bookkeeping_commit",
    "test_export_research.ResearchExportTests.test_run_inventory_keeps_all_jobs_with_unique_identifiers",
    "test_sources.SourceExportTests.test_cvk2_links_exact_registered_scorer_and_runtime_snapshot",
    "test_sources.SourceExportTests.test_every_record_has_valid_document_and_code_references",
    "test_sources.SourceExportTests.test_export_is_deterministic_and_every_href_resolves",
    "test_sources.SourceExportTests.test_public_content_has_no_private_accounts_home_paths_or_credentials",
}
LIST_SKIPS = """
import json, sys, unittest
suite = unittest.defaultTestLoader.loadTestsFromNames(["test_export_research", "test_sources"])
def walk(s):
    for t in s:
        yield from (walk(t) if isinstance(t, unittest.TestSuite) else [t])
out = {}
for t in walk(suite):
    method = getattr(t, t._testMethodName)
    out[t.id()] = getattr(method, "__unittest_skip_why__", None) if getattr(method, "__unittest_skip__", False) else None
print(json.dumps(out))
"""


class ExternalInputTests(unittest.TestCase):
    def test_an_input_runs_when_configured_or_present_and_skips_with_its_variable_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent = Path(tmp) / "absent"
            item = external_input("NOTEBOOK_WORKSPACE", absent, "the workspace", "outputs/x.csv", environ={})
            self.assertFalse(item.available)
            self.assertIn("set NOTEBOOK_WORKSPACE", item.reason)
            self.assertIn(str(absent / "outputs/x.csv"), item.reason)
            (absent / "outputs").mkdir(parents=True)
            (absent / "outputs/x.csv").write_text("id\n")
            self.assertTrue(external_input("NOTEBOOK_WORKSPACE", absent, "the workspace", "outputs/x.csv", environ={}).available)
            # A configured variable always runs the test, so a wrong path fails loudly instead of skipping.
            configured = external_input("NOTEBOOK_WORKSPACE", absent, "the workspace", "outputs/x.csv", environ={"NOTEBOOK_WORKSPACE": str(Path(tmp) / "typo")})
            self.assertTrue(configured.available)
            self.assertEqual(configured.path, Path(tmp) / "typo")

    def test_the_audit_and_inventory_default_to_the_configured_workspace(self):
        workspace, repo, audit, inventory = inputs({"NOTEBOOK_WORKSPACE": "/w", "NOTEBOOK_RESEARCH_REPO": "/r"})
        self.assertEqual((workspace.path, repo.path, audit.path, inventory.path),
                         (Path("/w"), Path("/r"), Path("/w/work/portal_data_audit.json"), Path("/w/outputs/tables/complete_run_inventory.csv")))
        # Their defaults sit inside the configured workspace, so they run (and fail loudly if that workspace lacks them).
        self.assertEqual((audit.configured, inventory.configured), (True, True))
        _, _, audit, _ = inputs({"NOTEBOOK_RESEARCH_REPO": "/r"})
        self.assertFalse(audit.configured)
        _, _, audit, inventory = inputs({"NOTEBOOK_DATA_AUDIT": "/a.json", "NOTEBOOK_RUN_INVENTORY": "/i.csv"})
        self.assertEqual((audit.path, inventory.path, audit.available, inventory.available), (Path("/a.json"), Path("/i.csv"), True, True))

    def test_a_fresh_clone_without_the_maintainer_inputs_skips_exactly_those_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A clone at <tmp>/clone: the default workspace <tmp> has no campaign tables, and HOME has no research repository.
            clone, home = Path(tmp) / "clone", Path(tmp) / "home"
            (clone / "tests").mkdir(parents=True)
            home.mkdir()
            for name in ["external_inputs.py", "test_export_research.py", "test_sources.py"]:
                (clone / "tests" / name).write_bytes((TESTS / name).read_bytes())
            for name in ["public", "scripts"]:
                (clone / name).symlink_to(PORTAL / name, target_is_directory=True)
            env = {key: value for key, value in os.environ.items() if not key.startswith("NOTEBOOK_")}
            env.update(HOME=str(home), PYTHONDONTWRITEBYTECODE="1")
            listed = subprocess.run([sys.executable, "-c", LIST_SKIPS], cwd=clone / "tests", env=env, capture_output=True, text=True, check=True)
            skips = {name: why for name, why in json.loads(listed.stdout).items() if why}
            self.assertEqual(set(skips), WORKSPACE_OR_REPO_TESTS)
            for name, why in skips.items():
                self.assertRegex(why, r"^needs .+: set NOTEBOOK_(WORKSPACE|RESEARCH_REPO|DATA_AUDIT|RUN_INVENTORY) \(not configured", name)
            # The runner reports them as skipped, not as errors; two portable tests still run and pass.
            portable = ["test_export_research.ResearchExportTests.test_complete_register_and_area_counts",
                        "test_export_research.ResearchExportTests.test_privacy_scan_allows_only_the_two_public_repository_prefixes"]
            result = subprocess.run([sys.executable, "-m", "unittest", "-v", *sorted(WORKSPACE_OR_REPO_TESTS), *portable],
                                    cwd=clone / "tests", env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            self.assertIn(f"OK (skipped={len(WORKSPACE_OR_REPO_TESTS)})", result.stderr)
            self.assertEqual(len(re.findall(r"\.\.\. skipped [\"']needs ", result.stderr)), len(WORKSPACE_OR_REPO_TESTS))
            self.assertEqual(result.stderr.count("... ok"), len(portable))
            self.assertNotIn("FileNotFoundError", result.stderr)

    def test_the_maintainer_inputs_run_every_data_test_when_configured(self):
        env = {**os.environ, "NOTEBOOK_WORKSPACE": "/configured/workspace", "NOTEBOOK_RESEARCH_REPO": "/configured/repo", "PYTHONDONTWRITEBYTECODE": "1"}
        listed = subprocess.run([sys.executable, "-c", LIST_SKIPS], cwd=TESTS, env=env, capture_output=True, text=True, check=True)
        self.assertEqual({name for name, why in json.loads(listed.stdout).items() if why}, set())


if __name__ == "__main__":
    unittest.main()
