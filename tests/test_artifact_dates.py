"""Maintainer refresh tests; no original research repository is changed."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
AT = '2026-09-15T13:27:30Z'
LATER = '2026-09-16T09:00:00Z'


def implementation(test):
    path = ROOT / 'scripts/export_artifact_dates.py'
    test.assertTrue(path.exists(), 'The verified date importer is not implemented')
    spec = importlib.util.spec_from_file_location('artifact_import', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    inventory = {'schemaVersion': 1, 'snapshotId': 'snapshot', 'snapshotSha256': 'a'*64, 'runsSha256': 'b'*64,
                 'publishedAt': '2026-09-15', 'exportedAt': AT,
                 'records': {'figure:f': {'kind': 'figure', 'id': 'f', 'contentSha256': 'c'*64, 'experimentIds': ['E'], 'runIds': ['r']},
                             'run:r': {'kind': 'run', 'id': 'r', 'contentSha256': 'd'*64, 'experimentIds': ['E'], 'runIds': ['r'], 'evidence': {'jobId': '1', 'originalLogSha256': 'e'*64, 'publicLogSha256': 'f'*64}},
                             'source:s': {'kind': 'source', 'id': 's', 'contentSha256': '1'*64, 'experimentIds': ['E'], 'runIds': ['r'], 'evidence': {'originalSha256': '2'*64, 'publicSha256': '3'*64}}}}
    date = {'value': AT, 'precision': 'second', 'basis': 'git_first_record_at_verified_path', 'commit': '4'*40}
    evidence = {'schemaVersion': 1, 'repositories': {'research': {'url': 'https://github.com/teshnizi2/hierarchical-metaoptimize'}, 'notebook': {'url': 'https://github.com/teshnizi2/metaoptimize-research-notebook'}},
                'sources': [{'id': 's', 'scope': 'original_file', 'repository': 'research', 'sha256': '2'*64, 'publicSha256': '3'*64, 'firstRecorded': date, 'lastContentUpdate': date}],
                'artifacts': [{'id': 'f', 'kind': 'figure', 'sha256': 'c'*64, 'firstRecorded': date, 'lastContentUpdate': date, 'repository': 'notebook',
                               'exported': {'value': AT, 'precision': 'second', 'basis': 'export_run_timestamp', 'publicSha256': 'c'*64}}],
                'runs': [{'id': 'r', 'jobId': '1', 'originalLogSha256': 'e'*64, 'publicLogSha256': 'f'*64, 'submitted': '2026-09-14T16:58:00+02:00', 'started': '2026-09-14T16:58:01+02:00', 'finished': '2026-09-14T18:00:35+02:00', 'precision': 'second', 'basis': 'slurm_accounting', 'reference': 'historical-accounting', 'logHref': '/assets/logs/r.txt'}]}
    return inventory, evidence


class ArtifactDateImportTests(unittest.TestCase):
    def test_default_evidence_discovery_and_at_alias_are_maintenance_only(self):
        module = implementation(self)
        self.assertTrue(hasattr(module, 'parse_arguments'), 'CLI does not expose the requested --at maintenance alias')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'portal'; root.mkdir()
            evidence = root.parent / 'artifact-date-evidence.json'; evidence.write_text('{}')
            args = module.parse_arguments(['--root', str(root), '--at', LATER])
            self.assertEqual(args.evidence, evidence.resolve())
            self.assertEqual(args.recorded_at, LATER)
            explicit = root / 'another.json'
            self.assertEqual(module.parse_arguments(['--root', str(root), '--evidence', str(explicit)]).evidence, explicit)
            evidence.unlink()
            self.assertIsNone(module.parse_arguments(['--root', str(root)]).evidence)

    def test_imports_only_hash_bound_facts_and_never_invents_creation(self):
        module = implementation(self); inventory, evidence = inputs()
        result = module.build_catalog(inventory, evidence, None, AT)
        self.assertNotIn('created', result['records']['figure:f'])
        self.assertEqual(result['records']['figure:f']['firstRecorded']['at'], AT)
        self.assertEqual(result['records']['run:r']['started']['at'], '2026-09-14T16:58:01+02:00')
        self.assertIn('verified path', result['records']['source:s']['updated']['basis'])
        self.assertEqual(result['records']['figure:f']['runWindow']['datedRuns'], 1)

    def test_unchanged_refresh_preserves_every_record_and_catalog_timestamp(self):
        module = implementation(self); inventory, evidence = inputs()
        first = module.build_catalog(inventory, evidence, None, AT)
        second = module.build_catalog(inventory, evidence, first, LATER)
        self.assertEqual(second, first)

    def test_refresh_without_private_fixture_preserves_unchanged_verified_facts(self):
        module = implementation(self); inventory, evidence = inputs()
        first = module.build_catalog(inventory, evidence, None, AT)
        self.assertEqual(module.build_catalog(inventory, {}, first, LATER), first)

    def test_changed_artifact_keeps_first_record_but_loses_unmatched_original_facts(self):
        module = implementation(self); inventory, evidence = inputs()
        first = module.build_catalog(inventory, evidence, None, AT)
        changed = copy.deepcopy(inventory); changed['records']['figure:f']['contentSha256'] = '9'*64
        refreshed = module.build_catalog(changed, evidence, first, LATER)['records']['figure:f']
        self.assertEqual(refreshed['firstRecorded'], first['records']['figure:f']['firstRecorded'])
        self.assertEqual(refreshed['updated']['at'], LATER)
        self.assertIn('notebook', refreshed['updated']['basis'])
        self.assertNotIn('exported', refreshed)
        self.assertNotIn('created', refreshed)

    def test_changed_original_source_hash_cannot_keep_its_old_git_update(self):
        module = implementation(self); inventory, evidence = inputs()
        first = module.build_catalog(inventory, evidence, None, AT)
        changed = copy.deepcopy(inventory); row = changed['records']['source:s']; row['contentSha256'] = '9'*64; row['evidence']['originalSha256'] = '0'*64
        refreshed = module.build_catalog(changed, evidence, first, LATER)['records']['source:s']
        self.assertEqual(refreshed['updated']['at'], LATER)
        self.assertNotIn('github.com', refreshed['updated'].get('href', ''))

    def test_changed_run_log_cannot_keep_unmatched_scheduler_facts(self):
        module = implementation(self); inventory, evidence = inputs()
        first = module.build_catalog(inventory, evidence, None, AT)
        changed = copy.deepcopy(inventory); row = changed['records']['run:r']; row['contentSha256'] = '9'*64; row['evidence']['originalLogSha256'] = '0'*64
        refreshed = module.build_catalog(changed, evidence, first, LATER)
        self.assertNotIn('started', refreshed['records']['run:r'])
        self.assertNotIn('runWindow', refreshed['records']['figure:f'])

    def test_accounting_job_identity_must_match_even_when_archived_bytes_match(self):
        module = implementation(self); inventory, evidence = inputs()
        inventory['records']['run:r']['evidence']['jobId'] = 'different-job'
        result = module.build_catalog(inventory, evidence, None, AT)
        self.assertNotIn('started', result['records']['run:r'])

    def test_mixed_day_and_second_precision_windows_preserve_the_coarser_fact(self):
        module = implementation(self); inventory, evidence = inputs()
        row = copy.deepcopy(inventory['records']['run:r']); row.update({'id': 'day', 'runIds': ['day']})
        inventory['records']['run:day'] = row
        inventory['records']['figure:f']['runIds'].append('day')
        dated = copy.deepcopy(evidence['runs'][0]); dated.update({'id': 'day', 'submitted': '2026-09-13', 'started': '2026-09-13', 'finished': '2026-09-13', 'precision': 'day'})
        evidence['runs'].append(dated)
        result = module.build_catalog(inventory, evidence, None, AT)['records']['figure:f']['runWindow']
        self.assertEqual(result['started']['at'], '2026-09-13')
        self.assertEqual(result['started']['precision'], 'day')
        self.assertEqual(result['datedRuns'], 2)

    def test_dated_original_csv_generation_keeps_original_and_public_hash_provenance(self):
        module = implementation(self); inventory, evidence = inputs()
        inventory['records'] = {'table:t': {'kind': 'table', 'id': 't', 'contentSha256': 'c'*64, 'experimentIds': [], 'runIds': []}}
        entry = evidence['artifacts'][0]; entry.update({'kind': 'table', 'id': 't', 'generated': {'value': '2026-09-14T15:39:50Z', 'precision': 'second', 'originalSha256': '5'*64, 'basis': 'producer_run_timestamp_for_original_csv'}})
        result = module.build_catalog(inventory, evidence, None, AT)['records']['table:t']
        self.assertEqual(result['created']['at'], '2026-09-14T15:39:50Z')
        self.assertIn('5'*64, result['created']['basis'])
        self.assertEqual(result['contentSha256'], 'c'*64)
        self.assertEqual(result['firstRecorded']['basis'], 'This file first recorded at its notebook Git path; this date does not establish original creation.')


if __name__ == '__main__':
    unittest.main()
