"""Public source-catalog contract and provenance/privacy regression tests."""
import csv
import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

from external_inputs import inputs, needs

PORTAL = Path(__file__).resolve().parents[1]
# The published-catalog tests rebuild it from the maintainer's workspace and research repository;
# they are skipped, with the variables named, when neither is configured nor present.
_WORKSPACE, _REPO, _, _ = inputs()
WORKSPACE, REPO = _WORKSPACE.path, _REPO.path
EXPORTER = PORTAL / 'scripts/export_sources.py'


def load_exporter():
    if not EXPORTER.exists():
        return None
    spec = importlib.util.spec_from_file_location('source_exporter', EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceExportTests(unittest.TestCase):
    def setUp(self):
        self.exporter = load_exporter()
        self.assertIsNotNone(self.exporter, 'The deterministic source exporter is not implemented')

    def fixture(self, directory):
        root = Path(directory)
        repo, workspace = root / 'repo', root / 'workspace'
        for p in [repo / 'docs', repo / 'analysis', repo / 'bin', workspace / 'outputs/tables']:
            p.mkdir(parents=True)
        (repo / 'docs/MASTER-TABLE.md').write_text('# Register\n| Trial | See `analysis/registered.py` and `bin/missing.py` |\n')
        (repo / 'analysis/registered.py').write_text('# Explicitly cited calculation\nANSWER = 22\n')
        (repo / 'bin/other_name.sh').write_text('#!/bin/sh\nrun_name="trial-${seed}"\n')
        (repo / 'bin/trial_name_only.sh').write_text('#!/bin/sh\necho unrelated\n')
        with (workspace / 'outputs/tables/complete_experiment_register.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'section', 'batches', 'sources'])
            writer.writeheader()
            writer.writerow({'id': 'MT002', 'section': '1', 'batches': '["trial"]',
                             'sources': json.dumps([{'path': str(repo / 'docs/MASTER-TABLE.md'), 'line': 2}])})
        return repo, workspace

    def test_explicit_citations_and_content_mentions_do_not_promote_filename_guesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, workspace = self.fixture(tmp)
            index, links, audit, assets = self.exporter.build_catalog(repo, workspace)
            by_id = {s['id']: s for s in index}
            linked = {by_id[s]['path'] for s in links['MT002']['codeIds']}
            self.assertIn('analysis/registered.py', linked)
            self.assertIn('bin/other_name.sh', linked)
            self.assertNotIn('bin/trial_name_only.sh', linked)
            labels = ' '.join(s['label'] for s in links['MT002']['sourceRefs'])
            self.assertIn('batch mention', labels.lower())
            self.assertIn('not execution provenance', labels.lower())
            self.assertTrue(any('bin/missing.py' in m for m in links['MT002']['missing']))

    def test_redaction_preserves_scientific_values_and_line_numbers(self):
        text = ('# contact private.person@example.org\n'
                'ROOT=/home/s5014158/metaopt\n'
                'OTHER=/data1/salehkaleybars/metaopt\n'
                'LOCAL="/Users/teshnizi/Saber Optimization/project"\n'
                'API_KEY="sk-privatecredential0123456789"\n'
                '# ssh alice2; ssh alice; password=private-password\n'
                'ALPHA0=1e-6; BETA_CLIP=-15:-2.3026; SEEDS="55 56 57"\n'
                'PREDICTED_CUT=22; PEAK_BAR=0.9573130207199779\n')
        result = self.exporter.redact(text)
        self.assertEqual(len(result.splitlines()), len(text.splitlines()))
        for value in ['1e-6', '-15:-2.3026', '55 56 57', 'PREDICTED_CUT=22', '0.9573130207199779']:
            self.assertIn(value, result)
        for private in ['private.person', 'example.org', 's5014158', 'salehkaleybars', 'teshnizi',
                        '/Users/', '/home/', '/data1/', 'sk-private', 'private-password', 'alice2']:
            self.assertNotIn(private, result)

    def test_private_hostnames_are_stably_redacted_without_changing_research_labels(self):
        text = ('gateway p-cfer-016105; alternate p-cfer-028009\n'
                'compute node851 and node887; repeat node851\n'
                'logins nodelogin01 login1 login2 login.alice.example.nl\n'
                'CIFAR-100 ResNet-18 PEAK-AT-22 node14420 chunk777\n'
                'SEEDS=55,56,57; plateau5=68.69333333333334\n')
        result = self.exporter.redact(text)
        self.assertEqual(len(result.splitlines()), len(text.splitlines()))
        for hostname in ['p-cfer-016105', 'p-cfer-028009', 'node851', 'node887',
                         'nodelogin01', 'login1', 'login2', 'login.alice.example.nl']:
            self.assertNotIn(hostname, result)
        for label in ['CIFAR-100', 'ResNet-18', 'PEAK-AT-22', 'node14420', 'chunk777',
                      'SEEDS=55,56,57', 'plateau5=68.69333333333334']:
            self.assertIn(label, result)
        nodes = re.findall(r'\[HOST_[0-9a-f]+\]', result.splitlines()[1])
        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[0], nodes[2])
        self.assertNotEqual(nodes[0], nodes[1])

    def test_unknown_zfsstore_account_roots_are_redacted_in_place(self):
        text = ('DATA="/zfsstore/user/z12345678/datasets/CIFAR100"\n'
                'ROOT=/zfsstore/user/another-account\n'
                'See `/zfsstore/user/arbitrary.user/runs/seed55.out`.\n'
                'SEEDS=55,56,57; plateau5=68.69333333333334\n')
        result = self.exporter.redact(text)
        self.assertEqual(result, (
            'DATA="/cluster/user/datasets/CIFAR100"\n'
            'ROOT=/cluster/user\n'
            'See `/cluster/user/runs/seed55.out`.\n'
            'SEEDS=55,56,57; plateau5=68.69333333333334\n'))
        self.assertEqual(len(result.splitlines()), len(text.splitlines()))
        self.assertTrue(any(re.search(pattern, text) for pattern in self.exporter.PRIVATE_PATTERNS))
        self.assertFalse(any(re.search(pattern, result) for pattern in self.exporter.PRIVATE_PATTERNS))

    def test_only_allowlisted_research_sources_are_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, workspace = self.fixture(tmp)
            for relative in ['.env', 'auth.json', 'results/raw.json', 'paper/refs/third_party.md',
                             'docs/PLAN-appendix-infra.md', 'unrelated.py']:
                p = repo / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('private material must not be exported')
            index, _, _, assets = self.exporter.build_catalog(repo, workspace)
            self.assertFalse(any('private material' in content for content in assets.values()))
            self.assertFalse(any(s['path'].startswith(('results/', 'paper/', '.')) for s in index))

    def test_deleted_proposed_and_corrected_names_are_historical_mentions(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, workspace = self.fixture(tmp)
            (repo / 'docs/CORRECTIONS.md').write_text(
                '# Corrections\n## 1. Historical source changes\n'
                'DELETED: `bin/deleted.py`\n\n\n\n\n'
                'HANDOFF FOR NEXT TICK: register `bin/proposed.py`.\n\n\n\n\n'
                'Correct filename: `analysis/registered.py`, not `bin/wrong.py`.\n'
                'Historical reducer `bin/unavailable.py` was used.\n')
            register = workspace / 'outputs/tables/complete_experiment_register.csv'
            rows = list(csv.DictReader(register.read_text().splitlines()))
            rows[0]['sources'] = json.dumps([{'path': str(repo / 'docs/CORRECTIONS.md'), 'line': 2}])
            with register.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            index, links, audit, _ = self.exporter.build_catalog(repo, workspace)
            missing = ' '.join(links['MT002']['missing'])
            self.assertIn('bin/unavailable.py', missing)
            for name in ['deleted.py', 'proposed.py', 'wrong.py']:
                self.assertNotIn(name, missing)
            classified = {r['reference']: r['status'] for r in audit['historicalUnavailableSources']}
            self.assertEqual(classified['bin/deleted.py'], 'documented deletion')
            self.assertEqual(classified['bin/proposed.py'], 'proposed source name')
            self.assertEqual(classified['bin/wrong.py'], 'corrected filename')

    def test_documents_named_in_cited_sections_are_connected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, workspace = self.fixture(tmp)
            (repo / 'docs/CLOSEOUT.md').write_text('# Final audit\nThe result was corrected.\n')
            p = repo / 'docs/MASTER-TABLE.md'
            p.write_text(p.read_text().replace('bin/missing.py', 'docs/CLOSEOUT.md'))
            index, links, _, _ = self.exporter.build_catalog(repo, workspace)
            source = next(s for s in index if s['path'] == 'docs/CLOSEOUT.md')
            self.assertIn('MT002', source['experimentIds'])
            self.assertTrue(any(r['sourceId'] == source['id'] for r in links['MT002']['sourceRefs']))

    def test_numbered_sections_stop_at_peers_despite_markdown_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, workspace = self.fixture(tmp)
            (repo / 'docs/CORRECTIONS.md').write_text(
                '# CORRECTIONS 96 — metric audit\nSee `analysis/within96.py`.\n'
                '### 96.1 Nested analysis\nSee `analysis/child96.py`.\n'
                '## 97. A different investigation\nSee `analysis/later97.py`.\n')
            for name in ['within96', 'child96', 'later97']:
                (repo / f'analysis/{name}.py').write_text('VALUE = 1\n')
            register = workspace / 'outputs/tables/complete_experiment_register.csv'
            rows = list(csv.DictReader(register.read_text().splitlines()))
            rows[0]['sources'] = json.dumps([{'path': str(repo / 'docs/CORRECTIONS.md'), 'line': 1}])
            with register.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            index, links, _, _ = self.exporter.build_catalog(repo, workspace)
            by_id = {s['id']: s['path'] for s in index}
            linked = {by_id[s] for s in links['MT002']['codeIds']}
            self.assertIn('analysis/within96.py', linked)
            self.assertIn('analysis/child96.py', linked)
            self.assertNotIn('analysis/later97.py', linked)

    def test_numbered_subsections_keep_descendants_and_ignore_fenced_headings(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, workspace = self.fixture(tmp)
            (repo / 'docs/CORRECTIONS.md').write_text(
                '# CORRECTIONS 96 — metric audit\n'
                '### 96.1 Selected subsection\nSee `analysis/selected.py`.\n'
                '# 96.1.1 Child with a shallower Markdown level\n'
                'See `analysis/descendant.py`.\n'
                '```python\n# 96.2 A Python comment, not a document boundary\n```\n'
                'See `analysis/after_fence.py`.\n'
                '#### 96.2 Actual peer subsection\nSee `analysis/outside.py`.\n')
            for name in ['selected', 'descendant', 'after_fence', 'outside']:
                (repo / f'analysis/{name}.py').write_text('VALUE = 1\n')
            register = workspace / 'outputs/tables/complete_experiment_register.csv'
            rows = list(csv.DictReader(register.read_text().splitlines()))
            rows[0]['sources'] = '["CORRECTIONS 96.1"]'
            with register.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            index, links, _, _ = self.exporter.build_catalog(repo, workspace)
            by_id = {s['id']: s['path'] for s in index}
            linked = {by_id[s] for s in links['MT002']['codeIds']}
            self.assertIn('analysis/selected.py', linked)
            self.assertIn('analysis/descendant.py', linked)
            self.assertIn('analysis/after_fence.py', linked)
            self.assertNotIn('analysis/outside.py', linked)

    def published_rows(self):
        return self.exporter.register_rows(REPO, WORKSPACE)

    @needs(_WORKSPACE, _REPO)
    def test_every_record_has_valid_document_and_code_references(self):
        rows = self.published_rows()
        index, links, audit, assets = self.exporter.build_catalog(REPO, WORKSPACE, rows)
        self.assertEqual(len(rows), 164)
        self.assertEqual(set(links), {row['id'] for row in rows})
        by_id = {s['id']: s for s in index}
        self.assertEqual(len(by_id), len(index))
        for row in rows:
            entry = links[row['id']]
            self.assertTrue(entry['codeIds'], row['id'])
            self.assertTrue(any(by_id[r['sourceId']]['referenceType'] == 'document'
                                for r in entry['sourceRefs']), row['id'])
            for source_id in entry['codeIds']:
                self.assertIn(source_id, by_id)
            for ref in entry['sourceRefs']:
                source = by_id[ref['sourceId']]
                self.assertIn(row['id'], source['experimentIds'])
                self.assertGreaterEqual(ref.get('line', 1), 1)
                self.assertLessEqual(ref.get('line', 1), source['lines'])
        for source in index:
            self.assertEqual(set(source), {'id', 'path', 'language', 'role', 'href', 'originalSha256',
                'publicSha256', 'redacted', 'lines', 'experimentIds', 'referenceType'})
            self.assertIn(source['referenceType'], ['explicit', 'shared', 'document'])
            self.assertEqual(source['href'], '/source/' + source['id'] + '.txt')
            self.assertEqual(hashlib.sha256(assets[source['id']].encode()).hexdigest(), source['publicSha256'])
            self.assertEqual(len(assets[source['id']].splitlines()), source['lines'])

    @needs(_WORKSPACE, _REPO)
    def test_cvk2_links_exact_registered_scorer_and_runtime_snapshot(self):
        index, links, audit, assets = self.exporter.build_catalog(REPO, WORKSPACE, self.published_rows())
        source = next(s for s in index if s['path'] == 'analysis/cVK2_vggcut_score.py')
        self.assertEqual(source['originalSha256'],
            'aca0cdc314734ed9e8fc611ba42f774e7f76a40294df33de4883c59cd7047338')
        self.assertIn(source['id'], links['CVK2']['codeIds'])
        labels = [r['label'] for r in links['CVK2']['sourceRefs'] if r['sourceId'] == source['id']]
        self.assertTrue(any('5629b171' in label and 'verified' in label.lower() for label in labels))
        self.assertTrue(audit['cvk2Registration']['verified'])
        self.assertEqual(audit['cvk2Registration']['scorerId'], source['id'])
        shared = [s for s in index if s['referenceType'] == 'shared']
        self.assertTrue(any(s['path'].endswith('/train.py') for s in shared))
        self.assertTrue(any(s['path'].endswith('/Optimizers/HF.py') for s in shared))
        self.assertTrue(any(s['path'].endswith('/Optimizers/build_optimizer.py') for s in shared))

    @needs(_WORKSPACE, _REPO)
    def test_public_content_has_no_private_accounts_home_paths_or_credentials(self):
        index, links, audit, assets = self.exporter.build_catalog(REPO, WORKSPACE, self.published_rows())
        payload = json.dumps(index) + json.dumps(links) + '\n'.join(assets.values())
        for pattern in [r'(?i)teshnizi|salehkaleybars|s5014158|hmkhd2', r'/(?:Users|home|data1|scratch)/',
                        r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', r'\b(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{16,}',
                        r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']:
            self.assertIsNone(re.search(pattern, payload), pattern)

    @needs(_WORKSPACE, _REPO)
    def test_export_is_deterministic_and_every_href_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            rows = self.published_rows()
            self.exporter.export_catalog(REPO, WORKSPACE, a, rows)
            self.exporter.export_catalog(REPO, WORKSPACE, b, rows)
            files_a = {p.relative_to(a): p.read_bytes() for p in a.rglob('*') if p.is_file()}
            files_b = {p.relative_to(b): p.read_bytes() for p in b.rglob('*') if p.is_file()}
            self.assertEqual(files_a, files_b)
            index = json.loads((a / 'data/source-index.json').read_text())
            for source in index:
                p = a / source['href'].lstrip('/')
                self.assertTrue(p.is_file())
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), source['publicSha256'])

    def test_scratch_roots_are_redacted_in_place(self):
        text = 'CIFAR10_DIR=/path/to/scratch/cifar10 python3 run.py\nROOT=/scratch/s5014158/runs\nALPHA0=1e-6\n'
        result = self.exporter.redact(text)
        self.assertNotIn('/scratch/', result)
        self.assertNotIn('s5014158', result)
        self.assertIn('ALPHA0=1e-6', result)
        self.assertEqual(len(result.splitlines()), len(text.splitlines()))

    def test_master_table_anchors_follow_row_content_not_line_numbers(self):
        model = self.exporter.register_model
        lines = ['# Register', '', '| Is the gap caused by the schedule? | varied |', '| Does scale matter? | varied |']
        rows = [{'id': 'MT002', 'original_question': 'Does scale matter?', 'sources': json.dumps([{'path': '/r/docs/MASTER-TABLE.md', 'line': 3}])},
                {'id': 'MT003', 'original_question': 'Is the gap caused by the schedule?', 'sources': json.dumps(['/r/docs/MASTER-TABLE.md:4'])}]
        anchored = model.anchor_master_table(rows, lines)
        self.assertEqual(json.loads(anchored[0]['sources'])[0]['line'], 4)
        self.assertEqual(json.loads(anchored[1]['sources'])[0], '/r/docs/MASTER-TABLE.md:3')
        with self.assertRaisesRegex(ValueError, 'not unique by content'):
            model.anchor_master_table([{**rows[0], 'original_question': 'A question that was never in the table'}], lines)


if __name__ == '__main__':
    unittest.main()
