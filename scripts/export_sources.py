#!/usr/bin/env python3
"""Export a deterministic, redacted research-source catalog without executing code.

The experiment register (scripts/register_model.load_register: the exported
register plus the pinned MASTER-TABLE section-10 rows) is the mapping authority. References resolve to
real files/line anchors; additional links explicitly disclose exact batch-name
mentions. Filename similarity never establishes an experiment association.
Shared implementation snapshots do not establish historical per-run provenance.
Only CVK2's separately verified frozen sources receive a registration claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import register_model  # noqa: E402  (shared outcome model and partition-audit import)

PORTAL = Path(__file__).resolve().parents[1]
WORKSPACE = PORTAL.parents[1]
DEFAULT_REPO = Path('/Users/teshnizi/Saber Optimization/alice-backup/hierarchical-metaoptimize')
CODE_ROOTS = ('analysis', 'bin', 'jobs', 'patches', 'tests', 'imagenet489')
CODE_SUFFIXES = {'.py', '.sh'}
DOCUMENTS = {
    'docs/MASTER-TABLE.md', 'docs/CORRECTIONS.md', 'docs/FINDINGS.md',
    'docs/CLOSEOUT.md', 'docs/PAPER-CONFIG.md', 'docs/ARGS-AUDIT.md',
    'docs/DATASETS.md', 'docs/MAIN-IDEA-LEDGER.md', 'docs/MAIN-IDEA-GAPS.md',
    'docs/MAIN-IDEA-REPORT.md', 'docs/REGISTER-class-count-law.md',
    'docs/REGISTER-c82-singleton-law.md', 'docs/REGISTER-ideas-ABC.md',
    'docs/REGISTER-c82b-singleton-law-WITHDRAWAL.md', 'docs/IDEA3-robustness.md',
}
SHARED_REPO = {
    'analysis/aggregate.py': 'Shared metric aggregation',
    'analysis/argsline_guard.py': 'Shared argument audit',
    'patches/HF_patched.py': 'Shared optimizer patch snapshot',
    'jobs/run_cifar.sh': 'Shared training runner snapshot',
    'imagenet489/train_in489.py': 'Shared ImageNet subset trainer',
}
RUNTIME_SOURCES = {
    'train.py': 'runtime/cifar10/train.py',
    'HF.py': 'runtime/cifar10/Optimizers/HF.py',
    'build_optimizer.py': 'runtime/cifar10/Optimizers/build_optimizer.py',
    'build_network.py': 'runtime/cifar10/build_network.py',
    'run_cifar.sh': 'runtime/jobs/run_cifar.sh',
}
CVK2_SCORER = 'analysis/cVK2_vggcut_score.py'
CVK2_LAUNCHER = 'bin/cVK2_vgg_cut_ladder.sh'
CVK2_HASH = 'aca0cdc314734ed9e8fc611ba42f774e7f76a40294df33de4883c59cd7047338'
CVK2_LAUNCHER_HASH = '4183f85eb8e468720f452e98404efda1e7d3379a68b9b8bb9416f40dad21573e'
FILE_RE = re.compile(r'(?<![\w.-])((?:analysis|bin|jobs|patches|tests|docs|imagenet489|release)/(?:[\w.+-]+/)*[\w.+-]+\.(?:py|sh|md|txt))(?::(\d+))?')
HEADING_RE = re.compile(r'^(#{1,6})\s+(?:(?:CORRECTIONS|FINDINGS)\s+)?'
                        r'(\d+(?:\.\d+)*)(?=\s|\.(?:\s|$)|[):])', re.I)
HOST_RE = re.compile(r'\b(?:p-cfer-\d+|node\d{3}|nodelogin\d+|login\d+|'
                     r'login\.[A-Za-z0-9_.…-]+)\b', re.I)
PRIVATE_PATTERNS = [
    r'(?i)teshnizi|salehkaleybars|s5014158|hmkhd2', r'/(?:Users|home|data1|scratch)/',
    r'/zfsstore/user/',
    r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',
    r'\b(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{16,}',
    r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    HOST_RE.pattern,
]


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_id(relative_path: str) -> str:
    return 'src-' + digest(relative_path.encode())[:16]


def redact(text: str) -> str:
    """Replace private infrastructure in place, retaining one-to-one line anchors."""
    original_lines = len(text.splitlines())
    text = re.sub(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----.*?'
                  r'-----END (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                  lambda m: '\n'.join('[REDACTED_PRIVATE_KEY]' for _ in m[0].split('\n')),
                  text, flags=re.S)
    text = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\b(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{16,}', '[REDACTED_CREDENTIAL]', text)
    # Stable pseudonyms retain whether two runs shared a machine without
    # publishing private gateway, login, or compute-node identifiers.
    text = HOST_RE.sub(lambda m: '[HOST_' + digest(m[0].lower().encode())[:10] + ']', text)
    text = re.sub(r'(?i)(\b(?:api[_-]?(?:key|token)|access[_-]?token|client[_-]?secret|'
                  r'password|passwd)\s*[:=]\s*)([\'\"]?)([^\s\'\";]+)([\'\"]?)',
                  lambda m: m[1] + m[2] + '[REDACTED_CREDENTIAL]' + m[4], text)
    for prefix, public in [('/home/s5014158', '/cluster/account_b'),
                           ('/data1/salehkaleybars', '/cluster/account_a')]:
        text = text.replace(prefix, public)
    text = re.sub(r'/Users/[^/\s]+(?:/Saber Optimization/alice-backup)?', '/workspace', text)
    # /scratch/ is a private cluster root too; the research exporter rejects it, so
    # rewrite it here in place (one line in, one line out) rather than weaken that check.
    text = re.sub(r'/(?:home|data1|scratch|zfsstore/user)/[^/\s\'\"`<>]+', '/cluster/user', text)
    for private, public in [('s5014158', 'account_b'), ('salehkaleybars', 'account_a'),
                            ('teshnizi', 'researcher'), ('hmkhd2', 'legacy_reviewer'),
                            ('alice2', 'cluster_b'), ('alice', 'cluster_a')]:
        text = re.sub(re.escape(private), public, text, flags=re.I)
    text = re.sub(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])', '[REDACTED_IP]', text)
    assert len(text.splitlines()) == original_lines, 'Redaction changed source line numbering'
    return text


def allowed_code(path: str) -> bool:
    p = Path(path)
    return p.suffix in CODE_SUFFIXES and (p.parts[0] in CODE_ROOTS or
        (p.parts[0] == 'release' and len(p.parts) > 1 and p.parts[1] in {'code', 'scripts'}))


def role(path: str) -> str:
    if path.startswith('runtime/'):
        return 'Shared runtime snapshot, verified for CVK2 only'
    if path in SHARED_REPO:
        return SHARED_REPO[path]
    if path.startswith('docs/'):
        return 'Historical research documentation; follow notebook corrections'
    if path.startswith(('tests/', 'release/code/tests/')) or Path(path).name.startswith('test_'):
        return 'Research validation test'
    if path.startswith(('patches/', 'release/code/patches/')):
        return 'Research implementation patch'
    if path.endswith('.sh'):
        return 'Experiment launcher or batch utility snapshot'
    return 'Research scorer or analysis snapshot'


def register_rows(repo: Path, workspace: Path):
    """The published register, with MASTER-TABLE anchors resolved by row content."""
    repo = Path(repo).resolve()
    lines = (repo / register_model.MASTER_TABLE).read_text().splitlines()
    return register_model.anchor_master_table(register_model.load_register(Path(workspace).resolve(), repo), lines)


def build_catalog(repo: Path, workspace: Path, rows=None):
    repo, workspace = Path(repo).resolve(), Path(workspace).resolve()
    if rows is None:
        rows = list(csv.DictReader((workspace / 'outputs/tables/complete_experiment_register.csv').read_text().splitlines()))
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate experiment IDs in the authoritative register')
    candidates: dict[str, bytes] = {}
    for root in CODE_ROOTS:
        for p in sorted((repo / root).rglob('*')):
            if p.is_file() and p.suffix in CODE_SUFFIXES and '__pycache__' not in p.parts:
                candidates[p.relative_to(repo).as_posix()] = p.read_bytes()
    for relative in sorted(DOCUMENTS):
        p = repo / relative
        if p.is_file():
            candidates[relative] = p.read_bytes()
    runtime_capture = workspace / 'work/cvk2_verification/sources'
    for filename, relative in RUNTIME_SOURCES.items():
        p = runtime_capture / filename
        if p.is_file():
            candidates[relative] = p.read_bytes()
    texts = {p: raw.decode('utf-8') for p, raw in candidates.items()}
    line_cache = {p: text.splitlines() for p, text in texts.items()}
    associations: dict[str, set[str]] = {}
    links = {eid: {'codeIds': [], 'sourceRefs': [], 'missing': []} for eid in ids}
    ref_keys = {eid: set() for eid in ids}
    ignored_data_refs: set[str] = set()
    historical_unavailable = []
    direct_associations: set[tuple[str, str]] = set()
    batch_associations: set[tuple[str, str]] = set()
    anchor_fallbacks = []
    heading_cache = {}

    def add_file(relative: str) -> bool:
        if relative in candidates:
            return True
        if not allowed_code(relative) and relative not in DOCUMENTS:
            return False
        p = (repo / relative).resolve()
        if not p.is_relative_to(repo) or not p.is_file():
            return False
        candidates[relative] = p.read_bytes()
        texts[relative] = candidates[relative].decode('utf-8')
        line_cache[relative] = texts[relative].splitlines()
        return True

    def missing(eid, value):
        safe = redact(value)
        if safe not in links[eid]['missing']:
            links[eid]['missing'].append(safe)

    def add_ref(eid: str, relative: str, line: int | None, label: str, method: str):
        if not add_file(relative):
            missing(eid, 'Referenced source not available: ' + relative)
            return
        line = line or 1
        if not 1 <= line <= len(line_cache[relative]):
            anchor_fallbacks.append({'experimentId': eid, 'source': redact(relative), 'requestedLine': line})
            label += '; original line unavailable, showing document start'
            line = 1
        sid = source_id(relative)
        key = (sid, line, label)
        if key not in ref_keys[eid]:
            links[eid]['sourceRefs'].append({'sourceId': sid, 'line': line, 'label': redact(label)})
            ref_keys[eid].add(key)
        associations.setdefault(relative, set()).add(eid)
        if not relative.startswith('docs/') and sid not in links[eid]['codeIds']:
            links[eid]['codeIds'].append(sid)
        if method == 'explicit':
            direct_associations.add((eid, relative))
        elif method == 'batch':
            batch_associations.add((eid, relative))

    def document_headings(relative: str):
        """Numbered record hierarchy takes precedence over inconsistent # levels."""
        if relative not in heading_cache:
            headings = {}
            fence = None
            for n, body in enumerate(line_cache.get(relative, []), 1):
                marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', body)
                if marker:
                    if fence is None:
                        fence = marker[1]
                    elif (marker[1][0] == fence[0] and len(marker[1]) >= len(fence)
                          and not body[marker.end():].strip()):
                        fence = None
                    continue
                if fence is not None:
                    continue
                heading = re.match(r'^(#{1,6})\s', body)
                if heading:
                    numbered = HEADING_RE.match(body)
                    number = tuple(map(int, numbered[2].split('.'))) if numbered else None
                    headings[n] = (len(heading[1]), number)
            heading_cache[relative] = headings
        return heading_cache[relative]

    def section_line(relative: str, number: str) -> int | None:
        requested = tuple(map(int, number.split('.')))
        for n, (_, value) in document_headings(relative).items():
            if value == requested:
                return n
        if '.' in number:
            return section_line(relative, number.split('.')[0])
        return None

    def context(relative: str, line: int):
        lines = line_cache[relative]
        if not 1 <= line <= len(lines):
            return []
        if relative == 'docs/MASTER-TABLE.md':
            return [(line, lines[line - 1])]
        headings = document_headings(relative)
        if line not in headings:
            return [(line, lines[line - 1])]
        depth, number = headings[line]
        end = len(lines)
        for n, (next_depth, next_number) in headings.items():
            if n <= line:
                continue
            if number is not None and next_number is not None:
                descendant = (len(next_number) > len(number)
                              and next_number[:len(number)] == number)
                boundary = not descendant
            else:
                boundary = next_depth <= depth
            if boundary:
                end = n - 1
                break
        return list(enumerate(lines[line - 1:end], line))

    def linked_code_in_context(eid: str, document: str, line: int):
        for n, body in context(document, line):
            for m in FILE_RE.finditer(body):
                relative = m[1]
                if allowed_code(relative):
                    if not add_file(relative):
                        nearby = '\n'.join(line_cache[document][max(0, n - 5):n]).replace('`', '')
                        status = None
                        if re.search(r'\bnot\s+' + re.escape(relative), body.replace('`', '')):
                            status = 'corrected filename'
                        elif re.search(r'\b(?:DELETED|withdrawn|removed)\b', nearby, re.I):
                            status = 'documented deletion'
                        elif re.search(r'\b(?:HANDOFF|NEXT TICK|proposed)\b', nearby, re.I):
                            status = 'proposed source name'
                        if status:
                            historical_unavailable.append({'experimentId': eid, 'reference': redact(relative),
                                'status': status, 'source': document, 'line': n})
                            add_ref(eid, document, n,
                                f'Historical source mention: {relative} ({status}); absent from this source snapshot', 'document')
                            continue
                    add_ref(eid, relative, int(m[2] or 1),
                        f'Code cited at {document}:{n}; source association, not execution provenance', 'explicit')
                elif relative in DOCUMENTS and relative != document:
                    add_ref(eid, relative, int(m[2] or 1),
                        f'Research document cited at {document}:{n}', 'document')

    def add_document(eid: str, relative: str, line: int | None, label: str):
        add_ref(eid, relative, line, label, 'document')
        if relative in line_cache:
            linked_code_in_context(eid, relative, line or 1)

    def parse_reference(eid: str, source):
        if isinstance(source, str):
            value, explicit_line, section = source, None, None
        else:
            value = source.get('path') or source.get('reference', '')
            explicit_line, section = source.get('line'), source.get('section')
            if source.get('reference'):
                value += ' [' + source['reference'] + ']'
        value = value.replace(str(repo) + '/', '')
        files = list(FILE_RE.finditer(value))
        if not files:
            for name in ['CORRECTIONS', 'FINDINGS', 'CLOSEOUT']:
                m = re.search(r'\b' + name + r'\s*(\d+(?:\.\d+)*)?', value)
                if m:
                    document = f'docs/{name}.md'
                    line = section_line(document, m[1]) if m[1] else 1
                    add_document(eid, document, line, f'Cited research record: {value}')
                    return
            if value.startswith('results/'):
                ignored_data_refs.add(value)
            return
        for m in files:
            relative = m[1]
            line = explicit_line or (int(m[2]) if m[2] else None)
            if relative.startswith('docs/'):
                if relative not in DOCUMENTS:
                    ignored_data_refs.add(relative)
                    continue
                sections = []
                if section:
                    sections.append(str(section))
                elif 'CORRECTIONS' in relative and not line:
                    tail = value[m.end():]
                    sections = re.findall(r'(?<![\w.])(\d+(?:\.\d+)*)(?![\w.])', tail)
                if sections and not line:
                    for number in dict.fromkeys(sections):
                        anchor = section_line(relative, number)
                        if anchor is None:
                            anchor_fallbacks.append({'experimentId': eid, 'source': relative, 'section': number})
                        add_document(eid, relative, anchor, f'Cited research record: {relative} section {number}')
                else:
                    add_document(eid, relative, line, f'Cited research record: {relative}' + (f' section {section}' if section else ''))
            elif allowed_code(relative):
                add_ref(eid, relative, line,
                        'Explicit source citation in the experiment register; not execution provenance', 'explicit')

    for row in rows:
        for source in json.loads(row.get('sources') or '[]'):
            parse_reference(row['id'], source)

    # Match exact batch prefixes in source text; never use source filenames.
    # Short strings such as "ms" and "sc" require a run-name suffix to avoid
    # matching ordinary optimizer terminology and local variable names.
    batch_to_ids: dict[str, set[str]] = {}
    for row in rows:
        for batch in json.loads(row.get('batches') or '[]'):
            if re.fullmatch(r'[A-Za-z][\w-]*', batch):
                batch_to_ids.setdefault(batch, set()).add(row['id'])
    for relative in sorted(texts):
        if not allowed_code(relative) or relative.startswith(('patches/', 'tests/', 'jobs/sweeps/reprio')):
            continue
        found = {}
        for n, body in enumerate(line_cache[relative], 1):
            for match in re.finditer(r'(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_-]*', body):
                token = match[0]
                ends = [m.start() for m in re.finditer('[-_]', token)] + [len(token)]
                for end in ends:
                    batch = token[:end]
                    if batch not in batch_to_ids or batch in found:
                        continue
                    if len(batch) <= 2:
                        suffix = body[match.start() + end:]
                        if not re.match(r'[-_][A-Za-z0-9${]', suffix):
                            continue
                    found[batch] = n
        for batch, line in sorted(found.items()):
            for eid in sorted(batch_to_ids[batch]):
                if (eid, relative) not in direct_associations:
                    add_ref(eid, relative, line,
                        f'Exact batch mention "{batch}" in source text; related source, not execution provenance', 'batch')

    for row in rows:
        eid = row['id']
        for relative in [*SHARED_REPO, *RUNTIME_SOURCES.values()]:
            if relative not in candidates:
                continue
            if relative.startswith('imagenet489/') and 'ImageNet' not in row.get('scope', ''):
                continue
            add_ref(eid, relative, 1,
                'Shared implementation snapshot for context; not historical per-run execution provenance', 'shared')

    cvk2_registration = {'verified': False, 'reason': 'CVK2 is not present in this input register'}
    if 'CVK2' in links:
        evidence = json.loads((workspace / 'work/cvk2_evidence.json').read_text())
        captured = runtime_capture / 'cVK2_vggcut_score.py'
        captured_launcher = runtime_capture / 'cVK2_vgg_cut_ladder.sh'
        if not (evidence['complete'] and evidence['independent_verification_passed'] and
                evidence['preregistration']['source_sha256'] == CVK2_HASH and
                digest(candidates[CVK2_SCORER]) == digest(captured.read_bytes()) == CVK2_HASH and
                digest(candidates[CVK2_LAUNCHER]) == digest(captured_launcher.read_bytes()) == CVK2_LAUNCHER_HASH):
            raise ValueError('CVK2 registered source or captured verification mismatch')
        for relative in [CVK2_SCORER, CVK2_LAUNCHER]:
            add_ref('CVK2', relative, 1,
                'Verified frozen registration: commit 5629b171 before submission; source hash matched unmodified execution on 2026-09-14', 'explicit')
        for filename, relative in RUNTIME_SOURCES.items():
            original_hash = digest(candidates[relative])
            if original_hash not in evidence['sourcehashes'].values():
                raise ValueError(f'CVK2 runtime source hash is not verified: {relative}')
            add_ref('CVK2', relative, 1,
                'CVK2 runtime snapshot: source hash verified before and after scoring on 2026-09-14; other experiments retain shared-context status', 'shared')
        cvk2_registration = {
            'verified': True, 'scorerId': source_id(CVK2_SCORER), 'launcherId': source_id(CVK2_LAUNCHER),
            'commit': evidence['preregistration']['commit'], 'scorerSha256': CVK2_HASH,
            'launcherSha256': CVK2_LAUNCHER_HASH, 'submittedUtc': evidence['preregistration']['submission_utc'],
            'validityPassed': evidence['registered_scorer_passed'],
            'scientificOutcome': evidence['registered_tokens'][0],
        }

    # Cited research records remain accessible as context; unrelated documents,
    # release duplicates, raw data, binaries, configuration and credentials do not.
    selected = set(associations)
    selected.update(p for p in DOCUMENTS if p in candidates)
    index, assets = [], {}
    for relative in sorted(selected):
        raw = candidates[relative]
        original = raw.decode('utf-8')
        public = redact(original)
        sid = source_id(relative)
        assets[sid] = public
        if relative.startswith('docs/'):
            kind = 'document'
        elif relative in SHARED_REPO or relative.startswith('runtime/'):
            kind = 'shared'
        else:
            kind = 'explicit'
        index.append({'id': sid, 'path': redact(relative),
            'language': {'.py': 'python', '.sh': 'bash', '.md': 'markdown'}.get(Path(relative).suffix, 'text'),
            'role': role(relative), 'href': f'/source/{sid}.txt', 'originalSha256': digest(raw),
            'publicSha256': digest(public.encode('utf-8')), 'redacted': original != public,
            'lines': len(original.splitlines()), 'experimentIds': sorted(associations.get(relative, set())),
            'referenceType': kind})
    for entry in links.values():
        entry['codeIds'].sort()
        entry['sourceRefs'].sort(key=lambda x: (x['sourceId'], x['line'], x['label']))
        entry['missing'].sort()
    serialized = json.dumps(index) + json.dumps(links) + '\n'.join(assets.values())
    for pattern in PRIVATE_PATTERNS:
        if re.search(pattern, serialized):
            raise ValueError(f'Public privacy check failed for pattern {pattern}')
    by_id = {s['id']: s for s in index}
    audit = {
        'schemaVersion': 1, 'deterministic': True, 'repositoryReadOnly': True,
        'registerSha256': digest((workspace / 'outputs/tables/complete_experiment_register.csv').read_bytes()),
        'experiments': len(rows), 'sources': len(index), 'publicBytes': sum(len(s.encode()) for s in assets.values()),
        'referenceTypes': dict(sorted(Counter(s['referenceType'] for s in index).items())),
        'redactedSources': sum(s['redacted'] for s in index),
        'experimentsWithDocuments': sum(any(by_id[r['sourceId']]['referenceType'] == 'document'
            for r in entry['sourceRefs']) for entry in links.values()),
        'experimentsWithCode': sum(bool(entry['codeIds']) for entry in links.values()),
        'explicitCitationAssociations': len(direct_associations), 'batchMentionAssociations': len(batch_associations),
        'unresolvedReferences': [{'experimentId': eid, 'reference': ref} for eid, entry in links.items() for ref in entry['missing']],
        'anchorFallbacks': anchor_fallbacks, 'excludedDataReferences': sorted(ignored_data_refs),
        'historicalUnavailableSources': historical_unavailable,
        'cvk2Registration': cvk2_registration,
        'provenancePolicy': 'Direct citations and exact batch-name mentions identify related sources, not runtime provenance. Shared files are current snapshots. CVK2 alone has the explicit frozen-source verification recorded here.',
        'privacyPolicy': 'Private account identifiers, email addresses, home/workspace roots, IP addresses, gateway/login/compute hostnames and credential literals are replaced with neutral placeholders. Host pseudonyms retain repeated-machine identity. Scientific values and line numbering are retained. Redacted source downloads are research records, not byte-identical executable reproductions.',
        'exclusions': ['Raw experiment data and logs', 'Binaries', 'Authentication and configuration files',
                       'Unrelated operational documents', 'Third-party papers', 'Unreferenced duplicate release sources'],
    }
    return index, links, audit, assets


def export_catalog(repo: Path, workspace: Path, public_root: Path, rows=None):
    index, links, audit, assets = build_catalog(repo, workspace, rows)
    public_root = Path(public_root)
    source_dir, data_dir = public_root / 'source', public_root / 'data'
    source_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = {sid + '.txt' for sid in assets}
    for stale in source_dir.glob('src-*.txt'):
        if stale.name not in wanted:
            stale.unlink()
    for sid, text in assets.items():
        (source_dir / f'{sid}.txt').write_bytes(text.encode('utf-8'))
    for filename, payload in [('source-index.json', index), ('source-links.json', links)]:
        target = data_dir / filename
        temp = target.with_suffix('.json.tmp')
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temp.replace(target)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--workspace', type=Path, default=WORKSPACE)
    parser.add_argument('--public', type=Path, default=PORTAL / 'public')
    parser.add_argument('--audit', type=Path, default=WORKSPACE / 'work/portal_source_audit.json')
    args = parser.parse_args()
    rows = register_rows(args.repo, args.workspace)
    audit = export_catalog(args.repo, args.workspace, args.public, rows)
    audit['partitionAuditCommit'] = register_model.PARTITION_AUDIT_COMMIT
    audit['registerModel'] = {'researchQuestions': sum(r['kind'] == 'research' for r in rows),
                              'methodChecks': sum(r['kind'] == 'method-check' for r in rows)}
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: audit[k] for k in ['experiments', 'sources', 'redactedSources',
        'experimentsWithDocuments', 'experimentsWithCode', 'cvk2Registration']}, indent=2))


if __name__ == '__main__':
    main()
