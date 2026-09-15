#!/usr/bin/env python3
"""Import hash-verified date evidence; maintain dates without touching research data.

The Node --inventory implementation owns canonical JSON hashes. Run this importer
with the private evidence fixture to seed or enrich the versioned catalog. Later
refreshes can omit the fixture: unchanged records retain their facts; changed
records receive a notebook-record update and no unverified historical facts.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
def fact(at, basis, precision='second', href=None):
    value = {'at': at, 'precision': precision, 'basis': basis}
    if href:
        value['href'] = href
    return value


def git_fact(value, evidence, repository, label):
    url = evidence.get('repositories', {}).get(repository, {}).get('url')
    href = url + '/commit/' + value['commit'] if url and value.get('commit') else None
    return fact(value['value'], label, value['precision'], href)


def historical_facts(item, evidence):
    """Return facts only when the fixture still identifies this exact evidence."""
    kind, eid = item['kind'], item['id']
    result = {}
    if kind == 'source':
        source = next((row for row in evidence.get('sources', []) if row['id'] == eid), None)
        current = item.get('evidence', {})
        if source:
            original = source['scope'] == 'original_file'
            expected = current.get('originalSha256' if original else 'publicSha256')
            if source['sha256'] == expected and source['publicSha256'] == current.get('publicSha256'):
                scope = 'Original source at the verified path' if original else 'Archived public copy at the verified path; original creation is unknown'
                result['firstRecorded'] = git_fact(source['firstRecorded'], evidence, source['repository'], scope + ': first Git record (copy ancestry excluded).')
                result['updated'] = git_fact(source['lastContentUpdate'], evidence, source['repository'], scope + ': last Git content change; not a runtime date.')
    elif kind == 'run':
        row = next((row for row in evidence.get('runs', []) if row['id'] == eid), None)
        current = item.get('evidence', {})
        if row and all(row[field] == current.get(field) for field in ['jobId', 'originalLogSha256', 'publicLogSha256']):
            for field, name in [('submitted', 'submission'), ('started', 'start'), ('finished', 'finish')]:
                timing = 'retained accounting evidence establishes UTC+02:00' if row['precision'] == 'second' else 'recorded calendar day; precise time and timezone are not asserted'
                result[field] = fact(row[field], f'Slurm accounting {name}; {timing}. Joined to the verified original/public log hashes.', row['precision'])
    else:
        artifact_id = 'report-pdf' if kind == 'download' and eid == 'pdf' else eid
        artifact_kind = 'pdf' if kind == 'download' and eid == 'pdf' else kind
        row = next((row for row in evidence.get('artifacts', []) if row['id'] == artifact_id and row['kind'] == artifact_kind), None)
        if row and row['sha256'] == item['contentSha256']:
            result['firstRecorded'] = git_fact(row['firstRecorded'], evidence, 'notebook', 'This file first recorded at its notebook Git path; this date does not establish original creation.')
            result['updated'] = git_fact(row['lastContentUpdate'], evidence, 'notebook', 'Last content change recorded at this notebook Git path.')
            exported = row.get('exported')
            if exported and exported['publicSha256'] == item['contentSha256']:
                result['exported'] = fact(exported['value'], 'Hash-matched public export run timestamp; not original creation or precise file-write completion.', exported['precision'])
            generated = row.get('generated')
            if kind == 'table' and generated:
                result['created'] = fact(generated['value'], 'Original CSV generated in the verified portfolio export; public copy may be redacted. Original SHA-256: ' + generated['originalSha256'] + '. Public SHA-256 is this record content hash.', generated['precision'], '/assets/tables/portfolio/file_manifest.csv')
    return result


def instant(value):
    parsed = datetime.fromisoformat(value['at'].replace('Z', '+00:00'))
    # UTC midnight is only an ordering convention for a day value; the output
    # keeps its day precision and does not manufacture a precise timestamp.
    return parsed.replace(tzinfo=timezone.utc) if value['precision'] == 'day' else parsed


def build_catalog(inventory, evidence, previous, recorded_at):
    previous = previous or {}
    previous_records = previous.get('records', {})
    evidence = evidence or {}
    snapshot = evidence.get('recordingSnapshot', {})
    snapshot_matches = snapshot.get('snapshotSha256') == inventory['snapshotSha256'] and snapshot.get('runsSha256') == inventory['runsSha256']
    records = {}
    for key, item in inventory['records'].items():
        prior = previous_records.get(key)
        unchanged = prior and prior['contentSha256'] == item['contentSha256'] and prior['experimentIds'] == item['experimentIds']
        fresh = historical_facts(item, evidence)
        if unchanged:
            row = copy.deepcopy(prior)
            # Verified new information can fill an unknown, never reset a known date.
            for field, value in fresh.items():
                if not row.get(field):
                    row[field] = value
        else:
            row = {field: copy.deepcopy(item[field]) for field in ['kind', 'id', 'contentSha256', 'experimentIds']}
            row.update(fresh)
            if prior:
                if prior.get('firstRecorded'):
                    row['firstRecorded'] = copy.deepcopy(prior['firstRecorded'])
                row['updated'] = fact(recorded_at, 'Changed artifact or metadata recorded by this notebook maintenance run; not original research creation.')
            elif not row.get('firstRecorded'):
                if snapshot_matches and item['kind'] in ['experiment', 'warning', 'chart', 'run']:
                    row['firstRecorded'] = git_fact(snapshot['firstRecorded'], evidence, 'notebook', 'This record is present in the hash-matched first notebook Git snapshot; not the original experiment or warning date.')
                else:
                    row['firstRecorded'] = fact(recorded_at, 'First recorded in the notebook date catalog; historical creation remains unknown.')
        row.pop('runWindow', None)
        records[key] = row
    for key, item in inventory['records'].items():
        if item['kind'] == 'run':
            continue
        dated = [records['run:' + rid] for rid in item.get('runIds', []) if records['run:' + rid].get('started') and records['run:' + rid].get('finished')]
        if dated:
            start = min((row['started'] for row in dated), key=instant)
            finish = max((row['finished'] for row in dated), key=instant)
            basis = 'Actual accounting range of linked batch runs; not an individual measurement, question creation, or validation date.'
            records[key]['runWindow'] = {'started': fact(start['at'], basis, start['precision']), 'finished': fact(finish['at'], basis, finish['precision']), 'datedRuns': len(dated), 'totalRuns': len(item['runIds'])}
    catalog = {'schemaVersion': 1, **{key: inventory[key] for key in ['snapshotId', 'snapshotSha256', 'runsSha256', 'publishedAt', 'exportedAt']}, 'recordedAt': recorded_at, 'records': records}
    # A no-op refresh is byte-stable, including the maintenance timestamp.
    comparison = copy.deepcopy(catalog)
    if previous:
        comparison['recordedAt'] = previous.get('recordedAt')
        if comparison == previous:
            return copy.deepcopy(previous)
    return catalog


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--evidence', type=Path, help='Private hash-verified evidence fixture; never copied to the website')
    parser.add_argument('--at', '--recorded-at', dest='recorded_at', help='Offset-qualified catalog/change-record timestamp only; never a true creation date')
    args = parser.parse_args(argv)
    if args.evidence is None:
        candidate = args.root.resolve().parent / 'artifact-date-evidence.json'
        if candidate.is_file():
            args.evidence = candidate
    return args


def main():
    args = parse_arguments()
    root = args.root.resolve()
    inventory = json.loads(subprocess.check_output(['node', str(root / 'scripts/sync-artifact-dates.mjs'), '--inventory', '--root', str(root)], text=True))
    evidence = json.loads(args.evidence.read_text()) if args.evidence else {}
    destination = root / 'content/artifact-dates.json'
    previous = json.loads(destination.read_text()) if destination.exists() else None
    recorded_at = args.recorded_at or datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    catalog = build_catalog(inventory, evidence, previous, recorded_at)
    content = json.dumps(catalog, indent=2, ensure_ascii=False) + '\n'
    # Validate against the same inventory before writing either public or content state.
    command = "import {validateCatalog} from './scripts/sync-artifact-dates.mjs'; let text=''; for await (const c of process.stdin) text+=c; const x=JSON.parse(text); validateCatalog(x.catalog,x.inventory);"
    subprocess.run(['node', '--input-type=module', '-e', command], cwd=root, input=json.dumps({'catalog': catalog, 'inventory': inventory}), text=True, check=True)
    if not destination.exists() or destination.read_text() != content:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(content)
        temporary.replace(destination)
    subprocess.run(['node', str(root / 'scripts/sync-artifact-dates.mjs'), '--root', str(root)], check=True)
    print(json.dumps({'records': len(catalog['records']), 'created': sum(bool(row.get('created')) for row in catalog['records'].values()), 'runStarts': sum(bool(row.get('started')) for row in catalog['records'].values()), 'changed': catalog != previous}))


if __name__ == '__main__':
    main()
