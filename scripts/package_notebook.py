"""Create a deterministic, portable public notebook ZIP; local tooling only."""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

PORTAL = Path(__file__).resolve().parents[1]
WORKSPACE = PORTAL.parents[1]
ROOT_FILES = ["README.md", "index.html", "package.json", "package-lock.json", "tsconfig.json", "vite.config.ts", "vercel.json", ".gitignore"]
SCRIPTS = ["journal.mjs", "sync-journal.mjs", "sync-artifact-dates.mjs"]
PORTABLE_COMMANDS = {"dev", "build", "preview", "test", "log", "dates:check"}
EXCLUDED_NAMES = {".DS_Store", "notebook-source.zip", "MetaOptimize_Research_Notebook_Source.zip"}
PRIVATE_PATTERNS = [
    ("private filesystem root", re.compile(r"/(?:Users|home|data1|scratch)/|/zfsstore/user/|/private/var/|[A-Za-z]:[\\/]+Users[\\/]+", re.I)),
    ("infrastructure identifier", re.compile(r"\b(?:salehkaleybars|s5014158|hmkhd2|teshnizi)\b", re.I)),
    ("contact address", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("private host", re.compile(r"\b100\.120\.248\.20\b")),
    ("credential literal", re.compile(r"\b(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{16,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe archive member name")


class PrivacyScan:
    def __init__(self):
        self.seen = set()
        self.files = self.pdf_pages = self.nested_archives = self.expanded_bytes = 0

    def text(self, value, context):
        for label, pattern in PRIVATE_PATTERNS:
            if pattern.search(value):
                raise ValueError(f"Privacy check failed ({label}) in {context}; matched content is not printed.")

    def scan(self, name, value, depth=0):
        safe_name(name)
        self.text(name, "archive member name")
        key = digest(value)
        if key in self.seen:
            return
        self.seen.add(key)
        self.files += 1
        self.expanded_bytes += len(value)
        if depth > 3 or self.files > 20000 or self.expanded_bytes > 768 * 1024 * 1024:
            raise ValueError("Archive privacy scan exceeds the bounded payload limit")
        self.text(value.decode("utf8", errors="replace"), name)
        if name.lower().endswith(".zip") or value.startswith(b"PK\x03\x04"):
            self.nested_archives += 1
            with zipfile.ZipFile(io.BytesIO(value)) as archive:
                for item in archive.infolist():
                    safe_name(item.filename)
                    if item.is_dir():
                        continue
                    if item.file_size > 256 * 1024 * 1024:
                        raise ValueError("Nested archive member exceeds the payload limit")
                    self.scan(item.filename, archive.read(item), depth + 1)
        elif name.lower().endswith(".pdf") or value.startswith(b"%PDF-"):
            try:
                from pypdf import PdfReader
            except ImportError as error:
                raise RuntimeError("pypdf is required for local PDF privacy verification") from error
            reader = PdfReader(io.BytesIO(value))
            if reader.attachments:
                raise ValueError("PDF embedded files are not permitted in the public package")
            self.text(str(reader.metadata), name + " metadata")
            for page in reader.pages:
                self.pdf_pages += 1
                self.text(page.extract_text() or "", name + " page text")
                if page.get("/Annots"):
                    self.text(str([item.get_object() for item in page["/Annots"]]), name + " annotations")

    def receipt(self):
        return {"status": "PASS", "uniquePayloadsScanned": self.files, "pdfPagesScanned": self.pdf_pages, "nestedArchivesScanned": self.nested_archives, "expandedBytesScanned": self.expanded_bytes,
                "scope": "Private identifier/path/contact/credential patterns in member names, text and raw bytes; nested ZIP contents and extracted PDF text, metadata and annotations. No OCR claim."}


def collect_payloads(root=PORTAL):
    root = Path(root).resolve()
    payloads, inputs = {}, {}

    def add(source, target=None):
        path = root / source
        relative = Path(source)
        if any((root / parent).is_symlink() for parent in [relative, *relative.parents] if parent != Path('.')):
            raise ValueError(f"Refusing symbolic link: {source}")
        if not path.is_file():
            raise ValueError(f"Required portable file is missing: {source}")
        value = path.read_bytes()
        inputs[source] = digest(value)
        payloads[target or source] = value

    for name in ROOT_FILES:
        add(name)
    if (root / ".vercelignore").exists():
        add(".vercelignore")
    add("docs/MAINTENANCE_PUBLIC.md", "docs/MAINTENANCE.md")
    add("content/journal.json")
    add("content/artifact-dates.json")
    for name in SCRIPTS:
        add("scripts/" + name)
    for folder in ["src", "public"]:
        base = root / folder
        if base.is_symlink():
            raise ValueError(f"Refusing symbolic link: {folder}")
        for path in sorted(base.rglob("*")):
            if path.name in EXCLUDED_NAMES or any(part.startswith(".") for part in path.relative_to(base).parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Refusing symbolic link: {path.relative_to(root).as_posix()}")
            if path.is_file():
                add(path.relative_to(root).as_posix())
    for path in sorted((root / "tests").glob("*.test.ts")):
        add(path.relative_to(root).as_posix())
    # The JavaScript tests read versioned JSON fixtures (published ID snapshots).
    for path in sorted((root / "tests/fixtures").glob("*.json")):
        add(path.relative_to(root).as_posix())
    for required in ["public/data/research.json", "public/data/runs.json", "public/data/journal.json", "public/data/artifact-dates.json"]:
        if required not in payloads:
            raise ValueError(f"Required evidence snapshot file is missing: {required}")
    if json.loads(payloads["content/journal.json"]) != json.loads(payloads["public/data/journal.json"]):
        raise ValueError("The journal is not synchronized. Run npm run build before packaging.")
    if json.loads(payloads["content/artifact-dates.json"]) != json.loads(payloads["public/data/artifact-dates.json"]):
        raise ValueError("Artifact dates are not synchronized. Run npm run build before packaging.")
    package = json.loads(payloads["package.json"])
    removed = sorted(set(package.get("scripts", {})) - PORTABLE_COMMANDS)
    package["scripts"] = {key: value for key, value in package.get("scripts", {}).items() if key in PORTABLE_COMMANDS}
    payloads["package.json"] = json_bytes(package)
    lock = json.loads(payloads["package-lock.json"])
    if "scripts" in lock.get("packages", {}).get("", {}):
        lock["packages"][""]["scripts"] = package["scripts"]
        payloads["package-lock.json"] = json_bytes(lock)
    scanner = PrivacyScan()
    for name, value in sorted(payloads.items()):
        scanner.scan(name, value)
    research, runs = json.loads(payloads["public/data/research.json"]), json.loads(payloads["public/data/runs.json"])
    metadata = {"snapshotId": research.get("meta", {}).get("snapshotId"), "experimentCount": len(research.get("experiments", [])), "runCount": len(runs), "publishedLogCount": sum(bool(run.get("logHref")) for run in runs),
                "removedPackageScripts": removed, "privacy": scanner.receipt(), "inputHashes": inputs}
    return payloads, metadata


def create_archive(destination, payloads, metadata):
    manifest = {"schemaVersion": 1, "snapshotId": metadata["snapshotId"], "payloadCount": len(payloads),
                "scope": "Portable notebook and its public evidence. MANIFEST.json excludes itself; the outer ZIP hash is reported separately.",
                "files": [{"path": name, "bytes": len(value), "sha256": digest(value)} for name, value in sorted(payloads.items())]}
    values = {**payloads, "MANIFEST.json": json_bytes(manifest)}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(values.items()):
            safe_name(name)
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return manifest


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("Archive duplicate member or CRC failure")
        for name in names:
            safe_name(name)
        manifest = json.loads(archive.read("MANIFEST.json"))
        expected = {row["path"] for row in manifest["files"]} | {"MANIFEST.json"}
        if set(names) != expected or manifest["payloadCount"] != len(manifest["files"]):
            raise ValueError("Archive manifest coverage mismatch")
        for row in manifest["files"]:
            value = archive.read(row["path"])
            if len(value) != row["bytes"] or digest(value) != row["sha256"]:
                raise ValueError(f"Archive manifest hash mismatch: {row['path']}")
    return {"status": "PASS", "files": len(names), "sha256": digest(Path(path).read_bytes()), "bytes": Path(path).stat().st_size}


def verify_portability(archive_path, log_path):
    results = []
    with tempfile.TemporaryDirectory(prefix="notebook-portable-") as temporary:
        root = Path(temporary) / "notebook"
        root.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(root)  # Names and hashes were verified before extraction.
        shutil.copyfile(archive_path, root / "public/assets/notebook-source.zip")
        user_config, global_config = Path(temporary) / "npm-user-config", Path(temporary) / "npm-global-config"
        user_config.touch(); global_config.touch()
        environment = {**os.environ, "CI": "true", "NPM_CONFIG_USERCONFIG": str(user_config), "NPM_CONFIG_GLOBALCONFIG": str(global_config)}
        with Path(log_path).open("w") as log:
            for command in [["npm", "ci"], ["npm", "test"], ["npm", "run", "build"]]:
                label = " ".join(command)
                print("Portable verification: " + label, flush=True)
                log.write("\n$ " + label + "\n"); log.flush()
                start = time.monotonic()
                result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                results.append({"command": label, "exitCode": result.returncode, "seconds": round(time.monotonic() - start, 3)})
                if result.returncode:
                    raise RuntimeError(f"Portable verification failed: {label}. See the local portability log.")
        if not (root / "dist/index.html").is_file():
            raise RuntimeError("Portable build did not produce dist/index.html")
        for name in ["data/research.json", "data/runs.json", "data/journal.json", "data/artifact-dates.json"]:
            if (root / "dist" / name).read_bytes() != (root / "public" / name).read_bytes():
                raise RuntimeError(f"Portable build changed evidence: {name}")
    return {"status": "PASS", "commands": results, "isolatedExtraction": True, "privateImportToolingRequired": False, "sourceDownloadRestoredFromOuterZip": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PORTAL)
    parser.add_argument("--verify", action="store_true", help="Run npm ci, npm test and npm run build in a clean extraction before publishing the ZIP")
    parser.add_argument("--output", type=Path, default=PORTAL / "public/assets/notebook-source.zip")
    parser.add_argument("--deliverable", type=Path, default=WORKSPACE / "outputs/MetaOptimize_Research_Notebook_Source.zip")
    parser.add_argument("--report", type=Path, default=WORKSPACE / "work/notebook_source_package_report.json")
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.report.with_name("notebook_source_portability.log")
    print("Packaging: collecting and checking the public payloads", flush=True)
    payloads, metadata = collect_payloads(args.root)
    print(f"Packaging: privacy checks passed for {len(payloads)} payload files; writing deterministic ZIPs", flush=True)
    with tempfile.TemporaryDirectory(prefix="notebook-package-") as temporary:
        candidate, repeated = Path(temporary) / "notebook.zip", Path(temporary) / "repeat.zip"
        manifest = create_archive(candidate, payloads, metadata)
        create_archive(repeated, payloads, metadata)
        archive = verify_archive(candidate)
        if digest(repeated.read_bytes()) != archive["sha256"]:
            raise RuntimeError("Deterministic ZIP verification failed")
        portability = verify_portability(candidate, log_path) if args.verify else {"status": "NOT_RUN", "command": "python3 scripts/package_notebook.py --verify"}
        for name, expected in metadata["inputHashes"].items():
            if digest((args.root / name).read_bytes()) != expected:
                raise RuntimeError(f"Source changed during packaging: {name}. Rerun after edits finish.")
        for destination in [args.output, args.deliverable]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = destination.with_suffix(".zip.tmp")
            shutil.copyfile(candidate, stage)
            stage.replace(destination)
        report = {"schemaVersion": 1, "status": "PASS", "snapshotId": metadata["snapshotId"], "archive": archive,
                  "deterministic": "PASS: two independently written ZIPs have identical SHA-256 hashes", "privacy": metadata["privacy"], "portability": portability,
                  "coverage": {key: metadata[key] for key in ["experimentCount", "runCount", "publishedLogCount"]},
                  "portablePackageScriptsRemoved": metadata["removedPackageScripts"], "sourceInputsUnchanged": True,
                  "excluded": ["recursive source ZIP", "dependency/build directories", "version-control and hosting associations", "private ingestion exporters", "Python tests", "internal specifications", "local packaging tooling"]}
        args.report.write_bytes(json_bytes(report))
        args.report.with_name("notebook_source_manifest.json").write_bytes(json_bytes(manifest))
    print(json.dumps({"status": "PASS", "files": archive["files"], "bytes": archive["bytes"], "sha256": archive["sha256"], "portability": portability["status"], **report["coverage"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
