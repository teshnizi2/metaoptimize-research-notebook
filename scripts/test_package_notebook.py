"""Local packaging checks; deliberately excluded from the portable notebook."""
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("package_notebook", Path(__file__).with_name("package_notebook.py"))
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


class NotebookPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="notebook-package-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        files = {
            "README.md": "Portable notebook\n",
            "docs/MAINTENANCE_PUBLIC.md": "Portable maintenance\n",
            "docs/MAINTENANCE.md": "PRIVATE INTERNAL GUIDE\n",
            "index.html": "<html></html>", "tsconfig.json": "{}", "vite.config.ts": "export default {};",
            "vercel.json": "{}", ".gitignore": "node_modules/\n",
            "src/main.tsx": "export {};", "tests/research.test.ts": "export {};",
            "tests/fixtures/register-ids.json": "{}\n",
            "scripts/journal.mjs": "export {};", "scripts/sync-journal.mjs": "export {};",
            "scripts/sync-artifact-dates.mjs": "export {};",
            "scripts/check-experiment-copy.ts": "export {};",
            "content/journal.json": "[]\n", "public/data/journal.json": "[]\n",
            "content/artifact-dates.json": "{}\n", "public/data/artifact-dates.json": "{}\n",
            "content/experiment-copy.json": "{}\n",
            "public/data/research.json": json.dumps({"meta": {"snapshotId": "test-snapshot"}, "experiments": [{"id": "CVK2"}]}),
            "public/data/runs.json": "[]", "public/source/example.txt": "Scientific evidence\n",
            "public/assets/notebook-source.zip": "old recursive archive",
            "node_modules/private.txt": "excluded", "dist/private.txt": "excluded", ".vercel/project.json": "excluded",
            "scripts/export_research.py": "excluded", "tests/test_private.py": "excluded",
            "docs/superpowers/internal.md": "excluded", "public/.DS_Store": "excluded",
            "package-lock.json": json.dumps({"name": "notebook", "lockfileVersion": 3, "packages": {}}),
            "package.json": json.dumps({"name": "notebook", "scripts": {"test": "node --test", "build": "node build.mjs", "copy:check": "tsx scripts/check-experiment-copy.ts", "test:data": "python3 private.py", "deploy": "vercel --prod"}}),
        }
        for name, text in files.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)

    def test_allowlist_is_portable_and_does_not_modify_inputs(self):
        original = (self.root / "package.json").read_bytes()
        payloads, meta = pack.collect_payloads(self.root)
        self.assertEqual((self.root / "package.json").read_bytes(), original)
        self.assertEqual(json.loads(payloads["package.json"])["scripts"], {"test": "node --test", "build": "node build.mjs", "copy:check": "tsx scripts/check-experiment-copy.ts"})
        self.assertEqual(payloads["docs/MAINTENANCE.md"], b"Portable maintenance\n")
        self.assertIn("public/source/example.txt", payloads)
        self.assertIn("content/journal.json", payloads)
        self.assertIn("tests/fixtures/register-ids.json", payloads, "JavaScript tests read their JSON fixtures")
        self.assertEqual(meta["snapshotId"], "test-snapshot")
        forbidden = ["node_modules/", "dist/", ".vercel/", "export_research", "test_private", "superpowers/", "notebook-source.zip", ".DS_Store"]
        self.assertFalse(any(term in name for name in payloads for term in forbidden))

    def test_zip_bytes_are_deterministic_and_manifest_hashes_match(self):
        payloads, meta = pack.collect_payloads(self.root)
        first, second = self.root / "first.zip", self.root / "second.zip"
        manifest = pack.create_archive(first, payloads, meta)
        pack.create_archive(second, dict(reversed(list(payloads.items()))), meta)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.namelist(), sorted(archive.namelist()))
            self.assertTrue(all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in archive.infolist()))
            self.assertEqual(json.loads(archive.read("MANIFEST.json")), manifest)
            for row in manifest["files"]:
                value = archive.read(row["path"])
                self.assertEqual(row["sha256"], hashlib.sha256(value).hexdigest())
                self.assertEqual(row["bytes"], len(value))
        self.assertEqual(pack.verify_archive(first)["files"], len(payloads) + 1)

    def test_private_content_in_text_is_rejected_without_printing_it(self):
        marker = "/Users/private-person/secret-project"
        (self.root / "public/source/example.txt").write_text(marker)
        with self.assertRaisesRegex(ValueError, "Privacy") as failure:
            pack.collect_payloads(self.root)
        self.assertNotIn(marker, str(failure.exception))

    def test_nested_zip_is_scanned_for_private_content(self):
        target = self.root / "public/assets/evidence.zip"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("notes.txt", "/home/private-account/research")
        with self.assertRaisesRegex(ValueError, "Privacy"):
            pack.collect_payloads(self.root)

    def test_symlinks_are_rejected(self):
        (self.root / "public/source/link.txt").symlink_to(self.root / "docs/MAINTENANCE.md")
        with self.assertRaisesRegex(ValueError, "symbolic"):
            pack.collect_payloads(self.root)

    def test_symlinked_parent_directories_are_rejected(self):
        (self.root / "docs").rename(self.root / "other-docs")
        (self.root / "docs").symlink_to(self.root / "other-docs", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            pack.collect_payloads(self.root)

    def test_optional_hosting_ignore_config_is_included(self):
        (self.root / ".vercelignore").write_text("scripts/*.py\n")
        payloads, _ = pack.collect_payloads(self.root)
        self.assertEqual(payloads[".vercelignore"], b"scripts/*.py\n")

    def test_unsynchronized_journal_is_rejected(self):
        (self.root / "content/journal.json").write_text('[{"id":"J-NEW"}]')
        with self.assertRaisesRegex(ValueError, "journal.*synchron|synchron.*journal"):
            pack.collect_payloads(self.root)

    def test_artifact_dates_and_validator_travel_with_the_portable_evidence(self):
        payloads, _ = pack.collect_payloads(self.root)
        self.assertIn("scripts/sync-artifact-dates.mjs", payloads)
        self.assertIn("content/artifact-dates.json", payloads)
        self.assertEqual(payloads["content/artifact-dates.json"], payloads["public/data/artifact-dates.json"])

    def test_experiment_copy_and_publication_gate_travel_together(self):
        payloads, _ = pack.collect_payloads(self.root)
        self.assertIn("content/experiment-copy.json", payloads)
        self.assertIn("scripts/check-experiment-copy.ts", payloads)

    def test_unsynchronized_artifact_dates_are_rejected(self):
        (self.root / "content/artifact-dates.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "[Dd]ate.*synchron|synchron.*date"):
            pack.collect_payloads(self.root)

    def test_manifest_tampering_is_detected(self):
        payloads, meta = pack.collect_payloads(self.root)
        target = self.root / "notebook.zip"
        pack.create_archive(target, payloads, meta)
        with zipfile.ZipFile(target) as archive:
            values = {name: archive.read(name) for name in archive.namelist()}
        values["src/main.tsx"] = b"changed"
        with zipfile.ZipFile(target, "w") as archive:
            for name, value in values.items():
                archive.writestr(name, value)
        with self.assertRaisesRegex(ValueError, "hash|manifest"):
            pack.verify_archive(target)


if __name__ == "__main__":
    unittest.main()
