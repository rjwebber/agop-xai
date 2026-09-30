"""Focused tests for the read-only public-release verifier."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.verify_release import (
    LEGACY_NUMBERED_TABLE,
    compare_inventory,
    verify_accepted_reports,
    verify_checksum_tree,
    verify_metadata,
)


class ReleaseVerifierTests(unittest.TestCase):
    def test_checksum_tree_checks_hashes_and_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.txt"
            payload.write_text("public\n", encoding="utf-8")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            (root / "SHA256SUMS").write_text(
                f"{digest}  ./payload.txt\n", encoding="utf-8"
            )
            errors: list[str] = []
            verify_checksum_tree(root, errors)
            self.assertEqual(errors, [])

            (root / "extra.txt").write_text("unlisted\n", encoding="utf-8")
            verify_checksum_tree(root, errors)
            self.assertTrue(any("missing from checksum" in error for error in errors))

    def test_inventory_rejects_missing_and_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "kept.txt").write_text("kept\n", encoding="utf-8")
            (root / "extra.txt").write_text("extra\n", encoding="utf-8")
            errors: list[str] = []
            compare_inventory(root, {"kept.txt", "missing.txt"}, "test", errors)
            self.assertTrue(any("missing test file" in error for error in errors))
            self.assertTrue(any("unexpected test file" in error for error in errors))

    def test_accepted_report_archive_is_json_and_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(
                json.dumps({"accepted_reports": {"member_count": 1}}),
                encoding="utf-8",
            )
            with zipfile.ZipFile(root / "accepted_reports.zip", "w") as archive:
                archive.writestr("figure/example.json", json.dumps({"path": "data/x"}))
            errors: list[str] = []
            verify_accepted_reports(root, errors)
            self.assertEqual(errors, [])

    def test_accepted_report_archive_rejects_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(
                json.dumps({"accepted_reports": {"member_count": 1}}),
                encoding="utf-8",
            )
            with zipfile.ZipFile(root / "accepted_reports.zip", "w") as archive:
                archive.writestr(
                    "figure/example.json", json.dumps({"path": "/Users/example/run"})
                )
            errors: list[str] = []
            verify_accepted_reports(root, errors)
            self.assertTrue(any("machine-local" in error for error in errors))

    def test_legacy_numbered_table_pattern_is_specific(self) -> None:
        self.assertIsNotNone(LEGACY_NUMBERED_TABLE.search("old " + "Table" + " 2"))
        self.assertIsNotNone(LEGACY_NUMBERED_TABLE.search("old_" + "table" + "2"))
        self.assertIsNone(LEGACY_NUMBERED_TABLE.search("table12"))
        self.assertIsNone(LEGACY_NUMBERED_TABLE.search("table1"))

    def test_metadata_versions_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".github/workflows").mkdir(parents=True)
            (root / "src/zc_xai").mkdir(parents=True)
            required_text = {
                ".gitattributes": "*.F -whitespace\n",
                ".gitignore": "\n",
                ".github/workflows/ci.yml": "name: CI\n",
                "CHANGELOG.md": "# Changelog\n\n## 1.0.0 — release\n",
                "CITATION.cff": "version: 1.0.0\n",
                "LICENSE": "BSD 3-Clause License\n",
                "LICENSE-DATA": "CC BY 4.0\n",
                "README.md": "release\n",
                "RELEASE_CONTENTS.md": "release\n",
                "THIRD_PARTY_NOTICES.md": "third-party notices\n",
                "pyproject.toml": '[project]\nversion = "1.0.0"\n',
                "src/zc_xai/__init__.py": '__version__ = "1.0.0"\n',
            }
            for relative, text in required_text.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")

            errors: list[str] = []
            verify_metadata(root, errors)
            self.assertEqual(errors, [])

            (root / "src/zc_xai/__init__.py").write_text(
                '__version__ = "0.1.0"\n', encoding="utf-8"
            )
            verify_metadata(root, errors)
            self.assertTrue(any("version mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
