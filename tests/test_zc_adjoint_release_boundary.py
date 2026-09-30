from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "adjoint" / "full_tangent_audit"
INVENTORY = AUDIT_ROOT / "RELEASE_FILES.txt"
TOOLCHAIN_ROOT = PROJECT_ROOT / "adjoint" / "tapenade_toolchain"


def _release_inventory() -> set[str]:
    return {
        line.strip()
        for line in INVENTORY.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


class AdjointReleaseBoundaryTests(unittest.TestCase):
    def test_full_tangent_audit_denies_generated_directories_by_default(
        self,
    ) -> None:
        rules = {
            line.strip()
            for line in (AUDIT_ROOT / ".gitignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("/*/", rules)
        self.assertIn("*_console.json", rules)
        self.assertIn("*_stdout.json", rules)

    def test_full_tangent_release_inventory_is_complete_and_safe(self) -> None:
        inventoried = _release_inventory()
        visible_files = {
            path.name
            for path in AUDIT_ROOT.iterdir()
            if path.is_file()
            and path.name != ".DS_Store"
            and not path.name.endswith(("_console.json", "_stdout.json"))
            and not path.name.endswith((".pyc", ".pyo"))
        }
        self.assertEqual(inventoried, visible_files)

        for relative in inventoried:
            self.assertEqual(relative, Path(relative).name)
            path = AUDIT_ROOT / relative
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())

    def test_toolchain_ignores_build_but_releases_authored_modifications(
        self,
    ) -> None:
        rules = {
            line.strip()
            for line in (TOOLCHAIN_ROOT / ".gitignore")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("/build/", rules)
        self.assertNotIn("/patches/", rules)
        self.assertNotIn("/coupled/", rules)

        # The reproducibility recipe, configuration contract, validation
        # drivers, authored ZC-derived modifications, and curated evidence
        # remain visible; only generated/local build products are excluded.
        for relative in (
            "README.md",
            "VERSION.env",
            "config/supported_runtime_config.json",
            "patches/cforce_ad_ready.patch",
            "coupled/zc_kernel_nino3.F",
            "scripts/compile_coupled_adjoint.sh",
            "validation/mloop_ad_test.F",
            "results/mloop_ad_validation.txt",
        ):
            path = TOOLCHAIN_ROOT / relative
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())


if __name__ == "__main__":
    unittest.main()
