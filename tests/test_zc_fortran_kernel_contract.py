from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "adjoint" / "fortran_kernel"


def _manifest() -> dict:
    return json.loads((KERNEL / "state_manifest.json").read_text())


class FortranKernelContractTests(unittest.TestCase):
    def test_manifest_segments_are_contiguous_and_complete(self) -> None:
        manifest = _manifest()
        for specification in manifest["arrays"].values():
            cursor = 0
            for segment in specification["segments"]:
                self.assertEqual(segment["start"], cursor)
                self.assertGreater(segment["stop"], segment["start"])
                self.assertEqual(segment["fortran_start"], segment["start"] + 1)
                self.assertEqual(segment["fortran_end"], segment["stop"])
                self.assertEqual(
                    math.prod(segment["shape"]),
                    segment["stop"] - segment["start"],
                )
                self.assertEqual(segment["order"], "F")
                self.assertIn(segment["activity"], {"active", "diagnostic", "passive"})
                cursor = segment["stop"]
            self.assertEqual(cursor, specification["length"])

    def test_named_scientific_slices_are_exact(self) -> None:
        real_segments = {
            item["name"]: item
            for item in _manifest()["arrays"]["real32"]["segments"]
        }
        expected = {
            "TO": (32121, 33141),
            "U1": (33141, 34161),
            "V1": (34161, 35181),
            "H1": (35181, 36201),
        }
        for name, (start, stop) in expected.items():
            segment = real_segments[name]
            self.assertEqual((segment["start"], segment["stop"]), (start, stop))
            self.assertEqual(segment["shape"], [30, 34])
            self.assertEqual(segment["activity"], "active")

    def test_fortran_lengths_match_manifest(self) -> None:
        include = (KERNEL / "zc_kernel_state.inc").read_text()
        manifest = _manifest()["arrays"]
        expected = {
            "ZC_STATE_NREAL": manifest["real32"]["length"],
            "ZC_STATE_NCOMPLEX": manifest["complex64"]["length"],
            "ZC_STATE_NDOUBLE": manifest["real64"]["length"],
            "ZC_STATE_NINTEGER": manifest["integer"]["length"],
            "ZC_STATE_NTIME": manifest["passive_time_real32"]["length"],
        }
        for name, value in expected.items():
            match = re.search(rf"PARAMETER \({name}=(\d+)\)", include)
            self.assertIsNotNone(match)
            assert match is not None
            self.assertEqual(int(match.group(1)), value)

    def test_scientific_control_mask_excludes_restart_workspace(self) -> None:
        real_segments = _manifest()["arrays"]["real32"]["segments"]
        independently_controlled = {
            segment["name"]
            for segment in real_segments
            if segment.get("independent_control", False)
        }
        self.assertIn("TO", independently_controlled)
        self.assertLessEqual({"AKB", "UB", "HB", "V"}, independently_controlled)
        self.assertNotIn("AK", independently_controlled)
        self.assertFalse(
            {
                "H1",
                "U1",
                "V1",
                "HTAU",
                "US",
                "VS",
                "WP",
                "QF",
            }
            & independently_controlled
        )

    def test_kelvin_wave_memory_closes_legacy_restart_gap(self) -> None:
        manifest = _manifest()
        real_segments = {
            segment["name"]: segment
            for segment in manifest["arrays"]["real32"]["segments"]
        }
        ak = real_segments["AK"]
        self.assertEqual((ak["start"], ak["stop"]), (59069, 59148))
        self.assertEqual(ak["shape"], [79])
        self.assertIsNone(ak["restart_record"])
        self.assertEqual(ak["activity"], "active")
        self.assertFalse(ak["independent_control"])
        self.assertNotIn("AK", manifest["excluded_overwritten_workspace"])

    def test_complex_atmosphere_arrays_are_restart_workspace(self) -> None:
        complex_segments = _manifest()["arrays"]["complex64"]["segments"]
        self.assertEqual(
            {segment["name"] for segment in complex_segments},
            {"Q", "E1", "E2"},
        )
        for segment in complex_segments:
            self.assertEqual(segment["activity"], "diagnostic")
            self.assertFalse(segment["independent_control"])


if __name__ == "__main__":
    unittest.main()
