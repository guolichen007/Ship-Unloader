"""Visibility is distinct from absence and sparse support keeps real gaps."""

import unittest
import numpy as np

from ship_perception.perception.region_model import EvidenceSource
from ship_perception.perception.structural_family import (
    StructuralBoundaryFamily,
    visibility_segments,
    merge_intervals,
)


class StructuralFamilyContracts(unittest.TestCase):
    def test_unobserved_longitudinal_segment_not_negative_evidence(self):
        segments = visibility_segments(
            np.arange(5.0) + 0.5, 1.0, [8, 0, 0, 0, 8], [(0, 1), (4, 5)]
        )
        self.assertEqual(
            segments,
            (
                (0.0, 1.0, "OBSERVED_SUPPORTED"),
                (1.0, 4.0, "UNOBSERVED"),
                (4.0, 5.0, "OBSERVED_SUPPORTED"),
            ),
        )
        self.assertNotIn("OBSERVED_CONTRADICTED", [s[2] for s in segments])

    def test_longitudinal_family_sparse_collinear_support(self):
        intervals = merge_intervals([(0, 1), (4, 5), (9, 10)])
        segments = visibility_segments(
            np.arange(10.0) + 0.5, 1.0, [10, 0, 0, 0, 10, 0, 0, 0, 0, 10], intervals
        )
        source = EvidenceSource(
            "map", "lidar", "1.0", (0, 1, 2), tuple(tuple(r) for r in np.eye(4)), ("V",)
        )
        family = StructuralBoundaryFamily(
            "F",
            "V",
            1,
            2.0,
            (2.0,),
            intervals,
            segments,
            ("P",),
            ("RAW3D_STRUCTURAL_PROVIDER", "BEV_RECTILINEAR_PROVIDER"),
            ("RAW3D_PROFILE_BREAK",),
            source,
            1.0,
            "UNRESOLVED",
        )
        self.assertEqual(family.observable_support_ratio, 1.0)
        self.assertEqual(family.audit()["gap_distribution_m"], [3.0, 4.0])
        self.assertEqual(family.topology().state, "SUPPORTING")
        self.assertFalse(family.audit()["rho_copy_allowed"])

    def test_visible_without_structure_is_not_automatic_contradiction(self):
        result = visibility_segments(np.array([0.5]), 1.0, [10], [])
        self.assertEqual(result[0][2], "OCCLUDED_OR_SPARSE")

    def test_other_vessel_or_context_cannot_own_family(self):
        for ownership, vessel in (
            ("ACTIVE_VESSEL_PRIMARY", "OTHER"),
            ("UNKNOWN_CONTEXT", "V"),
        ):
            source = EvidenceSource(
                "map",
                "lidar",
                "1.0",
                (0,),
                tuple(tuple(r) for r in np.eye(4)),
                (vessel,),
                ownership,
            )
            with self.assertRaises(ValueError):
                StructuralBoundaryFamily(
                    "F",
                    "V",
                    1,
                    2.0,
                    (2.0,),
                    ((0.0, 1.0),),
                    ((0.0, 1.0, "OBSERVED_SUPPORTED"),),
                    ("P",),
                    ("RAW3D_STRUCTURAL_PROVIDER",),
                    ("RAW3D_PROFILE_BREAK",),
                    source,
                    1.0,
                    "UNRESOLVED",
                )


if __name__ == "__main__":
    unittest.main()
