"""Physical counterexamples for observed topology and conservative precision."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ship_perception.io.frame import PointCloudFrame
from ship_perception.io.pcd_source import read_pcd
from ship_perception.perception.context import (
    VesselEvidence,
    freeze,
)
from ship_perception.perception.final_boundary import FinalBoundaryRefiner, guard_move
from ship_perception.perception.intensity import admission_stats
from ship_perception.perception.sections import physical_sections
from ship_perception.perception.validation import ExteriorTopology
from ship_perception.r1_static.run import resolve_config


class PhysicalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = resolve_config()

    def test_physical_section_scale(self):
        self.assertEqual(physical_sections((0, 0.4), 0.05, self.config), [])
        self.assertLess(
            len(physical_sections((0, 2), 0.05, self.config)),
            len(physical_sections((0, 80), 0.05, self.config)),
        )

    def test_cargo_ridge_cannot_crop_without_independent_sections(self):
        rho, status, risk = guard_move(0.0, 2.0, 1, 1, False, self.config)
        self.assertEqual(
            (rho, status, risk), (0.0, "KEEP_CONSERVATIVE_PREVIOUS_BOUNDARY", True)
        )
        self.assertEqual(guard_move(0.0, 2.0, 1, 6, True, self.config)[0], 2.0)

    def test_four_sides_raw_and_phase(self):
        u, v = np.meshgrid(np.arange(-2, 14, 0.08), np.arange(-2, 12, 0.08))
        inside = (u > 1) & (u < 11) & (v > 1) & (v < 9)
        xyz = np.column_stack(
            (u.ravel(), v.ravel(), np.where(inside, -4.0, 0.0).ravel())
        )
        proposal = dict(bounds_axial=[1.0, 1.0, 11.0, 9.0], vessel_hypothesis_id="V0")
        result = FinalBoundaryRefiner(xyz, [], self.config).refine(proposal, np.eye(2))
        self.assertEqual(set(result["sides"]), {"U0", "U1", "V0", "V1"})
        for side in result["sides"].values():
            self.assertEqual(len(side["phase_runs"]), 4)
            self.assertGreaterEqual(side["support_section_count"], 4)
            self.assertGreaterEqual(side["raw_subvoxel_section_count"], 4)
        # A rigid display transform does not change the raw UV inference.
        theta = 0.721
        axes = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        other = FinalBoundaryRefiner(xyz[::-1], [], self.config).refine(proposal, axes)
        np.testing.assert_allclose(
            np.array(other["boundary_xy"]) @ axes.T, result["boundary_uv"], atol=1e-9
        )

    def topology(self, *, side=1, spacing=0.2, missing=False, bridge=False):
        u, v = np.meshgrid(np.arange(0, 40, spacing), np.arange(-12, 12, spacing))
        boat = abs(v) <= 4
        hatch = (u > 8) & (u < 22) & (abs(v) < 2)
        z = np.where(boat, np.where(hatch, -5.0, 0.0), -2.0)
        if bridge:
            z[(abs(u - 15) < 1.0) & (v > 1) & (v < 8)] = -2.0
        keep = np.ones(u.shape, bool)
        if missing:
            keep &= ~((abs(v) > 4) & (abs(v) < 5))
        xyz = np.column_stack((u[keep], v[keep], z[keep]))
        owned = boat[keep]
        frame = PointCloudFrame(xyz, None, "test", 0, 0, "", ("x", "y", "z"), len(xyz))
        vessel = VesselEvidence(
            freeze(dict(vessel_hypothesis_id="V0", local_axes=[[1, 0], [0, 1]])),
            np.flatnonzero(owned),
            "test",
        )
        return ExteriorTopology.build(frame, vessel, self.config)

    def test_exterior_water_counterexamples(self):
        for spacing in (0.1, 0.45):
            for missing in (False, True):
                for bridge in (False, True):
                    with self.subTest(spacing=spacing, missing=missing, bridge=bridge):
                        topology = self.topology(
                            spacing=spacing, missing=missing, bridge=bridge
                        )
                        for box in ([2, 6, 30, 10], [2, -10, 30, -6]):
                            result = topology.review(box, 0.0)
                            self.assertEqual(
                                result["status"], "REJECT_EXTERIOR_LOW_REGION"
                            )
                            self.assertFalse(result["missing_cells_traversed"])
                        deep = topology.review([8, -2, 22, 2], 0.0)
                        self.assertEqual(
                            deep["status"],
                            (
                                "REJECT_EXTERIOR_LOW_REGION"
                                if bridge and not missing
                                else "ENCLOSED_OBSERVED_LOW_COMPONENT"
                            ),
                        )

    def test_constant_intensity_admission(self):
        stats = admission_stats(np.full(1000, 42.0))
        self.assertEqual(stats["status"], "UNAVAILABLE_CONSTANT_DEVICE_SIGNAL")
        self.assertEqual(stats["unique_count"], 1)
        self.assertEqual(stats["source_calibration"], "UNKNOWN")
        self.assertFalse(stats["used"])
        self.assertEqual(admission_stats(None)["finite_count"], 0)

    def test_unresolved_boundary_band_is_neither_known_exterior_nor_confirmed(self):
        topology = self.topology()
        u, v = topology.grid.centers()
        # A B1 split cuts the measured hold, but does not delete its points.
        unknown = topology.owned & (v < -0.5) & (v > -2)
        topology.owned[unknown] = False
        topology.unresolved_context = unknown
        review = topology.review([8, -2, 22, 2], 0)
        self.assertEqual(review["status"], "UNRESOLVED_EXTERIOR_CONNECTIVITY")
        self.assertIsNone(review["exterior_low_connected"])
        self.assertFalse(review["known_exterior_low_connected"])
        self.assertGreater(review["unresolved_context_component_cells"], 0)
        self.assertEqual(review["known_external_component_cells"], 0)
        self.assertEqual(
            topology.review([2, 6, 30, 10], 0)["status"], "REJECT_EXTERIOR_LOW_REGION"
        )

    def test_scene_boundary_still_seeds_exterior_even_for_unresolved_measurements(self):
        topology = self.topology()
        topology.unresolved_context = ~topology.owned
        review = topology.review([2, 6, 30, 10], 0)
        self.assertTrue(review["known_exterior_low_connected"])
        self.assertEqual(review["status"], "UNRESOLVED_EXTERIOR_CONNECTIVITY")
        self.assertFalse(review["missing_cells_traversed"])


if __name__ == "__main__":
    unittest.main()
