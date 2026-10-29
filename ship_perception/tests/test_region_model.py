"""Opening topology and measurement failure are independent contracts."""

import unittest

from ship_perception.perception.region_model import (
    OpeningRegion,
    TopologySupport,
    BoundaryMeasurement,
)


class RegionMeasurementContracts(unittest.TestCase):
    def region(self):
        return OpeningRegion(
            "R",
            "V",
            (0, 0, 12, 6),
            (-2, -2, 14, 8),
            (TopologySupport("BEV_ENCLOSURE", "SUPPORTING"),),
        )

    def test_region_survives_without_three_measured_sides(self):
        region = self.region()
        self.assertEqual(len(region.boundary_measurements), 4)
        self.assertTrue(
            all(
                s.state == "CONSERVATIVE_PRIOR"
                for s in region.boundary_measurements.values()
            )
        )
        self.assertEqual(region.status, "REGION_AMBIGUOUS")

    def test_topology_support_independent_from_boundary_measurement(self):
        region = self.region()
        region.boundary_measurements["V0"] = BoundaryMeasurement("V0", "UNOBSERVED")
        self.assertEqual(region.topology_support[0].state, "SUPPORTING")
        self.assertEqual(region.status, "REGION_AMBIGUOUS")
        with self.assertRaises(ValueError):
            TopologySupport("HEIGHT_TRANSITION", "STRONG")

    def test_precision_failure_preserves_region(self):
        region = self.region()
        before = region.audit()
        region.record_precision(
            dict(status="UNRESOLVED", error_code="NO_RAW_ROI_SUPPORT", sides={})
        )
        self.assertEqual(region.region_id, before["region_id"])
        self.assertEqual(
            region.conservative_bounds, tuple(before["conservative_bounds"])
        )
        self.assertEqual(region.precision_status, "UNRESOLVED")

    def test_search_envelope_survives_boundary_shrink(self):
        region = self.region()
        before = region.search_envelope
        region.record_precision(
            dict(status="UNRESOLVED", bounds_axial=[2, 2, 10, 4], sides={})
        )
        self.assertEqual(region.search_envelope, before)
        region.expand_search((3, 2, 8, 5))
        self.assertEqual(region.search_envelope, before)

    def test_context_only_and_degenerate_boxes_are_not_opening_identity(self):
        with self.assertRaises(ValueError):
            OpeningRegion(
                "R",
                "V",
                (0, 0, 12, 6),
                (0, 0, 12, 6),
                (TopologySupport("VESSEL_INTERIOR_CONTEXT", "STRONG"),),
            )
        with self.assertRaises(ValueError):
            OpeningRegion(
                "R",
                "V",
                (0, 0, 0, 6),
                (0, 0, 0, 6),
                (TopologySupport("BEV_ENCLOSURE", "WEAK"),),
            )


if __name__ == "__main__":
    unittest.main()
