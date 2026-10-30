"""Review identity requires owned current topology, not a measured steel rim."""

import unittest
from types import SimpleNamespace
import numpy as np

from ship_perception.perception.s4_topology_bridge import S4TopologyBridge
from ship_perception.r1_static.run import resolve_config


class ReviewIdentityContracts(unittest.TestCase):
    def bridge(self):
        b = S4TopologyBridge.__new__(S4TopologyBridge)
        b.config, _ = resolve_config()
        b.vessel_id = "V"
        b.axes = list(np.eye(2))
        b.grid = SimpleNamespace(cell_m=1.0)
        x, y = np.meshgrid(np.arange(12.0), np.arange(6.0))
        b.xy = np.stack((x, y), -1)
        b.projected = b.xy
        b.families = [
            dict(
                family_id="END" + str(i),
                axis_role="TRANSVERSE",
                direction_axis=1,
                rho=rho,
                rho_uncertainty=0.05,
                observed_intervals=[[0.0, 5.0]],
                support_span=[0.0, 5.0],
                evidence_types=[kind],
            )
            for i, (rho, kind) in enumerate(
                [(0.0, "OBSERVED_PROFILE_BREAK"), (11.0, "OBSERVED_3D_FACE")]
            )
        ]
        return b, x, y

    def test_opposite_current_raw_ends_supply_topology_without_four_line_product(self):
        b, x, y = self.bridge()
        self.assertEqual(
            b._opposite_terminals(["END0", "END1"], np.ones(x.shape, bool)),
            ["END0", "END1"],
        )
        self.assertEqual(b._opposite_terminals(["END0"], np.ones(x.shape, bool)), [])
        b.families[1]["evidence_types"] = ["HEIGHT_TRANSITION"]
        self.assertEqual(
            b._opposite_terminals(["END0", "END1"], np.ones(x.shape, bool)), []
        )

    def test_actual_parent_member_cells_attach_missing_terminal_search_evidence(self):
        b, x, y = self.bridge()
        b.nodes = {
            "LOW": SimpleNamespace(
                mask=(x >= 5) & (x <= 6) & (y < 2),
                bbox_xy=(5.0, 0.0, 6.0, 1.0),
                level_m=-2.0,
            ),
            "HIGH": SimpleNamespace(
                mask=np.ones(x.shape, bool), bbox_xy=(0.0, 0.0, 11.0, 5.0), level_m=-1.0
            ),
        }
        b.vertices = []
        b._vertices()
        low = next(v for v in b.vertices if v["node_id"] == "LOW")
        self.assertEqual(low["side_family_ids"], ["END0", "END1"])
        self.assertGreater(
            low["terminal_topology_assignment"][0]["observed_member_overlap"], 0
        )
        # An empty bbox overlap must not acquire the parent's wall evidence.
        b.nodes["HIGH"].mask = y >= 3
        b.vertices = []
        b._vertices()
        low = next(v for v in b.vertices if v["node_id"] == "LOW")
        self.assertEqual(low["side_family_ids"], [])

    def test_different_separator_domains_cannot_borrow_parent_terminals(self):
        b, x, y = self.bridge()
        b.nodes = {
            "LOW": SimpleNamespace(
                mask=(x >= 5) & (x <= 6) & (y < 2),
                bbox_xy=(5.0, 0.0, 6.0, 1.0),
                level_m=-2.0,
                opening_domain=dict(domain_id="D", side=0),
            ),
            "HIGH": SimpleNamespace(
                mask=np.ones(x.shape, bool),
                bbox_xy=(0.0, 0.0, 11.0, 5.0),
                level_m=-1.0,
                opening_domain=dict(domain_id="D", side=1),
            ),
        }
        b.vertices = []
        b._vertices()
        low = next(v for v in b.vertices if v["node_id"] == "LOW")
        self.assertEqual(low["side_family_ids"], [])


if __name__ == "__main__":
    unittest.main()
