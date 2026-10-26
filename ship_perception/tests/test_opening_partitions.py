"""Preserve independently measured cells without splitting ambiguous cargo bands."""

import copy
import unittest

import numpy as np

from ship_perception.perception.pipeline import precision_suppression
from ship_perception.r1_static.run import resolve_config


def candidate(ident, box, partial_side=None):
    u0, v0, u1, v1 = box
    names = ("U0", "U1", "V0", "V1")
    return dict(
        proposal_id=ident,
        vessel_hypothesis_id="V0",
        topology_status="INDEPENDENT_OPENING_GROUP",
        containment=dict(status="CONTAINED"),
        exterior=dict(status="ENCLOSED_OBSERVED_LOW_COMPONENT"),
        boundary_xy=[[u0, v0], [u1, v0], [u1, v1], [u0, v1]],
        undercoverage_risk=partial_side is not None,
        sides={
            name: dict(
                rho_after_m=rho,
                raw_subvoxel_section_count=2 if name == partial_side else 5,
                positive_height_section_count=2 if name == partial_side else 5,
                support_section_ratio=0.4 if name == partial_side else 1.0,
                phase_runs=[dict(physical_section_count=5)],
            )
            for name, rho in zip(names, (u0, u1, v0, v1))
        },
        deck_ring=dict(
            sides={
                name: dict(
                    raw3d_support=True,
                    bev_support=True,
                    coverage=1.0,
                    signed_height_rise_m=2.0,
                )
                for name in names
            }
        ),
    )


def scene():
    parent = candidate("envelope", [-20, 3, 35, 17])
    parent.update(topology_status="AMBIGUOUS_TOPOLOGY", topology_ambiguity=[1, 3.2])
    return [
        parent,
        candidate("left", [-20, 3, 1, 17], "U1"),
        candidate("right", [3.2, 3, 35, 17], "U0"),
    ]


class OpeningPartitionTests(unittest.TestCase):
    def setUp(self):
        self.config, _ = resolve_config()

    def test_supported_partial_cells_survive_stronger_ambiguous_envelope(self):
        rows = scene()
        before = copy.deepcopy(rows)
        selected, suppressed = precision_suppression(rows, np.eye(2), self.config)
        self.assertEqual([r["proposal_id"] for r in selected], ["left", "right"])
        self.assertEqual(selected, rows[1:])
        self.assertEqual(rows, before)
        self.assertEqual(suppressed[0]["explained_by_proposal_ids"], ["left", "right"])
        self.assertFalse(suppressed[0]["beam_width_gate_changed"])
        self.assertGreater(2.2, self.config["boundary"]["beam_max_width_m"])

    def test_broad_cargo_band_cannot_be_resolved_by_competing_children(self):
        rows = scene()
        rows[0]["topology_ambiguity"] = dict(unresolved_transverse_band_m=[1, 15])
        selected, _ = precision_suppression(rows, np.eye(2), self.config)
        self.assertEqual([r["proposal_id"] for r in selected], ["envelope"])

    def test_one_child_or_unmeasured_separator_cannot_resolve_parent(self):
        variants = [scene()[:2], scene(), scene(), scene()]
        variants[1][1]["deck_ring"]["sides"]["U1"]["raw3d_support"] = False
        variants[2][2]["deck_ring"]["sides"]["U0"]["signed_height_rise_m"] = -2.0
        variants[3][2]["vessel_hypothesis_id"] = "OTHER_VESSEL"
        for rows in variants:
            with self.subTest(rows=rows):
                selected, _ = precision_suppression(rows, np.eye(2), self.config)
                self.assertEqual([r["proposal_id"] for r in selected], ["envelope"])

    def test_supported_partition_is_invariant_to_input_yaw_and_translation(self):
        rows = scene()
        angle = 0.67
        axes = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        offset_uv = np.array([123.4, -51.2])
        for row in rows:
            row["boundary_xy"] = (
                (np.asarray(row["boundary_xy"]) + offset_uv) @ axes
            ).tolist()
            for name, side in row["sides"].items():
                side["rho_after_m"] += offset_uv[0 if name.startswith("U") else 1]
        rows[0]["topology_ambiguity"] = (
            np.asarray(rows[0]["topology_ambiguity"]) + offset_uv[0]
        ).tolist()
        selected, _ = precision_suppression(rows, axes, self.config)
        self.assertEqual([r["proposal_id"] for r in selected], ["left", "right"])


if __name__ == "__main__":
    unittest.main()
