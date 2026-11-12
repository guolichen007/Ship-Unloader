"""Falsifiable review-layer boundaries; no external steel accuracy claim."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ship_perception.r1_static.run import resolve_config
from ship_perception.perception.review_geometry_recovery import (
    recover_review_geometries,
    select_supported,
    joint_bounds,
)
from ship_perception.perception.review_labels import review_objects


def mode(position, score=1.0):
    return dict(
        mode_id=str(position),
        qualified_for_review=True,
        score=score,
        coverage=1.0,
        holdout_coverage=1.0,
        position_m=position,
        position_uncertainty_m=0.15,
    )


def test_ambiguous_parallel_transitions_remain_mutually_exclusive():
    cfg = resolve_config()[0]
    entry = dict(candidates=[mode(2.0), mode(4.0)], holdout_section_ids=[1, 3, 5, 7])
    best, alternatives = select_supported(entry, cfg)
    assert best is None and len(alternatives) == 2
    best, _ = select_supported(entry, cfg, [(4.0, 0.15)])
    assert best["position_m"] == 4.0


def test_genuine_unequal_widths_are_not_pulled_together():
    cfg = resolve_config()[0]
    cells = [dict(cell_id="a"), dict(cell_id="b")]
    plans = {
        key: dict(
            old_bounds=[0.0, 0.0, 24.0, width],
            vessel_id="v",
            selected={"V0": mode(0.0), "V1": mode(width)},
        )
        for key, width in (("a", 8.0), ("b", 14.0))
    }
    bounds, factors = joint_bounds(cells, plans, cfg)
    np.testing.assert_allclose(bounds["a"], [0.0, 0.0, 24.0, 8.0], atol=1e-10)
    np.testing.assert_allclose(bounds["b"], [0.0, 0.0, 24.0, 14.0], atol=1e-10)
    assert factors[0]["soft_weight"] < 1e-20


def fixture(flat=False):
    cfg = resolve_config()[0]
    u, v = np.meshgrid(np.arange(-1.0, 56.0, 0.1), np.arange(-1.0, 14.0, 0.1))
    z = np.zeros_like(u) if flat else np.where((v < 2.0) | (v > 10.0), 0.0, -2.0)
    points = np.c_[u.ravel(), v.ravel(), z.ravel()]
    ids = np.arange(len(points))
    masks = {}
    cells = []
    regions = {}
    for key, left in (("a", 0.0), ("b", 30.0)):
        polygon = [[left, 0.0], [left + 24.0, 0.0], [left + 24.0, 12.0], [left, 12.0]]
        cells.append(
            dict(
                cell_id=key,
                vessel_id="v",
                candidate_ids=[key],
                primary_region_id=key,
                strict_rectangle_state="INFEASIBLE",
                geometry=dict(
                    boundary_xy=polygon, corners_xyz=[[x, y, 7.0] for x, y in polygon]
                ),
                separator_constraints=[],
                width_recovery_requested_sides=["V0"],
            )
        )
        masks[key] = (
            (points[:, 0] >= left)
            & (points[:, 0] <= left + 24.0)
            & (points[:, 1] >= 0.0)
            & (points[:, 1] <= 12.0)
        )
        regions[key] = SimpleNamespace(
            search_envelope=[left - 1.0, -1.0, left + 25.0, 13.0]
        )
    local = SimpleNamespace(
        xyz=points,
        ids=ids,
        owned=np.ones(len(ids), bool),
        index=SimpleNamespace(
            query_indexes=lambda bounds: ids[
                np.all(
                    (points[:, :2] >= bounds[:2]) & (points[:, :2] <= bounds[2:]),
                    axis=1,
                )
            ]
        ),
    )
    measurement = SimpleNamespace(
        local=SimpleNamespace(vessel=SimpleNamespace(axes=np.eye(2)))
    )
    return cfg, cells, regions, masks, local, measurement


def run(cells, regions, masks, local, measurement, cfg):
    with patch(
        "ship_perception.perception.review_geometry_recovery.cell_members",
        side_effect=lambda cell, *args: (masks[cell["cell_id"]].copy(), {}),
    ):
        recover_review_geometries(cells, regions, measurement, cfg, local)


def test_changed_review_is_visible_without_mutating_strict_xy_z_or_ownership():
    cfg, cells, regions, masks, local, measurement = fixture()
    before = copy.deepcopy(cells)
    frozen = {k: v.copy() for k, v in masks.items()}
    run(cells, regions, masks, local, measurement, cfg)
    assert all(c["review_geometry"]["adopted_for_review"] for c in cells)
    for cell, old in zip(cells, before):
        assert cell["geometry"] == old["geometry"]
        np.testing.assert_array_equal(masks[cell["cell_id"]], frozen[cell["cell_id"]])
        r = cell["review_geometry"]
        assert all(p[2] is None for p in r["corners_xyz"])
        assert r["guards"]["spatial_core_loss"] == 0
        assert r["guards"]["neighbor_capture"] == 0
        assert not r["steel_certified"] and not r["control_eligible"]
        assert cell["strict_rectangle_state"] == "INFEASIBLE"
    displayed = review_objects(dict(hatches=[], selected_opening_cells=cells))
    assert displayed[0]["boundary_xy"] != before[0]["geometry"]["boundary_xy"]
    assert all(p["z_m"] is None for row in displayed for p in row["corner_heights"])


def test_two_same_hull_errors_without_own_transitions_cannot_force_equal_width():
    cfg, cells, regions, masks, local, measurement = fixture(flat=True)
    run(cells, regions, masks, local, measurement, cfg)
    assert all(not c["review_geometry"]["adopted_for_review"] for c in cells)


def test_single_hatch_and_foreign_vessels_skip_joint_recovery():
    cfg, cells, regions, masks, local, measurement = fixture()
    cells[1]["vessel_id"] = "foreign"
    run(cells, regions, masks, local, measurement, cfg)
    assert all(not c["review_geometry"]["adopted_for_review"] for c in cells)
    assert all("FAST_PATH" in c["review_geometry"]["reason"] for c in cells)


def test_normal_fast_path_does_not_scan_raw_membership():
    cfg, cells, regions, masks, local, measurement = fixture()
    for cell in cells:
        cell["width_recovery_requested_sides"] = []
    with patch(
        "ship_perception.perception.review_geometry_recovery.LocalRimEvidence",
        side_effect=AssertionError("UNNECESSARY_RAW_INDEX"),
    ), patch(
        "ship_perception.perception.review_geometry_recovery.cell_members",
        side_effect=AssertionError("UNNECESSARY_RAW_SCAN"),
    ):
        recover_review_geometries(cells, regions, None, cfg)
    assert all("FAST_PATH" in c["review_geometry"]["reason"] for c in cells)


def test_separator_conflict_cannot_be_removed_to_make_a_review_succeed():
    cfg, cells, regions, masks, local, measurement = fixture()
    cells[0]["separator_constraints"] = [dict(axis=1, sense="MIN", value=3.0)]
    run(cells, regions, masks, local, measurement, cfg)
    assert not cells[0]["review_geometry"]["adopted_for_review"]
    assert cells[0]["review_geometry"]["guards"]["separator_crossing"] == 1


def test_review_attachment_at_tolerance_boundary_is_rigid_invariant():
    cfg, cells, regions, masks, local, measurement = fixture()
    run(cells, regions, masks, local, measurement, cfg)
    baseline = {
        c["cell_id"]: np.asarray(c["review_geometry"]["boundary_xy"]) for c in cells
    }
    for yaw, translation in (
        (0.0, [31.0, -19.0]),
        (0.7, [-90.0, 145.0]),
        (-1.2, [500.0, -300.0]),
    ):
        cfg, cells, regions, masks, local, measurement = fixture()
        q = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        t = np.asarray(translation)
        shift = t @ q
        for cell in cells:
            geometry = cell["geometry"]
            xy = np.asarray(geometry["boundary_xy"]) @ q.T + t
            geometry["boundary_xy"] = xy.tolist()
            geometry["corners_xyz"] = [[x, y, 7.0] for x, y in xy]
        for region in regions.values():
            region.search_envelope = (
                np.asarray(region.search_envelope) + np.tile(shift, 2)
            ).tolist()
        local.xyz[:, :2] += shift
        points, ids = local.xyz, local.ids
        local.index = SimpleNamespace(
            query_indexes=lambda b: ids[
                np.all((points[:, :2] >= b[:2]) & (points[:, :2] <= b[2:]), axis=1)
            ]
        )
        measurement.local.vessel.axes = q.T
        run(cells, regions, masks, local, measurement, cfg)
        for cell in cells:
            assert cell["review_geometry"]["adopted_for_review"]
            inverse = (np.asarray(cell["review_geometry"]["boundary_xy"]) - t) @ q
            np.testing.assert_allclose(
                inverse, baseline[cell["cell_id"]], atol=1e-8, rtol=0
            )
