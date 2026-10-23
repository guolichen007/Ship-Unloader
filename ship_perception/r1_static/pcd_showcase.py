"""Reproduce the current static hatch height review directly from one PCD.

This is an orchestration entry point for existing algorithms. It never reads
intermediate review JSON and does not relax any calibration evidence gate.
"""

import argparse
import hashlib
import json
from pathlib import Path

from ship_perception.tools.v15_pcd import decode
from ship_perception.tools.stage_progress import stage

from .corner_topology_review import review_scene
from .family_reference import analyze_scene_s4r1
from .hatch_proposal import discover_scene_hatches
from .heightmap_batch import RESEARCH_CONFIG
from .heightmap_topology import detect_regions
from .opening_semantics import bridge_final_rectangles
from .rectangle_fusion import solve as fuse_rectangles
from .rectangle_refinement import refine as refine_rectangles
from .run import DEFAULT_CONFIG, _atomic_json, resolve_config
from .static_showcase import render_showcase


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(pcd, scene_id, output, expected_sha256=None):
    """Run the full current static chain and write one raw XYZ height figure."""
    pcd = Path(pcd)
    output = Path(output)
    digest = _sha256(pcd)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError("INPUT_SHA_MISMATCH:" + scene_id)
    config, config_sha = resolve_config(DEFAULT_CONFIG)
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    points, _ = decode(pcd)
    b2, b4, counts = prepare_geometry(points, scene_id, digest, config, research)
    report, _, private = review_scene(
        scene_id, pcd, b2, b4, config, research, height_rim=True
    )
    output.mkdir(parents=True, exist_ok=True)
    report["height_only_outputs"] = render_showcase(
        report, points, private, b2, b4, config, output, height_only=True
    )
    _atomic_json(output / "corner_topology.json", report)
    _atomic_json(
        output / "运行摘要.json",
        dict(
            scene_id=scene_id,
            input_sha256=digest,
            point_count=int(len(points)),
            config_sha256=config_sha,
            **counts,
            final_rectangle_count=sum(len(v["rectangles"]) for v in report["vessels"]),
            static_calibration_status=report["static_calibration_status"],
        ),
    )
    return report


def prepare_geometry(
    points,
    scene_id,
    digest,
    config,
    research,
    progress=None,
    *,
    scene_context=None,
    structural_cache=None,
    evidence_sink=None,
):
    """Share the unchanged XYZ providers between file and ROS entry points."""

    with stage(progress, "HEIGHT_PROVIDER"):
        height, _ = detect_regions(points, config, research)
    with stage(progress, "STRUCTURAL_PROVIDERS"):
        structural = analyze_scene_s4r1(points, config, research, height)
    if evidence_sink is not None:
        evidence_sink["structural"] = structural
    with stage(progress, "RECTANGLE_FUSION"):
        r2f = fuse_rectangles(
            points,
            structural["segment_resolutions"],
            structural["edges"],
            height["level_hierarchy"] + height.get("local_enclosure_proposals", []),
            config,
        )
    with stage(progress, "RECTANGLE_REFINEMENT"):
        r2g = (
            refine_rectangles(
                points,
                structural["segment_resolutions"],
                structural["edges"],
                r2f,
                config,
            )
            if r2f["rectangles"]
            else dict(rectangles=[], status="NO_LEGACY_RECTANGLE")
        )
    r2g["input_pcd_sha256"] = digest

    with stage(progress, "SCENE_VESSEL_DISCOVERY"):
        b2 = (
            discover_scene_hatches(points, config, research)
            if scene_context is None
            else discover_scene_hatches(
                points,
                config,
                research,
                scene_context=scene_context,
                structural_cache=structural_cache,
            )
        )
    b2.update(scene_id=scene_id, input_sha256=digest, raw_point_count=int(len(points)))
    with stage(progress, "OPENING_SIDE_REVIEW"):
        b4 = bridge_final_rectangles(points, r2g, b2, config)
    return (
        b2,
        b4,
        dict(
            height_region_count=height["region_count"],
            local_enclosure_proposal_count=len(
                height.get("local_enclosure_proposals", [])
            ),
            r2f_rectangle_count=len(r2f["rectangles"]),
            r2g_rectangle_count=len(r2g["rectangles"]),
            b4_rectangle_count=len(b4["rectangles"]),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    report = run(args.pcd, args.scene_id, args.output, args.expected_sha256)
    print(
        args.scene_id,
        [(v["vessel_hypothesis_id"], len(v["rectangles"])) for v in report["vessels"]],
        flush=True,
    )


if __name__ == "__main__":
    main()
