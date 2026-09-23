"""Enumerate physical 0/1/2/3-hatch hypotheses with duplicate conflicts."""
import itertools
import math


def _overlap(a, b):
    x0, y0, x1, y1 = a
    u0, v0, u1, v1 = b
    intersection = max(0, min(x1, u1) - max(x0, u0)) * max(0, min(y1, v1) - max(y0, v0))
    area_a, area_b = (x1 - x0) * (y1 - y0), (u1 - u0) * (v1 - v0)
    return intersection / max(min(area_a, area_b), 1e-9)


def _quality(row, config):
    deck = row["local_deck"]
    if deck["status"] != "RESOLVED" or row["status"] == "UNRESOLVED":
        return -math.inf
    edges = row["boundaries"]
    coverage = sum(edge["coverage"] for edge in edges)
    profile = sum("OBSERVED_PROFILE_BREAK" in edge["evidence_type"] for edge in edges)
    faces = sum("OBSERVED_3D_FACE" in edge["evidence_type"] for edge in edges)
    residual = sum(edge["fit_residual_p95_m"] for edge in edges) / len(edges)
    # Positive score requires several independently observed sides. There is
    # no scan name, GT count, or fixed hatch dimension in this score.
    return (1.5 * len(edges) + coverage + 0.25 * profile + 0.25 * faces
            - residual / config["boundary"]["line_inlier_m"]
            - deck["residual_p95_m"] / config["geometry"]["plane_inlier_m"] - 3.0)


def solve(rows, config):
    valid = [row for row in rows if math.isfinite(_quality(row, config))]
    valid.sort(key=lambda row: (-_quality(row, config), row["opening_id"]))
    if len(valid) > 24:
        # Bounded combinatorics. Discarded candidates remain in proposal debug
        # and this diagnostic; the output does not silently claim completeness.
        valid = valid[:24]
        limit_warning = "HYPOTHESIS_CANDIDATE_LIMIT"
    else:
        limit_warning = None
    conflicts = set()
    for first, second in itertools.combinations(valid, 2):
        boxes = first["bbox_xy"], second["bbox_xy"]
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        distance = math.dist(*centers)
        if (_overlap(*boxes) >= config["evaluation"]["duplicate_overlap_min"] or
                distance < config["geometry"]["candidate_voxel_m"]):
            conflicts.add(frozenset((first["opening_id"], second["opening_id"])))
    hypotheses = []
    for size in range(min(3, len(valid)) + 1):
        for combination in itertools.combinations(valid, size):
            ids = [row["opening_id"] for row in combination]
            if any(frozenset(pair) in conflicts for pair in itertools.combinations(ids, 2)):
                continue
            score = sum(_quality(row, config) for row in combination)
            hypotheses.append(dict(hatch_count=size, opening_ids=ids, score=float(score)))
    hypotheses.sort(key=lambda item: (-item["score"], item["hatch_count"], item["opening_ids"]))
    best = hypotheses[0]
    gap = best["score"] - hypotheses[1]["score"] if len(hypotheses) > 1 else math.inf
    ambiguous = gap < config["frame"]["candidate_score_gap"]
    selected = [] if ambiguous else [row for row in valid if row["opening_id"] in best["opening_ids"]]
    status = ("HATCH_HYPOTHESIS_AMBIGUOUS" if ambiguous else
              "RESOLVED" if any(row["status"] == "COMPLETE_OBSERVED" for row in selected) else
              "PARTIAL_DETECTION" if selected or any(row["status"] == "PARTIAL" for row in rows)
              else "NO_CONFIRMED_HATCH")
    return dict(scene_status=status, selected=selected, hypotheses=hypotheses[:2],
                candidate_conflicts=[sorted(pair) for pair in conflicts],
                score_gap=None if not math.isfinite(gap) else float(gap),
                warning=limit_warning)
