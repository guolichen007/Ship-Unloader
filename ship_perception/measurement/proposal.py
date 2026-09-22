"""Attributable proposal diagnostics; coarse boxes are never steel-edge truth."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import hypot, isfinite
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    provider: str
    candidate_id: str
    score: float
    rect: tuple[float, float, float, float]
    source: str
    rank_pre_nms: int | None = None
    rank_post_nms: int | None = None
    rejection_reason: str | None = None

    def __post_init__(self):
        x0, y0, x1, y1 = self.rect
        if not all(isfinite(v) for v in (*self.rect, self.score)) or x1 <= x0 or y1 <= y0:
            raise ValueError("INVALID_CANDIDATE_GEOMETRY")

    def record(self):
        data = asdict(self)
        x0, y0, x1, y1 = self.rect
        data.update(rectangle=list(self.rect), center=[(x0+x1)/2, (y0+y1)/2],
                    size=[x1-x0, y1-y0])
        del data["rect"]
        return data


def bbox(polygon: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    points = [tuple(p) for p in polygon]
    if len(points) < 3 or any(len(p) != 2 or not all(isfinite(x) for x in p) for p in points):
        raise ValueError("INVALID_ROUGH_POLYGON")
    xs, ys = zip(*points)
    box = min(xs), min(ys), max(xs), max(ys)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("DEGENERATE_ROUGH_POLYGON")
    return box


def intersection(a, b):
    return max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(0.0, min(a[3], b[3])-max(a[1], b[1]))


def iou(a, b):
    inter = intersection(a, b)
    aa = (a[2]-a[0])*(a[3]-a[1])
    bb = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa+bb-inter) if inter else 0.0


def nms(candidates: Iterable[Candidate], *, threshold: float = 0.3, top_k: int = 5):
    """Match legacy >0.3 NMS, but retain every suppressed record and reason."""
    if not 0 <= threshold <= 1 or top_k <= 0:
        raise ValueError("INVALID_NMS_CONFIG")
    ranked = sorted(candidates, key=lambda c: (-c.score, c.candidate_id))
    kept: list[Candidate] = []
    audited: list[Candidate] = []
    for pre, c in enumerate(ranked, 1):
        blocker = next((k for k in kept if iou(c.rect, k.rect) > threshold), None)
        reason = f"NMS_SUPPRESSED_BY:{blocker.candidate_id}" if blocker else None
        if reason is None and len(kept) >= top_k:
            reason = "TOP_K_TRUNCATED"
        result = Candidate(c.provider, c.candidate_id, c.score, c.rect, c.source,
                           pre, len(kept)+1 if reason is None else None, reason)
        audited.append(result)
        if reason is None:
            kept.append(result)
    return audited


def evaluate_instances(candidates: Iterable[Candidate], hatches: Iterable[dict],
                       *, min_iou: float = 0.1, min_gt_coverage: float = 0.5):
    """One-to-one coarse proposal audit. A box spanning two hatches is a merge, not two recalls.

    This diagnostic uses axis-aligned envelopes of adjudicated rough polygons. It does
    not measure physical opening-side edge accuracy. Missing ROIs must be rejected.
    """
    gt = []
    for h in hatches:
        poly = h.get("rough_polygon_raw_xy")
        if poly is None:
            raise ValueError("MISSING_ADJUDICATED_ROUGH_ROI")
        gt.append((h["local_id"], bbox(poly)))
    cs = list(candidates)
    merged = set()
    for c in cs:
        covered = [hid for hid, box in gt if intersection(c.rect, box) /
                   ((box[2]-box[0])*(box[3]-box[1])) >= min_gt_coverage]
        if len(covered) > 1:
            merged.add(c.candidate_id)
    pairs = sorted(((iou(c.rect, box), hid, c.candidate_id)
                    for c in cs for hid, box in gt if c.candidate_id not in merged), reverse=True)
    assigned_h, assigned_c = {}, set()
    for overlap, hid, cid in pairs:
        if overlap < min_iou:
            break
        if hid not in assigned_h and cid not in assigned_c:
            assigned_h[hid] = cid
            assigned_c.add(cid)
    rows = []
    by_id = {c.candidate_id: c for c in cs}
    for hid, box in gt:
        best = max(cs, key=lambda c: iou(c.rect, box), default=None)
        matched_id = assigned_h.get(hid)
        picked = by_id.get(matched_id)
        rows.append({"hatch_id": hid, "matched": picked is not None,
                     "matched_candidate_id": matched_id,
                     "matched_rank_pre_nms": picked.rank_pre_nms if picked else None,
                     "matched_rank_post_nms": picked.rank_post_nms if picked else None,
                     "best_candidate_id": best.candidate_id if best else None,
                     "best_candidate_rank": best.rank_pre_nms if best else None,
                     "best_candidate_rank_post_nms": best.rank_post_nms if best else None,
                     "coarse_bbox_iou": iou(best.rect, box) if best else 0.0,
                     "center_error_m": hypot((best.rect[0]+best.rect[2]-box[0]-box[2])/2,
                                              (best.rect[1]+best.rect[3]-box[1]-box[3])/2) if best else None})
    unmatched = [c for c in cs if c.candidate_id not in assigned_c and c.candidate_id not in merged]
    duplicates = [c.candidate_id for c in unmatched if any(iou(c.rect, box) >= min_iou for _, box in gt)]
    return {"hatches": rows, "matched": len(assigned_h), "total": len(gt),
            "merged_candidate_ids": sorted(merged),
            "duplicate_candidate_ids": sorted(duplicates),
            "false_or_unverified_candidate_count": len(unmatched)-len(duplicates)}
