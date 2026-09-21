"""Partial annotation evaluator. Product code never imports this module."""
import math
import numpy as np


def distance_to_segments(points, segments):
    p = np.asarray(points, dtype=float)
    if not len(segments):
        return np.full(len(p), np.inf)
    distances = np.full(len(p), np.inf)
    for a, b in segments:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        d = b-a
        t = np.clip((p-a)@d/max(float(d@d), 1e-20), 0, 1)
        distances = np.minimum(distances, np.linalg.norm(p-a-t[:, None]*d, axis=1))
    return distances


def polygon_segments(poly):
    return [(poly[i], poly[(i+1) % len(poly)]) for i in range(len(poly))]


def sample_segments(segments, spacing=.05):
    result = []
    for a, b in segments:
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(math.ceil(np.linalg.norm(b-a)/spacing))+1)
        result.extend(a+(b-a)*t for t in np.linspace(0, 1, n))
    return np.asarray(result)


def inside(points, polygon):
    points = np.asarray(points, float)
    hit = np.zeros(len(points), dtype=bool)
    for a, b in polygon_segments(np.asarray(polygon, float)):
        crosses = (a[1] > points[:, 1]) != (b[1] > points[:, 1])
        if abs(b[1]-a[1]) > 1e-20:
            hit ^= crosses & (points[:, 0] < (b[0]-a[0])*(points[:, 1]-a[1])/(b[1]-a[1])+a[0])
    return hit


def stats(values):
    a = np.asarray(values, float)
    if not len(a) or not np.isfinite(a).all():
        return dict(median=None, P75=None, P95=None, max=None, samples=len(a))
    return dict(median=float(np.quantile(a, .5)), P75=float(np.quantile(a, .75)),
                P95=float(np.quantile(a, .95)), max=float(a.max()), samples=len(a))


def input_segments(model, hatch, observed_only=False, inferred_only=False):
    t = np.linalg.inv(np.asarray(model["T_B_input"], float))
    result = []
    for edge in hatch["boundaries"]:
        if observed_only and not edge["evidence_flags"] & 1:
            continue
        if inferred_only and not edge["evidence_flags"] & 6:
            continue
        a3,b3=np.asarray(edge["a"],float),np.asarray(edge["b"],float)
        delta=b3-a3;length=float(np.linalg.norm(delta))
        if observed_only:
            if "support_intervals" not in edge:raise ValueError("MISSING_OBSERVED_SUPPORT_INTERVALS")
            intervals=edge["support_intervals"]
        else:intervals=[(0.,length)]
        for lo,hi in intervals:
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi <= length+1e-6):
                raise ValueError("INVALID_OBSERVED_SUPPORT_INTERVAL")
            if hi<=lo or length<=1e-12:continue
            a=(t@np.r_[a3+delta*(lo/length),1])[:2]
            b=(t@np.r_[a3+delta*(hi/length),1])[:2]
            result.append((a,b))
    return result


def evaluate_annotation(model, annotation, config):
    poly = np.asarray(annotation["corners"], float)
    if poly.shape != (4, 2) or not np.isfinite(poly).all():
        raise ValueError("INVALID_PARTIAL_ANNOTATION")
    truth_segments = polygon_segments(poly)
    label_points = sample_segments(truth_segments)
    margin = config["local_annotation_margin_m"]
    records = []
    for hatch in model["hatches"]:
        observed = input_segments(model, hatch, True)
        inferred = input_segments(model, hatch, inferred_only=True)
        samples = sample_segments(observed)
        rec = dict(candidate_id=hatch["candidate_id"], status="UNVERIFIED_CANDIDATE",
                   association_score=0., local_fraction=0., label_coverage=0., duplicate=False)
        if len(samples):
            d = distance_to_segments(samples, truth_segments)
            local = inside(samples, poly) | (d <= margin)
            coverage = float(np.mean(distance_to_segments(label_points, observed) <= margin))
            proximity = float(np.mean(np.maximum(0., 1-d/margin)))
            rec.update(local_fraction=float(local.mean()), label_coverage=coverage,
                       association_score=coverage*proximity,
                       interior_fraction=float(inside(samples, poly).mean()))
            if rec["local_fraction"] > 0:
                rec["status"] = "LOCAL_ASSOCIATION_UNCERTAIN"
            rec["observed_error"] = stats(np.r_[d, distance_to_segments(label_points, observed)])
            rec["per_edge"] = [stats(distance_to_segments(sample_segments([edge]), observed)) for edge in truth_segments]
        else:
            rec["observed_error"] = stats([])
            rec["per_edge"] = [stats([]) for _ in truth_segments]
        rec["inferred_error"] = stats(distance_to_segments(sample_segments(inferred), truth_segments)) if inferred else stats([])
        records.append(rec)
    ranked = sorted(records, key=lambda r: (-r["association_score"], r["candidate_id"]))
    eligible = [r for r in ranked if r["association_score"] >= config["association_min_score"]]
    status, matched = "UNMATCHED_LABEL", None
    if eligible:
        if len(eligible)>1 and eligible[0]["association_score"]-eligible[1]["association_score"] < config["association_gap"]:
            status = "AMBIGUOUS_MATCH"
            for r in eligible:r["status"] = "AMBIGUOUS_MATCH"
        else:
            status, matched = "MATCHED", eligible[0]["candidate_id"]
            eligible[0]["status"] = "MATCHED"
        winner = eligible[0]
        winner_hatch = next(h for h in model["hatches"] if h["candidate_id"] == winner["candidate_id"])
        winner_segments = input_segments(model, winner_hatch, True)
        for rec in records:
            if rec is winner:continue
            h = next(h for h in model["hatches"] if h["candidate_id"] == rec["candidate_id"])
            points = sample_segments(input_segments(model, h, True))
            if not len(points):continue
            same_support = float(np.mean(distance_to_segments(points, winner_segments) <= config.get("boundary_p95_m", .15)))
            # A shared boundary alone does not establish an entire second hatch.
            # Partial fragments repeating the matched support are nevertheless audited.
            duplicate = rec["local_fraction"] >= config["duplicate_overlap_min"] and (
                rec["label_coverage"] >= config["duplicate_overlap_min"] or
                (h["status"] == "PARTIAL" and same_support >= config["duplicate_overlap_min"] and rec.get("interior_fraction",0) >= config["duplicate_overlap_min"]))
            if duplicate:
                rec["duplicate"] = True
                if status != "AMBIGUOUS_MATCH":rec["status"] = "LOCAL_DUPLICATE_CANDIDATE"
    return dict(annotation_id=annotation.get("id"), status=status, matched_candidate_id=matched,
                local_duplicate_count=sum(r["duplicate"] for r in records), candidates=records,
                annotation_scope="PARTIAL_SINGLE_HATCH", outside_scope="UNLABELED_REGION",
                real_accuracy_gate="NO_15CM_HARD_GATE")


def disagreement(a, b):
    pa, pb = np.asarray(a["corners"], float), np.asarray(b["corners"], float)
    va = [pa[:,0].min(),pa[:,0].max(),pa[:,1].min(),pa[:,1].max()]
    vb = [pb[:,0].min(),pb[:,0].max(),pb[:,1].min(),pb[:,1].max()]
    result = {name:abs(float(x-y)) for name,x,y in zip(("x0","x1","y0","y1"),va,vb)}
    result["corner"] = stats(np.r_[np.linalg.norm(pa[:,None,:]-pb[None,:,:],axis=2).min(axis=1),np.linalg.norm(pb[:,None,:]-pa[None,:,:],axis=2).min(axis=1)])
    result["classification"] = "LABEL_DISAGREEMENT"
    return result
