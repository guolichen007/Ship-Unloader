"""独立合成评分通道。产品进程只接收 runtime XYZ 文件。"""
from dataclasses import dataclass,field
import math
import numpy as np
from .v15_partial import inside, polygon_segments, sample_segments, distance_to_segments, stats


@dataclass
class Fixture:
    runtime: np.ndarray
    polygons: list
    visible_deck: np.ndarray
    T_input_fixture: np.ndarray
    expected_complete: bool = True
    expected_beams: int = 0
    negative: bool = False
    boundary_observable: bool = True
    inferred_edge: bool = False
    beam_segments: list = field(default_factory=list)


QUICK_SCENARIOS = ("MULTI_HATCH_BEAM", "NONRECTANGULAR", "TOP_DOWN_VERTICAL_BLIND",
                   "INNER_OUTER_FACES", "MISSING_TWO_EDGES", "CARGO_BACKGROUND", "RAW_ROTATION")
FULL_SCENARIOS = QUICK_SCENARIOS + ("SINGLE_HATCH", "PENTAGON", "NOISE", "LOW_DENSITY",
                                  "PURE_DECK", "DECK_WHARF", "DECK_CRANE", "CARGO_ONLY",
                                  "RAMP_050", "RAMP_100", "SMOOTH_TRANSITION", "COMPETING_BREAKS",
                                  "INNER_ONLY", "OUTER_ONLY", "PARTIAL_BOTH_FACES", "TILTED_COAMING",
                                  "THICKNESS_020", "THICKNESS_040", "TOP_EDGE_SPARSE", "TRANSIENT_CLUSTERS")
PAIRED_SCENARIOS=tuple(f"PAIRED_{thickness}_{visibility}" for thickness in ("000","020","040")
                       for visibility in ("INNER","OUTER","BOTH","BLIND"))
FULL_SCENARIOS+=PAIRED_SCENARIOS+("ONE_OCCLUDED_SPAN","SOLID_PARTITION","CONCAVE_HATCH","OUTLIERS")


def fixture(name, seed):
    if name not in FULL_SCENARIOS:raise ValueError("UNKNOWN_SYNTHETIC_SCENARIO")
    rng=np.random.default_rng(np.random.SeedSequence([seed,FULL_SCENARIOS.index(name),15]))
    paired=name.startswith("PAIRED_")
    pair_thickness=float(name.split("_")[1])/100 if paired else None
    pair_visibility=name.split("_")[2] if paired else None
    step=.10
    x,y=np.meshgrid(np.arange(-7,7+step/2,step),np.arange(-5,5+step/2,step),indexing="ij")
    xy=np.column_stack((x.ravel(),y.ravel()))
    polygons=[np.array([[-4,-2],[4,-2],[4,2],[-4,2]],float)]
    if name in ("MULTI_HATCH_BEAM","TRANSIENT_CLUSTERS"):
        polygons=[np.array([[-5,-2],[-1,-2],[-1,2],[-5,2]],float),
                  np.array([[1,-2],[5,-2],[5,2],[1,2]],float)]
    if name=="NONRECTANGULAR":polygons=[np.array([[-4,-2],[4,-1.2],[3,2],[-3.4,2.6]],float)]
    if name=="PENTAGON":polygons=[np.array([[-4,-2],[3,-2],[4,0],[2.5,2.5],[-4,2]],float)]
    if name=="CONCAVE_HATCH":polygons=[np.array([[-4,-2],[4,-2],[4,0],[0,0],[0,2.5],[-4,2.5]],float)]
    if name=="SOLID_PARTITION":
        polygons=[np.array([[-4,-2],[-.5,-2],[-.5,2],[-4,2]],float),
                  np.array([[.5,-2],[4,-2],[4,2],[.5,2]],float)]
    negative=name in ("PURE_DECK","DECK_WHARF","DECK_CRANE","CARGO_ONLY")
    if negative:polygons=[]
    low=np.zeros(len(xy),bool)
    for poly in polygons:low|=inside(xy,poly)
    z=np.where(low,-3.,0.)
    if name in ("RAMP_050","RAMP_100","SMOOTH_TRANSITION","COMPETING_BREAKS"):
        depth=distance_to_segments(xy,polygon_segments(polygons[0]))
        width=.5 if name=="RAMP_050" else 1.
        t=np.minimum(depth[low]/width,1.)
        if name=="SMOOTH_TRANSITION":t=t*t*(3-2*t)
        if name=="COMPETING_BREAKS":t=np.where(t<.4,t*.3,.12+(t-.4)*1.4666666667)
        z[low]=-3*t
    coaming_top=np.zeros(len(xy),bool)
    face_scenarios=("INNER_OUTER_FACES","INNER_ONLY","OUTER_ONLY","PARTIAL_BOTH_FACES","TILTED_COAMING","THICKNESS_020","THICKNESS_040","TOP_EDGE_SPARSE")+tuple(s for s in PAIRED_SCENARIOS if "_000_" not in s)
    thickness=.2 if name=="THICKNESS_020" else .4 if name=="THICKNESS_040" else .3
    if paired:thickness=pair_thickness
    hidden_inner=name in ("OUTER_ONLY","TOP_EDGE_SPARSE") or (paired and thickness>0 and pair_visibility in ("OUTER","BLIND"))
    if name in face_scenarios:
        coaming_top=(~low)&(distance_to_segments(xy,polygon_segments(polygons[0]))<=thickness)
        z[coaming_top]=.8
    if name=="CARGO_BACKGROUND":
        z[low]+=3.8*np.exp(-(xy[low,0]**2/2+xy[low,1]**2))
    if name=="CARGO_ONLY":z=2*np.exp(-(xy[:,0]**2/10+xy[:,1]**2/8))
    points=np.column_stack((xy,z))
    deck=points[~low&~coaming_top].copy() if name!="CARGO_ONLY" else np.empty((0,3))
    extra=[]
    if name=="SOLID_PARTITION":
        # The divider has two actually sampled full-depth faces. A floating
        # beam fixture instead retains returns underneath and one opening.
        for xface in (-.5,.5):
            for by in np.arange(-2,2.001,.05):
                for h in np.arange(-3,.001,.05):extra.append([xface,by,h])
    if hidden_inner:
        # Visibility is applied to every physical surface, not only the wall.
        # A dense top ribbon or floor reaching the hidden inner edge would leak
        # its location through ordinary geometry and invalidate this control.
        floor_shadow=low&(distance_to_segments(xy,polygon_segments(polygons[0]))<.6)
        points=points[~coaming_top&~floor_shadow]
    if name in face_scenarios:
        for poly in polygons:
            for a,b in polygon_segments(poly):
                direction=(b-a)/np.linalg.norm(b-a);outward=np.array([direction[1],-direction[0]])
                for s in np.arange(0,np.linalg.norm(b-a)+.025,.05):
                    if name=="PARTIAL_BOTH_FACES" and .3< s/np.linalg.norm(b-a)<.7:continue
                    for h in np.arange(-.8,.801,.05):
                        if paired and pair_visibility=="BLIND":continue
                        if name=="TOP_EDGE_SPARSE" and (h<.75 or int(round(s/.05))%4):continue
                        q=a+direction*s
                        if name=="TILTED_COAMING":q=q+outward*.1*h
                        inner_visible=name!="OUTER_ONLY" and not (paired and pair_visibility=="OUTER")
                        if inner_visible:extra.append([q[0],q[1],h])
                        # The external face below Deck is physically occluded
                        # by the opaque deck/body and cannot enter runtime XYZ.
                        if h>=-1e-9 and name!="INNER_ONLY" and not (paired and pair_visibility=="INNER"):
                            q=q+outward*thickness;extra.append([q[0],q[1],h])
    if name=="MULTI_HATCH_BEAM":
        # A floating cross-member has actual lower returns underneath. Keeping
        # both returns is essential: its projection must not create two hatches.
        for bx in np.arange(-5,-.999,.05):
            for by in np.arange(-.2,.201,.05):extra.append([bx,by,.8])
        for bx in (-5.,-1.):
            for by in np.arange(-.2,.201,.05):
                for h in np.arange(0,.801,.05):extra.append([bx,by,h])
    if name in ("CARGO_BACKGROUND","DECK_WHARF"):
        for bx in np.arange(-10,10,.15):
            for by in np.arange(7,13,.15):extra.append([bx,by,1.5])
    if name=="DECK_CRANE":
        for h in np.arange(0,10,.1):
            for a in np.arange(0,2*math.pi,.2):extra.append([5+.15*math.cos(a),3+.15*math.sin(a),h])
    if name=="TRANSIENT_CLUSTERS":
        # A line-shaped accumulated moving object has no two supported
        # attachments. Its points stay in the runtime cloud.
        for cx in np.arange(-4.5,-1.4,.25):
            extra.extend(rng.normal([cx,0,1.2],[.08,.12,.15],(40,3)).tolist())
    if extra:points=np.vstack((points,np.array(extra)))
    if name=="MISSING_TWO_EDGES":
        keep=(points[:,0]<3.0)&(points[:,1]<1.0)
        points=points[keep];deck=deck[(deck[:,0]<3.0)&(deck[:,1]<1.0)]
    if name=="ONE_OCCLUDED_SPAN":
        def visible(p):return ~((np.abs(p[:,0])<3.65)&(p[:,1]>1.2)&(p[:,1]<2.8))
        points=points[visible(points)];deck=deck[visible(deck)]
    if name=="LOW_DENSITY":points=points[rng.random(len(points))>.35]
    if name=="NOISE":points+=rng.normal(0,.012,points.shape)
    if name=="OUTLIERS":
        # Sparse 3D contamination is not a coherent structural surface. Its
        # labels remain in the generator; runtime receives all these points.
        points=np.vstack((points,rng.uniform([-7,-5,-4],[7,5,3],size=(int(.02*len(points)),3))))
    transform=np.eye(4)
    if name=="RAW_ROTATION":
        ax,ay,az=.31,-.23,.37
        rx=np.array([[1,0,0],[0,math.cos(ax),-math.sin(ax)],[0,math.sin(ax),math.cos(ax)]])
        ry=np.array([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]])
        rz=np.array([[math.cos(az),-math.sin(az),0],[math.sin(az),math.cos(az),0],[0,0,1]])
        transform[:3,:3]=rz@ry@rx;transform[:3,3]=[2,-1,4]
    runtime=points@transform[:3,:3].T+transform[:3,3]
    runtime=runtime[rng.permutation(len(runtime))].astype("<f4")
    uncertain=name in ("SMOOTH_TRANSITION","COMPETING_BREAKS") or hidden_inner
    partial=uncertain or name in ("MISSING_TWO_EDGES","PARTIAL_BOTH_FACES")
    beams=[(np.array([-5,0,.8]),np.array([-1,0,.8]))] if name=="MULTI_HATCH_BEAM" else [(np.array([0,-2,0.]),np.array([0,2,0.]))] if name=="SOLID_PARTITION" else []
    return Fixture(runtime,polygons,deck,transform,not partial,len(beams),negative,not uncertain,name=="ONE_OCCLUDED_SPAN",beams)


def evaluate(model, truth, config):
    failures=[]
    def gate(condition, reason):
        if not condition:failures.append(reason)
    complete=[h for h in model["hatches"] if h["status"]!="PARTIAL"]
    if truth.negative:
        gate(not complete,"NEGATIVE_COMPLETE_HATCH")
        gate(not any(h["quality_score"]>=config["high_quality_threshold"] for h in model["hatches"]),"NEGATIVE_HIGH_QUALITY_HATCH")
        gate(not any(p["kind"] in ("COAMING_OR_HOLD_WALL","BEAM_OR_PARTITION") for p in model["structures"]),"NEGATIVE_CONFIRMED_STRUCTURE")
        return dict(status="PASS" if not failures else "FAIL",failures=failures)
    gate(model["frame_resolved"] and model["deck"]["valid"],"FRAME_OR_DECK_UNRESOLVED")
    if failures:return dict(status="FAIL",failures=failures)
    # Deck remains measurable even when the opening side is not. Unobservable
    # edges must not bypass the independent Deck accuracy gate.
    T_fixture_B=np.linalg.inv(truth.T_input_fixture)@np.linalg.inv(np.array(model["T_B_input"]))
    def transform(p):return (T_fixture_B@np.r_[p,1])[:3]
    deck=model["deck"]
    n=T_fixture_B[:3,:3]@np.array(deck["normal"]);origin=transform(-deck["offset"]*np.array(deck["normal"]));offset=-float(n@origin)
    center=truth.visible_deck.mean(axis=0);radius=np.linalg.norm(truth.visible_deck[:,:2]-center[:2],axis=1)
    r95=float(np.quantile(radius,.95));angle=math.acos(float(np.clip(n[2],-1,1)))
    distances=np.abs(truth.visible_deck@n+offset)
    gate(abs(float(n@center+offset))<=config["deck_offset_max_m"],"DECK_OFFSET")
    gate(angle<=min(math.radians(config["deck_angle_max_deg"]),math.atan2(config["deck_tilt_budget_m"],r95)),"DECK_ANGLE")
    gate(np.quantile(distances,.95)<=config["deck_distance_p95_m"],"DECK_DISTANCE")
    deck_metrics=dict(GT_visible_R95=r95,angle_rad=angle,distance=stats(distances))
    if not truth.boundary_observable:
        gate(bool(model["hatches"]),"MISSING_PARTIAL_OPENING")
        gate(not complete,"UNOBSERVABLE_EDGE_FORCED_COMPLETE")
        gate(not any(e["evidence_flags"]&1 for h in model["hatches"] for e in h["boundaries"]),"UNOBSERVABLE_EDGE_CALLED_OBSERVED")
        gate(not any(p["kind"] in ("COAMING_OR_HOLD_WALL","BEAM_OR_PARTITION") for p in model["structures"]),"UNOBSERVABLE_CONFIRMED_STRUCTURE")
        return dict(status="PASS" if not failures else "FAIL",failures=failures,expected="PARTIAL_UNOBSERVABLE_BOUNDARY",deck=deck_metrics)
    # Ground truth is transformed once using known fixture placement. There is
    # no ICP alignment or best-fit gauge correction in this scorer.
    predicted=[]
    for h in model["hatches"]:
        segments=[]
        for edge in h["boundaries"]:
            anchor_supported=edge["evidence_flags"]==4 and edge["evidence_mechanism"]=="TWO_OBSERVED_CORNER_FRAGMENTS"
            if not edge["evidence_flags"]&1 and not anchor_supported:continue
            gate(edge["side"]=="INNER_OPENING_FACE","OBSERVED_SIDE_UNRESOLVED")
            a,b=transform(edge["a"]),transform(edge["b"])
            delta=b-a;length=np.linalg.norm(delta)
            for lo,hi in edge["support_intervals"]:
                gate(0<=lo<=hi<=length+1e-5,"INVALID_SUPPORT_INTERVAL")
                if length>0:segments.append((a+delta*lo/length,a+delta*hi/length))
        predicted.append(segments)
    # Candidate association is unique by the nearest whole-boundary bidirectional
    # cost. Collisions remain failures, rather than selecting a favorable clone.
    assigned={};metrics=[]
    gt_segments=[[(np.r_[a,0],np.r_[b,0]) for a,b in polygon_segments(poly)] for poly in truth.polygons]
    visible_gt=[list(group) for group in gt_segments]
    if truth.inferred_edge:
        visible_gt[0]=[edge for j,edge in enumerate(gt_segments[0]) if j!=2]
        a,b=gt_segments[0][2];u=(b-a)/np.linalg.norm(b-a)
        visible_gt[0]+=[(a,a+u*.35),(b-u*.35,b)]
    for index,segments in enumerate(predicted):
        if not segments:continue
        samples=sample_segments(segments)
        costs=[float(np.mean(distance_to_segments(samples,g)))+float(np.mean(distance_to_segments(sample_segments(g),segments))) for g in gt_segments]
        target=int(np.argmin(costs))
        if target in assigned:gate(False,"DUPLICATE_HATCH");continue
        assigned[target]=index
        gt=visible_gt[target];d=np.r_[distance_to_segments(samples,gt),distance_to_segments(sample_segments(gt),segments)]
        coverage=[float(np.mean(distance_to_segments(sample_segments([edge]),segments)<=config["boundary_p95_m"])) for edge in gt]
        metric=dict(candidate_id=model["hatches"][index]["candidate_id"],gt_hatch=target,boundary=stats(d),edge_coverage=coverage)
        if truth.expected_complete:
            gate(np.quantile(d,.95)<=config["boundary_p95_m"],"BOUNDARY_15CM")
            gate(min(coverage)>=config["coverage_edge_min"],"EDGE_COVERAGE")
            gate(float(np.mean(distance_to_segments(sample_segments(gt),segments)<=config["boundary_p95_m"]))>=config["coverage_total_min"],"TOTAL_COVERAGE")
            h=model["hatches"][index]
            gate(h["status"]==("COMPLETE_WITH_INFERENCE" if truth.inferred_edge else "COMPLETE_OBSERVED"),"WRONG_COMPLETE_STATUS")
            if truth.inferred_edge:gate(h["inferred_edge_count"]==1,"WRONG_INFERRED_EDGE_COUNT")
            if h["nominal_polygon"]:
                corners=np.array([transform(v) for v in h["nominal_polygon"]]);expected=np.column_stack((truth.polygons[target],np.zeros(len(truth.polygons[target]))))
                corner_dist=np.r_[np.linalg.norm(corners[:,None]-expected[None,:],axis=2).min(axis=1),np.linalg.norm(expected[:,None]-corners[None,:],axis=2).min(axis=1)]
                gate(np.quantile(corner_dist,.95)<=config["corner_p95_m"],"CORNER_15CM");metric["corner"]=stats(corner_dist)
        metrics.append(metric)
    gate(len(assigned)==len(truth.polygons),"MISSING_HATCH")
    if truth.expected_complete:gate(len(model["hatches"])==len(truth.polygons),"EXTRA_HATCH")
    else:gate(not complete,"UNOBSERVABLE_EDGE_FORCED_COMPLETE")
    beams=sum(p["kind"]=="BEAM_OR_PARTITION" for p in model["structures"])
    gate(beams==truth.expected_beams,"BEAM_COUNT")
    matched_beams=set();coaming_support=[]
    for primitive in model["structures"]:
        if primitive["kind"]=="BEAM_OR_PARTITION":
            edge=primitive["segment"];segment=[(transform(edge["a"]),transform(edge["b"]))]
            if not truth.beam_segments:gate(False,"EXTRA_CONFIRMED_BEAM");continue
            samples=sample_segments(segment)
            distances=[np.r_[distance_to_segments(samples,[g]),distance_to_segments(sample_segments([g]),segment)] for g in truth.beam_segments]
            target=int(np.argmin([np.mean(d) for d in distances]))
            gate(np.quantile(distances[target],.95)<=config["boundary_p95_m"],"BEAM_GEOMETRY")
            gate(target not in matched_beams,"DUPLICATE_BEAM");matched_beams.add(target)
        if primitive["kind"]!="COAMING_OR_HOLD_WALL":continue
        edge=primitive["segment"];samples=sample_segments([(transform(edge["a"]),transform(edge["b"]))])
        gate(np.quantile(distance_to_segments(samples,[e for group in gt_segments for e in group]),.95)<=config["boundary_p95_m"],"EXTRA_CONFIRMED_COAMING")
        all_gt=[e for group in gt_segments for e in group]
        target=int(np.argmin([np.mean(distance_to_segments(samples,[g])) for g in all_gt]))
        # Confirmed duplicates cannot hide in the primitive table while the
        # hatch table reports only one clean candidate. Compare actual support
        # intervals, not the whole envelope across unobserved gaps.
        a,b=transform(edge["a"]),transform(edge["b"]);length=np.linalg.norm(b-a)
        supported=[(a+(b-a)*lo/length,a+(b-a)*hi/length) for lo,hi in edge["support_intervals"]] if length else []
        if supported:
            ss=sample_segments(supported)
            for previous_target,previous in coaming_support:
                if previous_target==target and min(np.mean(distance_to_segments(ss,previous)<=config["boundary_p95_m"]),np.mean(distance_to_segments(sample_segments(previous),supported)<=config["boundary_p95_m"]))>=config["duplicate_overlap_min"]:
                    gate(False,"DUPLICATE_CONFIRMED_COAMING")
            coaming_support.append((target,supported))
    return dict(status="PASS" if not failures else "FAIL",failures=sorted(set(failures)),hatches=metrics,
                deck=deck_metrics)
