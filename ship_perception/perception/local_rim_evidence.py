"""Cell-local Raw plane, step and continuation evidence in one review frame."""

import hashlib
import numpy as np

from .sections import physical_sections


def digest(indexes):
    return hashlib.sha256(np.sort(indexes).astype("<i8",copy=False).tobytes()).hexdigest()


def plane(points, config, horizontal):
    minimum=config["geometry"]["normal_min_points"]
    if len(points)<minimum:
        return None
    center=np.median(points,axis=0)
    _,s,v=np.linalg.svd(points-center,full_matrices=False)
    if len(s)<3 or s[1]<=np.finfo(float).eps:
        return None
    normal=v[-1]
    residual=float(np.median(abs((points-center)@normal)))
    tilt=float(abs(normal[2]))
    angular=np.sin(np.deg2rad(config["geometry"]["normal_angle_deg"]))
    qualified=(tilt>=np.cos(np.deg2rad(config["boundary"]["profile_deck_slope_max_deg"]))
               if horizontal else tilt<=angular)
    return dict(normal_stz=normal.tolist(),residual_mad_m=residual,
                point_count=len(points),qualified=bool(qualified and residual<=config["geometry"]["plane_inlier_m"]),
                role="HORIZONTAL_DECK" if horizontal else "VERTICAL_FACE")


class LocalRimEvidence:
    def __init__(self, measurement, config):
        self.measurement=measurement
        self.config=config
        primary=getattr(measurement.local,"primary_cache",measurement.local)
        self.owned_ids=primary.vessel.point_indexes
        self.index=measurement.spatial
        self.xyz=measurement.raw_uvz
        self.ids=measurement.raw_point_indexes

    def side(self, polygon, side):
        pairs={"U0":(0,3),"U1":(1,2),"V0":(0,1),"V1":(3,2)}
        vertices=np.asarray(polygon,float)
        a,b=vertices[list(pairs[side])]
        tangent=(b-a)/np.linalg.norm(b-a)
        normal=np.array([-tangent[1],tangent[0]])
        if (vertices.mean(0)-a)@normal<0:
            normal=-normal
        length=float(np.linalg.norm(b-a))
        radius=self.config["roi"]["support_search_m"]
        corners=np.array([a+normal*radius,a-normal*radius,b+normal*radius,b-normal*radius])
        margin=self.config["boundary"]["profile_half_length_m"]
        low=corners.min(0)-margin
        high=corners.max(0)+margin
        candidates=self.index.query_indexes(np.r_[low,high])
        owned=np.isin(self.ids[candidates],self.owned_ids)
        positives=candidates[owned]
        st=np.column_stack(((self.xyz[positives,:2]-a)@tangent,
                            (self.xyz[positives,:2]-a)@normal,self.xyz[positives,2]))
        keep=(st[:,0]>=-margin)&(st[:,0]<=length+margin)&(abs(st[:,1])<=radius)
        positives,st=positives[keep],st[keep]
        edges=physical_sections((0.,length),self.config["geometry"]["candidate_voxel_m"],self.config)
        sections=[]
        for lo,hi in zip(edges,edges[1:]):
            mask=(st[:,0]>=lo)&(st[:,0]<hi)
            points=st[mask]
            indexes=self.ids[positives[mask]]
            sections.append(self._section(points,indexes,(lo,hi)))
        minimum=self.config["boundary"]["profile_min_sections"]
        modes=[(i,m) for i,s in enumerate(sections) for m in s["modes"] if m["inner_rim_positive"]]
        tolerance=self.config["boundary"]["corner_join_m"]
        groups=[]
        for i,mode in sorted(modes,key=lambda row:row[1]["offset_m"]):
            if not groups or mode["offset_m"]-groups[-1][0][1]["offset_m"]>tolerance:
                groups.append([])
            groups[-1].append((i,mode))
        coherent=[]
        for group in groups:
            unique={i:m for i,m in group}
            if len(unique)<minimum:
                continue
            offsets=np.array([m["offset_m"] for m in unique.values()])
            value=float(np.median(offsets))
            coherent.append(dict(offset_m=value,independent_sections=len(unique),
                 section_ids=sorted(unique),uncertainty_m=max(
                     self.config["geometry"]["plane_inlier_m"],float(np.median(abs(offsets-value)))),
                 owned_raw_source_hashes=sorted({m["raw_index_digest"] for m in unique.values()})))
        band=self.config["boundary"]["side_probe_m"]
        extension=(abs(st[:,1])<=band)&((st[:,0]<0)|(st[:,0]>length))
        left=extension&(st[:,0]<0)
        right=extension&(st[:,0]>length)
        near=[r for r in coherent if abs(r["offset_m"])<=tolerance]
        deck=sum(s["legacy_deck_observed"] for s in sections)
        inside_only=sum(s["legacy_inside_occupied"] and not s["legacy_outside_occupied"] for s in sections)
        ridge=sum(s["legacy_both_sides_cargo"] for s in sections)
        has_extension=(left.sum()>=self.config["boundary"]["profile_min_points"]
                       and right.sum()>=self.config["boundary"]["profile_min_points"])
        if len(coherent)>1:
            state="MULTIMODE_COMPETING"
        elif near:
            state="INNER_RIM_OBSERVED"
        elif has_extension and deck<minimum and sections:
            state="HULL_OR_OUTER_EDGE_SUSPECT"
        elif ridge>=minimum:
            state="CARGO_RIDGE_SUSPECT"
        elif inside_only>=minimum or not len(positives):
            state="UNOBSERVED_OR_CROPPED"
        else:
            state="RETAINED_LEGACY_REVIEW"
        return dict(state=state,old_line=dict(origin_uv=a.tolist(),tangent_uv=tangent.tolist(),
                    inward_normal_uv=normal.tolist(),length_m=length),
                    selected_reason="LOCAL_RAW_ROLE_DIAGNOSIS_GEOMETRY_RETAINED",
                    new_line=None,rejected_alternatives=coherent,
                    sections=sections,independent_section_count=len(sections),
                    current_owned_raw_count=len(positives),raw_index_digest=digest(self.ids[positives]),
                    contamination_raw_count=int((~owned).sum()),
                    continuation=dict(before_count=int(left.sum()),after_count=int(right.sum()),
                                      raw_index_digest=digest(self.ids[positives[extension]])),
                    frame_contract="B1_UV_CONTINUOUS_CELL_LINE_V1",steel_certified=False)

    def _section(self, points, indexes, span):
        cfg=self.config
        minimum=cfg["boundary"]["profile_min_points"]
        probe=cfg["boundary"]["side_probe_m"]
        step=cfg["geometry"]["candidate_voxel_m"]
        def observations(offset):
            outward=(points[:,1]>=offset-probe)&(points[:,1]<offset)
            inward=(points[:,1]>=offset)&(points[:,1]<offset+probe)
            face=abs(points[:,1]-offset)<=cfg["geometry"]["plane_inlier_m"]
            deck=plane(points[outward],cfg,True)
            wall=plane(points[face],cfg,False)
            ready=min(outward.sum(),inward.sum())>=minimum
            contrast=float(np.median(points[inward,2])-np.median(points[outward,2])) if ready else None
            positive=bool(ready and deck and deck["qualified"] and wall and wall["qualified"]
                          and abs(contrast)>=cfg["boundary"]["profile_min_drop_m"]
                          and np.ptp(points[face,2])>=cfg["boundary"]["face_vertical_span_min_m"])
            return dict(offset_m=float(offset),deck_plane=deck,vertical_face=wall,
                        signed_height_change_m=contrast,inner_rim_positive=positive,
                        raw_index_digest=digest(indexes[outward|inward|face]),
                        deck_raw_index_digest=digest(indexes[outward]),
                        face_raw_index_digest=digest(indexes[face]),
                        inward_raw_index_digest=digest(indexes[inward]),
                        outward_count=int(outward.sum()),inward_count=int(inward.sum()))
        legacy=observations(0.) if len(points) else None
        modes=[]
        if len(points)>=2*minimum:
            origin=float(points[:,1].min())
            bins=np.floor((points[:,1]-origin)/step).astype(int)
            counts=np.bincount(bins)
            ordered=np.argsort(bins,kind="stable")
            ends=np.cumsum(counts)
            z=np.full(len(counts),np.nan)
            for k in np.flatnonzero(counts>=minimum):
                z[k]=np.median(points[ordered[ends[k]-counts[k]:ends[k]],2])
            jumps=np.flatnonzero(np.isfinite(z[1:])&np.isfinite(z[:-1])&
                                 (abs(np.diff(z))>=cfg["roi"]["opening_drop_m"]))+1
            for k in jumps:
                rho=origin+k*step
                band=abs(points[:,1]-rho)<=step
                if not band.any():
                    continue
                # Raw positions nominate a narrow candidate, while planes
                # and source ownership decide its physical eligibility.
                mode=observations(float(np.median(points[band,1])))
                modes.append(mode)
        return dict(span_m=list(span),raw_count=len(points),raw_index_digest=digest(indexes),modes=modes,
                    legacy_deck_observed=bool(legacy and legacy["deck_plane"] and legacy["deck_plane"]["qualified"]),
                    legacy_inside_occupied=bool(legacy and legacy["inward_count"]>=minimum),
                    legacy_outside_occupied=bool(legacy and legacy["outward_count"]>=minimum),
                    legacy_both_sides_cargo=bool(legacy and legacy["inward_count"]>=minimum and
                        legacy["outward_count"]>=minimum and not (legacy["deck_plane"] and legacy["deck_plane"]["qualified"])),
                    legacy=legacy)
