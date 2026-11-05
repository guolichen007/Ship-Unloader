"""Stable per-cell XY dimensions and nullable XYZ export for operator review."""

import csv
import math


def dimensions(points):
    if len(points)!=4 or any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError("INVALID_REVIEW_FOUR_CORNERS")
    edges=[(points[(i+1)%4][0]-p[0],points[(i+1)%4][1]-p[1]) for i,p in enumerate(points)]
    lengths=[math.hypot(*e) for e in edges]
    if min(lengths)<=0:
        raise ValueError("DEGENERATE_REVIEW_FOUR_CORNERS")
    pairs=[(lengths[0]+lengths[2])/2,(lengths[1]+lengths[3])/2]
    angles=[math.degrees(math.acos(max(-1.,min(1.,
        sum(a*b for a,b in zip(e,edges[(i+1)%4]))/(lengths[i]*lengths[(i+1)%4])))))
        for i,e in enumerate(edges)]
    local=[(p[0]-points[0][0],p[1]-points[0][1]) for p in points]
    area=abs(sum(p[0]*local[(i+1)%4][1]-p[1]*local[(i+1)%4][0] for i,p in enumerate(local))/2)
    return dict(length_m=max(pairs),width_m=min(pairs),area_m2=area,polygon_area_m2=area,
                opposite_side_mismatch_m=max(abs(lengths[0]-lengths[2]),abs(lengths[1]-lengths[3])),
                max_orthogonality_error_deg=max(abs(a-90) for a in angles),
                dimension_state="REVIEW_GEOMETRY")


def compact_hatches(payload):
    g=payload["geometry_only"]
    vessels={}
    for cell in g.get("selected_opening_cells",[]):
        geometry=cell.get("geometry")
        if geometry is None:
            continue
        strict=next((r for r in g["hatches"] if r["opening_identity"]==cell["primary_region_id"]),None)
        if strict:
            xyz=[[float(x),float(y),h["z_m"]] for (x,y),h in zip(strict["boundary_xy"],strict["corner_heights"])]
        else:
            xyz=geometry["corners_xyz"]
        poly=geometry["boundary_xy"]
        row=dict(opening_id=cell["cell_id"],geometry_state=cell["strict_rectangle_state"],
                 steel_edge_certified=False,**dimensions(poly),
                 corners_xyz=[dict(id="P"+str(i+1),x=float(p[0]),y=float(p[1]),
                                  z=None if p[2] is None else float(p[2])) for i,p in enumerate(xyz)],
                 edge_roles={side:cell.get("local_rim_evidence",{}).get(side,{}).get("state","RETAINED_LEGACY_REVIEW")
                             for side in ("U0","U1","V0","V1")})
        vessels.setdefault(cell["vessel_id"],[]).append(row)
    return dict(schema_version="r3h-hatch-geometry-v1",
                scene_source_sha256=payload["input_sha256"],
                runtime_source_sha256=payload["runtime_source_sha256"],
                coordinate_frame=payload["input"]["frame_id"],display_decimals=2,
                vessels=[dict(vessel_id=key,hatches=rows) for key,rows in vessels.items()])


def write_dimensions_csv(path, compact):
    fields=["vessel_id","opening_id","geometry_state","length_m","width_m","polygon_area_m2",
            "opposite_side_mismatch_m","max_orthogonality_error_deg"]
    fields += ["P"+str(i)+"_"+axis for i in range(1,5) for axis in ("x","y","z")]
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for vessel in compact["vessels"]:
            for hatch in vessel["hatches"]:
                row={k:hatch.get(k) for k in fields[:8]}
                row["vessel_id"]=vessel["vessel_id"]
                for point in hatch["corners_xyz"]:
                    for axis in ("x","y","z"):
                        row[point["id"]+"_"+axis]=point[axis]
                writer.writerow(row)
