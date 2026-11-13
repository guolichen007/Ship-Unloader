"""Export review dimensions from the native opening_cells.json contract."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ship_perception.perception.r3h_geometry import dimensions, write_dimensions_csv


def export_cells(path,input_sha=None,source_sha=None):
    document=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document,list):
        raise ValueError("OPENING_CELLS_MUST_BE_ARRAY")
    vessels={}
    for cell in document:
        g=cell.get("geometry")
        if g is None:
            continue
        corners=g.get("corners_xyz")
        if corners is None:
            raise ValueError("CELL_NULLABLE_XYZ_NOT_EXPORTED")
        row=dict(opening_id=cell["cell_id"],geometry_state=cell["strict_rectangle_state"],
                 steel_edge_certified=False,**dimensions(g["boundary_xy"]),
                 corners_xyz=[dict(id="P"+str(i+1),x=p[0],y=p[1],z=p[2]) for i,p in enumerate(corners)],
                 edge_roles={s:cell.get("local_rim_evidence",{}).get(s,{}).get("state","RETAINED_LEGACY_REVIEW")
                             for s in ("U0","U1","V0","V1")})
        vessels.setdefault(cell["vessel_id"],[]).append(row)
    return dict(schema_version="r3h-hatch-geometry-v1",scene_source_sha256=input_sha,
                runtime_source_sha256=source_sha,vessels=[dict(vessel_id=k,hatches=v) for k,v in vessels.items()])


def print_compact(compact):
    for vessel in compact["vessels"]:
        for hatch in vessel["hatches"]:
            print(vessel["vessel_id"]+" | "+hatch["opening_id"]+" | "+hatch["geometry_state"])
            print("Corner       X(m)       Y(m)       Z(m)")
            for p in hatch["corners_xyz"]:
                z="—" if p["z"] is None else format(p["z"],".2f")
                print("{id:>4} {x:11.2f} {y:11.2f} {z:>10}".format(**dict(p,z=z)))
            print("Length x Width = {:.2f} x {:.2f} m".format(hatch["length_m"],hatch["width_m"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opening_cells",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    native=args.opening_cells.parent/"03_results.json"
    identity=json.loads(native.read_text(encoding="utf-8")) if native.exists() else {}
    result=export_cells(args.opening_cells,identity.get("input_sha256"),identity.get("runtime_source_sha256"))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    write_dimensions_csv(args.output.with_suffix(".csv"),result)
    print_compact(result)


if __name__=="__main__":
    main()
