"""Development 候选生成/淘汰/排序取证；不使用标注给产品选择平面。"""
import argparse
import collections
import csv
import json
import os
from pathlib import Path
import subprocess
from v15_artifacts import dump, file_hash
from v15_config_identity import config_file_hash
from v15_pcd import decode, write_cache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[1]
    manifest=json.loads((source/'datasets/v15_dataset_manifest.json').read_text(encoding='utf-8'))
    scans=[s for s in manifest['scans'] if s['split_role']=='DEVELOPMENT']
    records=[];all_rows=[]
    for scan in scans:
        folder=args.output/scan['scan_id'];folder.mkdir(exist_ok=True)
        path=args.data_root/scan['pcd_path']
        if file_hash(path)!=scan['pcd_sha256']:raise ValueError('PCD_HASH_MISMATCH')
        cache=folder/'input.xyzbin';points,_=decode(path);write_cache(points,cache)
        process=subprocess.run([str(args.executable.resolve()),'offline',str(cache.resolve()),str((folder/'model.json').resolve())],
            capture_output=True,text=True,timeout=600,env=dict(os.environ,SHIP_V15_DIAGNOSTICS='1'))
        (folder/'process.log').write_text(process.stdout+process.stderr,encoding='utf-8')
        if process.returncode:raise ValueError('DIAGNOSTIC_EXECUTION_FAILED')
        candidates=[json.loads(line[len('DECK_AUDIT '):]) for line in process.stderr.splitlines() if line.startswith('DECK_AUDIT ')]
        selection=[json.loads(line[len('DECK_SELECTION '):]) for line in process.stderr.splitlines() if line.startswith('DECK_SELECTION ')]
        if len(selection)!=1 or selection[0]['candidate_count']!=len(candidates):raise ValueError('INCOMPLETE_CANDIDATE_AUDIT')
        selected=selection[0];eligible=sorted([c for c in candidates if c['reason']=='ELIGIBLE_FOR_RANKING'],key=lambda c:-c['total_score'])
        for candidate in candidates:
            candidate['manual_physical_class']='UNKNOWN'
            candidate['selected']=selected['valid'] and abs(candidate['offset']-selected['offset'])<1e-10 and all(abs(a-b)<1e-10 for a,b in zip(candidate['normal'],selected['normal']))
            all_rows.append(dict(scan_id=scan['scan_id'],**candidate))
        gap=(eligible[0]['total_score']-eligible[1]['total_score'])/max(1e-12,eligible[0]['total_score']) if len(eligible)>1 else None
        record=dict(scan_id=scan['scan_id'],candidate_count=len(candidates),eligible_count=len(eligible),selection=selected,
            top_two=eligible[:2],relative_top_two_gap=gap,rejection_counts=dict(collections.Counter(c['reason'] for c in candidates)),
            failure_stage='NO_ELIGIBLE_CANDIDATE' if not eligible else ('COMPETING_CANDIDATES' if not selected['valid'] else 'PHYSICAL_SELECTION_REQUIRES_REVIEW'),
            missing_metrics={'vertical_support':'NOT_COMPUTED_BY_CURRENT_PRODUCT','background_penalty':'NO_SEPARATE_TERM_IN_CURRENT_PRODUCT'},
            physical_deck_generation_verdict='UNKNOWN_REQUIRES_INDEPENDENT_GEOMETRY_REVIEW')
        dump(folder/'candidates.json',candidates);dump(folder/'summary.json',record);records.append(record)
        print(scan['scan_id'],len(candidates),'candidates;',len(eligible),'eligible;',record['failure_stage'],flush=True)
    report=dict(status='AUDIT_COMPLETE_NOT_ACCEPTANCE',config_hash=config_file_hash(source/'config/v15.json'),
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),executable_sha256=file_hash(args.executable),
        holdout_algorithm_scored=False,scans=records)
    dump(args.output/'report.json',report)
    if all_rows:
        with (args.output/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    return 0


if __name__=='__main__':raise SystemExit(main())
