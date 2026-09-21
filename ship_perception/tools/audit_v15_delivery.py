"""交付审查：开发测试通过不能替代完整版本覆盖、真实能力和正式验证。"""
import argparse
import json
from pathlib import Path
import subprocess
from v15_artifacts import dump,file_hash
from v15_holdout_ledger import DEFAULT_LEDGER,read_ledger,current_identity
from verify_v14_frozen import verify,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic-report',type=Path,required=True)
    p.add_argument('--development-report',type=Path,required=True)
    p.add_argument('--integration-report',type=Path)
    p.add_argument('--decoder-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();blockers=[];evidence={}
    def read(name,path):
        if path is None or not path.is_file():return None
        value=json.loads(path.read_text(encoding='utf-8'))
        evidence[name]=dict(path=str(path),sha256=file_hash(path))
        return value
    synthetic=read('synthetic',args.synthetic_report);real=read('development',args.development_report)
    integration=read('integration',args.integration_report);decoder=read('decoder',args.decoder_report)
    identity=current_identity()
    for name,report in (('synthetic',synthetic),('development',real)):
        if not report:raise ValueError('MISSING_'+name.upper()+'_REPORT')
        if report.get('config_hash')!=identity['config_hash']:blockers.append(name.upper()+'_CONFIG_MISMATCH')
    matrix=synthetic.get('results',[])
    expected={(scene,seed) for scene in synthetic.get('scenarios',[]) for seed in (42,1337,2026)}
    actual=[(r['scenario'],r['seed']) for r in matrix]
    complete=bool(expected) and len(actual)==len(set(actual)) and set(actual)==expected and synthetic.get('mode')=='full'
    matrix_gate='PASS' if complete and all(r['status']=='PASS' for r in matrix) else 'FAIL'
    # This marker is emitted by the runner until ALL plan capabilities have
    # actual producers, fixtures and counterexamples. A green subset is not Full.
    geometry_gate=matrix_gate if synthetic.get('acceptance_coverage')=='FULL_V15_ACCEPTANCE' else 'INCOMPLETE_COVERAGE'
    if geometry_gate!='PASS':blockers.append('SYNTHETIC_FULL_COVERAGE_NOT_ESTABLISHED')
    real_gate=real.get('REAL_DEVELOPMENT_OPERATIONAL_GATE','NOT_RUN')
    if real_gate!='PASS':blockers.append('REAL_DEVELOPMENT_OPERATIONAL_GATE')
    integration_gate=integration.get('status','NOT_RUN') if integration else 'NOT_RUN'
    if integration:
        qualified=[]
        for backend in ('GICP','VGICP'):
            rows=[r for r in integration.get('series',[]) if r['method']==backend]
            expected_series={(scene,seed) for scene in ('CLEAN','DYNAMIC') for seed in (42,1337,2026)}
            if len(rows)==6 and {(r['scene'],r['seed']) for r in rows}==expected_series and all(r['status']=='PASS' for r in rows):qualified.append(backend)
        if not qualified or integration.get('profile')!='full' or integration.get('reused_binary_outputs') or integration.get('both_backend_execution_integrity')!='PASS':
            integration_gate='FAIL'
        if integration.get('config_hash')!=identity['config_hash']:integration_gate='STALE_CONFIG'
    if integration_gate!='PASS':blockers.append('V14_INTEGRATION_GATE')
    ledger=read_ledger(DEFAULT_LEDGER)
    scored=any(r['access_purpose']=='ALGORITHM_SCORING' for r in ledger)
    contaminated=any(r['HOLDOUT_CONTAMINATED']=='YES' for r in ledger)
    holdout='SCORED_REQUIRES_COMPLETE_13_SCAN_23_ANNOTATION_REPORT' if scored else 'NOT_SCORED_CONFIG_NOT_FROZEN'
    # A ledger receipt alone is never proof of successful evaluation.
    all_scans=real.get('scans',[])
    full_real=(real.get('FULL_REAL_RUN_COMPLETENESS')=='PASS' and len(all_scans)==13 and
        len({r['scan_id'] for r in all_scans})==13 and sum(r['annotation_count'] for r in all_scans)==23 and
        all(r['execution_status']=='COMPLETED' for r in all_scans) and scored)
    if full_real:holdout='SCORED_WEAK_LABEL_DIAGNOSTICS'
    else:blockers.append('FULL_REAL_13_SCAN_23_ANNOTATION_REPORT_NOT_ARCHIVED')
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    if dirty:blockers.append('FINAL_CLEAN_EXACT_SHA_NOT_ESTABLISHED')
    try:frozen=verify()
    except ValueError as error:frozen=dict(status='FAIL',error=str(error));blockers.append('V14_FROZEN_PATH_GATE')
    if not decoder or decoder.get('pcl_crosscheck')!='PASS':blockers.append('UBUNTU_PCL_DECODER_CROSSCHECK_PENDING')
    report=dict(schema='ship_perception.v15.delivery_audit.1',**identity,evidence=evidence,worktree_dirty=dirty,
        status='NOT_READY' if blockers else 'READY_FOR_UNIFIED_REVIEW',blockers=blockers,
        SYNTHETIC_IMPLEMENTED_MATRIX_GATE=matrix_gate,SYNTHETIC_GEOMETRY_GATE=geometry_gate,
        REAL_DEVELOPMENT_OPERATIONAL_GATE=real_gate,real_counts=real['counts'],
        REAL_WEAK_LABEL_DIAGNOSTICS=real['RUN_COMPLETENESS'],SEALED_WEAK_HOLDOUT_STATUS=holdout,
        HOLDOUT_CONTAMINATED='YES' if contaminated else 'NO',V14_INTEGRATION_GATE=integration_gate,
        V14_FROZEN_PATH_GATE=frozen,REAL_FORMAL_15CM='PENDING_GOLDEN',SITE_ACCURACY='SITE_PENDING',
        IMPLEMENTATION_SHA=None,baseline_created=False,main_merge_allowed=False)
    dump(args.output,report);print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if blockers else 0


if __name__=='__main__':raise SystemExit(main())
