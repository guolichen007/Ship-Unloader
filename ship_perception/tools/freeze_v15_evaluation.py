"""只有真实能力及完整合成/集成报告通过后才签发封存集配置冻结记录。"""
import argparse
import json
from pathlib import Path
import subprocess
from v15_artifacts import dump,file_hash
from v15_holdout_ledger import current_identity


def freeze(reports,identity,clean):
    if not clean:raise ValueError('FREEZE_REQUIRES_CLEAN_GIT_SHA')
    for name,report in reports.items():
        if report.get('config_hash')!=identity['config_hash']:raise ValueError('PREREQUISITE_CONFIG_MISMATCH:'+name)
        sha=report.get('git_sha',report.get('compiled_sha'))
        if sha!=identity['git_sha'] or report.get('worktree_dirty',True):raise ValueError('PREREQUISITE_EXACT_SHA:'+name)
    d=reports['development'];s=reports['synthetic'];i=reports['integration']
    if d.get('REAL_DEVELOPMENT_OPERATIONAL_GATE')!='PASS' or d.get('RUN_COMPLETENESS')!='PASS':raise ValueError('DEVELOPMENT_OPERATIONAL_NOT_PASSED')
    if s.get('status')!='PASS' or s.get('mode')!='full' or s.get('acceptance_coverage')!='FULL_V15_ACCEPTANCE':raise ValueError('FULL_SYNTHETIC_COVERAGE_NOT_PASSED')
    if i.get('status')!='PASS' or i.get('profile')!='full' or i.get('reused_binary_outputs') or not i.get('qualified_backends') or i.get('both_backend_execution_integrity')!='PASS':raise ValueError('FULL_INTEGRATION_NOT_PASSED')
    return dict(status='FROZEN_FOR_HOLDOUT',**identity)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for lane in ('development','synthetic','integration'):p.add_argument('--'+lane+'-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('FREEZE_RECORD_ALREADY_EXISTS')
    paths={lane:getattr(args,lane+'_report') for lane in ('development','synthetic','integration')}
    reports={lane:json.loads(path.read_text(encoding='utf-8')) for lane,path in paths.items()}
    clean=not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    record=freeze(reports,current_identity(),clean)
    record['prerequisite_reports']={lane:dict(path=str(path.resolve()),sha256=file_hash(path)) for lane,path in paths.items()}
    dump(args.output,record);return 0


if __name__=='__main__':raise SystemExit(main())
