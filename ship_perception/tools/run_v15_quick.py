"""统一 Quick：7 场景、2 份 Development、两个后端的两类 50 帧集成。"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

TOOLS=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recognizer',type=Path,required=True)
    p.add_argument('--integration',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    # Each run owns a fresh artifact directory. An earlier PASS cannot survive
    # a failed producer by leaving old files in the same directory.
    directory=a.output/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True,exist_ok=False)
    commands={
        'synthetic':[str(TOOLS/'run_v15_synthetic.py'),str(a.recognizer.resolve()),'quick',str(directory/'synthetic')],
        'real_smoke':[str(TOOLS/'run_v15_development.py'),str(a.recognizer.resolve()),'--data-root',str(a.data_root.resolve()),'--smoke','--output',str(directory/'real_smoke')],
        'integration':[str(TOOLS/'run_v15_integration.py'),str(a.integration.resolve()),'quick',str(directory/'integration')],
    }
    result={'schema':'ship_perception.v15.quick.1','profile':'quick','formal_acceptance':False,
            'holdout_algorithm_scored':False,'status':'FAIL','lanes':{}}
    for lane,command in commands.items():
        print('QUICK',lane,flush=True)
        try:
            with (directory/(lane+'.log')).open('w',encoding='utf-8') as log:
                run=subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT,timeout=7200)
            path=directory/lane/'report.json'
            report=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
            status=report.get('QUICK_REAL_SMOKE' if lane=='real_smoke' else 'status') if report else None
            result['lanes'][lane]={'status':'PASS' if run.returncode==0 and status=='PASS' else 'FAIL',
                                   'returncode':run.returncode,'report':str(path),'reported_status':status}
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            result['lanes'][lane]={'status':'FAIL','error':str(error)}
        print(lane,result['lanes'][lane]['status'],flush=True)
    if all(v['status']=='PASS' for v in result['lanes'].values()):result['status']='PASS'
    (directory/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('QUICK_REPORT',directory/'report.json',flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
