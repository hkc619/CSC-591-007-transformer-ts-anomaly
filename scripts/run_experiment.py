"""單一實驗入口：外部 roots、可選 array row、即時 log；不提交 Slurm。"""
import argparse
import codecs
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paths import resolve_roots


def main():
    parser=argparse.ArgumentParser(description='執行單一 external-data TranAD v2 run')
    parser.add_argument('--manifest')
    parser.add_argument('--task-id',type=int)
    parser.add_argument('--project-root')
    parser.add_argument('--data-root')
    parser.add_argument('--results-root')
    parser.add_argument('--hpc',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args,remaining=parser.parse_known_args()
    roots=resolve_roots(args)
    selected=[]
    if args.manifest:
        import csv
        manifest=Path(args.manifest)
        if not manifest.is_absolute():manifest=roots.project/manifest
        with manifest.open() as f:rows=list(csv.DictReader(f))
        if args.task_id is None or not 0<=args.task_id<len(rows):parser.error('manifest 必須指定合法 --task-id')
        row=rows[args.task_id]
        expected_fixed={'protocol':'v2','seed':'1','epochs':'5','batch_size':'128','eval_batch_size':'128'}
        if any(row.get(k)!=v for k,v in expected_fixed.items()):raise ValueError('manifest 不符合固定 formal protocol')
        config=json.loads((ROOT/'configs/formal_protocol_v2.json').read_text())
        if len(rows)!=56 or row['target_machine'] not in config['machines']:raise ValueError('manifest 不是固定 56-run SMD 設計')
        if int(row['task_id'])!=args.task_id:raise ValueError('manifest row index 與 task_id 不一致')
        sources=json.loads(row['source_machines']);target=row['target_machine']
        if row['setting']=='per_machine':
            assert sources==[target];selected=['--machine',target]
        elif row['setting']=='cross_machine_zero_shot':
            assert sources==[m for m in config['machines'] if m!=target]
            selected=['--train-machines',*sources,'--test-machine',target]
        else:raise ValueError('未知 setting')
        # 不允許以 CLI 覆寫 manifest 的正式設定。
        if remaining:parser.error('manifest mode 不接受其他模型/訓練覆寫參數')
    elif args.task_id is not None:parser.error('--task-id 需要 --manifest')
    command=[sys.executable,'-u',str(ROOT/'main.py'),*selected,*remaining]
    env=dict(os.environ,PROJECT_ROOT=str(roots.project),DATA_ROOT=str(roots.data),RESULTS_ROOT=str(roots.results),PYTHONDONTWRITEBYTECODE='1')
    env.setdefault('MPLBACKEND','Agg');env.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'tranad-deployment-matplotlib'))
    if args.hpc:command+=['--hpc']
    if args.dry_run:
        return subprocess.run([*command,'--dry-run'],env=env,cwd=ROOT).returncode
    logs=roots.results/'logs';logs.mkdir(parents=True,exist_ok=True)
    token=uuid4().hex[:10];log=logs/f'launch-{token}.log'
    print('即時 log：',log,flush=True)
    folder=None;pending='';live=None;interrupted=False
    decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
    with log.open('xb') as output:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0,start_new_session=True)
        try:
            while True:
                chunk=os.read(child.stdout.fileno(),8192)
                if not chunk:break
                output.write(chunk);output.flush()
                if live:live.write(chunk);live.flush()
                text=decoder.decode(chunk);sys.stdout.write(text);sys.stdout.flush()
                if folder is None:
                    pending+=text
                    match=re.search(r'Run directory: ([^\r\n]+)[\r\n]',pending)
                    if match:
                        folder=Path(match.group(1));live=(folder/'run.log').open('xb');live.write(log.read_bytes());live.flush()
            code=child.wait()
        except KeyboardInterrupt:
            interrupted=True
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            code=130
        finally:
            if child.poll() is None:child.kill();child.wait()
            child.stdout.close()
            if live:live.close()
    record=dict(command=command,exit_code=code,interrupted=interrupted,run_path=str(folder) if folder else None,
                task_id=args.task_id,manifest=args.manifest,created_at=datetime.now(timezone.utc).isoformat(),
                roots={key:str(getattr(roots,key)) for key in ['project','data','results']})
    log.with_suffix('.json').write_text(json.dumps(record,indent=2))
    if folder:(folder/'launch.json').write_text(json.dumps(record,indent=2))
    if args.manifest:
        status=roots.results/'task_status';status.mkdir(exist_ok=True)
        (status/f'{args.task_id:03d}-{token}.json').write_text(json.dumps(record,indent=2))
    return code

if __name__=='__main__':raise SystemExit(main())
