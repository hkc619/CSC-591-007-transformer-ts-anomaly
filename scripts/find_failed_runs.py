"""列出缺失、失敗或缺少評估輸出的 array tasks；不重試、不提交工作。"""
import argparse
import csv
import json
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paths import resolve_roots


def latest_status(results):
    latest={}
    for path in (results/'task_status').glob('*.json'):
        r=json.loads(path.read_text());key=r['task_id']
        if key not in latest or path.stat().st_mtime>latest[key][0]:latest[key]=(path.stat().st_mtime,r)
    return {k:v[1] for k,v in latest.items()}


def main():
    parser=argparse.ArgumentParser()
    for name in ['project-root','data-root','results-root']:parser.add_argument('--'+name)
    parser.add_argument('--hpc',action='store_true');args=parser.parse_args();roots=resolve_roots(args)
    with (ROOT/'configs/formal_28_machine_runs.csv').open() as f:rows=list(csv.DictReader(f))
    statuses=latest_status(roots.results);failed=[]
    for row in rows:
        status=statuses.get(int(row['task_id']))
        if not status or status['exit_code']!=0 or not status['run_path'] or not (Path(status['run_path'])/'evaluation.json').is_file():failed.append(row['task_id'])
    print(json.dumps(dict(failed_or_missing_task_ids=failed,count=len(failed)),indent=2))
    print('可重跑的 --array='+','.join(failed) if failed else '全部完成；未提交任何 job。')

if __name__=='__main__':main()
