#!/usr/bin/env python3
"""Universal seeds 2/3 orchestration；不提交 Slurm、不重寫任何 scientific calculation。"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paths import resolve_roots
from src.universal_replication_protocol import valid_seed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=[1,2,3],required=True)
    parser.add_argument("--phase",choices=["plan","train","evaluate","threshold"],default="plan")
    parser.add_argument("--step",choices=["smoke","full"],help=argparse.SUPPRESS)
    for name in ("project-root","data-root","results-root"):
        parser.add_argument("--"+name)
    parser.add_argument("--hpc",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--retry",action="store_true")
    args=parser.parse_args();valid_seed(args.seed);roots=resolve_roots(args)
    if args.phase in ("plan","train"):
        from src.universal_training import execute,configuration,inspect_train,describe,namespace
        if args.phase=="plan":
            import json
            config=configuration(roots.project,args.seed)
            plan=describe(roots,config,inspect_train(roots,config))
            base=namespace(roots.results)
            plan["replication"]=dict(seed=args.seed,stage2_path=str(base/"evaluation"/f"seed{args.seed}"/"<checkpoint-sha256>/<evaluation-code-sha256>"),
                stage3_path=str(base/"thresholding"/f"seed{args.seed}"/"<checkpoint-sha256>/<stage3-code-sha256>"),
                dependency_gates=["same-seed Stage1 completed","same-seed Stage2 full completed"],pot_rng_seed=1,
                phases=["train","evaluate","threshold"],experiments_executed=False)
            print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
        if args.seed==1 and not args.dry_run and args.phase!="plan":
            parser.error("Replication 不重跑 seed 1；其既有入口保持不變")
        execute(roots,seed=args.seed,retry=args.retry,dry_run=args.dry_run or args.phase=="plan")
        return 0
    if args.seed==1:
        parser.error("Replication 不重跑 seed 1；aggregation 唯讀載入既有結果")
    if args.dry_run or args.step:
        step=args.step or "full"
        if args.phase=="evaluate":
            from src.universal_replication import evaluate
            evaluate(roots,args.seed,mode=step,dry_run=args.dry_run,retry=args.retry)
        else:
            from src.universal_thresholding import execute
            execute(roots,seed=args.seed,phase=step,dry_run=args.dry_run,retry=args.retry)
        return 0
    # Fresh process 隔離原 main.py 的 global args；不複製 numerical pipeline。
    from src.universal_replication import completed_attempt
    from src.universal_replication_protocol import resolve_training,evaluation_root
    if args.phase=="evaluate":
        bundle=resolve_training(roots,args.seed)
        parent=evaluation_root(roots.results,bundle)
    else:
        from src.universal_threshold_artifacts import resolve_inputs,root_path
        inputs=resolve_inputs(roots,seed=args.seed)
        parent=root_path(roots,inputs)
    for step in ("smoke","full"):
        if completed_attempt(parent/step) is not None:
            print(f"保留既有 completed {args.phase} {step}；不重跑",flush=True)
            continue
        command=[sys.executable,"-u",str(Path(__file__).resolve()),"--seed",str(args.seed),"--phase",args.phase,
            "--step",step,"--project-root",str(roots.project),"--data-root",str(roots.data),"--results-root",str(roots.results)]
        if args.hpc:command.append("--hpc")
        if args.retry:command.append("--retry")
        result=subprocess.run(command)
        if result.returncode:return result.returncode
    if args.phase=="evaluate":
        from src.universal_threshold_artifacts import resolve_inputs
        resolve_inputs(roots,seed=args.seed)
    else:
        from src.universal_replication_analysis import collect_seed
        collect_seed(roots,args.seed)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
