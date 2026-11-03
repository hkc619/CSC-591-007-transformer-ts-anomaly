"""集中解析外部路徑；解析本身不建立資料夾或讀取實驗資料。"""
from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class Roots:
    project: Path
    data: Path
    results: Path


def overlaps(a,b):
    return a==b or a.is_relative_to(b) or b.is_relative_to(a)


def resolve_roots(args=None):
    if args is None:
        from src.parser import args
    actual=Path(__file__).resolve().parents[1]
    supplied={name:getattr(args,name.lower(),None) or os.environ.get(name) for name in ['PROJECT_ROOT','DATA_ROOT','RESULTS_ROOT']}
    if args.hpc and not all(supplied.values()):
        missing=[name for name,value in supplied.items() if not value]
        raise ValueError('HPC 模式缺少明確路徑：'+', '.join(missing))
    project=Path(supplied['PROJECT_ROOT'] or actual).expanduser().resolve()
    data=Path(supplied['DATA_ROOT'] or project/'data/SMD').expanduser().resolve()
    results=Path(supplied['RESULTS_ROOT'] or project.parent/(project.name+'-results')).expanduser().resolve()
    if project!=actual:
        raise ValueError('PROJECT_ROOT 必須指向目前執行的 code checkout，不能默默改用另一份程式')
    if overlaps(project,results) or overlaps(data,results):
        raise ValueError('RESULTS_ROOT 必須與 code、dataset 彼此分離，不可互相包含')
    if args.hpc and overlaps(project,data):
        raise ValueError('HPC 模式的 DATA_ROOT 必須位於 code checkout 外')
    return Roots(project,data,results)


def scalar_label_path(data,machine):
    # 本機 validated copy 使用 labels，官方 SMD 也可能命名為 test_label。
    candidates=[data/folder/(machine+'.txt') for folder in ['labels','test_label']]
    present=[path for path in candidates if path.is_file()]
    if len(present)==2 and present[0].read_bytes()!=present[1].read_bytes():
        raise ValueError('labels 與 test_label 同時存在但內容不一致，拒絕靜默選擇')
    for path in present:
        return path
    raise FileNotFoundError(f'找不到 {machine} 的 labels 或 test_label')
