"""驗證 112-task config；--write 只建立不存在的 manifest，不覆寫原檔。"""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.seed_replication import FIELDS, expected_rows, manifest


def main():
    parser = argparse.ArgumentParser(description="驗證 seeds=2,3 的獨立 TranAD manifest")
    parser.add_argument("--write", action="store_true", help="只建立缺少的 CSV；已存在即拒絕")
    args = parser.parse_args()
    if args.write:
        rows = expected_rows(ROOT)
        with (ROOT / "configs/seed_replication_runs.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    rows = manifest(ROOT)
    print(f"manifest 檢查完成：{len(rows)} tasks；seed=2,3 各 56；原 seed=1 不重跑")


if __name__ == "__main__":
    main()
