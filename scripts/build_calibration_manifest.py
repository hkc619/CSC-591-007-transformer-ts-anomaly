"""驗證既有 84-row calibration manifest；只有明確 --write 才建立新檔，不覆寫。"""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration_protocol import FIELDS, load_configuration, manifest_rows, validate_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/calibration_runs.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--output", type=Path, help="建立新的 manifest；已存在則僅驗證，絕不覆寫")
    parser.add_argument("--write", action="store_true", help="建立不存在的 manifest；不覆寫既有檔案")
    parser.add_argument("--hpc", action="store_true", help="config/manifest 檢查不載入模型或資料")
    args = parser.parse_args()
    _, machines = load_configuration(ROOT)
    expected_rows = manifest_rows(machines, args.seeds)
    path = args.output or args.manifest
    if not path.is_absolute():
        path = ROOT / path
    if args.write or (args.output is not None and not path.exists()):
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(expected_rows)
    rows = validate_manifest(path, machines, args.seeds)
    print(f"Manifest/config validation：{len(rows)} tasks = 28 targets × 3 fractions × seeds={args.seeds}")
    print("僅檔案驗證；parameter / score / AUC / POT runtime checks 仍需 Hazel 執行")


if __name__ == "__main__":
    main()
