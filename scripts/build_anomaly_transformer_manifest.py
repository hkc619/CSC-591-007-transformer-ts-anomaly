"""核對固定 6×2 AT manifest；--write 只建立尚不存在檔案。"""
import argparse
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.anomaly_transformer.protocol import configuration, expected_rows, validate_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/anomaly_transformer_runs.csv")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config, _ = configuration(ROOT)
    if args.write:
        rows = expected_rows(config)
        with args.manifest.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    rows = validate_manifest(args.manifest, config)
    print(f"AT manifest：{len(rows)} tasks，6 fixed targets × 2 settings")


if __name__ == "__main__":
    main()
