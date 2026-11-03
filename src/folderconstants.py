"""僅供原始 SMD conversion 函式相容；prepare_run 會設定 per-run cache。"""
from src.paths import resolve_roots
roots=resolve_roots()
output_folder=str(roots.results/'cache/processed')
data_folder=str(roots.data)
