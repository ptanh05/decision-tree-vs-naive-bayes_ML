"""Chạy notebook tại chỗ bằng nbclient (thay cho `jupyter nbconvert`, vốn bị
Windows Application Control chặn trên máy này).

Cách dùng (từ thư mục gốc dự án):  python scripts/run_notebook.py notebooks/02_Preprocessing.ipynb
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.stdout.reconfigure(encoding="utf-8")

for path in map(Path, sys.argv[1:]):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    errors = sum(o.output_type == "error" for c in cells for o in c.outputs)
    print(f"{path}: {len(cells)} ô code đã chạy, {errors} lỗi")
    if errors:
        sys.exit(1)
