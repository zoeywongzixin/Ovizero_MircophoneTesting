import runpy
import sys
from pathlib import Path

from src.main import main


def test_main_entrypoint_is_callable():
    assert callable(main)


def test_main_file_loads_when_executed_by_path(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    script_path = src_dir / "main.py"
    search_path = [
        path
        for path in sys.path
        if path and Path(path).resolve() not in {project_root.resolve(), src_dir.resolve()}
    ]
    monkeypatch.setattr(sys, "path", [str(src_dir), *search_path])

    namespace = runpy.run_path(str(script_path), run_name="__not_main__")

    assert callable(namespace["main"])
