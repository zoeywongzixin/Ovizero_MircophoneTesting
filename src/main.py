from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.ui import run_app
else:
    from .ui import run_app


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
