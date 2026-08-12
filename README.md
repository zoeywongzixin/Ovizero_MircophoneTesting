# OviZero_MircophoneTesting

This repository contains the code and tests for mosquito audio analysis in the `src/` and `tests/` directories.

## Purpose

- Provide a supporting document and codebase for mosquito sound detection and machine learning workflows.
- Keep large datasets, generated outputs, and virtual environment files out of version control.
- Enable easy sharing and collaboration via GitHub.

## What is included

- `src/` — application source code
- `tests/` — automated tests
- `requirements.txt` — Python dependencies
- `pytest.ini` — test configuration
- `.gitignore` — ignores data, environment, and output artifacts

## How to run

- Analyze audio files from `input/`:

```bash
python src/analyze.py
```

- Run the GUI application:

```bash
python src/main.py
```

- Train a model:

```bash
python src/ml/train.py
```
```

- Evaluate a trained model:

```bash
python src/ml/evaluate.py
```

## Output and artifacts

- `output/models/` — trained model files and metrics artifacts
- `output/datasets/esc50/` — downloaded ESC-50 dataset used for training and evaluation
- `output/` is ignored by default to keep large files out of version control

- `input/` is the default source for sample audio files; place `.mp3`, `.wav`, or similar audio files there when running analysis or training.

## Guidance for GitHub

1. Verify that the remote is set to GitHub:

```bash
git remote -v
```

2. Fetch and rebase the existing remote branch if needed:

```bash
git fetch origin
git pull --rebase origin main
```

3. Push your local branch:

```bash
git push -u origin main
```

## Notes

- The directories `input/`, `data/`, and `output/` are ignored by default to avoid large dataset uploads.
- If you want to include specific data files, update `.gitignore` accordingly.
