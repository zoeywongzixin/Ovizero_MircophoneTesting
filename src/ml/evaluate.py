from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import WindowedFeatureDataset, build_sample_manifest, split_by_group
from .features import FeatureConfig
from .model import create_model, predict_probability


def compute_binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)

    true_positive = int(np.sum((predictions == 1) & (labels == 1)))
    false_positive = int(np.sum((predictions == 1) & (labels == 0)))
    true_negative = int(np.sum((predictions == 0) & (labels == 0)))
    false_negative = int(np.sum((predictions == 0) & (labels == 1)))

    accuracy = _safe_divide(true_positive + true_negative, labels.size)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": compute_roc_auc(probabilities, labels),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def compute_roc_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return 0.0
    wins = 0.0
    total = float(positives.size * negatives.size)
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return wins / total


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for features, batch_labels in dataloader:
        features = features.to(device)
        probabilities.append(predict_probability(model, features).cpu().numpy())
        labels.append(batch_labels.reshape(-1).cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def evaluate_artifact(
    artifact_path: str | Path = "output/models/mosquito_panns.pt",
    input_dir: str | Path = "input",
    esc50_dir: str | Path = "output/datasets/esc50",
    max_windows_per_class: int | None = 600,
) -> dict:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    feature_config = FeatureConfig.from_dict(artifact["feature_config"])
    model = create_model(artifact.get("model_type", "crnn"), artifact["model_config"])
    model.load_state_dict(artifact["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    samples = build_sample_manifest(input_dir=input_dir, esc50_dir=esc50_dir)
    _, validation = split_by_group(samples, seed=int(artifact.get("seed", 7)))
    dataset = WindowedFeatureDataset(
        validation,
        feature_config,
        max_windows_per_class=max_windows_per_class,
        seed=int(artifact.get("seed", 7)),
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    probabilities, labels = collect_predictions(model, dataloader, device)
    return compute_binary_metrics(probabilities, labels, float(artifact["threshold"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained mosquito sound model.")
    parser.add_argument("--model", type=Path, default=Path("output/models/mosquito_panns.pt"))
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--esc50-dir", type=Path, default=Path("output/datasets/esc50"))
    parser.add_argument("--max-windows-per-class", type=int, default=600)
    args = parser.parse_args()

    metrics = evaluate_artifact(
        artifact_path=args.model,
        input_dir=args.input_dir,
        esc50_dir=args.esc50_dir,
        max_windows_per_class=args.max_windows_per_class,
    )
    print(json.dumps(metrics, indent=2))


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


if __name__ == "__main__":
    main()
