from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (
    DEFAULT_HARD_NEGATIVE_CATEGORIES,
    WindowedFeatureDataset,
    build_sample_manifest,
    ensure_esc50_dataset,
    negative_sample_multiplier,
    split_by_group,
)
from .evaluate import collect_predictions, compute_binary_metrics
from .features import FeatureConfig
from .model import (
    MODEL_TYPE_CRNN,
    MODEL_TYPE_PANNS_CNN10,
    MODEL_TYPE_PANNS_CNN14,
    PANNsMosquitoClassifier,
    create_model,
    load_panns_pretrained_weights,
)


DEFAULT_MODEL_TYPE = MODEL_TYPE_PANNS_CNN10


def choose_conservative_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    min_precision: float = 0.90,
) -> tuple[float, dict[str, float | int]]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    thresholds = sorted({0.5, *[round(float(value), 6) for value in probabilities]}, reverse=True)
    candidates = []
    fallback_candidates = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(probabilities, labels, threshold)
        if metrics["true_positive"] > 0:
            fallback_candidates.append((threshold, metrics))
        if metrics["precision"] >= min_precision and metrics["true_positive"] > 0:
            candidates.append((threshold, metrics))
    if not candidates:
        if fallback_candidates:
            return max(
                fallback_candidates,
                key=lambda item: (item[1]["precision"], item[1]["f1"], item[0]),
            )
        fallback = 0.5
        return fallback, compute_binary_metrics(probabilities, labels, fallback)
    return max(candidates, key=lambda item: (item[1]["f1"], item[1]["precision"], item[0]))


def choose_operating_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    strategy: str = "f1",
    min_precision: float = 0.90,
) -> tuple[float, dict[str, float | int]]:
    if strategy == "precision":
        return choose_conservative_threshold(probabilities, labels, min_precision=min_precision)

    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    thresholds = sorted({0.5, *[round(float(value), 6) for value in probabilities]})
    scored = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(probabilities, labels, threshold)
        if metrics["true_positive"] == 0:
            continue
        if strategy == "recall":
            score = (metrics["recall"], metrics["f1"], metrics["precision"], -threshold)
        elif strategy == "f2":
            f2 = _fbeta(metrics["precision"], metrics["recall"], beta=2.0)
            metrics = {**metrics, "f2": f2}
            score = (f2, metrics["recall"], metrics["precision"], -threshold)
        elif strategy == "f1":
            score = (metrics["f1"], metrics["recall"], metrics["precision"], -threshold)
        else:
            raise ValueError("strategy must be one of: f1, f2, recall, precision")
        scored.append((score, threshold, metrics))

    if not scored:
        fallback = 0.5
        return fallback, compute_binary_metrics(probabilities, labels, fallback)
    _, threshold, metrics = max(scored, key=lambda item: item[0])
    return threshold, metrics


def should_replace_best_model(
    best_metrics: dict[str, float | int] | None,
    candidate_metrics: dict[str, float | int],
    selection_metric: str = "f1",
) -> bool:
    if best_metrics is None:
        return True
    return float(candidate_metrics.get(selection_metric, 0.0)) > float(
        best_metrics.get(selection_metric, 0.0)
    )


def make_training_report(artifact: dict) -> dict:
    return {
        key: value
        for key, value in artifact.items()
        if key != "model_state_dict"
    }


def _fbeta(precision: float, recall: float, beta: float) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return float((1.0 + beta_squared) * precision * recall / denominator)


def train_model(
    input_dir: str | Path = "input",
    esc50_dir: str | Path = "output/datasets/esc50",
    model_dir: str | Path = "output/models",
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_windows_per_class: int = 1600,
    min_precision: float = 0.90,
    threshold_strategy: str = "f1",
    selection_metric: str = "f1",
    model_type: str = DEFAULT_MODEL_TYPE,
    pretrained_checkpoint: str | Path | None = None,
    hard_negative_categories: tuple[str, ...] = DEFAULT_HARD_NEGATIVE_CATEGORIES,
    hard_negative_multiplier: int = 3,
    seed: int = 7,
    download_esc50: bool = True,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_config = FeatureConfig()
    if download_esc50:
        ensure_esc50_dataset(esc50_dir)

    samples = build_sample_manifest(input_dir=input_dir, esc50_dir=esc50_dir)
    if not any(sample.label == 1 for sample in samples):
        raise RuntimeError("no positive mosquito samples found")
    if not any(sample.label == 0 for sample in samples):
        raise RuntimeError("no negative ESC-50 samples found")

    train_samples, validation_samples = split_by_group(samples, seed=seed)
    train_dataset = WindowedFeatureDataset(
        train_samples,
        feature_config,
        max_windows_per_class=max_windows_per_class,
        hard_negative_categories=hard_negative_categories,
        hard_negative_multiplier=hard_negative_multiplier,
        seed=seed,
    )
    validation_dataset = WindowedFeatureDataset(
        validation_samples,
        feature_config,
        max_windows_per_class=max_windows_per_class // 2,
        hard_negative_categories=hard_negative_categories,
        hard_negative_multiplier=hard_negative_multiplier,
        seed=seed,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_type, {"n_mels": feature_config.n_mels})
    pretrained_report = None
    if pretrained_checkpoint is not None:
        if not isinstance(model, PANNsMosquitoClassifier):
            raise ValueError("--pretrained-checkpoint is only supported for PANNs models")
        pretrained_report = load_panns_pretrained_weights(model, pretrained_checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    best_state_dict = None
    best_threshold = 0.5
    best_metrics = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        probabilities, labels = collect_predictions(model, validation_loader, device)
        threshold, metrics = choose_operating_threshold(
            probabilities,
            labels,
            strategy=threshold_strategy,
            min_precision=min_precision,
        )
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        if should_replace_best_model(best_metrics, metrics, selection_metric=selection_metric):
            best_state_dict = copy.deepcopy(model.cpu().state_dict())
            model.to(device)
            best_threshold = threshold
            best_metrics = metrics
            best_epoch = epoch
        print(
            f"epoch={epoch} loss={np.mean(losses):.4f} "
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} threshold={threshold:.3f}"
        )

    if best_state_dict is None or best_metrics is None:
        raise RuntimeError("training did not produce validation metrics")

    model_path = Path(model_dir) / _model_filename(model_type)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_type": model_type,
        "model_state_dict": best_state_dict,
        "model_config": model.config(),
        "feature_config": feature_config.to_dict(),
        "threshold": best_threshold,
        "metrics": best_metrics,
        "history": history,
        "best_epoch": best_epoch,
        "threshold_strategy": threshold_strategy,
        "selection_metric": selection_metric,
        "pretrained_checkpoint": str(pretrained_checkpoint) if pretrained_checkpoint else None,
        "pretrained_report": pretrained_report,
        "negative_sampling": {
            "hard_negative_categories": list(hard_negative_categories),
            "hard_negative_multiplier": int(hard_negative_multiplier),
        },
        "seed": seed,
        "manifest": {
            "total_samples": len(samples),
            "positive_samples": sum(1 for sample in samples if sample.label == 1),
            "negative_samples": sum(1 for sample in samples if sample.label == 0),
            "hard_negative_samples": sum(
                1
                for sample in samples
                if negative_sample_multiplier(
                    sample,
                    hard_negative_categories,
                    hard_negative_multiplier,
                )
                > 1
            ),
            "local_negative_samples": sum(
                1 for sample in samples if sample.source == "local_negative"
            ),
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
        },
    }
    torch.save(artifact, model_path)
    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(make_training_report(artifact), indent=2), encoding="utf-8")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PANNs or CRNN mosquito sound detector.")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--esc50-dir", type=Path, default=Path("output/datasets/esc50"))
    parser.add_argument("--model-dir", type=Path, default=Path("output/models"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-windows-per-class", type=int, default=1600)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument(
        "--threshold-strategy",
        choices=("f1", "f2", "recall", "precision"),
        default="f1",
        help="How to choose the saved decision threshold.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("f1", "f2", "recall", "precision", "accuracy", "roc_auc"),
        default="f1",
        help="Validation metric used to choose which epoch is saved.",
    )
    parser.add_argument(
        "--model-type",
        choices=(MODEL_TYPE_PANNS_CNN10, MODEL_TYPE_PANNS_CNN14, MODEL_TYPE_CRNN),
        default=DEFAULT_MODEL_TYPE,
        help="Model backend to train. PANNs Cnn10 is the default realtime backend.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Optional official PANNs Cnn10/Cnn14 checkpoint; matching backbone tensors are loaded.",
    )
    parser.add_argument(
        "--hard-negative-categories",
        default=",".join(DEFAULT_HARD_NEGATIVE_CATEGORIES),
        help="Comma-separated ESC-50 categories to oversample as hard negatives.",
    )
    parser.add_argument(
        "--hard-negative-multiplier",
        type=int,
        default=3,
        help="How many times to repeat hard-negative windows during training.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-download", action="store_true", help="Do not download ESC-50 automatically")
    args = parser.parse_args()

    model_path = train_model(
        input_dir=args.input_dir,
        esc50_dir=args.esc50_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_windows_per_class=args.max_windows_per_class,
        min_precision=args.min_precision,
        threshold_strategy=args.threshold_strategy,
        selection_metric=args.selection_metric,
        model_type=args.model_type,
        pretrained_checkpoint=args.pretrained_checkpoint,
        hard_negative_categories=_parse_category_list(args.hard_negative_categories),
        hard_negative_multiplier=args.hard_negative_multiplier,
        seed=args.seed,
        download_esc50=not args.no_download,
    )
    print(f"saved model: {model_path}")


def _model_filename(model_type: str) -> str:
    if model_type == MODEL_TYPE_CRNN:
        return "mosquito_crnn.pt"
    if model_type in {MODEL_TYPE_PANNS_CNN10, MODEL_TYPE_PANNS_CNN14}:
        return "mosquito_panns.pt"
    raise ValueError(f"unsupported model_type: {model_type}")


def _parse_category_list(value: str) -> tuple[str, ...]:
    return tuple(category.strip() for category in value.split(",") if category.strip())


if __name__ == "__main__":
    main()
