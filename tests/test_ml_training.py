import inspect

import numpy as np

from src.ml.evaluate import compute_binary_metrics
from src.ml.train import (
    choose_conservative_threshold,
    choose_operating_threshold,
    make_training_report,
    should_replace_best_model,
    train_model,
)


def test_choose_conservative_threshold_prefers_high_precision():
    probabilities = np.array([0.10, 0.20, 0.55, 0.80, 0.95], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)

    threshold, metrics = choose_conservative_threshold(probabilities, labels, min_precision=0.95)

    assert threshold >= 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_choose_conservative_threshold_falls_back_to_best_available_precision():
    probabilities = np.array([0.40, 0.60, 0.90], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)

    threshold, metrics = choose_conservative_threshold(probabilities, labels, min_precision=0.95)

    assert threshold == 0.60
    assert metrics["true_positive"] == 1
    assert metrics["precision"] == 0.5


def test_choose_operating_threshold_balances_recall_and_precision_by_default():
    probabilities = np.array([0.05, 0.40, 0.60, 0.90], dtype=np.float32)
    labels = np.array([1, 1, 0, 0], dtype=np.int64)

    threshold, metrics = choose_operating_threshold(probabilities, labels, strategy="f1")

    assert threshold == 0.05
    assert metrics["recall"] == 1.0
    assert metrics["f1"] > 0.65


def test_should_replace_best_model_uses_validation_f1_not_last_epoch():
    best = {"f1": 0.40, "recall": 0.80}
    worse_late_epoch = {"f1": 0.10, "recall": 0.05}
    better_epoch = {"f1": 0.45, "recall": 0.60}

    assert not should_replace_best_model(best, worse_late_epoch, selection_metric="f1")
    assert should_replace_best_model(best, better_epoch, selection_metric="f1")


def test_train_model_defaults_prioritize_fewer_false_positives():
    signature = inspect.signature(train_model)

    assert signature.parameters["threshold_strategy"].default == "f1"
    assert signature.parameters["selection_metric"].default == "f1"


def test_train_model_defaults_to_lightweight_panns_backend():
    signature = inspect.signature(train_model)

    assert signature.parameters["model_type"].default == "panns_cnn10"


def test_train_model_exposes_hard_negative_sampling_defaults():
    signature = inspect.signature(train_model)

    categories = signature.parameters["hard_negative_categories"].default
    assert "wind" in categories
    assert "engine" in categories
    assert "chirping_birds" in categories
    assert "pouring_water" in categories
    assert signature.parameters["hard_negative_multiplier"].default == 3


def test_compute_binary_metrics_reports_confusion_matrix_values():
    probabilities = np.array([0.10, 0.70, 0.80, 0.30], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    metrics = compute_binary_metrics(probabilities, labels, threshold=0.5)

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_make_training_report_excludes_tensor_state_dict():
    artifact = {
        "model_state_dict": {"weight": object()},
        "metrics": {"precision": 0.9},
        "threshold": 0.5,
    }

    report = make_training_report(artifact)

    assert "model_state_dict" not in report
    assert report["metrics"]["precision"] == 0.9
