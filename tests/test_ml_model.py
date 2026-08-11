import torch

from src.ml.model import (
    MosquitoCRNN,
    PANNsMosquitoClassifier,
    create_model,
    load_panns_pretrained_weights,
    predict_probability,
)


def test_crnn_forward_returns_one_logit_per_window():
    model = MosquitoCRNN(n_mels=40, lstm_hidden_size=16)
    features = torch.zeros(3, 1, 40, 20)

    logits = model(features)

    assert logits.shape == (3, 1)


def test_predict_probability_returns_values_between_zero_and_one():
    model = MosquitoCRNN(n_mels=40, lstm_hidden_size=16)
    features = torch.zeros(2, 1, 40, 20)

    probabilities = predict_probability(model, features)

    assert probabilities.shape == (2,)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_panns_cnn10_forward_returns_one_logit_per_window():
    model = PANNsMosquitoClassifier(backbone="cnn10", n_mels=64)
    features = torch.zeros(2, 1, 64, 80)

    logits = model(features)

    assert logits.shape == (2, 1)


def test_create_model_restores_panns_cnn14_from_artifact_config():
    original = PANNsMosquitoClassifier(backbone="cnn14", n_mels=64, dropout=0.1)

    restored = create_model("panns_cnn14", original.config())

    assert isinstance(restored, PANNsMosquitoClassifier)
    assert restored.config()["backbone"] == "cnn14"


def test_load_panns_pretrained_weights_skips_unmatched_audioset_head(tmp_path):
    model = PANNsMosquitoClassifier(backbone="cnn10", n_mels=64)
    replacement = torch.full_like(model.state_dict()["conv_block1.conv1.weight"], 0.25)
    checkpoint_path = tmp_path / "panns_cnn10.pth"
    torch.save(
        {
            "model": {
                "module.conv_block1.conv1.weight": replacement,
                "fc_audioset.weight": torch.zeros(527, 512),
            }
        },
        checkpoint_path,
    )

    report = load_panns_pretrained_weights(model, checkpoint_path)

    assert report["loaded_tensors"] == 1
    assert torch.allclose(model.state_dict()["conv_block1.conv1.weight"], replacement)
