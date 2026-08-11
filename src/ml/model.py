from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


MODEL_TYPE_CRNN = "crnn"
MODEL_TYPE_PANNS_CNN10 = "panns_cnn10"
MODEL_TYPE_PANNS_CNN14 = "panns_cnn14"
SUPPORTED_MODEL_TYPES = (MODEL_TYPE_CRNN, MODEL_TYPE_PANNS_CNN10, MODEL_TYPE_PANNS_CNN14)


class MosquitoCRNN(nn.Module):
    def __init__(
        self,
        n_mels: int = 64,
        lstm_hidden_size: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout = dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("expected input shape [batch, channels, n_mels, frames]")
        features = self.cnn(x)
        features = features.mean(dim=2)
        sequence = features.transpose(1, 2)
        sequence_out, _ = self.lstm(sequence)
        pooled = sequence_out.mean(dim=1)
        return self.classifier(pooled)

    def config(self) -> dict:
        return {
            "n_mels": self.n_mels,
            "lstm_hidden_size": self.lstm_hidden_size,
            "dropout": self.dropout,
        }


class PANNsConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        pool_size: tuple[int, int] = (2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "avg":
            return F.avg_pool2d(x, kernel_size=pool_size)
        if pool_type == "max":
            return F.max_pool2d(x, kernel_size=pool_size)
        raise ValueError("pool_type must be 'avg' or 'max'")

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)
        nn.init.constant_(self.bn2.weight, 1.0)
        nn.init.constant_(self.bn2.bias, 0.0)


class PANNsMosquitoClassifier(nn.Module):
    """PANNs-style Cnn10/Cnn14 binary classifier over log-mel windows."""

    _BACKBONE_CHANNELS = {
        "cnn10": (64, 128, 256, 512),
        "cnn14": (64, 128, 256, 512, 1024, 2048),
    }

    def __init__(
        self,
        backbone: str = "cnn10",
        n_mels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if backbone not in self._BACKBONE_CHANNELS:
            raise ValueError("backbone must be one of: cnn10, cnn14")
        self.backbone = backbone
        self.n_mels = n_mels
        self.dropout = dropout

        channels = self._BACKBONE_CHANNELS[backbone]
        self.bn0 = nn.BatchNorm2d(n_mels)
        in_channels = 1
        for index, out_channels in enumerate(channels, start=1):
            setattr(self, f"conv_block{index}", PANNsConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.block_count = len(channels)
        self.embedding_size = channels[-1]
        self.fc1 = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.classifier = nn.Linear(self.embedding_size, 1, bias=True)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("expected input shape [batch, channels, n_mels, frames]")
        if x.shape[1] != 1:
            raise ValueError("expected a single log-mel channel")

        x = self._to_panns_layout(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        for index in range(1, self.block_count + 1):
            block = getattr(self, f"conv_block{index}")
            x = block(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        max_pooled, _ = torch.max(x, dim=2)
        mean_pooled = torch.mean(x, dim=2)
        x = max_pooled + mean_pooled
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def config(self) -> dict:
        return {
            "backbone": self.backbone,
            "n_mels": self.n_mels,
            "dropout": self.dropout,
        }

    def _to_panns_layout(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] == self.n_mels:
            return x.transpose(2, 3)
        if x.shape[3] == self.n_mels:
            return x
        raise ValueError(
            f"expected one spectrogram axis to contain {self.n_mels} mel bins"
        )

    def _init_weights(self) -> None:
        nn.init.constant_(self.bn0.weight, 1.0)
        nn.init.constant_(self.bn0.bias, 0.0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)


def create_model(model_type: str | None, model_config: dict | None) -> nn.Module:
    model_type = model_type or MODEL_TYPE_CRNN
    model_config = dict(model_config or {})
    if model_type == MODEL_TYPE_CRNN:
        return MosquitoCRNN(**_filter_config(model_config, {"n_mels", "lstm_hidden_size", "dropout"}))
    if model_type in {MODEL_TYPE_PANNS_CNN10, MODEL_TYPE_PANNS_CNN14}:
        default_backbone = "cnn14" if model_type == MODEL_TYPE_PANNS_CNN14 else "cnn10"
        model_config.setdefault("backbone", default_backbone)
        return PANNsMosquitoClassifier(
            **_filter_config(model_config, {"backbone", "n_mels", "dropout"})
        )
    raise ValueError(f"unsupported model_type: {model_type}")


def load_panns_pretrained_weights(
    model: PANNsMosquitoClassifier,
    checkpoint_path: str | Path,
) -> dict[str, int]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    source_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    target_state = model.state_dict()
    loadable_state = {}
    skipped_tensors = 0
    for key, value in source_state.items():
        clean_key = key.removeprefix("module.")
        if clean_key in target_state and target_state[clean_key].shape == value.shape:
            loadable_state[clean_key] = value
        else:
            skipped_tensors += 1
    model.load_state_dict({**target_state, **loadable_state})
    return {
        "loaded_tensors": len(loadable_state),
        "skipped_tensors": skipped_tensors,
    }


@torch.no_grad()
def predict_probability(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    logits = model(features)
    probabilities = torch.sigmoid(logits).reshape(-1)
    if was_training:
        model.train()
    return probabilities


def _filter_config(config: dict, allowed_keys: set[str]) -> dict:
    return {key: value for key, value in config.items() if key in allowed_keys}
