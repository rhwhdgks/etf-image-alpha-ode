from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SequenceCnn(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(32 * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class SequenceMultiScaleCnn(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        branch_channels = 16
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.Conv1d(branch_channels, branch_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv1d(branch_channels * 3, 48, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(48 * 4, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.cat([branch(inputs) for branch in self.branches], dim=1)
        return self.head(features).squeeze(-1)


class SequenceResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = dilation
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class SequenceDilatedCnn(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        hidden_channels = 32
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            SequenceResidualBlock(hidden_channels, dilation=1),
            SequenceResidualBlock(hidden_channels, dilation=2),
            SequenceResidualBlock(hidden_channels, dilation=4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.blocks(features)
        return self.head(features).squeeze(-1)


class ImageCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_head = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.output = nn.Linear(64, 1)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.feature_head(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.forward_features(inputs)).squeeze(-1)


class ImageResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class ImageResidualCnn(nn.Module):
    def __init__(self, base_channels: int = 16, dropout: float = 0.1):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block1 = ImageResidualBlock(c1)
        self.transition = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = ImageResidualBlock(c2)
        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(c2 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(64, 1)

    def _conv_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.block1(features)
        features = self.transition(features)
        return self.block2(features)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.feature_head(self._conv_features(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.forward_features(inputs)).squeeze(-1)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = inputs.mean(dim=2)
        scale = self.net(scale)
        return inputs * scale.unsqueeze(2)


class SequenceAttentionCnn(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        hidden = 32
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.se = SqueezeExcitation(hidden)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(hidden * 4, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.se(features)
        return self.head(features).squeeze(-1)


def build_cnn_model(model_name: str, input_shape: tuple[int, ...]) -> nn.Module:
    if model_name in {"cnn_1d_image_scale", "cnn_1d_cumulative_scale"}:
        return SequenceCnn(in_channels=input_shape[0])
    if model_name == "cnn_1d_multiscale_image_scale":
        return SequenceMultiScaleCnn(in_channels=input_shape[0])
    if model_name in {"cnn_1d_dilated_image_scale", "cnn_1d_dilated_cumulative_scale"}:
        return SequenceDilatedCnn(in_channels=input_shape[0])
    if model_name in {"cnn_1d_attention_image_scale", "cnn_1d_attention_cumulative_scale"}:
        return SequenceAttentionCnn(in_channels=input_shape[0])
    if model_name == "cnn_2d_rendered_images":
        return ImageCnn()
    if model_name == "cnn_2d_residual_images":
        return ImageResidualCnn()
    if model_name == "cnn_2d_residual_small":
        return ImageResidualCnn(base_channels=8, dropout=0.2)
    if model_name == "cnn_2d_residual_wd":
        return ImageResidualCnn()
    raise ValueError(f"unsupported CNN model name: {model_name}")


def _build_tensor_dataset(features: np.ndarray, targets: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(targets.astype(np.float32)).float(),
    )


def fit_torch_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    label_mode: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> nn.Module:
    train_loader = DataLoader(_build_tensor_dataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_build_tensor_dataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss() if label_mode == "classification" else nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    training_history = []

    for epoch_idx in range(epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_x)
                losses.append(float(criterion(predictions, batch_y).item()))

        val_loss = float(np.mean(losses)) if losses else float("inf")
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch_idx
            stale_epochs = 0
        else:
            stale_epochs += 1
        training_history.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "stale_epochs": stale_epochs,
            }
        )
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.training_history_ = training_history
    model.best_epoch_ = best_epoch + 1 if best_epoch >= 0 else None
    model.best_val_loss_ = best_val_loss
    return model.cpu()


def predict_torch_model(
    model: nn.Module,
    features: np.ndarray,
    label_mode: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    loader = DataLoader(TensorDataset(torch.from_numpy(features).float()), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            outputs.append(model(batch_x).cpu().numpy())

    raw = np.concatenate(outputs).astype(float)
    if label_mode == "classification":
        scores = 1.0 / (1.0 + np.exp(-raw))
        confidence = np.abs(scores - 0.5) * 2.0
        return scores, confidence
    return raw, None


def extract_torch_features(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if not hasattr(model, "forward_features"):
        raise ValueError(f"{model.__class__.__name__} does not expose forward_features()")

    loader = DataLoader(TensorDataset(torch.from_numpy(features).float()), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            outputs.append(model.forward_features(batch_x).cpu().numpy())
    return np.concatenate(outputs).astype(float)
