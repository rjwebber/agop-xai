"""Compact PyTorch forecasters for the revised Zebiak--Cane experiments.

The revised public model input is flat: the spatial fields come first in
channel-major order, followed by two scalar annual sine/cosine features. Thus
the four-field input has exactly ``4 * 20 * 27 + 2 = 2,162`` coordinates.
Models reshape only the spatial prefix internally.
"""

from __future__ import annotations

import math
from typing import Literal

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - dependency guidance
    raise ModuleNotFoundError(
        "PyTorch is required for forecasting experiments. Install the project "
        "with `python -m pip install -e .`."
    ) from error

ArchitectureName = Literal["mlp", "cnn", "vit"]
ARCHITECTURES: tuple[ArchitectureName, ...] = ("mlp", "cnn", "vit")
SPATIAL_SHAPE = (20, 27)
DEFAULT_PHASE_FEATURES = 0


def _split_inputs(
    inputs: torch.Tensor,
    *,
    spatial_channels: int,
    phase_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate a flat canonical input into spatial fields and phase scalars."""

    spatial_features = spatial_channels * math.prod(SPATIAL_SHAPE)
    expected_features = spatial_features + phase_features
    if inputs.ndim == 2 and inputs.shape[1] == expected_features:
        fields = inputs[:, :spatial_features].reshape(
            inputs.shape[0], spatial_channels, *SPATIAL_SHAPE
        )
        phase = inputs[:, spatial_features:]
        return fields, phase
    # Backward-compatible path for legacy, phase-free processed data.
    if (
        phase_features == 0
        and inputs.ndim == 4
        and tuple(inputs.shape[1:]) == (spatial_channels, *SPATIAL_SHAPE)
    ):
        return inputs, inputs.new_empty((inputs.shape[0], 0))
    raise ValueError(
        f"Expected flat model inputs with shape (batch, {expected_features}); "
        f"found {tuple(inputs.shape)}."
    )


def _mlp_width(spatial_channels: int, phase_features: int) -> int:
    """Choose a simple width keeping supported channel variants near 220k."""

    supported = {4: 100, 5: 80, 10: 40}
    if spatial_channels in supported:
        return supported[spatial_channels]
    input_features = spatial_channels * math.prod(SPATIAL_SHAPE) + phase_features
    candidates = range(16, 129)

    def parameter_distance(width: int) -> tuple[int, int]:
        count = input_features * width + 2 * width**2 + 4 * width + 1
        return abs(count - 220_000), width

    return min(candidates, key=parameter_distance)


class MLPForecaster(nn.Module):
    """Three-layer MLP with channel-dependent width and about 220k parameters."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        *,
        phase_features: int = DEFAULT_PHASE_FEATURES,
    ) -> None:
        super().__init__()
        channels, n_latitudes, n_longitudes = input_shape
        if (n_latitudes, n_longitudes) != SPATIAL_SHAPE:
            raise ValueError("The forecasting models expect a 20 x 27 spatial grid.")
        if phase_features < 0:
            raise ValueError("phase_features must be nonnegative.")
        self.phase_features = phase_features
        self.spatial_channels = channels
        self.hidden_width = _mlp_width(self.spatial_channels, phase_features)
        input_features = self.spatial_channels * n_latitudes * n_longitudes
        input_features += phase_features
        self.network = nn.Sequential(
            nn.Linear(input_features, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fields, phase = _split_inputs(
            inputs,
            spatial_channels=self.spatial_channels,
            phase_features=self.phase_features,
        )
        flattened = torch.cat((fields.flatten(start_dim=1), phase), dim=1)
        return self.network(flattened).squeeze(-1)


class CNNForecaster(nn.Module):
    """Four-convolution CNN with exact-cover pooling on the 20 x 27 grid."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        *,
        phase_features: int = DEFAULT_PHASE_FEATURES,
    ) -> None:
        super().__init__()
        channels, n_latitudes, n_longitudes = input_shape
        if (n_latitudes, n_longitudes) != SPATIAL_SHAPE:
            raise ValueError("The forecasting models expect a 20 x 27 spatial grid.")
        if phase_features < 0:
            raise ValueError("phase_features must be nonnegative.")
        self.phase_features = phase_features
        self.spatial_channels = channels
        convolution_channels = 50
        self.features = nn.Sequential(
            nn.Conv2d(
                self.spatial_channels, convolution_channels, 3, padding="same"
            ),
            nn.ReLU(),
            nn.Conv2d(convolution_channels, convolution_channels, 3, padding="same"),
            nn.ReLU(),
            # 20 x 27 -> 10 x 9, with no discarded row or column.
            nn.AvgPool2d(kernel_size=(2, 3), stride=(2, 3)),
            nn.Conv2d(convolution_channels, convolution_channels, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(convolution_channels, convolution_channels, 3, padding="same"),
            nn.ReLU(),
            # 10 x 9 -> 5 x 9; longitude is left untouched here.
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        regressor_input = convolution_channels * 5 * 9 + phase_features
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fields, phase = _split_inputs(
            inputs,
            spatial_channels=self.spatial_channels,
            phase_features=self.phase_features,
        )
        features = self.features(fields).flatten(start_dim=1)
        return self.regressor(torch.cat((features, phase), dim=1)).squeeze(-1)


class TransformerBlock(nn.Module):
    """Pre-normalized residual self-attention and feed-forward layers."""

    def __init__(self, embedding_dimension: int, n_heads: int, expansion: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dimension)
        self.attention = nn.MultiheadAttention(
            embedding_dimension,
            n_heads,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        hidden_dimension = embedding_dimension * expansion
        self.feed_forward_norm = nn.LayerNorm(embedding_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, embedding_dimension),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(inputs)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        outputs = inputs + attended
        return outputs + self.feed_forward(self.feed_forward_norm(outputs))


class ViTForecaster(nn.Module):
    """Exact-cover 2 x 3 patch ViT with a compact 64-dimensional embedding."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        *,
        phase_features: int = DEFAULT_PHASE_FEATURES,
    ) -> None:
        super().__init__()
        channels, n_latitudes, n_longitudes = input_shape
        if (n_latitudes, n_longitudes) != SPATIAL_SHAPE:
            raise ValueError("The forecasting models expect a 20 x 27 spatial grid.")
        if phase_features < 0:
            raise ValueError("phase_features must be nonnegative.")
        self.phase_features = phase_features
        self.spatial_channels = channels
        self.patch_height = 2
        self.patch_width = 3
        if (
            n_latitudes % self.patch_height != 0
            or n_longitudes % self.patch_width != 0
        ):
            raise ValueError("The 2 x 3 ViT patches must tile the grid exactly.")
        self.n_patch_rows = n_latitudes // self.patch_height
        self.n_patch_columns = n_longitudes // self.patch_width
        self.n_patches = self.n_patch_rows * self.n_patch_columns
        patch_features = self.spatial_channels * self.patch_height * self.patch_width
        embedding_dimension = 64
        self.patch_embedding = nn.Linear(patch_features, embedding_dimension)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.n_patches, embedding_dimension)
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_dimension, n_heads=4, expansion=4)
            for _ in range(4)
        )
        self.output_norm = nn.LayerNorm(embedding_dimension)
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dimension + phase_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def patches(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = inputs.shape
        patches = inputs.unfold(2, self.patch_height, self.patch_height).unfold(
            3, self.patch_width, self.patch_width
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        return patches.reshape(
            batch_size,
            self.n_patches,
            channels * self.patch_height * self.patch_width,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fields, phase = _split_inputs(
            inputs,
            spatial_channels=self.spatial_channels,
            phase_features=self.phase_features,
        )
        embedded = self.patch_embedding(self.patches(fields))
        embedded = embedded + self.position_embedding
        for block in self.blocks:
            embedded = block(embedded)
        pooled = self.output_norm(embedded).mean(dim=1)
        return self.regressor(torch.cat((pooled, phase), dim=1)).squeeze(-1)


def build_model(
    architecture: ArchitectureName,
    input_shape: tuple[int, int, int],
    *,
    phase_features: int = DEFAULT_PHASE_FEATURES,
) -> nn.Module:
    if architecture == "mlp":
        return MLPForecaster(input_shape, phase_features=phase_features)
    if architecture == "cnn":
        return CNNForecaster(input_shape, phase_features=phase_features)
    if architecture == "vit":
        return ViTForecaster(input_shape, phase_features=phase_features)
    raise ValueError(f"Unknown architecture: {architecture!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
