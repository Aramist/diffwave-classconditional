# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from math import sqrt
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ConvTranspose2d, Linear


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(max_steps), persistent=False
        )
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = swish(x)
        x = self.projection2(x)
        x = swish(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ClassEmbeddingUpsampler(nn.Module):
    def __init__(self, d_class):
        super().__init__()
        self.proj1 = Linear(d_class, 512)
        self.proj2 = Linear(512, d_class)

    def forward(self, x):
        x = self.proj1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.proj2(x)
        x = F.leaky_relu(x, 0.4)
        return x  # [B, d_cond]


class ResidualBlock(nn.Module):
    def __init__(
        self,
        d_cond,
        residual_channels,
        dilation,
        conditioning: Literal["none", "spec", "class"] = "nono",
    ):
        """
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        """
        super().__init__()
        ksize = 11
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            11,
            padding=5 * dilation,
            dilation=dilation,
        )
        self.diffusion_projection = Linear(512, residual_channels)

        if conditioning == "spec":  # conditional model
            self.conditioner_projection = Conv1d(d_cond, 2 * residual_channels, 1)
        elif conditioning == "class":  # conditional model
            self.conditioner_projection = nn.Sequential(
                Linear(d_cond, 2 * residual_channels),
                Unsqueeze(-1),
            )
        else:  # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None:  # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))

        self.spectrogram_upsampler = None
        self.class_upsampler = None
        if self.params.cond_type == "spec":
            self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)
        elif self.params.cond_type == "class":
            # TODO: technically there is no reason we can't have both
            self.class_upsampler = ClassEmbeddingUpsampler(
                params.class_embedding_dimension
            )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_cond=(
                        params.n_mels
                        if self.spectrogram_upsampler
                        else params.class_embedding_dimension
                    ),
                    residual_channels=params.residual_channels,
                    dilation=2 ** (i % params.dilation_cycle_length),
                    conditioning=params.cond_type,
                )
                for i in range(params.residual_layers)
            ]
        )
        self.skip_projection = Conv1d(
            params.residual_channels, params.residual_channels, 1
        )
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, conditioner=None):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler:  # use conditional model
            conditioner = self.spectrogram_upsampler(conditioner)
        elif self.class_upsampler:
            conditioner = self.class_upsampler(conditioner)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, conditioner)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
