"""PostDiffIO：基于条件扩散残差后验的惯性导航网络。

核心思路：
RoNIN 骨干先给出速度点估计和高层时序特征，条件扩散细化器再对速度残差
建立后验分布。普通前向传播使用确定性残差均值完成端到端训练；需要不确定性
时可采样残差后验，并将样本均值和方差交给轨迹积分或 EKF。
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from ...RoNIN.model.model_resnet1d import BasicBlock1D, FCOutputModule, ResNet1D
from .diffusion import DiffusionSchedule


def sinusoidal_time_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    """将扩散时间步编码为连续特征。"""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class ResidualBlock(nn.Module):
    """扩散细化器使用的残差多层感知机。"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.layers(self.norm(features))


class ConditionalResidualRefiner(nn.Module):
    """根据骨干特征和扩散时间步预测残差噪声。"""

    def __init__(
        self,
        output_dim: int,
        condition_dim: int = 128,
        time_dim: int = 64,
        hidden_dim: int = 256,
        block_count: int = 4,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_projection = nn.Linear(output_dim + condition_dim + time_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(block_count)])
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        residual: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time_features = self.time_encoder(
            sinusoidal_time_embedding(timestep, self.time_dim)
        )
        features = self.input_projection(
            torch.cat([residual, time_features, condition], dim=-1)
        )
        for block in self.blocks:
            features = block(features)
        return self.output_projection(features)


class PostDiffIO(nn.Module):
    """可由项目模型工厂直接创建的端到端 PostDiffIO。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        net_config: int,
        group_sizes: Optional[List[int]] = None,
        condition_dim: int = 128,
        diffusion_steps: int = 100,
    ):
        super().__init__()
        group_sizes = group_sizes or [2, 2, 2, 2]
        self.output_dim = output_dim
        self.backbone = ResNet1D(
            input_dim,
            output_dim,
            BasicBlock1D,
            group_sizes,
            base_plane=64,
            output_block=FCOutputModule,
            kernel_size=3,
            net_config=net_config,
        )
        backbone_dim = self.backbone.planes[-1] * BasicBlock1D.expansion
        self.condition_encoder = nn.Sequential(
            nn.Linear(backbone_dim + output_dim, 256),
            nn.SiLU(),
            nn.Linear(256, condition_dim),
        )
        self.refiner = ConditionalResidualRefiner(output_dim, condition_dim=condition_dim)
        self.log_std_head = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim),
        )
        self.diffusion = DiffusionSchedule(diffusion_steps)
        self.aux_loss = None

    def encode(self, imu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """提取基础速度估计与条件特征。"""
        features = self.backbone.input_block(imu)
        features = self.backbone.residual_groups(features)
        base_velocity = self.backbone.output_block(features)
        pooled_features = features.mean(dim=-1)
        condition = self.condition_encoder(
            torch.cat([pooled_features, base_velocity], dim=-1)
        )
        return base_velocity, condition

    def forward(
        self,
        imu: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回修正后的速度均值和对数标准差。"""
        base_velocity, condition = self.encode(imu)
        timestep = torch.zeros(imu.size(0), dtype=torch.long, device=imu.device)
        initial_residual = torch.zeros_like(base_velocity)
        residual_mean = self.refiner(initial_residual, timestep, condition)
        velocity = base_velocity + residual_mean
        log_std = self.log_std_head(condition).clamp(-6.0, 3.0)
        if self.training and target is not None:
            residual = target[:, : self.output_dim] - base_velocity
            self.aux_loss = self.diffusion.training_loss(
                self.refiner,
                residual,
                condition,
            )
        else:
            self.aux_loss = None
        return velocity, log_std

    def diffusion_loss(self, imu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算残差后验的扩散训练损失，供后续专用训练策略调用。"""
        base_velocity, condition = self.encode(imu)
        residual = target[:, : self.output_dim] - base_velocity
        return self.diffusion.training_loss(self.refiner, residual, condition)

    @torch.no_grad()
    def sample_posterior(
        self,
        imu: torch.Tensor,
        sample_count: int = 16,
        sampling_steps: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样速度后验并返回样本、均值和对角方差。"""
        base_velocity, condition = self.encode(imu)
        residual_samples = self.diffusion.sample(
            self.refiner,
            condition,
            self.output_dim,
            sample_count=sample_count,
            sampling_steps=sampling_steps,
        )
        samples = base_velocity.unsqueeze(1) + residual_samples
        mean = samples.mean(dim=1)
        variance = samples.var(dim=1, unbiased=sample_count > 1).clamp(min=1e-6)
        return samples, mean, variance
