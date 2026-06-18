"""PostDiffIO 的低维条件扩散工具。

核心思路：
对基础网络的速度残差建立条件扩散分布。训练时对真实残差加噪并预测噪声；
推理时通过 DDIM 采样得到多组残差，样本均值用于修正速度，样本方差表示不确定性。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    """生成余弦噪声调度。"""
    timeline = torch.arange(steps + 1, dtype=torch.float64) / steps
    alpha_bar = torch.cos((timeline + offset) / (1 + offset) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999).float()


class DiffusionSchedule(nn.Module):
    """可随模型设备迁移的扩散调度器。"""

    def __init__(self, steps: int = 100):
        super().__init__()
        self.steps = steps
        betas = cosine_beta_schedule(steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())

    def training_loss(
        self,
        refiner: nn.Module,
        residual: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """计算标准噪声预测损失。"""
        batch_size = residual.size(0)
        timestep = torch.randint(0, self.steps, (batch_size,), device=residual.device)
        noise = torch.randn_like(residual)
        scale = self.sqrt_alpha_bar[timestep].unsqueeze(1)
        noise_scale = self.sqrt_one_minus_alpha_bar[timestep].unsqueeze(1)
        noisy_residual = scale * residual + noise_scale * noise
        predicted_noise = refiner(noisy_residual, timestep, condition)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        refiner: nn.Module,
        condition: torch.Tensor,
        output_dim: int,
        sample_count: int = 16,
        sampling_steps: int = 10,
    ) -> torch.Tensor:
        """使用确定性 DDIM 采样残差后验。"""
        batch_size = condition.size(0)
        repeated_condition = condition.repeat_interleave(sample_count, dim=0)
        total = repeated_condition.size(0)
        residual = torch.randn(total, output_dim, device=condition.device)
        timeline = torch.linspace(
            self.steps - 1,
            0,
            sampling_steps + 1,
            device=condition.device,
        ).long()

        for index in range(sampling_steps):
            current = timeline[index].expand(total)
            following = timeline[index + 1].expand(total)
            current_alpha = self.alpha_bar[current].unsqueeze(1)
            following_alpha = self.alpha_bar[following].unsqueeze(1)
            predicted_noise = refiner(residual, current, repeated_condition)
            clean = (
                residual - (1.0 - current_alpha).sqrt() * predicted_noise
            ) / current_alpha.sqrt().clamp(min=1e-8)
            clean = clean.clamp(-3.0, 3.0)
            residual = (
                following_alpha.sqrt() * clean
                + (1.0 - following_alpha).clamp(min=0).sqrt() * predicted_noise
            )

        return residual.view(batch_size, sample_count, output_dim)

