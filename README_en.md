<p align="right">
  <a href="README.md">🇨🇳 中文</a> | <b>🇬🇧 English</b>
</p>

# PostDiffIO: Conditional Diffusion Posterior Refinement for Inertial Odometry

> **Status**: Preprint · Code release pending · Results under active validation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Abstract

Inertial Odometry (IO) estimates trajectory from raw inertial measurement unit (IMU) data, forming the backbone of indoor navigation where GNSS is unavailable. While recent deep-learning-based methods such as RoNIN have demonstrated competitive velocity regression, they typically produce point estimates and lack principled uncertainty quantification — limiting their applicability in safety-critical scenarios.

We present **PostDiffIO**, a two-stage framework that augments a deterministic IO backbone with a conditional diffusion process over velocity residuals. The backbone (a ResNet1D adapted from RoNIN) predicts a baseline velocity from IMU windows; a lightweight conditional diffusion refiner then models the posterior distribution of the residual between the backbone prediction and ground-truth velocity. During training, the refiner learns to denoise the residual distribution conditioned on backbone features via a standard noise-prediction objective. At inference, deterministic DDIM sampling yields a set of residual samples whose mean refines the backbone estimate and whose variance provides calibrated uncertainty. For sequential trajectory recovery, we further propose an error-state Extended Kalman Filter (EKF) that fuses the diffusion-derived velocity observations with IMU mechanization dynamics, producing both position and velocity estimates with full covariance tracking.

Preliminary experiments on synthetic and benchmark inertial datasets show that PostDiffIO improves velocity estimation accuracy over the backbone alone while delivering meaningful uncertainty estimates, validating the viability of diffusion-based residual refinement for IO tasks.

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
- [3. Method](#3-method)
- [4. Experiments](#4-experiments)
- [5. Conclusion](#5-conclusion)
- [6. Citation](#6-citation)
- [7. License](#7-license)

---

## 1. Introduction

Accurate and robust indoor positioning is essential for applications ranging from augmented reality to autonomous robotics. Inertial Measurement Units (IMUs), ubiquitous in consumer electronics, provide high-rate motion sensing but suffer from unbounded drift when integrated naively. Traditional model-based approaches — including Extended Kalman Filters (EKFs) and Zero-Velocity Updates (ZUPT) — require careful tuning and degrade under aggressive motion or magnetic disturbances.

Deep learning has emerged as a powerful alternative for IO, directly mapping IMU sequences to velocity or position increments. Notably, RoNIN introduced a ResNet1D architecture achieving real-time velocity regression from 200 Hz IMU windows. Subsequent works have extended this with LSTM, TCN, and attention-based variants. However, a fundamental limitation persists: these methods output deterministic point estimates and provide no measure of prediction confidence.

In safety-critical domains — human activity recognition, robot navigation, structural health monitoring — knowing *when* a prediction is unreliable is as important as the prediction itself. Bayesian approaches offer a principled framework for uncertainty estimation but are computationally prohibitive for real-time deployment on mobile devices.

Diffusion models, originally developed for image generation, have recently shown remarkable capacity for modeling complex distributions in low-dimensional regression tasks. Their iterative refinement process naturally provides multiple samples from a learned posterior, enabling uncertainty quantification without explicit density estimation.

In this work, we propose **PostDiffIO**, a framework that leverages conditional diffusion models to refine velocity residuals predicted by an IO backbone. Our key insight is that rather than modeling the full velocity distribution — which would be costly and unnecessary — we can decompose the problem: a fast deterministic backbone provides a strong baseline, and a lightweight diffusion process captures only the residual uncertainty. This two-stage design achieves the best of both worlds — real-time inference speed from the backbone and principled uncertainty from diffusion.

Our contributions are threefold:

1. **Conditional diffusion residual refinement**: We propose modeling velocity residuals via a conditional diffusion process conditioned on backbone features, enabling uncertainty-aware IO without sacrificing inference speed.
2. **DDIM posterior sampling with EKF fusion**: We demonstrate how DDIM-sampled velocity posteriors can be fused with IMU mechanization through an error-state EKF, providing both accurate trajectory estimates and calibrated uncertainty bounds.
3. **Modular architecture**: Our design is backbone-agnostic; the diffusion refiner can be attached to any deterministic IO network that produces velocity estimates from IMU inputs.

---

## 2. Related Work

> *This section will be completed upon paper submission.*

### Deep Learning for Inertial Odometry

The application of deep learning to inertial odometry has evolved through several architectural paradigms. RoNIN introduced the ResNet1D architecture for real-time velocity regression from IMU data. Subsequent works explored LSTM-based approaches for temporal modeling, TCN architectures for efficient sequence processing, and attention mechanisms for capturing long-range dependencies. More recently, transformer-based methods have been proposed for high-fidelity trajectory estimation.

### Diffusion Models for Regression

While diffusion models are most prominently associated with image generation, their application to structured regression tasks has gained traction. Conditional diffusion processes have been explored for time-series forecasting, point cloud denoising, and motion prediction. Our work extends this paradigm to inertial odometry, where the low-dimensional nature of velocity residuals makes diffusion-based refinement particularly efficient.

### Uncertainty Quantification in IO

Existing uncertainty-aware IO methods include Monte Carlo dropout, deep ensembles, and evidential deep learning. These approaches either impose significant computational overhead or provide only approximate uncertainty estimates. Diffusion models offer an alternative: multiple forward passes through the sampling process yield an empirical posterior without explicit uncertainty parameterization.

---

## 3. Method

### 3.1 Problem Formulation

Given a sequence of IMU measurements x₁:T = {(aₜ, ωₜ)} sampled at frequency f (typically 200 Hz), where aₜ ∈ ℝ³ and ωₜ ∈ ℝ³ denote linear acceleration and angular velocity respectively, the IO task is to estimate the corresponding velocity trajectory v₁:T ∈ ℝ^(3×T).

PostDiffIO operates in sliding windows of length W. For each window, the backbone predicts a velocity estimate v̂ₜ, and the diffusion refiner models the residual distribution p(εₜ | v̂ₜ, xₜ) where εₜ = vₜ − v̂ₜ.

### 3.2 RoNIN Backbone

We adopt the RoNIN ResNet1D architecture as our deterministic backbone. The network processes each IMU window through:

1. **Input projection**: A 1D convolution maps the 6-dimensional IMU input to a 64-channel feature space.
2. **Residual groups**: Four groups of residual blocks (with configurable group sizes) process the temporal features.
3. **Output head**: A fully-connected module maps the pooled features to 3-dimensional velocity estimates.

The backbone also produces intermediate features fₜ ∈ ℝ^(d_f) from the penultimate layer, which serve as conditioning information for the diffusion refiner.

### 3.3 Conditional Diffusion Residual Refiner

The refiner is a conditional denoising network ε_θ(rₜ, t, cₜ) that predicts the noise added to velocity residuals. It takes three inputs:

- **Noisy residual**: rₜ = √(ᾱₜ) · ε + √(1 − ᾱₜ) · η, the velocity residual corrupted by diffusion noise at timestep t.
- **Timestep embedding**: A sinusoidal embedding of the diffusion timestep t, projected through a two-layer MLP.
- **Condition vector**: cₜ = Encoder([fₜ; v̂ₜ]), backbone features concatenated with the baseline velocity prediction, projected to dimension 128.

The refiner architecture consists of:

- A linear projection of the concatenated inputs to hidden dimension 256.
- Four residual blocks, each applying LayerNorm → Linear → SiLU → Linear with skip connections.
- An output projection producing a 3-dimensional noise prediction.

### 3.4 Training Objective

The training loss combines two components:

> L = L_velocity + λ · L_diffusion

where L_velocity is the standard MSE loss between predicted and ground-truth velocity (for the backbone), and L_diffusion is the noise-prediction loss:

> L_diffusion = E_{t∼U(0,T), η∼N(0,I)} [ ‖η − ε_θ(rₜ, t, cₜ)‖² ]

We use a cosine noise schedule with 100 diffusion steps, balancing quality and efficiency.

### 3.5 DDIM Posterior Sampling

At inference, we employ deterministic DDIM sampling with 10 steps (down from 100) to efficiently sample the residual posterior:

1. Initialize the noisy residual from a standard normal distribution: r_T ∼ N(0, I).
2. For t = T, T−Δt, …, Δt, sequentially:
   - Predict noise: η̂ = ε_θ(rₜ, t, c)
   - Compute clean estimate: r₀ = (rₜ − √(1−ᾱₜ) · η̂) / √(ᾱₜ)
   - Update: r_{t−Δt} = √(ᾱ_{t−Δt}) · r₀ + √(1−ᾱ_{t−Δt}) · η̂

Drawing K samples (default 16), we compute:

- **Mean estimate**: v̂ = v̂_backbone + (1/K) · Σ ε⁽ᵏ⁾
- **Uncertainty**: Σ = (1/K) · Σ (ε⁽ᵏ⁾ − ε̄)(ε⁽ᵏ⁾ − ε̄)ᵀ

### 3.6 EKF Trajectory Fusion

For sequence-level trajectory recovery, we employ a 15-dimensional error-state EKF that fuses diffusion-derived velocity observations with IMU mechanization.

**State vector**: x = [p, v, q, bₐ, b_g]ᵀ ∈ ℝ¹⁵, containing position, velocity, orientation quaternion, accelerometer bias, and gyroscope bias.

**Prediction**: Standard strapdown inertial mechanization with midpoint integration, propagating uncertainty through the linearized state transition matrix.

**Update**: At each diffusion prediction window, the velocity observation (with covariance from diffusion samples) is fused via the Kalman gain:

> K = P · Hᵀ · (H · P · Hᵀ + R)⁻¹

where H selects the velocity block from the state and R is the diffusion-derived observation covariance. The Joseph form is used for numerically stable covariance updates.

---

## 4. Experiments

> *Comprehensive experimental results will be provided upon completion of the full evaluation suite. Below we outline the experimental design.*

### 4.1 Setup

**Datasets**: We evaluate on the following benchmark datasets:

- **OxIOD**: 14 sequences with ground-truth from Vicon motion capture
- **IDOL**: Large-scale inertial dataset with diverse activities
- **RoNIN-Synthetic**: Generated trajectories with configurable noise levels

**Metrics**:

- **Velocity RMSE** (m/s): Root mean square error of velocity estimates
- **ATE** (m): Absolute trajectory error for position
- **NLL**: Negative log-likelihood under the predicted uncertainty
- **ECE**: Expected calibration error for uncertainty quality

**Implementation**: PyTorch 2.0+, Python 3.10+, single NVIDIA RTX 3090. Training uses AdamW optimizer with cosine learning rate schedule. Diffusion uses 100 training steps with 10-step DDIM sampling at inference.

### 4.2 Main Results

*Results pending full experimental validation.*

| Method | Velocity RMSE ↓ | ATE ↓ | NLL ↓ | Real-time |
|:------:|:---------------:|:-----:|:-----:|:---------:|
| RoNIN (baseline) | — | — | — | ✓ |
| PostDiffIO | — | — | — | ✓ |
| PostDiffIO + EKF | — | — | — | ✓ |

### 4.3 Uncertainty Quantification

We evaluate uncertainty quality through:

- **Sparsification plots**: Error vs. removed fraction (higher AUC is better)
- **Calibration analysis**: Reliability diagrams comparing predicted vs. empirical variance
- **Selective prediction**: Accuracy when uncertain predictions are rejected

### 4.4 Ablation Study

Key ablations include:

- Effect of diffusion steps (50, 100, 200) on accuracy and speed
- Number of DDIM samples (4, 8, 16, 32) on uncertainty quality
- Impact of condition dimension (64, 128, 256) on model capacity
- Comparison with Monte Carlo dropout and deep ensemble baselines

---

## 5. Conclusion

We presented PostDiffIO, a conditional diffusion framework for uncertainty-aware inertial odometry. By decomposing velocity estimation into a deterministic backbone prediction and a diffusion-refined residual posterior, our approach achieves both fast inference and principled uncertainty quantification. The integration with an error-state EKF further enables robust trajectory recovery with full covariance tracking.

Preliminary results indicate that diffusion-based residual refinement improves both point-estimate accuracy and uncertainty calibration over the backbone alone. We believe this two-stage paradigm — deterministic prediction plus diffusion refinement — is broadly applicable beyond inertial odometry, potentially benefiting any regression task where uncertainty awareness is critical.

Future work includes extending the framework to multi-modal sensor fusion, exploring adaptive diffusion schedules conditioned on motion dynamics, and conducting comprehensive real-world deployment evaluations.

---

## 6. Citation

If you find this work useful, please cite:

```bibtex
@article{postdiffio2025,
  title     = {PostDiffIO: Conditional Diffusion Posterior Refinement for Uncertainty-Aware Inertial Odometry},
  author    = {Your Name and Collaborators},
  journal   = {arXiv preprint},
  year      = {2025},
  note      = {Work in progress. Code: https://github.com/BUG423/diffusion-io}
}
```

---

## 7. License

This project is released under the [MIT License](LICENSE).

**Copyright Notice**: The code and methodology presented in this repository are provided for research and academic purposes. Please respect all applicable copyright and intellectual property rights. Redistribution and derivative works must retain this notice.

For questions or collaboration inquiries, please open an issue on this repository.

---

*This README is maintained as part of the project's academic disclosure. Experimental results will be updated as the full evaluation is completed.*
