# Experiment Evolution

## VLSI Netra — Experimental Development

VLSI Netra was developed through multiple controlled model experiments rather than selecting a single architecture without comparison.

The experimental progression was used to study how changes in network capacity and configuration affected image-restoration quality.

---

## 1. Common Objective

The objective of all experiments was to restore a high-quality 256 x 256 grayscale image from a degraded 128 x 128 grayscale input.

```text
Degraded LR Image
128 x 128
     |
     v
KLAResNet
     |
     v
Restored Image
256 x 256
```

---

# 2. Experiment 1 — Baseline

Experiment 1 established the initial KLAResNet baseline.

### Configuration

| Parameter | Experiment 1 |
|---|---:|
| Architecture | KLAResNet |
| Residual blocks | 8 |
| Features | 64 |
| Parameters | 776,705 |
| Input | 1 x 128 x 128 |
| Output | 1 x 256 x 256 |

### Validation results

| Metric | Experiment 1 |
|---|---:|
| MAE | 0.030537 |
| PSNR | 28.3689 dB |
| SSIM | 0.765721 |
| Gradient | 0.559840 |

Experiment 1 provided the baseline against which subsequent experiments were evaluated.

---

# 3. Experiment 2 — Increased Network Depth

Experiment 2 increased the depth of the residual feature-extraction stage.

```text
Experiment 1
8 residual blocks
       |
       v
Experiment 2
12 residual blocks
```

The feature width remained 64.

### Configuration

| Parameter | Experiment 2 |
|---|---:|
| Architecture | KLAResNet |
| Residual blocks | 12 |
| Features | 64 |
| Parameters | 1,072,129 |
| Input | 1 x 128 x 128 |
| Output | 1 x 256 x 256 |

The increase in residual depth increased the trainable parameter count from 776,705 to 1,072,129.

### Validation results

| Metric | Experiment 2 |
|---|---:|
| MAE | 0.030200 |
| PSNR | 28.4556 dB |
| SSIM | 0.766358 |
| Gradient | 0.548285 |

Experiment 2 achieved the strongest overall balance across the primary restoration metrics and was selected as the final repository model.

### Selected checkpoint

```text
Model:
kla_resnet_exp2_best.pt

Residual blocks:
12

Parameters:
1,072,129

Best checkpoint epoch:
27

Best validation loss:
0.09600453078746796

SHA256:
90724a19cef86f35f2fa8eb5505d6def9b614dda05abe73589595300d8e9441f
```

---

# 4. Experiment 3 — Alternative Configuration

Experiment 3 was evaluated as an additional model configuration under the same overall restoration objective.

### Validation results

| Metric | Experiment 3 |
|---|---:|
| MAE | 0.032296 |
| PSNR | 27.8073 dB |
| SSIM | 0.749315 |
| Gradient | 0.531656 |

Experiment 3 did not provide a better overall combination of MAE, PSNR and SSIM than Experiment 2.

Therefore, Experiment 3 was not selected as the final model.

Architecture-specific Experiment 3 details are intentionally not stated here unless they are available from the original experiment record. This avoids presenting unverified configuration information.

---

# 5. Side-by-Side Comparison

| Metric | Exp 1 | Exp 2 | Exp 3 |
|---|---:|---:|---:|
| MAE | 0.030537 | 0.030200 | 0.032296 |
| PSNR | 28.3689 | 28.4556 | 27.8073 |
| SSIM | 0.765721 | 0.766358 | 0.749315 |
| Gradient | 0.559840 | 0.548285 | 0.531656 |
| Selection | Baseline | SELECTED | Rejected |

## Why Experiment 2 was selected

Experiment 2 was selected because it provided the strongest overall restoration performance.

- Lowest MAE among the three experiments.
- Highest PSNR among the three experiments.
- Highest SSIM among the three experiments.
- Improved gradient metric compared with Experiment 1.
- Higher model capacity than the baseline through additional residual blocks.

The final decision was therefore based on measured validation performance rather than model size alone.

---

# 6. Hardware and Training

All experiments were developed using GPU-accelerated PyTorch training.

### Hardware

```text
GPU: NVIDIA Tesla T4
```

### Training protocol

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| GPU | NVIDIA Tesla T4 |
| Training protocol | Minimum 50 epochs |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Adam betas | (0.9, 0.999) |

The best validation checkpoint was retained rather than simply selecting the final training epoch.

For Experiment 2, the validated best checkpoint is recorded at epoch 27.

---

# 7. Experiment 2 Loss Function

The selected model uses a composite restoration objective:

```text
Total Loss =
    0.55 x MAE
  + 0.25 x (1 - SSIM)
  + 0.20 x Gradient Loss
```

### Pixel fidelity

MAE encourages accurate reconstruction of image intensity values.

### Structural similarity

1 - SSIM encourages preservation of structural information.

### Gradient preservation

Gradient L1 encourages preservation of edges and fine image transitions.

---

# 8. Final Selected Pipeline

```text
INPUT
  |
  v
Noisy / Degraded LR Image
128 x 128
  |
  v
KLAResNet Head
  |
  v
12 Residual Processing Blocks
64 Features
  |
  v
Feature Reconstruction
  |
  v
2x PixelShuffle
  |
  v
Output Layer
  |
  v
Restored Image
256 x 256
```

---

# 9. Final Repository Model

The final GitHub repository uses:

```text
models/kla_resnet_exp2_best.pt
```

The original Experiment 1 checkpoint is no longer the selected repository model.

### Verified model

```text
Residual blocks : 12
Parameters      : 1,072,129
Input           : 1 x 128 x 128
Output          : 1 x 256 x 256
```

The checkpoint was verified using strict state-dictionary loading and a forward-pass compatibility test.

---

# 10. Engineering Significance

The experiment progression demonstrates an iterative engineering approach:

```text
EXP 1
Baseline
  |
  | Increase residual depth
  v
EXP 2
Higher-capacity model
  |
  | Comparative validation
  v
EXP 3
Alternative configuration
  |
  v
EXP 2 SELECTED
Best overall validation performance
```

This progression demonstrates that the final architecture was selected through comparative experimentation and validation rather than being chosen arbitrarily.
