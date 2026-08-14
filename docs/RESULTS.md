# VLSI Netra — Experimental Results

## Overview

This document records the validation results of the three model experiments conducted for VLSI Netra.

Experiments 1, 2, and 3 were evaluated using the same fair validation framework. Experiment 2 was selected as the current repository model because it achieved the strongest overall combination of MAE, PSNR, and SSIM.

---

## Experiment Comparison

| Experiment | MAE ↓ | PSNR ↑ | SSIM ↑ | Gradient ↓ |
|---|---:|---:|---:|---:|
| Experiment 1 | 0.030537 | 28.3689 | 0.765721 | 0.559840 |
| **Experiment 2** | **0.030200** | **28.4556** | **0.766358** | **0.548285** |
| Experiment 3 | 0.032296 | 27.8073 | 0.749315 | 0.531656 |

### Metric Interpretation

- **MAE:** Lower is better.
- **PSNR:** Higher is better.
- **SSIM:** Higher is better.
- **Gradient:** Lower indicates lower gradient reconstruction error.

---

## Experiment 1

Experiment 1 was the initial baseline restoration model.

Its fair validation results were:

- MAE: **0.030537**
- PSNR: **28.3689 dB**
- SSIM: **0.765721**
- Gradient: **0.559840**

Experiment 1 is retained as part of the experimental history but is not the current repository model.

---

## Experiment 2 — Selected Model

Experiment 2 increased the residual capacity of the restoration network and was selected after fair validation.

### Architecture

| Configuration | Value |
|---|---|
| Model | KLAResNet |
| Residual blocks | **12** |
| Features | **64** |
| Input channels | **1** |
| Output channels | **1** |
| Input resolution | **128 × 128** |
| Output resolution | **256 × 256** |
| Parameters | **1,072,129** |

### Training Configuration

| Configuration | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Betas | (0.9, 0.999) |
| Batch size | 16 |
| Epochs | 27 |

### Loss Function

The restoration objective is:

```text
L = 0.55 × L_MAE
  + 0.25 × L_SSIM
  + 0.20 × L_gradient
```

where the structural component is implemented as:

```text
L_SSIM = 1 − SSIM(prediction, target)
```

The gradient loss compares horizontal and vertical image gradients using L1 distance.

### Best Validation Checkpoint

- Best epoch: **27**
- Best validation loss: **0.09600453078746796**

### Fair Validation Metrics

- MAE: **0.030200**
- PSNR: **28.4556 dB**
- SSIM: **0.766358**
- Gradient: **0.548285**

---

## Experiment 3

Experiment 3 was evaluated using the same fair validation framework.

Its results were:

- MAE: **0.032296**
- PSNR: **27.8073 dB**
- SSIM: **0.749315**
- Gradient: **0.531656**

Experiment 3 did not exceed Experiment 2 on the primary overall restoration metrics.

---

## Why Experiment 2 Was Selected

Experiment 2 achieved:

1. The lowest MAE among the three experiments.
2. The highest PSNR among the three experiments.
3. The highest SSIM among the three experiments.

Although Experiment 3 produced a lower gradient metric, its MAE, PSNR, and SSIM were worse than Experiment 2.

Therefore, Experiment 2 was selected as the best overall restoration model.

---

## Checkpoint Verification

Published checkpoint:

`models/kla_resnet_exp2_best.pt`

Verified properties:

- 12 residual blocks
- 1,072,129 parameters
- Best epoch 27
- Best validation loss 0.09600453078746796
- Checkpoint contains training history metadata

SHA256:

```text
90724a19cef86f35f2fa8eb5505d6def9b614dda05abe73589595300d8e9441f
```

The repository checkpoint SHA256 matches the original validated Experiment 2 checkpoint.

---

## Architecture Verification

The repository training implementation was tested against the published checkpoint using strict state-dictionary loading.

Verification results:

- Parameter count: **1,072,129**
- Missing checkpoint keys: **0**
- Unexpected checkpoint keys: **0**
- Input tensor: **1 × 1 × 128 × 128**
- Output tensor: **1 × 1 × 256 × 256**
- Output range: **[0, 1]**

The architecture and checkpoint are therefore compatible.

---

## Inference Verification

The published `src/inference.py` was tested using the Experiment 2 checkpoint.

The verified pipeline is:

```text
128 × 128 .npy
        ↓
Experiment 2 KLAResNet
        ↓
256 × 256 grayscale PNG
```

The generated PNG was verified as:

- Resolution: **256 × 256**
- Mode: **L (grayscale)**
- Data type: **uint8**
- Pixel range: **0–255**

---

## Final Status

**Experiment 2 is the selected and validated repository model.**

The model checkpoint, training source, inference source, architecture, and validation results have been verified for consistency.
