# VLSI Netra

## Competition Inference

The repository provides the required competition entry point:

`python run.py <input-dir> <output-dir>`

The inference pipeline runs completely offline and does not require internet access, API keys, additional model downloads, user interaction, or manual configuration.

The validated model checkpoint is included in:

`models/kla_resnet_exp2_best.pt`

### Input

The script reads all `.npy` files from the input directory.

- Format: `.npy`
- Data type: `float32`
- Grayscale
- Expected input resolution: `128 x 128`
- Accepted shape: `(128, 128)` or `(1, 128, 128)`

### Output

One restored `.npy` file is generated for every input file.

- Same filename as the input
- Grayscale
- Shape: `(256, 256)`
- Data type: `float32`
- Values in `[0, 1]`
- No NaN or Inf values

Example:

`python run.py input output`

produces:

`input/001116.npy -> output/001116.npy`

### GPU Execution

The script automatically uses CUDA when an NVIDIA GPU is available and otherwise falls back to CPU.

The competition entry point was successfully tested on:

- NVIDIA Tesla T4
- PyTorch 2.11.0
- CUDA 12.8

### Installation

Install dependencies with:

`pip install -r requirements.txt`

No additional model download is required.

---


## Deep Learning-Based Image Restoration for VLSI Inspection

VLSI Netra is a deep-learning image-restoration system designed to reconstruct high-resolution grayscale images from noisy low-resolution inputs.

The system uses a residual convolutional neural network to restore image quality while preserving structural and edge information.

---

## Selected Model — Experiment 2

Experiment 2 is the current validated model selected for the repository based on fair three-model validation.

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
| Total training epochs | **50** |
| Best validation epoch | **27** |

### Restoration Loss

The model uses a weighted combination of pixel, structural, and gradient losses:

```text
L = 0.55 × L_MAE
  + 0.25 × L_SSIM
  + 0.20 × L_gradient
```

where:

- `L_MAE` measures pixel-level reconstruction error.
- `L_SSIM` is implemented as `1 − SSIM`.
- `L_gradient` measures horizontal and vertical gradient differences.

---

## Fair Validation Results

The three experiments were compared using the same evaluation framework.

| Model | MAE ↓ | PSNR ↑ | SSIM ↑ | Gradient ↓ |
|---|---:|---:|---:|---:|
| Experiment 1 | 0.030537 | 28.3689 | 0.765721 | 0.559840 |
| **Experiment 2** | **0.030200** | **28.4556** | **0.766358** | **0.548285** |
| Experiment 3 | 0.032296 | 27.8073 | 0.749315 | 0.531656 |

### Selected Result

**Experiment 2** provides the strongest overall restoration result in the fair comparison.

- **MAE:** 0.030200
- **PSNR:** 28.4556 dB
- **SSIM:** 0.766358
- **Gradient:** 0.548285

Experiment 2 achieves the best MAE, PSNR, and SSIM among the three compared models.

---

## Validation Checkpoint

The published Experiment 2 checkpoint is:

`models/kla_resnet_exp2_best.pt`

Checkpoint metadata:

```text
Best epoch       : 27
Total epochs     : 50
Best val loss    : 0.09600453078746796
Parameters       : 1,072,129
Residual blocks  : 12
```

The repository checkpoint was verified against the original validated checkpoint using SHA256:

`90724a19cef86f35f2fa8eb5505d6def9b614dda05abe73589595300d8e9441f`

---

## Inference

The published inference pipeline accepts a grayscale NumPy input:

`128 × 128`

and produces:

`256 × 256`

grayscale PNG output.

Example:

```bash
python src/inference.py \
    --input_dir /path/to/input \
    --output_dir /path/to/output \
    --checkpoint models/kla_resnet_exp2_best.pt
```

The inference implementation uses strict checkpoint loading to ensure that the published architecture matches the checkpoint.

---

## Repository Structure

```text
VLSI Netra/
├── src/
│   ├── train.py
│   └── inference.py
├── models/
│   └── kla_resnet_exp2_best.pt
├── docs/
│   └── RESULTS.md
├── outputs/
├── .gitignore
└── README.md
```

---

## Reproducibility

The repository contains:

1. Experiment 2 training source.
2. Experiment 2 inference source.
3. The validated Experiment 2 checkpoint.
4. Fair validation results for Experiments 1–3.
5. Checkpoint verification information.

The training source records epoch-level training and validation history in the checkpoint metadata.

---

## Project Goal

The goal of VLSI Netra is to improve the quality of degraded or noisy VLSI inspection imagery through learned image restoration.

The system focuses on:

- Noise reduction
- Resolution enhancement
- Structural preservation
- Edge preservation
- Quantitative image-quality improvement

---

## Status

**Current repository model: Experiment 2**

**Model status: Validated**

**Inference status: Verified**

**Checkpoint status: Verified**

---

## Experimental Development & Hardware

VLSI Netra was developed through comparative experiments rather than selecting a single architecture without validation.

Three model configurations were evaluated. Experiment 1 established the baseline, Experiment 2 increased residual-network depth, and Experiment 3 evaluated an alternative configuration.

### Hardware

- **GPU:** NVIDIA Tesla T4
- **Framework:** PyTorch
- **Training protocol:** 50 total epochs
- **Best validation epoch:** 27
- **Batch size:** 16
- **Optimizer:** AdamW
- **Learning rate:** 0.0001
- **Weight decay:** 0.0001

### Experiment progression

| Model | Residual Blocks | Parameters | MAE | PSNR | SSIM | Gradient |
|---|---:|---:|---:|---:|---:|---:|
| Experiment 1 | 8 | 776,705 | 0.030537 | 28.3689 | 0.765721 | 0.559840 |
| **Experiment 2** | **12** | **1,072,129** | **0.030200** | **28.4556** | **0.766358** | **0.548285** |
| Experiment 3 | — | — | 0.032296 | 27.8073 | 0.749315 | 0.531656 |

Experiment 2 was selected because it achieved the best overall combination of MAE, PSNR and SSIM on the common validation protocol.

See `docs/EXPERIMENTS.md` for the complete experimental evolution and methodology.
