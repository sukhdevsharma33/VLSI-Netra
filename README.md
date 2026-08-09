# VLSI Netra

## Deep Learning-Based Image Restoration for VLSI Inspection

VLSI Netra is a deep learning image-restoration project designed to reconstruct high-resolution images from noisy low-resolution inputs.

The project uses a lightweight residual convolutional neural network to transform:

- Noisy low-resolution input: 128 x 128
- Restored output: 256 x 256
- Image channels: 1 (grayscale)

The trained model was evaluated on a held-out validation set of 320 samples.

## Project Structure

```text
VLSI Netra/
|-- src/
|   |-- train.py
|   `-- inference.py
|-- models/
|   `-- kla_resnet_best.pt
|-- docs/
|-- outputs/
`-- README.md
```

## Model

The project uses the KLAResNet architecture.

### Configuration

| Parameter | Value |
|---|---:|
| Input channels | 1 |
| Output channels | 1 |
| Features | 64 |
| Residual blocks | 8 |
| Total parameters | 776,705 |
| Trainable parameters | 776,705 |

The model accepts a 128 x 128 noisy image and produces a 256 x 256 restored image.

## Dataset

The training dataset contains paired NoisyLR and ground-truth (GT) images.

| Split | Samples |
|---|---:|
| Training | 2,880 |
| Validation | 320 |
| Total | 3,200 |

Input and ground-truth dimensions:

```text
NoisyLR : 1 x 128 x 128
GT      : 1 x 256 x 256
```

## Training

| Parameter | Value |
|---|---:|
| Device | CUDA |
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Best epoch | 14 |

The best checkpoint is models/kla_resnet_best.pt.

The best model was selected using validation loss.

## Validation Results

Final evaluation was performed on all 320 validation samples.

| Metric | Result |
|---|---:|
| Mean PSNR | 27.863525 dB |
| Mean SSIM | 0.751410 |
| PSNR minimum | 10.856936 dB |
| PSNR maximum | 37.465970 dB |
| SSIM minimum | 0.243979 |
| SSIM maximum | 0.978146 |

## Standalone Inference

The project includes a standalone inference script:

src/inference.py

Run it from a terminal using:

```bash
python src/inference.py --input_dir <INPUT_DIRECTORY> --output_dir <OUTPUT_DIRECTORY> --checkpoint models/kla_resnet_best.pt
```

The script was tested on 320 validation inputs.

Verified results:

- Input files processed: 320
- Output PNG files: 320
- Output size: 256 x 256
- Image mode: grayscale (L)
- Inference return code: 0

## Training Script

The training implementation is provided in src/train.py.

The script contains the model, dataset, loss, training and validation components used for the project.

## Reproducibility

The following components were independently verified:

- Dataset paths
- Training and validation splits
- Dataset loading
- DataLoader
- Model architecture
- Model parameter count
- Best checkpoint
- Forward inference
- Restoration loss
- Validation evaluation
- Standalone inference script
- Generated output images

The saved best checkpoint is byte-for-byte identical to the validated original checkpoint.

## Output

The restored output images are grayscale PNG files with dimensions 256 x 256.

The standalone inference pipeline successfully generated 320 valid output images from 320 validation inputs.

## Project Goal

VLSI Netra aims to improve the quality and resolution of degraded VLSI inspection imagery using deep learning-based image restoration, enabling clearer visual information for downstream inspection and analysis.

## Status

Project pipeline verified successfully.

Best validation model:

- Epoch: 14
- PSNR: 27.863525 dB
- SSIM: 0.751410
