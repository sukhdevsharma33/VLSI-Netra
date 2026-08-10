# VLSI Netra - Experimental Results

## Dataset

- Total samples: 3,200
- Training samples: 2,880
- Validation samples: 320
- NoisyLR input: 1 x 128 x 128
- Ground truth: 1 x 256 x 256

## Model

- Architecture: KLAResNet
- Total parameters: 776,705
- Trainable parameters: 776,705
- Input channels: 1
- Output channels: 1

## Training Configuration

| Parameter | Value |
|---|---:|
| Device | CUDA |
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Best epoch | 14 |

## Best Checkpoint

The best validation checkpoint is:

models/kla_resnet_best.pt

Checkpoint epoch: 14

Best validation loss: 0.07613037005066872

Validation pixel loss: 0.03204418243840337

Validation SSIM loss: 0.2524751156568527

## Full Validation Evaluation

The best checkpoint was evaluated on all 320 validation samples.

| Metric | Result |
|---|---:|
| Mean PSNR | 27.863525 dB |
| Mean SSIM | 0.751410 |
| PSNR minimum | 10.856936 dB |
| PSNR maximum | 37.465970 dB |
| SSIM minimum | 0.243979 |
| SSIM maximum | 0.978146 |

## Standalone Inference Verification

The standalone inference script was executed independently from the training notebook.

- Script: src/inference.py
- Input files: 320
- Output files: 320
- Device: CUDA
- Return code: 0

## Output Verification

All 320 generated PNG files were verified.

- File count: 320
- Image size: 256 x 256
- Image mode: L (grayscale)
- Invalid images: 0

## Integrity Verification

The repository copies of train.py and inference.py were verified using SHA256 hashes.

The repository copy of kla_resnet_best.pt was also verified using SHA256.

The model copy matched the original checkpoint byte-for-byte.

## Conclusion

The complete VLSI Netra pipeline was successfully restored, trained, evaluated, and independently verified. The best model was obtained at epoch 14 and achieved a mean PSNR of 27.863525 dB and mean SSIM of 0.751410 on the 320-sample validation set.
