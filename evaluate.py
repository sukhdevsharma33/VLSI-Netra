
import os
import time
import argparse

import numpy as np
import torch
from PIL import Image


# ============================================================
# MODEL
# ============================================================

class ResidualBlock(torch.nn.Module):

    def __init__(self, features=64):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        return x + self.block(x)


class KLAResNet(torch.nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=64,
        num_blocks=12
    ):
        super().__init__()

        self.head = torch.nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1
        )

        self.body = torch.nn.Sequential(
            *[
                ResidualBlock(features)
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = torch.nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1
        )

        self.upsample = torch.nn.Sequential(
            torch.nn.Conv2d(
                features,
                features * 4,
                kernel_size=3,
                padding=1
            ),
            torch.nn.PixelShuffle(2),
            torch.nn.ReLU(inplace=True)
        )

        self.tail = torch.nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        head = self.head(x)

        body = self.body(head)

        body = self.body_conv(body)

        body = body + head

        body = self.upsample(body)

        output = self.tail(body)

        return torch.clamp(
            output,
            0.0,
            1.0
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path, device):

    model = KLAResNet(
        in_channels=1,
        out_channels=1,
        features=64,
        num_blocks=12
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# LOAD INPUT
# ============================================================

def load_input(path):

    array = np.load(path).astype(np.float32)

    if array.ndim == 2:
        array = array[None, ...]

    tensor = torch.from_numpy(array)

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# EVALUATION / INFERENCE
# ============================================================

def evaluate(
    model,
    input_dir,
    output_dir,
    device
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    files = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith(".npy")
        ]
    )

    if len(files) == 0:
        raise RuntimeError(
            f"No .npy input images found in: {input_dir}"
        )

    print("=" * 70)
    print("VLSI NETRA — EXPERIMENT 2 EVALUATION")
    print("=" * 70)

    print("Device:", device)
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    print("Input images:", len(files))
    print()

    total_start = time.time()

    with torch.no_grad():

        for index, filename in enumerate(files):

            input_path = os.path.join(
                input_dir,
                filename
            )

            sample_id = os.path.splitext(
                filename
            )[0]

            start = time.time()

            input_tensor = load_input(
                input_path
            ).to(device)

            prediction = model(
                input_tensor
            )

            prediction = (
                prediction
                .squeeze()
                .cpu()
                .numpy()
            )

            prediction = (
                prediction * 255.0
            ).round().clip(
                0,
                255
            ).astype(
                np.uint8
            )

            output_path = os.path.join(
                output_dir,
                f"{sample_id}.png"
            )

            Image.fromarray(
                prediction
            ).save(
                output_path
            )

            elapsed = time.time() - start

            print(
                f"[{index + 1:04d}/{len(files):04d}] "
                f"{filename} -> "
                f"{sample_id}.png | "
                f"{elapsed:.4f}s"
            )

    total_time = time.time() - total_start

    average_time = (
        total_time / len(files)
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "Total images:",
        len(files)
    )

    print(
        f"Total inference time: "
        f"{total_time:.4f} seconds"
    )

    print(
        f"Average inference time/image: "
        f"{average_time:.4f} seconds"
    )

    print(
        f"Average FPS: "
        f"{1.0 / average_time:.2f}"
    )

    print(
        "Outputs:",
        output_dir
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "VLSI Netra Experiment 2 "
            "standalone evaluation script"
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing input .npy images"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where restored PNG images are saved"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    repo_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    checkpoint_path = os.path.join(
        repo_dir,
        "models",
        "kla_resnet_exp2_best.pt"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print(
        "Checkpoint:",
        checkpoint_path
    )

    model = load_model(
        checkpoint_path,
        device
    )

    evaluate(
        model,
        args.input_dir,
        args.output_dir,
        device
    )


if __name__ == "__main__":
    main()
