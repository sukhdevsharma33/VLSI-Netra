
import os
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

            torch.nn.ReLU(
                inplace=True
            ),

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

            torch.nn.ReLU(
                inplace=True
            )
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

def load_model(
    checkpoint_path,
    device
):

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
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# LOAD INPUT
# ============================================================

def load_npy(path):

    array = np.load(
        path
    ).astype(
        np.float32
    )

    tensor = torch.from_numpy(
        array
    )

    if tensor.ndim == 2:

        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 3:

        raise ValueError(
            f"Expected input with shape "
            f"(H,W) or (C,H,W), got "
            f"{tuple(tensor.shape)}"
        )

    tensor = tensor.unsqueeze(0)

    if tensor.shape[1] != 1:

        raise ValueError(
            f"Expected 1 input channel, "
            f"got {tensor.shape[1]}"
        )

    return tensor


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_prediction(
    prediction,
    output_path
):

    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
    )

    prediction = np.clip(
        prediction,
        0.0,
        1.0
    )

    prediction = (
        prediction * 255.0
    ).round().astype(
        np.uint8
    )

    Image.fromarray(
        prediction
    ).save(
        output_path
    )


# ============================================================
# SINGLE IMAGE INFERENCE
# ============================================================

def predict_file(
    model,
    input_path,
    output_path,
    device
):

    input_tensor = load_npy(
        input_path
    ).to(device)

    with torch.no_grad():

        prediction = model(
            input_tensor
        )

    if tuple(prediction.shape) != (
        1,
        1,
        256,
        256
    ):

        raise RuntimeError(
            "Unexpected model output shape: "
            f"{tuple(prediction.shape)}"
        )

    save_prediction(
        prediction,
        output_path
    )


# ============================================================
# DIRECTORY INFERENCE
# ============================================================

def run_inference(
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
            if f.endswith(".npy")
        ]
    )

    print(
        "Input files:",
        len(files)
    )

    for index, filename in enumerate(files):

        input_path = os.path.join(
            input_dir,
            filename
        )

        sample_id = os.path.splitext(
            filename
        )[0]

        output_path = os.path.join(
            output_dir,
            f"{sample_id}.png"
        )

        predict_file(
            model,
            input_path,
            output_path,
            device
        )

        if (
            (index + 1) % 100 == 0
            or
            (index + 1) == len(files)
        ):

            print(
                f"Processed "
                f"{index + 1}/{len(files)}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "VLSI Netra Experiment 2 "
            "image restoration inference"
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    model = load_model(
        args.checkpoint,
        device
    )

    run_inference(
        model,
        args.input_dir,
        args.output_dir,
        device
    )


if __name__ == "__main__":
    main()
