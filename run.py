import os
import sys
import numpy as np
import torch


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

        return torch.clamp(output, 0.0, 1.0)


# ============================================================
# INPUT
# ============================================================

def load_input(path):
    array = np.load(path).astype(np.float32)

    if array.ndim == 2:
        tensor = torch.from_numpy(array).unsqueeze(0)

    elif array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(
                f"Expected grayscale input with shape (H,W) "
                f"or (1,H,W), got {array.shape}"
            )
        tensor = torch.from_numpy(array)

    else:
        raise ValueError(
            f"Expected input shape (H,W) or (1,H,W), "
            f"got {array.shape}"
        )

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# MODEL LOADING
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
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 3:
        print(
            "Usage: python run.py <input-dir> <output-dir>"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    repo_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    checkpoint_path = os.path.join(
        repo_dir,
        "models",
        "kla_resnet_exp2_best.pt"
    )

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = load_model(
        checkpoint_path,
        device
    )

    input_files = sorted(
        filename
        for filename in os.listdir(input_dir)
        if filename.endswith(".npy")
    )

    print("Input files:", len(input_files))

    for index, filename in enumerate(input_files, start=1):

        input_path = os.path.join(
            input_dir,
            filename
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        input_tensor = load_input(
            input_path
        ).to(device)

        with torch.no_grad():
            prediction = model(input_tensor)

        expected_shape = (
            1,
            1,
            256,
            256
        )

        if tuple(prediction.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected model output shape: "
                f"{tuple(prediction.shape)}"
            )

        prediction = prediction.squeeze().cpu().numpy()

        prediction = np.clip(
            prediction,
            0.0,
            1.0
        ).astype(np.float32)

        if not np.all(np.isfinite(prediction)):
            raise RuntimeError(
                f"NaN or Inf detected in output: {filename}"
            )

        np.save(
            output_path,
            prediction
        )

        if index % 100 == 0 or index == len(input_files):
            print(
                f"Processed {index}/{len(input_files)}"
            )

    print("Inference completed successfully.")


if __name__ == "__main__":
    main()
