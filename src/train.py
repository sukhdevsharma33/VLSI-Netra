
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pytorch_msssim import ssim


# ============================================================
# DATASET
# ============================================================

class KLADataset(Dataset):

    def __init__(
        self,
        ids,
        noisy_dir,
        gt_dir
    ):
        self.ids = ids
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        sample_id = self.ids[index]

        noisy_path = os.path.join(
            self.noisy_dir,
            f"{sample_id}.npy"
        )

        gt_path = os.path.join(
            self.gt_dir,
            f"{sample_id}.npy"
        )

        noisy = np.load(
            noisy_path
        ).astype(
            np.float32
        )

        gt = np.load(
            gt_path
        ).astype(
            np.float32
        )

        if noisy.ndim == 2:
            noisy = noisy[None, ...]

        if gt.ndim == 2:
            gt = gt[None, ...]

        noisy = torch.from_numpy(
            noisy
        )

        gt = torch.from_numpy(
            gt
        )

        return (
            noisy,
            gt,
            sample_id
        )


# ============================================================
# MODEL
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        features=64
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        return x + self.block(x)


class KLAResNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=64,
        num_blocks=8
    ):
        super().__init__()

        self.head = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    features
                )
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1
        )

        self.upsample = nn.Sequential(

            nn.Conv2d(
                features,
                features * 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(
                inplace=True
            )
        )

        self.tail = nn.Conv2d(
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
# LOSS
# ============================================================

def restoration_loss(
    prediction,
    target
):

    pixel_loss = nn.functional.l1_loss(
        prediction,
        target
    )

    structural_loss = 1.0 - ssim(
        prediction,
        target,
        data_range=1.0,
        size_average=True
    )

    total_loss = (
        0.5 * pixel_loss
        +
        0.5 * structural_loss
    )

    return (
        total_loss,
        pixel_loss,
        structural_loss
    )


# ============================================================
# LOAD IDS
# ============================================================

def load_ids(path):

    with open(
        path,
        "r"
    ) as f:

        return [
            line.strip()
            for line in f
            if line.strip()
        ]


# ============================================================
# VALIDATION
# ============================================================

def evaluate_validation_loss(
    model,
    val_loader,
    device
):

    model.eval()

    total_loss = 0.0
    total_pixel = 0.0
    total_ssim = 0.0
    batches = 0

    with torch.no_grad():

        for noisy, gt, _ in val_loader:

            noisy = noisy.to(
                device,
                non_blocking=True
            )

            gt = gt.to(
                device,
                non_blocking=True
            )

            prediction = model(
                noisy
            )

            loss_total, loss_pixel, loss_structural = (
                restoration_loss(
                    prediction,
                    gt
                )
            )

            total_loss += (
                loss_total.item()
            )

            total_pixel += (
                loss_pixel.item()
            )

            total_ssim += (
                loss_structural.item()
            )

            batches += 1

    return (
        total_loss / batches,
        total_pixel / batches,
        total_ssim / batches
    )


# ============================================================
# TRAINING
# ============================================================

def train(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    os.makedirs(
        args.model_dir,
        exist_ok=True
    )

    train_ids = load_ids(
        args.train_ids
    )

    val_ids = load_ids(
        args.val_ids
    )

    train_dataset = KLADataset(
        train_ids,
        args.noisy_dir,
        args.gt_dir
    )

    val_dataset = KLADataset(
        val_ids,
        args.noisy_dir,
        args.gt_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = KLAResNet(
        in_channels=1,
        out_channels=1,
        features=64,
        num_blocks=8
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    best_val_loss = float("inf")
    best_epoch = 0

    best_model_path = os.path.join(
        args.model_dir,
        "kla_resnet_best.pt"
    )

    final_model_path = os.path.join(
        args.model_dir,
        "kla_resnet_final.pt"
    )

    for epoch in range(
        1,
        args.epochs + 1
    ):

        start_time = time.time()

        model.train()

        running_train_loss = 0.0
        train_batches = 0

        for noisy, gt, _ in train_loader:

            noisy = noisy.to(
                device,
                non_blocking=True
            )

            gt = gt.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(
                noisy
            )

            loss_total, _, _ = (
                restoration_loss(
                    prediction,
                    gt
                )
            )

            loss_total.backward()

            optimizer.step()

            running_train_loss += (
                loss_total.item()
            )

            train_batches += 1

        train_loss = (
            running_train_loss
            /
            train_batches
        )

        val_loss, val_pixel, val_ssim = (
            evaluate_validation_loss(
                model,
                val_loader,
                device
            )
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        improved = (
            val_loss
            <
            best_val_loss
        )

        if improved:

            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_pixel_loss": val_pixel,
                    "val_ssim_loss": val_ssim
                },
                best_model_path
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Pixel: {val_pixel:.6f} | "
            f"Val SSIM: {val_ssim:.6f} | "
            f"Time: {elapsed:.1f}s"
            +
            (
                "  <-- BEST"
                if improved
                else ""
            )
        )

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss
        },
        final_model_path
    )

    print(
        "\n========== TRAINING COMPLETE =========="
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation loss:",
        best_val_loss
    )

    print(
        "Best model:",
        best_model_path
    )

    print(
        "Final model:",
        final_model_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2
    )

    args = parser.parse_args()

    args.dataset_dir = os.path.join(
        args.base_dir,
        "Datasets",
        "train_extracted",
        "train"
    )

    args.noisy_dir = os.path.join(
        args.dataset_dir,
        "NoisyLR"
    )

    args.gt_dir = os.path.join(
        args.dataset_dir,
        "GT"
    )

    args.train_ids = os.path.join(
        args.base_dir,
        "Datasets",
        "splits",
        "train_ids.txt"
    )

    args.val_ids = os.path.join(
        args.base_dir,
        "Datasets",
        "splits",
        "val_ids.txt"
    )

    args.model_dir = os.path.join(
        args.base_dir,
        "Models"
    )

    train(args)


if __name__ == "__main__":
    main()
