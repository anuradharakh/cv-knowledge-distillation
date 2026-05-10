"""
=============================================================================
ML2 Homework 2 — train_baby.py
=============================================================================
Train the deployable STUDENT model from scratch.

Important assignment contract:
  - The leaderboard/server calls the submitted model with x shaped
    (B, 3, 256, 256), float32 in [0, 1].
  - The submitted model must return raw logits shaped (B, 7).
  - Preprocessing must be inside the submitted module.
  - The submitted student must have strictly fewer than 500,000 parameters.

This file defines:
  - ImageDataset       : labeled image dataset
  - Preprocess         : resize + normalize wrapper included in model.pt
  - SmallCNN           : compact MobileNet-style CNN student
  - count_params       : parameter counter used by all scripts

Run:
  python train_baby.py

Output:
  model.pt  — TorchScript student model, ready to upload as a baseline.
=============================================================================
"""
from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

DATA_ROOT = Path(__file__).parent / "train"
NUM_CLASSES = 7
PARAM_LIMIT = 500_000
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Dataset
# =============================================================================
def read_rgb_image(path: Path) -> torch.Tensor:
    """Read an image as a float tensor shaped (3, H, W) in [0, 1]."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class ImageDataset(Dataset):
    """Reads labeled images from train/labels.csv.

    Augmentation is used only during training. The submitted model itself stays
    deterministic because augmentation is not part of Preprocess.
    """

    def __init__(self, root: Path, augment: bool = False):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / "labels.csv")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        # img: float tensor in [0, 1], shape (3, H, W)
        if torch.rand(()) < 0.5:
            img = torch.flip(img, dims=[2])

        # Small color jitter. Kept intentionally modest for production-line data.
        if torch.rand(()) < 0.8:
            brightness = 0.85 + 0.30 * torch.rand(())
            contrast = 0.85 + 0.30 * torch.rand(())
            mean = img.mean(dim=(1, 2), keepdim=True)
            img = (img - mean) * contrast + mean
            img = img * brightness
            img = img.clamp(0.0, 1.0)

        # Random erasing encourages robustness to small occlusions/reflections.
        if torch.rand(()) < 0.25:
            _, h, w = img.shape
            erase_h = int(h * (0.08 + 0.12 * torch.rand(())))
            erase_w = int(w * (0.08 + 0.12 * torch.rand(())))
            y0 = int(torch.randint(0, max(1, h - erase_h + 1), (1,)).item())
            x0 = int(torch.randint(0, max(1, w - erase_w + 1), (1,)).item())
            img[:, y0:y0 + erase_h, x0:x0 + erase_w] = img.mean()
        return img

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = read_rgb_image(self.root / row["filename"])
        if self.augment:
            img = self._augment(img)
        return img, int(row["label"])


# =============================================================================
# Preprocess wrapper — part of submitted model.pt
# =============================================================================
class Preprocess(nn.Module):
    """Resize server-shaped 256x256 images and normalize before SmallCNN."""

    def __init__(self, net: nn.Module, size: int = 96,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.net = net
        self.size = size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.net(x)


# =============================================================================
# Compact student architecture
# =============================================================================
class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int | None = None,
                 groups: int = 1):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class MBConvBlock(nn.Module):
    """MobileNet-style inverted bottleneck block.

    Architecture:
      1x1 expansion conv -> depthwise 3x3 conv -> 1x1 projection conv.
    Depthwise/grouped convolution keeps the student compact and fast.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 expand: int = 3, drop: float = 0.0):
        super().__init__()
        mid_ch = in_ch * expand
        self.use_residual = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            ConvBNAct(in_ch, mid_ch, kernel_size=1, stride=1, padding=0),
            ConvBNAct(mid_ch, mid_ch, kernel_size=3, stride=stride,
                      padding=1, groups=mid_ch),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.dropout = nn.Dropout2d(drop) if drop > 0.0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.block(x))
        if self.use_residual:
            y = x + y
        return self.act(y)


class SmallCNN(nn.Module):
    """Deployable student model.

    Parameter count is about 412k inside SmallCNN, still safely below the
    assignment limit after adding the non-learned Preprocess buffers.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 24, kernel_size=3, stride=2),      # 96 -> 48
            MBConvBlock(24, 32, stride=1, expand=2, drop=0.01),
            MBConvBlock(32, 48, stride=2, expand=3, drop=0.02),  # 48 -> 24
            MBConvBlock(48, 48, stride=1, expand=3, drop=0.02),
            MBConvBlock(48, 72, stride=2, expand=3, drop=0.03),  # 24 -> 12
            MBConvBlock(72, 72, stride=1, expand=3, drop=0.03),
            MBConvBlock(72, 96, stride=2, expand=4, drop=0.04),  # 12 -> 6
            MBConvBlock(96, 128, stride=1, expand=4, drop=0.05),
            MBConvBlock(128, 160, stride=1, expand=4, drop=0.05),
            nn.Conv2d(160, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# =============================================================================
# Training / evaluation helpers
# =============================================================================
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_one_epoch(model, loader, opt, device, label_smoothing: float = 0.05):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total, loss_sum / total


def save_torchscript(model: nn.Module, path: str = "model.pt") -> None:
    model_cpu = model.cpu().eval()
    with torch.inference_mode():
        dummy = torch.rand(2, 3, 256, 256)
        out = model_cpu(dummy)
        assert out.shape == (2, NUM_CLASSES), f"Output shape mismatch: {tuple(out.shape)}"
    torch.jit.save(torch.jit.script(model_cpu), path)
    print(f"Saved {path} — upload this student model to the leaderboard.")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--out", type=str, default="model.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same deterministic 80/20 split used by teacher and distillation scripts.
    full_plain = ImageDataset(DATA_ROOT, augment=False)
    n_val = max(1, len(full_plain) // 5)
    n_train = len(full_plain) - n_val
    train_subset, val_subset = random_split(
        full_plain, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Rebuild the training dataset with augmentation, then reuse the same indices.
    full_aug = ImageDataset(DATA_ROOT, augment=True)
    train_subset.dataset = full_aug

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = Preprocess(SmallCNN(), size=args.image_size).to(device)
    n_params = count_params(model)
    print(f"Student parameters: {n_params:,}")
    assert n_params < PARAM_LIMIT, f"Over assignment cap: {n_params:,} >= {PARAM_LIMIT:,}"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}  "
            f"lr={opt.param_groups[0]['lr']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val:.4f}")
    save_torchscript(model, args.out)


if __name__ == "__main__":
    main()
