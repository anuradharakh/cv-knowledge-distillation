"""
=============================================================================
ML2 Homework 2 — train_teacher.py
=============================================================================
Train a large pretrained TEACHER and cache its logits on unlabeled images.

The teacher is a training tool only. It is intentionally larger than the
500,000-parameter deployment limit and must NOT be submitted to the client.
Only the student model produced by train_baby.py or distill.py is submitted.

Default teacher:
  EfficientNet-B0 pretrained on ImageNet, with its classifier replaced by a
  7-class head.

Run:
  python train_teacher.py

Optional:
  python train_teacher.py --backbone resnet18 --epochs 15

Outputs:
  teacher_<backbone>.pth       — cached teacher weights for reuse
  teacher_soft_labels.npy      — pre-softmax teacher logits on unlabeled images
  teacher_filenames.txt        — filename order matching the logits
=============================================================================
"""
from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch

# Compatibility guard for environments where torchvision is installed without
# compiled custom ops such as torchvision::nms. It is harmless when the op
# already exists and lets torchvision import cleanly in CPU-only notebooks.
try:
    from torch.library import Library
    _tv_lib = Library("torchvision", "DEF")
    _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split

TRAIN_ROOT = Path(__file__).parent / "train"
UNLABELED_ROOT = Path(__file__).parent / "unlabeled"
NUM_CLASSES = 7
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class LabeledDataset(Dataset):
    def __init__(self, root: Path, size: int = 224, augment: bool = False):
        self.root = Path(root)
        self.df = pd.read_csv(root / "labels.csv")
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load_image(self, filename: str) -> torch.Tensor:
        with Image.open(self.root / filename) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            img = torch.flip(img, dims=[2])
        if torch.rand(()) < 0.8:
            brightness = 0.85 + 0.30 * torch.rand(())
            contrast = 0.85 + 0.30 * torch.rand(())
            mean = img.mean(dim=(1, 2), keepdim=True)
            img = (img - mean) * contrast + mean
            img = (img * brightness).clamp(0.0, 1.0)
        return img

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self._load_image(row["filename"])
        if self.augment:
            img = self._augment(img)
        img = F.interpolate(img.unsqueeze(0), size=(self.size, self.size),
                            mode="bilinear", align_corners=False).squeeze(0)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img, int(row["label"])


class UnlabeledDataset(Dataset):
    def __init__(self, root: Path, size: int = 224):
        self.root = Path(root)
        self.filenames = sorted(p.name for p in root.glob("*.jpg"))
        self.size = size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        fn = self.filenames[i]
        with Image.open(self.root / fn) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1)
        img = F.interpolate(img.unsqueeze(0), size=(self.size, self.size),
                            mode="bilinear", align_corners=False).squeeze(0)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img, fn


def build_teacher(backbone: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    backbone = backbone.lower()

    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m

    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m

    if backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m

    raise ValueError("Supported backbones: efficientnet_b0, resnet18, resnet50")


def fine_tune(model, train_loader, val_loader, epochs: int, device, lr: float, weight_decay: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.03)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:3d}  train_loss={loss_sum / total:.4f}  "
            f"train_acc={correct / total:.4f}  val_acc={val_acc:.4f}  "
            f"lr={opt.param_groups[0]['lr']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best teacher val_acc={best_val:.4f}")


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


@torch.inference_mode()
def dump_soft_labels(model, unlabeled_loader, device, out_path: Path, filenames_out: Path):
    model.eval()
    all_logits, all_filenames = [], []
    for x, fns in unlabeled_loader:
        x = x.to(device)
        logits = model(x).cpu().numpy().astype("float32")
        all_logits.append(logits)
        all_filenames.extend(fns)
    logits = np.concatenate(all_logits, axis=0)
    np.save(out_path, logits)
    filenames_out.write_text("\n".join(all_filenames) + "\n")
    print(f"Saved {out_path.name} shape={logits.shape}")
    print(f"Saved filename index -> {filenames_out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet18", "resnet50"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_path = Path(__file__).parent / f"teacher_{args.backbone}.pth"

    plain = LabeledDataset(TRAIN_ROOT, size=args.image_size, augment=False)
    n_val = max(1, len(plain) // 5)
    n_train = len(plain) - n_val
    train_ds, val_ds = random_split(
        plain, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    aug = LabeledDataset(TRAIN_ROOT, size=args.image_size, augment=True)
    train_ds.dataset = aug

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    teacher = build_teacher(args.backbone).to(device)
    print(f"Teacher backbone: {args.backbone}")
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")

    if state_path.exists() and not args.force_retrain:
        teacher.load_state_dict(torch.load(state_path, map_location=device))
        print(f"Loaded teacher weights from {state_path.name}")
    else:
        fine_tune(teacher, train_loader, val_loader, args.epochs, device, args.lr, args.weight_decay)
        torch.save(teacher.state_dict(), state_path)
        print(f"Saved teacher weights -> {state_path.name}")

    unlabeled = UnlabeledDataset(UNLABELED_ROOT, size=args.image_size)
    unlabeled_loader = DataLoader(unlabeled, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dump_soft_labels(
        teacher, unlabeled_loader, device,
        out_path=Path(__file__).parent / "teacher_soft_labels.npy",
        filenames_out=Path(__file__).parent / "teacher_filenames.txt",
    )


if __name__ == "__main__":
    main()
