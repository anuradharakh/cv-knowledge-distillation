from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

from src.utils import dataloader_kwargs, ensure_dir, get_device, load_config, set_seed


TRAIN_ROOT = Path("train")
UNLABELED_ROOT = Path("unlabeled")
CONFIG_PATH = Path("configs/teacher.yml")


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class TeacherLabeledDataset(Dataset):
    def __init__(self, root, image_size, indices=None):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / "labels.csv")

        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = torchvision.io.read_image(str(self.root / row["filename"]))

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]

        img = img.float() / 255.0

        img = F.interpolate(
            img.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        img = (img - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)

        return img, int(row["label"])


class TeacherUnlabeledDataset(Dataset):
    def __init__(self, root, image_size):
        self.root = Path(root)
        self.filenames = sorted(p.name for p in self.root.glob("*.jpg"))
        self.image_size = image_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        fn = self.filenames[i]
        img = torchvision.io.read_image(str(self.root / fn))

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]

        img = img.float() / 255.0

        img = F.interpolate(
            img.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        img = (img - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)

        return img, fn


def build_index_split(dataset_size, seed):
    n_val = max(1, dataset_size // 5)
    n_train = dataset_size - n_val

    indices = torch.randperm(
        dataset_size,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    return train_indices, val_indices


def build_teacher(backbone, num_classes=7, pretrained=True):
    if backbone != "efficientnet_b0":
        raise ValueError("Currently supported teacher backbone: efficientnet_b0")

    weights = (
        torchvision.models.EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )

    model = torchvision.models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def fine_tune(model, train_loader, val_loader, epochs, learning_rate, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        model.eval()
        correct, count = 0, 0

        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)

                correct += (logits.argmax(1) == y).sum().item()
                count += x.size(0)

        val_acc = correct / count

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={total_loss / total:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val


@torch.inference_mode()
def dump_soft_labels(model, loader, device, logits_path, filenames_path):
    model.eval()

    all_logits = []
    all_filenames = []

    for x, filenames in loader:
        x = x.to(device)
        logits = model(x).cpu().numpy()

        all_logits.append(logits)
        all_filenames.extend(filenames)

    logits_path = Path(logits_path)
    filenames_path = Path(filenames_path)

    ensure_dir(logits_path.parent)
    ensure_dir(filenames_path.parent)

    np.save(logits_path, np.concatenate(all_logits, axis=0))
    filenames_path.write_text("\n".join(all_filenames) + "\n")

    print(f"Saved teacher logits to {logits_path}")
    print(f"Saved teacher filenames to {filenames_path}")


def main():
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Using device: {device}")

    image_size = cfg["teacher"]["image_size"]
    batch_size = cfg["training"]["batch_size"]

    full_dataset = TeacherLabeledDataset(TRAIN_ROOT, image_size=image_size)
    train_indices, val_indices = build_index_split(
        len(full_dataset),
        seed=cfg["seed"],
    )

    train_ds = TeacherLabeledDataset(
        TRAIN_ROOT,
        image_size=image_size,
        indices=train_indices,
    )

    val_ds = TeacherLabeledDataset(
        TRAIN_ROOT,
        image_size=image_size,
        indices=val_indices,
    )

    loader_kwargs = dataloader_kwargs(cfg.get("data", {}).get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    teacher = build_teacher(
        backbone=cfg["teacher"]["backbone"],
        num_classes=7,
        pretrained=cfg["teacher"]["pretrained"],
    ).to(device)

    checkpoint_path = Path(cfg["outputs"]["checkpoint"])
    ensure_dir(checkpoint_path.parent)

    if checkpoint_path.exists():
        teacher.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded teacher checkpoint from {checkpoint_path}")
    else:
        best_val = fine_tune(
            teacher,
            train_loader,
            val_loader,
            epochs=cfg["training"]["epochs"],
            learning_rate=cfg["training"]["learning_rate"],
            device=device,
        )

        torch.save(teacher.state_dict(), checkpoint_path)
        print(f"Saved teacher checkpoint to {checkpoint_path}")
        print(f"Best teacher val_acc={best_val:.4f}")

    unlabeled = TeacherUnlabeledDataset(
        UNLABELED_ROOT,
        image_size=image_size,
    )

    unlabeled_loader = DataLoader(
        unlabeled,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    dump_soft_labels(
        teacher,
        unlabeled_loader,
        device,
        logits_path=cfg["outputs"]["logits"],
        filenames_path=cfg["outputs"]["filenames"],
    )


if __name__ == "__main__":
    main()