from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.datasets import ImageDataset
from src.models import build_student
from src.utils import count_params, dataloader_kwargs, get_device, load_config, set_seed


DATA_ROOT = Path("train")
CONFIG_PATH = Path("configs/student.yml")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return correct / total


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


def main():
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Using device: {device}")

    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
        ),
    ])

    full_dataset = ImageDataset(DATA_ROOT)
    train_indices, val_indices = build_index_split(
        len(full_dataset),
        seed=cfg["seed"],
    )

    train_ds = ImageDataset(
        DATA_ROOT,
        indices=train_indices,
        transform=train_transform,
    )

    val_ds = ImageDataset(
        DATA_ROOT,
        indices=val_indices,
        transform=None,
    )

    loader_kwargs = dataloader_kwargs(cfg.get("data", {}).get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    model = build_student(
        num_classes=cfg["model"]["num_classes"],
        image_size=cfg["model"]["image_size"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    n_params = count_params(model)
    print(f"Student parameters: {n_params:,}")

    assert n_params < cfg["model"]["max_params"], (
        f"Model has {n_params:,} parameters, over limit."
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )

    best_val = -1.0
    best_state = None

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    model_cpu = model.cpu().eval()

    with torch.inference_mode():
        dummy = torch.rand(2, 3, 256, 256)
        out = model_cpu(dummy)
        assert out.shape == (2, cfg["model"]["num_classes"])

    scripted = torch.jit.script(model_cpu)
    torch.jit.save(scripted, "model.pt")

    print(f"Best val_acc={best_val:.4f}")
    print("Saved model.pt")


if __name__ == "__main__":
    main()