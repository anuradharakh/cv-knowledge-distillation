from pathlib import Path

import csv
import shutil

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.datasets import ImageDataset, UnlabeledWithSoftLabels
from src.models import build_student
from src.utils import (
    count_params,
    dataloader_kwargs,
    ensure_dir,
    get_device,
    load_config,
    set_seed,
)


TRAIN_ROOT = Path("train")
UNLABELED_ROOT = Path("unlabeled")
STUDENT_CONFIG_PATH = Path("configs/student.yml")
DISTILL_CONFIG_PATH = Path("configs/distill.yml")
TEACHER_CONFIG_PATH = Path("configs/teacher.yml")


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


def train_step(
    student,
    x_lab,
    y_lab,
    x_un,
    teacher_logits,
    optimizer,
    temperature,
    alpha,
):
    student.train()

    student_lab_logits = student(x_lab)
    ce_loss = F.cross_entropy(student_lab_logits, y_lab)

    student_un_logits = student(x_un)
    kd_loss = F.kl_div(
        F.log_softmax(student_un_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)

    loss = (1 - alpha) * ce_loss + alpha * kd_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), ce_loss.item(), kd_loss.item()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")

        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return correct / total, loss_sum / total


def save_scripted_model(model, path):
    path = Path(path)
    ensure_dir(path.parent)

    scripted = torch.jit.script(model.cpu().eval())
    torch.jit.save(scripted, path)

    print(f"Saved model to {path}")


def run_distillation(temperature, save_model=True):
    student_cfg = load_config(STUDENT_CONFIG_PATH)
    distill_cfg = load_config(DISTILL_CONFIG_PATH)
    teacher_cfg = load_config(TEACHER_CONFIG_PATH)

    set_seed(distill_cfg["seed"])
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

    full_dataset = ImageDataset(TRAIN_ROOT)
    train_indices, val_indices = build_index_split(
        len(full_dataset),
        seed=distill_cfg["seed"],
    )

    train_ds = ImageDataset(
        TRAIN_ROOT,
        indices=train_indices,
        transform=train_transform,
    )

    val_ds = ImageDataset(
        TRAIN_ROOT,
        indices=val_indices,
        transform=None,
    )

    loader_kwargs = dataloader_kwargs(
        distill_cfg.get("data", {}).get("num_workers", 0)
    )

    lab_loader = DataLoader(
        train_ds,
        batch_size=distill_cfg["training"]["batch_size_labeled"],
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=distill_cfg["training"]["batch_size_labeled"],
        shuffle=False,
        **loader_kwargs,
    )

    unlabeled = UnlabeledWithSoftLabels(
        UNLABELED_ROOT,
        filenames_path=teacher_cfg["outputs"]["filenames"],
        soft_labels_path=teacher_cfg["outputs"]["logits"],
    )

    un_loader = DataLoader(
        unlabeled,
        batch_size=distill_cfg["training"]["batch_size_unlabeled"],
        shuffle=True,
        **loader_kwargs,
    )

    student = build_student(
        num_classes=student_cfg["model"]["num_classes"],
        image_size=student_cfg["model"]["image_size"],
        dropout=student_cfg["model"]["dropout"],
        channels=student_cfg["model"].get("channels"),
    ).to(device)

    n_params = count_params(student)
    print(f"Student parameters: {n_params:,}")

    assert n_params < student_cfg["model"]["max_params"], (
        f"Model has {n_params:,} parameters, over limit."
    )

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=distill_cfg["training"]["learning_rate"],
        weight_decay=distill_cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=distill_cfg["training"]["epochs"],
    )

    alpha = distill_cfg["distillation"]["alpha"]

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, distill_cfg["training"]["epochs"] + 1):
        un_iter = iter(un_loader)

        total_loss, ce_sum, kd_sum, steps = 0.0, 0.0, 0.0, 0

        for x_lab, y_lab in lab_loader:
            try:
                x_un, teacher_logits = next(un_iter)
            except StopIteration:
                un_iter = iter(un_loader)
                x_un, teacher_logits = next(un_iter)

            x_lab = x_lab.to(device)
            y_lab = y_lab.to(device)
            x_un = x_un.to(device)
            teacher_logits = teacher_logits.to(device)

            loss, ce_loss, kd_loss = train_step(
                student,
                x_lab,
                y_lab,
                x_un,
                teacher_logits,
                optimizer,
                temperature,
                alpha,
            )

            total_loss += loss
            ce_sum += ce_loss
            kd_sum += kd_loss
            steps += 1

        scheduler.step()

        val_acc, val_loss = evaluate(student, val_loader, device)

        print(
            f"T={temperature} "
            f"Epoch {epoch:03d} "
            f"loss={total_loss / steps:.4f} "
            f"ce={ce_sum / steps:.4f} "
            f"kd={kd_sum / steps:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in student.state_dict().items()
            }

    if best_state is not None:
        student.load_state_dict(best_state)

    checkpoint_name = f"distilled_T{temperature:g}_alpha{alpha:g}.pt"
    checkpoint_path = Path("outputs/checkpoints") / checkpoint_name

    save_scripted_model(student, checkpoint_path)

    if save_model:
        final_model_path = Path(distill_cfg["outputs"]["model"])
        shutil.copyfile(checkpoint_path, final_model_path)
        print(f"Copied best/current experiment model to {final_model_path}")

    return {
        "temperature": temperature,
        "alpha": alpha,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "params": n_params,
        "model_path": str(checkpoint_path),
    }


def write_results_csv(results, path):
    path = Path(path)
    ensure_dir(path.parent)

    fieldnames = [
        "temperature",
        "alpha",
        "best_val_acc",
        "best_val_loss",
        "params",
        "model_path",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved sweep results to {path}")


def main():
    distill_cfg = load_config(DISTILL_CONFIG_PATH)

    temperatures = distill_cfg["distillation"].get("temperature_sweep")

    if temperatures:
        results = []

        selected_temperature = float(distill_cfg["distillation"]["temperature"])

        for temperature in temperatures:
            temperature = float(temperature)

            result = run_distillation(
                temperature=temperature,
                save_model=(temperature == selected_temperature),
            )

            results.append(result)

        write_results_csv(results, distill_cfg["outputs"]["results_csv"])
    else:
        result = run_distillation(
            temperature=float(distill_cfg["distillation"]["temperature"]),
            save_model=True,
        )

        write_results_csv(
            [result],
            distill_cfg["outputs"]["results_csv"],
        )


if __name__ == "__main__":
    main()