from pathlib import Path

import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.datasets import ImageDataset, UnlabeledWithSoftLabels
from src.models import build_student
from src.utils import count_params, ensure_dir, get_device, load_config, set_seed


TRAIN_ROOT = Path("train")
UNLABELED_ROOT = Path("unlabeled")
STUDENT_CONFIG_PATH = Path("configs/student.yml")
DISTILL_CONFIG_PATH = Path("configs/distill.yml")
TEACHER_CONFIG_PATH = Path("configs/teacher.yml")


def train_step(student, x_lab, y_lab, x_un, teacher_logits, optimizer, temperature, alpha):
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


def run_distillation(temperature, save_model=True):
    student_cfg = load_config(STUDENT_CONFIG_PATH)
    distill_cfg = load_config(DISTILL_CONFIG_PATH)
    teacher_cfg = load_config(TEACHER_CONFIG_PATH)

    set_seed(distill_cfg["seed"])
    device = get_device()

    labeled = ImageDataset(TRAIN_ROOT)

    n_val = max(1, len(labeled) // 5)
    n_train = len(labeled) - n_val

    train_ds, val_ds = random_split(
        labeled,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(distill_cfg["seed"]),
    )

    lab_loader = DataLoader(
        train_ds,
        batch_size=distill_cfg["training"]["batch_size_labeled"],
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=distill_cfg["training"]["batch_size_labeled"],
        shuffle=False,
        num_workers=0,
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
        num_workers=0,
    )

    student = build_student(
        num_classes=student_cfg["model"]["num_classes"],
        image_size=student_cfg["model"]["image_size"],
        dropout=student_cfg["model"]["dropout"],
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

    if save_model:
        model_path = Path(distill_cfg["outputs"]["model"])
        scripted = torch.jit.script(student.cpu().eval())
        torch.jit.save(scripted, model_path)
        print(f"Saved distilled student to {model_path}")

    return {
        "temperature": temperature,
        "alpha": alpha,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "params": n_params,
    }


def write_results_csv(results, path):
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["temperature", "alpha", "best_val_acc", "best_val_loss", "params"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved sweep results to {path}")


def main():
    distill_cfg = load_config(DISTILL_CONFIG_PATH)

    temperatures = distill_cfg["distillation"].get("temperature_sweep")

    if temperatures:
        results = []

        for temperature in temperatures:
            result = run_distillation(
                temperature=float(temperature),
                save_model=(float(temperature) == float(distill_cfg["distillation"]["temperature"])),
            )
            results.append(result)

        write_results_csv(results, distill_cfg["outputs"]["results_csv"])
    else:
        run_distillation(
            temperature=float(distill_cfg["distillation"]["temperature"]),
            save_model=True,
        )


if __name__ == "__main__":
    main()