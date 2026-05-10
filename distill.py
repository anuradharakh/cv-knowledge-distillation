"""
=============================================================================
ML2 Homework 2 — distill.py
=============================================================================
Offline response-based logit distillation.

This script trains the deployable SmallCNN student using:
  1. Hard labels from train/labels.csv with cross-entropy.
  2. Cached teacher logits on unlabeled images with temperature-scaled KL loss.

Type of distillation:
  - Knowledge distillation
  - Response-based / logit-based distillation
  - Offline distillation, because the teacher is trained first and its logits
    are cached before student training.

Run teacher first:
  python train_teacher.py

Then train one distilled student:
  python distill.py --temperature 4 --alpha 0.7

Temperature sweep for the report:
  python distill.py --sweep-temperatures 1 2 4 8

Output:
  model.pt — TorchScript student model only. The teacher is not submitted.
=============================================================================
"""
from pathlib import Path
import argparse
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from train_baby import (
    ImageDataset,
    read_rgb_image,
    Preprocess,
    SmallCNN,
    count_params,
    save_torchscript,
    PARAM_LIMIT,
)

TRAIN_ROOT = Path(__file__).parent / "train"
UNLABELED_ROOT = Path(__file__).parent / "unlabeled"
SOFT_LABELS = Path(__file__).parent / "teacher_soft_labels.npy"
FILENAMES = Path(__file__).parent / "teacher_filenames.txt"
NUM_CLASSES = 7


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class UnlabeledWithSoftLabels(Dataset):
    def __init__(self, root: Path, filenames_path: Path, soft_labels_path: Path, augment: bool = True):
        if not filenames_path.exists() or not soft_labels_path.exists():
            raise FileNotFoundError(
                "Missing teacher_soft_labels.npy or teacher_filenames.txt. Run train_teacher.py first."
            )
        self.root = root
        self.filenames = filenames_path.read_text().strip().splitlines()
        self.logits = np.load(soft_labels_path).astype("float32")
        self.augment = augment
        assert len(self.filenames) == self.logits.shape[0], (
            "filenames and soft-labels length mismatch — rerun train_teacher.py"
        )
        assert self.logits.shape[1] == NUM_CLASSES, f"Expected {NUM_CLASSES} teacher logits per image"

    def __len__(self):
        return len(self.filenames)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            img = torch.flip(img, dims=[2])
        if torch.rand(()) < 0.5:
            brightness = 0.9 + 0.2 * torch.rand(())
            img = (img * brightness).clamp(0.0, 1.0)
        return img

    def __getitem__(self, i):
        fn = self.filenames[i]
        img = read_rgb_image(self.root / fn)
        if self.augment:
            img = self._augment(img)
        logits = torch.from_numpy(self.logits[i])
        return img, logits


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct, total, xent_sum = 0, 0, 0.0
    conf_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        correct += (logits.argmax(1) == y).sum().item()
        conf_sum += probs.max(dim=1).values.sum().item()
        xent_sum += F.cross_entropy(logits, y, reduction="sum").item()
        total += x.size(0)
    return correct / total, xent_sum / total, conf_sum / total


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def train_for_temperature(args, temperature: float, device):
    # Same split as train_baby.py for apples-to-apples validation.
    plain = ImageDataset(TRAIN_ROOT, augment=False)
    n_val = max(1, len(plain) // 5)
    n_train = len(plain) - n_val
    train_ds, val_ds = random_split(
        plain, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    aug = ImageDataset(TRAIN_ROOT, augment=True)
    train_ds.dataset = aug

    unlabeled = UnlabeledWithSoftLabels(UNLABELED_ROOT, FILENAMES, SOFT_LABELS, augment=True)

    lab_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    un_loader = DataLoader(unlabeled, batch_size=args.unlabeled_batch_size, shuffle=True, num_workers=0)

    student = Preprocess(SmallCNN(), size=args.image_size).to(device)
    n_params = count_params(student)
    print(f"Student parameters: {n_params:,}")
    assert n_params < PARAM_LIMIT, f"Over assignment cap: {n_params:,} >= {PARAM_LIMIT:,}"

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = -1.0
    best_state = None
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        student.train()
        un_iter = iter(un_loader)
        loss_sum, ce_sum, kd_sum, steps = 0.0, 0.0, 0.0, 0

        for x_lab, y_lab in lab_loader:
            try:
                x_un, t_logits = next(un_iter)
            except StopIteration:
                un_iter = iter(un_loader)
                x_un, t_logits = next(un_iter)

            x_lab, y_lab = x_lab.to(device), y_lab.to(device)
            x_un, t_logits = x_un.to(device), t_logits.to(device)

            z_lab = student(x_lab)
            z_un = student(x_un)

            loss_ce = F.cross_entropy(z_lab, y_lab, label_smoothing=args.label_smoothing)
            loss_kd = kd_loss(z_un, t_logits, temperature)
            loss = (1.0 - args.alpha) * loss_ce + args.alpha * loss_kd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            opt.step()

            loss_sum += loss.item()
            ce_sum += loss_ce.item()
            kd_sum += loss_kd.item()
            steps += 1

        scheduler.step()
        val_acc, val_loss, val_conf = evaluate(student, val_loader, device)

        if val_acc > best_val:
            best_val = val_acc
            best_metrics = (val_acc, val_loss, val_conf)
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        print(
            f"T={temperature:g} Epoch {epoch:3d}  "
            f"loss={loss_sum / steps:.4f}  ce={ce_sum / steps:.4f}  kd={kd_sum / steps:.4f}  "
            f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}  val_conf={val_conf:.4f}  "
            f"lr={opt.param_groups[0]['lr']:.6f}"
        )

    if best_state is not None:
        student.load_state_dict(best_state)
    assert best_metrics is not None
    return student, best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--sweep-temperatures", type=float, nargs="*", default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unlabeled-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--out", type=str, default="model.pt")
    args = parser.parse_args()

    assert 0.0 <= args.alpha <= 1.0, "alpha must be between 0 and 1"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temperatures = args.sweep_temperatures if args.sweep_temperatures else [args.temperature]
    results = []
    best_student = None
    best_score = -1.0
    best_t = None

    for t in temperatures:
        print("\n" + "=" * 80)
        print(f"Training distilled student with T={t:g}, alpha={args.alpha}")
        print("=" * 80)
        set_seed(args.seed)  # fair comparison across T values
        student, metrics = train_for_temperature(args, t, device)
        val_acc, val_loss, val_conf = metrics
        results.append((t, val_acc, val_loss, val_conf))
        if val_acc > best_score:
            best_score = val_acc
            best_student = student
            best_t = t

    print("\nTemperature sweep summary")
    print("T,val_acc,val_loss,mean_confidence")
    for t, val_acc, val_loss, val_conf in results:
        print(f"{t:g},{val_acc:.4f},{val_loss:.4f},{val_conf:.4f}")

    assert best_student is not None
    print(f"\nBest T={best_t:g} with val_acc={best_score:.4f}. Saving {args.out}")
    save_torchscript(best_student, args.out)


if __name__ == "__main__":
    main()
