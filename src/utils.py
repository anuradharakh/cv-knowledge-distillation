import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def dataloader_kwargs(num_workers=0):
    return {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }