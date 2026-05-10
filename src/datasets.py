from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / "labels.csv")

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
        return img, int(row["label"])


class UnlabeledWithSoftLabels(Dataset):
    def __init__(self, root, filenames_path, soft_labels_path):
        self.root = Path(root)
        self.filenames = Path(filenames_path).read_text().strip().splitlines()
        self.logits = np.load(soft_labels_path)

        assert len(self.filenames) == self.logits.shape[0], (
            "Mismatch between filenames and teacher logits."
        )

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
        logits = torch.tensor(self.logits[i], dtype=torch.float32)
        return img, logits