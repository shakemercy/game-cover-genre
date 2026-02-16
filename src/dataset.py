import os
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class TripletDataset(Dataset):
    def __init__(self, root_dir, indices, transform):
        self.samples = []
        self.transform = transform

        root_dir = Path(root_dir)

        class_to_idx = {}
        for idx, cls in enumerate(sorted(os.listdir(root_dir))):
            class_to_idx[cls] = idx

        all_triplets = []

        for cls in class_to_idx:
            cls_dir = root_dir / cls
            files = os.listdir(cls_dir)

            groups = {}

            for f in files:
                if not f.endswith(".jpg"):
                    continue

                base = f.split("_")[0].split(".")[0]

                if base not in groups:
                    groups[base] = []

                groups[base].append(cls_dir / f)

            for base in groups:
                imgs = groups[base]
                if len(imgs) == 3:
                    all_triplets.append((sorted(imgs), class_to_idx[cls]))

        self.all_triplets = [all_triplets[i] for i in indices]

    def __len__(self):
        return len(self.all_triplets)

    def __getitem__(self, idx):
        paths, label = self.all_triplets[idx]

        imgs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            img = self.transform(img)
            imgs.append(img)

        x = torch.cat(imgs, dim=0)
        return x, label


def get_loaders(data_dir, batch_size=32, img_size=224, seed=42):

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    root = Path(data_dir)

    temp_dataset = []
    class_names = sorted(os.listdir(root))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    for cls in class_names:
        cls_dir = root / cls
        files = os.listdir(cls_dir)

        groups = {}

        for f in files:
            if not f.endswith(".jpg"):
                continue

            base = f.split("_")[0].split(".")[0]

            if base not in groups:
                groups[base] = []

            groups[base].append(cls_dir / f)

        for base in groups:
            imgs = groups[base]
            if len(imgs) == 3:
                temp_dataset.append((sorted(imgs), class_to_idx[cls]))

    rng = random.Random(seed)
    by_class = {}

    for i, (_, y) in enumerate(temp_dataset):
        if y not in by_class:
            by_class[y] = []
        by_class[y].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for y in by_class:
        inds = by_class[y]
        rng.shuffle(inds)
        n = len(inds)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        train_idx.extend(inds[:n_train])
        val_idx.extend(inds[n_train:n_train + n_val])
        test_idx.extend(inds[n_train + n_val:])

    train_ds = TripletDataset(data_dir, train_idx, train_tf)
    val_ds = TripletDataset(data_dir, val_idx, eval_tf)
    test_ds = TripletDataset(data_dir, test_idx, eval_tf)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, len(class_names), class_names
