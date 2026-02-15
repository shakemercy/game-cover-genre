from pathlib import Path
import csv
import torch
from torchvision import datasets

ROOT = Path("data/raw")
OUT = Path("outputs/split_dist.csv")
SEED = 42

base = datasets.ImageFolder(str(ROOT))
n = len(base)

n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val

g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(n, generator=g).tolist()

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

idx_to_class = [y for _, y in base.samples]
class_names = base.classes

def count_classes(indices):
    counts = [0] * len(class_names)
    for i in indices:
        counts[idx_to_class[i]] += 1
    return counts

train_counts = count_classes(train_idx)
val_counts = count_classes(val_idx)
test_counts = count_classes(test_idx)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["class", "train", "val", "test", "total"])
    for k, name in enumerate(class_names):
        total = train_counts[k] + val_counts[k] + test_counts[k]
        w.writerow([name, train_counts[k], val_counts[k], test_counts[k], total])

print(f"Wrote {OUT} | images={n} train={n_train} val={n_val} test={n_test}")
