from pathlib import Path
import csv
import random
from torchvision import datasets

ROOT = Path("data/balanced_raw")
OUT = Path("outputs/split_dist.csv")
SEED = 42

random.seed(SEED)

base = datasets.ImageFolder(str(ROOT))
idx_to_class = []
for _, y in base.samples:
    idx_to_class.append(y)

class_names = base.classes

by_class = {}
for i in range(len(idx_to_class)):
    y = idx_to_class[i]
    if y not in by_class:
        by_class[y] = []
    by_class[y].append(i)

train_idx = []
val_idx = []
test_idx = []

for y in by_class:
    inds = by_class[y]
    random.shuffle(inds)

    n = len(inds)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_idx.extend(inds[:n_train])
    val_idx.extend(inds[n_train:n_train + n_val])
    test_idx.extend(inds[n_train + n_val:])

def count(indices):
    counts = [0] * len(class_names)
    for i in indices:
        c = idx_to_class[i]
        counts[c] += 1
    return counts

train_counts = count(train_idx)
val_counts = count(val_idx)
test_counts = count(test_idx)

OUT.parent.mkdir(parents=True, exist_ok=True)
f = OUT.open("w", newline="", encoding="utf-8")
w = csv.writer(f)
w.writerow(["class", "train", "val", "test", "total"])
for k in range(len(class_names)):
    total = train_counts[k] + val_counts[k] + test_counts[k]
    w.writerow([class_names[k], train_counts[k], val_counts[k], test_counts[k], total])
f.close()

print("Wrote", OUT)
print("images", len(base), "train", len(train_idx), "val", len(val_idx), "test", len(test_idx))
