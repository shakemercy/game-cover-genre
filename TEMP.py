from pathlib import Path
import random
from torchvision import datasets
import csv

ROOT = Path("data/balanced_raw")
CSV_PATH = Path("outputs/split_dist.csv")

random.seed(42)

base = datasets.ImageFolder(ROOT)

class_inds = []
for _, x in base.samples:
    class_inds.append(x)

class_names = base.classes

by_class = {}
for i in range(len(class_inds)):
    y = class_inds[i]
    if y not in by_class:
        by_class[y] = []
    by_class[y].append(i)

train_ind = []
val_ind = []
test_ind = []

for ind in by_class:
    inds = by_class[ind]

    random.shuffle(inds)

    n = len(inds)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_ind.extend(inds[:n_train])
    val_ind.extend(inds[n_train:n_train + n_val])
    test_ind.extend(inds[n_train + n_val:])

def count(indexes):
    counts = [0] * len(class_names)
    for i in indexes:
        c = class_inds[i]
        counts[c] += 1
    return counts

train_ct = count(train_ind)
val_ct = count(val_ind)
test_ct = count(test_ind)

with CSV_PATH.open("w", newline = "") as f:
    w = csv.writer(f)
    w.writerow(["class", "train", "val", "test", "total"])
    for i in range(len(class_names)):
        ct = train_ct[i] + val_ct[i] + test_ct[i]
        w.writerow((class_names[i], train_ct[i], val_ct[i], test_ct[i], ct))

print("NAPRAVLJEN -> ", CSV_PATH)