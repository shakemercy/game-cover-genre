import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("data/balanced_raw")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def count_images(folder: Path):
    n = 0
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            n += 1
    return n

class_dirs = []
for p in ROOT.iterdir():
    if p.is_dir():
        class_dirs.append(p)
class_dirs.sort(key=str)

names = []
counts = []

for cdir in class_dirs:
    names.append(cdir.name)
    counts.append(count_images(cdir))

csv_path = OUT_DIR / "ex_class_dist.csv"
f = csv_path.open("w", newline="", encoding="utf-8")
w = csv.writer(f)
w.writerow(["class", "images"])
for i in range(len(names)):
    w.writerow([names[i], counts[i]])
f.close()

plt.figure(figsize=(max(10, len(names) * 0.6), 6))
plt.bar(names, counts)
plt.xticks(rotation=45, ha="right")
plt.ylabel("images")
plt.title("Class distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "ex_class_dist.png", dpi=160)
plt.close()

split_csv = OUT_DIR / "split_dist.csv"
if split_csv.exists():
    split_names = []
    train = []
    val = []
    test = []

    f2 = split_csv.open("r", encoding="utf-8")
    r = csv.reader(f2)
    header = next(r, None)
    for row in r:
        if len(row) >= 4:
            split_names.append(row[0])
            train.append(int(row[1]))
            val.append(int(row[2]))
            test.append(int(row[3]))
    f2.close()

    x = list(range(len(split_names)))

    plt.figure(figsize=(max(10, len(split_names) * 0.6), 6))
    plt.bar(x, train, label="train")
    bottom1 = train[:]
    plt.bar(x, val, bottom=bottom1, label="val")
    bottom2 = []
    for i in range(len(train)):
        bottom2.append(train[i] + val[i])
    plt.bar(x, test, bottom=bottom2, label="test")

    plt.xticks(x, split_names, rotation=45, ha="right")
    plt.ylabel("images")
    plt.title("Split distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ex_split_dist.png", dpi=160)
    plt.close()

print("DONE:", OUT_DIR)
