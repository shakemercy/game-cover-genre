import csv
from pathlib import Path

ROOT = Path("data/raw")
OUT = Path("outputs/class_dist.csv")

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def count_images(folder: Path):
    n = 0
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS:
            n += 1
    return n

rows = []
total_images = 0
total_triplets = 0

class_folders = []
for p in ROOT.iterdir():
    if p.is_dir():
        class_folders.append(p)

class_folders.sort(key=str)

for class_dir in class_folders:
    subfolders = []
    for p in class_dir.iterdir():
        if p.is_dir():
            subfolders.append(p)

    if len(subfolders) > 0:
        imgs = 0
        for sf in subfolders:
            imgs += count_images(sf)
        triplets = len(subfolders)
    else:
        imgs = count_images(class_dir)
        triplets = imgs // 3

    rows.append((class_dir.name, imgs, imgs, triplets))
    total_images += imgs
    total_triplets += triplets

OUT.parent.mkdir(parents=True, exist_ok=True)
f = OUT.open("w", newline="", encoding="utf-8")
w = csv.writer(f)
w.writerow(["class", "total", "kept", "triplets"])
for r in rows:
    w.writerow(r)
f.close()

print("WROTE:", OUT)
print("classes:", len(rows), "images:", total_images, "triplets:", total_triplets)
