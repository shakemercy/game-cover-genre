from pathlib import Path
import csv

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ROOT = Path("data/raw")
OUT = Path("outputs/class_dist.csv")

def img_files(p: Path):
    return [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]

rows = []
total_triplets = 0
total_imgs = 0

for cdir in sorted([d for d in ROOT.iterdir() if d.is_dir()], key=lambda x: x.name.lower()):
    subdirs = [d for d in cdir.iterdir() if d.is_dir()]
    if subdirs:
        triplets = len(subdirs)
        imgs = sum(len(img_files(d)) for d in subdirs)
    else:
        imgs = len(img_files(cdir))
        triplets = imgs // 3

    rows.append((cdir.name, triplets, imgs))
    total_triplets += triplets
    total_imgs += imgs

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["class", "triplets", "image_files"])
    w.writerows(rows)

print(f"Wrote {OUT} | classes={len(rows)} triplets={total_triplets} images={total_imgs}")
