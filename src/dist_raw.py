from pathlib import Path
import csv

ROOT = Path("data/raw")
CSV_PATH = Path("outputs/raw_dist.csv")

rows = []
total_images = 0
total_games = 0

class_folders = [p for p in ROOT.iterdir() if p.is_dir()]
class_folders.sort(key=str)

for class_dir in class_folders:
    images = 0
    for img in class_dir.iterdir():
        if img.is_file() and img.suffix == ".jpg":
            images += 1

    games = images // 3

    rows.append((class_dir.name, images, games))
    total_images += images
    total_games += games

with CSV_PATH.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "images", "games"])
    for r in rows:
        w.writerow(r)

print("NAPRAVLJEN -> ", CSV_PATH)
print("classes:", len(rows), "images:", total_images, "games:", total_games)
