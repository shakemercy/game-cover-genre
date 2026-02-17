from pathlib import Path
import random
import re
import shutil
import csv

INPUT = Path("data/raw")
OUTPUT = Path("data/balanced_raw")
CSV_PATH = Path("outputs/sub_dist.csv")

random.seed(42)
OUTPUT.mkdir(exist_ok = True)
CSV_PATH.parent.mkdir(exist_ok = True)

MAX_GAMES = 1000

regex = re.compile(r"^(\d+)_?(gp[12])?$")

rows = []

classes = [d for d in INPUT.iterdir() if d.is_dir()]
classes.sort(key = lambda x : str(x))

def parse(img: Path):
    m = regex.match(img.stem)
    if not m:
        return None
    game_id = m.group(1)
    img_type = m.group(2) or "main"
    ext = img.suffix
    return game_id, img_type, ext

for class_dir in classes:
    games = {}
    for img in class_dir.iterdir():
        if not img.is_file() or img.suffix != ".jpg":
            continue
        game = parse(img)
        if game is None:
            continue
        game_id, img_type, ext = game
        if game_id not in games:
            games[game_id] = {}
        games[game_id][img_type] = img

    game_ids = list(games.keys())
    game_ids.sort()
    random.shuffle(game_ids)

    selected = game_ids[:MAX_GAMES]

    output = OUTPUT / class_dir.name
    output.mkdir(exist_ok = True)

    img_counter = 0
    for gid in selected:
        images = games[gid]
        for img_type in ["main", "gp1", "gp2"]:
            if img_type not in images:
                continue
            src = images[img_type]
            ext = src.suffix
            if img_type == "main":
                out_name = f"{gid}{ext}"
            else:
                out_name = f"{gid}_{img_type}{ext}"
            out = output / out_name
            shutil.copy(src, out)
            img_counter += 1
    
    img_total = sum(len(games[k]) for k in games.keys())
    rows.append((class_dir.name, len(game_ids), len(selected), img_total, img_counter))

with CSV_PATH.open("w", newline = "") as f:
    w = csv.writer(f)
    w.writerow(["class", "games_all", "games_selected", "images_all", "images_selected"])
    for r in rows:
        w.writerow(r)

print("NAPRAVLJEN -> ", OUTPUT)
print("DIST FAJL -> ", CSV_PATH)