import csv
import random
import re
import shutil
from pathlib import Path

IN_DIR = Path("data/raw")
OUT_DIR = Path("data/balanced_raw")
MAX_GAMES_PER_CLASS = 1000
SEED = 42

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
OUT_CSV = Path("outputs/sub_dist.csv")

random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

pat = re.compile(r"^(\d+)(?:_(gp[12]))?$", re.IGNORECASE)

def parse_file(p: Path):
    m = pat.match(p.stem)
    if not m:
        return None
    game_id = m.group(1)
    kind = m.group(2) or "cover"
    ext = p.suffix.lower()
    return game_id, kind.lower(), ext

rows = []

class_dirs = [d for d in IN_DIR.iterdir() if d.is_dir()]
class_dirs.sort(key=lambda x: str(x))

for class_dir in class_dirs:
    games = {}
    for p in class_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        info = parse_file(p)
        if info is None:
            continue
        game_id, kind, ext = info
        games.setdefault(game_id, {})
        if kind not in games[game_id]:
            games[game_id][kind] = p

    game_ids = list(games.keys())
    game_ids.sort()
    random.shuffle(game_ids)

    kept_ids = game_ids[:MAX_GAMES_PER_CLASS]

    target_dir = OUT_DIR / class_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)

    kept_images = 0
    for gid in kept_ids:
        items = games[gid]
        for kind in ["cover", "gp1", "gp2"]:
            if kind not in items:
                continue
            src = items[kind]
            ext = src.suffix.lower()
            if kind == "cover":
                dst_name = f"{gid}{ext}"
            else:
                dst_name = f"{gid}_{kind}{ext}"
            dst = target_dir / dst_name
            shutil.copy2(src, dst)
            kept_images += 1

    total_images = sum(len(games[k]) for k in games.keys())
    rows.append((class_dir.name, len(game_ids), len(kept_ids), total_images, kept_images))

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["class", "total_games", "kept_games", "total_images", "kept_images"])
    for r in rows:
        w.writerow(r)

print("DONE:", OUT_DIR)
print("WROTE:", OUT_CSV)
