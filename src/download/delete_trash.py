import re
from pathlib import Path

BASE_DIR = Path("data/raw")
DO_DELETE = False

exts = {".jpg", ".jpeg", ".png", ".webp"}

cover_files = []
gp1_bases = set()
gp2_bases = set()

gp_re = re.compile(r"^(?P<base>.+?)_gp(?P<n>[12])(?:_\d+)?$", re.IGNORECASE)

all_files = [p for p in BASE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts]

for p in all_files:
    stem = p.stem
    m = gp_re.match(stem)
    if m:
        base = m.group("base")
        if m.group("n") == "1":
            gp1_bases.add(base)
        else:
            gp2_bases.add(base)
    else:
        cover_files.append(p)

to_delete = []
for cover in cover_files:
    base = cover.stem
    if base not in gp1_bases or base not in gp2_bases:
        to_delete.append(cover)

print(f"BASE_DIR: {BASE_DIR.resolve()}")
print(f"Ukupno slika: {len(all_files)}")
print(f"Covers: {len(cover_files)} | gp1 baze: {len(gp1_bases)} | gp2 baze: {len(gp2_bases)}")
print(f"Za brisanje covera: {len(to_delete)}")

if not DO_DELETE:
    for p in to_delete[:30]:
        print(f"DRY {p}")
    if len(to_delete) > 30:
        print(f"... jos {len(to_delete) - 30}")
else:
    deleted = 0
    for p in to_delete:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"FAIL {p} -> {e}")
    print(f"Obrisano covera: {deleted}")
