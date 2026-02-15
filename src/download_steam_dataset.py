import time
import re
import json
import random
from pathlib import Path

import requests

# =========================
# PODESAVANJA
# =========================
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_DIR = Path("data/splits")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "download_state.json"
APPLIST_CACHE = STATE_DIR / "steamspy_all.json"

# Koliko covera želiš ukupno (npr. 5000–10000)
TARGET_IMAGES = 9000

# Koliko appid kandidata da uzmeš (uzmi mnogo više nego TARGET_IMAGES zbog miss-ova)
MAX_APPIDS_TO_TRY = 80000

# Ograničenje po žanru (da ne bude sve Action)
MAX_PER_GENRE = 1500

# Pauza između zahteva (smanji za brže, povećaj ako dobijaš 429)
SLEEP_BASE = 0.08  # 0.05–0.15 je ok

# Steam endpoints
STEAMSPY_URL = "https://steamspy.com/api.php"

# Cover URL-ovi (prvi je “lep poster”, drugi je fallback)
COVER_URLS = [
    "https://steamcdn-a.akamaihd.net/steam/apps/{appid}/library_600x900_2x.jpg",
    "https://steamcdn-a.akamaihd.net/steam/apps/{appid}/header.jpg",
]

# =========================
# HELPERS
# =========================
def safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:80] if name else "Unknown"


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done_appids": [], "per_genre": {}, "downloaded": 0}


def save_state(state: dict) -> None:
    # držimo done_appids u listi radi JSON-a
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

APPIDS_CACHE = STATE_DIR / "appids_cache.json"

def get_many_appids(limit: int) -> list[int]:
    print("Generating appid range candidates...")

    # Steam appid range (realne igre su uglavnom 1 do ~2000000)
    start = 1
    end = 2000000

    candidates = list(range(start, end))
    random.shuffle(candidates)

    print(f"Generated {len(candidates)} possible appids.")
    return candidates[:limit]


def get_json_with_retry(url: str, params: dict | None = None, tries: int = 6) -> dict | None:
    for t in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)

            # 429 / rate limit
            if r.status_code == 429:
                time.sleep(2.0 + t * 2.0 + random.random())
                continue

            # server error
            if r.status_code >= 500:
                time.sleep(1.0 + t * 1.5 + random.random())
                continue

            # bilo šta osim 200, preskoči
            if r.status_code != 200:
                time.sleep(0.5 + t * 0.8 + random.random())
                continue

            # Ako nije JSON (SteamSpy ponekad vrati HTML/prazno)
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "json" not in ctype:
                # često znači da su te privremeno blokirali / vratili HTML
                time.sleep(1.5 + t * 1.5 + random.random())
                continue

            text = r.text.strip()
            if not text:
                time.sleep(1.0 + t * 1.2 + random.random())
                continue

            return r.json()

        except Exception:
            time.sleep(1.5 + t * 1.5 + random.random())

    print(f"[get_json_with_retry] Failed url={url} params={params}")
    return None



def get_app_details_steamspy(appid: int) -> dict:
    params = {"request": "appdetails", "appid": appid}
    data = get_json_with_retry(STEAMSPY_URL, params=params, tries=6)
    return data or {}


def choose_main_genre(details: dict) -> str | None:
    # SteamSpy vraća npr. "genre": "Action, Adventure"
    genre = details.get("genre") or ""
    if not genre.strip():
        return None
    main = genre.split(",")[0].strip()
    return safe_name(main) if main else None


def download_cover(appid: int, out_path: Path) -> bool:
    # probaj više URL-ova (poster pa header)
    for tpl in COVER_URLS:
        url = tpl.format(appid=appid)
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.content and len(r.content) > 10_000:
                out_path.write_bytes(r.content)
                return True
        except Exception:
            pass
    return False


# =========================
# MAIN
# =========================
def main():
    state = load_state()
    done_set = set(state.get("done_appids", []))
    per_genre = state.get("per_genre", {})
    downloaded = int(state.get("downloaded", 0))

    print(f"Resume state: downloaded={downloaded}, done_appids={len(done_set)}, genres={len(per_genre)}")
    print(f"Target: {TARGET_IMAGES} images | Max candidates: {MAX_APPIDS_TO_TRY} | Max/genre: {MAX_PER_GENRE}")

    appids = get_many_appids(MAX_APPIDS_TO_TRY)

    for i, appid in enumerate(appids, start=1):
        if downloaded >= TARGET_IMAGES:
            break
        if appid in done_set:
            continue

        # označi kao “obrađen” odmah, da ne vrti isti ako pukne
        done_set.add(appid)

        try:
            details = get_app_details_steamspy(appid)
            genre = choose_main_genre(details)

            if genre is None:
                continue

            per_genre.setdefault(genre, 0)
            if per_genre[genre] >= MAX_PER_GENRE:
                continue

            genre_dir = OUT_DIR / genre
            genre_dir.mkdir(parents=True, exist_ok=True)
            out_path = genre_dir / f"{appid}.jpg"

            if out_path.exists():
                continue

            ok = download_cover(appid, out_path)
            if ok:
                per_genre[genre] += 1
                downloaded += 1
                print(f"[{i}/{len(appids)}] OK  appid={appid} genre={genre}  total={downloaded}/{TARGET_IMAGES}")
            else:
                # nema cover asset
                pass

        except Exception as e:
            print(f"[{i}/{len(appids)}] ERROR appid={appid}: {e}")

        # snimi state povremeno (na svakih 50 pokušaja)
        if i % 50 == 0:
            state["done_appids"] = list(done_set)
            state["per_genre"] = per_genre
            state["downloaded"] = downloaded
            save_state(state)

        time.sleep(SLEEP_BASE + random.random() * 0.04)

    # final save
    state["done_appids"] = list(done_set)
    state["per_genre"] = per_genre
    state["downloaded"] = downloaded
    save_state(state)

    print("\nDONE")
    print(f"Downloaded: {downloaded}")
    print("Top genres:")
    for g, c in sorted(per_genre.items(), key=lambda x: -x[1])[:20]:
        print(f"  {g}: {c}")


if __name__ == "__main__":
    main()
