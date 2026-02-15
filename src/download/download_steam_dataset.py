import json
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# =========================
# PODESAVANJA
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_DIR = BASE_DIR / "data" / "splits"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "download_state_fast.json"

STEAMSPY_URL = "https://steamspy.com/api.php"

COVER_URLS = [
    "https://steamcdn-a.akamaihd.net/steam/apps/{appid}/library_600x900_2x.jpg",
    "https://steamcdn-a.akamaihd.net/steam/apps/{appid}/header.jpg",
]

# Koliko paralelnih download-a
WORKERS = 32

# Koliko appid kandidata da uzmemo po žanru (uzmi više od "need" zbog failova)
CANDIDATE_MULT = 25

# Male pauze da ne bude agresivno
SLEEP_BETWEEN_GENRES = 0.5

# Ciljane klase i cap (koliko covera po klasi)
GENRE_CAPS = {
    "Action": 1447,
    "Adventure": 832,
    "Racing": 581,
    "RPG": 633,
    "Simulation": 779,
    "Strategy": 598,
}

# =========================
# STATE
# =========================
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"per_genre": {}, "done_appids": []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# =========================
# STEAMSPY
# =========================
def get_json_with_retry(params: dict, tries: int = 6) -> dict:
    for t in range(tries):
        try:
            r = requests.get(STEAMSPY_URL, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(2.0 + t * 2.0 + random.random())
                continue
            if r.status_code >= 500:
                time.sleep(1.0 + t * 1.5 + random.random())
                continue
            if r.status_code != 200:
                time.sleep(0.5 + t * 0.8 + random.random())
                continue
            return r.json() if r.text.strip() else {}
        except Exception:
            time.sleep(1.0 + t * 1.2 + random.random())
    return {}


def get_appids_for_genre(genre: str) -> list[int]:
    # SteamSpy: request=genre&genre=Action
    data = get_json_with_retry({"request": "genre", "genre": genre})
    appids = []
    for k in data.keys():
        if str(k).isdigit():
            appids.append(int(k))
    random.shuffle(appids)
    return appids


# =========================
# DOWNLOAD COVERS
# =========================
def download_cover(appid: int, out_path: Path) -> bool:
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


def cover_task(appid: int, genre: str, done_set: set[int]) -> tuple[int, bool]:
    # ako smo ga već pokušali ranije, preskoči
    if appid in done_set:
        return appid, False

    genre_dir = OUT_DIR / genre
    genre_dir.mkdir(parents=True, exist_ok=True)
    out_path = genre_dir / f"{appid}.jpg"

    if out_path.exists():
        done_set.add(appid)
        return appid, True

    ok = download_cover(appid, out_path)
    done_set.add(appid)
    return appid, ok


def count_existing_from_disk() -> dict:
    # Brojimo koliko već ima .jpg po folderu (da state ne laže)
    per = {}
    for g in GENRE_CAPS.keys():
        d = OUT_DIR / g
        if d.exists() and d.is_dir():
            per[g] = len(list(d.glob("*.jpg")))
        else:
            per[g] = 0
    return per


# =========================
# MAIN
# =========================
def main():
    state = load_state()
    done_set = set(state.get("done_appids", []))

    # Uskladi per_genre sa realnim fajlovima na disku (najjednostavnije i najtačnije)
    per_genre = count_existing_from_disk()

    print("Current counts:")
    for g in GENRE_CAPS:
        print(f"  {g}: {per_genre[g]}/{GENRE_CAPS[g]}")

    for genre, cap in GENRE_CAPS.items():
        need = cap - per_genre.get(genre, 0)
        if need <= 0:
            continue

        print(f"\n[GENRE] {genre} missing {need} covers...")

        appids = get_appids_for_genre(genre)
        # uzmi prvih need*CANDIDATE_MULT kandidata (da imamo buffer za failove)
        candidates = appids[: max(need * CANDIDATE_MULT, need)]

        added = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = [ex.submit(cover_task, appid, genre, done_set) for appid in candidates]

            for fut in as_completed(futures):
                appid, ok = fut.result()
                if ok:
                    per_genre[genre] += 1
                    added += 1

                    if per_genre[genre] % 20 == 0 or per_genre[genre] >= cap:
                        print(f"  {genre}: {per_genre[genre]}/{cap}")

                    if per_genre[genre] >= cap:
                        break

        print(f"[DONE] {genre} added {added}, now {per_genre[genre]}/{cap}")

        # save after each genre
        state["done_appids"] = list(done_set)
        state["per_genre"] = per_genre
        save_state(state)

        time.sleep(SLEEP_BETWEEN_GENRES)

    print("\nALL DONE. Final counts:")
    for g in GENRE_CAPS:
        print(f"  {g}: {per_genre[g]}/{GENRE_CAPS[g]}")


if __name__ == "__main__":
    main()
