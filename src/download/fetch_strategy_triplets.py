from pathlib import Path
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

STEAMSPY_GENRE_URL = "https://steamspy.com/api.php"
STEAM_APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"

def load_state(state_path: Path):
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"done": [], "failed": [], "seen": []}

def save_state(state_path: Path, state: dict):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

def get_json(session: requests.Session, url: str, params: dict, tries: int = 6, base_sleep: float = 0.7):
    last = None
    for i in range(tries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(base_sleep * (2 ** i))
                continue
            if r.status_code >= 500:
                time.sleep(base_sleep * (2 ** i))
                continue
            r.raise_for_status()
            if not r.text:
                return None
            return r.json()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2 ** i))
    raise last

def get_strategy_appids(session: requests.Session):
    data = get_json(session, STEAMSPY_GENRE_URL, {"request": "genre", "genre": "Strategy"}, tries=8)
    if not isinstance(data, dict):
        return []
    appids = []
    for k in data.keys():
        if str(k).isdigit():
            appids.append(int(k))
    return appids

def get_appdetails(session: requests.Session, appid: int):
    data = get_json(session, STEAM_APPDETAILS_URL, {"appids": appid}, tries=8)
    if not isinstance(data, dict):
        return None
    node = data.get(str(appid))
    if not isinstance(node, dict):
        return None
    if not node.get("success"):
        return None
    inner = node.get("data")
    if not isinstance(inner, dict):
        return None
    return inner

def download_file(session: requests.Session, url: str, dst: Path):
    r = session.get(url, stream=True, timeout=60)
    if r.status_code == 429:
        return False
    if r.status_code != 200:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 128):
            if chunk:
                f.write(chunk)
    return True

def make_triplet(session: requests.Session, appid: int, out_dir: Path):
    details = get_appdetails(session, appid)
    if details is None:
        return False

    header = details.get("header_image")
    if not isinstance(header, str) or not header:
        return False

    shots = details.get("screenshots")
    if not isinstance(shots, list) or len(shots) < 2:
        return False

    gp_urls = []
    for s in shots:
        if isinstance(s, dict):
            u = s.get("path_full") or s.get("path_thumbnail")
            if isinstance(u, str) and u:
                gp_urls.append(u)
        if len(gp_urls) >= 2:
            break

    if len(gp_urls) < 2:
        return False

    cover_path = out_dir / f"{appid}.jpg"
    gp1_path = out_dir / f"{appid}_gp1.jpg"
    gp2_path = out_dir / f"{appid}_gp2.jpg"

    if cover_path.exists() and gp1_path.exists() and gp2_path.exists():
        return True

    tmp_cover = out_dir / f"{appid}.tmp_cover"
    tmp_gp1 = out_dir / f"{appid}.tmp_gp1"
    tmp_gp2 = out_dir / f"{appid}.tmp_gp2"

    ok1 = download_file(session, header, tmp_cover)
    ok2 = download_file(session, gp_urls[0], tmp_gp1)
    ok3 = download_file(session, gp_urls[1], tmp_gp2)

    if not (ok1 and ok2 and ok3):
        for p in [tmp_cover, tmp_gp1, tmp_gp2]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        return False

    tmp_cover.replace(cover_path)
    tmp_gp1.replace(gp1_path)
    tmp_gp2.replace(gp2_path)

    return True

def worker(appid: int, out_dir: Path, sleep_min: float, sleep_max: float):
    session = requests.Session()
    ok = make_triplet(session, appid, out_dir)
    time.sleep(random.uniform(sleep_min, sleep_max))
    return appid, ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw/Strategy")
    ap.add_argument("--target-triplets", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--state", default="outputs/strategy_state.json")
    ap.add_argument("--sleep-min", type=float, default=0.10)
    ap.add_argument("--sleep-max", type=float, default=0.45)
    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    state_path = Path(args.state)
    state = load_state(state_path)

    done = set(int(x) for x in state.get("done", []))
    failed = set(int(x) for x in state.get("failed", []))
    seen = set(int(x) for x in state.get("seen", []))

    out_dir.mkdir(parents=True, exist_ok=True)

    base_session = requests.Session()
    all_appids = get_strategy_appids(base_session)

    all_appids = [a for a in all_appids if a not in done]
    random.shuffle(all_appids)

    current_triplets = len(list(out_dir.glob("*_gp2.jpg")))
    print(f"[INFO] Existing triplets in {out_dir}: {current_triplets}")
    print(f"[INFO] Target triplets: {args.target_triplets}")
    print(f"[INFO] Candidate appids (not done): {len(all_appids)}")

    if current_triplets >= args.target_triplets:
        print("[INFO] Already reached target.")
        return

    need = args.target_triplets - current_triplets

    queued = []
    for a in all_appids:
        if a in failed:
            continue
        queued.append(a)
        if len(queued) >= need * 6:
            break

    print(f"[INFO] Queued appids to try: {len(queued)}")

    success = 0
    tried = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for appid in queued:
            seen.add(appid)
            futs.append(ex.submit(worker, appid, out_dir, args.sleep_min, args.sleep_max))

        for fut in as_completed(futs):
            appid, ok = fut.result()
            tried += 1
            if ok:
                done.add(appid)
                success += 1
                print(f"[OK] appid={appid}  success={success}/{need}")
            else:
                failed.add(appid)
                print(f"[NO] appid={appid}  success={success}/{need}")

            state["done"] = sorted(done)
            state["failed"] = sorted(failed)
            state["seen"] = sorted(seen)
            save_state(state_path, state)

            if success >= need:
                print("[INFO] Target reached. You can stop now.")
                break

    print("[DONE]")
    print(f"saved_to={out_dir}")
    print(f"state={state_path}")

if __name__ == "__main__":
    main()
