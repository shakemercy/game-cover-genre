from pathlib import Path
import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

APP_LIST_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"

def load_state(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"done": [], "failed": [], "last_appid": 0}

def save_state(p: Path, state: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")

def get_json(session: requests.Session, url: str, params: dict | None, tries: int = 6, base_sleep: float = 0.8):
    last = None
    for i in range(tries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(base_sleep * (2 ** i))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2 ** i))
    raise last

def get_applist_page(session: requests.Session, key: str, last_appid: int, max_results: int):
    params = {
        "key": key,
        "include_games": "1",
        "include_dlc": "0",
        "include_software": "0",
        "include_videos": "0",
        "include_hardware": "0",
        "max_results": str(max_results),
        "last_appid": str(last_appid),
    }
    data = get_json(session, APP_LIST_URL, params=params, tries=6)
    resp = data.get("response", {})
    apps = resp.get("apps", [])
    out = []
    last = last_appid
    for a in apps:
        appid = a.get("appid")
        if isinstance(appid, int) and appid > 0:
            out.append(appid)
            if appid > last:
                last = appid
    have_more = bool(resp.get("have_more_results"))
    return out, last, have_more

def get_appdetails(session: requests.Session, appid: int):
    data = get_json(session, APPDETAILS_URL, {"appids": appid}, tries=6)
    node = data.get(str(appid))
    if not isinstance(node, dict):
        return None
    if not node.get("success"):
        return None
    inner = node.get("data")
    if not isinstance(inner, dict):
        return None
    return inner

def is_strategy(details: dict):
    if details.get("type") != "game":
        return False
    genres = details.get("genres")
    if not isinstance(genres, list):
        return False
    for g in genres:
        if isinstance(g, dict):
            d = g.get("description")
            if isinstance(d, str) and d.lower() == "strategy":
                return True
    return False

def pick_two_screenshots(details: dict):
    shots = details.get("screenshots")
    if not isinstance(shots, list) or len(shots) < 2:
        return None
    urls = []
    for s in shots:
        if isinstance(s, dict):
            u = s.get("path_full")
            if isinstance(u, str) and u:
                urls.append(u)
        if len(urls) >= 2:
            break
    if len(urls) < 2:
        return None
    return urls[0], urls[1]

def download_file(session: requests.Session, url: str, dst: Path):
    r = session.get(url, stream=True, timeout=90)
    if r.status_code != 200:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 128):
            if chunk:
                f.write(chunk)
    return True

def try_make_triplet(appid: int, out_dir: Path, sleep_min: float, sleep_max: float):
    session = requests.Session()

    cover = out_dir / f"{appid}.jpg"
    gp1 = out_dir / f"{appid}_gp1.jpg"
    gp2 = out_dir / f"{appid}_gp2.jpg"

    if cover.exists() and gp1.exists() and gp2.exists():
        time.sleep(random.uniform(sleep_min, sleep_max))
        return appid, True, "already"

    details = get_appdetails(session, appid)
    if details is None:
        time.sleep(random.uniform(sleep_min, sleep_max))
        return appid, False, "no_details"

    if not is_strategy(details):
        time.sleep(random.uniform(sleep_min, sleep_max))
        return appid, False, "not_strategy"

    header = details.get("header_image")
    if not isinstance(header, str) or not header:
        time.sleep(random.uniform(sleep_min, sleep_max))
        return appid, False, "no_header"

    pair = pick_two_screenshots(details)
    if pair is None:
        time.sleep(random.uniform(sleep_min, sleep_max))
        return appid, False, "no_screens"

    tmp_cover = out_dir / f"{appid}.tmp_cover"
    tmp_gp1 = out_dir / f"{appid}.tmp_gp1"
    tmp_gp2 = out_dir / f"{appid}.tmp_gp2"

    ok1 = download_file(session, header, tmp_cover)
    ok2 = download_file(session, pair[0], tmp_gp1)
    ok3 = download_file(session, pair[1], tmp_gp2)

    if not (ok1 and ok2 and ok3):
        for p in [tmp_cover, tmp_gp1, tmp_gp2]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        time.sleep(random.uniform(sleep_min, sleep_max))
        return appid, False, "dl_fail"

    tmp_cover.replace(cover)
    tmp_gp1.replace(gp1)
    tmp_gp2.replace(gp2)

    time.sleep(random.uniform(sleep_min, sleep_max))
    return appid, True, "ok"

def count_triplets(out_dir: Path):
    return len(list(out_dir.glob("*_gp2.jpg")))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw/Strategy")
    ap.add_argument("--target-triplets", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--state", default="outputs/strategy_state_v2.json")
    ap.add_argument("--sleep-min", type=float, default=0.10)
    ap.add_argument("--sleep-max", type=float, default=0.35)
    ap.add_argument("--page-size", type=int, default=50000)
    ap.add_argument("--pages", type=int, default=6)
    args = ap.parse_args()

    key = os.environ.get("STEAM_WEB_API_KEY", "").strip()
    if not key:
        raise SystemExit("Missing STEAM_WEB_API_KEY environment variable")

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state)
    state = load_state(state_path)

    done = set(int(x) for x in state.get("done", []))
    failed = set(int(x) for x in state.get("failed", []))
    last_appid = int(state.get("last_appid", 0))

    existing = count_triplets(out_dir)
    print(f"[INFO] Existing triplets in {out_dir}: {existing}")
    print(f"[INFO] Target triplets: {args.target_triplets}")

    if existing >= args.target_triplets:
        print("[INFO] Already reached target.")
        return

    base = requests.Session()

    all_appids = []
    have_more = True
    pages = 0
    while have_more and pages < args.pages:
        page, new_last, have_more = get_applist_page(base, key, last_appid, args.page_size)
        last_appid = new_last
        pages += 1
        all_appids.extend(page)

        state["last_appid"] = last_appid
        state["done"] = sorted(done)
        state["failed"] = sorted(failed)
        save_state(state_path, state)

        print(f"[INFO] applist page={pages} got={len(page)} last_appid={last_appid} have_more={have_more}")

    all_appids = [a for a in all_appids if a not in done and a not in failed]
    random.shuffle(all_appids)

    need = args.target_triplets - existing
    print(f"[INFO] Candidate appids: {len(all_appids)}")
    print(f"[INFO] Need new triplets: {need}")

    success = 0
    tried = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(try_make_triplet, a, out_dir, args.sleep_min, args.sleep_max) for a in all_appids]

        for fut in as_completed(futs):
            appid, ok, reason = fut.result()
            tried += 1

            if ok:
                done.add(appid)
                if reason != "already":
                    success += 1
                now = existing + success
                print(f"[OK] appid={appid} {reason} {now}/{args.target_triplets}")
            else:
                if reason != "not_strategy":
                    failed.add(appid)
                print(f"[NO] appid={appid} {reason} +{success}")

            state["done"] = sorted(done)
            state["failed"] = sorted(failed)
            state["last_appid"] = last_appid
            save_state(state_path, state)

            if existing + success >= args.target_triplets:
                print("[INFO] Target reached.")
                break

            if tried % 500 == 0:
                now = existing + success
                print(f"[INFO] progress tried={tried} new_triplets={success} total_triplets={now}")

    final = count_triplets(out_dir)
    print("[DONE]")
    print(f"triplets_now={final}")
    print(f"saved_to={out_dir}")
    print(f"state={state_path}")

if __name__ == "__main__":
    main()
