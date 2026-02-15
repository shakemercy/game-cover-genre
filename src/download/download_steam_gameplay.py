#!/usr/bin/env python3
"""
Download gameplay screenshots for already-downloaded Steam game covers.

Default behavior:
- Scans covers in --covers-dir (e.g. data/raw/<GENRE>/<APPID>.jpg)
- For each APPID, downloads N screenshots from Steam Store API (appdetails)
- Saves them into the SAME genre folder:
    data/raw/<GENRE>/<APPID>_gp1.jpg
    data/raw/<GENRE>/<APPID>_gp2.jpg
    ...

Speed-ups:
- Fetches appdetails in BATCHES (one request for many appids)
- Downloads images concurrently (ThreadPoolExecutor)
- Resume is automatic: if _gp1/_gp2 already exist (and file looks valid), it skips

Run:
  python src/download_steam_gameplay.py --covers-dir data/raw --per-app 2 --workers 32 --batch-size 50
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"

BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "data" / "splits" / "appdetails_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_meta_lock = threading.Lock()
_meta_next_time = 0.0

def rate_limit(min_interval: float):
    global _meta_next_time
    with _meta_lock:
        now = time.time()
        if now < _meta_next_time:
            time.sleep(_meta_next_time - now)
        _meta_next_time = time.time() + min_interval

def cache_path_for(appid: int) -> Path:
    return CACHE_DIR / f"{appid}.json"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
}

_tls = threading.local()


def make_session(pool_maxsize: int = 64, add_age_gate_cookies: bool = True) -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)

    if add_age_gate_cookies:
        s.cookies.set("birthtime", "315532801")    
        s.cookies.set("lastagecheckage", "1-January-1980")
        s.cookies.set("wants_mature_content", "1")

    retry = Retry(
        total=7,
        connect=7,
        read=7,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def get_thread_session() -> requests.Session:
    sess = getattr(_tls, "session", None)
    if sess is None:
        sess = make_session()
        _tls.session = sess
    return sess


def chunks(lst: List, n: int) -> Iterable[List]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def safe_suffix_from_url(url: str) -> str:
    path = urlparse(url).path
    suf = Path(path).suffix.lower()
    return suf if suf in IMG_EXTS else ".jpg"


def atomic_download(url: str, dst: Path, timeout: int = 45, min_bytes: int = 6_000) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size >= min_bytes:
        return True

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    sess = get_thread_session()
    try:
        with sess.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return False

            total = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total += len(chunk)

        if total < min_bytes:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return False

        tmp.replace(dst)
        return True
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def scan_covers(covers_dir: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for p in covers_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue

        name = p.stem
        if "_gp" in name:
            continue

        if not name.isdigit():
            continue

        appid = int(name)
        mapping[appid] = p.parent

    return mapping


def existing_gp_count(genre_dir: Path, appid: int) -> int:
    return len(list(genre_dir.glob(f"{appid}_gp*.*")))


def fetch_appdetails_single(appid: int, sess: requests.Session, timeout: int = 20, min_interval: float = 0.6) -> Dict:
    cp = cache_path_for(appid)
    if cp.exists() and cp.stat().st_size > 10:
        try:
            import json
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass

    params = {
        "appids": str(appid),
        "l": "english",
        "cc": "us",
        "filters": "basic,screenshots",
    }

    rate_limit(min_interval)

    backoff = 2.0
    for _ in range(6):
        try:
            r = sess.get(APPDETAILS_URL, params=params, timeout=timeout)
        except Exception:
            time.sleep(backoff)
            backoff = min(30.0, backoff * 1.7)
            continue

        if r.status_code == 200:
            try:
                j = r.json()
            except Exception:
                return {}
            try:
                cp.write_text(__import__("json").dumps(j), encoding="utf-8")
            except Exception:
                pass
            return j

        if r.status_code in (429, 500, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    time.sleep(float(ra))
                except Exception:
                    time.sleep(backoff)
            else:
                time.sleep(backoff)
            backoff = min(30.0, backoff * 1.7)
            continue

        return {}

    return {}



def fetch_appdetails_batch(appids: List[int], sess: requests.Session, timeout: int = 20, debug: bool = False) -> Dict:
    merged: Dict = {}
    for a in appids:
        one = fetch_appdetails_single(a, sess, timeout=timeout, min_interval=0.6)
        if isinstance(one, dict) and one:
            merged.update(one)
    return merged




@dataclass(frozen=True)
class DownloadTask:
    url: str
    path: Path

def build_tasks_for_app(app_payload: Dict, genre_dir: Path, appid: int, per_app: int) -> List[DownloadTask]:
    if not app_payload or not app_payload.get("success"):
        return []

    data = app_payload.get("data")
    if not isinstance(data, dict):
        return []

    screenshots = data.get("screenshots")
    if not isinstance(screenshots, list) or not screenshots:
        return []

    tasks: List[DownloadTask] = []
    used = 0
    for idx, s in enumerate(screenshots, start=1):
        if used >= per_app:
            break
        if not isinstance(s, dict):
            continue
        url = s.get("path_full") or s.get("path_thumbnail")
        if not url:
            continue
        ext = safe_suffix_from_url(url)
        dst = genre_dir / f"{appid}_gp{used+1}{ext}"
        tasks.append(DownloadTask(url=url, path=dst))
        used += 1

    return tasks


def count_images(root: Path) -> int:
    c = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            c += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers-dir", type=Path, default=Path("data/raw"),
                    help="Root folder that contains genre subfolders with cover images (default: data/raw).")
    ap.add_argument("--per-app", type=int, default=2, help="How many gameplay screenshots per game (default: 2).")
    ap.add_argument("--workers", type=int, default=32, help="Concurrent image download workers (default: 32).")
    ap.add_argument("--batch-size", type=int, default=50, help="How many appids per appdetails request (default: 50).")
    ap.add_argument("--max-total-images", type=int, default=15000,
                    help="Safety cap for total images in covers-dir (default: 15000). Use 0 to disable.")
    ap.add_argument("--min-bytes", type=int, default=6000,
                    help="Minimum file size to consider a download valid (default: 6000).")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be downloaded.")
    ap.add_argument("--debug-samples", type=int, default=0,
                    help="Print debug for first N appids that return no screenshots.")
    args = ap.parse_args()

    covers_dir: Path = args.covers_dir
    per_app: int = max(1, args.per_app)
    workers: int = max(1, args.workers)
    batch_size: int = max(1, args.batch_size)
    max_total: int = args.max_total_images
    min_bytes: int = max(1000, args.min_bytes)

    if not covers_dir.exists():
        raise SystemExit(f"[ERROR] covers-dir not found: {covers_dir}")

    app_to_genre = scan_covers(covers_dir)
    if not app_to_genre:
        raise SystemExit(
            f"[ERROR] No cover images found under {covers_dir}. "
            f"Expected files like data/raw/<GENRE>/<APPID>.jpg"
        )

    targets: List[Tuple[int, Path]] = []
    for appid, genre_dir in app_to_genre.items():
        if existing_gp_count(genre_dir, appid) >= per_app:
            continue
        targets.append((appid, genre_dir))

    if max_total and max_total > 0:
        current = count_images(covers_dir)
        room = max_total - current
        if room <= 0:
            print(f"[INFO] Already at/over max-total-images={max_total}. Nothing to do.")
            return
        max_apps = room // per_app
        if max_apps <= 0:
            print(f"[INFO] Not enough room for even 1 app (room={room}, per_app={per_app}).")
            return
        if len(targets) > max_apps:
            targets = targets[:max_apps]
            print(f"[INFO] Capped to {len(targets)} apps to stay <= {max_total} total images.")

    print(f"[INFO] Covers found: {len(app_to_genre)}")
    print(f"[INFO] Apps missing gameplay: {len(targets)} (per_app={per_app})")

    if not targets:
        print("[INFO] Nothing to download. (All apps already have gameplay screenshots.)")
        return

    meta_sess = make_session(pool_maxsize=32, add_age_gate_cookies=True)

    all_tasks: List[DownloadTask] = []
    debug_left = max(0, args.debug_samples)

    it_batches = list(chunks(targets, batch_size))
    bar_batches = tqdm(it_batches, desc="Fetching appdetails", unit="batch") if tqdm else it_batches

    for batch in bar_batches:
        batch_appids = [appid for appid, _ in batch]
        batch_data = fetch_appdetails_batch(batch_appids, meta_sess, debug=(args.debug_samples > 0))

        if not batch_data:
            if args.debug_samples > 0:
                print("[DEBUG] Batch returned empty dict after fallback too.")
            continue

        for appid, genre_dir in batch:
            payload = batch_data.get(str(appid)) or {}

            if debug_left > 0:
                succ = payload.get("success")
                d = payload.get("data")
                ss_len = 0
                if isinstance(d, dict):
                    ss = d.get("screenshots")
                    if isinstance(ss, list):
                        ss_len = len(ss)
                print(f"[DEBUG] appid={appid} success={succ} screenshots={ss_len}")
                debug_left -= 1

            tasks = build_tasks_for_app(payload, genre_dir, appid, per_app)

            # Skip tasks that already exist (resume-friendly)
            filtered: List[DownloadTask] = []
            for t in tasks:
                if t.path.exists() and t.path.stat().st_size >= min_bytes:
                    continue
                filtered.append(t)

            all_tasks.extend(filtered)


    if args.dry_run:
        print("[DRY RUN] Tasks to download:", len(all_tasks))
        for t in all_tasks[:40]:
            print("  ", t.path, "<-", t.url)
        if len(all_tasks) > 40:
            print("  ...")
        return

    if not all_tasks:
        print("[INFO] No new screenshots to download. "
              "Try --debug-samples 5 to see why screenshots are missing.")
        return

    print(f"[INFO] Total screenshot files to download: {len(all_tasks)}")

    ok = 0
    fail = 0

    bar_dl = tqdm(total=len(all_tasks), desc="Downloading screenshots", unit="img") if tqdm else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(atomic_download, task.url, task.path, 45, min_bytes) for task in all_tasks]
        for fut in as_completed(futures):
            try:
                res = bool(fut.result())
            except Exception:
                res = False
            if res:
                ok += 1
            else:
                fail += 1
            if bar_dl:
                bar_dl.update(1)

    if bar_dl:
        bar_dl.close()

    print(f"[DONE] Downloaded OK: {ok} | Failed: {fail}")
    if fail:
        print("[TIP] Failures are usually region locked / missing screenshots / temporary Steam issues. "
              "Re-run later; resume works automatically.")


if __name__ == "__main__":
    main()
