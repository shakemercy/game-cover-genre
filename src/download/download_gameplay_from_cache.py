import argparse
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import requests

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "data" / "splits" / "appdetails_cache"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
}

_tls = threading.local()

def get_session():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(DEFAULT_HEADERS)
        _tls.s = s
    return s

def safe_suffix_from_url(url: str) -> str:
    suf = Path(urlparse(url).path).suffix.lower()
    return suf if suf in IMG_EXTS else ".jpg"

def atomic_download(url: str, dst: Path, timeout: int = 45, min_bytes: int = 6000) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size >= min_bytes:
        return True

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    s = get_session()
    try:
        with s.get(url, stream=True, timeout=timeout) as r:
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

def load_cache(appid: int) -> dict:
    p = CACHE_DIR / f"{appid}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def build_urls_from_cache(appid: int, per_app: int) -> list[str]:
    j = load_cache(appid)
    payload = j.get(str(appid)) if isinstance(j, dict) else None
    if not isinstance(payload, dict):
        return []
    if not payload.get("success"):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    screenshots = data.get("screenshots")
    if not isinstance(screenshots, list):
        return []
    urls = []
    for s in screenshots:
        if len(urls) >= per_app:
            break
        if not isinstance(s, dict):
            continue
        url = s.get("path_full") or s.get("path_thumbnail")
        if url:
            urls.append(url)
    return urls

def scan_covers(covers_dir: Path):
    app_to_dir = {}
    for p in covers_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        stem = p.stem
        if "_gp" in stem:
            continue
        if stem.isdigit():
            app_to_dir[int(stem)] = p.parent
    return app_to_dir

def existing_gp_count(genre_dir: Path, appid: int) -> int:
    return len(list(genre_dir.glob(f"{appid}_gp*.*")))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--per-app", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-bytes", type=int, default=6000)
    args = ap.parse_args()

    covers_dir = args.covers_dir
    if not covers_dir.is_absolute():
        covers_dir = (BASE_DIR / covers_dir).resolve()

    app_to_dir = scan_covers(covers_dir)
    targets = []
    for appid, gdir in app_to_dir.items():
        if existing_gp_count(gdir, appid) >= args.per_app:
            continue
        targets.append((appid, gdir))

    tasks = []
    for appid, gdir in targets:
        urls = build_urls_from_cache(appid, args.per_app)
        for i, url in enumerate(urls, start=1):
            ext = safe_suffix_from_url(url)
            dst = gdir / f"{appid}_gp{i}{ext}"
            tasks.append((url, dst))

    if not tasks:
        print("[INFO] No tasks found from cache yet (maybe stop later / cache is empty).")
        return

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(atomic_download, url, dst, 45, args.min_bytes) for url, dst in tasks]
        for fut in as_completed(futs):
            if fut.result():
                ok += 1
            else:
                fail += 1

    print(f"[DONE] ok={ok} fail={fail} tasks={len(tasks)} cache_dir={CACHE_DIR}")

if __name__ == "__main__":
    main()
