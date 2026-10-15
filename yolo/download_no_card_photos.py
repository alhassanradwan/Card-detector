"""Download free-licensed photos of card-like objects that are NOT cards.

Searches Wikimedia Commons (all files there are freely licensed) for things the
model might confuse with a card: paper, sticky notes, envelopes, phones... and
saves them with a credits file (author, licence, source page) for each photo.

Usage:
    uv run python yolo/download_no_card_photos.py
"""

import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests

OUTPUT_DIR = Path(r"C:\Users\Lenovo\Downloads\no card photos\internet")
PER_SEARCH = 6
MAX_SIDE = 1280
SEARCHES = [
    "sticky note on desk",
    "post-it notes",
    "index card",
    "blank paper sheet on table",
    "envelope on table",
    "paper receipt",
    "ticket stub paper",
    "smartphone lying on table",
    "notebook on desk",
    "light switch wall",
    "paper tag label",
    "sheet of paper wooden table",
]
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "card-detector-dataset/1.0 (student project; training negatives)"}


def clean(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()


def search(query: str) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": 30,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": MAX_SIDE,
    }
    pages = requests.get(API, params=params, headers=HEADERS, timeout=60).json().get("query", {}).get("pages", {})
    return sorted(pages.values(), key=lambda p: p.get("index", 0))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    credits = []
    for query in SEARCHES:
        saved = 0
        for page in search(query):
            if saved == PER_SEARCH:
                break
            info = page.get("imageinfo", [{}])[0]
            if info.get("mime") not in {"image/jpeg", "image/png"} or min(info.get("width", 0), info.get("height", 0)) < 480:
                continue
            try:
                data = requests.get(info.get("thumburl") or info["url"], headers=HEADERS, timeout=60).content
            except requests.RequestException:
                continue
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            scale = MAX_SIDE / max(image.shape[:2])
            if scale < 1:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            saved += 1
            name = f"{query.replace(' ', '_')}_{saved}.jpg"
            cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(OUTPUT_DIR / name)
            meta = info.get("extmetadata", {})
            credits.append({
                "file": name,
                "search": query,
                "title": page.get("title", ""),
                "author": clean(meta.get("Artist", {}).get("value", "")),
                "license": clean(meta.get("LicenseShortName", {}).get("value", "")),
                "source": info.get("descriptionurl", ""),
            })
            time.sleep(0.3)  # be polite to the Wikimedia servers
        print(f"{query:<32} {saved} photos")

    with open(OUTPUT_DIR / "credits.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "search", "title", "author", "license", "source"])
        writer.writeheader()
        writer.writerows(credits)
    print(f"\n{len(credits)} photos -> {OUTPUT_DIR} (credits in credits.csv)")


if __name__ == "__main__":
    main()
