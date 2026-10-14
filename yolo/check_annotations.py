"""Check VoTT annotations: every card must have exactly 4 corner points.

Usage:
    uv run python yolo/check_annotations.py [annotations_folder]
"""

import json
import math
import sys
from pathlib import Path
from urllib.parse import unquote

ANNOTATIONS_DIR = Path(r"C:\Users\Lenovo\Downloads\annotated")
# Photos are looked up by file name here, so the folder can be renamed or moved
# without editing the paths VoTT saved inside the JSON files.
IMAGES_DIR = Path(r"C:\Users\Lenovo\Downloads\cards before annotation")
DUPLICATE_DISTANCE_PX = 5  # points closer than this count as an accidental double click


def image_path_for(asset: dict) -> Path:
    return IMAGES_DIR / unquote(asset["name"])


def load_latest_annotations(folder: Path) -> list[dict]:
    """Load one annotation per photo.

    A photo annotated in more than one VoTT project has several JSON files;
    only the most recently saved one is used.
    """
    latest: dict[str, tuple[float, dict]] = {}
    for file in folder.glob("*-asset.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        name = unquote(data["asset"]["name"])
        saved_at = file.stat().st_mtime
        if name not in latest or saved_at > latest[name][0]:
            latest[name] = (saved_at, data)
    return [latest[name][1] for name in sorted(latest)]


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ANNOTATIONS_DIR
    files = list(folder.glob("*-asset.json"))
    annotations = load_latest_annotations(folder)
    problems = 0

    for data in annotations:
        image_name = unquote(data["asset"]["name"])
        image_path = image_path_for(data["asset"])
        issues = []

        if not image_path.exists():
            issues.append(f"image not found at {image_path}")

        regions = data["regions"]
        if len(regions) != 1:
            issues.append(f"{len(regions)} regions (expected 1 card)")

        for region in regions:
            points = [(p["x"], p["y"]) for p in region["points"]]
            if len(points) != 4:
                issues.append(f"{len(points)} points (expected 4)")
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    if math.dist(points[i], points[j]) < DUPLICATE_DISTANCE_PX:
                        issues.append(
                            f"points {i + 1} and {j + 1} are on top of each other "
                            f"({points[i][0]:.0f}, {points[i][1]:.0f})"
                        )

        if issues:
            problems += 1
            print(f"FIX  {image_name}")
            for issue in issues:
                print(f"       - {issue}")
        else:
            print(f"OK   {image_name}")

    skipped = len(files) - len(annotations)
    if skipped:
        print(f"\n{skipped} older duplicate annotations ignored (same photo annotated again later).")
    print(f"{len(annotations)} annotations checked, {problems} need fixing.")


if __name__ == "__main__":
    main()
