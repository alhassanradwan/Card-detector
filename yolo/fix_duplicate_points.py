"""Remove accidental double-click points from VoTT polygon annotations.

A point is removed when it is closer than DUPLICATE_DISTANCE_PX to the point
kept before it, so the shape of the polygon does not change.

Usage:
    uv run python yolo/fix_duplicate_points.py [annotations_folder]
"""

import json
import math
import sys
from pathlib import Path
from urllib.parse import unquote

from check_annotations import ANNOTATIONS_DIR, DUPLICATE_DISTANCE_PX


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ANNOTATIONS_DIR

    for file in sorted(folder.glob("*-asset.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        changed = False

        for region in data["regions"]:
            kept = []
            for point in region["points"]:
                xy = (point["x"], point["y"])
                if any(math.dist(xy, (k["x"], k["y"])) < DUPLICATE_DISTANCE_PX for k in kept):
                    continue
                kept.append(point)
            if len(kept) != len(region["points"]):
                print(f"{unquote(data['asset']['name'])}: {len(region['points'])} -> {len(kept)} points")
                region["points"] = kept
                changed = True

        if changed:
            file.write_text(json.dumps(data, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()
