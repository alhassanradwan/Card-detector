"""Convert VoTT polygon annotations into a YOLO segmentation dataset.

Creates:
    yolo/dataset/images/{train,val}/   copies of the photos
    yolo/dataset/labels/{train,val}/   one .txt per photo: "0 x1 y1 x2 y2 x3 y3 x4 y4" (0..1)
    yolo/dataset/data.yaml             dataset description for Ultralytics
    yolo/dataset/preview/              photos with the converted label drawn on top

Usage:
    uv run python yolo/build_dataset.py [annotations_folder]
"""

import glob
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from check_annotations import ANNOTATIONS_DIR, image_path_for, load_latest_annotations

DATASET_DIR = Path(__file__).parent / "dataset"
CLASS_NAMES = ["card"]
VAL_FRACTION = 0.2  # share of photos kept aside to check the model
SEED = 42  # fixed seed so the train/val split is the same every run


def load_annotations(folder: Path) -> list[dict]:
    items = []
    for data in load_latest_annotations(folder):
        asset = data["asset"]
        width, height = asset["size"]["width"], asset["size"]["height"]
        polygons = []
        for region in data["regions"]:
            class_id = CLASS_NAMES.index(region["tags"][0])
            points = [
                (min(max(p["x"] / width, 0.0), 1.0), min(max(p["y"] / height, 0.0), 1.0))
                for p in region["points"]
            ]
            polygons.append((class_id, points))
        items.append({
            "image": image_path_for(asset),
            "polygons": polygons,
        })
    return items


def write_split(items: list[dict], split: str) -> None:
    images_dir = DATASET_DIR / "images" / split
    labels_dir = DATASET_DIR / "labels" / split
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for item in items:
        shutil.copy2(item["image"], images_dir / item["image"].name)
        lines = [
            f"{class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
            for class_id, points in item["polygons"]
        ]
        (labels_dir / f"{item['image'].stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_previews() -> None:
    """Draw each label file back onto its image, to check the conversion by eye."""
    preview_dir = DATASET_DIR / "preview"
    preview_dir.mkdir()
    for label_file in sorted((DATASET_DIR / "labels").glob("*/*.txt")):
        split = label_file.parent.name
        image_file = next((DATASET_DIR / "images" / split).glob(f"{glob.escape(label_file.stem)}.*"))
        image = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image {image_file}")
        height, width = image.shape[:2]
        for line in label_file.read_text(encoding="utf-8").splitlines():
            values = [float(v) for v in line.split()[1:]]
            points = np.array(
                [(values[i] * width, values[i + 1] * height) for i in range(0, len(values), 2)],
                dtype=np.int32,
            )
            overlay = image.copy()
            cv2.fillPoly(overlay, [points], (0, 255, 0))
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            cv2.polylines(image, [points], True, (0, 255, 0), 3)
        cv2.putText(image, split, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError(f"could not encode preview for {image_file}")
        encoded.tofile(preview_dir / f"{split}_{label_file.stem}.jpg")


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ANNOTATIONS_DIR
    items = load_annotations(folder)

    # Check every photo before deleting the old dataset, so a missing photo can't leave it empty.
    missing = [item["image"] for item in items if not item["image"].exists()]
    if missing:
        print("Photos not found - check IMAGES_DIR in check_annotations.py:")
        for path in missing:
            print(f"  {path}")
        sys.exit(1)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    random.Random(SEED).shuffle(items)
    val_count = max(1, round(len(items) * VAL_FRACTION))
    val, train = items[:val_count], items[val_count:]
    write_split(train, "train")
    write_split(val, "val")

    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    (DATASET_DIR / "data.yaml").write_text(
        f"train: images/train\nval: images/val\n\nnames:\n{names}\n", encoding="utf-8"
    )

    write_previews()

    print(f"train: {len(train)} images, val: {len(val)} images")
    print("val images:")
    for item in val:
        print(f"  {item['image'].name}")
    print(f"dataset written to {DATASET_DIR}")


if __name__ == "__main__":
    main()
