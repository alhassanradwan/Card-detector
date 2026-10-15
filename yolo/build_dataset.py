"""Build the YOLO segmentation dataset from the real and the fake-card photos.

Sources:
    VoTT annotations             real card photos with the card outlined
    <FAKES_DIR>/card plus fake/  real card photo + an unlabelled fake paper card;
                                 goes to the same split as the photo it was made from
    <FAKES_DIR>/fake only/       no card at all (empty label = "nothing to find here")
    <FAKES_DIR>/internet/        no card at all

Creates:
    yolo/dataset/images/{train,val}/   copies of the photos
    yolo/dataset/labels/{train,val}/   one .txt per photo: "0 x1 y1 x2 y2 x3 y3 x4 y4" (0..1), empty if no card
    yolo/dataset/data.yaml             dataset description for Ultralytics
    yolo/dataset/preview/              photos with the label drawn on top

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
from make_fake_cards import OUTPUT_DIR as FAKES_DIR

DATASET_DIR = Path(__file__).parent / "dataset"
CLASS_NAMES = ["card"]
VAL_FRACTION = 0.2  # share of photos kept aside to check the model
SEED = 42  # fixed seed so the train/val split is the same every run
NO_CARD_FOLDERS = ["fake only", "internet"]
# Too many photos without a card can make the model miss real cards. 60 keeps them
# at about a third of the training photos (the card-plus-fake photos add more fake
# paper). Raise it if the model still mistakes paper for cards; lower it if it
# starts missing real cards. None = use them all.
MAX_NO_CARD_PHOTOS: int | None = 60


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


def load_card_plus_fake() -> dict[str, list[dict]]:
    """Card-plus-fake photos grouped by the name of the real photo they came from."""
    by_source: dict[str, list[dict]] = {}
    folder = FAKES_DIR / "card plus fake"
    for image in sorted(folder.glob("*.jpg")):
        label = folder / "labels" / f"{image.stem}.txt"
        polygons = []
        for line in label.read_text(encoding="utf-8").splitlines():
            values = line.split()
            coords = [float(v) for v in values[1:]]
            polygons.append((int(values[0]), list(zip(coords[::2], coords[1::2]))))
        source = image.stem.removesuffix(" + fake")
        by_source.setdefault(source, []).append({"image": image, "polygons": polygons})
    return by_source


def load_no_card() -> list[dict]:
    """Photos without a card; their empty label teaches the model what is not a card."""
    images = sorted(p for name in NO_CARD_FOLDERS for p in (FAKES_DIR / name).glob("*.jpg"))
    random.Random(SEED).shuffle(images)
    return [{"image": image, "polygons": []} for image in images[:MAX_NO_CARD_PHOTOS]]


def split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Shuffles with the fixed seed and returns (train, val)."""
    items = list(items)
    random.Random(SEED).shuffle(items)
    val_count = max(1, round(len(items) * VAL_FRACTION))
    return items[val_count:], items[:val_count]


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
        text = "".join(f"{line}\n" for line in lines)  # empty file = no card in this photo
        (labels_dir / f"{item['image'].stem}.txt").write_text(text, encoding="utf-8")


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
        caption = split if label_file.read_text(encoding="utf-8").strip() else f"{split} - no card"
        cv2.putText(image, caption, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError(f"could not encode preview for {image_file}")
        encoded.tofile(preview_dir / f"{split}_{label_file.stem}.jpg")


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ANNOTATIONS_DIR
    real = load_annotations(folder)

    # Check every photo before deleting the old dataset, so a missing photo can't leave it empty.
    missing = [item["image"] for item in real if not item["image"].exists()]
    if missing:
        print("Photos not found - check IMAGES_DIR in check_annotations.py:")
        for path in missing:
            print(f"  {path}")
        sys.exit(1)

    # Real photos are split on their own first, so the val photos stay the same as
    # in earlier runs and old and new models can be compared fairly.
    real_train, real_val = split(real)
    pairs = load_card_plus_fake()
    pairs_train = [p for item in real_train for p in pairs.get(item["image"].stem, [])]
    pairs_val = [p for item in real_val for p in pairs.get(item["image"].stem, [])]
    no_card_train, no_card_val = split(load_no_card())

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    train = real_train + pairs_train + no_card_train
    val = real_val + pairs_val + no_card_val
    write_split(train, "train")
    write_split(val, "val")

    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    (DATASET_DIR / "data.yaml").write_text(
        f"train: images/train\nval: images/val\n\nnames:\n{names}\n", encoding="utf-8"
    )

    write_previews()

    print(f"{'':<16}{'train':>7}{'val':>6}")
    for name, (t, v) in {
        "real cards": (real_train, real_val),
        "card plus fake": (pairs_train, pairs_val),
        "no card": (no_card_train, no_card_val),
        "total": (train, val),
    }.items():
        print(f"{name:<16}{len(t):>7}{len(v):>6}")
    print("real val photos:")
    for item in real_val:
        print(f"  {item['image'].name}")
    print(f"dataset written to {DATASET_DIR}")


if __name__ == "__main__":
    main()
