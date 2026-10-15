"""Generate "fake card" photos to teach the model what is NOT a card.

Card-sized pieces of paper (white, coloured, ruled, squares like sticky notes)
are drawn with tilt, shadow and lighting onto real backgrounds cut from the
annotated photos, away from the real card. Two kinds of output:

    <OUTPUT_DIR>/fake only/        no real card: training needs no label at all
    <OUTPUT_DIR>/card plus fake/   a real card photo with a fake paper card added;
                                   labels/ holds the real card's outline (YOLO
                                   format), the fake is deliberately unlabelled

Usage:
    uv run python yolo/make_fake_cards.py
"""

import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from check_annotations import ANNOTATIONS_DIR, image_path_for, load_latest_annotations  # noqa: E402

OUTPUT_DIR = Path(r"C:\Users\Lenovo\Downloads\no card photos")
FAKE_ONLY_COUNT = 80
SEED = 7
OUT_W, OUT_H = 960, 1280  # same portrait size as the phone photos

PAPER_COLOURS = [  # BGR
    (245, 245, 245), (235, 238, 240), (250, 250, 248), (225, 228, 230),  # whites / light greys
    (150, 230, 255), (170, 240, 250), (200, 230, 180), (230, 200, 160),  # yellow, cream, green, blue
    (190, 180, 250), (210, 210, 255),  # pink, light red
]


def read_image(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(path)


def card_polygon(annotation: dict) -> np.ndarray:
    return np.array([(p["x"], p["y"]) for p in annotation["regions"][0]["points"]], np.float32)


def background_crop(image: np.ndarray, card: np.ndarray, rng: random.Random) -> np.ndarray | None:
    """A portrait crop of the photo that does not touch the real card."""
    h, w = image.shape[:2]
    x0, y0 = card.min(axis=0) - 20
    x1, y1 = card.max(axis=0) + 20
    for _ in range(40):
        crop_h = int(h * rng.uniform(0.35, 0.7))
        crop_w = int(crop_h * OUT_W / OUT_H)
        if crop_w >= w:
            continue
        cx, cy = rng.randint(0, w - crop_w), rng.randint(0, h - crop_h)
        if cx + crop_w < x0 or cx > x1 or cy + crop_h < y0 or cy > y1:
            crop = image[cy:cy + crop_h, cx:cx + crop_w]
            return cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)
    return None


def paper_texture(width: int, height: int, rng: random.Random) -> np.ndarray:
    """A flat piece of paper: plain, ruled, grid or slightly crumpled."""
    colour = np.array(rng.choice(PAPER_COLOURS), np.float32)
    paper = np.ones((height, width, 3), np.float32) * colour
    paper += np.random.default_rng(rng.randint(0, 10**6)).normal(0, 3, paper.shape)  # paper grain
    style = rng.random()
    if style < 0.25:  # ruled lines
        gap = rng.randint(14, 26)
        for y in range(gap, height, gap):
            cv2.line(paper, (0, y), (width, y), (200, 170, 140), 1, cv2.LINE_AA)
    elif style < 0.35:  # grid paper
        gap = rng.randint(12, 20)
        for v in range(gap, max(width, height), gap):
            cv2.line(paper, (v, 0), (v, height), (210, 200, 190), 1, cv2.LINE_AA)
            cv2.line(paper, (0, v), (width, v), (210, 200, 190), 1, cv2.LINE_AA)
    elif style < 0.5:  # soft creases
        for _ in range(rng.randint(1, 3)):
            p1 = (rng.randint(0, width), 0)
            p2 = (rng.randint(0, width), height)
            cv2.line(paper, p1, p2, tuple(float(c) - 18 for c in colour), 2, cv2.LINE_AA)
        paper = cv2.GaussianBlur(paper, (5, 5), 0)
    return np.clip(paper, 0, 255)


def add_fake_card(image: np.ndarray, avoid: np.ndarray | None, rng: random.Random) -> bool:
    """Draws a card-sized piece of paper onto image (in place). Returns False if it
    could not be placed away from the polygon `avoid`."""
    h, w = image.shape[:2]
    square = rng.random() < 0.15  # sticky-note shape
    aspect = 1.0 if square else rng.uniform(1.45, 1.75)  # real cards are 1.59
    # Smaller next to a real card, which often fills much of the photo
    long_side = w * (rng.uniform(0.2, 0.42) if avoid is not None else rng.uniform(0.3, 0.65))
    pw, ph = int(long_side), int(long_side / aspect)
    radius = int(min(pw, ph) * (0.02 if square else rng.uniform(0.04, 0.08)))

    # Paper image with rounded corners (drawn upright, then warped into place)
    paper = paper_texture(pw, ph, rng)
    shape = np.zeros((ph, pw), np.uint8)
    cv2.rectangle(shape, (radius, 0), (pw - radius, ph), 255, -1)
    cv2.rectangle(shape, (0, radius), (pw, ph - radius), 255, -1)
    for cx, cy in [(radius, radius), (pw - radius, radius), (radius, ph - radius), (pw - radius, ph - radius)]:
        cv2.circle(shape, (cx, cy), radius, 255, -1, cv2.LINE_AA)

    # Random position, rotation and a little perspective (phone not perfectly flat)
    angle = math.radians(rng.uniform(0, 360))
    corners = np.array([(-pw / 2, -ph / 2), (pw / 2, -ph / 2), (pw / 2, ph / 2), (-pw / 2, ph / 2)])
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    for _ in range(200):
        centre = np.array([rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h])
        jitter = np.array([(rng.uniform(-0.06, 0.06) * pw, rng.uniform(-0.06, 0.06) * ph) for _ in range(4)])
        target = ((corners + jitter) @ rot.T + centre).astype(np.float32)
        if target.min() < 0 or target[:, 0].max() >= w or target[:, 1].max() >= h:
            continue
        if avoid is not None:
            fake_area = np.zeros((h, w), np.uint8)
            cv2.fillPoly(fake_area, [target.astype(np.int32)], 1)
            real_area = np.zeros((h, w), np.uint8)
            cv2.fillPoly(real_area, [avoid.astype(np.int32)], 1)
            real_area = cv2.dilate(real_area, np.ones((41, 41), np.uint8))
            if (fake_area & real_area).any():
                continue
        break
    else:
        return False

    src = np.array([(0, 0), (pw, 0), (pw, ph), (0, ph)], np.float32)
    warp = cv2.getPerspectiveTransform(src, target)
    warped = cv2.warpPerspective(paper, warp, (w, h))
    alpha = cv2.warpPerspective(shape, warp, (w, h)).astype(np.float32) / 255
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[..., None]

    # Soft shadow under the paper
    offset = np.float32([[1, 0, rng.uniform(4, 14)], [0, 1, rng.uniform(4, 14)]])
    shadow = cv2.GaussianBlur(cv2.warpAffine(alpha[..., 0], offset, (w, h)), (31, 31), 0)[..., None]
    base = image.astype(np.float32) * (1 - shadow * rng.uniform(0.25, 0.45))

    # Lighting: brighter on one side, matched roughly to the background brightness
    gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    tilt = rng.uniform(0, 2 * math.pi)
    light = 1 + rng.uniform(0.05, 0.2) * (gx * math.cos(tilt) + gy * math.sin(tilt))
    scene = float(image.mean()) / 150
    warped *= (light * min(max(scene, 0.55), 1.05))[..., None]

    image[:] = np.clip(base * (1 - alpha) + warped * alpha, 0, 255).astype(np.uint8)
    return True


def main() -> None:
    rng = random.Random(SEED)
    annotations = load_latest_annotations(ANNOTATIONS_DIR)
    photos = [(read_image(image_path_for(a["asset"])), card_polygon(a), a) for a in annotations]

    fake_dir = OUTPUT_DIR / "fake only"
    pair_dir = OUTPUT_DIR / "card plus fake"
    (pair_dir / "labels").mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    while made < FAKE_ONLY_COUNT:
        image, card, _ = rng.choice(photos)
        background = background_crop(image, card, rng)
        if background is None or not add_fake_card(background, None, rng):
            continue
        made += 1
        write_image(fake_dir / f"fake_{made:03d}.jpg", background)

    paired = 0
    for image, card, annotation in photos:
        combined = image.copy()
        if not add_fake_card(combined, card, rng):
            continue
        paired += 1
        stem = Path(image_path_for(annotation["asset"])).stem
        write_image(pair_dir / f"{stem} + fake.jpg", combined)
        h, w = image.shape[:2]
        coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in card)
        (pair_dir / "labels" / f"{stem} + fake.txt").write_text(f"0 {coords}\n", encoding="utf-8")

    print(f"fake only:      {made} photos -> {fake_dir}")
    print(f"card plus fake: {paired} photos -> {pair_dir}")


if __name__ == "__main__":
    main()
