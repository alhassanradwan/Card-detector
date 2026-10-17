"""Analyse a finished training run and print a summary in the terminal.

Runs automatically at the end of train.py. Covers:
    1. best epoch          the epoch saved as best.pt and its scores, plus the top 5
    2. training length     whether it stopped improving early, kept improving, or overfitted
    3. mistakes            missed cards and false alarms on photos it never trained on
                           (saved with the model's guesses drawn on in <run>/mistakes/)
    4. confidence cut-off  the range that separates real cards from fakes
    5. comparison          against the previous run
    6. chart               the score of every epoch, drawn with text
    7. prediction time     time per photo on this PC and the average over all photos
    8. runtime             training, predicting and analysis times, and the software running the model
    9. suggestions         what to try next

Mistakes and the comparison use only photos the model never trained on:
    real cards        the real validation photos
    card plus fake    the card-plus-fake validation photos (only the real card should be found)
    no card (val)     the no-card validation photos
    no card (unused)  no-card photos left out of the dataset (MAX_NO_CARD_PHOTOS)
and the same confidence cut-off as the app (_minConfidence in card_detector.dart).

Usage:
    uv run python yolo/analyze_run.py [run] [previous_run]
"""

import csv
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, TypeVar, cast

import cv2
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Results

from build_dataset import DATASET_DIR, MAX_NO_CARD_PHOTOS, NO_CARD_FOLDERS
from make_fake_cards import OUTPUT_DIR as FAKES_DIR

RUNS_DIR = Path(__file__).parent / "runs"
PROJECT_DIR = Path(__file__).parent.parent
APP_DETECTOR = PROJECT_DIR / "lib" / "detection" / "card_detector.dart"
APP_MODEL = PROJECT_DIR / "assets" / "models" / "card_seg.tflite"
MEANINGFUL_GAIN = 0.005  # a score gain smaller than this is treated as noise
OVERFIT_RISE = 0.15  # validation outline error this far above its lowest point = overfitting


def app_confidence() -> float:
    """The confidence cut-off the app uses (_minConfidence in card_detector.dart)."""
    match = re.search(r"_minConfidence\s*=\s*([0-9.]+)", APP_DETECTOR.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else 0.5


CONFIDENCE = app_confidence()


def test_groups() -> dict[str, list[Path]]:
    """Photos no model trained on, grouped by what a good model should do with them."""
    val_dir = DATASET_DIR / "images" / "val"
    val_labels = DATASET_DIR / "labels" / "val"

    def has_card(photo: Path) -> bool:
        return bool((val_labels / f"{photo.stem}.txt").read_text(encoding="utf-8").strip())

    in_dataset = {p.name for p in (DATASET_DIR / "images").glob("*/*")}
    val_photos = sorted(val_dir.glob("*"))
    return {
        "real cards": [p for p in val_photos if "+ fake" not in p.stem and has_card(p)],
        "card plus fake": [p for p in val_photos if "+ fake" in p.stem],
        "no card (val)": [p for p in val_photos if not has_card(p)],
        "no card (unused)": sorted(p for name in NO_CARD_FOLDERS for p in (FAKES_DIR / name).glob("*.jpg")
                                   if p.name not in in_dataset),
    }


def detect(model: YOLO, photo: Path, confidence: float = 0.05) -> Results:
    """The model's result for one photo, keeping guesses down to `confidence`."""
    return cast(list[Results], model.predict(str(photo), imgsz=640, conf=confidence, verbose=False))[0]


def scores(result: Results) -> list[float]:
    """Confidence of every object found, highest first."""
    boxes = result.boxes
    return [] if boxes is None else sorted((float(c) for c in boxes.conf), reverse=True)


def summarize(group: str, photo_scores: list[list[float]]) -> str:
    """One comparison table cell: how a model did on a group at the app's cut-off."""
    counts = [sum(s >= CONFIDENCE for s in found) for found in photo_scores]
    total = len(photo_scores)
    if group == "real cards":
        return f"found {sum(c >= 1 for c in counts)}/{total}"
    if group == "card plus fake":
        return f"fake outlined {sum(c > 1 for c in counts)}/{total}"
    return f"false alarm {sum(c >= 1 for c in counts)}/{total}"


def heading(title: str) -> None:
    print(f"\n{'=' * 64}\n {title}\n{'=' * 64}")


def load_epochs(run_dir: Path) -> list[dict[str, float]]:
    """results.csv as numbers, plus Ultralytics' best.pt score (box + mask mAP50-95)."""
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as f:
        epochs = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    for epoch in epochs:
        epoch["score"] = epoch["metrics/mAP50-95(B)"] + epoch["metrics/mAP50-95(M)"]
    return epochs


def previous_run(run: str) -> str | None:
    """The most recent other run trained before this one."""
    this = RUNS_DIR / run / "weights" / "best.pt"
    older = [p for p in RUNS_DIR.glob("*/weights/best.pt")
             if p != this and p.stat().st_mtime < this.stat().st_mtime]
    return max(older, key=lambda p: p.stat().st_mtime).parent.parent.name if older else None


def smooth(values: list[float], window: int = 5) -> list[float]:
    return [sum(values[max(0, i - window + 1):i + 1]) / len(values[max(0, i - window + 1):i + 1])
            for i in range(len(values))]


def show_best_epoch(epochs: list[dict[str, float]]) -> dict[str, float]:
    heading("1. BEST EPOCH")
    best = max(epochs, key=lambda e: e["score"])  # first epoch with the top score, as Ultralytics keeps it
    print(f"Best epoch: {int(best['epoch'])} of {len(epochs)}  (saved as weights/best.pt)")
    print(f"  Precision (right when it says 'card') ...... {best['metrics/precision(M)']:.3f}")
    print(f"  Recall (real cards found) .................. {best['metrics/recall(M)']:.3f}")
    print(f"  Outline roughly right (mAP50) .............. {best['metrics/mAP50(M)']:.3f}")
    print(f"  Outline fits tightly (mask mAP50-95) ....... {best['metrics/mAP50-95(M)']:.3f}")
    print(f"  Box fits tightly (box mAP50-95) ............ {best['metrics/mAP50-95(B)']:.3f}")
    print(f"  Training time: {epochs[-1]['time'] / 60:.1f} minutes")

    print(f"\n  {'epoch':>5}  {'score':>6}  {'mask':>5}  {'box':>5}  {'P':>5}  {'R':>5}")
    top = sorted(epochs, key=lambda e: e["score"], reverse=True)[:5]
    for e in top + ([epochs[-1]] if epochs[-1] not in top else []):
        tag = "  <- best" if e is best else ("  <- last (last.pt)" if e is epochs[-1] else "")
        print(f"  {int(e['epoch']):>5}  {e['score']:>6.3f}  {e['metrics/mAP50-95(M)']:>5.3f}  "
              f"{e['metrics/mAP50-95(B)']:>5.3f}  {e['metrics/precision(M)']:>5.3f}  "
              f"{e['metrics/recall(M)']:>5.3f}{tag}")
    return best


def show_training_length(epochs: list[dict[str, float]]) -> list[str]:
    heading("2. TRAINING LENGTH")
    advice = []
    total = len(epochs)
    best_so_far, last_gain = 0.0, 1
    for e in epochs:
        if e["score"] > best_so_far + MEANINGFUL_GAIN:
            best_so_far, last_gain = e["score"], int(e["epoch"])
    if last_gain <= 0.75 * total:
        enough = min(total, last_gain + 10)
        print(f"Stopped improving around epoch {last_gain}; about {enough} epochs would have been enough.")
        advice.append(f"Training stopped improving around epoch {last_gain}: EPOCHS = {enough} would save time.")
    elif last_gain >= 0.9 * total:
        print(f"Still improving near the end (epoch {last_gain} of {total}); more epochs may help.")
        advice.append(f"Still improving at epoch {last_gain} of {total}: try EPOCHS = {int(total * 1.5)}.")
    else:
        print(f"Last real improvement at epoch {last_gain} of {total}: the training length looks right.")

    val_loss = smooth([e["val/seg_loss"] for e in epochs])
    train_loss = smooth([e["train/seg_loss"] for e in epochs])
    lowest = min(range(total), key=lambda i: val_loss[i])
    if val_loss[-1] > val_loss[lowest] * (1 + OVERFIT_RISE) and train_loss[-1] < train_loss[lowest]:
        print(f"OVERFITTING: from epoch {lowest + 1} the outline error on test photos rose "
              f"({val_loss[lowest]:.3f} -> {val_loss[-1]:.3f}) while it kept falling on training photos.")
        advice.append(f"Overfitting from epoch {lowest + 1}: train fewer epochs or add more photos.")
    else:
        print("No overfitting: the outline error on test photos did not rise while training continued.")
    return advice


class Timing(NamedTuple):
    """How long one prediction took, in milliseconds."""

    group: str
    photo: str
    prepare: float  # resize and convert the photo
    model: float  # the model itself
    outline: float  # turn the model output into boxes and outlines

    @property
    def total(self) -> float:
        return self.prepare + self.model + self.outline


def show_mistakes(
    run_dir: Path, model: YOLO
) -> tuple[dict[str, int], list[float], list[float], list[str], list[Timing]]:
    heading(f"3. MISTAKES (photos it never trained on, cut-off {CONFIDENCE:.0%})")
    mistakes_dir = run_dir / "mistakes"
    shutil.rmtree(mistakes_dir, ignore_errors=True)
    mistakes_dir.mkdir()

    groups = test_groups()
    first_photo = next(photo for photos in groups.values() for photo in photos)
    for _ in range(3):  # warm-up: the first predictions are slow while the model loads
        detect(model, first_photo)

    counts = {"cards": 0, "missed": 0, "fake outlined": 0, "no-card photos": 0, "false alarms": 0}
    card_scores, fake_scores, missed_names = [], [], []
    timings: list[Timing] = []
    for group, photos in groups.items():
        for photo in photos:
            result = detect(model, photo)
            speed = result.speed
            timings.append(Timing(group, photo.name, speed["preprocess"] or 0.0,
                                  speed["inference"] or 0.0, speed["postprocess"] or 0.0))
            found = scores(result)
            top = found[0] if found else 0.0
            problem = None
            if group in ("real cards", "card plus fake"):
                counts["cards"] += 1
                if top > 0:
                    card_scores.append(top)
                if top < CONFIDENCE:
                    counts["missed"] += 1
                    missed_names.append(photo.name)
                    problem = f"MISSED card (best guess {top:.0%})"
                if group == "card plus fake" and len(found) > 1:
                    fake_scores.append(found[1])  # the real card scores highest, the fake second
                    if found[1] >= CONFIDENCE:
                        counts["fake outlined"] += 1
                        problem = f"FAKE outlined next to card ({found[1]:.0%})"
            else:
                counts["no-card photos"] += 1
                if found:
                    fake_scores.append(top)
                if top >= CONFIDENCE:
                    counts["false alarms"] += 1
                    problem = f"FALSE ALARM ({top:.0%})"
            if problem:
                print(f"  {problem:<36} {photo.name}")
                image = result.plot()
                cv2.putText(image, problem, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imencode(".jpg", image)[1].tofile(mistakes_dir / f"{problem.split(' (')[0]} - {photo.stem}.jpg")

    print(f"\n  Real cards found ............ {counts['cards'] - counts['missed']}/{counts['cards']}")
    print(f"  Fakes outlined next to cards  {counts['fake outlined']}")
    print(f"  False alarms ................ {counts['false alarms']}/{counts['no-card photos']} no-card photos")
    if counts["missed"] or counts["fake outlined"] or counts["false alarms"]:
        print(f"  Photos with the model's guesses drawn on: {mistakes_dir}")
    return counts, card_scores, fake_scores, missed_names, timings


def show_cutoff(card_scores: list[float], fake_scores: list[float]) -> list[str]:
    heading("4. CONFIDENCE CUT-OFF")
    advice = []
    if not card_scores:
        print("No real card was found at all, so no cut-off can be recommended.")
        return advice
    lowest_card = min(card_scores)
    highest_fake = max(fake_scores, default=0.0)
    print(f"  Real cards score ........ {lowest_card:.0%} to {max(card_scores):.0%}"
          "  (cards the model missed completely are not counted)")
    print(f"  Fakes score up to ....... {highest_fake:.0%}")
    low = round(highest_fake + 0.05, 2)
    high = round(lowest_card - 0.05, 2)
    if low <= high:
        print(f"  Safe cut-off range ...... {low:.0%} to {high:.0%}"
              f"  (the app uses {CONFIDENCE:.0%})")
        if CONFIDENCE < low:
            advice.append(f"The app's cut-off ({CONFIDENCE:.0%}) is close to the highest fake ({highest_fake:.0%}): "
                          f"if paper still gets outlined on the phone, raise _minConfidence in "
                          f"card_detector.dart towards {low:.2f}-{high:.2f}.")
        elif CONFIDENCE > high:
            advice.append(f"The app's cut-off ({CONFIDENCE:.0%}) is close to the lowest real card ({lowest_card:.0%}): "
                          f"real cards may be missed in poor light; consider _minConfidence = {low:.2f}-{high:.2f}.")
    else:
        print("  No cut-off separates them: some fakes score as high as real cards.")
        advice.append("Fakes and real cards overlap in confidence: add more no-card and card-plus-fake photos.")
    return advice


def show_comparison(run: str, previous: str | None) -> tuple[int, int] | None:
    heading("5. COMPARISON WITH THE PREVIOUS RUN")
    if previous is None:
        print("No earlier run to compare with.")
        return None
    models = {name: YOLO(RUNS_DIR / name / "weights" / "best.pt") for name in (previous, run)}
    print(f"{'':<18}{'photos':>7}   {previous:>22}   {run:>22}")
    real_found = (0, 0)
    for group, photos in test_groups().items():
        results = [[scores(detect(models[name], p)) for p in photos] for name in (previous, run)]
        cells = [summarize(group, photo_scores) for photo_scores in results]
        print(f"{group:<18}{len(photos):>7}   {cells[0]:>22}   {cells[1]:>22}")
        if group == "real cards":
            old, new = (sum(1 for found in photo_scores if found and found[0] >= CONFIDENCE)
                        for photo_scores in results)
            real_found = (old, new)
    return real_found  # real cards found by the previous and this run


def show_chart(epochs: list[dict[str, float]], best: dict[str, float]) -> None:
    heading("6. SCORE PER EPOCH (box + mask mAP50-95, higher is better)")
    width, height = min(len(epochs), 60), 10
    per_column = len(epochs) / width
    columns = [max(e["score"] for e in epochs[int(c * per_column):int((c + 1) * per_column) or 1])
               for c in range(width)]
    best_column = min(int((best["epoch"] - 1) / per_column), width - 1)
    low = min(e["score"] for e in epochs[len(epochs) // 10:])  # skip the first epochs so the end is readable
    high = max(columns)
    span = (high - low) or 1
    print(" " * 7 + " " * best_column + f"* best (epoch {int(best['epoch'])})")
    for row in range(height, 0, -1):
        level = low + span * (row - 0.5) / height
        label = f"{high:.3f} |" if row == height else (f"{low:.3f} |" if row == 1 else "      |")
        print(label + "".join("#" if value >= level else " " for value in columns))
    print("      +" + "-" * width)
    print(f"       epoch 1{' ' * (width - len(str(len(epochs))) - 7)}{len(epochs)}")


def show_prediction_time(timings: list[Timing]) -> None:
    device = f"GPU {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    heading(f"7. PREDICTION TIME ON THIS PC ({device})")
    print("  total ms = prepare photo + model + build outline (the phone is usually slower)\n")
    groups = list(dict.fromkeys(t.group for t in timings))
    for group in groups:
        print(f"  {group}:")
        for t in timings:
            if t.group == group:
                print(f"    {t.total:6.1f} ms   (model {t.model:5.1f})   {t.photo}")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    print(f"\n  {'average per group':<20}{'photos':>7}{'total ms':>10}{'model ms':>10}")
    for group in groups:
        items = [t for t in timings if t.group == group]
        print(f"  {group:<20}{len(items):>7}{mean([t.total for t in items]):>10.1f}"
              f"{mean([t.model for t in items]):>10.1f}")
    total = mean([t.total for t in timings])
    print(f"\n  Average over all {len(timings)} photos: {total:.1f} ms per photo "
          f"(prepare {mean([t.prepare for t in timings]):.1f} + model {mean([t.model for t in timings]):.1f} "
          f"+ outline {mean([t.outline for t in timings]):.1f}), about {1000 / total:.0f} photos per second")
    print(f"  Fastest {min(t.total for t in timings):.1f} ms, slowest {max(t.total for t in timings):.1f} ms")
    print(f"  Total for all {len(timings)} photos: {sum(t.total for t in timings) / 1000:.1f} s")


def package_version(name: str) -> str:
    """A Flutter package's version from pubspec.lock."""
    lock = (PROJECT_DIR / "pubspec.lock").read_text(encoding="utf-8")
    match = re.search(rf"\n  {name}:\n(?:    .*\n)*?    version: \"([^\"]+)\"", lock)
    return match.group(1) if match else "?"


def show_runtime(run_dir: Path, epochs: list[dict[str, float]], timings: list[Timing],
                 part_times: dict[str, float]) -> None:
    heading("8. RUNTIME")
    training = epochs[-1]["time"]
    print(f"  Training ................... {training / 60:.1f} min "
          f"({len(epochs)} epochs, {training / len(epochs):.1f} s per epoch)")
    print(f"  Predicting all {len(timings)} photos .. {sum(t.total for t in timings) / 1000:.1f} s")

    print("\n  This analysis:")
    for part, seconds in part_times.items():
        print(f"    {part:<28}{seconds:>7.1f} s")
    print(f"    {'total':<28}{sum(part_times.values()):>7.1f} s")

    device = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})" if torch.cuda.is_available() else "CPU"
    pt_file = run_dir / "weights" / "best.pt"
    print("\n  Software running the model:")
    print(f"    PC:    PyTorch {torch.__version__} on {device}, Ultralytics {ultralytics.__version__}, "
          f"{pt_file.name} {pt_file.stat().st_size / 2**20:.1f} MB")
    app_model = f"{APP_MODEL.name} {APP_MODEL.stat().st_size / 2**20:.1f} MB" if APP_MODEL.exists() else "no model"
    print(f"    Phone: TensorFlow Lite via tflite_flutter {package_version('tflite_flutter')} "
          f"on the phone's CPU (4 threads), {app_model}")


def show_suggestions(advice: list[str]) -> None:
    heading("9. SUGGESTIONS")
    if not advice:
        print("No problems found. Next: export this model for the app (yolo/export_tflite.py).")
    for item in advice:
        print(f"  - {item}")


T = TypeVar("T")


def analyze(run: str, previous: str | None = None) -> None:
    part_times: dict[str, float] = {}

    def timed(part: str, show: Callable[..., T], *args: object) -> T:
        """Runs one part of the analysis and records how long it took."""
        start = time.perf_counter()
        result = show(*args)
        part_times[part] = time.perf_counter() - start
        return result

    run_dir = RUNS_DIR / run
    start = time.perf_counter()
    epochs = load_epochs(run_dir)
    model = YOLO(run_dir / "weights" / "best.pt")
    previous = previous or previous_run(run)
    part_times["loading the model"] = time.perf_counter() - start

    best = timed("1. best epoch", show_best_epoch, epochs)
    advice = timed("2. training length", show_training_length, epochs)
    counts, card_scores, fake_scores, missed, timings = timed("3. mistakes", show_mistakes, run_dir, model)
    advice += timed("4. confidence cut-off", show_cutoff, card_scores, fake_scores)
    found = timed("5. comparison", show_comparison, run, previous)
    timed("6. chart", show_chart, epochs, best)
    timed("7. prediction time", show_prediction_time, timings)
    show_runtime(run_dir, epochs, timings, part_times)

    if counts["missed"]:
        advice.insert(0, f"Missed {counts['missed']} real card(s), e.g. {missed[0]}: add more training photos like them.")
    if found and found[1] < found[0]:
        advice.insert(0, f"Finds fewer real cards than {previous} ({found[1]} vs {found[0]}): lower "
                         f"MAX_NO_CARD_PHOTOS in build_dataset.py (now {MAX_NO_CARD_PHOTOS}) or add card photos.")
    if counts["false alarms"]:
        advice.append(f"{counts['false alarms']} false alarm(s): add more no-card photos like them (see mistakes/).")
    if counts["fake outlined"]:
        advice.append(f"Outlined fake paper next to a card {counts['fake outlined']} time(s): "
                      "make more card-plus-fake photos (make_fake_cards.py).")
    show_suggestions(advice)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "card_seg_fakes", sys.argv[2] if len(sys.argv) > 2 else None)
