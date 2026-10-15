"""Train a YOLO segmentation model to outline cards.

Run it from VS Code (Run Python File) or with:
    uv run python yolo/train.py

Results are saved to yolo/runs/<RUN_NAME>/ (the trained model is weights/best.pt).
When training ends, analyze_run.py prints a summary of the run in the terminal.
"""

from pathlib import Path

from ultralytics import YOLO

from analyze_run import analyze

YOLO_DIR = Path(__file__).parent

# Training settings - change these and run again
MODEL = "yolo26n-seg.pt"  # pre-trained nano segmentation model (downloads on first run)
EPOCHS = 100              # how many times to go through all training photos
IMAGE_SIZE = 640          # photos are resized to this many pixels for training
BATCH = 8                 # photos processed at the same time
DEVICE = 0                # 0 = first NVIDIA GPU (RTX 4050); use "cpu" to train without the GPU
RUN_NAME = "card_seg_fakes"  # results go to yolo/runs/<RUN_NAME>; use a new name to keep older runs


def main() -> None:
    model = YOLO(MODEL)
    model.train(
        data=str(YOLO_DIR / "dataset" / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=0,  # avoids data-loader problems on Windows
        project=str(YOLO_DIR / "runs"),
        name=RUN_NAME,
        exist_ok=True,  # running again with the same RUN_NAME overwrites it instead of making card_seg_552...
    )
    analyze(RUN_NAME)  # summary of the run: best epoch, mistakes, cut-off, comparison, suggestions


if __name__ == "__main__":
    main()
