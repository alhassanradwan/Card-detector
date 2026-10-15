# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "ultralytics==8.4.175",
#     "torch==2.14.1",
#     "tensorflow==2.19.0",
#     "tf_keras==2.19.0",
#     "onnx2tf>=1.26.3,<1.29.0",
#     "sng4onnx>=1.0.1",
#     "onnx_graphsurgeon>=0.3.26",
#     "ai-edge-litert>=1.2.0",
#     "onnx>=1.12.0,<2.0.0",
#     "onnxslim>=0.1.82",
#     "onnxruntime",
#     "protobuf>=5",
# ]
# ///
"""Export the trained card model to TensorFlow Lite (.tflite) for the phone app.

Ultralytics only exports .tflite directly on Linux/macOS, so on Windows this uses the
TensorFlow SavedModel export, which also writes .tflite files. The export tools are
listed above and installed by uv into a separate environment, not the project's .venv.

Usage:
    uv run yolo/export_tflite.py
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

YOLO_DIR = Path(__file__).parent
MODEL = YOLO_DIR / "runs" / "card_seg_fakes" / "weights" / "best.pt"
IMAGE_SIZE = 640  # must match the size the app feeds to the model
OUTPUT_DIR = YOLO_DIR / "export"


def main() -> None:
    saved_model_dir = Path(YOLO(MODEL).export(format="saved_model", imgsz=IMAGE_SIZE, device="cpu"))

    OUTPUT_DIR.mkdir(exist_ok=True)
    for tflite in sorted(saved_model_dir.glob("*.tflite")):
        shutil.copy2(tflite, OUTPUT_DIR / tflite.name)
        print(f"{tflite.name}: {tflite.stat().st_size / 1e6:.1f} MB -> {OUTPUT_DIR / tflite.name}")


if __name__ == "__main__":
    main()
