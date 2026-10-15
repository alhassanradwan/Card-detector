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
"""Check the app's .tflite model against best.pt on the validation photos.

Ultralytics' own val() misreads these .tflite files (it expects 0-1 box coordinates,
the export gives 640-pixel ones), so this reads the raw outputs by hand, the same way
the app does, and compares the top card's confidence, box and mask with best.pt.

Uses the same tools as export_tflite.py, so uv reuses that environment.

Usage:
    uv run yolo/check_tflite.py
"""

from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
PT_MODEL = ROOT / "yolo" / "runs" / "card_seg_fakes" / "weights" / "best.pt"
TFLITE_MODEL = ROOT / "assets" / "models" / "card_seg.tflite"
VAL_IMAGES = ROOT / "yolo" / "dataset" / "images" / "val"
SIZE = 640


def letterbox(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping the aspect ratio and pad to SIZE x SIZE with grey (as YOLO does)."""
    h, w = image.shape[:2]
    scale = SIZE / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    return canvas, scale, top, left


def main() -> None:
    pt = YOLO(PT_MODEL)
    interpreter = Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    outputs = interpreter.get_output_details()
    print("TFLite input:", interpreter.get_input_details()[0]["shape"].tolist())
    print("TFLite outputs:", [o["shape"].tolist() for o in outputs])

    for image_path in sorted(VAL_IMAGES.glob("*")):
        image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        canvas, scale, top, left = letterbox(image)
        interpreter.set_tensor(input_index, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255)
        interpreter.invoke()
        # detections: 4 box values (centre x, centre y, width, height in 640 px), 1 score, 32 mask coefficients
        det = next(interpreter.get_tensor(o["index"]) for o in outputs if o["shape"][1] == 37)[0]
        protos = next(interpreter.get_tensor(o["index"]) for o in outputs if o["shape"][-1] == 32)[0]

        best = int(det[4].argmax())
        cx, cy, bw, bh = det[:4, best]
        mask = 1 / (1 + np.exp(-(protos @ det[5:37, best])))  # 160 x 160
        crop = np.zeros_like(mask)
        crop[int(max((cy - bh / 2) / 4, 0)):int((cy + bh / 2) / 4), int(max((cx - bw / 2) / 4, 0)):int((cx + bw / 2) / 4)] = 1
        tflite_mask = cv2.resize(((mask * crop) > 0.5).astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        result = pt.predict(image, imgsz=SIZE, verbose=False, conf=0.01)[0]
        if result.boxes is None or result.masks is None or len(result.boxes) == 0:
            # No-card photo: best.pt finds nothing, the .tflite model should score low too
            print(f"{image_path.name[:45]:<45} score tflite {det[4, best]:.3f} / pt nothing found (no card)")
            continue
        pt_score = float(result.boxes.conf[0])
        pt_mask_small = (result.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
        pt_mask_full = cv2.resize(pt_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        nh, nw = round(h * scale), round(w * scale)
        pt_mask = np.zeros((SIZE, SIZE), np.uint8)
        pt_mask[top:top + nh, left:left + nw] = cv2.resize(pt_mask_full, (nw, nh), interpolation=cv2.INTER_NEAREST)
        iou = (tflite_mask & pt_mask).sum() / max((tflite_mask | pt_mask).sum(), 1)
        print(f"{image_path.name[:45]:<45} score tflite {det[4, best]:.3f} / pt {pt_score:.3f}   mask overlap {iou:.3f}")


if __name__ == "__main__":
    main()
