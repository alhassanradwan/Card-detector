# Card Detector

Detects a card in a live camera view, checks zoom, brightness and angle, and automatically captures three frames 0.5 seconds apart.

This folder contains two parts:

- **Flutter app** (`pubspec.yaml`, `lib/`, `android/`, `ios/`, ...) - the mobile app.
- **Python prototype** (`pyproject.toml`, `uv.lock`, `main.py`) - OpenCV and Ultralytics YOLO, used to build and test card detection on the PC.

## Flutter app

```
flutter pub get
flutter run
```

## Python prototype

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run python main.py
```
