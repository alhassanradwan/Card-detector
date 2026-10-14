# Card Detector

The Flutter app provides a live camera preview and a manual still-image capture flow. The Python prototype contains the OpenCV and Ultralytics dependencies for developing card detection on a PC.

This folder contains two parts:

- **Flutter app** (`pubspec.yaml`, `lib/`, `android/`, `ios/`, ...) - the mobile app.
- **Python prototype** (`pyproject.toml`, `uv.lock`, `main.py`) - OpenCV and Ultralytics YOLO, used to build and test card detection on the PC.

## Flutter app

```
flutter pub get
flutter run
```

The Flutter app uses the `camera` plugin. Camera access is requested at runtime on
Android and declared for iOS and macOS. On macOS, camera access is also enabled
for the sandboxed app. Web browsers request camera access through their normal
permission prompt.

## Python prototype

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run python main.py
```
