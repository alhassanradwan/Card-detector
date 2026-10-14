import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Offset, Size;

import 'package:camera/camera.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';

/// A card found in a camera frame.
class CardDetection {
  const CardDetection({
    required this.corners,
    required this.confidence,
    required this.frameSize,
  });

  /// The 4 card corners, clockwise from the top-left, as fractions (0..1) of
  /// the upright camera frame.
  final List<Offset> corners;

  /// How sure the model is that this is a card (0..1).
  final double confidence;

  /// Size of the upright camera frame in pixels.
  final Size frameSize;
}

/// Finds a card in camera frames with the YOLO segmentation model.
///
/// The model and all image processing run in a background isolate so the
/// camera preview stays smooth. Only one frame is processed at a time: check
/// [isBusy] and skip frames that arrive while it is true.
class CardDetector {
  CardDetector._(this._responses) {
    _responses.listen((message) {
      final pending = _pending;
      _pending = null;
      pending?.complete(message);
    });
  }

  static const modelAsset = 'assets/models/card_seg.tflite';

  final ReceivePort _responses;
  late final SendPort _requests;
  Completer<Object?>? _pending;
  bool _closed = false;

  /// Loads the model and starts the background isolate.
  static Future<CardDetector> create() async {
    final model = await rootBundle.load(modelAsset);
    final responses = ReceivePort();
    await Isolate.spawn(_workerMain, responses.sendPort, debugName: 'CardDetector');
    final detector = CardDetector._(responses);

    detector._requests = await detector._receive() as SendPort;
    final ready = detector._receive();
    detector._requests.send(
      TransferableTypedData.fromList([
        model.buffer.asUint8List(model.offsetInBytes, model.lengthInBytes),
      ]),
    );
    final reply = await ready;
    if (reply is String) {
      detector.dispose();
      throw StateError('Could not load the card model: $reply');
    }
    return detector;
  }

  /// True while a frame is being processed.
  bool get isBusy => _pending != null;

  /// Looks for a card in [image].
  ///
  /// [rotationDegrees] turns the camera frame upright (usually the camera's
  /// sensor orientation, 90 on most phones in portrait). Returns null when no
  /// card is found, or when the detector is busy or disposed.
  Future<CardDetection?> detect(
    CameraImage image, {
    required int rotationDegrees,
  }) async {
    if (_closed || isBusy) return null;

    final format = switch (image.format.group) {
      ImageFormatGroup.yuv420 => 'yuv420',
      ImageFormatGroup.bgra8888 => 'bgra8888',
      final other => throw UnsupportedError(
        'Camera format $other is not supported, use yuv420 or bgra8888.',
      ),
    };
    final reply = _receive();
    _requests.send([
      image.width,
      image.height,
      rotationDegrees,
      format,
      [
        for (final plane in image.planes)
          TransferableTypedData.fromList([plane.bytes]),
      ],
      [for (final plane in image.planes) plane.bytesPerRow],
      [for (final plane in image.planes) plane.bytesPerPixel ?? 1],
    ]);

    final result = await reply;
    if (result is String) throw StateError('Card detection failed: $result');
    if (result is! Float64List) return null;
    return CardDetection(
      confidence: result[0],
      corners: [
        for (var i = 0; i < 4; i++) Offset(result[1 + i * 2], result[2 + i * 2]),
      ],
      frameSize: Size(result[9], result[10]),
    );
  }

  /// Stops the background isolate and frees the model.
  void dispose() {
    if (_closed) return;
    _closed = true;
    _requests.send(null);
    _pending?.complete(null);
    _pending = null;
    _responses.close();
  }

  Future<Object?> _receive() {
    final pending = Completer<Object?>();
    _pending = pending;
    return pending.future;
  }
}

/// Entry point of the background isolate.
void _workerMain(SendPort replies) {
  final requests = ReceivePort();
  replies.send(requests.sendPort);
  _CardModel? model;

  requests.listen((message) {
    try {
      if (message is TransferableTypedData) {
        model = _CardModel(message.materialize().asUint8List());
        replies.send(true);
      } else if (message == null) {
        model?.close();
        requests.close();
        Isolate.exit();
      } else {
        replies.send(model!.process(message as List<Object?>));
      }
    } catch (error, stack) {
      replies.send('$error\n$stack');
    }
  });
}

/// Where the upright frame sits inside the square model input.
typedef _Letterbox = ({int left, int top, int width, int height});

/// Runs the model on one frame and turns its output into 4 card corners.
class _CardModel {
  _CardModel(Uint8List modelBytes)
    : _interpreter = Interpreter.fromBuffer(
        modelBytes,
        options: InterpreterOptions()..threads = 4,
      ) {
    _interpreter.allocateTensors();
    final outputs = _interpreter.getOutputTensors();
    _detectionsIndex = outputs.indexWhere(
      (t) => t.shape.length == 3 && t.shape[1] == _detectionValues,
    );
    _protosIndex = outputs.indexWhere(
      (t) => t.shape.length == 4 && t.shape[3] == _maskCoefficients,
    );
    if (_detectionsIndex < 0 || _protosIndex < 0) {
      throw StateError(
        'Unexpected model outputs: ${outputs.map((t) => t.shape).toList()}',
      );
    }
    _candidateCount = outputs[_detectionsIndex].shape[2];
  }

  static const _inputSize = 640; // the model was trained and exported at 640x640
  static const _protoSize = 160; // mask building blocks are 160x160 (640 / 4)
  static const _maskCoefficients = 32;
  static const _detectionValues = 4 + 1 + _maskCoefficients; // box, score, mask coefficients

  static const _minConfidence = 0.5; // ignore guesses the model is less than 50% sure about
  static const _duplicateOverlap = 0.5; // boxes overlapping more than this are the same object
  static const _maxObjectsChecked = 5; // distinct objects checked for a card shape
  static const _minCardArea = 0.02; // card must cover at least 2% of the frame
  // A card is 85.6 x 54 mm (ratio 1.59); allow for tilt and imperfect outlines.
  // This also rejects long thin rectangles such as the laptop's logo sticker.
  static const _minAspectRatio = 1.25;
  static const _maxAspectRatio = 2.1;

  final Interpreter _interpreter;
  late final int _detectionsIndex;
  late final int _protosIndex;
  late final int _candidateCount;

  void close() => _interpreter.close();

  /// Returns [confidence, x0, y0, x1, y1, x2, y2, x3, y3, frameWidth, frameHeight]
  /// or null when no card is found.
  Float64List? process(List<Object?> message) {
    final width = message[0] as int;
    final height = message[1] as int;
    final rotation = message[2] as int;
    final format = message[3] as String;
    final planes = [
      for (final plane in message[4] as List)
        (plane as TransferableTypedData).materialize().asUint8List(),
    ];
    final rowStrides = (message[5] as List).cast<int>();
    final pixelStrides = (message[6] as List).cast<int>();

    final rgb = format == 'yuv420'
        ? _yuvToRgb(width, height, planes, rowStrides, pixelStrides)
        : _bgraToRgb(width, height, planes[0], rowStrides[0]);
    final upright = _rotate(rgb, rotation);
    final letterbox = _fillModelInput(upright);
    final frameWidth = upright.cols.toDouble();
    final frameHeight = upright.rows.toDouble();
    if (!identical(upright, rgb)) upright.dispose();
    rgb.dispose();

    _interpreter.invoke();
    final detections = Float32List.sublistView(
      _interpreter.getOutputTensor(_detectionsIndex).data,
    );
    final protos = Float32List.sublistView(
      _interpreter.getOutputTensor(_protosIndex).data,
    );

    for (final index in _distinctObjects(detections)) {
      final corners = _cardCorners(detections, protos, index, letterbox);
      if (corners == null) continue;
      return Float64List.fromList([
        detections[4 * _candidateCount + index],
        for (final corner in corners) ...[
          (corner.x - letterbox.left) / letterbox.width,
          (corner.y - letterbox.top) / letterbox.height,
        ],
        frameWidth,
        frameHeight,
      ]);
    }
    return null;
  }

  /// Android YUV_420_888 (3 planes) or iOS NV12 (2 planes) -> RGB.
  cv.Mat _yuvToRgb(
    int width,
    int height,
    List<Uint8List> planes,
    List<int> rowStrides,
    List<int> pixelStrides,
  ) {
    final i420 = cv.Mat.zeros(height + height ~/ 2, width, cv.MatType.CV_8UC1);
    final out = i420.data;
    for (var row = 0; row < height; row++) {
      out.setRange(row * width, row * width + width, planes[0], row * rowStrides[0]);
    }

    // Colour is stored at half resolution, as separate U and V planes or one
    // interleaved UV plane; copy it into the planar I420 layout OpenCV expects.
    final interleaved = planes.length == 2;
    final uPlane = planes[1];
    final vPlane = interleaved ? planes[1] : planes[2];
    final vOffset = interleaved ? 1 : 0;
    final uvRowStride = rowStrides[1];
    final uvPixelStride = interleaved ? 2 : pixelStrides[1];
    final halfWidth = width ~/ 2;
    final halfHeight = height ~/ 2;
    var uIndex = width * height;
    var vIndex = uIndex + halfWidth * halfHeight;
    for (var row = 0; row < halfHeight; row++) {
      var source = row * uvRowStride;
      for (var col = 0; col < halfWidth; col++) {
        out[uIndex++] = uPlane[source];
        out[vIndex++] = vPlane[source + vOffset];
        source += uvPixelStride;
      }
    }

    final rgb = cv.cvtColor(i420, cv.COLOR_YUV2RGB_I420);
    i420.dispose();
    return rgb;
  }

  /// iOS BGRA -> RGB.
  cv.Mat _bgraToRgb(int width, int height, Uint8List bytes, int rowStride) {
    final bgra = cv.Mat.zeros(height, width, cv.MatType.CV_8UC4);
    final out = bgra.data;
    final rowBytes = width * 4;
    for (var row = 0; row < height; row++) {
      out.setRange(row * rowBytes, row * rowBytes + rowBytes, bytes, row * rowStride);
    }
    final rgb = cv.cvtColor(bgra, cv.COLOR_BGRA2RGB);
    bgra.dispose();
    return rgb;
  }

  cv.Mat _rotate(cv.Mat image, int degrees) {
    final code = switch (degrees % 360) {
      90 => cv.ROTATE_90_CLOCKWISE,
      180 => cv.ROTATE_180,
      270 => cv.ROTATE_90_COUNTERCLOCKWISE,
      _ => null,
    };
    return code == null ? image : cv.rotate(image, code);
  }

  /// Resizes the frame to fit 640x640 (keeping its shape), pads the rest with
  /// grey like YOLO does during training, and copies it into the model input.
  _Letterbox _fillModelInput(cv.Mat image) {
    final scale = math.min(_inputSize / image.cols, _inputSize / image.rows);
    final width = (image.cols * scale).round();
    final height = (image.rows * scale).round();
    final left = (_inputSize - width) ~/ 2;
    final top = (_inputSize - height) ~/ 2;

    final resized = cv.resize(image, (width, height), interpolation: cv.INTER_LINEAR);
    final padded = cv.copyMakeBorder(
      resized,
      top,
      _inputSize - height - top,
      left,
      _inputSize - width - left,
      cv.BORDER_CONSTANT,
      value: cv.Scalar(114, 114, 114, 0),
    );
    final input = padded.convertTo(cv.MatType.CV_32FC3, alpha: 1 / 255);
    _interpreter.getInputTensor(0).data = input.data;
    resized.dispose();
    padded.dispose();
    input.dispose();
    return (left: left, top: top, width: width, height: height);
  }

  /// Candidate indexes above [_minConfidence], best first, with duplicates of
  /// the same object removed.
  List<int> _distinctObjects(Float32List detections) {
    final n = _candidateCount;
    final candidates = [
      for (var i = 0; i < n; i++)
        if (detections[4 * n + i] >= _minConfidence) i,
    ]..sort((a, b) => detections[4 * n + b].compareTo(detections[4 * n + a]));

    final kept = <int>[];
    for (final candidate in candidates) {
      if (kept.length == _maxObjectsChecked) break;
      if (kept.every((k) => _boxOverlap(detections, candidate, k) < _duplicateOverlap)) {
        kept.add(candidate);
      }
    }
    return kept;
  }

  double _boxOverlap(Float32List d, int a, int b) {
    final n = _candidateCount;
    final ax0 = d[a] - d[2 * n + a] / 2, ax1 = d[a] + d[2 * n + a] / 2;
    final ay0 = d[n + a] - d[3 * n + a] / 2, ay1 = d[n + a] + d[3 * n + a] / 2;
    final bx0 = d[b] - d[2 * n + b] / 2, bx1 = d[b] + d[2 * n + b] / 2;
    final by0 = d[n + b] - d[3 * n + b] / 2, by1 = d[n + b] + d[3 * n + b] / 2;
    final overlapWidth = math.max(0.0, math.min(ax1, bx1) - math.max(ax0, bx0));
    final overlapHeight = math.max(0.0, math.min(ay1, by1) - math.max(ay0, by0));
    final overlap = overlapWidth * overlapHeight;
    final union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - overlap;
    return union <= 0 ? 0 : overlap / union;
  }

  /// Builds the object's mask, simplifies its outline to 4 corners (in model
  /// input pixels, clockwise from the top-left), and returns null if it is not
  /// card-shaped.
  List<math.Point<double>>? _cardCorners(
    Float32List detections,
    Float32List protos,
    int index,
    _Letterbox letterbox,
  ) {
    final mask = _objectMask(detections, protos, index);
    final (contours, hierarchy) = cv.findContours(
      mask,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );
    mask.dispose();
    hierarchy.dispose();

    cv.VecPoint? outline;
    var outlineArea = 0.0;
    for (final contour in contours) {
      final area = cv.contourArea(contour);
      if (area > outlineArea) {
        outlineArea = area;
        outline = contour;
      }
    }
    final frameArea = letterbox.width * letterbox.height;
    if (outline == null || outlineArea < _minCardArea * frameArea) {
      contours.dispose();
      return null;
    }

    final corners = _fourCorners(outline);
    contours.dispose();

    double side(int a, int b) => corners[a].distanceTo(corners[b]);
    final sideA = (side(0, 1) + side(2, 3)) / 2;
    final sideB = (side(1, 2) + side(3, 0)) / 2;
    final ratio = math.max(sideA, sideB) / math.max(math.min(sideA, sideB), 1e-6);
    if (ratio < _minAspectRatio || ratio > _maxAspectRatio) return null;
    return _clockwiseFromTopLeft(corners);
  }

  /// The object's mask at 640x640: each mask building block weighted by the
  /// object's 32 coefficients, limited to its box, smoothed and thresholded.
  cv.Mat _objectMask(Float32List detections, Float32List protos, int index) {
    final n = _candidateCount;
    final coefficients = Float32List.fromList([
      for (var k = 0; k < _maskCoefficients; k++) detections[(5 + k) * n + index],
    ]);
    const toProto = _protoSize / _inputSize;
    final cx = detections[index], cy = detections[n + index];
    final halfWidth = detections[2 * n + index] / 2;
    final halfHeight = detections[3 * n + index] / 2;
    final x0 = math.max(0, ((cx - halfWidth) * toProto).floor());
    final x1 = math.min(_protoSize, ((cx + halfWidth) * toProto).ceil());
    final y0 = math.max(0, ((cy - halfHeight) * toProto).floor());
    final y1 = math.min(_protoSize, ((cy + halfHeight) * toProto).ceil());

    final small = cv.Mat.zeros(_protoSize, _protoSize, cv.MatType.CV_32FC1);
    final values = Float32List.sublistView(small.data);
    for (var y = y0; y < y1; y++) {
      for (var x = x0; x < x1; x++) {
        final base = (y * _protoSize + x) * _maskCoefficients;
        var sum = 0.0;
        for (var k = 0; k < _maskCoefficients; k++) {
          sum += protos[base + k] * coefficients[k];
        }
        values[y * _protoSize + x] = 1 / (1 + math.exp(-sum));
      }
    }

    final large = cv.resize(small, (_inputSize, _inputSize), interpolation: cv.INTER_LINEAR);
    final (_, binary) = cv.threshold(large, 0.5, 255, cv.THRESH_BINARY);
    final mask = binary.convertTo(cv.MatType.CV_8UC1);
    small.dispose();
    large.dispose();
    binary.dispose();
    return mask;
  }

  /// Simplifies an outline to 4 corners; falls back to the smallest rotated
  /// rectangle around it when it does not simplify to a clean quadrilateral.
  List<math.Point<double>> _fourCorners(cv.VecPoint outline) {
    final approx = cv.approxPolyDP(outline, 0.02 * cv.arcLength(outline, true), true);
    try {
      if (approx.length == 4 && cv.isContourConvex(approx)) {
        return [for (final p in approx) math.Point(p.x.toDouble(), p.y.toDouble())];
      }
    } finally {
      approx.dispose();
    }
    final box = cv.boxPoints(cv.minAreaRect(outline));
    final corners = [for (final p in box) math.Point(p.x, p.y)];
    box.dispose();
    return corners;
  }

  /// Orders corners clockwise (on screen, where y points down) starting with
  /// the top-left one.
  List<math.Point<double>> _clockwiseFromTopLeft(List<math.Point<double>> corners) {
    final cx = corners.fold(0.0, (sum, p) => sum + p.x) / corners.length;
    final cy = corners.fold(0.0, (sum, p) => sum + p.y) / corners.length;
    final sorted = [...corners]
      ..sort((a, b) => math.atan2(a.y - cy, a.x - cx).compareTo(math.atan2(b.y - cy, b.x - cx)));
    var start = 0;
    for (var i = 1; i < sorted.length; i++) {
      if (sorted[i].x + sorted[i].y < sorted[start].x + sorted[start].y) start = i;
    }
    return [...sorted.sublist(start), ...sorted.sublist(0, start)];
  }
}
