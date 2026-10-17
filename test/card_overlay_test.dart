import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:flutter_application_1/detection/card_detection.dart';
import 'package:flutter_application_1/detection/card_overlay.dart';

void main() {
  const previewSize = Size(400, 800);

  test('no detected card keeps the frame white', () {
    expect(
      CardOverlayPainter.statusFor(null, previewSize),
      CardFrameStatus.noCard,
    );
  });

  test('card fully inside the guide frame is green', () {
    final detection = CardDetection(
      corners: const [
        Offset(0.25, 0.4),
        Offset(0.75, 0.4),
        Offset(0.75, 0.55),
        Offset(0.25, 0.55),
      ],
      confidence: 0.9,
      frameSize: previewSize,
    );

    expect(
      CardOverlayPainter.statusFor(detection, previewSize),
      CardFrameStatus.fullyInFrame,
    );
  });

  test('detected card extending outside the guide frame is red', () {
    final detection = CardDetection(
      corners: const [
        Offset(0.02, 0.4),
        Offset(0.5, 0.4),
        Offset(0.5, 0.55),
        Offset(0.02, 0.55),
      ],
      confidence: 0.9,
      frameSize: previewSize,
    );

    expect(
      CardOverlayPainter.statusFor(detection, previewSize),
      CardFrameStatus.partiallyInFrame,
    );
  });
}
