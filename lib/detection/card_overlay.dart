import 'package:flutter/material.dart';

import 'card_detection.dart';

enum CardFrameStatus {
  noCard,
  partiallyInFrame,
  fullyInFrame;

  Color get color => switch (this) {
    CardFrameStatus.noCard => Colors.white,
    CardFrameStatus.partiallyInFrame => Colors.red,
    CardFrameStatus.fullyInFrame => const Color(0xFF00E676),
  };
}

/// Draws the card-positioning frame, colored according to card detection.
///
/// The painter's area must show exactly the upright camera frame (not cropped),
/// because the card corners are fractions of that frame.
class CardOverlayPainter extends CustomPainter {
  const CardOverlayPainter(this.detection);

  final CardDetection? detection;

  static Rect frameRect(Size size) {
    final width = size.width * 0.86;
    final height = width / 1.58;
    return Rect.fromCenter(
      center: Offset(size.width / 2, size.height * 0.47),
      width: width,
      height: height,
    );
  }

  static CardFrameStatus statusFor(CardDetection? detection, Size size) {
    if (detection == null) return CardFrameStatus.noCard;
    final frame = frameRect(size).inflate(1);
    final isFullyInside = detection.corners.every(
      (corner) => frame.contains(
        Offset(corner.dx * size.width, corner.dy * size.height),
      ),
    );
    return isFullyInside
        ? CardFrameStatus.fullyInFrame
        : CardFrameStatus.partiallyInFrame;
  }

  @override
  void paint(Canvas canvas, Size size) {
    final detection = this.detection;
    final status = statusFor(detection, size);
    final color = status.color;

    _drawFrame(canvas, frameRect(size), color);
  }

  void _drawFrame(Canvas canvas, Rect rect, Color color) {
    final length = rect.width * 0.15;
    final path = Path()
      ..moveTo(rect.left + length, rect.top)
      ..lineTo(rect.left, rect.top)
      ..lineTo(rect.left, rect.top + length)
      ..moveTo(rect.right - length, rect.top)
      ..lineTo(rect.right, rect.top)
      ..lineTo(rect.right, rect.top + length)
      ..moveTo(rect.right, rect.bottom - length)
      ..lineTo(rect.right, rect.bottom)
      ..lineTo(rect.right - length, rect.bottom)
      ..moveTo(rect.left + length, rect.bottom)
      ..lineTo(rect.left, rect.bottom)
      ..lineTo(rect.left, rect.bottom - length);
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 6
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round,
    );
  }

  @override
  bool shouldRepaint(CardOverlayPainter oldDelegate) =>
      oldDelegate.detection != detection;
}
