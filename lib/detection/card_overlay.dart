import 'package:flutter/material.dart';

import 'card_detector.dart';

/// Draws a green outline around a detected card.
///
/// The painter's area must show exactly the upright camera frame (not cropped),
/// because the card corners are fractions of that frame.
class CardOverlayPainter extends CustomPainter {
  const CardOverlayPainter(this.detection);

  final CardDetection? detection;

  static const _color = Color(0xFF00E676);

  @override
  void paint(Canvas canvas, Size size) {
    final detection = this.detection;
    if (detection == null) return;

    final corners = [
      for (final corner in detection.corners)
        Offset(corner.dx * size.width, corner.dy * size.height),
    ];
    final outline = Path()..addPolygon(corners, true);

    canvas.drawPath(outline, Paint()..color = _color.withValues(alpha: 0.18));
    canvas.drawPath(
      outline,
      Paint()
        ..color = _color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 4
        ..strokeJoin = StrokeJoin.round,
    );
    final cornerPaint = Paint()..color = _color;
    for (final corner in corners) {
      canvas.drawCircle(corner, 6, cornerPaint);
    }
  }

  @override
  bool shouldRepaint(CardOverlayPainter oldDelegate) =>
      oldDelegate.detection != detection;
}
