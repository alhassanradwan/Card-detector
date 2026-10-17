import 'dart:ui' show Offset, Size;

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
