import 'dart:async';
import 'dart:io' show Platform;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'detection/card_detector.dart';
import 'detection/card_overlay.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  CameraController? _controller;
  Uint8List? _capturedImage;
  Object? _cameraError;
  bool _isInitializing = true;
  int _initializationGeneration = 0;
  CardDetector? _detector;
  CardDetection? _detection;
  int? _detectionDurationMs;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
    _loadDetector();
  }

  Future<void> _loadDetector() async {
    try {
      final detector = await CardDetector.create();
      if (!mounted) {
        detector.dispose();
        return;
      }
      _detector = detector;
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Card detection is unavailable: $error')),
      );
    }
  }

  /// Streams camera frames to the card detector. Frames that arrive while the
  /// previous one is still being processed are skipped.
  Future<void> _startDetection(CameraController controller) async {
    if (!controller.value.isInitialized || controller.value.isStreamingImages) {
      return;
    }
    await controller.startImageStream((image) => _processFrame(controller, image));
  }

  Future<void> _processFrame(
    CameraController controller,
    CameraImage image,
  ) async {
    final detector = _detector;
    if (detector == null || detector.isBusy) return;
    try {
      final stopwatch = Stopwatch()..start();
      final detection = await detector.detect(
        image,
        rotationDegrees: controller.description.sensorOrientation,
      );
      stopwatch.stop();
      if (!mounted || _controller != controller || _capturedImage != null) return;
      setState(() {
        _detection = detection;
        _detectionDurationMs = stopwatch.elapsedMilliseconds;
      });
    } catch (error) {
      debugPrint('Card detection failed: $error');
    }
  }

  Future<void> _initializeCamera() async {
    final generation = ++_initializationGeneration;
    setState(() {
      _isInitializing = true;
      _cameraError = null;
    });

    CameraController? initializingController;
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw CameraException('NoCameraAvailable', 'No cameras were found.');
      }

      final camera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.yuv420
            : ImageFormatGroup.bgra8888,
      );
      initializingController = controller;
      await controller.initialize();

      if (!mounted || generation != _initializationGeneration) {
        unawaited(controller.dispose());
        return;
      }

      setState(() {
        _controller = controller;
        _isInitializing = false;
      });
      await _startDetection(controller);
    } catch (error) {
      if (initializingController != null) {
        unawaited(initializingController.dispose());
      }
      if (!mounted || generation != _initializationGeneration) return;
      setState(() {
        _cameraError = error;
        _isInitializing = false;
      });
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (state == AppLifecycleState.inactive && controller != null) {
      ++_initializationGeneration;
      setState(() {
        _controller = null;
        _capturedImage = null;
        _detection = null;
        _detectionDurationMs = null;
        _isInitializing = false;
      });
      unawaited(controller.dispose());
    } else if (state == AppLifecycleState.resumed && controller == null) {
      _initializeCamera();
    }
  }

  Future<void> _captureImage() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;

    try {
      if (controller.value.isStreamingImages) await controller.stopImageStream();
      final image = await controller.takePicture();
      final bytes = await image.readAsBytes();
      if (!mounted) return;
      setState(() {
        _capturedImage = bytes;
        _detection = null;
        _detectionDurationMs = null;
      });
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not capture image: $error')),
      );
      unawaited(_startDetection(controller));
    }
  }

  void _retake() {
    setState(() => _capturedImage = null);
    final controller = _controller;
    if (controller != null) unawaited(_startDetection(controller));
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    final controller = _controller;
    if (controller != null) unawaited(controller.dispose());
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    final capturedImage = _capturedImage;

    return Scaffold(
      appBar: AppBar(title: const Text('Card Detector')),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: capturedImage != null
                    ? Image.memory(capturedImage, fit: BoxFit.contain)
                    : controller != null && controller.value.isInitialized
                    ? _CameraPreviewLayer(
                        controller: controller,
                        detection: _detection,
                        detectionDurationMs: _detectionDurationMs,
                      )
                    : _cameraContent(),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16),
              child: capturedImage != null
                  ? OutlinedButton.icon(
                      onPressed: _retake,
                      icon: const Icon(Icons.refresh),
                      label: const Text('Retake'),
                    )
                  : controller != null && controller.value.isInitialized
                  ? FilledButton.icon(
                      onPressed: _captureImage,
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Capture'),
                    )
                  : const SizedBox(height: 48),
            ),
          ],
        ),
      ),
    );
  }

  Widget _cameraContent() {
    if (_isInitializing) {
      return const CircularProgressIndicator();
    }

    final error = _cameraError;
    if (error != null) {
      return Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.no_photography, size: 48),
            const SizedBox(height: 16),
            const Text(
              'Unable to start the camera. Check camera access and try again.',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text('$error', textAlign: TextAlign.center),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: _initializeCamera,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    return const SizedBox.shrink();
  }
}

class _CameraPreviewLayer extends StatelessWidget {
  const _CameraPreviewLayer({
    required this.controller,
    required this.detection,
    required this.detectionDurationMs,
  });

  final CameraController controller;
  final CardDetection? detection;
  final int? detectionDurationMs;

  @override
  Widget build(BuildContext context) {
    return CameraPreview(
      controller,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final size = Size(constraints.maxWidth, constraints.maxHeight);
          final status = CardOverlayPainter.statusFor(detection, size);
          return Stack(
            fit: StackFit.expand,
            children: [
              CustomPaint(painter: CardOverlayPainter(detection)),
              Positioned(
                bottom: 24,
                left: 16,
                right: 16,
                child: _DetectionStatus(
                  status: status,
                  detectionDurationMs: detectionDurationMs,
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _DetectionStatus extends StatelessWidget {
  const _DetectionStatus({
    required this.status,
    required this.detectionDurationMs,
  });

  final CardFrameStatus status;
  final int? detectionDurationMs;

  @override
  Widget build(BuildContext context) {
    final (message, icon) = switch (status) {
      CardFrameStatus.noCard => ('put the card in the frame', Icons.center_focus_strong),
      CardFrameStatus.fullyInFrame => ('the card is detected', Icons.check_circle_outline),
      CardFrameStatus.partiallyInFrame => (
        'the card is not fully detected',
        Icons.error_outline,
      ),
    };
    final color = status.color;

    return Center(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.72),
          borderRadius: BorderRadius.circular(28),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: color, size: 22),
            const SizedBox(width: 10),
            Flexible(
              child: Text(
                detectionDurationMs == null
                    ? message
                    : '$message  ·  $detectionDurationMs ms',
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
