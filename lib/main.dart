import 'dart:async';
import 'dart:io' show Platform;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'detection/card_detector.dart';
import 'detection/card_overlay.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Card detection maps camera frames to the screen assuming portrait.
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const CardDetectorApp());
}

class CardDetectorApp extends StatelessWidget {
  const CardDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Card Detector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const CameraPage(),
    );
  }
}

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

  Future<void> _processFrame(CameraController controller, CameraImage image) async {
    final detector = _detector;
    if (detector == null || detector.isBusy) return;
    try {
      final detection = await detector.detect(
        image,
        rotationDegrees: controller.description.sensorOrientation,
      );
      if (!mounted || _controller != controller || _capturedImage != null) return;
      if (detection == null && _detection == null) return;
      setState(() => _detection = detection);
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
        // The formats the card detector reads.
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
      // Taking a photo while frames are streaming fails on some phones.
      if (controller.value.isStreamingImages) await controller.stopImageStream();
      final image = await controller.takePicture();
      final bytes = await image.readAsBytes();
      if (!mounted) return;
      setState(() {
        _capturedImage = bytes;
        _detection = null;
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
  });

  final CameraController controller;
  final CardDetection? detection;

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Keep the detection outline aligned with the camera image.
        CameraPreview(
          controller,
          child: CustomPaint(painter: CardOverlayPainter(detection)),
        ),
        Positioned(
          bottom: 24,
          left: 16,
          right: 16,
          child: Text(
            detection != null ? 'Card detected' : 'Position the card in view',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: detection != null ? const Color(0xFF00E676) : Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w600,
              shadows: [Shadow(color: Colors.black54, blurRadius: 4)],
            ),
          ),
        ),
      ],
    );
  }
}
