import 'package:flutter/material.dart';
import 'dart:typed_data';
import 'dart:ui' as ui;
import '../equipment_classifier.dart';

class AROverlayScreen extends StatefulWidget {
  final Uint8List imageBytes;
  final EquipmentDetection? detection;
  final EquipmentClassifier classifier;

  const AROverlayScreen({
    Key? key,
    required this.imageBytes,
    required this.detection,
    required this.classifier,
  }) : super(key: key);

  @override
  State<AROverlayScreen> createState() => _AROverlayScreenState();
}

class _AROverlayScreenState extends State<AROverlayScreen> {
  late EquipmentDetection? _detection;
  bool _isLoading = true;
  String? _error;

  // Mapeo de clases a colores y posiciones relativas en el cuerpo
  final Map<String, Map<String, dynamic>> equipmentMap = {
    'casco': {'color': Colors.blue, 'region': 'cabeza', 'y': 0.15},
    'gafas': {'color': Colors.cyan, 'region': 'cara', 'y': 0.18},
    'buff': {'color': Colors.purple, 'region': 'cuello', 'y': 0.25},
    'chaleco': {'color': Colors.green, 'region': 'torso', 'y': 0.40},
    'uniforme': {'color': Colors.amber, 'region': 'torso', 'y': 0.45},
    'guantes': {'color': Colors.pink, 'region': 'manos', 'y': 0.50},
    'cinturon': {'color': Colors.brown, 'region': 'cintura', 'y': 0.55},
    'botas': {'color': Colors.grey, 'region': 'pies', 'y': 0.90},
    'armaP': {'color': Colors.red, 'region': 'cintura', 'y': 0.60},
    'fusil': {'color': Colors.deepOrange, 'region': 'mano', 'y': 0.45},
    'portafusil': {'color': Colors.orange, 'region': 'espalda', 'y': 0.50},
    'pantalon': {'color': Colors.indigo, 'region': 'piernas', 'y': 0.65},
    'cinturon_tactico': {'color': Colors.lime, 'region': 'cintura', 'y': 0.57},
    'buzo': {'color': Colors.teal, 'region': 'torso', 'y': 0.42},
  };

  // Notas del inspector por clase
  final Map<String, String> inspectorNotes = {
    'casco': 'Protección craneal obligatoria\nVerificar ajuste correcto',
    'gafas': 'Protección ocular requerida\nLentes anti-reflectantes',
    'buff': 'Armamento secundario visible\nDebe estar cubierto',
    'chaleco': 'Chaleco balístico obligatorio\nVerificar cobertura completa',
    'uniforme': 'Uniforme base requerido\nDebe estar limpio y en orden',
    'guantes': 'Protección de manos obligatoria\nVerificar integridad',
    'cinturon': 'Cinturón de carga requerido\nDebe sostener equipo',
    'botas': 'Botas tácticas obligatorias\nSuela antideslizante',
    'armaP': 'Arma principal visible\nDebe estar asegurada',
    'fusil': 'Fusil detectado\nVerificar serial y estado',
    'portafusil': 'Portafusil requerido\nDebe estar visible',
    'pantalon': 'Pantalón táctico requerido\nBolsillos de carga',
    'cinturon_tactico': 'Cinturón táctico con arnés\nVerificar sistemas de sujeción',
    'buzo': 'Buzo base o uniforme requerido\nColores permitidos: verde, gris',
  };

  @override
  void initState() {
    super.initState();
    _detection = widget.detection;

    if (_detection == null) {
      _processImage();
    } else {
      _isLoading = false;
    }
  }

  Future<void> _processImage() async {
    try {
      print('🔍 Procesando imagen en AROverlayScreen...');
      final detection = await widget.classifier.detect(widget.imageBytes);

      if (mounted) {
        setState(() {
          _detection = detection;
          _isLoading = false;
        });
        print('✅ Detección completada en AROverlayScreen');
      }
    } catch (e) {
      print('❌ Error procesando imagen: $e');
      if (mounted) {
        setState(() {
          _error = 'Error: $e';
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('AR Overlay - Detección'),
          centerTitle: true,
        ),
        body: const Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Detectando equipos...'),
            ],
          ),
        ),
      );
    }

    if (_error != null) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('AR Overlay - Error'),
          centerTitle: true,
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error, color: Colors.red, size: 64),
              const SizedBox(height: 16),
              Text(_error!),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Volver'),
              ),
            ],
          ),
        ),
      );
    }

    final detection = _detection;
    final detectedItems = detection?.detectedClasses.entries.toList() ?? [];

    return Scaffold(
      appBar: AppBar(
        title: const Text('AR Overlay - Detección'),
        centerTitle: true,
        backgroundColor: Colors.black87,
      ),
      body: Stack(
        children: [
          // Imagen con overlay
          SingleChildScrollView(
            child: Column(
              children: [
                // Imagen + AR overlay
                Container(
                  width: double.infinity,
                  color: Colors.black,
                  child: Stack(
                    children: [
                      // Imagen base
                      Image.memory(
                        widget.imageBytes,
                        fit: BoxFit.fitWidth,
                      ),

                      // Overlay AR con flechas y etiquetas
                      Positioned.fill(
                        child: CustomPaint(
                          painter: AROverlayPainter(
                            detectedItems: detectedItems,
                            equipmentMap: equipmentMap,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                // Panel de estado APTO/NO APTO
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  color: detection?.isApto == true
                      ? Colors.green.withOpacity(0.2)
                      : Colors.red.withOpacity(0.2),
                  child: Column(
                    children: [
                      Text(
                        detection?.isApto == true ? '✅ APTO' : '❌ NO APTO',
                        style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: detection?.isApto == true
                              ? Colors.green
                              : Colors.red,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        detection?.isApto == true
                            ? 'Tiene todo el equipo requerido'
                            : 'Faltan elementos de seguridad',
                        style: const TextStyle(fontSize: 14),
                      ),
                    ],
                  ),
                ),

                // Lista de equipos detectados con notas
                Container(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'EQUIPOS DETECTADOS',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.black87,
                        ),
                      ),
                      const SizedBox(height: 12),
                      ...detectedItems.map((entry) {
                        final equipment = entry.key;
                        final confidence = entry.value;
                        final color = equipmentMap[equipment]?['color'] ??
                            Colors.grey;

                        return Container(
                          margin: const EdgeInsets.only(bottom: 12),
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: (color as Color).withOpacity(0.1),
                            borderLeft: BorderSide(
                              color: color as Color,
                              width: 4,
                            ),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(
                                    '✓ ${equipment.toUpperCase()}',
                                    style: TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.bold,
                                      color: color as Color,
                                    ),
                                  ),
                                  Text(
                                    '${(confidence * 100).toStringAsFixed(1)}%',
                                    style: const TextStyle(
                                      fontSize: 12,
                                      color: Colors.grey,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 8),
                              Text(
                                inspectorNotes[equipment] ??
                                    'Equipo detectado correctamente',
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: Colors.black54,
                                  height: 1.4,
                                ),
                              ),
                            ],
                          ),
                        );
                      }).toList(),
                      if (detectedItems.isEmpty)
                        const Text(
                          'No se detectó equipo',
                          style: TextStyle(color: Colors.grey),
                        ),
                    ],
                  ),
                ),

                // Equipos faltantes
                if (detection?.missingClasses.isNotEmpty ?? false)
                  Container(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'EQUIPOS FALTANTES',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.red,
                          ),
                        ),
                        const SizedBox(height: 12),
                        ...((detection?.missingClasses ?? []).map((item) {
                          return Container(
                            margin: const EdgeInsets.only(bottom: 12),
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.red.withOpacity(0.1),
                              border: Border.all(
                                color: Colors.red,
                              ),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    const Icon(Icons.close,
                                        color: Colors.red, size: 20),
                                    const SizedBox(width: 8),
                                    Text(
                                      item.toUpperCase(),
                                      style: const TextStyle(
                                        fontSize: 14,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.red,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  inspectorNotes[item] ??
                                      'Equipo requerido no detectado',
                                  style: const TextStyle(
                                    fontSize: 12,
                                    color: Colors.red,
                                    height: 1.4,
                                  ),
                                ),
                              ],
                            ),
                          );
                        }).toList()),
                      ],
                    ),
                  ),

                // Botones
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      SizedBox(
                        width: double.infinity,
                        height: 48,
                        child: ElevatedButton(
                          onPressed: () {
                            Navigator.pop(context);
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF1976D2),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                          child: const Text(
                            'Volver a Capturar',
                            style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(height: 12),
                      SizedBox(
                        width: double.infinity,
                        height: 48,
                        child: OutlinedButton(
                          onPressed: () {
                            Navigator.of(context).pushNamedAndRemoveUntil(
                              '/',
                              (route) => false,
                            );
                          },
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(
                              color: Color(0xFF1976D2),
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                          child: const Text(
                            'Salir',
                            style: TextStyle(
                              color: Color(0xFF1976D2),
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Custom painter para dibujar flechas y etiquetas AR
class AROverlayPainter extends CustomPainter {
  final List<MapEntry<String, double>> detectedItems;
  final Map<String, Map<String, dynamic>> equipmentMap;

  AROverlayPainter({
    required this.detectedItems,
    required this.equipmentMap,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Dibujar flechas y etiquetas para cada equipo detectado
    for (int i = 0; i < detectedItems.length; i++) {
      final equipment = detectedItems[i].key;
      final data = equipmentMap[equipment];

      if (data == null) continue;

      final color = data['color'] as Color;
      final yPos = (data['y'] as double) * size.height;

      // Posición horizontal distribuida
      final xPos = (size.width / (detectedItems.length + 1)) * (i + 1);

      // Dibujar flecha
      _drawArrow(canvas, Offset(xPos, 20), Offset(xPos, yPos), color, size);

      // Dibujar etiqueta
      _drawLabel(canvas, equipment, Offset(xPos, 10), color, size);

      // Dibujar círculo en punto de destino
      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;
      canvas.drawCircle(Offset(xPos, yPos), 8, paint);
    }
  }

  void _drawArrow(Canvas canvas, Offset start, Offset end, Color color, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    // Línea principal
    canvas.drawLine(start, end, paint);

    // Punta de flecha
    const arrowSize = 12.0;
    final dx = end.dx - start.dx;
    final dy = end.dy - start.dy;
    final angle = dy > 0 ? 0.5236 : -0.5236; // 30 grados en radianes

    final path = Path();
    path.moveTo(end.dx, end.dy);
    path.lineTo(
      end.dx - arrowSize * (dy / (dy.abs() + 0.01)) * 0.866,
      end.dy - arrowSize * 0.5,
    );
    path.lineTo(
      end.dx + arrowSize * (dy / (dy.abs() + 0.01)) * 0.866,
      end.dy - arrowSize * 0.5,
    );
    path.close();

    canvas.drawPath(path, paint..style = PaintingStyle.fill);
  }

  void _drawLabel(Canvas canvas, String text, Offset offset, Color color, Size size) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: text.toUpperCase(),
        style: TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          backgroundColor: color.withOpacity(0.8),
        ),
      ),
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(
        offset.dx - textPainter.width / 2,
        offset.dy,
      ),
    );
  }

  @override
  bool shouldRepaint(AROverlayPainter oldDelegate) {
    return oldDelegate.detectedItems.length != detectedItems.length;
  }
}
