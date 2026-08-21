import os
import torch
from flask import Flask, request, jsonify
import cv2
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

# Variables globales
model = None

def load_model():
    """Carga el modelo de detección"""
    global model
    if model is None:
        try:
            # Buscar best.pt en diferentes ubicaciones
            model_paths = ["./best.pt", "/app/best.pt", "best.pt"]
            model_path = None

            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✅ Modelo encontrado en: {model_path}")
                    break

            if not model_path:
                raise FileNotFoundError(f"best.pt no encontrado")

            print("⏳ Cargando modelo...")
            model = torch.jit.load(model_path)
            model.eval()
            print("✅ Modelo cargado correctamente")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise

print("🚀 Modelo se cargará bajo demanda")

REQUIRED_CLASSES = {"armaP", "botas", "buff", "casco", "chaleco", "gafas", "uniforme"}
CLASS_NAMES = {0: "armaP", 1: "botas", 2: "buff", 3: "casco", 4: "chaleco", 5: "gafas", 6: "uniforme"}

# Umbrales inteligentes por clase
CLASS_CONFIDENCES = {
    "gafas": 0.65,      # Alto para evitar falsos positivos
    "chaleco": 0.40,
    "botas": 0.40,
    "casco": 0.25,      # Bajado porque el casco se dificulta detectar
    "armaP": 0.15,      # Muy bajo porque las armas se deforman/ocultan mucho
    "uniforme": 0.30,   # Bajo porque el uniforme se deforma
    "buff": 0.50,       # Más alto para evitar falsos positivos en distancias cortas
}

def detect_uniform_hsv(image, bbox=None):
    """
    Detecta uniforme usando análisis de color HSV (verde militar/caqui)
    Busca píxeles con tonos verdes/cafés típicos del camuflaje
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        roi = image[int(y1):int(y2), int(x1):int(x2)]
    else:
        roi = image

    # Convertir a HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Rango para verde militar/caqui/olivo
    lower_green1 = np.array([35, 30, 30])
    upper_green1 = np.array([90, 255, 255])

    lower_brown = np.array([10, 30, 30])
    upper_brown = np.array([25, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green1, upper_green1)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_combined = cv2.bitwise_or(mask_green, mask_brown)

    # Calcular porcentaje de píxeles con color militar
    total_pixels = mask_combined.size
    military_pixels = cv2.countNonZero(mask_combined)
    percentage = (military_pixels / total_pixels) * 100

    # Si más del 15% tiene color militar, es uniforme
    return percentage > 15.0, percentage

def simple_detect(image):
    """
    Detección simple con el modelo YOLO
    """
    try:
        # Redimensionar imagen si es muy grande
        height, width = image.shape[:2]
        if max(height, width) > 640:
            scale = 640 / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        # Normalizar y convertir a tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        # Ejecutar modelo
        with torch.no_grad():
            predictions = model(img_tensor)

        detected_classes = {}

        # Procesar predicciones (formato puede variar)
        if isinstance(predictions, dict):
            if 'detections' in predictions:
                dets = predictions['detections']
                if dets is not None and len(dets) > 0:
                    for det in dets:
                        # det: [x1, y1, x2, y2, conf, class_id]
                        confidence = float(det[4])
                        class_id = int(det[5])
                        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

                        threshold = CLASS_CONFIDENCES.get(class_name, 0.3)
                        if confidence >= threshold:
                            if class_name not in detected_classes:
                                detected_classes[class_name] = confidence
                            else:
                                detected_classes[class_name] = max(detected_classes[class_name], confidence)

        return detected_classes
    except Exception as e:
        print(f"⚠️ Error en detección: {e}")
        return {}

@app.route('/detect', methods=['POST'])
def detect():
    """Detecta equipamiento militar en imagen"""
    try:
        load_model()

        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data'}), 400

        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400

        print(f"🔍 Procesando imagen ({len(image_base64)} bytes)...")

        # Decodificar
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        print(f"✅ Imagen decodificada: {image.shape}")

        # Detectar
        detected_classes = simple_detect(image)

        # Análisis HSV para uniforme
        has_uniform, pct = detect_uniform_hsv(image)
        if has_uniform and "uniforme" not in detected_classes:
            detected_classes["uniforme"] = 0.85

        # Determinar APTO
        detected_set = set(detected_classes.keys())
        missing = list(REQUIRED_CLASSES - detected_set)
        is_apto = len(missing) == 0

        print(f"📊 Detectados: {list(detected_set)}")
        print(f"📊 Faltantes: {missing}")
        print(f"📊 Estado: {'APTO ✅' if is_apto else 'NO APTO ❌'}")

        return jsonify({
            'apto': is_apto,
            'detected': detected_classes,
            'missing': missing,
            'message': 'APTO ✅' if is_apto else 'NO APTO ❌'
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'OK'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 Flask API iniciado en http://0.0.0.0:{port}")
    print("POST /detect - Detectar equipo")
    print("GET /health - Status")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
