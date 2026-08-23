import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
from io import BytesIO
import base64
from ultralytics import YOLO

app = Flask(__name__)

# Variables globales
model_specialized = None
model_generic = None

def load_models():
    """Carga ambos modelos YOLO"""
    global model_specialized, model_generic

    if model_specialized is None or model_generic is None:
        try:
            # Modelo especializado (equipamiento militar)
            model_paths = ["./best.pt", "/app/best.pt", "best.pt"]
            model_path = None

            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✅ Modelo especializado encontrado en: {model_path}")
                    break

            if not model_path:
                raise FileNotFoundError("best.pt no encontrado")

            print("⏳ Cargando modelo especializado...")
            model_specialized = YOLO(model_path)
            print("✅ Modelo especializado cargado")

            # Modelo genérico (detección de personas)
            print("⏳ Cargando modelo genérico...")
            model_generic = YOLO('yolov8n.pt')
            print("✅ Modelo genérico cargado")

        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            raise

print("🚀 Modelos se cargarán bajo demanda")

REQUIRED_CLASSES = {"armaP", "botas", "buff", "casco", "chaleco", "gafas", "uniforme"}
CLASS_NAMES = {0: "armaP", 1: "botas", 2: "buff", 3: "casco", 4: "chaleco", 5: "gafas", 6: "uniforme"}

# Umbrales inteligentes por clase (BALANCEADOS para máxima detección)
CLASS_CONFIDENCES = {
    "gafas": 0.45,      # Detección óptima de gafas
    "chaleco": 0.45,    # Detección óptima de chaleco
    "botas": 0.45,      # Detección óptima de botas
    "casco": 0.25,      # Bajado para detectar cascos con cualquier ángulo
    "armaP": 0.20,      # Muy bajo para armas
    "uniforme": 0.25,   # Bajado para detectar uniformes mejor
    "buff": 0.45,       # Detección óptima de buff
}

def detect_uniform_hsv(image, bbox=None):
    """
    Detecta uniforme usando análisis de color HSV (SOLO verde/caqui real)
    Excluye azul, negro y otros colores
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        roi = image[int(y1):int(y2), int(x1):int(x2)]
    else:
        roi = image

    # Convertir a HSV (en OpenCV: H 0-180, S 0-255, V 0-255)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # RANGO ESTRICTO: Solo verde real (no azul, no negro)
    # Verde puro: H 40-80 con saturación media-alta
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Caqui/marrón: H 12-25 con saturación media
    lower_brown = np.array([12, 40, 40])
    upper_brown = np.array([25, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Combinar rangos
    mask_combined = cv2.bitwise_or(mask_green, mask_brown)

    # Calcular porcentaje de píxeles con color militar
    total_pixels = mask_combined.size
    military_pixels = cv2.countNonZero(mask_combined)
    percentage = (military_pixels / total_pixels) * 100

    # Más exigente: 20% mínimo para detectar uniforme
    return percentage > 20.0, percentage

def detect_in_crops(image, person_bbox):
    """
    Aplica recortes estratégicos para detectar mejor objetos pequeños
    """
    x1, y1, x2, y2 = person_bbox
    h = y2 - y1
    w = x2 - x1

    detected_classes = {}

    # 1. Imagen completa de la persona
    crop_full = image[int(y1):int(y2), int(x1):int(x2)]
    results_full = model_specialized.predict(crop_full, conf=0.15, verbose=False)

    # 2. Mitad superior (75% de arriba) - cabeza y torso
    crop_top = image[int(y1):int(y1 + h*0.75), int(x1):int(x2)]
    results_top = model_specialized.predict(crop_top, conf=0.15, verbose=False)

    # 3. Mitad inferior (70% de abajo) - armas y botas
    crop_bottom = image[int(y2 - h*0.70):int(y2), int(x1):int(x2)]
    results_bottom = model_specialized.predict(crop_bottom, conf=0.15, verbose=False)

    # Procesar resultados
    for results in [results_full, results_top, results_bottom]:
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names.get(class_id, "unknown")

                    # Aplicar umbral específico por clase
                    threshold = CLASS_CONFIDENCES.get(class_name, 0.3)

                    if class_name in CLASS_NAMES.values() and confidence >= threshold:
                        if class_name not in detected_classes:
                            detected_classes[class_name] = confidence
                        else:
                            detected_classes[class_name] = max(detected_classes[class_name], confidence)

    return detected_classes

@app.route('/detect', methods=['POST'])
def detect():
    """
    Pipeline completo: Detecta personas, aplica recortes estratégicos,
    usa umbrales inteligentes, análisis HSV para uniforme
    """
    try:
        # Cargar modelos si no están cargados
        if model_specialized is None or model_generic is None:
            print("🔄 Cargando modelos por primera vez...")
            load_models()

        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data'}), 400

        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400

        print(f"🔍 Imagen recibida, tamaño base64: {len(image_base64)} bytes")

        # Decodificar imagen
        try:
            image_bytes = base64.b64decode(image_base64)
            print(f"✅ Decodificación exitosa: {len(image_bytes)} bytes")
        except Exception as e:
            print(f"❌ Error decodificando base64: {e}")
            return jsonify({'error': f'Base64 decode error: {e}'}), 400

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print(f"⚠️ cv2.imdecode falló, intentando con PIL...")
            try:
                from PIL import Image as PILImage
                pil_image = PILImage.open(BytesIO(image_bytes))
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                print(f"✅ Imagen procesada con PIL: {image.shape}")
            except Exception as e:
                print(f"❌ Tampoco funcionó con PIL: {e}")
                return jsonify({'error': f'Invalid image format: {e}'}), 400
        else:
            print(f"✅ Imagen procesada con cv2: {image.shape}")

        # PASO 1: Detectar personas con modelo genérico
        print("🔍 Paso 1: Detectando personas...")
        results_persons = model_generic.predict(image, classes=0, conf=0.5, verbose=False)  # clase 0 = persona

        detected_classes = {}
        person_count = 0

        if results_persons and len(results_persons) > 0:
            result = results_persons[0]
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"✅ Detectadas {len(result.boxes)} persona(s)")

                for box in result.boxes:
                    person_count += 1
                    person_bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    print(f"\n👤 Analizando Persona {person_count}...")

                    # PASO 2: Aplicar recortes estratégicos
                    print("📐 Paso 2: Aplicando recortes estratégicos...")
                    detected_in_crops = detect_in_crops(image, person_bbox)

                    # Fusionar detecciones
                    for class_name, conf in detected_in_crops.items():
                        if class_name not in detected_classes:
                            detected_classes[class_name] = conf
                        else:
                            detected_classes[class_name] = max(detected_classes[class_name], conf)

                    # PASO 3: Análisis HSV para uniforme
                    print("🎨 Paso 3: Analizando color para uniforme...")
                    has_uniform, uniform_percent = detect_uniform_hsv(image, person_bbox)
                    if has_uniform and "uniforme" not in detected_classes:
                        detected_classes["uniforme"] = 0.85  # Confianza alta si HSV lo detecta
                        print(f"✅ Uniforme detectado por HSV ({uniform_percent:.1f}% verde militar)")

                    print(f"✅ Equipos detectados en Persona {person_count}: {list(detected_in_crops.keys())}")
            else:
                print("⚠️ No se detectaron personas en la imagen")
        else:
            print("⚠️ No se detectaron personas en la imagen")

        # PASO 4: Determinar APTO/NO APTO
        detected_set = set(detected_classes.keys())
        missing_classes = list(REQUIRED_CLASSES - detected_set)
        is_apto = len(missing_classes) == 0

        print(f"\n📊 RESULTADO FINAL:")
        print(f"   Detectados: {list(detected_set)}")
        print(f"   Faltantes: {missing_classes}")
        print(f"   Estado: {'APTO ✅' if is_apto else 'NO APTO ❌'}")

        return jsonify({
            'apto': is_apto,
            'detected': detected_classes,
            'missing': missing_classes,
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
    print("POST /detect - Detectar equipo militar")
    print("GET /health - Estado del servidor")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
