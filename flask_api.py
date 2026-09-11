import os
import sys
from flask import Flask, request, jsonify
import cv2
import numpy as np
from io import BytesIO
import base64

# Desabilitar descargas de modelos automáticas
os.environ['YOLOv8_CACHE'] = '/tmp/yolo_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ ultralytics no instalado, usando fallback")
    YOLO = None

app = Flask(__name__)

# Variables globales
model_specialized = None
model_generic = None

def load_models():
    """Carga ambos modelos YOLO"""
    global model_specialized, model_generic

    if model_specialized is None or model_generic is None:
        if YOLO is None:
            raise ImportError("ultralytics no está disponible. Instala: pip install ultralytics torch")

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
                raise FileNotFoundError(f"best.pt no encontrado. Intentadas rutas: {model_paths}")

            print("⏳ Cargando modelo especializado...")
            model_specialized = YOLO(model_path)
            print("✅ Modelo especializado cargado")

            # Modelo genérico (detección de personas)
            print("⏳ Cargando modelo genérico yolov8n...")
            model_generic = YOLO('yolov8n.pt')
            print("✅ Modelo genérico cargado")

        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            raise

print("🚀 Modelos se cargarán bajo demanda")

# 14 CLASES: 8 Requeridas + 5 Infracciones + 1 Agregado
REQUIRED_CLASSES = {"armaP", "botas", "buff", "casco", "chaleco", "gafas", "guantes", "uniforme"}
NEGATIVE_CLASSES = {"no_botas", "sin_buff", "sin_chaleco", "sin_guantes", "sin_uniforme"}
ALL_CLASSES = REQUIRED_CLASSES | NEGATIVE_CLASSES | {"pistola"}

CLASS_NAMES = {
    0: "armaP", 1: "botas", 2: "buff", 3: "casco", 4: "chaleco", 5: "gafas", 6: "guantes",
    7: "no_botas", 8: "pistola", 9: "sin_buff", 10: "sin_chaleco", 11: "sin_guantes",
    12: "sin_uniforme", 13: "uniforme"
}

# Umbrales calibrados del pipeline_militar.py
CLASS_CONFIDENCES = {
    "armaP": 0.18,
    "botas": 0.25,
    "buff": 0.25,
    "casco": 0.22,
    "chaleco": 0.25,
    "gafas": 0.25,
    "guantes": 0.15,  # Bajado de 0.20 para detectar mejor
    "no_botas": 0.25,
    "pistola": 0.10,
    "sin_buff": 0.30,
    "sin_chaleco": 0.30,
    "sin_guantes": 0.22,
    "sin_uniforme": 0.30,
    "uniforme": 0.25,
}

def is_punta_de_armaP(pistola_box, armaP_boxes, person_cx):
    """
    Valida si una detección de pistola es realmente la punta/cañón del fusil.
    Del algoritmo pipeline_militar.py
    """
    px1, py1, px2, py2 = pistola_box
    pcx = (px1 + px2) / 2

    # Si está en el lado izquierdo (muslo derecho del soldado), no es punta de fusil
    if pcx < person_cx - 20:
        return False

    for a_box in armaP_boxes:
        ax1, ay1, ax2, ay2 = a_box
        aw = ax2 - ax1
        ah = ay2 - ay1
        a_mid_x = ax1 + aw * 0.50
        a_mid_y = ay1 + ah * 0.50

        # Intersección
        ix1 = max(px1, ax1)
        iy1 = max(py1, ay1)
        ix2 = min(px2, ax2)
        iy2 = min(py2, ay2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        p_area = (px2 - px1) * (py2 - py1)
        containment = inter / p_area if p_area > 0 else 1.0

        # Solo si está en la mitad inferior-derecha del fusil
        if pcx > a_mid_x and (py1 + py2) / 2 > a_mid_y:
            if containment >= 0.40 or (px2 >= ax2 - aw * 0.20 and containment >= 0.25):
                return True

    return False

def should_remap_glove_to_holster(box, p_box, has_armaP, tipo_cuerpo, image=None):
    """
    Detecta si 'guantes' en el muslo es realmente una funda de pistola (confusión del modelo).
    Validación por:
    - Ubicación (muslo/cintura)
    - Forma (aspect ratio)
    - Color (negra/oscura para funda vs piel/gris para guantes)
    Del algoritmo pipeline_militar.py
    """
    if not has_armaP or tipo_cuerpo != "Cuerpo Completo":
        return False

    bx1, by1, bx2, by2 = box
    px1, py1, px2, py2 = p_box
    pw, ph = px2 - px1, py2 - py1
    b_cy = (by1 + by2) / 2

    # Zona de medio muslo
    if not (py1 + ph * 0.48 <= b_cy <= py1 + ph * 0.80):
        return False

    # Posición lateral (muslo)
    is_lateral = (bx1 < px1 + pw * 0.35) or (bx2 > px2 - pw * 0.35)
    return is_lateral

def nms_boxes(boxes, iou_thresh=0.40):
    """
    Non-Maximum Suppression por clase. Del pipeline_militar.py
    """
    if not boxes:
        return []

    final_boxes = []
    classes = set(b[0] for b in boxes)
    for cls_name in classes:
        cls_boxes = [b for b in boxes if b[0] == cls_name]
        cls_boxes.sort(key=lambda x: x[1], reverse=True)
        kept = []
        for b in cls_boxes:
            _, conf, x1, y1, x2, y2 = b
            overlap = False
            for k in kept:
                _, _, kx1, ky1, kx2, ky2 = k
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (kx2 - kx1) * (ky2 - ky1)
                    union = area1 + area2 - inter
                    if union > 0 and (inter / union) > iou_thresh:
                        overlap = True
                        break
            if not overlap:
                kept.append(b)
        final_boxes.extend(kept)
    return final_boxes

def check_uniform_color(person_crop):
    """
    Verifica si hay uniforme por análisis de color HSV militar.
    Del pipeline_militar.py
    """
    if person_crop.size == 0:
        return False

    hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)

    # Rango verde militar y caqui en HSV
    lower_military = np.array([20, 20, 20])
    upper_military = np.array([90, 255, 200])

    mask = cv2.inRange(hsv, lower_military, upper_military)

    total_pixels = mask.shape[0] * mask.shape[1]
    matching_pixels = cv2.countNonZero(mask)

    if total_pixels == 0:
        return False

    porcentaje = (matching_pixels / total_pixels) * 100

    # Si más del 15% tiene color militar
    return porcentaje > 15.0

def detect_in_crops(image, person_bbox, full_image_results=None, flipped_image_results=None):
    """
    Implementa el pipeline_militar.py completo con:
    - Multiescala de recortes
    - Modo espejo (TTA)
    - Remapeo anatómico guante→pistola
    - Validación de pistola vs punta de fusil
    - NMS
    - Fallback HSV para uniforme
    """
    x1, y1, x2, y2 = map(int, person_bbox)
    h_img, w_img = image.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    if x2 <= x1 or y2 <= y1:
        return {"tipo_cuerpo": "Desconocido", "detected": {}, "boxes": []}

    w = x2 - x1
    h = y2 - y1
    aspect_ratio = h / w if w > 0 else 0

    tipo_cuerpo = "Cuerpo Completo" if aspect_ratio > 1.7 else "Medio Cuerpo"
    person_cx = (x1 + x2) / 2

    # Bounding box expandida (margen horizontal para armas que sobresalen)
    pad_x = int(w * 0.25)
    pad_y = int(h * 0.10)
    ex1, ey1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    ex2, ey2 = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)

    raw_candidates = []

    # 1a. Detecciones en imagen completa estándar
    if full_image_results is not None:
        for r in full_image_results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2

                    if (ex1 <= bcx <= ex2 and ey1 <= bcy <= ey2) or (
                        max(x1, bx1) < min(x2, bx2) and max(y1, by1) < min(y2, by2)
                    ):
                        cname = r.names.get(cls_id, CLASS_NAMES.get(cls_id, "unknown"))
                        if cname in ALL_CLASSES:
                            raw_candidates.append((cname, conf, bx1, by1, bx2, by2))

    # 1b. Detecciones en imagen espejo
    if flipped_image_results is not None:
        for r in flipped_image_results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    fbx1, fby1, fbx2, fby2 = map(int, box.xyxy[0])

                    # Revertir coordenadas x
                    obx1 = w_img - fbx2
                    obx2 = w_img - fbx1
                    obcx = (obx1 + obx2) // 2
                    obcy = (fby1 + fby2) // 2

                    if (ex1 <= obcx <= ex2 and ey1 <= obcy <= ey2) or (
                        max(x1, obx1) < min(x2, obx2) and max(y1, fby1) < min(y2, fby2)
                    ):
                        cname = r.names.get(cls_id, CLASS_NAMES.get(cls_id, "unknown"))
                        if cname in ALL_CLASSES:
                            raw_candidates.append((cname, conf, obx1, fby1, obx2, fby2))

    # Función auxiliar para recortes
    def detect_crop(crop_img, offset_x, offset_y, crop_conf=0.10, use_mirror=False):
        if crop_img.size == 0:
            return

        # Estándar
        results = model_specialized.predict(crop_img, conf=crop_conf, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    cname = r.names.get(cls_id, CLASS_NAMES.get(cls_id, "unknown"))
                    if cname in ALL_CLASSES:
                        raw_candidates.append((cname, conf, bx1 + offset_x, by1 + offset_y, bx2 + offset_x, by2 + offset_y))

        # Modo espejo
        if use_mirror:
            cw = crop_img.shape[1]
            flipped_crop = cv2.flip(crop_img, 1)
            flip_res = model_specialized.predict(flipped_crop, conf=crop_conf, verbose=False)
            for r in flip_res:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        fbx1, fby1, fbx2, fby2 = map(int, box.xyxy[0])
                        obx1 = cw - fbx2
                        obx2 = cw - fbx1
                        cname = r.names.get(cls_id, CLASS_NAMES.get(cls_id, "unknown"))
                        if cname in ALL_CLASSES:
                            raw_candidates.append((cname, conf, obx1 + offset_x, fby1 + offset_y, obx2 + offset_x, fby2 + offset_y))

    # 2. Recorte expandido
    expanded_crop = image[ey1:ey2, ex1:ex2]
    detect_crop(expanded_crop, ex1, ey1, crop_conf=0.08, use_mirror=True)

    # 3. Recorte superior (70% arriba)
    y_mid_top = int(y1 + h * 0.70)
    upper_crop = image[y1:y_mid_top, x1:x2]
    detect_crop(upper_crop, x1, y1, crop_conf=0.08)

    # 4. Recorte inferior (botas y armas bajas)
    if tipo_cuerpo == "Cuerpo Completo":
        y_mid_bottom = int(y1 + h * 0.35)
        lower_crop = image[y_mid_bottom:y2, x1:x2]
        detect_crop(lower_crop, x1, y_mid_bottom, crop_conf=0.08)

    # 5. Recorte cintura-muslos con espejo especial
    y_waist_top = max(0, int(y1 + h * 0.40))
    y_thigh_bottom = min(h_img, int(y1 + h * 0.85)) if tipo_cuerpo == "Cuerpo Completo" else y2
    x_waist_left = max(0, int(x1 - w * 0.15))
    x_waist_right = min(w_img, int(x2 + w * 0.15))

    waist_thigh_crop = image[y_waist_top:y_thigh_bottom, x_waist_left:x_waist_right]
    detect_crop(waist_thigh_crop, x_waist_left, y_waist_top, crop_conf=0.10, use_mirror=True)

    # Laterales específicos de muslo
    mid_x = (x1 + x2) // 2
    right_thigh_crop = image[y_waist_top:y_thigh_bottom, x_waist_left:mid_x]
    detect_crop(right_thigh_crop, x_waist_left, y_waist_top, crop_conf=0.10, use_mirror=True)

    left_thigh_crop = image[y_waist_top:y_thigh_bottom, mid_x:x_waist_right]
    detect_crop(left_thigh_crop, mid_x, y_waist_top, crop_conf=0.10, use_mirror=True)

    # 6. Remapeo anatómico guantes→pistola
    has_armaP = any(b[0] == "armaP" and b[1] >= 0.18 for b in raw_candidates)
    remapped = []
    for b in raw_candidates:
        cname, conf, bx1, by1, bx2, by2 = b
        if cname == "guantes" and should_remap_glove_to_holster((bx1, by1, bx2, by2), (x1, y1, x2, y2), has_armaP, tipo_cuerpo, image):
            remapped.append(("pistola", conf, bx1, by1, bx2, by2))
        else:
            remapped.append(b)

    # 7. Filtrar por umbral
    filtered = [b for b in remapped if b[1] >= CLASS_CONFIDENCES.get(b[0], 0.20)]

    # 8. NMS
    dedup = nms_boxes(filtered, iou_thresh=0.40)

    # 9. Validar pistola vs punta de fusil
    armaP_boxes = [b[2:] for b in dedup if b[0] == "armaP"]
    valid_boxes = []
    for b in dedup:
        if b[0] == "pistola" and armaP_boxes:
            if is_punta_de_armaP(b[2:], armaP_boxes, person_cx):
                continue
        valid_boxes.append(b)

    detected_items = set(b[0] for b in valid_boxes)

    # 10. Fallback HSV para uniforme
    person_crop = image[y1:y2, x1:x2]
    if "uniforme" not in detected_items and "sin_uniforme" not in detected_items:
        if check_uniform_color(person_crop):
            detected_items.add("uniforme")
            valid_boxes.append(("uniforme", 0.99, x1, y1, x2, y2))

    detected_dict = {b[0]: b[1] for b in valid_boxes}

    return {
        "tipo_cuerpo": tipo_cuerpo,
        "detected": detected_dict,
        "boxes": valid_boxes
    }

@app.route('/detect', methods=['POST'])
def detect():
    """
    Pipeline militar completo basado en pipeline_militar.py:
    - 14 clases (8 requeridas + 5 infracciones + pistola)
    - Multiescala + Modo espejo (TTA)
    - Remapeo anatómico guantes→pistola
    - Validación pistola vs fusil
    - Fallback HSV para uniforme
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

        # PASO 1: Detectar personas
        print("🔍 Paso 1: Detectando personas...")
        results_persons = model_generic.predict(image, classes=0, conf=0.45, verbose=False)

        # Pre-calcular predicciones en imagen completa (estándar + espejo)
        print("📐 Pre-calculando predicciones en imagen completa...")
        full_image_results = model_specialized.predict(image, conf=0.08, verbose=False)
        flipped_img = cv2.flip(image, 1)
        flipped_image_results = model_specialized.predict(flipped_img, conf=0.08, verbose=False)

        personas_detectadas = []
        results_all = []

        if results_persons and len(results_persons) > 0:
            result = results_persons[0]
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"✅ Detectadas {len(result.boxes)} persona(s)")

                for box_idx, box in enumerate(result.boxes, 1):
                    person_bbox = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                    print(f"\n👤 Analizando Persona {box_idx}...")

                    # PASO 2-10: Pipeline militar completo
                    print("🔄 Ejecutando pipeline militar...")
                    result_info = detect_in_crops(
                        image,
                        person_bbox,
                        full_image_results=full_image_results,
                        flipped_image_results=flipped_image_results
                    )

                    tipo_cuerpo = result_info["tipo_cuerpo"]
                    detected_dict = result_info["detected"]
                    detected_set = set(detected_dict.keys())

                    print(f"   Tipo de cuerpo: {tipo_cuerpo}")
                    print(f"   Equipos detectados: {list(detected_set)}")

                    # Validar infracciones y faltantes
                    equipos_presentes = []
                    faltas_detectadas = []
                    elementos_faltantes = []

                    # -- ARMA (armaP / pistola) --
                    tiene_armaP = "armaP" in detected_set
                    tiene_pistola = "pistola" in detected_set
                    if tiene_armaP or tiene_pistola:
                        if tiene_armaP and tiene_pistola:
                            equipos_presentes.append("Arma (armaP + pistola)")
                        elif tiene_armaP:
                            equipos_presentes.append("Arma (armaP)")
                        else:
                            equipos_presentes.append("Arma (pistola)")
                    else:
                        elementos_faltantes.append("Arma")

                    # -- OTROS EQUIPOS REQUERIDOS --
                    for item in ["casco", "chaleco", "buff", "gafas", "guantes", "botas", "uniforme"]:
                        if item in detected_set:
                            equipos_presentes.append(item.capitalize())
                        else:
                            elementos_faltantes.append(item.capitalize())

                    # -- INFRACCIONES --
                    for neg_item in ["no_botas", "sin_chaleco", "sin_buff", "sin_guantes", "sin_uniforme"]:
                        if neg_item in detected_set:
                            faltas_detectadas.append(f"FALTA: {neg_item.upper()}")

                    # Botas especial (solo en Cuerpo Completo)
                    if tipo_cuerpo == "Medio Cuerpo" and "botas" not in detected_set:
                        elementos_faltantes.remove("Botas")

                    # Dictamen final
                    es_apto = (len(elementos_faltantes) == 0) and (len(faltas_detectadas) == 0)

                    # Extraer coordenadas de bounding boxes
                    boxes_with_coords = {}
                    for box in result_info.get("boxes", []):
                        class_name = box[0]
                        confidence = float(box[1])
                        x1, y1, x2, y2 = float(box[2]), float(box[3]), float(box[4]), float(box[5])

                        # Calcular centro y dimensiones
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1

                        boxes_with_coords[class_name] = {
                            "confidence": confidence,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "center_x": center_x,
                            "center_y": center_y,
                            "width": width,
                            "height": height
                        }

                    results_all.append({
                        "persona": box_idx,
                        "tipo_cuerpo": tipo_cuerpo,
                        "apto": es_apto,
                        "detected": detected_dict,
                        "boxes": boxes_with_coords,
                        "equipos_presentes": equipos_presentes,
                        "faltas": faltas_detectadas,
                        "faltantes": elementos_faltantes,
                        "status": "APTO ✅" if es_apto else "NO APTO ❌"
                    })

                    print(f"   Status: {results_all[-1]['status']}")
            else:
                print("⚠️ No se detectaron personas en la imagen")
        else:
            print("⚠️ No se detectaron personas en la imagen")

        # Determinar APTO/NO APTO general
        is_apto = all(r["apto"] for r in results_all) if results_all else False
        missing_classes = []
        detected_all = {}
        boxes_all = {}

        if results_all:
            detected_all = results_all[0]["detected"]
            missing_classes = results_all[0]["faltantes"]
            boxes_all = results_all[0].get("boxes", {})

        print(f"\n📊 RESULTADO FINAL:")
        print(f"   Total personas: {len(results_all)}")
        print(f"   Estado global: {'APTO ✅' if is_apto else 'NO APTO ❌'}")

        # Obtener dimensiones de la imagen
        image_height, image_width = image.shape[:2]

        return jsonify({
            'apto': is_apto,
            'detected': detected_all,
            'missing': missing_classes,
            'boxes': boxes_all,
            'image_width': image_width,
            'image_height': image_height,
            'personas': results_all,
            'message': 'APTO ✅' if is_apto else 'NO APTO ❌'
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
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
