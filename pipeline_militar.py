import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Configuraciones
DEFAULT_CUSTOM_MODEL_PATH = r"best.pt"
PERSON_MODEL_PATH = "yolov8n.pt"
OUTPUT_DIR = "Resultados"

CUSTOM_CLASSES = {0: "armaP", 1: "botas", 2: "buff", 3: "casco", 4: "chaleco", 5: "gafas", 6: "uniforme"}

# Colores para las cajas (BGR)
COLORS = {
    "persona": (255, 0, 0),    # Azul
    "armaP": (0, 0, 255),      # Rojo
    "botas": (0, 255, 0),      # Verde
    "buff": (255, 255, 0),     # Cian
    "casco": (0, 255, 255),    # Amarillo
    "chaleco": (255, 0, 255),  # Magenta
    "gafas": (128, 0, 128),    # Púrpura
    "uniforme": (128, 128, 0)  # Verde oscuro
}

def check_uniform_color(person_crop):
    """
    Verifica si un porcentaje significativo de la persona tiene colores de camuflaje militar
    (verdes oscuros, olivo, marrones).
    """
    if person_crop.size == 0:
        return False
        
    hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
    
    # Rango de colores verde militar y caqui en HSV
    lower_military = np.array([20, 20, 20])
    upper_military = np.array([90, 255, 200])
    
    mask = cv2.inRange(hsv, lower_military, upper_military)
    
    total_pixels = mask.shape[0] * mask.shape[1]
    matching_pixels = cv2.countNonZero(mask)
    
    if total_pixels == 0:
        return False
        
    porcentaje = (matching_pixels / total_pixels) * 100
    
    # Si más del 15% del cuerpo tiene el color, asumimos que tiene uniforme
    return porcentaje > 15.0

def verify_person(image, person_box, custom_model, conf_threshold=0.20):
    """
    Recibe la imagen original y la bounding box de una persona.
    Realiza recortes (zoom) y devuelve los objetos detectados y sus coordenadas absolutas.
    """
    x1, y1, x2, y2 = map(int, person_box)
    
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    
    if x2 <= x1 or y2 <= y1:
        return {"tipo_cuerpo": "Desconocido", "items": set(), "boxes_to_draw": []}
    
    w_box = x2 - x1
    h_box = y2 - y1
    aspect_ratio = h_box / w_box if w_box > 0 else 0
    
    tipo_cuerpo = "Cuerpo Completo" if aspect_ratio > 1.8 else "Medio Cuerpo"
    
    # Recortes
    person_crop = image[y1:y2, x1:x2]
    
    # Ampliamos la mitad superior al 75% de la altura para no cortar armas que lleguen hasta la cadera
    y_mid_top = int(y1 + h_box * 0.75)
    upper_crop = image[y1:y_mid_top, x1:x2]
    
    # Empezamos la mitad inferior más arriba (desde el 30%) para asegurar que la cintura y el arma entren completas
    y_mid_bottom = int(y1 + h_box * 0.30)
    lower_crop = image[y_mid_bottom:y2, x1:x2]
    
    detected_items = set()
    boxes_to_draw = []
    
    def detect_on_crop(crop_img, offset_x, offset_y):
        if crop_img.size == 0:
            return
        results = custom_model.predict(crop_img, conf=conf_threshold, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    
                    if cls_id in CUSTOM_CLASSES:
                        class_name = CUSTOM_CLASSES[cls_id]
                        
                        # REGLA DE CONFIANZA POR CLASE
                        min_conf = {
                            "armaP": 0.20,
                            "botas": 0.40,
                            "buff": 0.40,
                            "casco": 0.20,
                            "chaleco": 0.40,
                            "gafas": 0.65,
                            "uniforme": 0.30
                        }.get(class_name, 0.30)
                        
                        if conf < min_conf:
                            continue
                            
                        detected_items.add(class_name)
                        
                        abs_x1 = bx1 + offset_x
                        abs_y1 = by1 + offset_y
                        abs_x2 = bx2 + offset_x
                        abs_y2 = by2 + offset_y
                        
                        boxes_to_draw.append((class_name, conf, abs_x1, abs_y1, abs_x2, abs_y2))

    # Analizar cuerpo entero
    detect_on_crop(person_crop, x1, y1)
    
    # Analizar parte superior
    detect_on_crop(upper_crop, x1, y1)
    
    # Analizar parte inferior
    detect_on_crop(lower_crop, x1, y_mid_bottom)
    
    # ====== LÓGICA EXTRA HÍBRIDA ======
    
    # 1. Comprobación del Uniforme por Color
    if "uniforme" not in detected_items:
        if check_uniform_color(person_crop):
            detected_items.add("uniforme")
            # Agregamos una caja de dibujo ficticia (la persona misma)
            boxes_to_draw.append(("uniforme", 0.99, x1, y1, x2, y2))

    # ==================================
    
    return {
        "tipo_cuerpo": tipo_cuerpo,
        "items": detected_items,
        "boxes_to_draw": boxes_to_draw
    }

def process_image(image_path, person_model, custom_model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error al leer la imagen: {image_path}")
        return False

    print(f"\n📷 Analizando imagen: {os.path.basename(image_path)}")
    print("-" * 60)

    # 1. Detectar personas
    results_person = person_model.predict(img, classes=[0], conf=0.5, verbose=False)
    
    personas_detectadas = []
    for r in results_person:
        if r.boxes is not None:
            for box in r.boxes:
                personas_detectadas.append(box.xyxy[0].cpu().numpy())

    if not personas_detectadas:
        print("❌ No se detectaron personas en la imagen.")
        return False
        
    print(f"👥 Se detectaron {len(personas_detectadas)} persona(s) en total.")
    
    img_draw = img.copy()

    # 2. Analizar cada persona
    for i, p_box in enumerate(personas_detectadas, 1):
        px1, py1, px2, py2 = map(int, p_box)
        cv2.rectangle(img_draw, (px1, py1), (px2, py2), COLORS["persona"], 2)
        cv2.putText(img_draw, f"Persona {i}", (px1, max(py1 - 10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["persona"], 2)

        info = verify_person(img, p_box, custom_model)
        items = info["items"]
        tipo = info["tipo_cuerpo"]
        
        print(f"\n👤 PERSONA {i} ({tipo}):")
        
        tiene = []
        falta = []
        
        for class_id, class_name in CUSTOM_CLASSES.items():
            if class_name in items:
                tiene.append(class_name)
            else:
                falta.append(class_name)
                
        if tiene:
            for item in tiene:
                print(f"  ✅ {item.capitalize():15}")
        else:
            print("  ❌ No se detectó ningún equipo.")
            
        if falta:
            print("\n  ⚠️ Faltante:")
            for item in falta:
                print(f"  ❌ {item.capitalize():15}")
                
        if not falta:
            print("\n  🌟 APTO - ¡Tiene todo el equipo militar!")

        # Dibujar cajas de los equipos detectados
        for box_info in info["boxes_to_draw"]:
            class_name, conf, bx1, by1, bx2, by2 = box_info
            color = COLORS.get(class_name, (255, 255, 255))
            
            cv2.rectangle(img_draw, (bx1, by1), (bx2, by2), color, 2)
            
            label = f"{class_name} {conf:.2f}" if conf < 1.0 else f"{class_name} (Color)"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_draw, (bx1, by1 - 20), (bx1 + w, by1), color, -1)
            cv2.putText(img_draw, label, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Guardar imagen resultante
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"res_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, img_draw)
    print(f"\n💾 Imagen resultante guardada en: {out_path}")
    print("=" * 60)
    return True

def main(path_input, custom_model_path=DEFAULT_CUSTOM_MODEL_PATH):
    print("🚀 Cargando modelos...")
    person_model = YOLO(PERSON_MODEL_PATH)
    
    try:
        custom_model = YOLO(custom_model_path)
    except Exception as e:
        print(f"❌ Error al cargar tu modelo custom en: {custom_model_path}")
        print(f"Detalles: {e}")
        return
        
    print("✅ Modelos cargados exitosamente.\n")

    if os.path.isfile(path_input):
        process_image(path_input, person_model, custom_model)
    elif os.path.isdir(path_input):
        archivos = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📁 Se encontraron {len(archivos)} imágenes en el directorio '{path_input}'")
        for archivo in archivos:
            ruta_completa = os.path.join(path_input, archivo)
            process_image(ruta_completa, person_model, custom_model)
    else:
        print(f"❌ La ruta proporcionada no es válida: {path_input}")

if __name__ == "__main__":
    target = "Deteccion"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    main(target)
