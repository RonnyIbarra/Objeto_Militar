from ultralytics import YOLO
import cv2
import os

# Cargar el modelo entrenado (v2 - última versión)
model_path = r"C:\Users\ronny\PycharmProjects\ProyectoDeteccionObjetos\runs\detect\train_roboflow-3\weights\best.pt"

print("🚀 Cargando modelo entrenado...")
model = YOLO(model_path)
print("✅ Modelo cargado\n")

# Clases requeridas
clases_requeridas = {0: "armaP", 1: "botas", 2: "buff", 3: "casco", 4: "chaleco", 5: "gafas", 6: "uniforme"}
confianza_minima = 0.5  # 50% de confianza mínima

print("=" * 60)
print("PRUEBA DE DETECCIÓN DE EQUIPO DE SEGURIDAD")
print("=" * 60)
print(f"Clases requeridas: {list(clases_requeridas.values())}")
print(f"Confianza mínima: {confianza_minima * 100}%\n")

def verificar_equipo(image_path):
    """Analiza una imagen y verifica si tiene todo el equipo"""

    if not os.path.exists(image_path):
        print(f"❌ Imagen no encontrada: {image_path}\n")
        return

    print(f"📷 Analizando: {os.path.basename(image_path)}")
    print("-" * 60)

    # Hacer predicción
    results = model.predict(image_path, conf=confianza_minima)

    # Extraer detecciones
    clases_detectadas = {}

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]

                if class_id not in clases_detectadas:
                    clases_detectadas[class_id] = {
                        "nombre": class_name,
                        "confianza": confidence,
                        "cantidad": 1
                    }
                else:
                    clases_detectadas[class_id]["cantidad"] += 1
                    # Mantener la confianza más alta
                    if confidence > clases_detectadas[class_id]["confianza"]:
                        clases_detectadas[class_id]["confianza"] = confidence

    # Mostrar detecciones
    print("\n🔍 DETECCIONES:")
    if clases_detectadas:
        for class_id in sorted(clases_detectadas.keys()):
            info = clases_detectadas[class_id]
            confianza_pct = info["confianza"] * 100
            print(f"  ✅ {info['nombre']:15} - Confianza: {confianza_pct:5.1f}% (encontrado {info['cantidad']}x)")
    else:
        print("  ❌ No se detectaron objetos")

    # Verificar si tiene todo el equipo
    print("\n📋 VERIFICACIÓN:")
    clases_detectadas_ids = set(clases_detectadas.keys())
    clases_requeridas_ids = set(clases_requeridas.keys())

    faltantes = clases_requeridas_ids - clases_detectadas_ids

    if not faltantes:
        print("  ✅ APTO - ¡¡¡Tiene TODO el equipo de seguridad!!!")
        print(f"  Detectadas {len(clases_detectadas)}/7 clases")
    else:
        print("  ❌ NO APTO - Falta equipo de seguridad")
        print(f"  Falta: {', '.join([clases_requeridas[cid] for cid in sorted(faltantes)])}")
        print(f"  Detectadas {len(clases_detectadas)}/7 clases")

    print("-" * 60 + "\n")

# MODO INTERACTIVO
print("\n" + "=" * 60)
print("MODO INTERACTIVO - INGRESA LA RUTA DE TU IMAGEN")
print("=" * 60)

while True:
    print("\n📷 Ingresa la ruta de una imagen (o 'salir' para terminar):")
    print("   Ejemplo: C:\\Users\\ronny\\Downloads\\Deteccion\\prueba.jpg")

    ruta = input("\n➜ Ruta: ").strip()

    if ruta.lower() == "salir":
        print("\n✅ ¡Hasta luego!")
        break

    # Limpiar comillas si las tiene
    ruta = ruta.strip('"\'')

    # Verificar la imagen
    verificar_equipo(ruta)

print("\n" + "=" * 60)
print("PROGRAMA FINALIZADO")
print("=" * 60)
