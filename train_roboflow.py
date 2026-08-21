from ultralytics import YOLO
import torch
import os

# Verificar si GPU está disponible
print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Se usará CPU (más lento)")

print("\n🚀 Iniciando entrenamiento de YOLO con datos de Roboflow...\n")

# Ruta del archivo data.yaml de Roboflow v2 (editado)
data_yaml = r"C:\Users\USER\Documents\Objeto_Militar\DeteccionEquipoSeguridad.v3i.yolov8\data.yaml"

# Verificar que el archivo existe
if not os.path.exists(data_yaml):
    print(f"❌ Error: No se encontró {data_yaml}")
    print("Asegúrate de que exista la carpeta correcta de tu dataset (v3i)")
    exit(1)

print(f"✅ Encontrado: {data_yaml}\n")

# Cargar tu modelo PREVIAMENTE ENTRENADO (best.pt) para no perder lo que ya sabía
# Si no existe, puedes cambiar a 'yolov8n.pt'
if os.path.exists('best.pt'):
    print("🔄 Re-entrenando a partir de tu modelo anterior (best.pt)...")
    model = YOLO('best.pt')
else:
    print("⚠️ No se encontró 'best.pt', empezando desde cero con yolov8n.pt...")
    model = YOLO('yolov8n.pt')

# Entrenar el modelo
results = model.train(
    data=data_yaml,
    epochs=100,                   # Aumentamos a 100 para que aprenda bien los casos difíciles
    imgsz=640,                    # 640 es mucho mejor que 416 para detectar detalles pequeños (armas, gafas, buff)
    batch=16,                     # Tamaño de lote (si te da error de memoria de GPU, bájalo a 8)
    patience=15,                  # Early stopping: si en 15 épocas no mejora, se detiene
    device=0 if torch.cuda.is_available() else 'cpu',
    save=True,
    workers=4,
    verbose=True,
    name='train_v3_roboflow',     # Nombre diferente para no sobreescribir tus entrenamientos anteriores
    
    # === TÉCNICAS PARA EVITAR EL SOBREENTRENAMIENTO (OVERFITTING) ===
    optimizer='AdamW',            # AdamW maneja mejor la regularización que el optimizador por defecto
    weight_decay=0.0005,          # Penaliza pesos muy grandes (Regularización L2)
    dropout=0.15,                 # Apaga aleatoriamente el 15% de las neuronas para forzar a la red a no memorizar
    
    # --- Aumento de Datos (Data Augmentation) Fuerte ---
    # Esto distorsiona las imágenes en cada época para que el modelo nunca vea la "misma" foto dos veces
    mosaic=1.0,                   # Pega 4 imágenes juntas (excelente para el contexto)
    mixup=0.1,                    # Mezcla dos imágenes semitransparentes (previene memorización)
    degrees=10.0,                 # Rota las imágenes hasta 10 grados
    translate=0.1,                # Mueve la imagen 10%
    scale=0.5,                    # Acera/Aleja la imagen (zoom out/in)
    fliplr=0.5,                   # Voltea la imagen horizontalmente (espejo) el 50% de las veces
    hsv_h=0.015,                  # Cambia ligeramente los colores (ayuda con diferentes luces)
    hsv_s=0.7,
    hsv_v=0.4
)

print("\n✅ Entrenamiento completado")
print(f"Modelo guardado en: runs/detect/train_roboflow/weights/best.pt")
print("\nAhora puedes usar el modelo para hacer predicciones:")
print("  python predict.py")
