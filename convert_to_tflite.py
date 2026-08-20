from ultralytics import YOLO
import os

# Ruta del modelo entrenado
model_path = r"C:\Users\ronny\Downloads\Modelo Entrenado\best.pt"

# Verificar que existe
if not os.path.exists(model_path):
    print(f"❌ Error: No se encontró {model_path}")
    exit(1)

print("🚀 Cargando modelo YOLOv8...")
model = YOLO(model_path)

print("🔄 Convirtiendo a TFLite...")
export_path = model.export(format="tflite", imgsz=416, device="cpu")

print(f"✅ Conversión completada!")
print(f"📁 Archivo guardado en: {export_path}")
print(f"\nAhora copia este archivo a tu carpeta Flutter:")
print(f"   -> C:\\Users\\ronny\\OneDrive\\Documents\\Copia de seguridad\\Copia\\Codigo Deteccion\\codigo_deteccion\\assets\\")
