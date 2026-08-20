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

print("🔄 Convirtiendo a ONNX...")
export_path = model.export(format="onnx", imgsz=416, device="cpu")

print(f"✅ Conversión completada!")
print(f"📁 Archivo guardado en: {export_path}")
print(f"\n💡 Ahora copia este archivo .onnx a tu carpeta Flutter:")
print(f"   -> C:\\Users\\ronny\\OneDrive\\Documents\\Copia de seguridad\\Copia\\Codigo Deteccion\\codigo_deteccion\\assets\\")
print(f"\nLuego en Flutter usarás tflite_flutter para cargar modelos ONNX.")
