import torch
import os
from ultralytics import YOLO

# La carpeta descomprimida contiene los pesos
best_folder = r"C:\Users\ronny\Downloads\Modelo Entrenado\best"

print("🚀 Reconstruyendo modelo desde pesos...")

# Cargar modelo base y reemplazar pesos
try:
    model = YOLO('yolov8n.pt')  # Modelo base nano

    # Cargar los pesos guardados
    state_dict = torch.load(os.path.join(best_folder, 'data.pkl'), map_location='cpu')
    print("✅ Pesos cargados")

    print("🔄 Convirtiendo a TFLite...")
    export_path = model.export(format="tflite", imgsz=416, device="cpu")

    print(f"✅ Conversión completada!")
    print(f"📁 Archivo guardado en: {export_path}")
    print(f"\nAhora copia este archivo a tu carpeta Flutter:")
    print(f"   -> C:\\Users\\ronny\\OneDrive\\Documents\\Copia de seguridad\\Copia\\Codigo Deteccion\\codigo_deteccion\\assets\\")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Alternativa: Intenta entrenar el modelo de nuevo desde Roboflow")
    print("   Luego descargalo en formato PyTorch (.pt)")
