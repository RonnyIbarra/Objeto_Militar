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
data_yaml = r"C:\Users\ronny\PycharmProjects\ProyectoDeteccionObjetos\DeteccionEquipoSeguridad.v2i.yolov8\data.yaml"

# Verificar que el archivo existe
if not os.path.exists(data_yaml):
    print(f"❌ Error: No se encontró {data_yaml}")
    print("Asegúrate de que exista la carpeta 'DeteccionEquipoSeguridad.v1.yolov8'")
    exit(1)

print(f"✅ Encontrado: {data_yaml}\n")

# Cargar modelo YOLO pequeño (yolov8n = nano)
model = YOLO('yolov8n.pt')

# Entrenar el modelo
results = model.train(
    data=data_yaml,
    epochs=50,                    # Número de épocas
    imgsz=416,                    # Tamaño de imagen
    batch=16,                     # Tamaño de lote (ajusta según tu RAM)
    patience=10,                  # Early stopping
    device=0 if torch.cuda.is_available() else 'cpu',
    save=True,
    workers=4,
    verbose=True,
    name='train_roboflow'
)

print("\n✅ Entrenamiento completado")
print(f"Modelo guardado en: runs/detect/train_roboflow/weights/best.pt")
print("\nAhora puedes usar el modelo para hacer predicciones:")
print("  python predict.py")
