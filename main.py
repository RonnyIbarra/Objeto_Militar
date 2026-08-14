from ultralytics import YOLO
import cv2

print("✓ YOLO importado correctamente")
print("✓ OpenCV importado correctamente")

# Intenta cargar un modelo YOLO preentrenado (pequeño)
model = YOLO('yolov8n.pt')
print("✓ Modelo YOLO cargado correctamente")