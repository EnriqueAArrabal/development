import os
import tempfile
import glob
from ultralytics import YOLO
from transformers import AutoModelForZeroShotObjectDetection

# ===============================
# YOLOv11
# ===============================
print("=== YOLOv11 ===")

# Ruta al archivo del modelo YOLOv11
yolo_model_path = "yolo11n.pt"

# Tamaño en disco
yolo_size_bytes = os.path.getsize(yolo_model_path)
yolo_size_mb = yolo_size_bytes / (1024 * 1024)

# Cargar modelo y contar parámetros
yolo_model = YOLO(yolo_model_path)
yolo_num_params = sum(p.numel() for p in yolo_model.model.parameters())

print(f"Tamaño en disco: {yolo_size_mb:.2f} MB")
print(f"Número de parámetros: {yolo_num_params:,}")

# ===============================
# GroundingDINO-Tiny
# ===============================
print("\n=== GroundingDINO-Tiny ===")

# ID del modelo en Hugging Face
grounding_model_id = "IDEA-Research/grounding-dino-tiny"

# Descargar y cargar modelo
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id)

# Número de parámetros
gd_num_params = sum(p.numel() for p in grounding_model.parameters())
print(f"Número de parámetros: {gd_num_params:,}")

# Guardar el modelo en una carpeta temporal para calcular el tamaño total
with tempfile.TemporaryDirectory() as tmpdirname:
    grounding_model.save_pretrained(tmpdirname)

    total_size_bytes = 0
    for root, dirs, files in os.walk(tmpdirname):
        for file in files:
            filepath = os.path.join(root, file)
            total_size_bytes += os.path.getsize(filepath)

    gd_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Tamaño en disco: {gd_size_mb:.2f} MB")
