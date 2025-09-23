import torch
import os
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import requests

# ===============================
# Función para medir memoria en GPU
# ===============================
def medir_memoria_inferencia(modelo, inputs, dispositivo):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dispositivo)

    modelo.to(dispositivo)
    inputs = {k: v.to(dispositivo) for k, v in inputs.items()}

    with torch.no_grad():
        _ = modelo(**inputs)

    mem_pico = torch.cuda.max_memory_allocated(dispositivo) / (1024 ** 2)
    return mem_pico

# ===============================
# YOLOv11
# ===============================
print("=== YOLOv11 ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo YOLOv11
yolo_model_path = "yolo11n.pt"
yolo_model = YOLO(yolo_model_path)

# Crear imagen de prueba (tensor aleatorio)
yolo_input = torch.randn(1, 3, 640, 448)

# Medir memoria
yolo_memoria = medir_memoria_inferencia(yolo_model.model, {"x": yolo_input}, device)
print(f"Memoria pico usada en GPU: {yolo_memoria:.2f} MB")

# ===============================
# GroundingDINO-Tiny
# ===============================
print("\n=== GroundingDINO-Tiny ===")

# Cargar modelo y procesador
modelo_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(modelo_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(modelo_id)

# Usar imagen real para inferencia
url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
text_prompt = ["a person", "a face"]

# Preparar inputs correctamente
inputs = processor(images=image, text=text_prompt, return_tensors="pt")

# Medir memoria
dino_memoria = medir_memoria_inferencia(dino_model, inputs, device)
print(f"Memoria pico usada en GPU: {dino_memoria:.2f} MB")
