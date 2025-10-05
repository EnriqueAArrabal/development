import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import time

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Imagen local
image_path = "./datasets/trafic_data/train/images/41_jpg.rf.cccfea1ad9ef5ff8ad3b5468c36b2709.jpg"
image = Image.open(image_path)

# Etiquetas (puedes usar tus clases de tráfico)
text_labels = [["ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus",
                "car", "garbage van", "human hauler", "minibus", "minivan",
                "motorbike", "pickup truck", "police car", "rickshaw", "scooter",
                "SUV", "taxi", "three wheeler CNG", "truck", "van", "wheelbarrow"]]

num_runs = 100
total_time = 0.0

for _ in range(num_runs):
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    total_time += time.time() - start_time

avg_time = total_time / num_runs
fps = 1.0 / avg_time

print(f"Tiempo medio por imagen: {avg_time:.3f} s")
print(f"FPS aproximados: {fps:.2f}")
