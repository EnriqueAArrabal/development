import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pycocotools.coco import COCO
import json, os
from tqdm import tqdm

# -----------------------------
# Rutas y configuración
# -----------------------------
IMG_DIR = "/mnt/c/Users/Tito/Desktop/Master/TFM/development/COCO/images/val2017"
ANN_FILE = "/mnt/c/Users/Tito/Desktop/Master/TFM/development/COCO/annotations/instances_val2017.json"
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
CONF_THRESHOLD = 0.25
OUTPUT_JSON = "dino_tiny_predictions.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Cargar modelo
# -----------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# -----------------------------
# Cargar COCO
# -----------------------------
coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

categories = coco.loadCats(coco.getCatIds())
class_name_to_id = {cat["name"].lower(): cat["id"] for cat in categories}
text_labels = [cat["name"] for cat in categories]  # lista plana

# -----------------------------
# Generar predicciones
# -----------------------------
results = []

for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMG_DIR, img_info["file_name"])
    image = Image.open(img_path).convert("RGB")

    # Preprocesar entrada
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(
            pixel_values=inputs.pixel_values,
            input_ids=inputs.input_ids
        )

    # Post-procesar resultados
    postprocessed = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=CONF_THRESHOLD,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]

    for box, score, label_text in zip(postprocessed["boxes"], postprocessed["scores"], postprocessed["labels"]):
        class_name = label_text.lower()
        category_id = class_name_to_id.get(class_name, None)
        if category_id is None:
            continue

        # Convertir formato [x1, y1, x2, y2] → [x, y, w, h]
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1

        results.append({
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "score": round(score.item(), 4)
        })

# -----------------------------
# Guardar predicciones
# -----------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f)

print(f"✅ Predicciones guardadas en {OUTPUT_JSON}")
