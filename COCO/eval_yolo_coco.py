from ultralytics import YOLO
from pycocotools.coco import COCO
import json, os
from PIL import Image
from tqdm import tqdm

# Rutas
IMG_DIR = "C:/Users/Tito/Desktop/Master/TFM/development/COCO/images/val2017"
ANN_FILE = "C:/Users/Tito/Desktop/Master/TFM/development/COCO/annotations/instances_val2017.json"
MODEL_PATH = "yolo11n.pt"  # Cambia según el modelo que uses

# Cargar modelo
model = YOLO(MODEL_PATH)

# Cargar COCO
coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

# Diccionario de clases COCO
categories = coco.loadCats(coco.getCatIds())
class_name_to_id = {cat["name"]: cat["id"] for cat in categories}

results = []

for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMG_DIR, img_info["file_name"])
    image = Image.open(img_path).convert("RGB")

    # Inferencia
    pred = model.predict(image, imgsz=640, conf=0.25)[0]

    for box in pred.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        score = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        category_id = class_name_to_id.get(class_name, None)
        if category_id is None:
            continue

        results.append({
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [x1, y1, w, h],
            "score": score
        })

# Guardar predicciones
with open("yolo11_predictions.json", "w") as f:
    json.dump(results, f)
