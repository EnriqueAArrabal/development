from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Cargar anotaciones reales
coco_gt = COCO("./annotations/instances_val2017.json")

# Cargar predicciones del modelo
coco_dt = coco_gt.loadRes("./yolo11_predictions.json")

# Evaluar
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


# Cargar anotaciones reales
coco_gt_dino = COCO("./annotations/instances_val2017.json")

# Cargar predicciones del modelo
coco_dt_dino = coco_gt_dino.loadRes("./dino_tiny_predictions.json")

# Evaluar
coco_eval_dino = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval_dino.evaluate()
coco_eval_dino.accumulate()
coco_eval_dino.summarize()
