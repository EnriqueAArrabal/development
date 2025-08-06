import time
import csv
from dataset_loader import load_images_from_folders
from models import load_vit_model, load_resnet50_model, infer_vit, infer_resnet
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    folders = [
        #"./datasets/Vehicle_Detection_Image_Dataset/train/images",
        # "./datasets/trafic_data/train/images",
        # "./dataset3/validation"
    ]
    images, names = load_images_from_folders(folders, max_images_per_folder=1, recursive=True)
    print(f"Cargadas {len(images)} imágenes.")

    vit_processor, vit_model = load_vit_model()
    resnet_transform, resnet_model = load_resnet50_model()

    results = []

    for i, (img, name) in enumerate(zip(images, names)):
        start = time.time()
        vit_pred = infer_vit(vit_processor, vit_model, img)
        vit_time = (time.time() - start) * 1000

        start = time.time()
        resnet_pred = infer_resnet(resnet_transform, resnet_model, img)
        resnet_time = (time.time() - start) * 1000

        print(f"Imagen {i+1}: ViT: {vit_pred} ({vit_time:.1f} ms), ResNet: {resnet_pred} ({resnet_time:.1f} ms)")

        results.append({
            "imagen": name,
            "vit_pred": vit_pred,
            "vit_time_ms": round(vit_time, 2),
            "resnet_pred": resnet_pred,
            "resnet_time_ms": round(resnet_time, 2)
        })

    # Guardar en CSV
    # Guardar en CSV
    output_file = "resultados.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "Nombre de imagen",
            "Predicción ViT",
            "Tiempo ViT (ms)",
            "Predicción ResNet50",
            "Tiempo ResNet50 (ms)"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Nombre de imagen": r["imagen"],
                "Predicción ViT": r["vit_pred"],
                "Tiempo ViT (ms)": r["vit_time_ms"],
                "Predicción ResNet50": r["resnet_pred"],
                "Tiempo ResNet50 (ms)": r["resnet_time_ms"]
            })


        print(f"\nResultados guardados en {output_file}")

if __name__ == "__main__":
    main()
