import time
from ultralytics import YOLO

if __name__ == "__main__":
    # Cargar el modelo
    model = YOLO("yolo11n.pt")

    # Imagen de prueba
    img = "./datasets/trafic_data/train/images/41_jpg.rf.cccfea1ad9ef5ff8ad3b5468c36b2709.jpg"

    # Calentar la GPU (primera pasada no cuenta)
    model(img)

    # Medir tiempo de inferencia
    N = 100  # número de repeticiones
    start = time.time()
    for _ in range(N):
        results = model(img)
    end = time.time()

    avg_time = (end - start) / N
    fps = 1 / avg_time

    print(f"Tiempo medio por imagen: {avg_time:.4f} s ({avg_time*1000:.2f} ms)")
    print(f"FPS aproximados: {fps:.2f}")
