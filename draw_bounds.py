import cv2
import matplotlib.pyplot as plt

def draw_yolo_boxes(image_path, label_path):
    # Cargar imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Leer cajas desde el txt
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height

            # Convertir a formato (x1, y1, x2, y2)
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Dibujar rectángulo
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(image, f'Class {class_id}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Mostrar la imagen con matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Bounding Boxes")
    plt.show()

# Ejemplo de uso
draw_yolo_boxes(
    "./datasets/trafic_data/train/images/41_jpg.rf.cccfea1ad9ef5ff8ad3b5468c36b2709.jpg", 
    "./datasets/trafic_data/train/labels/41_jpg.rf.cccfea1ad9ef5ff8ad3b5468c36b2709.txt"
)
