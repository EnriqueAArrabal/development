import os
from PIL import Image

def load_images_from_folders(folder_paths, max_images_per_folder=None, recursive=False):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = []
    image_names = []

    for folder_path in folder_paths:
        count = 0
        if recursive:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        img_path = os.path.join(root, file)
                        try:
                            img = Image.open(img_path).convert('RGB')
                            images.append(img)
                            image_names.append(file)
                            count += 1
                            if max_images_per_folder and count >= max_images_per_folder:
                                break
                        except Exception as e:
                            print(f"Error cargando {img_path}: {e}")
                if max_images_per_folder and count >= max_images_per_folder:
                    break
        else:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
            for file in files:
                img_path = os.path.join(folder_path, file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    image_names.append(file)
                    count += 1
                    if max_images_per_folder and count >= max_images_per_folder:
                        break
                except Exception as e:
                    print(f"Error cargando {img_path}: {e}")

    return images, image_names
