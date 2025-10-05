from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo y procesador
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to(device)

# Cargar la imagen
image = Image.open("./datasets/trafic_data/train/images/41_jpg.rf.cccfea1ad9ef5ff8ad3b5468c36b2709.jpg")

# Construir conversación en formato correcto
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is shown in this image?"},
            {"type": "image"}
        ]
    }
]

# Crear prompt con plantilla
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Preparar inputs para el modelo
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

# Generar salida
output = model.generate(**inputs, max_new_tokens=100)

# Mostrar respuesta
print(processor.decode(output[0], skip_special_tokens=True))
