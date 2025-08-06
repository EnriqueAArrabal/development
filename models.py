from transformers import AutoImageProcessor, AutoModelForImageClassification
import torchvision.models as models
import torchvision.transforms as transforms
import torch

def load_vit_model():
    model_name = "google/vit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

def load_resnet50_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return transform, model

def infer_vit(processor, model, image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    return model.config.id2label[predicted_class]

def infer_resnet(transform, model, image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    predicted_class = outputs.argmax(-1).item()
    return predicted_class
