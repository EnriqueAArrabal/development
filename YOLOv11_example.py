from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n.pt")

    train_results = model.train(
        data="C:/Users/Tito/Desktop/Master/TFM/development/datasets/trafic_data/data_1.yaml",  # YAML del dataset
        epochs=100,                
        batch=16,
        device="0",                
        project="yolo_training",
        name="vehicle_model",
        exist_ok=True              
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("./datasets/trafic_data/train/images/41_jpg.rf.cccfea1ad9ef5ff8ad3b5468c36b2709.jpg")
    results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model
