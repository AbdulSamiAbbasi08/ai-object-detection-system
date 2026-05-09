from ultralytics import YOLO
import os

# Load pretrained YOLOv8 small model
model = YOLO('yolov8s.pt')

# Train the model
results = model.train(
    data='data/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,        # increase to 16 since we have GPU
    device=0,        # 0 = use first GPU
    name='coco_detection',
    project='model/runs',
    patience=10,
    save=True,
    plots=True,
    verbose=True
)

print("Training complete!")
print(f"Best weights saved to: model/runs/coco_detection/weights/best.pt")