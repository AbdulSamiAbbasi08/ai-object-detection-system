from ultralytics import YOLO
import os

# Load pretrained YOLOv8 small model
model = YOLO('yolov8s.pt')

# Train the model
results = model.train(
    data='data/dataset.yaml',   # our dataset config
    epochs=50,                   # number of training rounds
    imgsz=640,                   # image size
    batch=8,                     # batch size (reduce to 4 if you get memory errors)
    name='coco_detection',       # run name
    project='model/runs',        # save location
    patience=10,                 # stop early if no improvement for 10 epochs
    save=True,                   # save best weights
    plots=True,                  # save training plots
    verbose=True                 # print progress
)

print("Training complete!")
print(f"Best weights saved to: model/runs/coco_detection/weights/best.pt")