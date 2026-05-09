from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')
    
    results = model.train(
    data='data/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    name='coco_detection_v2',
    project='model/runs',
    patience=20,         # give it more time
    lr0=0.001,           # lower learning rate
    lrf=0.01,
    optimizer='SGD',     # switch to SGD
    save=True,
    plots=True,
    verbose=True,
    workers=0
)

    print("Training complete!")
    print(f"Best weights saved to: model/runs/coco_detection/weights/best.pt")

if __name__ == '__main__':
    main()