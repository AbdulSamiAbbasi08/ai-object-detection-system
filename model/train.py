from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')
    
    results = model.train(
        data='data/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        name='coco_detection',
        project='model/runs',
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        workers=0        # fix for Windows multiprocessing issue
    )

    print("Training complete!")
    print(f"Best weights saved to: model/runs/coco_detection/weights/best.pt")

if __name__ == '__main__':
    main()