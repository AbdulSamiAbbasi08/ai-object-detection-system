from ultralytics import YOLO
import os

try:
    # Load model
    print("Loading model...")
    model = YOLO('model/best.pt')
    print("Model loaded!")

    # Check test images exist
    test_dir = 'data/test/images'
    images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    print(f"Found {len(images)} test images")

    # Run predictions
    print("Running predictions...")
    results = model.predict(
        source=test_dir,
        conf=0.25,
        save=True,
        save_txt=True,
        project='results/predictions',
        name='test_run',
        exist_ok=True,
        verbose=True
    )

    print(f"Done! Processed {len(results)} images")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()