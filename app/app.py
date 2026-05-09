import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Our 4 environments and their classes
ENVIRONMENTS = {
    'Traffic':   ['car', 'motorcycle', 'bus', 'truck', 'person', 'traffic light', 'bicycle'],
    'Home':      ['tv', 'laptop', 'cell phone', 'chair', 'bed', 'bottle', 'cup'],
    'Wildlife':  ['bird', 'cat', 'dog', 'horse', 'elephant', 'bear', 'zebra', 'giraffe'],
    'Classroom': ['backpack', 'book', 'clock', 'scissors', 'keyboard', 'mouse', 'handbag', 'potted plant'],
}

ENVIRONMENT_COLORS = {
    'Traffic':   '🚦',
    'Home':      '🏠',
    'Wildlife':  '🦒',
    'Classroom': '🏫',
    'General':   '📦',
}

# Load model once at startup
print("Loading model...")
model = YOLO('model/best.pt')
print("Model ready!")


def detect_environment(detected_classes):
    """Determine which environment the detected objects belong to."""
    env_counts = {env: 0 for env in ENVIRONMENTS}
    for cls in detected_classes:
        for env, classes in ENVIRONMENTS.items():
            if cls in classes:
                env_counts[env] += 1
    max_env = max(env_counts, key=env_counts.get)
    if env_counts[max_env] == 0:
        return 'General'
    return max_env


def predict(image, confidence):
    """Run detection on uploaded image."""
    if image is None:
        return None, "Please upload an image.", ""

    # Run YOLO prediction
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Get annotated image
    annotated = result.plot()
    annotated_rgb = Image.fromarray(annotated[..., ::-1])

    # Build detection summary
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated_rgb, "No objects detected.", "Unknown"

    # Count detections per class
    class_counts = {}
    detected_classes = []
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf_score = float(box.conf[0])
        detected_classes.append(cls_name)
        if cls_name not in class_counts:
            class_counts[cls_name] = {'count': 0, 'max_conf': 0}
        class_counts[cls_name]['count'] += 1
        class_counts[cls_name]['max_conf'] = max(
            class_counts[cls_name]['max_conf'], conf_score
        )

    # Determine environment
    environment = detect_environment(detected_classes)
    env_icon = ENVIRONMENT_COLORS.get(environment, '📦')

    # Build results text
    lines = [f"Detected {len(boxes)} object(s):\n"]
    for cls_name, info in sorted(class_counts.items()):
        count = info['count']
        conf  = info['max_conf']
        lines.append(f"  • {cls_name}: {count}x  (confidence: {conf:.1%})")

    summary = "\n".join(lines)
    env_text = f"{env_icon} Environment: {environment}"

    return annotated_rgb, summary, env_text


# Build Gradio UI
with gr.Blocks(title="AI Object Detection System") as app:

    gr.Markdown("# 🔍 AI-Based Multi-Environment Object Detection")
    gr.Markdown("Upload an image to detect objects across Traffic, Home, Wildlife, and Classroom environments.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Image",
                type="pil"
            )
            confidence = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )
            detect_btn = gr.Button("🔍 Detect Objects", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Detection Result")
            environment_label = gr.Textbox(label="Detected Environment", interactive=False)
            detection_summary = gr.Textbox(
                label="Detected Objects",
                lines=10,
                interactive=False
            )

    detect_btn.click(
        fn=predict,
        inputs=[input_image, confidence],
        outputs=[output_image, detection_summary, environment_label]
    )

    gr.Markdown("### Sample test images are in `data/test/images/`")

if __name__ == '__main__':
    app.launch(share=False)