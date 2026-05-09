import json
import os
from pathlib import Path

# Paths
annotations_file = 'data/annotations/instances_val2017.json'
images_dir = 'data/images'
labels_dir = 'data/labels'

# Load COCO annotations
print("Loading COCO annotations...")
with open(annotations_file, 'r') as f:
    coco = json.load(f)

# Build category ID to index mapping
categories = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}

# Print all 80 categories so we can see them
print(f"\nTotal categories: {len(categories)}")
for cat in coco['categories']:
    print(f"  ID {cat['id']:3d} -> Index {categories[cat['id']]:2d} : {cat['name']}")

# Build image ID to filename mapping
images = {img['id']: img for img in coco['images']}

# Group annotations by image ID
print("\nGrouping annotations by image...")
annotations_by_image = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

# Convert and save YOLO format labels
print("Converting to YOLO format...")
converted = 0
skipped = 0

for img_id, anns in annotations_by_image.items():
    img_info = images[img_id]
    img_width = img_info['width']
    img_height = img_info['height']
    filename = img_info['file_name']

    # Check image actually exists in our folder
    img_path = os.path.join(images_dir, filename)
    if not os.path.exists(img_path):
        skipped += 1
        continue

    # Build YOLO label lines
    label_lines = []
    for ann in anns:
        # Skip crowd annotations
        if ann.get('iscrowd', 0):
            continue

        cat_id = ann['category_id']
        class_idx = categories[cat_id]

        # COCO bbox: [x_min, y_min, width, height]
        x_min, y_min, w, h = ann['bbox']

        # Convert to YOLO: normalized center x, center y, width, height
        cx = (x_min + w / 2) / img_width
        cy = (y_min + h / 2) / img_height
        nw = w / img_width
        nh = h / img_height

        # Clamp values to [0, 1] to avoid out-of-bound errors
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        nw = max(0, min(1, nw))
        nh = max(0, min(1, nh))

        label_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if label_lines:
        # Save label file with same name as image but .txt extension
        label_filename = Path(filename).stem + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
        converted += 1

print(f"\nDone!")
print(f"  Converted : {converted} images")
print(f"  Skipped   : {skipped} images (not found in data/images)")
print(f"  Labels saved to: {labels_dir}")