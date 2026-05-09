import os
import shutil
import random
from pathlib import Path

# Paths
images_dir = 'data/images'
labels_dir = 'data/labels'

train_img = 'data/train/images'
train_lbl = 'data/train/labels'
val_img   = 'data/val/images'
val_lbl   = 'data/val/labels'
test_img  = 'data/test/images'
test_lbl  = 'data/test/labels'

# Split ratios
TRAIN = 0.70
VAL   = 0.20
TEST  = 0.10

# Get all images that have a corresponding label file
print("Finding matched image-label pairs...")
all_images = []
for img_file in Path(images_dir).glob('*.jpg'):
    label_file = Path(labels_dir) / (img_file.stem + '.txt')
    if label_file.exists():
        all_images.append(img_file.stem)

print(f"Total matched pairs: {len(all_images)}")

# Shuffle randomly
random.seed(42)
random.shuffle(all_images)

# Calculate split sizes
n = len(all_images)
n_train = int(n * TRAIN)
n_val   = int(n * VAL)
n_test  = n - n_train - n_val

train_set = all_images[:n_train]
val_set   = all_images[n_train:n_train + n_val]
test_set  = all_images[n_train + n_val:]

print(f"  Train : {len(train_set)} images")
print(f"  Val   : {len(val_set)} images")
print(f"  Test  : {len(test_set)} images")

# Copy files to split folders
def copy_split(file_stems, src_img, src_lbl, dst_img, dst_lbl, split_name):
    print(f"\nCopying {split_name} set...")
    for stem in file_stems:
        shutil.copy(f"{src_img}/{stem}.jpg", f"{dst_img}/{stem}.jpg")
        shutil.copy(f"{src_lbl}/{stem}.txt", f"{dst_lbl}/{stem}.txt")
    print(f"  Done — {len(file_stems)} files copied")

copy_split(train_set, images_dir, labels_dir, train_img, train_lbl, 'train')
copy_split(val_set,   images_dir, labels_dir, val_img,   val_lbl,   'val')
copy_split(test_set,  images_dir, labels_dir, test_img,  test_lbl,  'test')

print("\nDataset split complete!")
print(f"  data/train/ -> {len(train_set)} images")
print(f"  data/val/   -> {len(val_set)} images")
print(f"  data/test/  -> {len(test_set)} images")