import os
import shutil
import random
from pathlib import Path

# Source
images_dir = Path("archive/dataset/input_images")
labels_dir = Path("archive/dataset/annotations")

# Destination
target_dir = Path("yolo_dataset")
for split in ["train", "val"]:
    (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# List image-label pairs
image_files = list(images_dir.glob("*.jpg"))
random.shuffle(image_files)

# 80/20 split
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_pairs(files, split):
    for img_path in files:
        label_path = labels_dir / (img_path.name + ".txt") if (labels_dir / (img_path.name + ".txt")).exists() else labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(img_path, target_dir / "images" / split / img_path.name)
            shutil.copy(label_path, target_dir / "labels" / split / (img_path.stem + ".txt"))

move_pairs(train_files, "train")
move_pairs(val_files, "val")

print("✅ Dataset prepared for YOLOv8!")
