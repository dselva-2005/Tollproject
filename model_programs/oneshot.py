from ultralytics import YOLO
import os
import cv2
from random import randint
import matplotlib.pyplot as plt

# --- Load a random image from the test_samples directory ---
image_dir = "../test_samples"
valid_exts = {"png", "jpg", "jpeg"}
file_list = [f for f in os.listdir(image_dir) if f.split(".")[-1].lower() in valid_exts]

if not file_list:
    raise FileNotFoundError(f"No image files found in {image_dir}")

filename = file_list[randint(0, len(file_list) - 1)]
image_path = os.path.join(image_dir, filename)

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read image at {image_path}")

# Convert BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Load YOLO model and run prediction ---
# model = YOLO("yolo11n.pt")
model = YOLO("../nanomodel/best.pt")
results = model(image)[0]

# --- Add padding ---
pad = 50
padded = cv2.copyMakeBorder(image_rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

# --- Draw predictions ---
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    conf = box.conf.item()
    cls = int(box.cls.item())
    label = f"{model.names[cls]} {conf:.2f}"

    # Draw box
    cv2.rectangle(padded, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Draw label
    cv2.putText(padded, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# --- Plot using matplotlib with image dimensions ---
plt.figure(figsize=(12, 8))
plt.imshow(padded)
plt.title(f"{filename} - {image.shape[1]}x{image.shape[0]} (W x H)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
