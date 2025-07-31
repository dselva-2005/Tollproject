import cv2
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

# Load YOLO model
model = YOLO("/home/dselva/MINIPROJECTTE/weights/best.pt")

# Load TrOCR (small, printed version)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed").to("cuda" if torch.cuda.is_available() else "cpu")

# Extract valid plate number
def extract_plate(text):
    text = text.upper().replace(" ", "")
    text = re.sub(r"[^A-Z0-9]", "", text)
    pattern = r"\b([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4})\b"
    matches = re.findall(pattern, text)
    return max(matches, key=len) if matches else text

# OCR using TrOCR
def recognize_plate(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img_pil, return_tensors="pt").pixel_values.to(trocr.device)
    ids = trocr.generate(inputs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return extract_plate(text)

# Stream from IP cam
cap = cv2.VideoCapture("http://192.168.1.100:8080/video")
if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for r in model(frame, stream=True):
        for b in r.boxes:
            if b.conf[0] < 0.5: continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            try:
                plate = recognize_plate(crop)
                if plate:
                    print(plate)
            except:
                pass

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
