import cv2
import torch
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
from datetime import datetime
import re

# Load YOLOv8 model (number plate detector)
model = YOLO("/home/dselva/MINIPROJECTTE/weights/best.pt")

# Load TrOCR
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed").to(device)

# Create folder and text file
os.makedirs("detected_plates", exist_ok=True)
plate_log_file = open("detected_plates/plates.txt", "a")

stream_url = "http://10.144.70.59:8080/video"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print(f"❌ Failed to open video stream at {stream_url}")
    exit()

print("✅ Stream opened. Press 'q' to quit.")

def extract_plate_number(text):
    text = text.upper().replace(" ", "")
    text = text.replace("O", "0").replace("I", "1").replace("Z", "2")
    text = re.sub(r"[^A-Z0-9]", "", text)
    pattern = r"\b([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4})\b"
    matches = re.findall(pattern, text)
    return max(matches, key=len) if matches else text

def run_trocr(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    generated_ids = trocr.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < 0.5: continue

            plate_img = frame[y1:y2, x1:x2]
            try:
                raw_text = run_trocr(plate_img)
                final_plate = extract_plate_number(raw_text)

                if final_plate:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"detected_plates/{final_plate}_{timestamp}.jpg", plate_img)
                    plate_log_file.write(f"{timestamp}: {final_plate}\n")
                    plate_log_file.flush()

                    print(f"[{timestamp}] '{raw_text}' → '{final_plate}'")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, final_plate, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"⚠️ OCR failed: {e}")

    cv2.imshow("ANPR - TrOCR + YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plate_log_file.close()
