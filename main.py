import cv2
from ultralytics import YOLO
import easyocr
import os
from datetime import datetime
import re

# Load your fine-tuned YOLOv8 model
model = YOLO("/home/dselva/MINIPROJECTTE/runs/detect/train6/weights/best.pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Create folder to save plates
os.makedirs("detected_plates", exist_ok=True)

# Camera stream
stream_url = "http://192.168.1.100:8080/video"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print(f"❌ Failed to open video stream at {stream_url}")
    exit()

print("✅ Stream opened successfully. Press 'q' to quit.\n")

def clean_ocr_text(text):
    # Fix common OCR misreads
    text = text.upper().replace(" ", "")
    text = text.replace("O", "0")
    text = text.replace("I", "1")
    text = text.replace("Z", "2")
    return text

def extract_plate_number(raw_text):
    # Try to match an Indian plate-like pattern
    text = re.sub(r"[^A-Z0-9\s-]", "", raw_text.upper())
    pattern = r"\b([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4})\b"
    matches = re.findall(pattern, text)
    if matches:
        return max(matches, key=len)
    return clean_ocr_text(raw_text)

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

            if conf < 0.5:
                continue

            plate_img = frame[y1:y2, x1:x2]

            ocr_result = reader.readtext(plate_img, detail=0)
            plate_text_raw = " ".join(ocr_result)
            final_plate = extract_plate_number(plate_text_raw)

            # Save the cropped plate
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detected_plates/{final_plate}_{timestamp}.jpg"
            cv2.imwrite(filename, plate_img)

            print(f"[{timestamp}] OCR Raw: '{plate_text_raw}' → Final Plate: '{final_plate}'")

            # Draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, final_plate, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Real-Time ANPR - India", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
