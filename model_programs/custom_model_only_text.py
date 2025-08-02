import os
import re
import cv2
import time
import torch
import logging
import warnings
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import logging as hf_logging
from ultralytics.utils import LOGGER

# ========== SUPPRESS LOGS ========== #
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LOGGER.setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# ========== CONFIGURATION ========== #
YOLO_MODEL_PATH = "/home/dselva/MINIPROJECTTE/weights/best.pt"
TROCR_MODEL_NAME = "microsoft/trocr-small-printed"
VIDEO_SOURCE = "../test_samples/video1.mp4"
CONFIDENCE_THRESHOLD = 0.75

VALID_STATES = {
    "AP", "AR", "AS", "BR", "CH", "CT", "DN", "DD", "DL", "GA", "GJ", "HR", "HP",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

# ========== INITIALIZE MODELS ========== #
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading YOLO model from: {YOLO_MODEL_PATH}")
detector = YOLO(YOLO_MODEL_PATH)

print(f"[INFO] Loading TrOCR model from: {TROCR_MODEL_NAME}")
processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
trocr = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME).to(device)

# ========== UTILITIES ========== #
def extract_plate(text: str) -> str | None:
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    match = re.fullmatch(r"([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})", text)
    if match and match.group(1) in VALID_STATES:
        return "".join(match.groups())
    return None

def recognize_plate(image) -> str | None:
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    output_ids = trocr.generate(inputs)
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return extract_plate(decoded)

def format_timestamp(ms: float) -> str:
    total_seconds = int(ms / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# ========== MAIN FUNCTION ========== #
def main():
    print(f"[INFO] Starting video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Failed to open video.")
        return

    plates_detected = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector(frame, stream=True, verbose=False)
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        for result in results:
            for box in result.boxes:
                if box.conf[0] < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_plate = frame[y1:y2, x1:x2]

                try:
                    plate = recognize_plate(cropped_plate)
                    if plate and plate not in plates_detected:
                        plates_detected.add(plate)
                        readable_time = format_timestamp(timestamp_ms)
                        print(f"[DETECTED] {plate} at {readable_time}")
                except Exception as e:
                    continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print("\n[RESULT] All detected plates:")
    for plate in sorted(plates_detected):
        print(plate)

# ========== ENTRY POINT ========== #
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
