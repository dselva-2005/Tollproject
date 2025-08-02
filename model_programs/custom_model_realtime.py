import os
import re
import cv2
import time
import torch
import logging
import warnings
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from paddleocr import PaddleOCR
from transformers.utils import logging as hf_logging
from ultralytics.utils import LOGGER

# ========== SUPPRESS LOGS ==========
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LOGGER.setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# ========== CONSTANTS ==========
CONFIDENCE_THRESHOLD = 0.75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_STATES = {
    "AP", "AR", "AS", "BR", "CH", "CT", "DN", "DD", "DL", "GA",
    "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB"
}

# ========== MODEL LOADER ==========
class ModelLoader:
    def __init__(self, yolo_path: str, model_type: str, model_name: str):
        self.detector = YOLO(yolo_path, verbose=False)
        self.model_type = model_type.lower()
        if self.model_type == "trocr":
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)
        elif self.model_type == "paddleocr":
            self.model = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
        else:
            raise ValueError("Unsupported OCR model type: choose 'trocr' or 'paddleocr'")

    def detect_plates(self, frame):
        return self.detector(frame, stream=True)

    def recognize_text(self, plate_img):
        if self.model_type == "trocr":
            pil_img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)
            output_ids = self.model.generate(inputs)
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return text.strip()
        elif self.model_type == "paddleocr":
            results = self.model.predict(plate_img)
            for result in results:
                if result['rec_texts'][0]: return result['rec_texts'][0]
                else : return ""

# ========== UTILITIES ==========
def extract_valid_plate(text: str) -> str | None:
    text = re.sub(r"[^A-Z0-9]", "", text.upper().replace(" ", ""))
    pattern = re.compile(r"^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$")
    match = pattern.fullmatch(text)
    if not match or match.group(1) not in VALID_STATES:
        return None
    return "".join(match.groups())

def draw_plate_box(frame, plate_text, box_coords):
    x1, y1, x2, y2 = box_coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def format_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ========== ANPR SYSTEM ==========
class ANPRSystem:
    def __init__(self, yolo_path: str, ocr_type: str, ocr_model: str, source: str):
        self.model_loader = ModelLoader(yolo_path, ocr_type, ocr_model)
        self.capture = cv2.VideoCapture(source)
        self.plates_detected = set()

        if not self.capture.isOpened():
            raise RuntimeError(f"❌ Failed to open video stream: {source}")
        print("✅ Stream opened. Press 'q' to quit.")

    def process_frame(self, frame):
        # timestamp_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)
        # timestamp = datetime.utcfromtimestamp(timestamp_ms / 1000).strftime("%H:%M:%S.%f")[:-3]
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        detections = self.model_loader.detect_plates(frame)
        for result in detections:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]
                try:
                    raw_text = self.model_loader.recognize_text(plate_img)
                    plate = extract_valid_plate(raw_text)
                    if plate:
                        print(f"[{timestamp}] '{raw_text}' → '{plate}'")
                        draw_plate_box(frame, plate, (x1, y1, x2, y2))
                except:
                    continue
        return frame

    def run(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            frame = self.process_frame(frame)
            cv2.namedWindow("ANPR System", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ANPR System", 1280, 720)
            cv2.imshow("ANPR System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cleanup()

    def cleanup(self):
        self.capture.release()
        cv2.destroyAllWindows()

# ========== MAIN FUNCTION ==========
def main():
    YOLO_PATH = "/home/dselva/MINIPROJECTTE/nano640/best.pt"
    OCR_TYPE = "paddleocr"  # "trocr" or "paddleocr"
    OCR_MODEL = "microsoft/trocr-base-printed"  # ignored if paddleocr
    VIDEO_SOURCE = "../test_samples/video3.mp4"
    # VIDEO_SOURCE = "http://192.168.1.100:8080/video"

    try:
        anpr = ANPRSystem(YOLO_PATH, OCR_TYPE, OCR_MODEL, VIDEO_SOURCE)
        anpr.run()
    except Exception as e:
        print(f"🚨 Application error: {e}")


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Elapsed time: {time.time() - start_time:.4f} seconds")
