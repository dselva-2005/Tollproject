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
    def __init__(self, yolo_path: str, trocr_model: str):
        self.detector = YOLO(yolo_path, verbose=False)

    def detect_plates(self, frame):
        return self.detector(frame, stream=True)


def draw_plate_box(frame, plate_text, box_coords):
    x1, y1, x2, y2 = box_coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def format_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ========== ANPR SYSTEM ==========
class ANPRSystem:
    def __init__(self, yolo_path: str, trocr_model: str, source: str):
        self.model_loader = ModelLoader(yolo_path, trocr_model)
        self.capture = cv2.VideoCapture(source)
        self.plates_detected = set()

        if not self.capture.isOpened():
            raise RuntimeError(f"❌ Failed to open video stream: {source}")
        print("✅ Stream opened. Press 'q' to quit.")

    def process_frame(self, frame):

        detections = self.model_loader.detect_plates(frame)
        for result in detections:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_plate_box(frame, "plate", (x1, y1, x2, y2))

        return frame

    def run(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            frame = self.process_frame(frame)
            cv2.namedWindow("ANPR - TrOCR + YOLOv8", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ANPR - TrOCR + YOLOv8", 1280, 720)
            cv2.imshow("ANPR - TrOCR + YOLOv8", frame)
            # time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cleanup()

    def cleanup(self):
        self.capture.release()
        cv2.destroyAllWindows()

# ========== MAIN FUNCTION ==========
def main():
    YOLO_PATH = "/home/dselva/MINIPROJECTTE/nanomodel/best.pt"
    TYOCR_MODEL = "microsoft/trocr-base-printed"
    VIDEO_SOURCE = "../test_samples/video3.mp4"  # or use live stream URL

    try:
        anpr = ANPRSystem(YOLO_PATH, TYOCR_MODEL, VIDEO_SOURCE)
        anpr.run()
    except Exception as e:
        print(f"🚨 Application error: {e}")

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")