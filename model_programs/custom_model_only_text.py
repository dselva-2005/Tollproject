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

# ========== SUPPRESS LOGS ==========
os.environ["YOLO_VERBOSE"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
LOGGER.setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# ========== CONFIGURATION ==========
YOLO_MODEL_PATH = "/home/dselva/MINIPROJECTTE/small_weights/best.pt"
VIDEO_SOURCE = "../test_samples/video1.mp4"
CONFIDENCE_THRESHOLD = 0.75
OCR_BACKEND = "paddle"  # "trocr", "paddle", "easyocr"
TROCR_MODEL = "microsoft/trocr-base-printed"

VALID_STATES = {
    "AP", "AR", "AS", "BR", "CH", "CT", "DN", "DD", "DL", "GA", "GJ", "HR", "HP",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== OCR BACKEND WRAPPER ==========
class OCRRecognizer:
    def __init__(self, backend="trocr"):
        self.backend = backend.lower()

        if self.backend == "trocr":
            print("[INFO] Loading TrOCR...")
            self.processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
            self.model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)

        elif self.backend == "paddle":
            print("[INFO] Loading PaddleOCR...")
            from paddleocr import PaddleOCR
            self.model = PaddleOCR(
                use_angle_cls=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )

        elif self.backend == "easyocr":
            print("[INFO] Loading EasyOCR...")
            import easyocr
            self.model = easyocr.Reader(["en"], gpu=(device == "cuda"))

        else:
            raise ValueError(f"OCR backend '{self.backend}' is not supported.")

    def recognize(self, image):
        if self.backend == "trocr":
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
            output_ids = self.model.generate(inputs)
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        elif self.backend == "paddle":
            result = self.model.predict(image)
            try:
                text = result[0]["rec_texts"][0]
            except:
                text = ""

        elif self.backend == "easyocr":
            result = self.model.readtext(image)
            text = result[0][1] if result else ""

        else:
            raise NotImplementedError(f"OCR backend '{self.backend}' not implemented.")

        return self.clean_plate(text)

    @staticmethod
    def clean_plate(text):
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        match = re.fullmatch(r"([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})", text)
        return "".join(match.groups()) if match and match.group(1) in VALID_STATES else None

# ========== UTILITY ==========
def format_timestamp(ms: float) -> str:
    total_seconds = int(ms / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# ========== MAIN ==========
def main():
    print(f"[INFO] Starting video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Failed to open video.")
        return

    detector = YOLO(YOLO_MODEL_PATH)
    ocr = OCRRecognizer(backend=OCR_BACKEND)
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
                    plate = ocr.recognize(cropped_plate)
                    if plate and plate not in plates_detected:
                        plates_detected.add(plate)
                        print(f"[DETECTED] {plate} at {format_timestamp(timestamp_ms)}")
                except:
                    continue

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    print("\n[RESULT] All detected plates:")
    for plate in sorted(plates_detected):
        print(plate)

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Elapsed time: {time.time() - start_time:.4f} seconds")
