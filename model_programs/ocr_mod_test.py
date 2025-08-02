import os
import time
import torch
import pandas
import warnings
import logging
from PIL import Image
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import logging as hf_logging

# ========= Suppress Warnings and Logs =========
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()

# ========= Load Models =========
print("[INFO] Loading PaddleOCR...")
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

print("[INFO] Loading TrOCR...")
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-small-printed"
).to(device)

# ========= Input Image =========
df = pandas.read_csv('../TrOcr_training/archive/dataset/label.csv')
sample = df.sample(n=1)

image_path = '../TrOcr_training/archive/dataset/input_images/' + sample.iloc[0]['imageFile']
true_value = sample.iloc[0]['imageLabel']
print(image_path, true_value)


print(" ========= True Value =========")
print(true_value)


# ========= Display Image =========
def plot_image(image_path, title="Input Image"):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(6, 3))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


plot_image(image_path)

# ========= PaddleOCR Prediction =========
start_paddle = time.time()
paddle_result = ocr.predict(input=image_path)
end_paddle = time.time()

print("\n===== PaddleOCR Output =====")
for res in paddle_result:
    print(res["rec_texts"][0])
print(f"PaddleOCR Inference Time: {end_paddle - start_paddle:.3f} seconds")

# ========= TrOCR Prediction =========
image = Image.open(image_path).convert("RGB")
pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values.to(
    device
)

start_trocr = time.time()
with torch.no_grad():
    generated_ids = trocr_model.generate(pixel_values)
    trocr_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ]
end_trocr = time.time()

print("\n===== TrOCR Output =====")
print(trocr_text)
print(f"TrOCR Inference Time: {end_trocr - start_trocr:.3f} seconds")
