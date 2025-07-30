from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("runs/detect/train6/weights/best.pt")  # or yolov8s.pt if you want more accuracy

# Train
model.train(
    data="data.yaml",        # path to your YAML file
    epochs=100,              # total training epochs
    imgsz=640,               # image size
    batch=16,                # batch size (adjust if OOM)
    device=0,                # GPU id or 'cpu'
    name="yolo_plate_train", # run folder name
    project="runs/train",    # output folder
    workers=4,               # number of dataloader workers
    patience=20              # early stopping patience
)
