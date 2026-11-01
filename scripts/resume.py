from ultralytics import YOLO

model = YOLO("runs/detect/runs/vehicle/vehicle-detection/weights/best.pt")

model.train(resume=True)