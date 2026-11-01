from ultralytics import YOLO

model = YOLO("runs/detect/runs/vehicle/vehicle-detection/weights/best.pt")

# Predict objects in a new image
results = model.predict(
    source="data/vehicle-detection/test/images",
    conf=0.5,
    save=True
)

print("Prediction completed!")