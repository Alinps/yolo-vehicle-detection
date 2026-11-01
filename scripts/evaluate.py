from ultralytics import YOLO

model = YOLO("runs/detect/runs/vehicle/vehicle-detection/weights/last.pt")

metrics = model.val()

print("mAP50", metrics.box.map50)
print("mAP50-95", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)