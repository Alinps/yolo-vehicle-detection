from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="data/vehicle-detection/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,

    #Experiment name
    project="runs/vehicle",
    name="vehicle-detection",

    # Saving training result
    save=True,
    plots=True
)

print("YOLO training complete")