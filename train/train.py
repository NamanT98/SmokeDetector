from ultralytics import YOLO

print("Imported YOLO Successfully")

model=YOLO("yolov8m.pt")

print("Created Model Successfully")

model.train(data="Data\data.yaml",epochs=100,optimizer="Adam")
