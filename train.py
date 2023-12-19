from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# model.train(data="config.yaml", epochs=10)
model.train(data="data.yaml", epochs=1)
