from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8m.pt')

# Export to ONNX format
model.export(format='engine', device="cpu")
