import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    return box_annotator.annotate(frame.copy(), detections=detections)

sv.process_video(
    source_path="sample.mp4",
    target_path="result.mp4",
    callback=callback
)