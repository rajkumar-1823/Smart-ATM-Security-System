import cv2
from ultralytics import YOLO
import numpy as np
import pygame
from concurrent.futures import ThreadPoolExecutor

# Initialize pygame for sound playback
pygame.mixer.init()

# List of selected points (polygon for people detection)
points = [(104, 317), (460, 112), (734, 287), (1277, 530), (1211, 718), (644, 717), (189, 499)]
polygon = np.array(points, dtype=np.int32)

# Load YOLO models for person, helmet, and mask detection
person_model = YOLO("yolov8n.pt")
helmet_model = YOLO("helmet.pt")
mask_model = YOLO("mask_all.pt")

# Load the video file
video_path = "main_720.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define color and thickness for lines and rectangles
line_color = (0, 255, 0)
line_thickness = 2
rect_thickness = 2
min_size = 100  # Minimum size for bounding boxes (100x100)

# Preload sound files
pygame.mixer.music.load("alert_mask.mp3")
three_men_sound = pygame.mixer.Sound("alert_tamil.mp3")
helmet_sound = pygame.mixer.Sound("alert_helmet.mp3")

# Variables to track sound playback
sound_played = False

frame_skip = 4  # Process every 4th frame (higher skip for faster processing)
frame_count = 0  # Initialize the frame counter

# ThreadPoolExecutor for parallel processing of YOLO models
executor = ThreadPoolExecutor(max_workers=3)

def detect_person(frame):
    return person_model.track(frame, classes=[0], conf=0.5, verbose=False, persist=True)

def detect_helmet(frame):
    return helmet_model.track(frame, classes=[0], conf=0.5, verbose=False, persist=True)

def detect_mask(frame):
    return mask_model.track(frame, classes=[0], conf=0.5, verbose=False, persist=True)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Downscale the frame to improve performance
    frame = cv2.resize(frame, (1280, 720))

    # Run YOLO model inferences in parallel
    future_person = executor.submit(detect_person, frame)
    future_helmet = executor.submit(detect_helmet, frame)
    future_mask = executor.submit(detect_mask, frame)

    # Get the results from each YOLO model
    person_results = future_person.result()
    helmet_results = future_helmet.result()
    mask_results = future_mask.result()

    annotated_frame = frame.copy()

    # Draw lines connecting the points on the frame
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]
        cv2.line(annotated_frame, start_point, end_point, line_color, line_thickness)

    for point in points:
        cv2.circle(annotated_frame, point, 5, (0, 0, 255), -1)

    people_inside_polygon_count = 0
    helmet_detected = False
    mask_detected_with_min_size = False

    # Process person detection results
    for result in person_results:
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            is_inside = cv2.pointPolygonTest(polygon, (center_x, center_y), False)
            if is_inside >= 0:
                people_inside_polygon_count += 1

    # Process helmet detection results
    for result in helmet_results:
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            if (x_max - x_min) >= min_size and (y_max - y_min) >= min_size:
                helmet_detected = True
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Process mask detection results
    for result in mask_results:
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            if (x_max - x_min) >= min_size and (y_max - y_min) >= min_size:
                mask_detected_with_min_size = True
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Sound logic: mask > helmet > three men
    if mask_detected_with_min_size and not sound_played:
        pygame.mixer.music.play()
        sound_played = True
    elif helmet_detected and not sound_played:
        helmet_sound.play()
        sound_played = True
    elif people_inside_polygon_count >= 3 and not sound_played:
        three_men_sound.play()
        sound_played = True

    if not pygame.mixer.music.get_busy():
        sound_played = False

    cv2.imshow("YOLOv8 Combined Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
