import cv2
from ultralytics import YOLO
import numpy as np
import pygame

# Initialize pygame for sound playback
pygame.mixer.init()

# Load the first YOLO model
model1 = YOLO("yolov8m.pt")  # Replace with your first model path
# Load the second YOLO model or another model
model2 = YOLO("yolov8m.pt")  # Replace with your second model path

# Load the video file
video_path = "sam.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # Run inference on the first model
    results1 = model1(frame, classes=[0], verbose=False)  # Adjust classes as needed

    # Run inference on the second model
    results2 = model2(frame, classes=[1], verbose=False)  # Adjust classes as needed

    # Process results from both models
    # For model 1
    annotated_frame1 = results1[0].plot()
    
    # For model 2
    annotated_frame2 = results2[0].plot()

    # Combine the annotations (this is a basic example; you may need to adjust)
    combined_frame = cv2.addWeighted(annotated_frame1, 0.5, annotated_frame2, 0.5, 0)

    # Display the combined frame with detections
    cv2.imshow("Combined Model Detection", combined_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  # Clean up the pygame mixer
