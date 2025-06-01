###############################Crowd###############################################

import cv2
from ultralytics import YOLO
import numpy as np
import pygame

# Initialize pygame for sound playback
pygame.mixer.init()

# List of selected points (ensure proper order to avoid crossing lines)
# points = [(678, 309), (357, 533), (483, 717), (1100, 716), (1141, 628)]
points =[(104, 317), (460, 112), (734, 287), (1277, 530), (1211, 718), (644, 717), (189, 499)]

# Convert points to a numpy array for use with cv2 functions
polygon = np.array(points, dtype=np.int32)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Load the video file
video_path = "main_720.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define color and thickness for lines and rectangles
line_color = (0, 255, 0)  # Green color in BGR
line_thickness = 2  # Line thickness

# Sound file to play when the condition is true
sound_file = "alert_tamil.mp3"  # Replace with your sound file path
pygame.mixer.music.load(sound_file)

# Variable to track whether the sound has been played for the current frame
sound_played = False

frame_skip = 2  # Process every 2nd frame
frame_count = 0  # Initialize the frame counter

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        # Break the loop if the end of the video is reached or there’s an error
        break

    frame_count += 1

    # Skip frames that are not divisible by the frame_skip value
    if frame_count % frame_skip != 0:
        continue  # Skip this frame

    # Run YOLOv8 inference on the frame
    results = model.track(frame, classes=[0], verbose=False)  # Adjust classes as needed, 0 represents person class

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Draw lines connecting the points on the annotated frame
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]  # Loop back to the first point
        cv2.line(annotated_frame, start_point, end_point, line_color, line_thickness)

    # Draw the points on the frame for better visualization
    for point in points:
        cv2.circle(annotated_frame, point, 5, (0, 0, 255), -1)  # Red points

    # Initialize a counter for the number of detected people inside the polygon
    people_inside_polygon_count = 0

    # Check if any detected person is inside the polygon
    for result in results:
        for bbox in result.boxes:
            # Extract the bounding box coordinates (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])

            # Calculate the center of the bounding box
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Check if the center of the bounding box is inside the polygon
            is_inside = cv2.pointPolygonTest(polygon, (center_x, center_y), False)

            if is_inside >= 0:  # 1 for inside, 0 for on the edge
                people_inside_polygon_count += 1  # Increment the counter

    # Play sound if two or more people are detected inside the polygon and the sound is not already playing
    if people_inside_polygon_count > 2 and not sound_played and not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()
        sound_played = True

    # Reset sound_played only after the sound has finished playing
    if not pygame.mixer.music.get_busy():
        sound_played = False

    # Display the annotated frame with detected objects and custom lines
    cv2.imshow("YOLOv8 Detection with Custom Lines", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  # Clean up the pygame mixer


# import cv2
# from ultralytics import YOLO
# import numpy as np
# import pygame

# # Initialize pygame for sound playback
# pygame.mixer.init()

# # List of selected points (ensure proper order to avoid crossing lines)
# points = [(678, 309), (357, 533), (483, 717), (1100, 716), (1141, 628)]

# # Convert points to a numpy array for use with cv2 functions
# polygon = np.array(points, dtype=np.int32)

# # Load the YOLOv8 model for person detection
# person_model = YOLO("yolov8l.pt")

# # Load the second YOLO model for helmet detection
# helmet_model = YOLO("helmet.pt")

# # Load the video file
# video_path = "val.mp4"  # Replace with your video file path
# cap = cv2.VideoCapture(video_path)

# # Check if the video opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Define color and thickness for lines and rectangles
# line_color = (0, 255, 0)  # Green color in BGR for lines
# line_thickness = 2  # Line thickness

# # Colors for bounding boxes
# person_color = (0, 255, 0)  # Green for person
# helmet_color = (0, 0, 255)  # Red for helmet

# # Sound file to play when the condition is true
# sound_file = "alert_tamil.mp3"  # Replace with your sound file path
# pygame.mixer.music.load(sound_file)

# # Variable to track whether the sound has been played for the current frame
# sound_played = False

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if not success:
#         # Break the loop if the end of the video is reached or there’s an error
#         break

#     # Run YOLOv8 inference on the frame for person detection (class 0 represents person)
#     person_results = person_model(frame, classes=[0], verbose=False)

#     # Run YOLOv8 inference on the frame for helmet detection (adjust class index as per your model)
#     helmet_results = helmet_model(frame, verbose=False)

#     # Draw lines connecting the points on the frame
#     for i in range(len(points)):
#         start_point = points[i]
#         end_point = points[(i + 1) % len(points)]  # Loop back to the first point
#         cv2.line(frame, start_point, end_point, line_color, line_thickness)

#     # Draw the points on the frame for better visualization
#     for point in points:
#         cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Red points

#     # Initialize a counter for the number of detected people inside the polygon
#     people_inside_polygon_count = 0
#     helmet_detected = False

#     # Process person detections
#     for result in person_results:
#         for bbox in result.boxes:
#             # Extract the bounding box coordinates (x_min, y_min, x_max, y_max)
#             x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])

#             # Draw the person bounding box
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), person_color, 2)

#             # Calculate the center of the bounding box
#             center_x = (x_min + x_max) // 2
#             center_y = (y_min + y_max) // 2

#             # Check if the center of the bounding box is inside the polygon
#             is_inside = cv2.pointPolygonTest(polygon, (center_x, center_y), False)

#             if is_inside >= 0:  # 1 for inside, 0 for on the edge
#                 people_inside_polygon_count += 1  # Increment the counter

#     # Process helmet detections
#     for result in helmet_results:
#         for bbox in result.boxes:
#             # Extract the bounding box coordinates for helmet
#             x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])

#             # Draw the helmet bounding box
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), helmet_color, 2)

#             # A helmet was detected
#             helmet_detected = True

#     # Play sound if two or more people are detected inside the polygon and a helmet is detected, and the sound is not already playing
#     if people_inside_polygon_count >= 2 and helmet_detected and not sound_played and not pygame.mixer.music.get_busy():
#         pygame.mixer.music.play()
#         sound_played = True

#     # Reset sound_played only after the sound has finished playing
#     if not pygame.mixer.music.get_busy():
#         sound_played = False

#     # Display the frame with detected objects and custom lines
#     cv2.imshow("YOLOv8 Detection with Custom Lines", frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
# pygame.mixer.quit()  # Clean up the pygame mixer
