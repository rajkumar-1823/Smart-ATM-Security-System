import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("toolsf.pt")

# Open the video file
video_path = "tools_2.mp4"
cap = cv2.VideoCapture(video_path)

# Define the target frame size for 720p
target_width, target_height = 1280, 720

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Resize the frame to 720p
        frame = cv2.resize(frame, (target_width, target_height))

        # Run YOLO tracking on the resized frame, persisting tracks between frames
        results = model.track(frame, conf=0.5, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
