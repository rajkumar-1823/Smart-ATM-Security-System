import cv2
from ultralytics import YOLO
import pygame


pygame.mixer.init()

# Load the YOLOv8 model
model = YOLO("mask_all.pt")


video_path = 0  
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


rect_color = (0, 255, 0)  
rect_thickness = 2  
min_size = 100  


sound_file = "alert_mask.mp3"  
pygame.mixer.music.load(sound_file)


sound_played = False

frame_skip = 2  
frame_count = 0 


while cap.isOpened():
   
    success, frame = cap.read()

    if not success:
        
        break

    frame_count += 1

    
    if frame_count % frame_skip != 0:
        continue  

    
    results = model.track(frame, classes=[0], conf=0.5, verbose=False,persist=True)  

  
    annotated_frame = results[0].plot()

 
    mask_detected_with_min_size = False

    
    for result in results:
        for bbox in result.boxes:
            
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])

            
            width = x_max - x_min
            height = y_max - y_min

            
            if width >= min_size and height >= min_size:
                mask_detected_with_min_size = True


    if mask_detected_with_min_size and not sound_played and not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()
        sound_played = True


    if not pygame.mixer.music.get_busy():
        sound_played = False


    cv2.imshow("YOLOv8 mask Detection", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit() 
