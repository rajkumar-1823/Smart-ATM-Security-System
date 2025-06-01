import cv2
from ultralytics import YOLO
import numpy as np
import pygame


pygame.mixer.init()


# points = [(678, 309), (357, 533), (483, 717), (1100, 716), (1141, 628)]
points =[(104, 317), (460, 112), (734, 287), (1277, 530), (1211, 718), (644, 717), (189, 499)]


polygon = np.array(points, dtype=np.int32)


model = YOLO("yolov8n.pt")


video_path = "main_720.mp4"  
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


line_color = (0, 255, 0)  
line_thickness = 2  


sound_file = "alert_tamil.mp3"  
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

    
    results = model.track(frame, classes=[0], verbose=False)  

    
    annotated_frame = results[0].plot()

   
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]  
        cv2.line(annotated_frame, start_point, end_point, line_color, line_thickness)


    for point in points:
        cv2.circle(annotated_frame, point, 5, (0, 0, 255), -1)  

    
    people_inside_polygon_count = 0

    
    for result in results:
        for bbox in result.boxes:
            
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])

            
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            
            is_inside = cv2.pointPolygonTest(polygon, (center_x, center_y), False)

            if is_inside >= 0:  
                people_inside_polygon_count += 1  

    
    if people_inside_polygon_count > 2 and not sound_played and not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()
        sound_played = True

    
    if not pygame.mixer.music.get_busy():
        sound_played = False

    
    cv2.imshow("YOLOv8 Detection with Custom Lines", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  
