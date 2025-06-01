# import cv2
# from ultralytics import YOLO
# import numpy as np
# import pygame


# pygame.mixer.init()


# points = [(104, 317), (460, 112), (734, 287), (1277, 530), (1211, 718), (644, 717), (189, 499)]
# polygon = np.array(points, dtype=np.int32)


# person_model = YOLO("yolov8n.pt")  
# helmet_model = YOLO("helmet.pt")   
# mask_model = YOLO("mask_all.pt")   


# video_path = "mask_720.mp4"
# cap = cv2.VideoCapture(video_path)


# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()


# line_color = (0, 255, 0)  
# line_thickness = 2  
# rect_thickness = 2  
# min_size = 100  


# three_men_sound = "alert_tamil.mp3"   
# helmet_sound = "alert_helmet.mp3"     
# mask_sound = "alert_mask.mp3"         
# pygame.mixer.music.load(mask_sound)


# sound_played = False

# frame_skip = 4  
# frame_count = 0  



# def put_label_on_box(annotated_frame, label, x_min, y_min):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.6
#     color = (255, 255, 255)  
#     thickness = 2
#     label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

    
#     label_y_min = max(y_min, label_size[1] + 10)  
#     cv2.putText(annotated_frame, label, (x_min, label_y_min - 10), font, font_scale, color, thickness)


# while cap.isOpened():

#     success, frame = cap.read()

#     if not success:
#         break  

#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         continue  


#     person_results = person_model.track(frame, classes=[0], verbose=False)  # Detect people
#     helmet_results = helmet_model.track(frame, classes=[0], conf=0.7, verbose=False, persist=True)  # Detect helmets
#     mask_results = mask_model.track(frame, classes=[0], conf=0.7, verbose=False, persist=True)  # Detect masks

#     # Create a copy of the frame for annotation
#     annotated_frame = frame.copy()


#     for i in range(len(points)):
#         start_point = points[i]
#         end_point = points[(i + 1) % len(points)]  
#         cv2.line(annotated_frame, start_point, end_point, line_color, line_thickness)


#     for point in points:
#         cv2.circle(annotated_frame, point, 5, (0, 0, 255), -1)  


#     people_inside_polygon_count = 0


#     mask_detected_with_min_size = False


#     for result in person_results:
#         for bbox in result.boxes:
#             x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
#             center_x = (x_min + x_max) // 2
#             center_y = (y_min + y_max) // 2


#             cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  # Yellow box
#             put_label_on_box(annotated_frame, "Person", x_min, y_min)  # Add "Person" label



#             is_inside = cv2.pointPolygonTest(polygon, (center_x, center_y), False)
#             if is_inside >= 0:
#                 people_inside_polygon_count += 1 


#     helmet_detected = False
#     for result in helmet_results:
#         for bbox in result.boxes:
#             x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
#             width = x_max - x_min
#             height = y_max - y_min

#             if width >= min_size and height >= min_size:
#                 helmet_detected = True

#                 cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red box
#                 put_label_on_box(annotated_frame, "Helmet", x_min, y_min)  # Add "Helmet" label
                


#     for result in mask_results:
#         for bbox in result.boxes:
#             x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
#             width = x_max - x_min
#             height = y_max - y_min

#             if width >= min_size and height >= min_size:
#                 mask_detected_with_min_size = True

#                 cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue box
#                 put_label_on_box(annotated_frame, "Mask", x_min, y_min)  


#     if mask_detected_with_min_size and not sound_played:
#         pygame.mixer.music.load(mask_sound)
#         pygame.mixer.music.play()
#         sound_played = True
#     elif helmet_detected and not sound_played:
#         pygame.mixer.music.load(helmet_sound)
#         pygame.mixer.music.play()
#         sound_played = True
#     elif people_inside_polygon_count >= 3 and not sound_played:
#         pygame.mixer.music.load(three_men_sound)
#         pygame.mixer.music.play()
#         sound_played = True


#     if not pygame.mixer.music.get_busy():
#         sound_played = False


#     cv2.imshow("YOLOv8 Combined Detection", annotated_frame)


#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break


# cap.release()
# cv2.destroyAllWindows()
# pygame.mixer.quit()


import cv2
from ultralytics import YOLO
import numpy as np
import pygame

pygame.mixer.init()

points = [(104, 317), (460, 112), (734, 287), (1277, 530), (1211, 718), (644, 717), (189, 499)]
polygon = np.array(points, dtype=np.int32)

person_model = YOLO("yolov8n.pt")  
helmet_model = YOLO("helmet.pt")   
mask_model = YOLO("mask_all.pt")   
gun_model = YOLO("gun100.pt")  
tools_model = YOLO("Ftools.pt")  

video_path = 0
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

line_color = (0, 255, 0)  
line_thickness = 2  
rect_thickness = 2  
min_size = 100  

three_men_sound = "alert_tamil.mp3"   
helmet_sound = "alert_helmet.mp3"     
mask_sound = "alert_mask.mp3"         
pygame.mixer.music.load(mask_sound)

sound_played = False

frame_skip = 4
frame_count = 0  

def put_label_on_box(annotated_frame, label, x_min, y_min):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  
    thickness = 2
    label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    label_y_min = max(y_min, label_size[1] + 10)  
    cv2.putText(annotated_frame, label, (x_min, label_y_min - 10), font, font_scale, color, thickness)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break  

    if np.mean(frame) < 10:  # Check if frame is almost black
        print("Black screen detected!")

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  

    person_results = person_model.track(frame, classes=[0], verbose=False)  
    helmet_results = helmet_model.track(frame, classes=[0], conf=0.7, verbose=False, persist=True)  
    mask_results = mask_model.track(frame, classes=[0], conf=0.7, verbose=False, persist=True)  
    gun_results = gun_model.track(frame, classes=[0], conf=0.7, verbose=False, persist=True)  
    tools_results = tools_model.track(frame, classes=[0], conf=0.7, verbose=False, persist=True)  
    
    annotated_frame = frame.copy()

    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]  
        cv2.line(annotated_frame, start_point, end_point, line_color, line_thickness)

    for point in points:
        cv2.circle(annotated_frame, point, 5, (0, 0, 255), -1)  

    people_inside_polygon_count = 0
    mask_detected_with_min_size = False

    for result in person_results:
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  
            put_label_on_box(annotated_frame, "Person", x_min, y_min)  
            is_inside = cv2.pointPolygonTest(polygon, (center_x, center_y), False)
            if is_inside >= 0:
                people_inside_polygon_count += 1  

    helmet_detected = False
    for result in helmet_results:
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            if (x_max - x_min) >= min_size and (y_max - y_min) >= min_size:
                helmet_detected = True
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  
                put_label_on_box(annotated_frame, "Helmet", x_min, y_min)  
    
    for result in mask_results:
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            if (x_max - x_min) >= min_size and (y_max - y_min) >= min_size:
                mask_detected_with_min_size = True
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  
                put_label_on_box(annotated_frame, "Mask", x_min, y_min)  
    
    for result in gun_results:
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)  
            put_label_on_box(annotated_frame, "Gun", x_min, y_min)  
    
    for result in tools_results:
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (128, 0, 128), 2)  
            put_label_on_box(annotated_frame, "Tool", x_min, y_min)  

    cv2.imshow("YOLOv8 Combined Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
