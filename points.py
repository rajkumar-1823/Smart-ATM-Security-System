import cv2

# List to store points
points = []

# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the point to the list
        points.append((x, y))
        print(f"Point selected: ({x}, {y})")

        # Display the point on the image
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", param)

# Load the image file
image_path = "scs.png"  # Replace with your image file path
image = cv2.imread(image_path)

if image is not None:
    # Display the image and set up the mouse callback
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event, image)

    # Wait until 'q' is pressed to exit
    cv2.waitKey(0)

    # Print the selected points
    print("Selected points:", points)
else:
    print("Error loading image.")

cv2.destroyAllWindows()


# Point selected: (678, 309)
# Point selected: (357, 533)
# Point selected: (483, 717)
# Point selected: (1100, 716)
# Point selected: (1141, 628)
# Selected points: [(678, 309), (357, 533), (483, 717), (1100, 716), (1141, 628)]