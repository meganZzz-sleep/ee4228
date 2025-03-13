import cv2
import os

# Create a new folder to save the images
folder_name = 'captured_images'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Number of images to capture
num_images = 25

# Loop to capture 25 images
for i in range(num_images):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the frame to 480x480
    frame_resized = cv2.resize(frame, (480, 480))

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Display the captured grayscale image
    cv2.imshow('Captured Grayscale Frame', gray_frame)

    # Save the captured grayscale image to the new folder
    filename = os.path.join(folder_name, f"image_{i+1}.jpg")
    cv2.imwrite(filename, gray_frame)
    print(f"Saved {filename}")

    # Wait for 1 ms to capture the next image (press 'q' to quit early)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
