import cv2
import joblib
import numpy as np

# Define the path to the folder containing the .pkl files
model_path = r'C:\Users\yelhs\source\repos\EE4228clone\Assignment 1 test'

# Load the trained PCA and SVM models
pca = joblib.load(f'{model_path}\\pca_model.pkl')  # Use the full path to your saved PCA model
svm = joblib.load(f'{model_path}\\svm_model.pkl')  # Use the full path to your saved SVM model
label_encoder = joblib.load(f'{model_path}\\label_encoder.pkl')  # Use the full path to your label encoder

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a live camera feed
cap = cv2.VideoCapture(0)  # 0 is the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop, resize, and preprocess the detected face
        face = gray_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (90, 90))
        face_flattened = face_resized.flatten().reshape(1, -1)  # Flatten and reshape for PCA

        # Apply PCA transformation
        face_pca = pca.transform(face_flattened)

        # Predict the label using SVM
        label_encoded = svm.predict(face_pca)[0]
        label = label_encoder.inverse_transform([label_encoded])[0]  # Decode the label

        # Display the label on the frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
