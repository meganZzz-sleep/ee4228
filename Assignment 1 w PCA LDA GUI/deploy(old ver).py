import sys
import os
import cv2
import joblib
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QLineEdit
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
import time

THRESHOLD = 7000

# Define paths
dataset_path = "dataset"
model_path = "pca_lda_model.pkl"

# Load PCA+LDA model if available
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Face Recognition System")
        self.setGeometry(100, 100, 1920, 1440)  # Increase window size

        self.image_label = QLabel()
        self.image_label.setFixedSize(1920, 1440) #size of camera UI

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter name for new person")

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.capture_button = QPushButton("Capture New Person")
        self.exit_button = QPushButton("Exit")

        # Sidebar for confidence levels
        self.sidebar_layout = QVBoxLayout()  # Layout for the sidebar
        self.sidebar_widget = QWidget(self)
        self.sidebar_widget.setLayout(self.sidebar_layout)

        hbox = QHBoxLayout()
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.stop_button)
        hbox.addWidget(self.capture_button)
        hbox.addWidget(self.exit_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.name_input)
        vbox.addLayout(hbox)

        main_layout = QHBoxLayout(self)
        main_layout.addLayout(vbox)
        main_layout.addWidget(self.sidebar_widget)

        self.setLayout(main_layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.capture_button.clicked.connect(self.capture_images)
        self.exit_button.clicked.connect(self.close)

        self.image_counter = 0
        self.current_name = ""
        self.capture_mode = False

        self.last_update_time = time.time()  # Variable to store last update time
        self.update_interval = 3  # Update every 3 seconds

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()

    def capture_images(self):
        self.current_name = self.name_input.text().strip()
        if not self.current_name:
            print("Error: Name field is empty.")
            return

        person_path = os.path.join(dataset_path, self.current_name)
        os.makedirs(person_path, exist_ok=True)
        self.image_counter = 0
        self.capture_mode = True
        print(f"Capturing images for: {self.current_name}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

        predicted_faces = []  # List to store predicted names and confidences for all faces

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Face recognition
            face = gray_frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (480, 480))  # Ensure proper size

            predicted_name = "Unknown"
            confidence = 0  # Default confidence

            if model:
                mean = model['mean']
                eigenfaces = model['eigenfaces']
                lda_transform = model['lda']
                projections = model['projections']
                labels = model['labels']
                label_names = model['label_names']

                # Preprocess the captured face
                face_pca = np.dot(face_resized.flatten() - mean, eigenfaces)  # Project onto PCA space
                face_lda = np.dot(face_pca, lda_transform)  # Project onto LDA space
                distances = np.linalg.norm(projections - face_lda, axis=1)  # Compute distance

                # Find closest match
                sorted_indices = np.argsort(distances)
                min_dist = distances[sorted_indices[0]]

                if min_dist < THRESHOLD:  # Adjust threshold if needed
                    predicted_name = label_names[labels[sorted_indices[0]]]
                    confidence = 100 * (1 - min_dist / THRESHOLD)  # Example confidence calculation

            predicted_faces.append((predicted_name, confidence))  # Store predicted name and confidence

        # Update sidebar with all detected faces
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            # Only update the sidebar when the interval has passed
            self.last_update_time = current_time
            self.update_sidebar(predicted_faces)  # Pass the list of faces to update_sidebar

        # Don't display confidence level on face directly
        for (x, y, w, h), (predicted_name, confidence) in zip(faces, predicted_faces):
            cv2.putText(frame, f"{predicted_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize frame before displaying in UI
        frame_resized = cv2.resize(frame, (1920, 1440))
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = QImage(frame_resized, frame_resized.shape[1], frame_resized.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.image_label.setPixmap(pix)


    def update_sidebar(self, predicted_faces):
        # Clear the sidebar before updating
        for i in reversed(range(self.sidebar_layout.count())):
            widget = self.sidebar_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Add new predicted names and confidence levels to the sidebar
        for predicted_name, confidence in predicted_faces:
            label = QLabel(f"{predicted_name}: {confidence:.2f}%", self)
            self.sidebar_layout.addWidget(label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
