import sys
import os
import cv2
import joblib
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
                              QLineEdit, QDialog, QMessageBox, QSpacerItem, QSizePolicy, QDesktopWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from sklearn.model_selection import train_test_split
import pickle
import time

THRESHOLD = 7000
IMAGE_SIZE = (480, 480)

# Define paths
dataset_path = "dataset"
model_path = "pca_lda_model.pkl"

# Load PCA+LDA model if available, else None
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def load_dataset(dataset_path):
    X, y, labels = [], [], []
    label_map = {}
    label_counter = 0
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        if person_name not in label_map:
            label_map[person_name] = label_counter
            labels.append(person_name)
            label_counter += 1
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMAGE_SIZE)
            X.append(img.flatten() / 255.0)
            y.append(label_map[person_name])
    return np.array(X, dtype='float32'), np.array(y), labels


def pca(X, variance_threshold=0.98):
    # Calculate the mean face
    mean_face = np.mean(X, axis=0)
    
    # Center the data by subtracting the mean face
    X_centered = X - mean_face
    
    # Compute the covariance matrix
    cov_matrix = np.dot(X_centered, X_centered.T)
    
    # Perform eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    eigvecs = np.dot(X_centered.T, eigvecs)  # Align eigenvectors to the data
    eigvecs = eigvecs[:, ::-1]
    eigvals = eigvals[::-1]
    
    # Calculate the explained variance ratio
    explained_variance_ratio = eigvals / np.sum(eigvals)
    
    # Calculate the cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find the number of components needed to explain at least the threshold variance
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1  # +1 since index starts from 0
    
    # Select the number of components
    eigvecs = eigvecs[:, :num_components]
    
    # Normalize eigenvectors
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)
    
    return mean_face, eigvecs


def lda(X_proj, y):
    class_labels = np.unique(y)
    mean_total = np.mean(X_proj, axis=0)
    Sw = np.zeros((X_proj.shape[1], X_proj.shape[1]))
    Sb = np.zeros((X_proj.shape[1], X_proj.shape[1]))

    for c in class_labels:
        Xc = X_proj[y == c]
        mean_class = np.mean(Xc, axis=0)
        Sw += np.dot((Xc - mean_class).T, (Xc - mean_class))
        n_c = Xc.shape[0]
        mean_diff = (mean_class - mean_total).reshape(-1, 1)
        Sb += n_c * np.dot(mean_diff, mean_diff.T)

    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
    eigvecs = eigvecs[:, eigvals.argsort()[::-1]]
    return eigvecs.real


def project(X, mean, eigenfaces):
    return np.dot(X - mean, eigenfaces)


def train_model_and_save_pkl():
    X, y, labels = load_dataset(dataset_path)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Set the variance threshold (e.g., 98%)
    variance_threshold = 0.98
    
    # Use the updated PCA function to select the number of components based on the variance threshold
    mean_face, eigenfaces = pca(X_train, variance_threshold=variance_threshold)
    
    # Project the training data onto the PCA components
    X_train_pca = project(X_train, mean_face, eigenfaces)

    # Apply LDA to the PCA-transformed data
    lda_transform = lda(X_train_pca, y_train)
    num_lda_components = len(np.unique(y_train)) - 1
    lda_transform = lda_transform[:, :num_lda_components]
    
    # Project the data onto the LDA components
    X_train_lda = np.dot(X_train_pca, lda_transform)

    # Save the model as a pickle file
    with open(model_path, "wb") as f:
        pickle.dump({
            'mean': mean_face,
            'eigenfaces': eigenfaces,
            'lda': lda_transform,
            'projections': X_train_lda,
            'labels': y_train,
            'label_names': labels
        }, f)
    
    # Reload the model to update the global model
    model = joblib.load(model_path)  # Update global model
    print("Model trained and saved as pca_lda_model.pkl")




class CaptureWindow(QDialog):
    def __init__(self, parent=None, camera=None):
        super().__init__(parent)
        self.setWindowTitle("Capture New Person")
        self.setGeometry(300, 300, 600, 200)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.camera = camera

        self.layout = QVBoxLayout()
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter new person's name")
        self.capture_button = QPushButton("Take Photo")
        self.status_label = QLabel("Photos taken: 0/50")

        self.layout.addWidget(self.name_input)
        self.layout.addWidget(self.capture_button)
        self.layout.addWidget(self.status_label)
        self.setLayout(self.layout)

        self.capture_button.clicked.connect(self.capture_photo)
        self.image_counter = 0
        self.person_name = ""

        button_style = """
        QPushButton {
            background-color: white;
            border: 1px solid #ccc;
            padding: 8px;
            border-radius: 6px;
            font-size: 30px;
        }
        QPushButton:hover {
            background-color: #e6f2ff; /* Light blue on hover */
        }
        """
        self.capture_button.setStyleSheet(button_style)

        input_style = """
        QLineEdit {
            background-color: white;
            border: 1px solid #ccc;
            padding: 8px;
            border-radius: 6px;
            font-size: 30px;
        }
        QLineEdit:focus {
            border: 1px solid #66a3ff; /* Blue border on focus */
        }
        """
        self.name_input.setStyleSheet(input_style)

    def capture_photo(self):
        if not self.person_name:
            self.person_name = self.name_input.text().strip()
            if not self.person_name:
                self.status_label.setText("Name is required.")
                return
            os.makedirs(os.path.join(dataset_path, self.person_name), exist_ok=True)

        ret, frame = self.camera.read()
        if not ret:
            self.status_label.setText("Failed to read from camera.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            self.status_label.setText("No face detected. Try again.")
            return

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, IMAGE_SIZE)
            filename = os.path.join(dataset_path, self.person_name, f"img{self.image_counter}.jpg")
            cv2.imwrite(filename, face_resized)
            self.image_counter += 1
            self.status_label.setText(f"Photos taken: {self.image_counter}/50")
            break  # Only save one face per click

        if self.image_counter >= 50:
            self.status_label.setText("50 photos captured. Training model...")
            train_model_and_save_pkl()

            # Create the QMessageBox
            msg_box = QMessageBox(self.parent())  # Message box appears on the main window
            msg_box.setWindowTitle("Training Complete")
            msg_box.setText("Training complete. \n Please Refresh Model to view changes.")

            # Apply custom styles (matching button styles)
            msg_box.setStyleSheet("""
                QMessageBox {
                    border-radius: 15px;  /* Rounded corners */
                    padding: 10px;
                }
                QLabel {
                    font-size: 25px;  /* Optional: adjust font size */
                }
                QPushButton {
                    background-color: white;
                    border: 1px solid #ccc;
                    padding: 8px;
                    border-radius: 6px;
                    font-size: 20px;
                }
                QPushButton:hover {
                    background-color: #e6f2ff; /* Light blue on hover */
                }
            """)

            # Show the message box
            msg_box.exec_()

            self.accept()  # Close the window



# Main app updated to use the new dialog

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("F43 Group 31's Face Recognition System")
        self.setGeometry(100, 100, 2560, 1440)

        self.image_label = QLabel()
        self.image_label.setFixedSize(1920, 1440)

        # Buttons
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Freeze Camera")
        self.capture_button = QPushButton("Capture New Person")
        self.exit_button = QPushButton("Exit")
        self.refresh_button = QPushButton("Refresh Model")

        button_style = """
        QPushButton {
            background-color: white;
            border: 1px solid #ccc;
            padding: 8px;
            border-radius: 6px;
            font-size: 30px;
        }
        QPushButton:hover {
            background-color: #e6f2ff; /* Light blue on hover */
        }
        """

        self.start_button.setStyleSheet(button_style)
        self.stop_button.setStyleSheet(button_style)
        self.capture_button.setStyleSheet(button_style)
        self.exit_button.setStyleSheet(button_style)
        self.refresh_button.setStyleSheet(button_style)

        # Connect buttons
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.capture_button.clicked.connect(self.open_capture_window)
        self.exit_button.clicked.connect(self.close)
        self.refresh_button.clicked.connect(self.refresh_model)

        # Sidebar layout
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_widget = QWidget(self)
        self.sidebar_widget.setLayout(self.sidebar_layout)
        self.sidebar_widget.setStyleSheet("background-color: #f1f3f5;")  # Light grey sidebar
        self.sidebar_widget.setFixedWidth(470)  # Fixed width for sidebar

        # For the camera box (image_label):
        self.image_label.setStyleSheet("""
            border-radius: 15px;  /* Apply rounded corners to the box */
            border: 2px solid #ccc;  /* Optional: border around the camera box */
            background-color: white;  /* Optional: set a background color */
        """)

        # Apply a mask to the QLabel to make the camera feed rounded:
        self.image_label.setMask(self.image_label.mask())

        # For the confidence level box (sidebar_widget):
        self.sidebar_widget.setStyleSheet("""
            background-color: #f1f3f5;
            border-radius: 15px;  /* Add rounded corners */
            padding: 0px;  /* Optional: padding inside the box */
        """)

        # Confidence Level UI
        confidence_label = QLabel("Confidence Level")
        confidence_label.setStyleSheet("font-weight: bold; font-size: 30px; margin-bottom: 10px;")
        confidence_layout = QVBoxLayout()  # Use QVBoxLayout to stack elements vertically
        confidence_layout.addWidget(confidence_label, alignment=Qt.AlignTop | Qt.AlignCenter)  # Align the label to the top and center
        self.sidebar_layout.addLayout(confidence_layout)

        # Main layout (camera UI on the left and sidebar on the right)
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label)  # Camera feed on the left
        main_layout.addWidget(self.sidebar_widget)  # Confidence level on the right

        # Buttons at the bottom spanning the entire width of the window
        bottom_button_layout = QHBoxLayout()  # Use HBox for buttons to go in a row
        bottom_button_layout.addWidget(self.start_button)
        bottom_button_layout.addWidget(self.stop_button)
        bottom_button_layout.addWidget(self.capture_button)
        bottom_button_layout.addWidget(self.refresh_button)
        bottom_button_layout.addWidget(self.exit_button)

        # Create a container for the bottom buttons and set its size
        bottom_buttons_widget = QWidget()
        bottom_buttons_widget.setLayout(bottom_button_layout)
        bottom_buttons_widget.setFixedHeight(100)  # Set a fixed height for the bottom buttons

        # Set up the overall layout (camera + sidebar + bottom buttons)
        overall_layout = QVBoxLayout()
        overall_layout.addLayout(main_layout)
        overall_layout.addWidget(bottom_buttons_widget)

        self.setLayout(overall_layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Slow down confidence level updates
        self.last_update_time = time.time()
        self.update_interval = 3

        self.setStyleSheet("background-color: #c7d1d5;")  # Blue Grey Background

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()

    def open_capture_window(self):
        if self.cap is None or not self.cap.isOpened():
            print("Camera not started.")
            return
        capture_window = CaptureWindow(self, self.cap)
        capture_window.exec_()

    def update_frame(self):
        global model

        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

        predicted_faces = []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray_frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, IMAGE_SIZE)

            predicted_name = "Unknown"
            confidence = 100

            if model:
                mean = model['mean']
                eigenfaces = model['eigenfaces']
                lda_transform = model['lda']
                projections = model['projections']
                labels = model['labels']
                label_names = model['label_names']

                face_pca = np.dot(face_resized.flatten() - mean, eigenfaces)
                face_lda = np.dot(face_pca, lda_transform)
                distances = np.linalg.norm(projections - face_lda, axis=1)

                sorted_indices = np.argsort(distances)
                min_dist = distances[sorted_indices[0]]

                if min_dist < THRESHOLD:
                    predicted_name = label_names[labels[sorted_indices[0]]]
                    confidence = 100 - (100 * (1 - min_dist / THRESHOLD))

            predicted_faces.append((predicted_name, confidence))

        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            self.update_sidebar(predicted_faces)

        for (x, y, w, h), (predicted_name, _) in zip(faces, predicted_faces):
            cv2.putText(frame, f"{predicted_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_resized = cv2.resize(frame, (1920, 1440))
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = QImage(frame_resized, frame_resized.shape[1], frame_resized.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.image_label.setPixmap(pix)

    def update_sidebar(self, predicted_faces):
        # Clear the existing predicted faces (not the confidence level UI)
        for i in reversed(range(self.sidebar_layout.count())):
            widget = self.sidebar_layout.itemAt(i).widget()
            if widget and widget != self.sidebar_layout.itemAt(0).widget():  # Ignore the "Confidence Level" UI
                widget.deleteLater()

        # Add new predicted faces (names and confidence) directly below the Confidence Level UI
        for predicted_name, confidence in predicted_faces:
            label = QLabel(f"{predicted_name}: {confidence:.2f}%", self)
            label.setStyleSheet("font-size: 30px; margin: 2px 0;")  # Reduced spacing
            self.sidebar_layout.addWidget(label, alignment=Qt.AlignTop | Qt.AlignCenter)  # Align to the top and center


    def refresh_model(self):
        global model
        train_model_and_save_pkl()
        model = joblib.load(model_path)
    
        # Create the QMessageBox
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Model Refreshed")
        msg_box.setText("The Face Recognition Model has been retrained and updated.")
    
        # Create an 'Ok' button manually to ensure it's available
        button = msg_box.addButton(QMessageBox.Ok)
    
        # Apply custom style to the 'Ok' button
        button.setStyleSheet("""
            background-color: white;
            border: 1px solid #ccc;
            padding: 8px;
            border-radius: 6px;
            font-size: 20px;
        """)

        # Show the message box
        msg_box.exec_()


if __name__ == "__main__":
    print("Training model from dataset...")
    train_model_and_save_pkl()
    model = joblib.load(model_path)

    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())