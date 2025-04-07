import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QInputDialog, QMessageBox, QDialog, QProgressBar
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

from fine_tuned1 import get_embedding  # Import your function


class FaceRecognition:
    def __init__(self, database_path="face_database.pkl"):
        self.database_path = database_path
        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.known_faces = self.load_face_database()

        # Enable OpenCV multi-threading
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)

    def load_face_database(self):
        print("📂 Loading face database...")
        try:
            with open("face_database.pkl", "rb") as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} known faces!")
            return data
        except FileNotFoundError:
            print("ERROR: face_database.pkl not found! Run `python face_database.py` first.")
            exit()
    
    def del_from_database(self,name):
        try:
            with open("face_database.pkl", "rb") as f:
                data = pickle.load(f)
            if name in data:
                del data[name]
                with open("face_database.pkl", "wb") as f:
                    pickle.dump(data, f)
                print(f"Deleted {name} from the database.")
            else:
                print(f"{name} not found in the database.")
        except FileNotFoundError:
            print("ERROR: face_database.pkl not found! Run `python face_database.py` first.")
            exit()

    def real_time_face_recognition(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img_rgb)
        results = []
        last_face = None
        last_embedding = None

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y) # new
            face_img = img_rgb[y:y+h, x:x+w]
            
            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                continue
            face_img = cv2.resize(face_img, (160, 160))

            # Use cached embedding if the same face appears
            if last_face is not None and np.array_equal(last_face, face_img):
                embedding = last_embedding
            else:
                embedding = self.embedder.embeddings([face_img])[0]
                last_face = face_img.copy()
                last_embedding = embedding

            min_dist = float("inf")
            identity = "Unknown"

            for name, db_embedding in self.known_faces.items():
                dist = cosine(db_embedding, embedding)
                if dist < 0.4:
                    min_dist = dist
                    identity = name

            results.append((x, y, w, h, identity))

        return results
    
class RecognitionThread(QThread):
    registration_done = pyqtSignal(str, np.ndarray)
    progress_updated = pyqtSignal(int)  # New signal for progress updates
    
    def __init__(self, cap, recognizer, required_frames=10, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.recognizer = recognizer  # Instance of your FaceRecognition class
        self.required_frames = required_frames
        self.person_name = None  # To be set from GUI

    def run(self):
        embeddings = []
        count = 0
        while count < self.required_frames:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            # Process the frame to detect a face using your recognizer method
            results = self.recognizer.real_time_face_recognition(frame)
            if len(results) == 0:
                continue

            # For simplicity, choose the first detected face
            x, y, w, h, _ = results[0]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_img = img_rgb[y:y+h, x:x+w]

            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                continue

            try:
                embedding = get_embedding(face_img)
                embeddings.append(embedding)
                count += 1
                # Emit progress as a percentage
                progress = int((count / self.required_frames) * 100)
                self.progress_updated.emit(progress)
                
            except Exception as e:
                print("Error in embedding generation:", e)
                continue

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            self.registration_done.emit(self.person_name, avg_embedding)

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Face Recognition System")
        self.setGeometry(100, 100, 800, 600)
        self.recognizer = FaceRecognition()

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        self.start_button = QPushButton("Start Camera")
        self.freeze_button = QPushButton("Freeze Camera")
        self.add_recognition_button = QPushButton("Add Recognition")
        self.delete_recognition_button = QPushButton("Delete Recognition")
        self.exit_button = QPushButton("Exit")

        hbox = QHBoxLayout()
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.freeze_button)
        hbox.addWidget(self.add_recognition_button)
        hbox.addWidget(self.delete_recognition_button)
        hbox.addWidget(self.exit_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Connect buttons
        self.start_button.clicked.connect(self.start_camera)
        self.freeze_button.clicked.connect(self.freeze_camera)
        self.add_recognition_button.clicked.connect(self.add_recognition)
        self.delete_recognition_button.clicked.connect(self.delete_recognition)
        self.exit_button.clicked.connect(self.close)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)  # 30ms = ~33fps

    def freeze_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()

    def pause_camera(self):
        self.timer.stop()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        results = self.recognizer.real_time_face_recognition(frame)

        for (x, y, w, h, identity) in results:
            color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Resize frame to fit the image_label (640x480) while preserving aspect ratio
        label_width, label_height = 640, 480
        frame_height, frame_width = frame.shape[:2]
        scale = min(label_width / frame_width, label_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Convert to RGB and display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.image_label.setPixmap(pix)

    def add_recognition(self):
        self.pause_camera()

        # Ask user for a name
        name, ok = QInputDialog.getText(self, "Register Face", "Enter name for the new face:")
        if not ok or not name.strip():
            self.start_camera()  # Resume camera if no name provided
            return

        # Create and show a dialog with a progress bar
        capture_dialog = QDialog(self)
        capture_dialog.setWindowTitle("Capturing Face")
        capture_dialog.setModal(True)
        label = QLabel(f"Capturing 50 images for '{name}'...\nPlease face the camera.")
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)  # Percentage
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(progress_bar)
        capture_dialog.setLayout(layout)
        capture_dialog.show()

        # Start the registration thread
        self.registration_thread = RecognitionThread(self.cap, self.recognizer, required_frames=50)
        self.registration_thread.person_name = name.strip()
        self.registration_thread.progress_updated.connect(progress_bar.setValue)  # Update progress bar
        self.registration_thread.registration_done.connect(lambda n, e: self.handle_registration_done(n, e, capture_dialog))
        self.registration_thread.start()
    
    def delete_recognition(self):
        name, ok = QInputDialog.getText(self, "Delete Face", "Enter name of the face to delete:")
        if not ok or not name.strip():
            return
        self.recognizer.del_from_database(name.strip())

        #reload the database
        self.recognizer.known_faces = self.recognizer.load_face_database()
        QMessageBox.information(self, "Delete Face", f"Face for '{name}' deleted successfully!")

    def handle_registration_done(self, name, avg_embedding, capture_dialog):
        capture_dialog.accept()  # Close the dialog
        self.recognizer.known_faces[name] = avg_embedding
        with open(self.recognizer.database_path, "wb") as f:
            pickle.dump(self.recognizer.known_faces, f)
        QMessageBox.information(self, "Add Face", f"Face for '{name}' added successfully!")
        self.start_camera()

    def closeEvent(self, event):
        self.freeze_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
