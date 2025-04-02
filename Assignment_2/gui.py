import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer, QThread
from PyQt5.QtGui import QImage, QPixmap
import cv2
from recognition_gui import *

class RecognitionThread(QThread):
    def __init__(self):
        super().__init__()
        self.fr = FaceRecognition()

    def run(self):
        self.fr.real_time_face_recognition()

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Face Recognition System")
        self.setGeometry(100, 100, 800, 600)
        self.recognizer = FaceRecognition()

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        # self.recognition_button = QPushButton("Run Real-Time Recognition")
        self.exit_button = QPushButton("Exit")

        hbox = QHBoxLayout()
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.stop_button)
        # hbox.addWidget(self.recognition_button)
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
        self.stop_button.clicked.connect(self.stop_camera)
        # self.recognition_button.clicked.connect(self.run_recognition)
        self.exit_button.clicked.connect(self.close)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)  # 30ms = ~33fps

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()

    def run_recognition(self):
        self.stop_camera()
        self.recognition_thread = RecognitionThread()
        self.recognition_thread.start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        results = self.recognizer.real_time_face_recognition(frame)

        for (x, y, w, h, identity) in results:
            color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.image_label.setPixmap(pix)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())