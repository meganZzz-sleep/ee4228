import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

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

    def real_time_face_recognition(self, frame):
        # # known_faces = load_face_database()
        # print("Opening webcam...")
        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # if not cap.isOpened():
        #     print("ERROR: Could not open webcam! Check camera permissions.")
        #     exit()

        # frame_count = 0
        # last_face = None
        # last_embedding = None

        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         print("ERROR: Failed to read frame!")
        #         break

        #     frame = cv2.resize(frame, (640, 480))  # Reduce frame size
        #     frame_count += 1

        #     # Process every 3rd frame for better speed
        #     if frame_count % 3 != 0:
        #         continue

        #     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     faces = self.detector.detect_faces(img_rgb)

        #     print(f"Detected {len(faces)} faces.")

        # new 
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img_rgb)
        results = []
        last_face = None
        last_embedding = None

        for face in faces:
            x, y, w, h = face['box']
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

            #     color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            #     cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            #     cv2.putText(frame, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # cv2.imshow("Face Recognition", frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # cap.release()
        # cv2.destroyAllWindows()

# if __name__ == "__main__":
#     print("Running Face Recognition...")
#     fr = FaceRecognition()
#     fr.real_time_face_recognition()
