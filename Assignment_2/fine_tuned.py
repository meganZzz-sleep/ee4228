import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# Enable OpenCV multi-threading
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# Load models
print("✅ Loading MTCNN for face detection...")
detector = MTCNN()
print("✅ Loading FaceNet for face recognition...")
embedder = FaceNet()

# Load face database
def load_face_database():
    print("📂 Loading face database...")
    try:
        with open("face_database.pkl", "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded {len(data)} known faces!")
        return data
    except FileNotFoundError:
        print("❌ ERROR: face_database.pkl not found! Run `python face_database.py` first.")
        exit()

def real_time_face_recognition():
    known_faces = load_face_database()
    print("🎥 Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam! Check camera permissions.")
        exit()

    frame_count = 0
    last_face = None
    last_embedding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read frame!")
            break

        frame = cv2.resize(frame, (640, 480))  # Reduce frame size
        frame_count += 1

        # Process every 3rd frame for better speed
        if frame_count % 3 != 0:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        print(f"🔍 Detected {len(faces)} faces.")

        for face in faces:
            x, y, w, h = face['box']
            face_img = img_rgb[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))

            # Use cached embedding if the same face appears
            if last_face is not None and np.array_equal(last_face, face_img):
                embedding = last_embedding
            else:
                embedding = embedder.embeddings([face_img])[0]
                last_face = face_img.copy()
                last_embedding = embedding

            min_dist = float("inf")
            identity = "Unknown"

            for name, db_embedding in known_faces.items():
                dist = cosine(db_embedding, embedding)
                if dist < 0.4:
                    min_dist = dist
                    identity = name

            color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🚀 Running Face Recognition...")
    real_time_face_recognition()
