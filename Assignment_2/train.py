import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
import os
import pickle
from tqdm import tqdm  # Progress bar

# Load MTCNN for face detection
detector = MTCNN()

# Load FaceNet for face recognition
embedder = FaceNet()

dataset_path = "Latest_processed"

def create_face_database():
    known_faces = {}

    # Use tqdm for progress bars
    for person_name in tqdm(os.listdir(dataset_path), desc="Processing people"):
        person_folder = os.path.join(dataset_path, person_name)
        embeddings = []

        for img_name in tqdm(os.listdir(person_folder), desc=f"Processing {person_name} images"):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces = detector.detect_faces(img_rgb)

            if faces:
                x, y, w, h = faces[0]['box']
                face = img_rgb[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))

                embedding = embedder.embeddings([face])[0]
                embeddings.append(embedding)

        if embeddings:
            known_faces[person_name] = np.mean(embeddings, axis=0)

    with open("face_database.pkl", "wb") as f:
        pickle.dump(known_faces, f)

    print("Face database created and saved!")

if __name__ == "__main__":
    create_face_database()
