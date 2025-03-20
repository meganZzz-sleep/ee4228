import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
import pickle
from tqdm import tqdm
from albumentations import Compose, Rotate, HorizontalFlip, RandomBrightnessContrast, GaussNoise, Blur
from sklearn.cluster import DBSCAN  # Add this import

# Load MTCNN for face detection
detector = MTCNN()
# Load FaceNet for face recognition
embedder = FaceNet()
dataset_path = "Latest_processed"

augment = Compose([
    Rotate(limit=15),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    GaussNoise(p=0.3),
    Blur(blur_limit=3, p=0.3),
])

def augment_image(image):
    augmented = augment(image=image)
    return augmented['image']

def create_face_database():
    known_faces = {}
    # Filter only directories in the dataset path
    for person_name in tqdm([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))], desc="Processing people"):
        person_folder = os.path.join(dataset_path, person_name)
        embeddings = []
        
        # Filter only valid image files (e.g., .jpg, .jpeg, .png), skip hidden files
        valid_extensions = ('.jpg', '.jpeg', '.png')
        for img_name in tqdm([f for f in os.listdir(person_folder) if f.lower().endswith(valid_extensions) and not f.startswith('.')], desc=f"Processing {person_name} images"):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Failed to load image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            
            for face in faces:
                if face['confidence'] > 0.85:
                    x, y, w, h = face['box']
                    # Crop the face from the image
                    face_img = img_rgb[y:y+h, x:x+w]
                    # Apply augmentation to the cropped face
                    augmented_face = augment_image(face_img)
                    # Resize the augmented face to 160x160 for FaceNet
                    face_resized = cv2.resize(augmented_face, (160, 160))
                    # Generate embedding
                    embedding = embedder.embeddings([face_resized])[0]
                    embeddings.append(embedding)
        
        if embeddings:
            # Apply DBSCAN clustering to remove outliers
            embeddings_array = np.array(embeddings)  # Convert to numpy array for clustering
            clustering = DBSCAN(eps=0.5, min_samples=4).fit(embeddings_array)
            valid_indices = clustering.labels_ != -1  # -1 indicates noise/outliers
            if np.any(valid_indices):
                # Average only the embeddings in valid clusters
                known_faces[person_name] = np.mean(embeddings_array[valid_indices], axis=0)
                print(f"✅ {person_name}: {np.sum(valid_indices)} valid embeddings out of {len(embeddings)}")
            else:
                # Fallback: Use all embeddings if no valid clusters are found
                known_faces[person_name] = np.mean(embeddings_array, axis=0)
                print(f"⚠️ {person_name}: No valid clusters found, using all {len(embeddings)} embeddings")
        else:
            print(f"No embeddings generated for {person_name}")
    
    with open("face_database.pkl", "wb") as f:
        pickle.dump(known_faces, f)
    print("✅ Face database created and saved!")
    return known_faces

if __name__ == "__main__":
    known_faces = create_face_database()
