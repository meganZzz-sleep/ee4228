#This code consists of saving the augmented and face alignment images 

import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
import pickle
from tqdm import tqdm
from albumentations import Compose, Rotate, HorizontalFlip, RandomBrightnessContrast, GaussNoise, Blur, Resize, ElasticTransform
from scipy.spatial.distance import cosine

# Load MTCNN for face detection
detector = MTCNN()
# Load FaceNet for face recognition
embedder = FaceNet()
dataset_path = "photos"
output_path = "face_database.pkl"
preprocessed_dir = "pre-processed"  # Directory to save preprocessed images

# Augmentation pipeline (minor adjustments)
augment = Compose([
    Rotate(limit=15, p=0.6),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    GaussNoise(var_limit=(5.0, 30.0), p=0.3),
    Blur(blur_limit=3, p=0.2),
    ElasticTransform(alpha=1, sigma=45, alpha_affine=45, p=0.15),
    Resize(160, 160, interpolation=cv2.INTER_AREA, p=1)  # Keep final size small
])

def augment_image(image):
    augmented = augment(image=image)
    return augmented['image']

def preprocess_image(image, img_name, person_name):
    """Convert to grayscale, crop, and apply adaptive equalization, and save the preprocessed image."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        crop_size = min(h, w)
        y = (h - crop_size) // 2
        x = (w - crop_size) // 2
        cropped_gray = gray[y:y+crop_size, x:x+crop_size]
        resized_gray = cv2.resize(cropped_gray, (480, 480), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_gray = clahe.apply(resized_gray)
        preprocessed_image = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2RGB)  # Back to RGB

        # Save preprocessed image
        save_dir = os.path.join(preprocessed_dir, person_name)
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, preprocessed_image)

        return preprocessed_image
    except cv2.error as e:
        print(f"⚠️ Preprocess error: {e}, using original image")
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #returning gray scale

def align_face(image, keypoints):
    """Aligns the face based on eye landmarks."""
    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    if left_eye and right_eye:
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        return aligned_face
    else:
        print("⚠️ Eye keypoints missing, skipping alignment")
        return image

def robust_face_detection(img_rgb):
    """Cascade face detection with size filtering."""
    faces = detector.detect_faces(img_rgb)
    valid_faces = [f for f in faces if f['box'][2] > 60 and f['box'][3] > 60]
    return valid_faces

def create_face_database():
    known_faces = {}
    valid_extensions = ('.jpg', '.jpeg', '.png')
    face_counts = {}  # To store the number of faces detected per person
    total_images_processed = 0
    total_faces_detected = 0

    # Directory to save aligned faces
    aligned_faces_dir = "aligned_faces"
    os.makedirs(aligned_faces_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Create the preprocessed directory if it doesn't exist
    os.makedirs(preprocessed_dir, exist_ok=True)

    for person_name in tqdm([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))], desc="Processing people"):
        person_folder = os.path.join(dataset_path, person_name)
        embeddings = []
        person_face_count = 0  # Count faces for the current person

        for img_name in tqdm([f for f in os.listdir(person_folder) if f.lower().endswith(valid_extensions) and not f.startswith('.')], desc=f"Processing {person_name} images"):
            img_path = os.path.join(person_folder, img_name)
            total_images_processed += 1
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Failed to load image: {img_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb = preprocess_image(img_rgb, img_name, person_name)  # Apply preprocessing and save

                faces = robust_face_detection(img_rgb)

                for i, face in enumerate(faces):
                    if face['confidence'] > 0.8:
                        x, y, w, h = face['box']
                        keypoints = face.get("keypoints", {})

                        face_img = img_rgb[y:y+h, x:x+w]
                        face_img = align_face(face_img, keypoints)  # Align the face

                        # Save the aligned face
                        align_save_dir = os.path.join(aligned_faces_dir, person_name)
                        os.makedirs(align_save_dir, exist_ok=True)  # Create person-specific subfolder
                        align_filename = f"{os.path.splitext(img_name)[0]}_aligned_face_{i}.jpg"
                        align_save_path = os.path.join(align_save_dir, align_filename)
                        cv2.imwrite(align_save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                        print(f"✅ Saved aligned face: {align_save_path}")

                        # Continue with augmentation and further processing
                        face_img = augment_image(face_img)
                        face_img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)
                        embedding = embedder.embeddings([face_img])[0]
                        embeddings.append(embedding)
                        person_face_count += 1
                        total_faces_detected += 1
                    else:
                        print(f"⚠️ Low confidence face in {img_path}, skipping.")

            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")

        if embeddings:
            known_faces[person_name] = np.median(embeddings, axis=0)
            face_counts[person_name] = person_face_count
            print(f"✅ {person_name}: {len(embeddings)} embeddings processed, {person_face_count} faces detected.")
        else:
            print(f"⚠️ No embeddings generated for {person_name}")

    with open(output_path, "wb") as f:
        pickle.dump(known_faces, f)

    print("✅ Face database created and saved!")

    # Print Statistics
    print("\n----- Database Creation Statistics -----")
    print(f"Total Images Processed: {total_images_processed}")
    print(f"Total Faces Detected: {total_faces_detected}")
    print("\nFaces Detected per Person:")
    for person, count in face_counts.items():
        print(f"  - {person}: {count} faces")

    return known_faces

if __name__ == "__main__":
    known_faces = create_face_database()
