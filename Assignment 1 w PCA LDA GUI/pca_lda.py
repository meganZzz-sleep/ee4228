# face_recognition_pca.py

import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

IMAGE_SIZE = (90, 90) #can change this 
THRESHOLD = 2000  #can change this parameter, distance threshold for recognition
DISTANCE_LOG = [] # log the distances for plotting of result
LABEL_LOG = []    # log the labels for plotting of result

# --- Training Functions ---

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
            X.append(img.flatten()/255.0)
            y.append(label_map[person_name])
    return np.array(X, dtype='float32'), np.array(y), labels

def pca(X, num_components):
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    cov_matrix = np.dot(X_centered, X_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvecs = np.dot(X_centered.T, eigvecs)
    eigvecs = eigvecs[:, ::-1]  # descending order
    eigvecs = eigvecs[:, :num_components]
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

    # Explained variance
    explained_variance = np.var(np.dot(X_centered, eigvecs), axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)

    return mean_face, eigvecs, X_centered, explained_variance_ratio

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

def show_mean_face(mean_face):
    mean_img = mean_face.reshape(IMAGE_SIZE)
    plt.imshow(mean_img, cmap='gray')
    plt.title("Mean Face")
    plt.axis('off')
    plt.show()

def show_top_eigenfaces(eigenfaces, num):
    rows = 2
    cols = (num+1)//2
    plt.figure(figsize=(2.5*cols, 5))
    for i in range(num):
        plt.subplot(rows, cols, i+1)
        eig_img = eigenfaces[:, i].reshape(IMAGE_SIZE)
        plt.imshow(eig_img, cmap='gray')
        plt.title(f'PC {i+1}')
        plt.axis('off')
    plt.suptitle("Top Eigenfaces")
    plt.tight_layout()
    plt.show()

def plot_pca_class_distribution(X_pca, y, label_names):
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(
            X_pca[y == label, 0],
            X_pca[y == label, 1],
            label=label_names[label],
            alpha=0.6
        )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Class Distribution in PCA and LDA Space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_pca_variance(evr):
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(evr), marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_model():
    X, y, labels = load_dataset("dataset")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) #can change test_size

    # PCA on training data
    num_pca_components = 120                           
    mean_face, eigenfaces, X_train_centered, evr = pca(X_train, num_components=num_pca_components)
    X_train_pca = project(X_train, mean_face, eigenfaces)

    # LDA on training data
    lda_transform = lda(X_train_pca, y_train)
    num_lda_components = len(np.unique(y_train)) - 1
    lda_transform = lda_transform[:, :num_lda_components]
    X_train_lda = np.dot(X_train_pca, lda_transform)

    # Project test data
    X_test_pca = project(X_test, mean_face, eigenfaces)
    X_test_lda = np.dot(X_test_pca, lda_transform)

    # Predict test set
    y_pred = []
    for test_vec in X_test_lda:
        distances = np.linalg.norm(X_train_lda - test_vec, axis=1)
        best_match_idx = np.argmin(distances)
        pred_label = y_train[best_match_idx]
        y_pred.append(pred_label)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[labels[i] for i in np.unique(y)])
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.show()

    # Save model
    with open("pca_lda_model.pkl", "wb") as f:
        pickle.dump({
            'mean': mean_face,
            'eigenfaces': eigenfaces,
            'lda': lda_transform,
            'projections': X_train_lda,
            'labels': y_train,
            'label_names': labels
        }, f)

    print("\n--- Training Summary ---")
    print(f"Total people (classes): {len(labels)}")
    print(f"Total images: {len(X)}")
    for i, name in enumerate(labels):
        print(f"  - {name}: {(y == i).sum()} images")
    print(f"Image vector size: {X.shape[1]}")
    print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
    print(f"PCA components used: {num_pca_components}")
    print(f"LDA components used: {num_lda_components}")
    print(f"Validation Accuracy on Test Set: {acc * 100:.2f}%")
    print("Model trained and saved as pca_lda_model.pkl\n")

    show_mean_face(mean_face)
    show_top_eigenfaces(eigenfaces, num=10)
    plot_pca_variance(evr)
    plot_pca_class_distribution(X_train_lda, y_train, labels)


# --- Live Recognition Functions ---

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return [], []
    face_vecs = []
    face_rects = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, IMAGE_SIZE)
        face_vecs.append(face_resized.flatten())
        face_rects.append((x, y, w, h))
    return face_vecs, face_rects

def recognize(face_vec, mean, eigenfaces, lda_transform, projections, labels, label_names):
    face_pca = np.dot(face_vec - mean, eigenfaces)
    face_lda = np.dot(face_pca, lda_transform)
    distances = np.linalg.norm(projections - face_lda, axis=1)
    sorted_indices = np.argsort(distances)
    min_dist = distances[sorted_indices[0]]
    second_dist = distances[sorted_indices[1]]
    best_match = label_names[labels[sorted_indices[0]]]
    second_best = label_names[labels[sorted_indices[1]]]

    if min_dist < THRESHOLD:
        name = best_match
    else:
        name = "Unknown"

    print(f"Detected: {name}, Distance: {min_dist:.2f}, 2nd Best: {second_best}, Dist: {second_dist:.2f}")
    DISTANCE_LOG.append(min_dist)
    LABEL_LOG.append(name)
    return name, min_dist

def plot_distance_distribution():
    known_dists = [d for d, l in zip(DISTANCE_LOG, LABEL_LOG) if l != "Unknown"]
    unknown_dists = [d for d, l in zip(DISTANCE_LOG, LABEL_LOG) if l == "Unknown"]

    plt.hist(known_dists, bins=20, alpha=0.7, label='Known')
    plt.hist(unknown_dists, bins=20, alpha=0.7, label='Unknown')
    plt.axvline(x=THRESHOLD, color='r', linestyle='--', label=f'Threshold = {THRESHOLD}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution of Recognized Faces')
    plt.legend()
    plt.show()

def live_recognition():
    with open("pca_lda_model.pkl", "rb") as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_vecs, face_rects = preprocess(frame)
        for face_vec, face_rect in zip(face_vecs, face_rects):
            name, dist = recognize(face_vec, model['mean'], model['eigenfaces'], model['lda'],
                                   model['projections'], model['labels'], model['label_names'])
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    plot_distance_distribution()

# --- Entrypoint ---
if __name__ == "__main__":
    mode = input("Enter 'train' to train the model or 'test' to start camera: ").strip().lower()
    if mode == 'train':
        train_model()
    elif mode == 'test':
        live_recognition()
    else:
        print("Invalid option.")
