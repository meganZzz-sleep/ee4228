import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib  # Import joblib to save models

# Define dataset path
dataset_path = r'C:\Users\yelhs\source\repos\EE4228clone\Assignment 1 test\Photos_Renamed'

# Define output path for saving models
output_path = r'C:\Users\yelhs\source\repos\EE4228clone\Assignment 1 test'
os.makedirs(output_path, exist_ok=True)  # Create the folder if it doesn't exist

# Initialize lists for images and labels
images = []
labels = []
label_encoder = LabelEncoder()

# Get all folder names (people's names)
folder_names = [folder_name for folder_name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder_name))]

# Fit the LabelEncoder on all folder names (so each person gets a unique label)
label_encoder.fit(folder_names)

# Load and preprocess images
for folder_name in folder_names:
    folder_path = os.path.join(dataset_path, folder_name)
    
    # Label each person using the folder name (group member's name)
    label_encoded = label_encoder.transform([folder_name])[0]
        
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize the image to 90x90 if it's not already
        img_resized = cv2.resize(img, (90, 90))
        
        # Flatten the image (convert to vector)
        images.append(img_resized.flatten())
        labels.append(label_encoded)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Print the unique labels
print("Unique labels in dataset:", np.unique(labels))

# Apply PCA to reduce dimensionality
pca = PCA(n_components=100)  # You can adjust the number of components as needed
pca_images = pca.fit_transform(images)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(pca_images, labels, test_size=0.2, random_state=42)

# Print unique labels in the training set
print("Unique labels in y_train:", np.unique(y_train))

# Train a classifier (e.g., Support Vector Machine)
svm = SVC(kernel='linear')

# Check if we have more than one class
if len(np.unique(y_train)) > 1:
    svm.fit(X_train, y_train)
    # Evaluate the classifier
    accuracy = svm.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Save the PCA model, SVM model, and LabelEncoder to the desired output path
    joblib.dump(pca, os.path.join(output_path, 'pca_model.pkl'))
    print("PCA model saved as 'pca_model.pkl'")
    
    joblib.dump(svm, os.path.join(output_path, 'svm_model.pkl'))
    print("SVM model saved as 'svm_model.pkl'")
    
    joblib.dump(label_encoder, os.path.join(output_path, 'label_encoder.pkl'))
    print("Label encoder saved as 'label_encoder.pkl'")
else:
    print("Error: There should be more than one class in y_train.")