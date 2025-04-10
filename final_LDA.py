# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 02:21:46 2025

@author: ranen
"""

import numpy as np
import os
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Path to image dataset - use your own
dataset_path = "C:/Users/ranen/OneDrive - Nanyang Technological University/01 EE4228 INT SYS DES/Dataset"  # Change this to your folder path

# Image processing parameters
img_size = (90, 90)  # Resize all images to 64x64
num_classes = len(os.listdir(dataset_path))

# Initialize empty lists
X_train = []
X_test = []
y_train = []
y_test = []

# Assign labels
label_dict = {}

for label, class_name in enumerate(sorted(os.listdir(dataset_path))):
    class_path = os.path.join(dataset_path, class_name)
    
    if os.path.isdir(class_path):
        label_dict[class_name] = label
        images = []
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img.flatten() / 255.0)  # Normalize and flatten
        
        # Convert to numpy array
        images = np.array(images)
        labels = np.full(len(images), label)  # Create label array
        
        # Split within this class
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            images, labels, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Append to overall list
        X_train.append(X_train_class)
        X_test.append(X_test_class)
        y_train.append(y_train_class)
        y_test.append(y_test_class)

# Stack all classes together
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)
y_train = np.hstack(y_train)
y_test = np.hstack(y_test)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

#%% 3

# Compute class means
class_means = {}
for label in np.unique(y_train):
    class_means[label] = np.mean(X_train[y_train == label], axis=0)

#%% 4

# Compute within-class scatter matrix Sw
S_w = np.zeros((X_train.shape[1], X_train.shape[1]))
for i, img in enumerate(X_train):
    diff = (img - class_means[y_train[i]]).reshape(-1, 1)
    S_w += diff @ diff.T

print("Done 4")

#%% 5
# Compute overall mean
overall_mean = np.mean(X_train, axis=0)

print("Done 5")

#%% 6
# Compute between-class scatter matrix Sb
S_b = np.zeros((X_train.shape[1], X_train.shape[1]))
for label in np.unique(y_train):
    N = np.sum(y_train == label)  # Number of samples in class
    mean_diff = (class_means[label] - overall_mean).reshape(-1, 1)
    S_b += N * (mean_diff @ mean_diff.T)

print("Done 6")

#%% 7
# Solve for the eigenvectors and eigenvalues
eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(S_w).dot(S_b))
print("Done 7")

#%% 8
# Sort eigenvectors by eigenvalues in descending order
sorted_indices = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, sorted_indices]

# Choose top components (num_classes - 1)
num_components = num_classes - 1
lda_components = eigvecs[:, :num_components]

print("Done 8")

#%% 9

# Project data onto LDA components
X_train_lda = X_train.dot(lda_components)
X_test_lda = X_test.dot(lda_components)

print("Done 9")

#%% 10

# Train k-NN classifier on LDA features
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_lda, y_train)

# Evaluate the model
accuracy = knn.score(X_test_lda, y_test)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

print("Done 10")

#%% 11

# Visualize LDA components as images
fig, axes = plt.subplots(1, min(5, num_components), figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(lda_components[:, i].reshape(img_size), cmap='gray')
    ax.set_title(f"LDA Component {i+1}")
    ax.axis("off")
plt.show()

print("Done 11")
#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = knn.predict(X_test_lda)  # Predict the labels for test data
accuracy = np.mean(y_pred == y_test)  # Same as knn.score but manual

print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Optional: Display the confusion matrix as a plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_dict.keys())
disp.plot(cmap=plt.cm.Blues)
ax.tick_params(axis='both', labelsize=4)  # <-- This controls the tick font size
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.title("Confusion Matrix")
plt.show()

print("Done 12")
