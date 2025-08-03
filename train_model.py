import os
import numpy as np
import cv2
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
# Initialize keras-facenet embedder
embedder = FaceNet()

# Path to your dataset
dataset_path = 'D:/FaceAttendence/augmented_data'  # format: dataset/person_name/image.jpg

# X = embeddings, y = labels
X, y = [], []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        results = embedder.extract(img)
        if results:
            emb = results[0]['embedding']
            X.append(emb)
            y.append(person)

# Convert to numpy arrays
X = np.asarray(X)
y = np.asarray(y)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X, y_encoded)

# Save to embeddings folder
os.makedirs("embeddings", exist_ok=True)
with open("embeddings/face_classifier.pkl", "wb") as f:
    pickle.dump((encoder, classifier), f)

print("Face embeddings (512-d) extracted and classifier saved.")