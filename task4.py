# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

dataset, info = tfds.load("cats_vs_dogs", split="train", with_info=True, as_supervised=True)

IMG_SIZE = 64 # small for SVM

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0
    return image, label

dataset = dataset.map(preprocess)

X = []
y = []

for img, label in dataset.take(5000):
    X.append(img.numpy().reshape(-1))
    y.append(label.numpy())

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Dog', 'Cat'],
            yticklabels=['Dog', 'Cat'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM Dog vs Cat")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["Dog", "Cat"]))

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


dataset = tfds.load("cats_vs_dogs", split="train", as_supervised=True)


plt.figure(figsize=(10, 10))

for i, (image, label) in enumerate(dataset.take(9)):
    plt.subplot(3, 3, i+1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Dog" if label.numpy() == 1 else "Cat")

plt.show()