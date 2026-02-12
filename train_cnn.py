# train_cnn.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_model import build_model

data = []
labels = []

for label, folder in enumerate(["closed", "open"]):
    path = os.path.join("dataset", folder)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64,64))

        data.append(image)
        labels.append(label)

data = np.array(data).reshape(-1,64,64,1) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

model = build_model()

model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_test, y_test))

model.save("eye_cnn.h5")

