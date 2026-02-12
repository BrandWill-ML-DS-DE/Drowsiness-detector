# evaluate.py
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

model = load_model("eye_cnn.h5")

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

predictions = model.predict(data)
predictions = (predictions > 0.5).astype(int)

print(classification_report(labels, predictions))

