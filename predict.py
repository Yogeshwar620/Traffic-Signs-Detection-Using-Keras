import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model('traffic_sign_model.h5')

def predict_sign(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return None, None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}.")
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_id, confidence

# User input
image_path = input("Enter the image path: ")
class_id, confidence = predict_sign(image_path)
if class_id is not None:
    print(f'Predicted class: {class_id} | Confidence: {confidence:.2f}')
