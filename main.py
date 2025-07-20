import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import os

# Data preparation
def load_data(data_dir):
    images = []
    labels = []
    for label in range(43):
        path = os.path.join(data_dir, str(label))
        for img_file in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (30, 30))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset
train_path = "Train"
test_path = "Test"
X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)

# Preprocessing
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Model architecture
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(30,30,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

# Compile and train
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
history = model.fit(X_train, y_train, 
                    epochs=15, 
                    validation_split=0.2, 
                    batch_size=64)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Save model
model.save('traffic_sign_recognition.h5')
