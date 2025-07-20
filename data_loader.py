import os
import cv2
import numpy as np
import pandas as pd

def load_test_data(test_dir, csv_path):
    test_data = pd.read_csv(csv_path)
    images = []
    labels = []
    for idx, row in test_data.iterrows():
        # Adjust this line based on your CSV:
        # If Path column is 'Test/00000.png', use os.path.join('data', row['Path'])
        # If Path column is '00000.png', use os.path.join(test_dir, row['Path'])
        if row['Path'].startswith('Test/'):
            img_path = os.path.join('data', row['Path'])
        else:
            img_path = os.path.join(test_dir, row['Path'])

        if not os.path.exists(img_path):
            print(f"Warning: {img_path} does not exist. Skipping.")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}. Skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (30, 30))
        images.append(img)
        labels.append(row['ClassId'])
    return np.array(images), np.array(labels)
def load_train_data(data_dir):
    images = []
    labels = []
    # List all class folders (0 to 42)
    class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for class_folder in sorted(class_folders, key=int):
        class_path = os.path.join(data_dir, class_folder)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load {img_path}. Skipping.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (30, 30))
            images.append(img)
            labels.append(int(class_folder))
    return np.array(images), np.array(labels)

