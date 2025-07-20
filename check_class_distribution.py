import os

train_dir = 'data/Train'
counts = {}
for class_id in range(43):
    class_path = os.path.join(train_dir, str(class_id))
    counts[class_id] = len(os.listdir(class_path))
print("Class distribution in training set:")
for k, v in counts.items():
    print(f"Class {k}: {v} images")
