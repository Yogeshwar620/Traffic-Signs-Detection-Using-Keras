from data_loader import load_train_data, load_test_data
from preprocess import preprocess_data
from model import create_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# 1. Load datasets
print("Loading training data...")
X_train_full, y_train_full = load_train_data('data/Train')

print("Loading test data...")
X_test, y_test = load_test_data('data/Test', 'data/Test.csv')  # Path to Test.csv

# 2. Preprocessing
print("Preprocessing data...")
X_train_full, X_test, y_train_full, y_test = preprocess_data(X_train_full, X_test, y_train_full, y_test)

# 3. Split training data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=np.argmax(y_train_full, axis=1)
)

# 4. Compute class weights to address class imbalance
print("Computing class weights...")
y_train_labels = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights_dict = dict(enumerate(class_weights))

# 5. Set up data augmentation
print("Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)
datagen.fit(X_train)

# 6. Create and compile the model
print("Creating model...")
model = create_model()

# 7. Train the model with data augmentation and class weights
print("Training model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(X_val, y_val),
    class_weight=class_weights_dict,
    verbose=1
)

# 8. Evaluate on the test set
print("Evaluating model on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')

# 9. Per-class evaluation report
print("Classification report on test data:")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred_classes))

# 10. Save model
model.save('traffic_sign_model.h5')
print("Model saved!")

# 11. Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('training_history.png')
plt.show()
