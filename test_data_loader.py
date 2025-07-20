from data_loader import load_train_data, load_test_data

# Load training data
X_train, y_train = load_train_data('data/Train')
print(f"Loaded {len(X_train)} training images and {len(y_train)} labels.")

# Load test data
X_test, y_test = load_test_data('data/Test', 'data/Test.csv')
print(f"Loaded {len(X_test)} test images and {len(y_test)} labels.")
