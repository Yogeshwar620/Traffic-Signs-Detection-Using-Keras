from data_loader import load_train_data, load_test_data
from preprocess import preprocess_data
import numpy as np

# Load the data first!
X_train, y_train = load_train_data('data/Train')
X_test, y_test = load_test_data('data/Test', 'data/Test.csv')

# Now preprocess
X_train_norm, X_test_norm, y_train_cat, y_test_cat = preprocess_data(X_train, X_test, y_train, y_test)

print(f"X_train normalized shape: {X_train_norm.shape}, dtype: {X_train_norm.dtype}, min: {np.min(X_train_norm)}, max: {np.max(X_train_norm)}")
print(f"y_train one-hot shape: {y_train_cat.shape}")
