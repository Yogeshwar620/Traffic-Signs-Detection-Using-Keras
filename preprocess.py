from tensorflow.keras.utils import to_categorical

def preprocess_data(X_train, X_test, y_train, y_test):
    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    
    return X_train, X_test, y_train, y_test
