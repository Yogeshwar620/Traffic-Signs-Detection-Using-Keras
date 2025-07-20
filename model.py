from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model():
    model = Sequential([
        Input(shape=(30, 30, 3)),  # Explicit input layer
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        
        Conv2D(128, (3,3), activation='relu'),
        
        Flatten(),
        
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(43, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
