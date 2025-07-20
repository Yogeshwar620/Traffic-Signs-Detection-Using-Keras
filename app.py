import streamlit as st

# Call set_page_config only once, at the very top
st.set_page_config(page_title="Traffic Sign Detection", layout="centered")

from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
@st.cache_resource
def load_keras_model():
    return load_model('traffic_sign_model.h5')

model = load_keras_model()

# Class ID to sign name mapping for GTSRB
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicles > 3.5 tons'
}

# Streamlit UI
st.title("🚦 Traffic Sign Detection")
st.write("Upload a traffic sign image and the model will predict the sign.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        img_resized = image.resize((30, 30))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        class_id = int(np.argmax(pred))
        sign_name = classes.get(class_id, "Unknown sign")
        confidence = float(np.max(pred))
        st.success(f"**Prediction:** {sign_name} (Class {class_id})")
        st.info(f"**Confidence:** {confidence:.2f}")
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image file to get started.")
