import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model (update the filename if needed)
model = load_model('traffic_sign_model.h5')

# Class ID to sign name mapping for GTSRB (0-based)
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

# --- GUI Functions ---

def classify(file_path):
    try:
        image = Image.open(file_path).convert('RGB')
        image = image.resize((30, 30))
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)
        preds = model.predict(image_array)
        class_id = int(np.argmax(preds))
        sign_name = classes.get(class_id, "Unknown sign")
        confidence = float(np.max(preds))
        result_label.config(
            text=f"Prediction: {sign_name}\n(Class {class_id}, Confidence: {confidence:.2f})",
            fg="green"
        )
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}", fg="red")

def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        result_label.config(text="No file selected.", fg="red")
        return
    try:
        img = Image.open(file_path)
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        image_panel.config(image=img_tk)
        image_panel.image = img_tk
        result_label.config(text="")  # Clear previous result
        classify(file_path)
    except Exception as e:
        result_label.config(text=f"Error loading image: {str(e)}", fg="red")

# --- GUI Layout ---

root = tk.Tk()
root.title("Traffic Sign Detection")
root.geometry("450x500")
root.configure(bg='#f0f0f0')

heading = tk.Label(root, text="Traffic Sign Detection", pady=20, font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#364156')
heading.pack()

upload_btn = tk.Button(
    root, text="Upload Image", command=upload_image,
    bg='#364156', fg='white', font=('Arial', 12, 'bold'), padx=10, pady=5
)
upload_btn.pack(pady=10)

image_panel = tk.Label(root, bg='#f0f0f0')
image_panel.pack(pady=10)

result_label = tk.Label(root, text="", font=('Arial', 14), bg='#f0f0f0')
result_label.pack(pady=20)

root.mainloop()
