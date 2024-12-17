from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model

# Ensure your model path is correct
MODEL_PATH = r"D:\Dev\Deep Learning\Food-Recognition-Using-Deep-Learning\models\food_recognition_model.h5"
model = load_model(MODEL_PATH)

# Path to an image to test
img_path = r'D:\Dev\Deep Learning\Food-Recognition-Using-Deep-Learning\data\train\fries\fries_01.jpg'  # Corrected path

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))  # Resize the image to the input shape
img = image.img_to_array(img)  # Convert image to array
img = img / 255.0  # Normalize the image (same as in your training code)
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Make prediction
predictions = model.predict(img)
predicted_class_index = np.argmax(predictions)  # Get index of the highest probability

# Assuming the 'train_generator' object is still available
train_generator = None  # This should be the object from your training script
# Map the predicted class index to the class label (ensure class order is correct)
class_names = ['burger', 'fries', 'pizza']  # Use the exact class names used in your model
predicted_class = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class}")
