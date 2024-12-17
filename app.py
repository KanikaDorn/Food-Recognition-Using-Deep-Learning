from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Define model path and load the model
MODEL_PATH = r'D:\Food Recognition Using Deep Learning\models\food_recognition_model.h5'

# Verify that the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")
else:
    print(f"Model file found at {MODEL_PATH}, size: {os.path.getsize(MODEL_PATH)} bytes")

try:
    model = load_model(MODEL_PATH)  # Load the model
    print("Model loaded successfully!")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

# Set upload folder for images
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# List of food classes (ensure this matches the model's training classes)
CLASSES = ['Pizza', 'Pasta', 'Paella', 'Schnitzel', 'Croissant', 
           'Waffles', 'Ratatouille', 'Baguette', 'Bratwurst', 'Goulash']

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Save the uploaded file
        file = request.files['file']
        if not file:
            return render_template("index.html", food_name="No file uploaded")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess the image
        img = cv2.imread(filepath)
        if img is None:
            return render_template("index.html", food_name="Invalid image file")
        
        img = cv2.resize(img, (224, 224))  # Resize to model input size
        img = img / 255.0                 # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        food_name = CLASSES[class_index]

        return render_template("index.html", food_name=food_name, image_path=filepath)

    return render_template("index.html", food_name=None)

if __name__ == "__main__":
    app.run(debug=True)
