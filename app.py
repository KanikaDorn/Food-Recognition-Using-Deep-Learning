from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Define model path and load the model
MODEL_PATH = r"D:\Dev\Deep Learning\Food-Recognition-Using-Deep-Learning\models\food_recognition_model.h5"

# Verify that the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}")
else:
    print(f"Model file found at {MODEL_PATH}, size: {os.path.getsize(MODEL_PATH)} bytes")

# Attempt to load the model
try:
    model = load_model(MODEL_PATH)  # Load the model
    print("Model loaded successfully!")
    model.summary()  # Print the model summary for verification
except Exception as e:
    raise Exception(f"Error loading model: {e}")

# Set upload folder for images
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# List of food classes (ensure this matches the model's training classes)
CLASSES = ['Burger', 'Pizza', 'Fries']

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Ensure a file is provided
        file = request.files['file']
        if not file:
            return render_template("index.html", food_name="No file uploaded")

        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Read and preprocess the image
        img = cv2.imread(filepath)
        if img is None:
            return render_template("index.html", food_name="Invalid image file")

        # Resize the image to match the model input size (224x224 for many CNNs)
        img = cv2.resize(img, (224, 224))

        # Normalize the image by dividing by 255.0 (same as training preprocessing)
        img = img / 255.0

        # Convert the image to a batch of 1 sample (expand dimensions)
        img = np.expand_dims(img, axis=0)

        # Make prediction
        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions)  # Get the index of the class with the highest probability

        # Map the predicted index to the class label
        predicted_class = CLASSES[predicted_class_index]

        # Return the result to the user
        return render_template("index.html", food_name=predicted_class)

    return render_template("index.html", food_name=None)

if __name__ == "__main__":
    app.run(debug=True)
