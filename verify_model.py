import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = tf.keras.models.load_model("street_issue_classifier.h5")

# Class indices
class_indices = {v: k for k, v in model.class_names.items()} if hasattr(model, 'class_names') else None

# Define class labels (optional fallback if `model.class_names` doesn't exist)
# You can replace this with the actual folder names in your "dataset" folder
class_labels = ['Flooding', 'Road_damage', 'Street_light','Garbage']  # example

# Load and preprocess image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # make batch of 1
    img_array /= 255.0  # same rescaling as training
    return img_array

# Prediction function
def classify_image(img_path):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    label = class_labels[predicted_class] if class_indices is None else class_indices[predicted_class]
    
    print(f"Predicted Class: {label} (Confidence: {confidence:.2f})")

# Example usage
img_path = "D:\garbage\images.jpeg"# <-- change this path to your test image
classify_image(img_path)
