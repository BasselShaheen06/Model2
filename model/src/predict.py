import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PyQt5.QtWidgets import QLabel
from model import load_model

def predict_image(model, img_path, img_size=(224, 224)):
    # Load and preprocess the image
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Get prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)

    return predicted_class, confidence

def display_prediction(img_path, model, label_widget: QLabel):
    predicted_class, confidence = predict_image(model, img_path)
    label_widget.setText(f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}")
