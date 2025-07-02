import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model (update path if needed)
model = tf.keras.models.load_model("satellite_mobilenetv2_base_best.h5")

# Define class labels
class_names = ['cloudy', 'desert', 'green_area', 'water']

# Page layout
st.set_page_config(page_title="Satellite Classifier", layout="centered")
st.title("🌍 Satellite Image Classifier")
st.write("Upload a satellite image to classify the land type.")

# Upload image
uploaded_file = st.file_uploader("Choose a .jpg/.png image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image_pil.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
