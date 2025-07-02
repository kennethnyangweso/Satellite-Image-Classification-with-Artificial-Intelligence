import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image  # only for preprocessing

# Load ONNX model
session = ort.InferenceSession("satellite_model.onnx")

# Class labels
class_names = ['cloudy', 'desert', 'green_area', 'water']

# Streamlit page
st.set_page_config(page_title="Satellite Classifier", layout="centered")
st.title("🌍 Satellite Image Classifier")
st.write("Upload a satellite image to classify the land type.")

# Upload
uploaded_file = st.file_uploader("Choose a .jpg/.png image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)

    # Inference
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: arr})[0]

    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Show result
    st.success(f"Prediction: `{predicted_class}`")
    st.info(f"Confidence: `{confidence:.2f}%`")
