import streamlit as st
from PIL import Image
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_image

st.title("🍎 Fruit Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        result = predict_image(image)
        st.success(f"Prediction: {result}")