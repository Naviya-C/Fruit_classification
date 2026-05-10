import streamlit as st
from PIL import Image
import requests

API = "http://127.0.0.1:8000/predict"

st.title("Fruit classification")

input_image = st.file_uploader("Upload the image: ", type = ['jpg', 'jpeg', 'png'])


if input_image is not None:
    
    image = Image.open(input_image)
    st.image(image, caption = "Image Uploaded")
    
    input_image.seek(0)
    
    response = requests.post(API, files = {"userImage":input_image})
    
    if response.status_code == 200:
        prediction = response.json()
        st.success(f"prediction: {prediction['prediction']}")
        
    else:
        st.error(response.text)