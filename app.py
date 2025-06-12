import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import joblib

# Custom CSS for styling the page
st.markdown("""
    <style>
        .main {
            font-family: "Arial", sans-serif;
            color: #333;
        }
        .header {
            text-align: center;
            font-size: 32px;
            color: #4B9CD3;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .result {
            text-align: center;
            font-size: 20px;
            color: #444;
            font-weight: bold;
        }
        .image-container {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)



st.markdown("<p class='header'>Upload an image of potato leaves to classify its condition</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    img_resized = image.resize((224, 224))
    
    img_array = np.array(img_resized)

    if img_array.ndim == 2:  
        img_array = np.expand_dims(img_array, axis=-1)  
        img_array = np.repeat(img_array, 3, axis=-1)   

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)  

    model = tf.keras.models.load_model("potato_6_class.keras")
    
    if st.button("Predict Category"):
        prediction = model.predict(img_array)
        label = np.argmax(prediction)

        leaf_category = joblib.load("class_name.joblib")
        
        leaf_category_name = leaf_category[label]
        
        st.markdown(f"<p class='result'>The predicted category for the uploaded image is: {leaf_category_name}</p>", unsafe_allow_html=True)
