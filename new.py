import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

model = EfficientNetB0(weights="imagenet")  

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the image for EfficientNetB0.
    """
    image = image.resize((224, 224))  
    image_array = np.array(image)
    if image_array.shape[-1] == 4: 
        image_array = image_array[..., :3]
    image_array = preprocess_input(image_array) 
    image_array = np.expand_dims(image_array, axis=0)  
    return image_array


st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f7;
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #FF6F61;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 30px;
    }
    .upload-area {
        text-align: center;
        margin: 20px auto;
        padding: 20px;
        border: 2px dashed #FF6F61;
        border-radius: 10px;
        background-color: #fff;
        color: #000;
        font-size: 1rem;
        font-weight: bold;
    }
    .image-container {
        text-align: center;
        margin-top: 20px;
    }
    .result-box {
        padding: 20px;
        margin: 20px auto;
        border-radius: 10px;
        background-color: #ffebe6;
        text-align: center;
        color: #333;
        max-width: 600px;
    }
    .result-box h2 {
        font-size: 1.5rem;
        color: #FF6F61;
    }
    .result-box p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main app title
st.markdown('<h1 class="main-title">AI Image Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image and let AI identify objects for you!</p>', unsafe_allow_html=True)

# Upload image section
st.markdown(
    '<div class="upload-area">Drag and drop or browse to upload an image (JPG, JPEG, PNG)</div>',
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Add a button to toggle image visibility
    if st.button("View Image"):
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Process image and predict
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    top_prediction = decoded_predictions[0]

    # Display prediction results
    st.markdown(
        f"""
        <div class="result-box">
            <h2>Prediction Results</h2>
            <p><b>Object:</b> {top_prediction[1].replace('_', ' ').capitalize()}</p>
            <p><b>Confidence:</b> {top_prediction[2]:.2%}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
