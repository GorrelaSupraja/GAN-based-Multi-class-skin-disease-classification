import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import base64

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the pre-trained model
def load_pretrained_model():
    return load_model('dense.h5')

model = load_pretrained_model()

# Preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((100, 100))  # Resize the image to match the input size of the model
    img = img.convert('RGB')  # Convert image to RGB format
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the class name and accuracy
def predict_class_and_accuracy(image):
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    classes = ['Basal cell carcinoma', 'Melanocytic Nevus', 'Melanoma',  'Actinic keratosis' ,'Benign keratosis']
    
    if prediction[0][class_index] >= 0.5:
        class_name = classes[class_index]
        accuracy = prediction[0][class_index]
    else:
        class_name = "Unknown"
        accuracy = 0.0
    
    return class_name, accuracy

# Streamlit app
def main():
    df = px.data.iris()

    
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()


    img = get_img_as_base64(r"bg.jpg")
    #data:image/png;base64,{img}
    page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.pinimg.com/736x/ef/b0/a3/efb0a3f19ee40192019e2c299b8a4e34.jpg");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}


.title {{
    font-size: 32px; /* Change this value to adjust the title size */
}}
</style>
"""
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown('<h1 class="title">SKIN DISEASE CLASSIFICATION</h1>', unsafe_allow_html=True)
    
    st.write("Upload an image to predict its class and accuracy.")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.write("Image uploaded successfully!")
        
        try:
            # Preprocess the uploaded image
            preprocessed_image = preprocess_image(uploaded_image)
            
            # Predict the class name and accuracy
            class_name, accuracy = predict_class_and_accuracy(preprocessed_image)
            
            # Display the predicted class name and accuracy
            st.write(f"Predicted Class: {class_name}")
            st.write(f"Accuracy: {accuracy:.2f}")
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
