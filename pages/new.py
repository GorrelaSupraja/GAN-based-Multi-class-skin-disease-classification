import streamlit as st
import torch
import numpy as np
import plotly.express as px
from PIL import Image
from torchvision.transforms import functional as F
from transformers import ViTForImageClassification
import base64
import streamlit as st
import pandas as pd
import os
from time import sleep
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import base64
import os
tf.compat.v1.reset_default_graph()




# Function to display balanced data statistics
def show_balanced_data():
    st.subheader("Balanced Data Information")
    balanced_data = pd.Series({
        'Benign keratosis': 400,
        'Actinic keratosis': 400,
        'Melanocytic nevus': 400,
        'Melanoma': 400,
        'Basal cell carcinoma': 400
    })
    st.write(balanced_data)  # Corrected variable name here

def show_unbalanced_data():
    st.subheader("Unbalanced Data Information")
    unbalanced_data = pd.Series({
        'Benign keratosis': 168,
        'Actinic keratosis': 167,
        'Melanocytic nevus': 505,
        'Melanoma': 330,
        'Basal cell carcinoma': 248
    })
    st.write(unbalanced_data)
def plot_balanced_data():
    balanced_data = pd.Series({
        'Benign keratosis': 400,
        'Actinic keratosis': 400,
        'Melanocytic nevus': 400,
        'Melanoma': 400,
        'Basal cell carcinoma': 400
    })
    st.bar_chart(balanced_data)

# Function to plot unbalanced data
def plot_unbalanced_data():
    unbalanced_data = pd.Series({
        'Benign keratosis': 168,
        'Actinic keratosis': 167,
        'Melanocytic nevus': 505,
        'Melanoma': 330,
        'Basal cell carcinoma': 248
    })
    st.bar_chart(unbalanced_data)
import matplotlib.pyplot as plt
def plot_comparison():
    st.subheader("Comparison of Balanced vs Unbalanced Data")
    
    epochs = [2,4,6,8,10]
    accuracy_1 = [0.75, 0.80, 0.85, 0.88, 0.90]
    accuracy_2 = [0.70, 0.75, 0.78, 0.80, 0.82]
    accuracy_3 = [0.65, 0.70, 0.75, 0.78, 0.80]
    accuracy_4 = [0.60, 0.65, 0.70, 0.72, 0.75]

    plt.plot(epochs, accuracy_1, label='Balanced Densenet')
    plt.plot(epochs, accuracy_2, label='Balanced VGG16')
    plt.plot(epochs, accuracy_3, label='Unbalanced Densenet')
    plt.plot(epochs, accuracy_4, label='Unabalanced VGG16')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.show()
    # Display the plot in Streamlit
    st.pyplot(plt)
    

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
    #st.title("Admin Page")
    st.write("Welcome, admin!")
    # Your admin page content here
    st.write("You can add your admin functionalities here.")
    show_balanced_button = st.button("Show Balanced Data")
    show_unbalanced_button = st.button("Show Unbalanced Data")
    
    if show_balanced_button:
        show_balanced_data()
        plot_balanced_data()
    elif show_unbalanced_button:
        show_unbalanced_data()
        plot_unbalanced_data()
    elif plot_comparison:
        plot_comparison()
   
    

   

    
if __name__ == "__main__":
    main()