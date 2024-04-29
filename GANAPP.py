import streamlit as st
import pandas as pd
import os
from time import sleep
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import base64
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Replace this line
tf.compat.v1.reset_default_graph()
def model_load_bal( image_paths, labels):
    model = load_model('dense.h5')
    
    images = load_images(image_paths)
    encoded_labels = encode_labels(labels)
    
    loss, accuracy = model.evaluate(images, encoded_labels)
    
    return accuracy

def preprocess_uploaded_image(uploaded_image, target_size=(90, 90)):
    img = Image.open(uploaded_image)
    img = img.resize(target_size)  # Resize the image to the specified target size  # Add a channel dimension
    img = img / 255.0  # Normalize the image
    return img
def predict_uploaded_image(image):
    model = load_model('trained_model.h5')
    prediction = model.predict(np.expand_dims(image,axis=0))
    class_index = np.argmax(prediction)
    classes = ['Melanocytic Nevus', 'Melanoma', 'Basal cell carcinoma', 'Benign keratosis', 'Actinic keratosis']  # Adjust based on your model's classes
    
    # Check if class_index is within the range of classes list
    if 0 <= class_index < len(classes):
        class_name = classes[class_index]
        accuracy = prediction[0][class_index]
        return f"{class_name}", f"{accuracy:.2f}"
    else:
        return "Unknown", "0.0"


def directory_to_dataframe(directory):
    # Initialize lists to store data
    file_paths = []
    classes = []

    # Traverse through the directory
    for root, dirs, files in os.walk(directory):
        # Extract class name from the current directory
        class_name = os.path.basename(root)
        
        # Iterate through files
        for file in files:
            # Append file path
            file_paths.append(os.path.join(root, file))
            # Append class
            classes.append(class_name)

    # Create DataFrame
    df = pd.DataFrame({'File_Path': file_paths, 'Class': classes})
    
    return df


# Function to register a new user and save their details in an Excel file
def register_and_save(username, password):
    try:
        user_data = pd.read_excel("user_data.xlsx")
    except FileNotFoundError:
        user_data = pd.DataFrame(columns=["Username", "Password"])
    
    if username in user_data["Username"].values:
        st.error("Username already exists. Please choose a different username.")
        return
    
    new_user = pd.DataFrame({"Username": [username], "Password": [password]})
    user_data = pd.concat([user_data, new_user], ignore_index=True)
    
    user_data.to_excel("user_data.xlsx", index=False)
    st.success("Registration successful. You can now login.")

# Function to load the pre-trained model
def model_load( image_paths, labels):  # Ensure this matches the original architecture
    model = load_model('/content/mydrive/MyDrive/NEW_ISIC - 2019/NEW_ISIC - 2019/trained_model.h5')
    
    images = load_images(image_paths)
    encoded_labels = encode_labels(labels)
    
    loss, accuracy = model.evaluate(images, encoded_labels)
    
    return accuracy

# Function to load images into arrays
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=(100, 100))
        img = img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)

# Function to encode labels
def encode_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels

# Function to login
import pandas as pd
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")
    import pandas as pd
    if login_button:
            user_data = pd.read_excel("user_data.xlsx")
            if username == "admin" and password == "admin":
                # If so, redirect to the new page
                st.success(f"Logged in as {username}")
                st.switch_page("pages/new.py")
            elif (username in user_data["Username"].values) and \
               (password == user_data.loc[user_data["Username"] == username, "Password"].values[0]):
                st.success(f"Logged in as {username}")
                st.switch_page("pages/app.py")
                
            else:
                st.error("Invalid username or password")        
    

    bdf=directory_to_dataframe(r"C:\Users\gorre\Downloads\NEW_ISIC - 2019\NEW_ISIC - 2019\newbalance")
    

# Function to register page
def register_page():
    st.title("Register Page")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    register_button = st.button("Register")
    if register_button:
        if new_password != confirm_password:
            st.error("Passwords do not match. Please try again.")
        else:
            register_and_save(new_username, new_password)



# Main function to switch between login, register, and user pages
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
background-image: url("https://cdn.sanity.io/images/ay6gmb6r/production/36d84e202d6c836424b90f8631f23fc28eee51c6-2240x1260.png");
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
    page = st.sidebar.selectbox("Select Page", ["Register","Login"])
    if page == "Login":
        login()
    elif page == "Register":
        register_page()
    
    

if __name__ == "__main__":
    main()
