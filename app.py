import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("chest_model.keras")

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input shape of your model (224x224)
    image = image.resize((224, 224))
    # Convert the image to a numpy array and scale pixel values to [0, 1]
    image = np.array(image) / 255.0
    # Expand dimensions to match the model's expected input shape (e.g., (1, 224, 224, 3))
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title("Lung Cancer Classification")

st.write("Upload an image to predict the class")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Predict using the loaded model
    predictions = model.predict(processed_image)
    
    # Convert the predictions into class names
    classes = ['Adenocarcinoma', 'Squamous Cell Carcinoma', 'Normal', 'Large Cell Carcinoma']
    predicted_class = classes[np.argmax(predictions)]

    # Display the result
    st.write(f"Predicted Class: **{predicted_class}**")
