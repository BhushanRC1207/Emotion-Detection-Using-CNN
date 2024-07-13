from tensorflow.keras.models import model_from_json
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf


st.title('Emotion Detection System')
json_file = open('custom_cnn_augmentation_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("custom_cnn_augmentation_model.weights.h5")




# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# File uploader widget
img = st.file_uploader("Choose a file", type=["jpg", "png"])

if img is not None:
    # Open the image file
    st.image(Image.open(img))
    img = Image.open(img).convert('L')  # Convert to grayscale
    img = img.resize((48,48))

    # Convert image to NumPy array
    img_array = np.array(img)

    # Normalize the image data
    img_array = img_array.astype("float32") / 255.0

    # Expand dimensions to match the input shape for the model
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension



    # Make prediction
    pred = model.predict(img_array)
    emotion = emotion_labels[np.argmax(pred)]

    # Display the emotion label
    #st.title(emotion)

    st.title(f"Detected Emotion: {emotion}")

    # Additional statement based on the detected emotion
    if emotion == 'Happy':
        st.write("You look happy! 😄")
    elif emotion == 'Sad':
        st.write("Seems like you're feeling sad. 😞")
    else:
        st.write(f"It looks like you're feeling {emotion.lower()}.")
