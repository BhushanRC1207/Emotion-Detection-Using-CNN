import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import streamlit as st

# Initialize session state
if 'button_pressed' not in st.session_state:
    st.session_state.button_pressed = False

# Optional: Add a restart button to reset the state
if st.sidebar.button('Start'):
    st.session_state.button_pressed = False


# Create button
if st.sidebar.button('Stop'):
    st.session_state.button_pressed = True


# Conditional logic to prevent rerun
if not st.session_state.button_pressed:
    # Streamlit app title
    st.title('Real Time Emotion Detection')

    # Load the emotion detection model
    json_file = open('custom_cnn_augmentation_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("custom_cnn_augmentation_model.weights.h5")

    # Emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    # Load the face detection model (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Function to process each frame
    def process_frame(frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face_gray = gray[y:y + h, x:x + w]

            # Resize to match model input size
            face_resized = cv2.resize(face_gray, (48, 48))

            # Normalize the image data
            img_array = face_resized.astype("float32") / 255.0

            # Expand dimensions to match the input shape for the model
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            pred = model.predict(img_array)
            emotion = emotion_labels[np.argmax(pred)]

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the emotion label
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

        return frame



    # Real-time processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Process the frame for emotion detection
        processed_frame = process_frame(frame)

        # Display the processed frame
        stframe.image(processed_frame, channels="BGR")



    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

