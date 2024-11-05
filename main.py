import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pandas as pd
import datetime
import streamlit as st
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained emotion detection model
model = load_model('pradde_model.h5')  # Adjust the path to the model if necessary
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Path for the CSV file to store the analysis
csv_file = "C:/Users/Danish Khan/Downloads/project facial/customer_feedback_analysis.csv"

# Create the CSV file and write the header if it doesn't exist
if os.path.exists(csv_file):
    try:
        df = pd.read_csv(csv_file)
        logging.info("CSV file read successfully.")
    except Exception as e:
        logging.error(f"An error occurred while reading the CSV file: {e}")
        df = pd.DataFrame(columns=["Date", "Detected Emotion", "Emotion Percentages"])
else:
    logging.warning(f"CSV file not found. Creating a new one at {csv_file}.")
    df = pd.DataFrame(columns=["Date", "Detected Emotion", "Emotion Percentages"])
    df.to_csv(csv_file, index=False)

def gen_frames():  # No need to pass df here
    cap = cv2.VideoCapture(0)  # Open the default camera
    emotion_counts = {label: 0 for label in emotion_labels}

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to grayscale for emotion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Initialize predicted_emotion and confidence
        predicted_emotion = "Neutral"
        confidence = 0.0

        for (x, y, w, h) in faces:
            roi = gray_frame[y:y + h, x:x + w]
            img_array = cv2.resize(roi, (48, 48))
            img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to 3 channels
            img_array = img_to_array(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize

            # Predict the emotion
            predictions = model.predict(img_array)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotion_labels[max_index]
            confidence = np.max(predictions[0])

            # Increment the count for the detected emotion
            emotion_counts[predicted_emotion] += confidence

            # Draw a rectangle around the face and label the emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{predicted_emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the overall emotion percentages
        total_confidence = sum(emotion_counts.values())
        if total_confidence > 0:
            for emotion in emotion_labels:
                emotion_counts[emotion] = (emotion_counts[emotion] / total_confidence) * 100  # Convert to percentage

        # Overlay customer feedback based on emotions
        feedback_label = interpret_emotion(predicted_emotion)
        cv2.putText(frame, feedback_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue  # If encoding fails, skip to the next frame
        frame = buffer.tobytes()
        yield frame, emotion_counts  # Yield frame and the current emotion counts

    cap.release()  # Release the camera when done

def interpret_emotion(predicted_emotion):
    """Interpret the detected emotion to determine customer feedback."""
    feedback_messages = {
        'Angry': "Customer seems upset.",
        'Disgust': "Consider offering assistance.",
        'Fear': "Customer appears anxious.",
        'Happy': "Customer is happy! Great service!",
        'Sad': "Offering assistance may help.",
        'Surprise':"Check if they need anything.",
        'Neutral': "Monitor for changes."
    }
    return feedback_messages.get(predicted_emotion, "Emotion not recognized.")

def live_emotion_detection():
    st.title("Live Emotion Detection")

    # Declare df as global to modify it within this function
    global df  

    # Start video capture
    stframe = st.empty()  # Placeholder for video stream
    emotion_counts = {label: 0 for label in emotion_labels}

    # Start the frame generator
    frame_generator = gen_frames()

    # Display frames from the generator
    for frame, counts in frame_generator:
        # Convert the frame from bytes to image
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Update the emotion counts
        for label in emotion_labels:
            emotion_counts[label] += counts[label]

    # After the stream ends, calculate percentages and save to CSV
    total_confidence = sum(emotion_counts.values())
    if total_confidence > 0:
        emotion_percentages = {label: (count / total_confidence) * 100 for label, count in emotion_counts.items()}
        detected_emotion = max(emotion_percentages, key=emotion_percentages.get)
    else:
        emotion_percentages = {label: 0 for label in emotion_labels}
        detected_emotion = "Neutral"

    # Log the interaction with the timestamp and store in CSV
    current_time = datetime.datetime.now()
    date_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Create a new DataFrame for the new row
    new_row = pd.DataFrame({
        "Date": [date_str],
        "Detected Emotion": [detected_emotion],
        "Emotion Percentages": [emotion_percentages]
    })

    # Append the new row to the existing DataFrame
    global df  # Declare df as global to modify it
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file, index=False)  # Save to CSV file

# Run the application
if __name__ == '__main__':
    live_emotion_detection()
