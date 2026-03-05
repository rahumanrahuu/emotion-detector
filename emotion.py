# Import necessary libraries
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import os

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the face classifier and emotion detection model using robust paths
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
model_path = os.path.join(script_dir, 'Emotion_Detection.h5')

face_classifier = cv2.CascadeClassifier(cascade_path)
classifier = load_model(model_path)
print("Model loaded successfully.")

# Class labels for emotions
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture (camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully. Press 'q' to quit.")

# Function to display images (using OpenCV for real-time video)
def display_image(img):
    cv2.imshow('Emotion Detector', img)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Waiting for camera or no frame captured...")
        cv2.waitKey(1000) # Wait a bit before retrying
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            # Preprocess the face region for model prediction
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Make a prediction on the face
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)

            # Display emotion label on the image
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Show the resulting frame
    display_image(frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()