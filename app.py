import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Get absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
model_path = os.path.join(script_dir, 'Emotion_Detection.h5')

# Load models
face_classifier = cv2.CascadeClassifier(cascade_path)
classifier = load_model(model_path)
print("Model loaded successfully.")

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    img_data = data['image']
    # Format of data url: "data:image/jpeg;base64,...base64-string..."
    if "," in img_data:
        header, encoded = img_data.split(",", 1)
    else:
        encoded = img_data
    
    # Decode base64 bytes to image
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    results = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            
            results.append({
                'box': [int(x), int(y), int(w), int(h)],
                'emotion': label
            })

    return jsonify({'predictions': results})

if __name__ == '__main__':
    # Using threaded=False because Keras models sometimes have issues being shared across threads
    app.run(debug=True, port=8000, threaded=False)
