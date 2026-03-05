# 😊 Emotion Detection

A real-time facial emotion detection application using a pre-trained deep learning model and OpenCV. The app captures live video from your webcam, detects faces, and classifies the emotion displayed into one of five categories: **Angry, Happy, Neutral, Sad, or Surprise**.

---

## 🗂️ Project Structure

```
Emotion-Detection/
├── app.py                            # Flask server backend
├── emotion.py                        # (Legacy) Local OpenCV script
├── templates/
│   └── index.html                    # Web UI frontend
├── Emotion_Detection.h5              # Pre-trained Keras model
├── haarcascade_frontalface_default.xml  # OpenCV face detector
├── requirements.txt                  # Python dependencies
├── Procfile                          # Deployment config (Render/Heroku)
├── .gitignore                        # Git ignore rules
└── README.md
```

---

## ⚙️ Requirements

- Python 3.9 – 3.11
- A working webcam
- Dependencies listed in `requirements.txt`

---

## 🚀 Setup & Run

### 1. Clone the Repository
```bash
git clone https://github.com/Karshavarthini/Emotion-Detection.git
cd Emotion-Detection
```

### 2. Create a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application locally
```bash
python app.py
```

Open your browser and navigate to:
**http://localhost:8000**

Click **"Start Camera"** and allow webcam permissions in your browser to begin real-time detection.

---

## 🌐 Deploying to Render

This repository is fully configured for a "Web Service" deployment on platforms like [Render](https://render.com).

1. Push your code to GitHub.
2. Link your repo on Render and create a **Web Service**.
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `gunicorn app:app`
5. Click **Deploy**. Your app will be live and accessible from any browser!

---

## 🎭 Supported Emotions

| Label    | Description              |
|----------|--------------------------|
| Angry    | Angry or frustrated face |
| Happy    | Smiling or joyful face   |
| Neutral  | Expressionless face      |
| Sad      | Sad or upset face        |
| Surprise | Surprised or shocked face|

---

## 🧠 Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Input:** 48×48 grayscale face image
- **Output:** Softmax probabilities over 5 emotion classes
- **Format:** Keras `.h5` (legacy format, compatible with TF 2.x + Keras 2.x)

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| Camera not opening | Grant camera permissions to Terminal / IDE in System Settings |
| Model load error | Ensure you're using `keras==2.15.0` with `tensorflow==2.15.1` |
| No window appears | Make sure you're not running in a headless/SSH environment |
| Multiple cameras | Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `emotion.py` |

---

## 📄 License

This project is open-source. Feel free to use and modify it.
