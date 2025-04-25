from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
import cv2

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
try:
    # Emotion recognition model
    emotion_model = load_model("model/fer_model.keras")
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    # Face detection model
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    emotion_model = None
    face_cascade = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def contains_face(img_array):
    """Check if image contains at least one face"""
    try:
        # Convert PIL image to OpenCV format
        gray = np.array(img_array.convert('L'))
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))
        return len(faces) > 0
    except Exception as e:
        print(f"Face detection error: {e}")
        return False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check if request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check if models are loaded
    if not emotion_model or not face_cascade:
        return jsonify({"error": "Models not loaded"}), 500
    
    try:
        # Read image
        img = Image.open(io.BytesIO(file.read()))
        
        # Check if image contains a face
        if not contains_face(img):
            return jsonify({"error": "No face detected in the image. Please upload a clear photo of a face."}), 400
        
        # Convert to grayscale and resize
        img = img.convert("L").resize((48, 48))
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=(0, -1))  # Add dimensions
        
        # Make prediction
        predictions = emotion_model.predict(img_array)[0]
        
        # Format results
        results = {
            emotion: float(prob) * 100  # Convert to percentage
            for emotion, prob in zip(EMOTIONS, predictions)
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

# CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)