import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from preprocess import extract_features

app = Flask(__name__)

# -------------------- CONFIG --------------------
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model("model.h5")

# Labels mapping
labels = {
    0: "Human Voice",
    1: "Non-Human Sound",
    2: "Possible / Uncertain"
}

# -------------------- ROUTES --------------------

@app.route("/")
def home():
    """Render home page with file upload and microphone"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle audio file upload and model prediction"""
    if "audio" not in request.files:
        return "No file uploaded"

    file = request.files["audio"]

    if file.filename == "":
        return "No selected file"

    # Save file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Feature extraction
    features = extract_features(filepath)
    if features is None:
        return "Error processing audio"

    # Reshape for CNN input
    features = np.expand_dims(features, axis=0)
    features = features[..., np.newaxis]

    # Make prediction
    prediction = model.predict(features)
    class_index = int(np.argmax(prediction))
    confidence = round(float(np.max(prediction)) * 100, 2)
    result = labels[class_index]

    # URL for audio playback
    audio_url = f"/uploads/{file.filename}"

    # Render result page
    return render_template("result.html",
                           result=result,
                           confidence=confidence,
                           audio_url=audio_url)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded audio files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(debug=True)