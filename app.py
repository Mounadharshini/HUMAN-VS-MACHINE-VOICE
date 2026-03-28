import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from preprocess import extract_features

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("model.h5")

labels = {
    0: "Human Voice",
    1: "Non-Human Sound",
    2: "Possible / Uncertain"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return "No file uploaded"
    file = request.files["audio"]
    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    features = extract_features(filepath)
    if features is None:
        return "Error processing audio"

    features = np.expand_dims(features, axis=0)
    features = features[..., np.newaxis]

    prediction = model.predict(features)
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = round(float(np.max(prediction)) * 100, 2)
    result = labels[class_index]

    audio_url = f"/uploads/{file.filename}"

    return render_template("result.html", result=result, confidence=confidence, audio_url=audio_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
