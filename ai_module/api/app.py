from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "asl_landmark_model.keras")
LABEL_PATH = os.path.join(BASE_DIR, "model", "label_classes.npy")

model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABEL_PATH, allow_pickle=True)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict-landmarks", methods=["POST"])
def predict_landmarks():
    data = request.get_json()

    landmarks = data.get("landmarks")

    if not landmarks:
        return jsonify({"error": "No landmarks provided"}), 400

    landmarks = np.array(landmarks).reshape(1, -1)

    prediction = model.predict(landmarks, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": str(predicted_label),
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)