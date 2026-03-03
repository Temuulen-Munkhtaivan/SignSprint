import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from collections import deque

# ===== Path Setup =====
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "..", "model")

model_path = os.path.join(model_dir, "asl_landmark_model.keras")
label_path = os.path.join(model_dir, "label_classes.npy")

model = tf.keras.models.load_model(model_path)
label_classes = np.load(label_path, allow_pickle=True)

# ===== MediaPipe Setup =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===== Prediction Smoothing =====
prediction_history = deque(maxlen=10)
CONFIDENCE_THRESHOLD = 0.75

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted_letter = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            wrist = hand_landmarks.landmark[0]
            coords = []

            for lm in hand_landmarks.landmark:
                coords.append([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            coords = np.array(coords)
            max_val = np.max(np.abs(coords))
            coords = coords / max_val
            landmark_list = coords.flatten().tolist()

            input_data = np.array(landmark_list).reshape(1, -1)
            prediction = model.predict(input_data, verbose=0)

            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > CONFIDENCE_THRESHOLD:
                prediction_history.append(label_classes[class_index])

    if len(prediction_history) > 0:
        predicted_letter = max(set(prediction_history),
                                key=prediction_history.count)

    cv2.putText(frame, f"Prediction: {predicted_letter}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0), 2)

    cv2.imshow("ASL Real-Time Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()