import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import string

# ===== Path Setup =====
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "..", "dataset")
os.makedirs(dataset_dir, exist_ok=True)
file_path = os.path.join(dataset_dir, "asl_landmark_data.csv")

# ===== Settings =====
LETTERS = [l for l in string.ascii_uppercase if l not in ['J', 'Z']]
SAMPLES_PER_LETTER = 200

# ===== MediaPipe Setup =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

data = []
labels = []

current_letter_index = 0
sample_count = 0
capturing = False

print("Press SPACE to start capturing.")
print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_letter = LETTERS[current_letter_index]

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

            if capturing:
                data.append(landmark_list)
                labels.append(current_letter)
                sample_count += 1

    cv2.putText(frame, f"Letter: {current_letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Samples: {sample_count}/{SAMPLES_PER_LETTER}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key == 32:
        capturing = True

    if sample_count >= SAMPLES_PER_LETTER:
        capturing = False
        sample_count = 0
        current_letter_index += 1

        if current_letter_index >= len(LETTERS):
            break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv(file_path, index=False)

print("Dataset saved to:", file_path)