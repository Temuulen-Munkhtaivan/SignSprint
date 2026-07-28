import os

import numpy as np

_base_dir = os.path.dirname(os.path.abspath(__file__))
_weights_path = os.path.join(_base_dir, "model", "weights.npz")
_labels_path = os.path.join(_base_dir, "model", "label_classes.npy")

_weights = np.load(_weights_path)
W1, B1 = _weights["w1"], _weights["b1"]
W2, B2 = _weights["w2"], _weights["b2"]
W3, B3 = _weights["w3"], _weights["b3"]

LABELS = np.load(_labels_path, allow_pickle=True)

NUM_FEATURES = W1.shape[0]  # 63: 21 landmarks x (x, y, z), wrist-relative, max-abs normalized


def _relu(x):
    return np.maximum(x, 0)


def _softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def predict(landmarks):
    """landmarks: sequence of NUM_FEATURES floats (already normalized client-side,
    matching ai_module/data_collection/data_collector.py's preprocessing).
    Returns (letter: str, confidence: float, probs: dict[str, float])."""
    x = np.asarray(landmarks, dtype=np.float32).reshape(1, -1)
    if x.shape[1] != NUM_FEATURES:
        raise ValueError(f"expected {NUM_FEATURES} features, got {x.shape[1]}")

    h1 = _relu(x @ W1 + B1)
    h2 = _relu(h1 @ W2 + B2)
    logits = h2 @ W3 + B3
    probs = _softmax(logits)[0]

    class_index = int(np.argmax(probs))
    letter = str(LABELS[class_index])
    confidence = float(probs[class_index])
    return letter, confidence
