import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.metrics import classification_report, confusion_matrix

# ===== Path Setup =====
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "..", "dataset", "asl_landmark_data.csv")
model_dir = os.path.join(base_dir, "..", "model")
os.makedirs(model_dir, exist_ok=True)

# ===== Load Dataset =====
df = pd.read_csv(dataset_path)

#cleaning column names
df.columns = df.columns.str.strip()

#cleaning and encoding hand column
df["hand"] = df["hand"].astype(str).str.strip().str.lower()
df["hand"] = df["hand"].map({
    "left": 0,
    "right": 1
})

df = df.dropna(subset=["hand"])

X = df.drop("label", axis=1)
y = df["label"].values

# Force all X columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Remove rows with bad values
bad_rows = X.isna().any(axis=1)
print("Bad rows removed:", bad_rows.sum())

X = X[~bad_rows]
y = y[~bad_rows]

# Convert to NumPy arrays
X = X.to_numpy(dtype=np.float32)


# ===== Encode Labels =====
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ===== Train/Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ===== Build Model =====
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===== Train =====
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print(confusion_matrix(y_test, y_pred))

# ===== Evaluate =====
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# ===== Save =====
model_path = os.path.join(model_dir, "asl_landmark_model.keras")
label_path = os.path.join(model_dir, "label_classes.npy")

model.save(model_path)
np.save(label_path, encoder.classes_)

print("Model saved to:", model_path)