import os
import shutil

import numpy as np
import tensorflow as tf

# ===== Path Setup =====
base_dir = os.path.dirname(os.path.abspath(__file__))
ai_model_dir = os.path.join(base_dir, "..", "..", "ai_module", "model")
out_dir = os.path.join(base_dir, "model")
os.makedirs(out_dir, exist_ok=True)

keras_path = os.path.join(ai_model_dir, "asl_landmark_model.keras")
labels_src_path = os.path.join(ai_model_dir, "label_classes.npy")
weights_path = os.path.join(out_dir, "weights.npz")
labels_dst_path = os.path.join(out_dir, "label_classes.npy")

# ===== Extract weights =====
# Backend serves this model with a plain NumPy forward pass (no TF/TFLite
# runtime at serving time) since the network is tiny (63->128->64->24 dense)
# and this avoids any ML-framework version/availability risk on the deploy host.
model = tf.keras.models.load_model(keras_path, compile=False)

dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
assert len(dense_layers) == 3, f"expected 3 Dense layers, found {len(dense_layers)}"

w1, b1 = dense_layers[0].get_weights()
w2, b2 = dense_layers[1].get_weights()
w3, b3 = dense_layers[2].get_weights()

np.savez(
    weights_path,
    w1=w1, b1=b1,
    w2=w2, b2=b2,
    w3=w3, b3=b3,
)
shutil.copyfile(labels_src_path, labels_dst_path)

print("Wrote", weights_path)
print("Shapes:", w1.shape, b1.shape, w2.shape, b2.shape, w3.shape, b3.shape)
print("Wrote", labels_dst_path)
print("Labels:", list(np.load(labels_dst_path, allow_pickle=True)))
