# SignSprint – ASL Fingerspelling Recognition

A real-time American Sign Language (ASL) fingerspelling recognition game with
a browser-based camera interface.

**Live demo:** https://signsprint.onrender.com

**Run the web app:** see [webapp/README.md](webapp/README.md) for setup,
running locally, testing, and deployment — that's the actual product.

**Model training / data collection:** `ai_module/` holds the offline
pipeline used to build the model the web app serves:

- `ai_module/data_collection/data_collector.py` — collects labelled hand-landmark data from a webcam
- `ai_module/training/train_model.py` — trains the ASL letter classification model
- `webapp/backend/export_weights.py` — exports a retrained model to the NumPy weights the web app actually serves

See the "Retraining / re-exporting the model" section in
[webapp/README.md](webapp/README.md) for the full retrain → re-export →
redeploy workflow.
