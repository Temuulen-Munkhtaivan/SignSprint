SignSprint – ASL Real-Time Recognition
Requirements

- Python 3.11

- Webcam

Setup: 

git clone <repo>
cd SignSprint
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Train Model:

python ai_module/training/train_model.py

Run Real-Time Prediction:

python ai_module/api/realtime_predict.py