import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import model_service

base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(base_dir, "..", "frontend")

app = FastAPI()


@app.websocket("/ws/predict")
async def predict_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            landmarks = payload.get("landmarks")
            mode = payload.get("mode", "letters")  # letters or words

            if not isinstance(landmarks, list) or len(landmarks) != model_service.NUM_FEATURES:
                await websocket.send_json({
                    "error": f"expected 'landmarks' as a list of {model_service.NUM_FEATURES} floats"
                })
                continue

            if mode not in ["letters", "words"]:
                await websocket.send_json({
                    "error": "invalid mode, must be 'letters' or 'words'"
                })
                continue

            try:
                letter, confidence = model_service.predict(landmarks)
            except (TypeError, ValueError):
                await websocket.send_json({"error": "invalid landmark values"})
                continue

            await websocket.send_json({
                "mode": mode,
                "letter": letter,
                "confidence": confidence
            })
    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")