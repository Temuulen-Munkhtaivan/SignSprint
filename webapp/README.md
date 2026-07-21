# SignSprint — Web Game

Real-time ASL fingerspelling game. Hand tracking runs in the browser
(MediaPipe Tasks Vision); the browser sends only the 63 normalized landmark
floats to a small FastAPI backend, which runs the trained model (as a plain
NumPy forward pass — no TensorFlow/TFLite needed at serving time) and returns
the predicted letter.

## Project layout

```
webapp/
  backend/
    main.py            FastAPI app: serves the frontend + /ws/predict websocket
    model_service.py    NumPy forward pass over the exported weights
    export_weights.py   One-time: ai_module/model/asl_landmark_model.keras -> model/weights.npz
    model/               weights.npz + label_classes.npy (generated, committed so the
                          backend needs no ML framework installed at all)
    requirements.txt
  frontend/
    index.html, style.css, app.js
    assets/letters/      optional per-letter reference images (see its README)
  render.yaml, backend/Procfile   deploy config
```

## Run locally

Requires Python 3.11+ (any version — the backend only needs `fastapi`,
`uvicorn`, and `numpy`; it does not need TensorFlow).

```bash
cd webapp/backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 and click **Start Game**. Localhost is treated as
a secure context by browsers, so camera access works without HTTPS here.

## Retraining / re-exporting the model

If `ai_module/model/asl_landmark_model.keras` changes (retrained on new data),
re-export the weights the backend actually serves:

```bash
# from a Python env that has tensorflow installed (matching ai_module/requirements.txt)
python webapp/backend/export_weights.py
```

This overwrites `webapp/backend/model/weights.npz` and `label_classes.npy`.
Commit those two files — the deployed backend reads them directly and never
imports TensorFlow.

## Deploying (Render)

1. Push this repo to GitHub (already connected: `Temuulen-Munkhtaivan/SignSprint`).
2. On [render.com](https://render.com), New -> Web Service -> connect the repo.
3. Render should auto-detect `render.yaml` at the repo root's sibling path —
   if not, set manually:
   - **Root Directory**: `webapp/backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy. Render provisions HTTPS automatically — required for camera access
   on a real domain (only `localhost` gets a free pass from browsers).
5. Open the deployed URL, click Start Game, allow camera access.

Free-tier specifics (RAM/cold-start/pricing) shift over time — check Render's
current plan details before deploying. If Render's free tier ever becomes too
constrained, this backend has no heavy dependencies (just numpy), so it also
runs fine on most other Python-friendly hosts (Railway, Fly.io, a plain VPS).

## Known calibration point

The browser flips the camera frame horizontally before running hand
detection (`app.js`'s `flipCanvas`), mirroring `cv2.flip(frame, 1)` in
`ai_module/data_collection/data_collector.py` — this has to match, since the
model has no handedness feature and the raw landmark coordinates do encode
left/right orientation. If predictions come out systematically confused on
asymmetric letters (e.g. D/G-shaped letters), check that flip is still in
place before assuming it's a model accuracy issue.
