import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import { HAND_CONNECTIONS } from "./config.js";

let handLandmarker = null;

export async function setupHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  const modelAssetPath =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
    });
  } catch {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath, delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 1,
    });
  }
  return handLandmarker;
}

export async function setupCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();
  return stream;
}

// Offscreen flipped canvas: mirrors cv2.flip(frame, 1) from the training
// pipeline so browser landmark coordinates match what the model was trained on.
const flipCanvas = document.createElement("canvas");
const flipCtx = flipCanvas.getContext("2d");

export function getFlippedFrame(video) {
  flipCanvas.width = video.videoWidth;
  flipCanvas.height = video.videoHeight;
  flipCtx.save();
  flipCtx.translate(flipCanvas.width, 0);
  flipCtx.scale(-1, 1);
  flipCtx.drawImage(video, 0, 0, flipCanvas.width, flipCanvas.height);
  flipCtx.restore();
  return flipCanvas;
}

export function detectHands(flippedCanvas) {
  return handLandmarker.detectForVideo(flippedCanvas, performance.now());
}

export function drawSkeleton(ctx, landmarks, width, height) {
  ctx.strokeStyle = "#34e4c7";
  ctx.lineWidth = 3;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
    ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
    ctx.stroke();
  }
  ctx.fillStyle = "#ff5da2";
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Renders a per-letter reference vector (from letter_references.json --
// wrist-relative mean landmark positions computed from the training data)
// as a static skeleton diagram, reusing drawSkeleton so it visually matches
// the live camera overlay. This is what teaches a beginner the handshape
// when no reference photo has been supplied (see assets/letters/README.md).
export function drawGhostHand(canvas, referenceFlat63) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!referenceFlat63) return;

  const points = [];
  for (let i = 0; i < 21; i++) {
    points.push({ x: referenceFlat63[i * 3], y: referenceFlat63[i * 3 + 1] });
  }

  // Reference coords are wrist-relative offsets (wrist = origin), roughly
  // in [-1, 1] after per-sample normalization -- map that onto the canvas
  // around a fixed center point rather than the raw [0,1] image-fraction
  // space drawSkeleton normally expects.
  const CENTER_X = 0.5;
  const CENTER_Y = 0.38;
  const SCALE = 0.36;
  const canvasPoints = points.map((p) => ({
    x: CENTER_X + p.x * SCALE,
    y: CENTER_Y + p.y * SCALE,
  }));

  drawSkeleton(ctx, canvasPoints, canvas.width, canvas.height);
}

// Must mirror ai_module/data_collection/data_collector.py's normalization exactly:
// wrist-relative coordinates, divided by the max absolute value across all of them.
export function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const rel = landmarks.map((lm) => [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]);
  let maxVal = 1e-6;
  for (const [x, y, z] of rel) {
    maxVal = Math.max(maxVal, Math.abs(x), Math.abs(y), Math.abs(z));
  }
  const flat = [];
  for (const [x, y, z] of rel) flat.push(x / maxVal, y / maxVal, z / maxVal);
  return flat;
}
