const letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"];

let targetLetter = "A";
let score = 0;
let streak = 0;
let lastPredictionTime = 0;
let canScore = true;

const CONFIDENCE_THRESHOLD = 0.75;
const PREDICTION_DELAY = 500;

const targetEl = document.getElementById("targetLetter");
const predictionEl = document.getElementById("prediction");
const scoreEl = document.getElementById("score");
const streakEl = document.getElementById("streak");
const messageEl = document.getElementById("message");
const webcam = document.getElementById("webcam");

let timeLeft = 60;
let timerInterval = null;
const timerEl = document.getElementById("timer");

function startTimer() {
  if (timerInterval) return;

  timerInterval = setInterval(() => {
    timeLeft--;
    timerEl.textContent = timeLeft;

    if (timeLeft <= 0) {
      clearInterval(timerInterval);
      messageEl.textContent = `Game Over! Final Score: ${score}`;
    }
  }, 1000);
}

function nextLetter() {
  targetLetter = letters[Math.floor(Math.random() * letters.length)];
  targetEl.textContent = targetLetter;
  canScore = true;
}

function updateScore() {
  if (!canScore) return;

  canScore = false;
  score += 10;
  streak += 1;

  scoreEl.textContent = score;
  streakEl.textContent = streak;
  messageEl.textContent = "Correct! +10 points";

  setTimeout(() => {
    nextLetter();
    messageEl.textContent = "Show the next sign!";
  }, 1000);
}

async function sendLandmarks(landmarks) {

  if (timeLeft <= 0) return;
  const now = Date.now();

  if (now - lastPredictionTime < PREDICTION_DELAY) {
    return;
  }

  lastPredictionTime = now;

  try {
    const response = await fetch("http://127.0.0.1:5000/predict-landmarks", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        landmarks: landmarks
      })
    });

    const data = await response.json();

    if (data.error) {
      console.error(data.error);
      return;
    }

    const confidencePercent = Math.round(data.confidence * 100);

    predictionEl.textContent = `${data.prediction} (${confidencePercent}%)`;
    console.log(data.prediction, data.confidence);

    if (
      data.confidence >= CONFIDENCE_THRESHOLD &&
      data.prediction === targetLetter
    ) {
      updateScore();
    }

  } catch (error) {
    console.error("Prediction error:", error);
  }
}

function flattenLandmarks(handLandmarks) {
  let coords = [];

  const wrist = handLandmarks[0];

  for (const lm of handLandmarks) {
    coords.push([
      -(lm.x - wrist.x),
      lm.y - wrist.y,
      lm.z - wrist.z
    ]);
  }

  let maxVal = 0;

  for (const point of coords) {
    for (const value of point) {
      maxVal = Math.max(maxVal, Math.abs(value));
    }
  }

  if (maxVal === 0) {
    maxVal = 1;
  }

  let flat = [];

  for (const point of coords) {
    flat.push(point[0] / maxVal);
    flat.push(point[1] / maxVal);
    flat.push(point[2] / maxVal);
  }

  return flat;
}

async function startGame() {
  startTimer();
  nextLetter();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true
    });

    webcam.srcObject = stream;

    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7
    });

    hands.onResults(async (results) => {
      if (
        results.multiHandLandmarks &&
        results.multiHandLandmarks.length > 0
      ) {
        const landmarks = flattenLandmarks(results.multiHandLandmarks[0]);
        await sendLandmarks(landmarks);
      } else {
        predictionEl.textContent = "-";
      }
    });

    const camera = new Camera(webcam, {
      onFrame: async () => {
        await hands.send({ image: webcam });
      },
      width: 640,
      height: 480
    });

    camera.start();

    messageEl.textContent = "Show the correct sign!";

  } catch (error) {
    console.error("Webcam error:", error);
    messageEl.textContent = "Could not access webcam.";
  }
}