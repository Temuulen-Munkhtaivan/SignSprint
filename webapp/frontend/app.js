import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ===== Config =====
const LETTERS = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"];
const ROUNDS_TOTAL = 15;
const ROUND_TIME_MS = 12000;
const HOLD_MS = 700;
const CONFIDENCE_THRESHOLD = 0.75;
const SEND_INTERVAL_MS = 100; // ~10fps to the backend
const HISTORY_LEN = 8;

// Short ASL fingerspelling cues used as a fallback when no reference image is
// supplied at assets/letters/<LETTER>.png (see that folder's README).
const LETTER_CUES = {
  A: "Fist with thumb resting alongside the fingers.",
  B: "Flat hand, fingers together pointing up, thumb folded across the palm.",
  C: "Curve the hand into a C shape.",
  D: "Index finger up, other fingers touch the thumb in a circle.",
  E: "Fingers curled down, thumb tucked in front.",
  F: "Thumb and index finger touch in a circle, other three fingers up.",
  G: "Index finger and thumb point sideways, like a small gun shape.",
  H: "Index and middle finger extended together, pointing sideways.",
  I: "Pinky finger up, other fingers in a fist.",
  K: "Index and middle finger up in a V, thumb between them.",
  L: "Thumb and index finger form an L, other fingers folded.",
  M: "Thumb tucked under three fingers (index, middle, ring).",
  N: "Thumb tucked under two fingers (index, middle).",
  O: "Fingers and thumb curved to form an O shape.",
  P: "Like K, but pointing downward.",
  Q: "Like G, but pointing downward.",
  R: "Index and middle finger crossed.",
  S: "Fist with thumb across the front of the fingers.",
  T: "Fist with thumb tucked between index and middle finger.",
  U: "Index and middle finger up together, straight.",
  V: "Index and middle finger up in a V shape.",
  W: "Index, middle, and ring finger up, spread apart.",
  X: "Index finger curled into a hook shape.",
  Y: "Thumb and pinky extended, other fingers folded down.",
};

// ===== DOM =====
const startScreen = document.getElementById("startScreen");
const gameScreen = document.getElementById("gameScreen");
const endScreen = document.getElementById("endScreen");
const startBtn = document.getElementById("startBtn");
const playAgainBtn = document.getElementById("playAgainBtn");
const statsEl = document.getElementById("stats");
const scoreVal = document.getElementById("scoreVal");
const streakVal = document.getElementById("streakVal");
const roundVal = document.getElementById("roundVal");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const noHandBanner = document.getElementById("noHandBanner");
const roundTimerBar = document.getElementById("roundTimerBar");
const targetLetterEl = document.getElementById("targetLetter");
const predictedLetterEl = document.getElementById("predictedLetter");
const holdBar = document.getElementById("holdBar");
const feedbackPanel = document.getElementById("feedbackPanel");
const feedbackTitle = document.getElementById("feedbackTitle");
const referencePanel = document.getElementById("referencePanel");
const referenceImg = document.getElementById("referenceImg");
const referenceCue = document.getElementById("referenceCue");
const masteryRow = document.getElementById("masteryRow");
const finalScore = document.getElementById("finalScore");
const finalAccuracy = document.getElementById("finalAccuracy");
const finalBestStreak = document.getElementById("finalBestStreak");
const connectionBanner = document.getElementById("connectionBanner");

// ===== Audio (created lazily on first user gesture) =====
let audioCtx = null;
function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
}
function beep(freqs, duration = 0.12, gap = 0.09) {
  if (!audioCtx) return;
  freqs.forEach((freq, i) => {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;
    const start = audioCtx.currentTime + i * gap;
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.2, start + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(start);
    osc.stop(start + duration + 0.02);
  });
}
const playCorrectSound = () => beep([523.25, 659.25, 783.99]);
const playMissSound = () => beep([220, 160]);

// ===== Confetti (tiny self-contained canvas effect) =====
function burstConfetti() {
  let canvas = document.getElementById("confettiCanvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.id = "confettiCanvas";
    document.body.appendChild(canvas);
  }
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx = canvas.getContext("2d");
  const colors = ["#ff5da2", "#34e4c7", "#ffd166", "#8b6bff"];
  const particles = Array.from({ length: 80 }, () => ({
    x: canvas.width / 2 + (Math.random() - 0.5) * 200,
    y: canvas.height * 0.35,
    vx: (Math.random() - 0.5) * 10,
    vy: Math.random() * -8 - 2,
    size: Math.random() * 7 + 3,
    color: colors[Math.floor(Math.random() * colors.length)],
    rotation: Math.random() * Math.PI,
    vr: (Math.random() - 0.5) * 0.3,
  }));

  let frame = 0;
  const maxFrames = 70;
  function step() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of particles) {
      p.vy += 0.35;
      p.x += p.vx;
      p.y += p.vy;
      p.rotation += p.vr;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rotation);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
      ctx.restore();
    }
    frame++;
    if (frame < maxFrames) requestAnimationFrame(step);
    else ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  requestAnimationFrame(step);
}

// ===== Normalization: must mirror ai_module/data_collection/data_collector.py =====
function normalizeLandmarks(landmarks) {
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

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

function drawSkeleton(landmarks, width, height) {
  overlayCtx.strokeStyle = "#34e4c7";
  overlayCtx.lineWidth = 3;
  for (const [a, b] of HAND_CONNECTIONS) {
    overlayCtx.beginPath();
    overlayCtx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
    overlayCtx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
    overlayCtx.stroke();
  }
  overlayCtx.fillStyle = "#ff5da2";
  for (const lm of landmarks) {
    overlayCtx.beginPath();
    overlayCtx.arc(lm.x * width, lm.y * height, 4, 0, Math.PI * 2);
    overlayCtx.fill();
  }
}

// ===== WebSocket client =====
let ws = null;
let wsReady = false;
let lastSentAt = 0;
let pendingResolve = null;

function connectWs() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws/predict`);
  ws.onopen = () => {
    wsReady = true;
    connectionBanner.hidden = true;
  };
  ws.onclose = () => {
    wsReady = false;
    connectionBanner.textContent = "Connection lost — reconnecting…";
    connectionBanner.hidden = false;
    setTimeout(connectWs, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (event) => {
    if (!pendingResolve) return;
    const data = JSON.parse(event.data);
    pendingResolve(data);
    pendingResolve = null;
  };
}

function requestPrediction(landmarks) {
  if (!wsReady) return Promise.resolve(null);
  return new Promise((resolve) => {
    pendingResolve = resolve;
    ws.send(JSON.stringify({ landmarks }));
  });
}

// ===== Prediction smoothing (mirrors realtime_predict.py's prediction_history) =====
const predictionHistory = [];
function pushPrediction(letter, confidence) {
  if (confidence > CONFIDENCE_THRESHOLD) {
    predictionHistory.push(letter);
    if (predictionHistory.length > HISTORY_LEN) predictionHistory.shift();
  }
}
function smoothedPrediction() {
  if (predictionHistory.length === 0) return null;
  const counts = {};
  let best = predictionHistory[0];
  for (const l of predictionHistory) {
    counts[l] = (counts[l] || 0) + 1;
    if (counts[l] > counts[best]) best = l;
  }
  return best;
}

// ===== Game state =====
let handLandmarker = null;
let stream = null;
let running = false;
let sendInFlight = false;

let score = 0;
let streak = 0;
let bestStreak = 0;
let roundIndex = 0;
let currentTarget = null;
let holdMs = 0;
let roundStartTime = 0;
let roundActive = false;
let correctCount = 0;
const mastered = new Set();

function pickNextLetter() {
  let letter;
  do {
    letter = LETTERS[Math.floor(Math.random() * LETTERS.length)];
  } while (letter === currentTarget && LETTERS.length > 1);
  return letter;
}

function renderMasteryRow() {
  masteryRow.innerHTML = "";
  for (const letter of LETTERS) {
    const tile = document.createElement("div");
    tile.className = "mastery-tile";
    if (mastered.has(letter)) tile.classList.add("mastered");
    if (letter === currentTarget) tile.classList.add("current");
    tile.textContent = letter;
    masteryRow.appendChild(tile);
  }
}

function updateStatsUI() {
  scoreVal.textContent = String(score);
  streakVal.textContent = String(streak);
  roundVal.textContent = `${Math.min(roundIndex, ROUNDS_TOTAL)}/${ROUNDS_TOTAL}`;
}

function showReferenceFor(letter) {
  referencePanel.hidden = false;
  referenceCue.textContent = LETTER_CUES[letter] || "";
  referenceImg.hidden = false;
  referenceImg.src = `assets/letters/${letter}.png`;
  referenceImg.onerror = () => { referenceImg.hidden = true; };
}

function startRound() {
  currentTarget = pickNextLetter();
  targetLetterEl.textContent = currentTarget;
  holdMs = 0;
  holdBar.style.width = "0%";
  roundStartTime = performance.now();
  roundActive = true;
  feedbackPanel.hidden = true;
  feedbackPanel.classList.remove("correct", "miss");
  referencePanel.hidden = true;
  predictionHistory.length = 0;
  predictedLetterEl.textContent = "—";
  renderMasteryRow();
}

function endRound(wasCorrect) {
  roundActive = false;
  roundIndex++;

  feedbackPanel.hidden = false;
  if (wasCorrect) {
    correctCount++;
    streak++;
    bestStreak = Math.max(bestStreak, streak);
    const bonus = Math.min(streak * 2, 20);
    score += 10 + bonus;
    mastered.add(currentTarget);
    feedbackPanel.classList.add("correct");
    feedbackPanel.classList.remove("miss");
    feedbackTitle.textContent = `Correct! +${10 + bonus} pts`;
    playCorrectSound();
    burstConfetti();
  } else {
    streak = 0;
    feedbackPanel.classList.add("miss");
    feedbackPanel.classList.remove("correct");
    feedbackTitle.textContent = "Not quite — here's the correct handshape:";
    showReferenceFor(currentTarget);
    playMissSound();
    document.querySelector(".camera-wrap").classList.add("shake");
    setTimeout(() => document.querySelector(".camera-wrap").classList.remove("shake"), 400);
  }
  updateStatsUI();
  renderMasteryRow();

  if (roundIndex >= ROUNDS_TOTAL) {
    setTimeout(showEndScreen, wasCorrect ? 900 : 1800);
  } else {
    setTimeout(startRound, wasCorrect ? 900 : 1800);
  }
}

function showEndScreen() {
  running = false;
  gameScreen.hidden = true;
  statsEl.hidden = true;
  endScreen.hidden = false;
  finalScore.textContent = String(score);
  finalAccuracy.textContent = `${Math.round((correctCount / ROUNDS_TOTAL) * 100)}%`;
  finalBestStreak.textContent = String(bestStreak);
}

function resetGameState() {
  score = 0;
  streak = 0;
  bestStreak = 0;
  roundIndex = 0;
  correctCount = 0;
  mastered.clear();
}

// ===== Camera + detection loop =====
async function setupHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
    });
  } catch (e) {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
    });
  }
}

async function setupCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

// Offscreen flipped canvas: mirrors cv2.flip(frame, 1) from the training
// pipeline so browser landmark coordinates match what the model was trained on.
const flipCanvas = document.createElement("canvas");
const flipCtx = flipCanvas.getContext("2d");

function detectLoop() {
  if (!running) return;

  flipCanvas.width = video.videoWidth;
  flipCanvas.height = video.videoHeight;
  flipCtx.save();
  flipCtx.translate(flipCanvas.width, 0);
  flipCtx.scale(-1, 1);
  flipCtx.drawImage(video, 0, 0, flipCanvas.width, flipCanvas.height);
  flipCtx.restore();

  const result = handLandmarker.detectForVideo(flipCanvas, performance.now());

  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.drawImage(flipCanvas, 0, 0, overlay.width, overlay.height);

  if (result.landmarks && result.landmarks.length > 0) {
    noHandBanner.hidden = true;
    const landmarks = result.landmarks[0];
    drawSkeleton(landmarks, overlay.width, overlay.height);

    const now = performance.now();
    if (!sendInFlight && now - lastSentAt >= SEND_INTERVAL_MS) {
      lastSentAt = now;
      sendInFlight = true;
      const features = normalizeLandmarks(landmarks);
      requestPrediction(features).then((resp) => {
        sendInFlight = false;
        if (resp && resp.letter) {
          pushPrediction(resp.letter, resp.confidence);
        }
      });
    }
  } else {
    noHandBanner.hidden = false;
  }

  const smoothed = smoothedPrediction();
  predictedLetterEl.textContent = smoothed || "—";

  if (roundActive) {
    const elapsed = performance.now() - roundStartTime;
    const remainingPct = Math.max(0, 100 - (elapsed / ROUND_TIME_MS) * 100);
    roundTimerBar.style.width = `${remainingPct}%`;

    if (smoothed && smoothed === currentTarget) {
      holdMs += 1000 / 30; // approx per-frame delta at ~30fps detection loop
      holdBar.style.width = `${Math.min(100, (holdMs / HOLD_MS) * 100)}%`;
      if (holdMs >= HOLD_MS) {
        endRound(true);
      }
    } else {
      holdMs = Math.max(0, holdMs - 1000 / 15);
      holdBar.style.width = `${Math.min(100, (holdMs / HOLD_MS) * 100)}%`;
    }

    if (elapsed >= ROUND_TIME_MS) {
      endRound(false);
    }
  }

  requestAnimationFrame(detectLoop);
}

// ===== Flow control =====
async function startGame() {
  ensureAudio();
  startBtn.disabled = true;
  startBtn.textContent = "Loading…";
  try {
    if (!handLandmarker) await setupHandLandmarker();
    if (!stream) await setupCamera();
  } catch (err) {
    alert("Camera or model failed to load: " + err.message);
    startBtn.disabled = false;
    startBtn.textContent = "Start Game";
    return;
  }

  resetGameState();
  startScreen.hidden = true;
  endScreen.hidden = true;
  gameScreen.hidden = false;
  statsEl.hidden = false;
  updateStatsUI();

  running = true;
  startRound();
  requestAnimationFrame(detectLoop);
}

startBtn.addEventListener("click", startGame);
playAgainBtn.addEventListener("click", startGame);

connectWs();
