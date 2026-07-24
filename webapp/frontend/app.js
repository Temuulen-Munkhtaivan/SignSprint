import { resetMotionDetector, updateMotionDetector, } from "./js/motionDetector.js";
import { LETTERS, LETTER_CUES, ROUNDS_TOTAL, SEND_INTERVAL_MS, HISTORY_LEN, DIFFICULTIES, XP_CORRECT, XP_MISS, unlockedThemes, CORRECT_ADVANCE_DELAY_MS, MISS_ADVANCE_DELAY_MS } from "./js/config.js";
import {
  loadProfile, saveProfile, getLevelInfo, addXp,
  recordRoundResult, recordCombo, recordGameEnd,
} from "./js/profile.js";
import { checkAchievements, showAchievementToasts } from "./js/achievements.js";
import { createComboState, registerCorrect, registerMiss, maybeShowComboPopup } from "./js/combo.js";
import { ensureAudio, setVolume, setMuted, playCorrectSound, playMissSound, playTick } from "./js/audio.js";
import { burstConfetti, glowPulse } from "./js/confetti.js";
import {
  setupHandLandmarker, setupCamera, getFlippedFrame, detectHands, drawSkeleton, normalizeLandmarks,
} from "./js/handTracking.js";
import { connectWs, requestPrediction, createSmoother } from "./js/websocketClient.js";
import { loadReferences, coachFor } from "./js/coaching.js";
import { createCircularTimer } from "./js/timer.js";
import { renderEndReport, renderStatsScreen } from "./js/statsView.js";

// ===== DOM =====
const screens = {
  start: document.getElementById("startScreen"),
  game: document.getElementById("gameScreen"),
  end: document.getElementById("endScreen"),
  stats: document.getElementById("statsScreen"),
  settings: document.getElementById("settingsScreen"),
};
let previousScreen = "start";

function showScreen(name) {
  for (const [key, el] of Object.entries(screens)) el.hidden = key !== name;
  document.getElementById("stats").hidden = name !== "game";
}

const startBtn = document.getElementById("startBtn");
const playAgainBtn = document.getElementById("playAgainBtn");
const viewStatsFromEndBtn = document.getElementById("viewStatsFromEndBtn");
const backFromStatsBtn = document.getElementById("backFromStatsBtn");
const backFromSettingsBtn = document.getElementById("backFromSettingsBtn");
const statsBtn = document.getElementById("statsBtn");
const settingsBtn = document.getElementById("settingsBtn");
const muteBtn = document.getElementById("muteBtn");
const modeToggleBtn = document.getElementById("modeToggleBtn");

const scoreVal = document.getElementById("scoreVal");
const comboVal = document.getElementById("comboVal");
const roundVal = document.getElementById("roundVal");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const noHandBanner = document.getElementById("noHandBanner");
const targetLetterEl = document.getElementById("targetLetter");
const predictedLetterEl = document.getElementById("predictedLetter");
const confidenceValEl = document.getElementById("confidenceVal");
const holdBar = document.getElementById("holdBar");
const feedbackPanel = document.getElementById("feedbackPanel");
const feedbackTitle = document.getElementById("feedbackTitle");
const responseTimeValEl = document.getElementById("responseTimeVal");
const poseSimilarityValEl = document.getElementById("poseSimilarityVal");
const coachingHintEl = document.getElementById("coachingHint");
const referencePanel = document.getElementById("referencePanel");
const referenceImg = document.getElementById("referenceImg");
const referenceCue = document.getElementById("referenceCue");
const masteryRow = document.getElementById("masteryRow");
const connectionBanner = document.getElementById("connectionBanner");
const levelNumEl = document.getElementById("levelNum");
const xpFillEl = document.getElementById("xpFill");

const timerCircleEl = document.getElementById("timerCircle");
const timerTextEl = document.getElementById("timerText");
const circularTimer = createCircularTimer(timerCircleEl, timerTextEl);

// ===== Profile / settings =====
let profile = loadProfile();

function applyMode(mode) {
  document.body.dataset.mode = mode;
  modeToggleBtn.textContent = mode === "light" ? "☀️" : "🌙";
  const modeSelect = document.getElementById("modeSelect");
  if (modeSelect) modeSelect.value = mode;
}

function applySettingsToUI() {
  document.body.classList.toggle("colorblind", profile.settings.colorblind);
  document.body.dataset.theme = profile.settings.theme;
  applyMode(profile.settings.mode);
  setVolume(profile.settings.volume);
  setMuted(profile.settings.muted);

  document.getElementById("difficultySelect").value = profile.settings.difficulty;
  document.getElementById("sensitivitySlider").value = profile.settings.sensitivity;
  document.getElementById("volumeSlider").value = profile.settings.volume;
  document.getElementById("muteCheckbox").checked = profile.settings.muted;
  document.getElementById("colorblindCheckbox").checked = profile.settings.colorblind;
  muteBtn.textContent = profile.settings.muted ? "🔇" : "🔊";

  const level = getLevelInfo(profile).level;
  const themeSelect = document.getElementById("themeSelect");
  const unlocked = unlockedThemes(level);
  themeSelect.innerHTML = unlocked.map((t) => `<option value="${t.id}">${t.label}</option>`).join("");
  themeSelect.value = profile.settings.theme;
  document.getElementById("themeUnlockHint").textContent = `(${unlocked.length} unlocked)`;
}

function renderLevelBadge() {
  const level = getLevelInfo(profile);
  levelNumEl.textContent = `Lv ${level.level}`;
  xpFillEl.style.width = `${Math.round((level.xpIntoLevel / level.xpForNext) * 100)}%`;
}

function currentDifficulty() {
  return DIFFICULTIES[profile.settings.difficulty] || DIFFICULTIES.normal;
}

// ===== Settings screen wiring =====
document.getElementById("difficultySelect").addEventListener("change", (e) => {
  profile.settings.difficulty = e.target.value;
  saveProfile(profile);
});
document.getElementById("sensitivitySlider").addEventListener("input", (e) => {
  profile.settings.sensitivity = parseFloat(e.target.value);
  smoother.setThreshold(profile.settings.sensitivity);
  saveProfile(profile);
});
document.getElementById("volumeSlider").addEventListener("input", (e) => {
  profile.settings.volume = parseFloat(e.target.value);
  setVolume(profile.settings.volume);
  saveProfile(profile);
});
document.getElementById("muteCheckbox").addEventListener("change", (e) => {
  profile.settings.muted = e.target.checked;
  setMuted(profile.settings.muted);
  muteBtn.textContent = profile.settings.muted ? "🔇" : "🔊";
  saveProfile(profile);
});
document.getElementById("colorblindCheckbox").addEventListener("change", (e) => {
  profile.settings.colorblind = e.target.checked;
  document.body.classList.toggle("colorblind", profile.settings.colorblind);
  saveProfile(profile);
});
document.getElementById("themeSelect").addEventListener("change", (e) => {
  profile.settings.theme = e.target.value;
  document.body.dataset.theme = profile.settings.theme;
  saveProfile(profile);
});
document.getElementById("modeSelect").addEventListener("change", (e) => {
  profile.settings.mode = e.target.value;
  applyMode(profile.settings.mode);
  saveProfile(profile);
});
modeToggleBtn.addEventListener("click", () => {
  profile.settings.mode = profile.settings.mode === "light" ? "dark" : "light";
  applyMode(profile.settings.mode);
  saveProfile(profile);
});
muteBtn.addEventListener("click", () => {
  profile.settings.muted = !profile.settings.muted;
  setMuted(profile.settings.muted);
  document.getElementById("muteCheckbox").checked = profile.settings.muted;
  muteBtn.textContent = profile.settings.muted ? "🔇" : "🔊";
  saveProfile(profile);
});

statsBtn.addEventListener("click", () => {
  previousScreen = running ? "game" : "start";
  renderStatsScreen(profile);
  showScreen("stats");
});
settingsBtn.addEventListener("click", () => {
  previousScreen = running ? "game" : "start";
  applySettingsToUI();
  showScreen("settings");
});
backFromStatsBtn.addEventListener("click", () => showScreen(previousScreen));
backFromSettingsBtn.addEventListener("click", () => showScreen(previousScreen));
viewStatsFromEndBtn.addEventListener("click", () => {
  previousScreen = "end";
  renderStatsScreen(profile);
  showScreen("stats");
});

// ===== WebSocket + smoother =====
const smoother = createSmoother(profile.settings.sensitivity, HISTORY_LEN);
connectWs((connected) => {
  connectionBanner.hidden = connected;
  if (!connected) connectionBanner.textContent = "Connection lost — reconnecting…";
});

let letterReferences = null;
loadReferences().then((refs) => (letterReferences = refs));

// ===== Game state =====
let handLandmarker = null;
let stream = null;
let running = false;
let sendInFlight = false;
let lastSentAt = 0;
let lastConfidence = null;
let lastUserVector = null;
let lastHandLabel = null; // "left" | "right", matches ai_module's hand encoding
let motionCompletedAt = null;

let score = 0;
const comboState = createComboState();
let bestComboSession = 0;
let roundIndex = 0;
let currentTarget = null;
let holdMs = 0;
let roundStartTime = 0;
let roundActive = false;
let correctCount = 0;
let missCount = 0;
let roundHadWrongPrediction = false;
let lastTickSecond = null;
const mastered = new Set();
const achievementsEarnedThisSession = [];
let sumResponseMs = 0;
let countResponses = 0;
let fastestResponseMs = null;
let fastestLetter = null;
let xpAtGameStart = 0;

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
  comboVal.textContent = String(comboState.combo);
  roundVal.textContent = `${Math.min(roundIndex, ROUNDS_TOTAL)}/${ROUNDS_TOTAL}`;
}

function showReferenceFor(letter) {
  referencePanel.hidden = false;
  referenceCue.textContent = LETTER_CUES[letter] || "";
  referenceImg.hidden = false;
  referenceImg.src = `assets/letters/${letter}.png`;
  referenceImg.onerror = () => { referenceImg.hidden = true; };
}

function runAchievementCheck(extra = {}) {
  const ctx = {
    lifetimeCorrect: profile.stats.lifetimeCorrect,
    sessionStreak: comboState.combo,
    combo: comboState.combo,
    sessionEnded: false,
    sessionMisses: missCount,
    sessionRounds: roundIndex,
    lifetimeMasteredCount: profile.stats.masteredAllTime.length,
    lastResponseMs: null,
    roundHadWrongPrediction,
    lastRoundWasCorrect: null,
    ...extra,
  };
  const earned = checkAchievements(profile, ctx);
  if (earned.length) {
    achievementsEarnedThisSession.push(...earned);
    showAchievementToasts(earned);
    earned.forEach(() => addXp(profile, 25));
  }
}

function startRound() {
  currentTarget = pickNextLetter();
  resetMotionDetector(currentTarget);

  targetLetterEl.textContent = currentTarget;
  holdMs = 0;
  holdBar.style.width = "0%";
  roundStartTime = performance.now();
  roundActive = true;
  roundHadWrongPrediction = false;
  lastTickSecond = null;
  feedbackPanel.hidden = true;
  feedbackPanel.classList.remove("correct", "miss");
  referencePanel.hidden = true;
  coachingHintEl.textContent = "";
  smoother.reset();
  predictedLetterEl.textContent = "—";
  confidenceValEl.textContent = "";
  circularTimer.reset();
  renderMasteryRow();
}

function endRound(wasCorrect) {
  roundActive = false;
  roundIndex++;

  const responseMs = wasCorrect ? performance.now() - roundStartTime : null;
  const tier = wasCorrect ? registerCorrect(comboState) : (registerMiss(comboState), null);
  const multiplier = tier ? tier.multiplier : 1;
  bestComboSession = Math.max(bestComboSession, comboState.combo);

  feedbackPanel.hidden = false;
  responseTimeValEl.textContent = wasCorrect ? `⏱ ${(responseMs / 1000).toFixed(2)}s` : "";
  poseSimilarityValEl.textContent = lastConfidence != null ? `Confidence: ${Math.round(lastConfidence * 100)}%` : "";

  if (wasCorrect) {
    correctCount++;
    const pointsAwarded = Math.round(10 * multiplier);
    score += pointsAwarded;
    mastered.add(currentTarget);
    sumResponseMs += responseMs;
    countResponses++;
    if (fastestResponseMs == null || responseMs < fastestResponseMs) {
      fastestResponseMs = responseMs;
      fastestLetter = currentTarget;
    }

    feedbackPanel.classList.add("correct");
    feedbackPanel.classList.remove("miss");
    feedbackTitle.textContent = `Correct! +${pointsAwarded} pts`;
    addXp(profile, XP_CORRECT * multiplier);
    maybeShowComboPopup(comboState, tier, comboState.combo);
    playCorrectSound();
    burstConfetti();
    glowPulse(document.querySelector(".camera-wrap"));
  } else {
    missCount++;
    feedbackPanel.classList.add("miss");
    feedbackPanel.classList.remove("correct");
    feedbackTitle.textContent = "Not quite — here's the correct handshape:";
    showReferenceFor(currentTarget);
    addXp(profile, XP_MISS);
    playMissSound();
    document.querySelector(".camera-wrap").classList.add("shake");
    setTimeout(() => document.querySelector(".camera-wrap").classList.remove("shake"), 400);

    const reference = letterReferences && lastHandLabel
      ? letterReferences[currentTarget]?.[lastHandLabel]
      : null;
    if (reference && lastUserVector) {
      const { hints, similarity } = coachFor(lastUserVector, reference);
      coachingHintEl.textContent = hints.length ? hints[0] : "Keep practicing — you're close!";
      poseSimilarityValEl.textContent += (poseSimilarityValEl.textContent ? " · " : "") + `Pose match: ${similarity}%`;
    }
  }

  recordCombo(profile, comboState.combo);
  recordRoundResult(profile, { letter: currentTarget, correct: wasCorrect, responseMs });
  runAchievementCheck({ lastResponseMs: responseMs, lastRoundWasCorrect: wasCorrect });
  saveProfile(profile);
  renderLevelBadge();
  updateStatsUI();
  renderMasteryRow();

  const advanceDelay = wasCorrect ? CORRECT_ADVANCE_DELAY_MS : MISS_ADVANCE_DELAY_MS;
  if (roundIndex >= ROUNDS_TOTAL) {
    setTimeout(showEndScreen, advanceDelay);
  } else {
    setTimeout(startRound, advanceDelay);
  }
}

function showEndScreen() {
  running = false;
  recordGameEnd(profile, { score });
  runAchievementCheck({ sessionEnded: true });
  saveProfile(profile);
  renderLevelBadge();

  const session = {
    score,
    rounds: roundIndex,
    correct: correctCount,
    bestCombo: bestComboSession,
    fastestLetter,
    fastestResponseMs,
    avgResponseMs: countResponses > 0 ? sumResponseMs / countResponses : null,
    achievementsEarned: achievementsEarnedThisSession,
  };
  const xpGained = profile.xpTotal - xpAtGameStart;

  showScreen("end");
  renderEndReport(session, profile, xpGained);
}

function resetGameState() {
  score = 0;
  comboState.combo = 0;
  comboState.lastAnnouncedTier = null;
  bestComboSession = 0;
  roundIndex = 0;
  correctCount = 0;
  missCount = 0;
  mastered.clear();
  achievementsEarnedThisSession.length = 0;
  sumResponseMs = 0;
  countResponses = 0;
  fastestResponseMs = null;
  fastestLetter = null;
  xpAtGameStart = profile.xpTotal;
}

// ===== Camera + detection loop =====
function detectLoop() {
  if (!running) return;

  const flipCanvas = getFlippedFrame(video);
  const result = detectHands(flipCanvas);

  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.drawImage(flipCanvas, 0, 0, overlay.width, overlay.height);

  if (result.landmarks && result.landmarks.length > 0) {
    noHandBanner.hidden = true;
    const landmarks = result.landmarks[0];
    drawSkeleton(overlayCtx, landmarks, overlay.width, overlay.height);

    const now = performance.now();
    const motionPrediction = updateMotionDetector(
      landmarks,
      currentTarget,
      now
    );

    const isMotionTarget =
      currentTarget === "J" || currentTarget === "Z";

    if (
      isMotionTarget &&
      motionPrediction === currentTarget
    ) {
      console.log("Motion letter completed:", motionPrediction);

      // Instantly complete the progress bar
      holdBar.style.width = "100%";

      // J/Z movements are already complete,
      // so they do not need the static-letter hold timer.
      endRound(true);
    }
    if (!sendInFlight && now - lastSentAt >= SEND_INTERVAL_MS) {
      lastSentAt = now;
      sendInFlight = true;
      const features = normalizeLandmarks(landmarks);
      lastUserVector = features;

      // Model input is [hand, ...63 landmarks] -- hand is 0=left/1=right,
      // matching ai_module/training/train_model.py's df["hand"].map({"left":0,"right":1}).
      // detectHands() already runs on the flipped canvas (mirroring cv2.flip in the
      // training pipeline), so this handedness reads the same way Python's did.
      const handednessLabel = result.handedness?.[0]?.[0]?.categoryName;
      lastHandLabel = handednessLabel ? handednessLabel.toLowerCase() : null;
      const handValue = lastHandLabel === "right" ? 1 : 0;
      const payload = [handValue, ...features];

      requestPrediction(payload).then((resp) => {
        sendInFlight = false;
        if (resp && resp.letter) {
          lastConfidence = resp.confidence;
          smoother.push(resp.letter, resp.confidence);
        }
      });
    }
  } else {
    noHandBanner.hidden = false;
  }

  const staticPrediction = smoother.smoothed();

  const isMotionTarget =
    currentTarget === "J" || currentTarget === "Z";

  // J and Z are handled immediately above when their
  // complete movement is detected.
  const finalPrediction =
    isMotionTarget ? null : staticPrediction;

  predictedLetterEl.textContent =
  finalPrediction || "—";

  confidenceValEl.textContent =
    !isMotionTarget &&
    finalPrediction &&
    lastConfidence != null
      ? `${Math.round(lastConfidence * 100)}%`
      : "";

  if (roundActive) {
    const diff = currentDifficulty();
    const elapsed = performance.now() - roundStartTime;
    const remainingMs = Math.max(0, diff.roundTimeMs - elapsed);
    circularTimer.update(remainingMs, diff.roundTimeMs);

    const remainingSec = Math.ceil(remainingMs / 1000);
    if (remainingMs <= 3000 && remainingMs > 0 && remainingSec !== lastTickSecond) {
      lastTickSecond = remainingSec;
      playTick();
    }

    if (
      finalPrediction &&
      finalPrediction === currentTarget
    ) {
      const isMotionTarget =
        currentTarget === "J" || currentTarget === "Z";

      if (
        isMotionTarget &&
        motionPrediction === currentTarget
      ) {
        holdBar.style.width = "100%";
        endRound(true);
      }

      // Existing behaviour for static letters
      holdMs += 1000 / 30;

      holdBar.style.width =
        `${Math.min(100, (holdMs / diff.holdMs) * 100)}%`;

      if (holdMs >= diff.holdMs) {
        endRound(true);
      }
    }else {
      if (
        finalPrediction &&
        finalPrediction !== currentTarget
      ) {
        roundHadWrongPrediction = true;
      }
      holdMs = Math.max(0, holdMs - 1000 / 15);
      holdBar.style.width = `${Math.min(100, (holdMs / diff.holdMs) * 100)}%`;
    }

    if (elapsed >= diff.roundTimeMs) {
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
    if (!handLandmarker) handLandmarker = await setupHandLandmarker();
    if (!stream) {
      stream = await setupCamera(video);
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
    }
  } catch (err) {
    alert("Camera or model failed to load: " + err.message);
    startBtn.disabled = false;
    startBtn.textContent = "Start Game";
    return;
  }

  smoother.setThreshold(profile.settings.sensitivity);
  resetGameState();
  showScreen("game");
  updateStatsUI();

  running = true;
  startRound();
  requestAnimationFrame(detectLoop);

  startBtn.disabled = false;
  startBtn.textContent = "Start Game";
}

startBtn.addEventListener("click", startGame);
playAgainBtn.addEventListener("click", startGame);

applySettingsToUI();
renderLevelBadge();
