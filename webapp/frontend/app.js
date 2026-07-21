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

const playAgainBtn = document.getElementById("playAgainBtn");
const backToMenuBtn = document.getElementById("backToMenuBtn");
const backToMenuFromEndBtn = document.getElementById("backToMenuFromEndBtn");
const viewStatsFromEndBtn = document.getElementById("viewStatsFromEndBtn");
const backFromStatsBtn = document.getElementById("backFromStatsBtn");
const backFromSettingsBtn = document.getElementById("backFromSettingsBtn");
const statsBtn = document.getElementById("statsBtn");
const settingsBtn = document.getElementById("settingsBtn");
const muteBtn = document.getElementById("muteBtn");
const modeToggleBtn = document.getElementById("modeToggleBtn");
const lettersModeBtn = document.getElementById("lettersModeBtn");
const wordsModeBtn = document.getElementById("wordsModeBtn");
const modeBadge = document.getElementById("modeBadge");
const targetLabelEl = document.getElementById("targetLabel");
const wordProgressEl = document.getElementById("wordProgress");

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
const WORD_EXTRA_TIME_MS = 8000; 

// ===== Profile / settings =====
let profile = loadProfile();

function applyMode(mode) {
  document.body.dataset.mode = mode;
  modeToggleBtn.textContent = mode === "light" ? "☀️" : "🌙";
  const modeSelect = document.getElementById("modeSelect");
  if (modeSelect) modeSelect.value = mode;
}
function goToMainMenu() {
  running = false;
  roundActive = false;
  holdMs = 0;
  holdBar.style.width = "0%";
  circularTimer.reset();
  feedbackPanel.hidden = true;
  predictedLetterEl.textContent = "—";
  confidenceValEl.textContent = "";
  wordProgressEl.hidden = true;
  wordProgressEl.textContent = "";
  currentWord = "";
  currentWordTarget = "";
  lastAcceptedLetter = null;
  gameMode = null;
  updateModeBadge(null);
  showScreen("start");
}

backToMenuBtn?.addEventListener("click", goToMainMenu);
backToMenuFromEndBtn?.addEventListener("click", goToMainMenu);

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

const LETTER_ROUNDS_TOTAL = ROUNDS_TOTAL;
const WORD_ROUNDS_TOTAL = 5;

function currentRoundsTotal() {
  return gameMode === "words" ? WORD_ROUNDS_TOTAL : LETTER_ROUNDS_TOTAL;
}

function currentRoundTimeMs() {
  const base = currentDifficulty().roundTimeMs;
  return gameMode === "words" ? base + WORD_EXTRA_TIME_MS : base;
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
loadReferences().then((refs) => {
  letterReferences = refs;
});

// ===== Game state =====
let handLandmarker = null;
let stream = null;
let running = false;
let sendInFlight = false;
let lastSentAt = 0;
let lastConfidence = null;
let lastUserVector = null;
let lastHandLabel = null;

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
let lastLetterAcceptedAt = 0;
const REPEATED_LETTER_DELAY_MS = 700;
let lastTickSecond = null;
const mastered = new Set();
const achievementsEarnedThisSession = [];
let sumResponseMs = 0;
let countResponses = 0;
let fastestResponseMs = null;
let fastestLetter = null;
let xpAtGameStart = 0;

let gameMode = null;
const WORDS = {
    easy: [
        "cat", "dog", "hat", "pen", "cup",
        "sun", "map", "fish", "book", "tree",
        "ball", "milk", "shoe", "bird", "star"
    ],

    normal: [
        "apple", "banana", "school", "orange", "friend",
        "camera", "window", "teacher", "laptop", "garden",
        "bottle", "rabbit", "pencil", "guitar", "library"
    ],

    hard: [
        "elephant", "beautiful", "adventure", "technology",
        "development", "communication", "environment",
        "information", "responsibility", "university",
        "recognition", "application", "performance",
        "programming", "artificial"
    ]
};
let currentWord = "";
let currentWordTarget = "";
let lastAcceptedLetter = null;
let roundWords = [];
let currentRound = 0;


function getWordsForCurrentDifficulty() {
  const difficulty = String(
    profile.settings.difficulty || "easy"
  ).toLowerCase();

  return WORDS[difficulty] || WORDS.easy;
}

function prepareRoundWords() {
  const availableWords = getWordsForCurrentDifficulty()
    .map((word) => word.toUpperCase());

  roundWords = [...availableWords]
    .sort(() => Math.random() - 0.5)
    .slice(0, WORD_ROUNDS_TOTAL);
}

function pickNextWord() {
  if (!roundWords.length) {
    prepareRoundWords();
  }

  return roundWords[roundIndex] || roundWords[0];
}


function updateModeBadge(mode = null) {
  if (mode === "letters") {
    modeBadge.textContent = `Letter Mode · ${LETTER_ROUNDS_TOTAL} rounds`;
  } else if (mode === "words") {
    modeBadge.textContent = `Word Mode · ${WORD_ROUNDS_TOTAL} rounds`;
  } else {
    modeBadge.textContent = "Choose a mode";
  }

  lettersModeBtn?.classList.toggle(
    "selected",
    mode === "letters"
  );

  wordsModeBtn?.classList.toggle(
    "selected",
    mode === "words"
  );
}

function displayCurrentLetter(letter) {
  const targetLabel = document.getElementById("targetLabel");
  const targetLetter = document.getElementById("targetLetter");
  const wordProgress = document.getElementById("wordProgress");

  targetLabel.textContent = "Sign this letter";

  targetLetter.hidden = false;
  wordProgress.hidden = true;

  targetLetter.textContent = letter.toUpperCase();
}

function pickNextLetter() {
  let letter;
  do {
    letter = LETTERS[Math.floor(Math.random() * LETTERS.length)];
  } while (letter === currentTarget && LETTERS.length > 1);
  return letter;
}



function renderMasteryRow() {
  masteryRow.innerHTML = "";

  if (gameMode !== "letters") return;

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
  const totalRounds = currentRoundsTotal();
  roundVal.textContent = `${Math.min(roundIndex, totalRounds)}/${totalRounds}`;
}

function showReferenceFor(letter) {
  referencePanel.hidden = false;
  referenceCue.textContent = LETTER_CUES[letter] || "";
  referenceImg.hidden = false;
  referenceImg.src = `assets/letters/${letter}.png`;
  referenceImg.onerror = () => {
    referenceImg.hidden = true;
  };
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

  if (gameMode === "letters") {
    currentTarget = pickNextLetter();
    currentWord = "";
    currentWordTarget = "";
    lastAcceptedLetter = null;
    targetLabelEl.textContent = "Sign this letter";
    targetLetterEl.textContent = currentTarget;
    wordProgressEl.hidden = true;
    wordProgressEl.textContent = "";
  } else {
    currentWord = "";
    currentWordTarget = pickNextWord();
    currentTarget = currentWordTarget;
    lastAcceptedLetter = null;
    lastLetterAcceptedAt = 0;

    targetLabelEl.textContent = "Sign this word";
    targetLetterEl.hidden = false;
    targetLetterEl.textContent = currentWordTarget;

    wordProgressEl.hidden = false;
    wordProgressEl.textContent = "Current: —";
  }

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

    if (gameMode === "letters") {
      mastered.add(currentTarget);
      if (fastestResponseMs == null || responseMs < fastestResponseMs) {
        fastestResponseMs = responseMs;
        fastestLetter = currentTarget;
      }
    }

    sumResponseMs += responseMs;
    countResponses++;

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

    if (gameMode === "letters") {
      feedbackTitle.textContent = "Not quite — here's the correct handshape:";
      showReferenceFor(currentTarget);

      const reference = letterReferences && lastHandLabel
        ? letterReferences[currentTarget]?.[lastHandLabel]
        : null;
      if (reference && lastUserVector) {
        const { hints, similarity } = coachFor(lastUserVector, reference);
        coachingHintEl.textContent = hints.length ? hints[0] : "Keep practicing — you're close!";
        poseSimilarityValEl.textContent += (poseSimilarityValEl.textContent ? " · " : "") + `Pose match: ${similarity}%`;
      }
    } else {
      feedbackTitle.textContent = `Missed word: ${currentWordTarget}`;
      coachingHintEl.textContent = "Try signing each letter clearly and steadily.";
      referencePanel.hidden = true;
    }

    addXp(profile, XP_MISS);
    playMissSound();
    document.querySelector(".camera-wrap").classList.add("shake");
    setTimeout(() => document.querySelector(".camera-wrap").classList.remove("shake"), 400);
  }

  recordCombo(profile, comboState.combo);

  if (gameMode === "letters") {
    recordRoundResult(profile, { letter: currentTarget, correct: wasCorrect, responseMs });
  }

  runAchievementCheck({ lastResponseMs: responseMs, lastRoundWasCorrect: wasCorrect });
  saveProfile(profile);
  renderLevelBadge();
  updateStatsUI();
  renderMasteryRow();

  const advanceDelay = wasCorrect ? CORRECT_ADVANCE_DELAY_MS : MISS_ADVANCE_DELAY_MS;
  if (roundIndex >= currentRoundsTotal()) {
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
  currentWord = "";
  currentWordTarget = "";
  lastAcceptedLetter = null;
  roundWords = [];

    if (gameMode === "words") {
      prepareRoundWords();
    }
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
    if (!sendInFlight && now - lastSentAt >= SEND_INTERVAL_MS) {
      lastSentAt = now;
      sendInFlight = true;
      const features = normalizeLandmarks(landmarks);
      lastUserVector = features;

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

  const smoothed = smoother.smoothed();
  predictedLetterEl.textContent = smoothed || "—";
  confidenceValEl.textContent = smoothed && lastConfidence != null ? `${Math.round(lastConfidence * 100)}%` : "";

  if (roundActive) {
    const diff = currentDifficulty();
    const roundTimeMs = currentRoundTimeMs();
    const elapsed = performance.now() - roundStartTime;
    const remainingMs = Math.max(0, roundTimeMs - elapsed);
    circularTimer.update(remainingMs, roundTimeMs);

    const remainingSec = Math.ceil(remainingMs / 1000);
    if (remainingMs <= 3000 && remainingMs > 0 && remainingSec !== lastTickSecond) {
      lastTickSecond = remainingSec;
      playTick();
    }

    if (gameMode === "letters") {
      if (smoothed && smoothed === currentTarget) {
        holdMs += 1000 / 30;
        holdBar.style.width = `${Math.min(100, (holdMs / diff.holdMs) * 100)}%`;
        if (holdMs >= diff.holdMs) {
          endRound(true);
        }
      } else {
        if (smoothed && smoothed !== currentTarget) roundHadWrongPrediction = true;
        holdMs = Math.max(0, holdMs - 1000 / 15);
        holdBar.style.width = `${Math.min(100, (holdMs / diff.holdMs) * 100)}%`;
      }
    } else {
      const nextNeeded = currentWordTarget[currentWord.length];

      if (smoothed && smoothed === nextNeeded) {
         const now = performance.now();

         const repeatedLetter =
            lastAcceptedLetter === smoothed;

         const repeatDelayFinished =
            now - lastLetterAcceptedAt >= REPEATED_LETTER_DELAY_MS;

         if (!repeatedLetter || repeatDelayFinished) {
            holdMs += 1000 / 30;
            holdBar.style.width = `${Math.min(100, (holdMs / diff.holdMs) * 100)}%`;

            if (holdMs >= diff.holdMs) {
                currentWord += smoothed;
                lastAcceptedLetter = smoothed;
                lastLetterAcceptedAt = now;

                holdMs = 0;
                holdBar.style.width = "0%";
                wordProgressEl.textContent = `Current: ${currentWord}`;

                if (currentWord === currentWordTarget) {
                    endRound(true);
                }
            }
         }
      } else {
          if (smoothed && smoothed !== nextNeeded)
            roundHadWrongPrediction = true;

          if (smoothed !== lastAcceptedLetter)
            lastAcceptedLetter = null;

          holdMs = Math.max(0, holdMs - 1000 / 15);
          holdBar.style.width = `${Math.min(100, (holdMs / diff.holdMs) * 100)}%`;
      } 
    }

    if (elapsed >= roundTimeMs) {
      endRound(false);
    }
  }

  requestAnimationFrame(detectLoop);
}

// ===== Flow control =====
async function startGame() {
  if (!gameMode) return;

  ensureAudio();

  try {
    if (!handLandmarker) handLandmarker = await setupHandLandmarker();
    if (!stream) {
      stream = await setupCamera(video);
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
    }
  } catch (err) {
    alert("Camera or model failed to load: " + err.message);
    return;
  }

  smoother.setThreshold(profile.settings.sensitivity);
  resetGameState();
  showScreen("game");
  updateStatsUI();

  running = true;
  startRound();
  requestAnimationFrame(detectLoop);
}

lettersModeBtn?.addEventListener("click", async () => {
  gameMode = "letters";
  updateModeBadge(gameMode);
  await startGame();
});

wordsModeBtn?.addEventListener("click", async () => {
  gameMode = "words";
  updateModeBadge(gameMode);
  await startGame();
});

playAgainBtn.addEventListener("click", startGame);

updateModeBadge(null);
applySettingsToUI();
renderLevelBadge();