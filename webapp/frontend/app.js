import { resetMotionDetector, updateMotionDetector, } from "./js/motionDetector.js";
import { LETTERS, LETTER_CUES, ROUNDS_TOTAL, SEND_INTERVAL_MS, HISTORY_LEN, DIFFICULTIES, XP_CORRECT, XP_MISS, unlockedThemes, CORRECT_ADVANCE_DELAY_MS, MISS_ADVANCE_DELAY_MS } from "./js/config.js";
import {
  loadProfile, saveProfile, getLevelInfo, addXp,
  recordRoundResult, recordCombo, recordGameEnd, markLetterMastered,
} from "./js/profile.js";
import { checkAchievements, showAchievementToasts } from "./js/achievements.js";
import { createComboState, registerCorrect, registerMiss, maybeShowComboPopup } from "./js/combo.js";
import { ensureAudio, setVolume, setMuted, playCorrectSound, playMissSound, playTick } from "./js/audio.js";
import { burstConfetti, glowPulse } from "./js/confetti.js";
import {
  setupHandLandmarker, setupCamera, getFlippedFrame, detectHands, drawSkeleton, normalizeLandmarks, drawGhostHand,
} from "./js/handTracking.js";
import { connectWs, requestPrediction, createSmoother } from "./js/websocketClient.js";
import { loadReferences, coachFor } from "./js/coaching.js";
import { createCircularTimer } from "./js/timer.js";
import { renderEndReport, renderStatsScreen } from "./js/statsView.js";
import { isMotionLetter, motionLetterFrame, MOTION_LETTER_BASE, LOOP_MS as GHOST_LOOP_MS } from "./js/ghostAnimation.js";

// ===== DOM =====
const screens = {
  start: document.getElementById("startScreen"),
  game: document.getElementById("gameScreen"),
  end: document.getElementById("endScreen"),
  stats: document.getElementById("statsScreen"),
  settings: document.getElementById("settingsScreen"),
  learnComplete: document.getElementById("learnCompleteScreen"),
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
const learnModeBtn = document.getElementById("learnModeBtn");
const modeBadge = document.getElementById("modeBadge");
const targetLabelEl = document.getElementById("targetLabel");
const wordProgressEl = document.getElementById("wordProgress");
const learnControls = document.getElementById("learnControls");
const learnHowTo = document.getElementById("learnHowTo");
const learnHowToText = document.getElementById("learnHowToText");
const learnPrevBtn = document.getElementById("learnPrevBtn");
const learnSkipBtn = document.getElementById("learnSkipBtn");
const learnProgressLabel = document.getElementById("learnProgressLabel");
const learnCompleteToLettersBtn = document.getElementById("learnCompleteToLettersBtn");
const learnCompleteToMenuBtn = document.getElementById("learnCompleteToMenuBtn");

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
const ghostHandCanvas = document.getElementById("ghostHandCanvas");
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
  stopGhostAnimation();
  predictedLetterEl.textContent = "—";
  confidenceValEl.textContent = "";
  wordProgressEl.hidden = true;
  wordProgressEl.textContent = "";
  currentWord = "";
  currentWordTarget = "";
  lastAcceptedLetter = null;
  learnControls.hidden = true;
  learnIndex = 0;
  gameMode = null;
  updateModeBadge(null);
  completedLearnLetters.clear();
  learnHowTo.hidden = true;
  learnHowToText.textContent = "";
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
let lastLetterAcceptedAt = 0;
const REPEATED_LETTER_DELAY_MS = 1000;
let lastTickSecond = null;
const mastered = new Set();
const achievementsEarnedThisSession = [];
let sumResponseMs = 0;
let countResponses = 0;
let fastestResponseMs = null;
let fastestLetter = null;
let xpAtGameStart = 0;

let gameMode = null;
let learnIndex = 0;
// Letters the user has successfully demonstrated during this Learn Mode session.
const completedLearnLetters = new Set();
const LEARN_INSTRUCTIONS = {
  A: "Make a fist with your thumb resting against the side of your index finger.",
  B: "Hold all four fingers straight together and fold your thumb across your palm.",
  C: "Curve your fingers and thumb to form the shape of the letter C.",
  D: "Touch your thumb to your middle finger while keeping your index finger pointing upward.",
  E: "Curl all four fingers downward and tuck your thumb underneath them.",
  F: "Touch your thumb and index finger together. Keep the other three fingers raised.",
  G: "Point your index finger and thumb sideways, keeping them parallel.",
  H: "Extend your index and middle fingers together sideways.",
  I: "Make a fist and raise only your little finger.",
  J: "Start with the sign for I, then draw a J shape in the air using your little finger.",
  K: "Raise your index and middle fingers. Place your thumb between them.",
  L: "Extend your thumb and index finger to form an L shape.",
  M: "Place your thumb underneath your first three fingers.",
  N: "Place your thumb underneath your index and middle fingers.",
  O: "Curve all fingers and your thumb together to create an O shape.",
  P: "Use the K handshape, then point it downward.",
  Q: "Use the G handshape, then point it downward.",
  R: "Cross your index and middle fingers while keeping the other fingers folded.",
  S: "Make a fist with your thumb folded across the front of your fingers.",
  T: "Make a fist with your thumb placed between your index and middle fingers.",
  U: "Raise your index and middle fingers together, keeping them straight.",
  V: "Raise your index and middle fingers apart to form a V.",
  W: "Raise your index, middle and ring fingers while keeping the others folded.",
  X: "Raise your index finger and bend it into a hook shape.",
  Y: "Extend your thumb and little finger while keeping the middle fingers folded.",
  Z: "Use your index finger to draw a Z shape in the air."
};

function showLearnInstructions(letter) {
  if (gameMode !== "learn") {
    learnHowTo.hidden = true;
    learnHowToText.textContent = "";
    return;
  }

  learnHowTo.hidden = false;

  const instruction =
    LEARN_INSTRUCTIONS[letter] ||
    LETTER_CUES[letter] ||
    "Copy the reference handshape and hold it steadily.";

  learnHowToText.innerHTML =
    `<strong>${letter}:</strong> ${instruction}`;
}
const WORDS = {
    // Includes a couple of J/Z words per tier now that motion detection
    // supports those letters, so Word Mode can exercise them too instead
    // of only ever appearing in Letter Mode/Learn Mode.
    easy: [
        "cat", "dog", "hat", "pen", "cup",
        "sun", "map", "fish", "book", "tree",
        "ball", "milk", "shoe", "bird", "star",
        "jam", "zip"
    ],

    normal: [
        "apple", "banana", "school", "orange", "friend",
        "camera", "window", "teacher", "laptop", "garden",
        "bottle", "rabbit", "pencil", "guitar", "library",
        "jacket", "zebra"
    ],

    hard: [
        "elephant", "beautiful", "adventure", "technology",
        "development", "communication", "environment",
        "information", "responsibility", "university",
        "recognition", "application", "performance",
        "programming", "artificial",
        "journey", "citizen"
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
  } else if (mode === "learn") {
    modeBadge.textContent = `Learn Mode · ${LETTERS.length} letters`;
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

  learnModeBtn?.classList.toggle(
    "selected",
    mode === "learn"
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

  if (gameMode === "words") return;

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
  const isLearn = gameMode === "learn";
  const scoreStat = scoreVal.closest(".stat");
  const comboStat = comboVal.closest(".stat");
  if (scoreStat) scoreStat.hidden = isLearn;
  if (comboStat) comboStat.hidden = isLearn;

  scoreVal.textContent = String(score);
  comboVal.textContent = String(comboState.combo);

  if (isLearn) {
    roundVal.textContent = `${Math.min(learnIndex + 1, LETTERS.length)}/${LETTERS.length}`;
  } else {
    const totalRounds = currentRoundsTotal();
    roundVal.textContent = `${Math.min(roundIndex, totalRounds)}/${totalRounds}`;
  }
}

let ghostAnimHandle = null;

function stopGhostAnimation() {
  if (ghostAnimHandle != null) {
    cancelAnimationFrame(ghostAnimHandle);
    ghostAnimHandle = null;
  }
}

function showReferenceFor(letter) {
  referencePanel.hidden = false;
  referenceCue.textContent = LETTER_CUES[letter] || "";
  referenceImg.hidden = false;
  // Public-domain (CC0) handshape diagrams from Wikimedia Commons -- see
  // assets/letters/README.md for source/license details.
  referenceImg.src = `assets/letters/${letter}.svg`;

  stopGhostAnimation();
  ghostHandCanvas.hidden = true; // only shown where it adds something the image can't (below)
  const handForDiagram = lastHandLabel || "right";
  const motion = isMotionLetter(letter);

  function playGhostHand() {
    ghostHandCanvas.hidden = false;
    if (motion) {
      // J and Z have no static pose -- animate the fingertip through the
      // actual motion path so the player can watch and copy the movement,
      // something the (still-shown) static reference image can't convey.
      const baseVector = letterReferences?.[MOTION_LETTER_BASE[letter]]?.[handForDiagram];
      if (!baseVector) {
        ghostHandCanvas.hidden = true;
        return;
      }
      const startedAt = performance.now();
      const tick = () => {
        const t = ((performance.now() - startedAt) % GHOST_LOOP_MS) / GHOST_LOOP_MS;
        drawGhostHand(ghostHandCanvas, motionLetterFrame(letter, baseVector, t));
        ghostAnimHandle = requestAnimationFrame(tick);
      };
      tick();
    } else {
      drawGhostHand(ghostHandCanvas, letterReferences?.[letter]?.[handForDiagram]);
    }
  }

  referenceImg.onerror = () => {
    referenceImg.hidden = true;
    // Fallback for static letters only -- motion letters already always
    // show the animated ghost-hand below, image or no image.
    if (!motion) playGhostHand();
  };

  if (motion) {
    playGhostHand();
  }
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

function updateWordProgress() {
  if (gameMode !== "words") {
    wordProgressEl.hidden = true;
    return;
  }

  const completed = currentWord
    .split("")
    .map((letter) => `<span class="word-letter completed">${letter}</span>`)
    .join("");

  const activeIndex = currentWord.length;

  const remaining = currentWordTarget
    .slice(activeIndex)
    .split("")
    .map((letter, index) => {
      const className =
        index === 0 ? "word-letter active" : "word-letter";

      return `<span class="${className}">${letter}</span>`;
    })
    .join("");

  wordProgressEl.innerHTML = completed + remaining;
}

function acceptWordLetter(letter) {
  if (gameMode !== "words" || !roundActive) {
    return;
  }

  const expectedLetter =
    currentWordTarget[currentWord.length];

  if (letter !== expectedLetter) {
    return;
  }

  const now = performance.now();

  // Prevent one held pose from being accepted repeatedly.
  if (
    lastAcceptedLetter === letter &&
    now - lastLetterAcceptedAt < REPEATED_LETTER_DELAY_MS
  ) {
    return;
  }

  currentWord += letter;
  lastAcceptedLetter = letter;
  lastLetterAcceptedAt = now;

  playCorrectSound();

  holdMs = 0;
  holdBar.style.width = "0%";

  updateWordProgress();

  // Whole word completed.
  if (currentWord === currentWordTarget) {
    holdBar.style.width = "100%";
    endRound(true);
    return;
  }

  // Move to the next expected letter.
  currentTarget =
    currentWordTarget[currentWord.length];

  resetMotionDetector(currentTarget);
  smoother.reset();

  predictedLetterEl.textContent = "—";
  confidenceValEl.textContent = "";
}

// Learn Mode never times out and never "misses" -- this is its only
// success path, kept separate from endRound() since none of the
// scoring/combo/XP/session-report logic there applies to untimed practice.
function completeLearnLetter() {
  if (!roundActive) return;
  roundActive = false;

  holdBar.style.width = "100%";

  completedLearnLetters.add(currentTarget);

  markLetterMastered(profile, currentTarget);
  mastered.add(currentTarget);
  saveProfile(profile);
  renderLevelBadge();
  renderMasteryRow();

  runAchievementCheck({ lastRoundWasCorrect: true });

  feedbackPanel.hidden = false;
  feedbackPanel.classList.add("correct");
  feedbackPanel.classList.remove("miss");
  feedbackTitle.textContent = "Nice! You've got it ✅";
  responseTimeValEl.textContent = "";
  poseSimilarityValEl.textContent = "";
  coachingHintEl.textContent = "";

  playCorrectSound();
  burstConfetti();
  glowPulse(document.querySelector(".camera-wrap"));

  setTimeout(() => advanceLearnMode(1), CORRECT_ADVANCE_DELAY_MS);
}

function showLearnCompleteScreen() {
  running = false;
  roundActive = false;
  showScreen("learnComplete");
}

function getFirstIncompleteLearnIndex() {
  return LETTERS.findIndex(
    (letter) => !completedLearnLetters.has(letter)
  );
}

function advanceLearnMode(direction) {
  const movingForward = direction > 0;
  const atLastLetter = learnIndex >= LETTERS.length - 1;

  if (movingForward && atLastLetter) {
    const completedAllLetters =
      completedLearnLetters.size === LETTERS.length;

    if (completedAllLetters) {
      showLearnCompleteScreen();
      return;
    }

    // Z was reached, but one or more letters were skipped.
    const firstIncompleteIndex = getFirstIncompleteLearnIndex();

    feedbackPanel.hidden = false;
    feedbackPanel.classList.remove("correct", "miss");
    feedbackTitle.textContent =
      "You still have letters left to complete";

    responseTimeValEl.textContent = "";
    poseSimilarityValEl.textContent = "";

    const remainingLetters = LETTERS.filter(
      (letter) => !completedLearnLetters.has(letter)
    );

    coachingHintEl.innerHTML =
      `Complete every letter before finishing Learn Mode. ` +
      `Remaining: <span class="learn-incomplete-message">` +
      `${remainingLetters.join(", ")}</span>`;

    // Return to the first letter that was skipped.
    setTimeout(() => {
      learnIndex =
        firstIncompleteIndex >= 0
          ? firstIncompleteIndex
          : 0;

      startRound();
      updateStatsUI();
    }, 1600);

    return;
  }

  learnIndex = Math.min(
    LETTERS.length - 1,
    Math.max(0, learnIndex + direction)
  );

  startRound();
  updateStatsUI();
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
  stopGhostAnimation();
  coachingHintEl.textContent = "";

  smoother.reset();
  predictedLetterEl.textContent = "—";
  confidenceValEl.textContent = "";
  circularTimer.reset();
  learnControls.hidden = true;
  learnHowTo.hidden = gameMode !== "learn";
  document.getElementById("gameScreen").classList.toggle("learn-active", gameMode === "learn");

  if (gameMode === "letters") {
    currentTarget = pickNextLetter();

    currentWord = "";
    currentWordTarget = "";
    lastAcceptedLetter = null;

    targetLabelEl.textContent = "Sign this letter";
    targetLetterEl.classList.remove("word-target");
    targetLetterEl.hidden = false;
    targetLetterEl.textContent = currentTarget;

    wordProgressEl.hidden = true;
    wordProgressEl.textContent = "";

    resetMotionDetector(currentTarget);
  } else if (gameMode === "learn") {
    currentTarget = LETTERS[learnIndex];

    currentWord = "";
    currentWordTarget = "";
    lastAcceptedLetter = null;

    targetLabelEl.textContent = "Sign this letter";
    targetLetterEl.classList.remove("word-target");
    targetLetterEl.hidden = false;
    targetLetterEl.textContent = currentTarget;

    wordProgressEl.hidden = true;
    wordProgressEl.textContent = "";

    learnControls.hidden = false;
    learnPrevBtn.disabled = learnIndex === 0;
    const completedCount = completedLearnLetters.size;

    learnProgressLabel.textContent =
      `Letter ${learnIndex + 1} of ${LETTERS.length} · ` +
      `${completedCount}/${LETTERS.length} completed`;

    showReferenceFor(currentTarget);
    showLearnInstructions(currentTarget);
    resetMotionDetector(currentTarget);
  } else {
    currentWord = "";
    currentWordTarget = pickNextWord();

    // The current target must be one letter, not the whole word.
    currentTarget = currentWordTarget[0];

    lastAcceptedLetter = null;
    lastLetterAcceptedAt = 0;

    targetLabelEl.textContent = "Spell this word";
    targetLetterEl.hidden = false;
    targetLetterEl.classList.add("word-target");
    targetLetterEl.textContent = currentWordTarget;

    wordProgressEl.hidden = false;
    updateWordProgress();

    resetMotionDetector(currentTarget);
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
    if (gameMode !== "words") {
      playCorrectSound();
    }

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

  completedLearnLetters.clear();
  learnIndex = 0;
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
      // Instantly complete the progress bar
      holdBar.style.width = "100%";

      if (gameMode === "words") {
      acceptWordLetter(motionPrediction);
      } else if (gameMode === "learn") {
        completeLearnLetter();
      } else {
        endRound(true);
      }
  }
    if (!sendInFlight && now - lastSentAt >= SEND_INTERVAL_MS) {
      lastSentAt = now;
      sendInFlight = true;
      const features = normalizeLandmarks(landmarks);
      lastUserVector = features;

      const handednessLabel = result.handedness?.[0]?.[0]?.categoryName;
      lastHandLabel = handednessLabel ? handednessLabel.toLowerCase() : null;
      const handValue = lastHandLabel === "right" ? 1 : 0;
      const payload = [handValue, ...features];

      requestPrediction(payload, gameMode === "words" ? "words" : "letters").then((resp) => {
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
    const roundTimeMs = currentRoundTimeMs();
    const elapsed = performance.now() - roundStartTime;
    const remainingMs = Math.max(0, roundTimeMs - elapsed);
    circularTimer.update(remainingMs, roundTimeMs);

    const remainingSec = Math.ceil(remainingMs / 1000);
    if (remainingMs <= 3000 && remainingMs > 0 && remainingSec !== lastTickSecond && gameMode !== "learn") {
      lastTickSecond = remainingSec;
      playTick();
    }

    if (
      finalPrediction &&
      finalPrediction === currentTarget
    ) {

      holdMs += 1000 / 30;

      holdBar.style.width =
        `${Math.min(
          100,
          (holdMs / diff.holdMs) * 100
        )}%`;

      if (holdMs >= diff.holdMs) {

        if (gameMode === "words") {

          acceptWordLetter(finalPrediction);

        } else if (gameMode === "learn") {

          completeLearnLetter();

        } else {

          endRound(true);

        }

      }

    } else {

      if (
        finalPrediction &&
        finalPrediction !== currentTarget
      ) {
        roundHadWrongPrediction = true;
      }

      holdMs = Math.max(
        0,
        holdMs - 1000 / 15
      );

      holdBar.style.width =
        `${Math.min(
          100,
          (holdMs / diff.holdMs) * 100
        )}%`;

    }

    if (elapsed >= roundTimeMs && gameMode !== "learn") {
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

learnModeBtn?.addEventListener("click", async () => {
  gameMode = "learn";
  learnIndex = 0;
  updateModeBadge(gameMode);
  await startGame();
});

learnPrevBtn?.addEventListener("click", () => advanceLearnMode(-1));
learnSkipBtn?.addEventListener("click", () => advanceLearnMode(1));
learnCompleteToMenuBtn?.addEventListener("click", goToMainMenu);
learnCompleteToLettersBtn?.addEventListener("click", async () => {
  gameMode = "letters";
  updateModeBadge(gameMode);
  await startGame();
});

playAgainBtn.addEventListener("click", startGame);

updateModeBadge(null);
applySettingsToUI();
renderLevelBadge();