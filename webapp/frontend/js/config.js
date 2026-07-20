export const LETTERS = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"];

// Short ASL fingerspelling cues used as a fallback when no reference image is
// supplied at assets/letters/<LETTER>.png (see that folder's README).
export const LETTER_CUES = {
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

export const ROUNDS_TOTAL = 15;
export const SEND_INTERVAL_MS = 100; // ~10fps to the backend
export const HISTORY_LEN = 8;

// Pause before advancing to the next round, so there's time to actually read
// the feedback. Misses get much longer: there's a coaching hint + reference
// handshape to read, not just a "Correct!" flash.
export const CORRECT_ADVANCE_DELAY_MS = 1300;
export const MISS_ADVANCE_DELAY_MS = 4500;

// ===== Difficulty presets =====
// Selected in Settings; changes hold time, round time, and how strict the
// model's confidence threshold is before a prediction counts at all.
export const DIFFICULTIES = {
  easy:   { label: "Easy",   holdMs: 500, roundTimeMs: 16000, confidenceThreshold: 0.65 },
  normal: { label: "Normal", holdMs: 700, roundTimeMs: 12000, confidenceThreshold: 0.75 },
  hard:   { label: "Hard",   holdMs: 900, roundTimeMs: 8000,  confidenceThreshold: 0.85 },
};
export const DEFAULT_DIFFICULTY = "normal";

// ===== Combo / multiplier tiers =====
// Evaluated against the current consecutive-correct combo count.
export const COMBO_TIERS = [
  { min: 10, multiplier: 3, label: "3x Combo!" },
  { min: 5, multiplier: 2, label: "2x Combo!" },
  { min: 3, multiplier: 1, label: "On Fire!" },
  { min: 0, multiplier: 1, label: null },
];

export function comboTierFor(combo) {
  return COMBO_TIERS.find((tier) => combo >= tier.min);
}

// ===== XP / Levels =====
// XP required to reach level n+1 from level n grows linearly: level*100.
export function xpForNextLevel(level) {
  return level * 100;
}

export function levelFromTotalXp(totalXp) {
  let level = 1;
  let remaining = totalXp;
  while (remaining >= xpForNextLevel(level)) {
    remaining -= xpForNextLevel(level);
    level += 1;
  }
  return { level, xpIntoLevel: remaining, xpForNext: xpForNextLevel(level) };
}

export const XP_CORRECT = 10;
export const XP_MISS = 2;

// ===== Cosmetic theme unlocks =====
// Applied as a class on <body>; see style.css for each theme's palette.
export const THEMES = [
  { level: 1, id: "default", label: "Aurora (default)" },
  { level: 2, id: "sunset", label: "Sunset" },
  { level: 4, id: "neon", label: "Neon" },
  { level: 6, id: "ocean", label: "Ocean" },
];

export function unlockedThemes(level) {
  return THEMES.filter((t) => level >= t.level);
}

// ===== Achievements =====
// `check(ctx)` returns true the moment an achievement should be granted.
// `ctx` is assembled by app.js from the round/session/profile state.
export const ACHIEVEMENTS = [
  {
    id: "first_correct",
    title: "First Correct Sign",
    description: "Get your first letter right.",
    check: (ctx) => ctx.lifetimeCorrect >= 1,
  },
  {
    id: "five_in_a_row",
    title: "5 Correct in a Row",
    description: "Reach a streak of 5 in a single session.",
    check: (ctx) => ctx.sessionStreak >= 5,
  },
  {
    id: "ten_combo",
    title: "10 Combo",
    description: "Reach a combo of 10.",
    check: (ctx) => ctx.combo >= 10,
  },
  {
    id: "perfect_game",
    title: "Perfect Game",
    description: "Finish a Classic session with zero misses.",
    check: (ctx) => ctx.sessionEnded && ctx.sessionMisses === 0 && ctx.sessionRounds > 0,
  },
  {
    id: "alphabet_master",
    title: "Alphabet Master",
    description: "Get every letter right at least once (lifetime).",
    check: (ctx) => ctx.lifetimeMasteredCount >= LETTERS.length,
  },
  {
    id: "hundred_correct",
    title: "100 Correct Signs",
    description: "Get 100 correct signs total, across all sessions.",
    check: (ctx) => ctx.lifetimeCorrect >= 100,
  },
  {
    id: "fast_hands",
    title: "Fast Hands",
    description: "Get a letter right in under 1.5 seconds.",
    check: (ctx) => ctx.lastResponseMs != null && ctx.lastResponseMs < 1500,
  },
  {
    id: "no_mistakes",
    title: "No Mistakes",
    description: "Get a letter right on the very first attempt, no fumbling.",
    check: (ctx) => ctx.roundHadWrongPrediction === false && ctx.lastRoundWasCorrect === true,
  },
];

// ===== Grading (end-of-game report) =====
export function gradeFor(accuracyPct) {
  if (accuracyPct >= 95) return "S";
  if (accuracyPct >= 85) return "A";
  if (accuracyPct >= 70) return "B";
  if (accuracyPct >= 50) return "C";
  return "D";
}

// MediaPipe hand landmark index groups, used by js/coaching.js and skeleton drawing.
export const FINGER_GROUPS = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

export const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];
