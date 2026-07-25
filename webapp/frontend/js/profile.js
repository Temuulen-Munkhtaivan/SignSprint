import { LETTERS, DEFAULT_DIFFICULTY, DIFFICULTIES, levelFromTotalXp } from "./config.js";

const STORAGE_KEY = "signsprint_profile_v1";

function defaultProfile() {
  const perLetter = {};
  for (const letter of LETTERS) perLetter[letter] = { attempts: 0, correct: 0, bestTimeMs: null };

  return {
    version: 1,
    xpTotal: 0,
    achievements: {}, // id -> earned-at timestamp (ms)
    settings: {
      colorblind: false,
      difficulty: DEFAULT_DIFFICULTY,
      sensitivity: DIFFICULTIES[DEFAULT_DIFFICULTY].confidenceThreshold,
      volume: 0.8,
      muted: false,
      theme: "default",
      mode: "dark",
    },
    stats: {
      gamesPlayed: 0,
      lifetimeCorrect: 0,
      lifetimeAttempts: 0,
      bestCombo: 0,
      highestScore: 0,
      sumResponseMs: 0,
      countResponses: 0,
      fastestResponseMs: null,
      fastestLetter: null,
      perLetter,
      masteredAllTime: [],
    },
  };
}

export function loadProfile() {
  let raw;
  try {
    raw = JSON.parse(localStorage.getItem(STORAGE_KEY));
  } catch {
    raw = null;
  }
  const fresh = defaultProfile();
  if (!raw || typeof raw !== "object") return fresh;

  // Shallow-merge so new fields introduced later don't break existing saves.
  return {
    ...fresh,
    ...raw,
    settings: { ...fresh.settings, ...(raw.settings || {}) },
    stats: {
      ...fresh.stats,
      ...(raw.stats || {}),
      perLetter: { ...fresh.stats.perLetter, ...((raw.stats && raw.stats.perLetter) || {}) },
    },
    achievements: { ...(raw.achievements || {}) },
  };
}

export function saveProfile(profile) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
}

export function getLevelInfo(profile) {
  return levelFromTotalXp(profile.xpTotal);
}

export function addXp(profile, amount) {
  profile.xpTotal = Math.max(0, profile.xpTotal + amount);
  return getLevelInfo(profile);
}

export function recordRoundResult(profile, { letter, correct, responseMs }) {
  const s = profile.stats;
  s.lifetimeAttempts += 1;
  const letterStats = s.perLetter[letter] || { attempts: 0, correct: 0, bestTimeMs: null };
  letterStats.attempts += 1;

  if (correct) {
    s.lifetimeCorrect += 1;
    letterStats.correct += 1;
    if (responseMs != null) {
      s.sumResponseMs += responseMs;
      s.countResponses += 1;
      if (s.fastestResponseMs == null || responseMs < s.fastestResponseMs) {
        s.fastestResponseMs = responseMs;
        s.fastestLetter = letter;
      }
      if (letterStats.bestTimeMs == null || responseMs < letterStats.bestTimeMs) {
        letterStats.bestTimeMs = responseMs;
      }
    }
    if (!s.masteredAllTime.includes(letter)) s.masteredAllTime.push(letter);
  }

  s.perLetter[letter] = letterStats;
}

/**
 * Learn Mode's equivalent of recordRoundResult -- marks a letter as mastered
 * without touching lifetime attempts/accuracy/response-time stats, since
 * Learn Mode is untimed practice, not a performance measurement. Still
 * updates masteredAllTime so it counts toward the Alphabet Master achievement
 * and the shared mastery grid.
 */
export function markLetterMastered(profile, letter) {
  if (!profile.stats.masteredAllTime.includes(letter)) {
    profile.stats.masteredAllTime.push(letter);
  }
}

export function recordCombo(profile, combo) {
  profile.stats.bestCombo = Math.max(profile.stats.bestCombo, combo);
}

export function recordGameEnd(profile, { score }) {
  profile.stats.gamesPlayed += 1;
  profile.stats.highestScore = Math.max(profile.stats.highestScore, score);
}

export function markAchievement(profile, id) {
  if (profile.achievements[id]) return false;
  profile.achievements[id] = Date.now();
  return true;
}

export function avgResponseMs(profile) {
  const s = profile.stats;
  return s.countResponses > 0 ? Math.round(s.sumResponseMs / s.countResponses) : null;
}

export function accuracyPct(profile) {
  const s = profile.stats;
  return s.lifetimeAttempts > 0 ? Math.round((s.lifetimeCorrect / s.lifetimeAttempts) * 100) : 0;
}

export function lettersNeedingPractice(profile, limit = 5) {
  const s = profile.stats;
  return Object.entries(s.perLetter)
    .map(([letter, st]) => ({
      letter,
      attempts: st.attempts,
      correct: st.correct,
      accuracy: st.attempts > 0 ? st.correct / st.attempts : null,
    }))
    .filter((row) => row.attempts >= 2 && row.accuracy != null)
    .sort((a, b) => a.accuracy - b.accuracy)
    .slice(0, limit);
}
