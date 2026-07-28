import { gradeFor, LETTERS } from "./config.js";
import { avgResponseMs, accuracyPct, lettersNeedingPractice, getLevelInfo } from "./profile.js";

function set(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

export function renderEndReport(session, profile, xpGained) {
  const accuracy = session.rounds > 0 ? Math.round((session.correct / session.rounds) * 100) : 0;

  set("finalScore", session.score);
  set("finalAccuracy", `${accuracy}%`);
  set("finalFastestLetter", session.fastestLetter ? `${session.fastestLetter} (${(session.fastestResponseMs / 1000).toFixed(2)}s)` : "—");
  set("finalBestCombo", session.bestCombo);
  set("finalAvgResponse", session.avgResponseMs != null ? `${(session.avgResponseMs / 1000).toFixed(2)}s` : "—");
  set("finalGrade", gradeFor(accuracy));

  const practiceList = document.getElementById("finalPracticeList");
  if (practiceList) {
    const practice = lettersNeedingPractice(profile, 5);
    practiceList.innerHTML = practice.length
      ? practice.map((p) => `<li>${p.letter} <span>(${Math.round(p.accuracy * 100)}% over ${p.attempts} tries)</span></li>`).join("")
      : "<li>Nothing yet — keep playing!</li>";
  }

  const achievementsList = document.getElementById("finalAchievementsList");
  if (achievementsList) {
    achievementsList.innerHTML = session.achievementsEarned.length
      ? session.achievementsEarned.map((a) => `<li>🏆 ${a.title}</li>`).join("")
      : "<li>None this round — try again!</li>";
  }

  set("finalXpGained", `+${xpGained} XP`);
  const level = getLevelInfo(profile);
  set("finalLevelText", `Level ${level.level} — ${level.xpIntoLevel}/${level.xpForNext} XP`);
  const bar = document.getElementById("finalLevelBar");
  if (bar) bar.style.width = `${Math.round((level.xpIntoLevel / level.xpForNext) * 100)}%`;
}

export function renderStatsScreen(profile) {
  const s = profile.stats;
  set("statGamesPlayed", s.gamesPlayed);
  set("statAvgAccuracy", `${accuracyPct(profile)}%`);
  set("statFastestResponse", s.fastestResponseMs != null ? `${(s.fastestResponseMs / 1000).toFixed(2)}s (${s.fastestLetter})` : "—");
  const avg = avgResponseMs(profile);
  set("statAvgResponse", avg != null ? `${(avg / 1000).toFixed(2)}s` : "—");
  set("statBestCombo", s.bestCombo);
  set("statHighestScore", s.highestScore);
  set("statLettersMastered", `${s.masteredAllTime.length}/${LETTERS.length}`);

  const practiceList = document.getElementById("statPracticeList");
  if (practiceList) {
    const practice = lettersNeedingPractice(profile, 5);
    practiceList.innerHTML = practice.length
      ? practice.map((p) => `<li>${p.letter} <span>(${Math.round(p.accuracy * 100)}%)</span></li>`).join("")
      : "<li>Not enough data yet.</li>";
  }

  drawLetterChart(profile);
}

function drawLetterChart(profile) {
  const canvas = document.getElementById("statLetterChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 600;
  const cssHeight = 160;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const barWidth = cssWidth / LETTERS.length;
  LETTERS.forEach((letter, i) => {
    const st = profile.stats.perLetter[letter];
    const acc = st && st.attempts > 0 ? st.correct / st.attempts : 0;
    const barHeight = Math.max(2, acc * (cssHeight - 20));
    const x = i * barWidth + barWidth * 0.15;
    const w = barWidth * 0.7;
    const y = cssHeight - barHeight - 16;

    ctx.fillStyle = acc === 0 ? "rgba(255,255,255,0.12)" : `rgba(52, 228, 199, ${0.35 + acc * 0.65})`;
    ctx.fillRect(x, y, w, barHeight);

    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(letter, x + w / 2, cssHeight - 4);
  });
}
