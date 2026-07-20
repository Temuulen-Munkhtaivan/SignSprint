import { ACHIEVEMENTS } from "./config.js";
import { markAchievement } from "./profile.js";

export function checkAchievements(profile, ctx) {
  const newlyEarned = [];
  for (const achievement of ACHIEVEMENTS) {
    if (profile.achievements[achievement.id]) continue;
    if (achievement.check(ctx)) {
      if (markAchievement(profile, achievement.id)) newlyEarned.push(achievement);
    }
  }
  return newlyEarned;
}

let toastContainer = null;
function getContainer() {
  if (!toastContainer) toastContainer = document.getElementById("achievementToasts");
  return toastContainer;
}

export function showAchievementToast(achievement) {
  const container = getContainer();
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = "achievement-toast";
  toast.innerHTML = `
    <div class="achievement-toast-icon">🏆</div>
    <div>
      <div class="achievement-toast-title">Achievement unlocked</div>
      <div class="achievement-toast-name">${achievement.title}</div>
      <div class="achievement-toast-desc">${achievement.description}</div>
    </div>
  `;
  container.appendChild(toast);

  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 400);
  }, 3800);
}

export function showAchievementToasts(achievements) {
  achievements.forEach((a, i) => setTimeout(() => showAchievementToast(a), i * 600));
}
