import { comboTierFor } from "./config.js";

export function createComboState() {
  return { combo: 0, lastAnnouncedTier: null };
}

export function registerCorrect(state) {
  state.combo += 1;
  return comboTierFor(state.combo);
}

export function registerMiss(state) {
  state.combo = 0;
  state.lastAnnouncedTier = null;
}

let popupEl = null;
function getPopupEl() {
  if (!popupEl) popupEl = document.getElementById("comboPopup");
  return popupEl;
}

/**
 * Shows the combo popup only when entering a new tier (not on every correct
 * answer within the same tier), so "On Fire!" doesn't re-fire every round.
 */
export function maybeShowComboPopup(state, tier, combo) {
  const el = getPopupEl();
  if (!el || !tier || !tier.label) return;
  if (state.lastAnnouncedTier === tier.label) return;
  state.lastAnnouncedTier = tier.label;

  el.textContent = tier.multiplier > 1 ? `${tier.label} (${combo} combo)` : `${tier.label} ${combo} combo`;
  el.classList.remove("pop");
  void el.offsetWidth; // restart animation
  el.classList.add("pop");
}
