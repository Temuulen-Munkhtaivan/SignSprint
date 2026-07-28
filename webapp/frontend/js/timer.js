const COLOR_STOPS = [
  { at: 1.0, color: [52, 228, 199] },  // green (--success)
  { at: 0.4, color: [255, 209, 102] }, // yellow (--accent-3)
  { at: 0.0, color: [255, 99, 99] },   // red (--danger)
];

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function colorForFraction(fraction) {
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const hi = COLOR_STOPS[i];
    const lo = COLOR_STOPS[i + 1];
    if (fraction <= hi.at && fraction >= lo.at) {
      const span = hi.at - lo.at || 1;
      const t = (fraction - lo.at) / span;
      const [r, g, b] = [0, 1, 2].map((c) => Math.round(lerp(lo.color[c], hi.color[c], t)));
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  return `rgb(${COLOR_STOPS[COLOR_STOPS.length - 1].color.join(", ")})`;
}

export function createCircularTimer(circleEl, textEl) {
  const radius = circleEl.r.baseVal.value;
  const circumference = 2 * Math.PI * radius;
  circleEl.style.strokeDasharray = `${circumference}`;

  return {
    update(remainingMs, totalMs) {
      const fraction = Math.max(0, Math.min(1, remainingMs / totalMs));
      circleEl.style.strokeDashoffset = `${circumference * (1 - fraction)}`;
      circleEl.style.stroke = colorForFraction(fraction);
      if (textEl) textEl.textContent = Math.ceil(remainingMs / 1000);
    },
    reset() {
      circleEl.style.strokeDashoffset = "0";
      circleEl.style.stroke = colorForFraction(1);
    },
  };
}
