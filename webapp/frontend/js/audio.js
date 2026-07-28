let audioCtx = null;
let masterGain = null;
let volume = 0.8;
let muted = false;

export function ensureAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  masterGain = audioCtx.createGain();
  masterGain.gain.value = muted ? 0 : volume;
  masterGain.connect(audioCtx.destination);
}

export function setVolume(v) {
  volume = Math.min(1, Math.max(0, v));
  if (masterGain) masterGain.gain.value = muted ? 0 : volume;
}

export function setMuted(m) {
  muted = m;
  if (masterGain) masterGain.gain.value = muted ? 0 : volume;
}

function beep(freqs, duration = 0.12, gap = 0.09, type = "sine") {
  if (!audioCtx) return;
  freqs.forEach((freq, i) => {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    const start = audioCtx.currentTime + i * gap;
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.2, start + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
    osc.connect(gain).connect(masterGain);
    osc.start(start);
    osc.stop(start + duration + 0.02);
  });
}

export const playCorrectSound = () => beep([523.25, 659.25, 783.99]);
export const playMissSound = () => beep([220, 160]);
export const playTick = () => beep([880], 0.05, 0, "square");
export const playLevelUp = () => beep([523.25, 659.25, 783.99, 1046.5], 0.14, 0.1);
export const playComboMilestone = () => beep([659.25, 987.77], 0.1, 0.08, "triangle");
