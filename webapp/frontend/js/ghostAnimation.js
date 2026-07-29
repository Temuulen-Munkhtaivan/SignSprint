import { FINGER_GROUPS } from "./config.js";

// J and Z aren't in letter_references.json -- they're motion letters with no
// single static pose, so a frozen diagram can't show anything meaningful for
// them (and previously showed nothing at all). This animates just the
// relevant fingertip through the same path motionDetector.js actually checks
// for, layered on top of a real static letter's pose as the resting hand
// shape: J starts from the "I" handshape in ASL, and Z has no direct
// equivalent among the 24 static letters, so "D" (index already extended)
// is used as the closest approximation for the non-moving fingers.
export const MOTION_LETTER_BASE = { J: "I", Z: "D" };

export function isMotionLetter(letter) {
  return letter === "J" || letter === "Z";
}

const PINKY_TIP = FINGER_GROUPS.pinky[FINGER_GROUPS.pinky.length - 1]; // 20
const INDEX_TIP = FINGER_GROUPS.index[FINGER_GROUPS.index.length - 1]; // 8

export const LOOP_MS = 2200;
const HOLD_FRACTION = 0.18; // brief pause at the end of the stroke before looping back

// [t, dx, dy] keyframes -- offsets added to the tip's base (x, y), in the
// same wrist-relative normalized units as the reference vectors. Sized to
// clear motionDetector.js's own movedDown/hookedSideways/diagonal thresholds,
// so the demo traces the same motion that actually gets recognized.
const J_PATH = [
  [0.0, 0, 0],
  [0.45, 0, 0.16],
  [0.75, 0.11, 0.2],
  [1.0, 0.11, 0.2],
];
const Z_PATH = [
  [0.0, 0, 0],
  [0.3, 0.12, 0],
  [0.6, 0.02, 0.09],
  [0.85, 0.14, 0.09],
  [1.0, 0.14, 0.09],
];

function interpolatePath(path, t) {
  for (let i = 0; i < path.length - 1; i++) {
    const [t0, x0, y0] = path[i];
    const [t1, x1, y1] = path[i + 1];
    if (t >= t0 && t <= t1) {
      const span = t1 - t0 || 1;
      const f = (t - t0) / span;
      return [x0 + (x1 - x0) * f, y0 + (y1 - y0) * f];
    }
  }
  const [, x, y] = path[path.length - 1];
  return [x, y];
}

/**
 * Returns a 63-float landmark vector for a motion letter at loop progress
 * `t` (0-1): `baseFlat63` (a real static letter's reference pose) with just
 * the relevant fingertip offset along its demonstration path.
 */
export function motionLetterFrame(letter, baseFlat63, t) {
  const tipIndex = letter === "J" ? PINKY_TIP : INDEX_TIP;
  const path = letter === "J" ? J_PATH : Z_PATH;
  const loopT = t <= 1 - HOLD_FRACTION ? t / (1 - HOLD_FRACTION) : 1;
  const [dx, dy] = interpolatePath(path, Math.min(1, loopT));

  const frame = baseFlat63.slice();
  frame[tipIndex * 3] += dx;
  frame[tipIndex * 3 + 1] += dy;
  return frame;
}
