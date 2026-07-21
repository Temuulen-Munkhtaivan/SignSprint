import { FINGER_GROUPS } from "./config.js";

let referencesPromise = null;

export function loadReferences() {
  if (!referencesPromise) {
    referencesPromise = fetch("assets/letter_references.json").then((r) => r.json());
  }
  return referencesPromise;
}

function toPoints(flat63) {
  const points = [];
  for (let i = 0; i < 21; i++) {
    points.push({ x: flat63[i * 3], y: flat63[i * 3 + 1], z: flat63[i * 3 + 2] });
  }
  return points;
}

function magnitude(p) {
  return Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

const FINGER_LABELS = { thumb: "thumb", index: "index finger", middle: "middle finger", ring: "ring finger", pinky: "pinky finger" };
const CURL_THRESHOLD = 0.18;
const SPREAD_THRESHOLD = 0.12;
const ROTATION_DOT_THRESHOLD = 0.6;

/**
 * Compares the player's live (already wrist-relative, max-abs normalized)
 * 63-float landmark vector against the per-letter mean reference vector
 * computed by export_letter_references.py, and returns ranked coaching
 * hints plus an overall 0-100 pose-similarity score.
 */
export function coachFor(userFlat63, referenceFlat63) {
  const user = toPoints(userFlat63);
  const ref = toPoints(referenceFlat63);

  const candidates = [];

  // Per-finger curl: since coordinates are wrist-relative, the tip's distance
  // from the origin directly measures how extended (far) vs curled (near) it is.
  for (const [finger, indices] of Object.entries(FINGER_GROUPS)) {
    const tipIdx = indices[indices.length - 1];
    const curlUser = magnitude(user[tipIdx]);
    const curlRef = magnitude(ref[tipIdx]);
    const delta = curlUser - curlRef;
    if (Math.abs(delta) >= CURL_THRESHOLD) {
      candidates.push({
        magnitude: Math.abs(delta),
        text: delta > 0
          ? `Curl your ${FINGER_LABELS[finger]} in a little more.`
          : `Straighten your ${FINGER_LABELS[finger]} a bit more.`,
      });
    }
  }

  // Thumb position relative to the index finger's base knuckle.
  const thumbTip = user[4], thumbTipRef = ref[4];
  const indexMcp = user[5], indexMcpRef = ref[5];
  const thumbDistUser = dist(thumbTip, indexMcp);
  const thumbDistRef = dist(thumbTipRef, indexMcpRef);
  if (thumbDistUser - thumbDistRef >= CURL_THRESHOLD) {
    candidates.push({ magnitude: thumbDistUser - thumbDistRef, text: "Move your thumb inward, closer to your hand." });
  }

  // Fingertip spread between adjacent fingers.
  const adjacentPairs = [["index", "middle"], ["middle", "ring"], ["ring", "pinky"]];
  for (const [a, b] of adjacentPairs) {
    const tipA = FINGER_GROUPS[a][FINGER_GROUPS[a].length - 1];
    const tipB = FINGER_GROUPS[b][FINGER_GROUPS[b].length - 1];
    const spreadUser = dist(user[tipA], user[tipB]);
    const spreadRef = dist(ref[tipA], ref[tipB]);
    if (spreadRef - spreadUser >= SPREAD_THRESHOLD) {
      candidates.push({ magnitude: spreadRef - spreadUser, text: "Spread your fingers apart slightly." });
      break; // one spread hint is enough
    }
  }

  // Coarse hand-plane orientation via the normal of (indexMcp, pinkyMcp) from the wrist.
  const pinkyMcp = user[17], pinkyMcpRef = ref[17];
  const normalUser = cross(indexMcp, pinkyMcp);
  const normalRef = cross(indexMcpRef, pinkyMcpRef);
  const cosAngle = dot(normalUser, normalRef) / ((magnitude(normalUser) * magnitude(normalRef)) || 1);
  if (cosAngle < ROTATION_DOT_THRESHOLD) {
    candidates.push({ magnitude: 1 - cosAngle, text: "Rotate your hand to face the camera more directly." });
  }

  candidates.sort((a, b) => b.magnitude - a.magnitude);
  const hints = candidates.slice(0, 2).map((c) => c.text);

  // Overall similarity score for display alongside model confidence.
  let sumDist = 0;
  for (let i = 0; i < 21; i++) sumDist += dist(user[i], ref[i]);
  const avgDist = sumDist / 21;
  const similarity = Math.round(Math.max(0, Math.min(1, 1 - avgDist / 0.5)) * 100);

  return { hints, similarity };
}

function cross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function dot(a, b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
