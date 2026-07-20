let ws = null;
let wsReady = false;
let pendingResolve = null;

export function connectWs(onStatusChange) {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws/predict`);
  ws.onopen = () => {
    wsReady = true;
    onStatusChange(true);
  };
  ws.onclose = () => {
    wsReady = false;
    onStatusChange(false);
    setTimeout(() => connectWs(onStatusChange), 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (event) => {
    if (!pendingResolve) return;
    const data = JSON.parse(event.data);
    pendingResolve(data);
    pendingResolve = null;
  };
}

export function requestPrediction(landmarks) {
  if (!wsReady) return Promise.resolve(null);
  return new Promise((resolve) => {
    pendingResolve = resolve;
    ws.send(JSON.stringify({ landmarks }));
  });
}

/**
 * Rolling majority-vote smoother over recent above-threshold predictions,
 * mirroring realtime_predict.py's `prediction_history` deque. `threshold`
 * and `historyLen` can be adjusted live (e.g. when difficulty changes).
 */
export function createSmoother(threshold, historyLen) {
  let history = [];

  return {
    setThreshold(t) { threshold = t; },
    setHistoryLen(n) { historyLen = n; },
    push(letter, confidence) {
      if (confidence > threshold) {
        history.push(letter);
        if (history.length > historyLen) history.shift();
      }
    },
    reset() { history = []; },
    smoothed() {
      if (history.length === 0) return null;
      const counts = {};
      let best = history[0];
      for (const l of history) {
        counts[l] = (counts[l] || 0) + 1;
        if (counts[l] > counts[best]) best = l;
      }
      return best;
    },
  };
}
