export function burstConfetti() {
  let canvas = document.getElementById("confettiCanvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.id = "confettiCanvas";
    document.body.appendChild(canvas);
  }
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx = canvas.getContext("2d");
  const colors = ["#ff5da2", "#34e4c7", "#ffd166", "#8b6bff"];
  const particles = Array.from({ length: 80 }, () => ({
    x: canvas.width / 2 + (Math.random() - 0.5) * 200,
    y: canvas.height * 0.35,
    vx: (Math.random() - 0.5) * 10,
    vy: Math.random() * -8 - 2,
    size: Math.random() * 7 + 3,
    color: colors[Math.floor(Math.random() * colors.length)],
    rotation: Math.random() * Math.PI,
    vr: (Math.random() - 0.5) * 0.3,
  }));

  let frame = 0;
  const maxFrames = 70;
  function step() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of particles) {
      p.vy += 0.35;
      p.x += p.vx;
      p.y += p.vy;
      p.rotation += p.vr;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rotation);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
      ctx.restore();
    }
    frame++;
    if (frame < maxFrames) requestAnimationFrame(step);
    else ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  requestAnimationFrame(step);
}

/** Briefly adds a glow class to an element, e.g. the camera frame on a correct answer. */
export function glowPulse(el) {
  if (!el) return;
  el.classList.remove("glow-pulse");
  void el.offsetWidth; // restart animation
  el.classList.add("glow-pulse");
}
