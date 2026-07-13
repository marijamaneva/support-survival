// Shared helpers for the "bedside monitor" UI: the ambient waveform decoration
// and the risk -> color/label mapping used by both the predictor and the
// triage panel, so the two pages agree on what "urgent" looks like.

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function statusForRisk(p) {
  if (p < 0.50) return { label: "Nominal", color: "var(--trace)", glow: "var(--trace-glow)" };
  if (p < 0.65) return { label: "Elevated", color: "var(--amber)", glow: "var(--amber-glow)" };
  if (p < 0.80) return { label: "High", color: "var(--amber)", glow: "var(--amber-glow)" };
  return { label: "Critical", color: "var(--critical)", glow: "var(--critical-glow)" };
}

// Draws a looping EKG-style trace on a <canvas>. Purely decorative and
// ambient -- it never encodes real data, so it freezes on a single static
// beat when the viewer prefers reduced motion instead of animating forever.
export function drawWaveform(canvas, colorVar = "--trace") {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth;
  const cssHeight = canvas.clientHeight;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  ctx.scale(dpr, dpr);

  const style = getComputedStyle(document.documentElement);
  const color = style.getPropertyValue(colorVar).trim() || "#35e0a0";

  // One "beat" shape, tileable, in unit coordinates (x: 0..1, y: -1..1).
  function beat(x) {
    if (x < 0.55) return 0;
    if (x < 0.60) return -0.25;
    if (x < 0.64) return 1.0;
    if (x < 0.68) return -0.6;
    if (x < 0.74) return 0.15;
    if (x < 0.80) return 0;
    return 0;
  }

  function frame(x) {
    const midY = cssHeight / 2;
    const amp = cssHeight * 0.38;
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    const period = cssWidth * 0.4;
    for (let px = 0; px <= cssWidth; px++) {
      const phase = (((px + x) % period) / period + 1) % 1;
      const y = midY - beat(phase) * amp;
      if (px === 0) ctx.moveTo(px, y); else ctx.lineTo(px, y);
    }
    ctx.stroke();
  }

  if (REDUCED_MOTION) {
    frame(0);
    return;
  }

  let x = 0;
  function tick() {
    x = (x + 1.6) % 100000;
    frame(x);
    requestAnimationFrame(tick);
  }
  tick();
}
