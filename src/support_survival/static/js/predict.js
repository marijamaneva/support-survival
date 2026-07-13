import { statusForRisk, drawWaveform } from "./monitor.js";

const METER_SEGMENTS = 24;

const waveform = document.getElementById("waveform");
drawWaveform(waveform);

const meterTrack = document.getElementById("meter-track");
for (let i = 0; i < METER_SEGMENTS; i++) {
  const seg = document.createElement("div");
  seg.className = "meter-seg";
  meterTrack.appendChild(seg);
}

function paintResult(pct, status) {
  const resultBox = document.getElementById("result");
  resultBox.style.setProperty("--readout-color", status.color);
  resultBox.style.setProperty("--readout-glow", status.glow);

  document.getElementById("hero-value").textContent = pct.toFixed(1) + "%";
  document.getElementById("status-dot").classList.toggle("pulsing", status.label === "Critical");
  document.getElementById("status-label").textContent = status.label;

  const litCount = Math.round((pct / 100) * METER_SEGMENTS);
  [...meterTrack.children].forEach((seg, i) => seg.classList.toggle("lit", i < litCount));
}

document.getElementById("predict-form").addEventListener("submit", async function (e) {
  e.preventDefault();
  const form = e.target;
  const errorBox = document.getElementById("error");
  const resultBox = document.getElementById("result");
  errorBox.hidden = true;
  resultBox.hidden = true;

  const payload = {
    age: parseFloat(form.age.value),
    sex: parseInt(form.sex.value, 10),
    n_comorbidities: parseInt(form.n_comorbidities.value, 10),
    diabetes: parseInt(form.diabetes.value, 10),
    dementia: parseInt(form.dementia.value, 10),
    cancer: parseInt(form.cancer.value, 10),
    mean_bp: parseFloat(form.mean_bp.value),
    heart_rate: parseFloat(form.heart_rate.value),
    resp_rate: parseFloat(form.resp_rate.value),
    temperature: parseFloat(form.temperature.value),
    serum_sodium: parseFloat(form.serum_sodium.value),
    wbc: parseFloat(form.wbc.value),
    serum_creatinine: parseFloat(form.serum_creatinine.value),
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json();

    if (!res.ok) {
      const detail = body.detail;
      const messages = Array.isArray(detail)
        ? detail.map((d) => (d.loc || []).slice(-1)[0] + ": " + d.msg)
        : [String(detail)];
      errorBox.textContent = messages.join(" · ");
      errorBox.hidden = false;
      return;
    }

    const pct = body.risk_probability * 100;
    paintResult(pct, statusForRisk(body.risk_probability));
    document.getElementById("model-version").textContent = body.model_version;
    resultBox.hidden = false;
  } catch (err) {
    errorBox.textContent = "Could not reach the API: " + err;
    errorBox.hidden = false;
  }
});
