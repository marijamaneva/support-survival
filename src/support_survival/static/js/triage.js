import { drawWaveform } from "./monitor.js";

drawWaveform(document.getElementById("waveform"));

const tierRowClass = { Urgent: "row-urgent", Monitor: "row-monitor", Routine: "row-routine" };
const tierBadgeClass = { Urgent: "tier-urgent", Monitor: "tier-monitor", Routine: "tier-routine" };

fetch("/triage")
  .then((res) => res.json())
  .then((data) => {
    const rows = document.getElementById("rows");
    rows.innerHTML = data.patients.map((p) => `
      <tr class="${tierRowClass[p.tier] || ""}">
        <td class="num">#${p.patient_id}</td>
        <td class="num">${p.age}</td>
        <td>${p.cancer}</td>
        <td class="num">${(p.overall_risk * 100).toFixed(1)}%</td>
        <td class="num">${(p.risk_30d * 100).toFixed(1)}%</td>
        <td class="num">${(p.risk_90d * 100).toFixed(1)}%</td>
        <td><span class="tier ${tierBadgeClass[p.tier] || ""}"><span class="tier-dot"></span>${p.tier}</span></td>
      </tr>
    `).join("");

    const counts = { Urgent: 0, Monitor: 0, Routine: 0 };
    data.patients.forEach((p) => { counts[p.tier] = (counts[p.tier] || 0) + 1; });
    document.getElementById("count-urgent").textContent = counts.Urgent;
    document.getElementById("count-monitor").textContent = counts.Monitor;
    document.getElementById("count-routine").textContent = counts.Routine;
    document.getElementById("stats").hidden = false;

    document.getElementById("meta").textContent =
      `${data.patients.length} patients, sampled from the held-out test set. model_version: ${data.model_version}`;
  })
  .catch((err) => {
    document.getElementById("meta").textContent = "Could not load the triage panel: " + err;
  });
