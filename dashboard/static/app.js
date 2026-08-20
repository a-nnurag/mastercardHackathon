const app = document.getElementById("app");
const navSessions = document.getElementById("nav-sessions");
const navMetrics = document.getElementById("nav-metrics");

let allSessions = null;

function setActiveNav(which) {
  navSessions.classList.toggle("active", which === "sessions");
  navMetrics.classList.toggle("active", which === "metrics");
}

navSessions.onclick = () => { setActiveNav("sessions"); renderSessionList(); };
navMetrics.onclick = () => { setActiveNav("metrics"); renderMetrics(); };

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function ensureSessions() {
  if (!allSessions) allSessions = await fetchJSON("/api/sessions");
  return allSessions;
}

function subtletyBadge(subtlety) {
  return `<span class="badge ${subtlety}">${subtlety}</span>`;
}

function fmt(n, digits = 3) {
  return typeof n === "number" ? n.toFixed(digits) : "N/A";
}

// ---------- Sessions view ----------

async function renderSessionList() {
  app.innerHTML = `<p class="hint">Loading sessions...</p>`;
  const sessions = await ensureSessions();

  const categories = [...new Set(sessions.map(s => s.category))].sort();
  const subtleties = [...new Set(sessions.map(s => s.subtlety))].sort();

  app.innerHTML = `
    <div class="filters">
      <select id="f-subtlety"><option value="">All subtleties</option>${subtleties.map(s => `<option value="${s}">${s}</option>`).join("")}</select>
      <select id="f-category"><option value="">All categories</option>${categories.map(d => `<option value="${d}">${d}</option>`).join("")}</select>
      <select id="f-rules"><option value="">Any rules result</option><option value="flagged">Rules: flagged</option><option value="clean">Rules: clean</option></select>
      <span class="hint" id="count-hint"></span>
    </div>
    <table>
      <thead><tr><th>Agent ID</th><th>Category</th><th>Subtlety</th><th>Rules</th><th>LightGBM</th><th>Content</th><th>GNN</th></tr></thead>
      <tbody id="session-rows"></tbody>
    </table>
  `;

  const subtletySel = document.getElementById("f-subtlety");
  const categorySel = document.getElementById("f-category");
  const rulesSel = document.getElementById("f-rules");
  const tbody = document.getElementById("session-rows");
  const countHint = document.getElementById("count-hint");

  function applyFilters() {
    const subtlety = subtletySel.value;
    const category = categorySel.value;
    const rules = rulesSel.value;
    const filtered = sessions.filter(s =>
      (!subtlety || s.subtlety === subtlety) &&
      (!category || s.category === category) &&
      (!rules || (rules === "flagged") === s.signals.rules_flagged)
    );
    countHint.textContent = `${filtered.length} / ${sessions.length} sessions`;
    tbody.innerHTML = filtered.map(s => `
      <tr class="row-clickable" data-id="${s.agent_id}">
        <td>${s.agent_id}</td>
        <td>${s.category}</td>
        <td>${subtletyBadge(s.subtlety)}</td>
        <td><span class="badge ${s.signals.rules_flagged ? "flagged" : "clean"}">${s.signals.rules_flagged ? "flagged" : "clean"}</span></td>
        <td>${fmt(s.signals.lightgbm_prob)}</td>
        <td>${fmt(s.signals.content_score)}</td>
        <td>${s.signals.gnn_prob != null ? fmt(s.signals.gnn_prob) : "N/A"}</td>
      </tr>
    `).join("");
    tbody.querySelectorAll("tr").forEach(tr => {
      tr.onclick = () => renderSessionDetail(tr.dataset.id);
    });
  }

  subtletySel.onchange = applyFilters;
  categorySel.onchange = applyFilters;
  rulesSel.onchange = applyFilters;
  applyFilters();
}

async function renderSessionDetail(agentId) {
  const sessions = await ensureSessions();
  const s = sessions.find(x => x.agent_id === agentId);
  if (!s) { app.innerHTML = `<p>Session not found.</p>`; return; }

  app.innerHTML = `
    <button class="action back" id="back-btn">&larr; Back to sessions</button>
    <div class="panel">
      <h2>${s.agent_id} <span style="float:right">${subtletyBadge(s.subtlety)}</span></h2>
      <div class="signal-grid">
        <div class="signal-card"><div class="label">Rules</div><div class="value">${s.signals.rules_flagged ? "FLAGGED" : "clean"}</div></div>
        <div class="signal-card"><div class="label">Constraint drift</div><div class="value">${fmt(s.signals.constraint_drift)}</div></div>
        <div class="signal-card"><div class="label">Ingestion trust</div><div class="value">${fmt(s.signals.ingestion_source_trust_score)}</div></div>
        <div class="signal-card"><div class="label">Utterance/artifact divergence</div><div class="value">${fmt(s.signals.utterance_artifact_divergence)}</div></div>
        <div class="signal-card"><div class="label">LightGBM prob.</div><div class="value">${fmt(s.signals.lightgbm_prob)}</div></div>
        <div class="signal-card"><div class="label">Content score</div><div class="value">${fmt(s.signals.content_score)}</div></div>
        <div class="signal-card"><div class="label">GNN (merchant ring) prob.</div><div class="value">${s.signals.gnn_prob != null ? fmt(s.signals.gnn_prob) : "N/A"}</div></div>
      </div>
      ${s.signals.rules_reasons && s.signals.rules_reasons.length ? `<p class="hint">Rules reasons: ${s.signals.rules_reasons.join(", ")}</p>` : ""}
    </div>
    <div class="panel">
      <h2>Raw utterance</h2>
      <div class="readonly">${escapeHtml(s.raw_utterance)}</div>
    </div>
    <div class="panel">
      <h2>Signed artifact</h2>
      <div class="readonly">${escapeHtml(s.signed_artifact_text)}</div>
    </div>
    ${s.injection_payload_text ? `
    <div class="panel">
      <h2>Injection payload (ground truth, hidden from Defend)</h2>
      <div class="readonly">${escapeHtml(s.injection_payload_text)}</div>
    </div>` : ""}
    <div class="panel">
      <h2>Task origin</h2>
      <p class="hint">${escapeHtml(s.task_origin_url)}</p>
    </div>
    <div class="panel">
      <h2>AI Verdict</h2>
      <button class="action" id="verdict-btn">Get AI Verdict</button>
      <div id="verdict-result"></div>
    </div>
  `;

  document.getElementById("back-btn").onclick = renderSessionList;
  document.getElementById("verdict-btn").onclick = async (e) => {
    const btn = e.target;
    const resultEl = document.getElementById("verdict-result");
    btn.disabled = true;
    btn.textContent = "Calling LLM verdict layer...";
    resultEl.innerHTML = `<p class="hint">This calls the live LLM adapter — may take a few seconds.</p>`;
    try {
      const data = await fetchJSON(`/api/sessions/${agentId}/verdict`, { method: "POST" });
      const v = data.verdict;
      resultEl.innerHTML = `
        <div class="verdict-box ${v.risk_level}">
          <strong>${v.risk_level}</strong> — recommendation: <strong>${v.recommendation}</strong>
          <p>${escapeHtml(v.explanation)}</p>
        </div>
      `;
    } catch (err) {
      resultEl.innerHTML = `<p class="hint">Verdict failed: ${escapeHtml(err.message)}</p>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Get AI Verdict";
    }
  };
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// ---------- Metrics view ----------

function metricsRow(label, m) {
  if (!m) return `<tr><td>${label}</td><td colspan="4" class="hint">not available</td></tr>`;
  return `<tr><td>${label}</td><td>${fmt(m.precision)}</td><td>${fmt(m.recall)}</td><td>${fmt(m.f1)}</td><td>${fmt(m.auc)}</td></tr>`;
}

function mutationChart(rounds) {
  if (!rounds || !rounds.length) return `<p class="hint">No mutation-round data yet — run the Mutator (mutator/mutate.py) to populate data/mutation_rounds.json.</p>`;

  const w = 560, h = 240, pad = 40;
  const xs = rounds.map((_, i) => pad + i * ((w - 2 * pad) / Math.max(rounds.length - 1, 1)));
  const metricKeys = [
    { key: "precision", color: "#3a5bff" },
    { key: "recall", color: "#5fd97a" },
    { key: "f1", color: "#ffb84d" },
  ];
  const yFor = v => h - pad - v * (h - 2 * pad);

  const lines = metricKeys.map(({ key, color }) => {
    const points = rounds.map((r, i) => `${xs[i]},${yFor(r[key])}`).join(" ");
    const circles = rounds.map((r, i) => `<circle cx="${xs[i]}" cy="${yFor(r[key])}" r="4" fill="${color}"/>`).join("");
    return `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2"/>${circles}`;
  }).join("");

  const labels = rounds.map((r, i) => `<text x="${xs[i]}" y="${h - pad + 18}" fill="#9aa0ac" font-size="11" text-anchor="middle">Round ${r.round} (n=${r.n})</text>`).join("");
  const legend = metricKeys.map(({ key, color }, i) => `
    <circle cx="${pad + i * 90}" cy="16" r="5" fill="${color}"/>
    <text x="${pad + i * 90 + 10}" y="20" fill="#e4e6eb" font-size="12">${key}</text>
  `).join("");

  return `<svg class="chart" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${legend}${lines}${labels}</svg>`;
}

async function renderMetrics() {
  app.innerHTML = `<p class="hint">Computing metrics (re-runs cross-validation + GNN evaluation)...</p>`;
  const data = await fetchJSON("/api/metrics");
  const a1 = data.attack1_prompt_injection;
  const a2 = data.attack2_merchant_laundering;

  app.innerHTML = `
    <div class="panel metrics-table">
      <h2>Attack #1 — Prompt injection (Intent Artifact corruption)</h2>
      <table>
        <thead><tr><th>Slice</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr></thead>
        <tbody>
          ${metricsRow("Overall", a1 && a1.overall)}
          ${metricsRow("Obvious hijacks", a1 && a1.obvious)}
          ${metricsRow("Subtle hijacks", a1 && a1.subtle)}
        </tbody>
      </table>
      <p class="hint">False-positive rate on benign sessions: ${a1 ? fmt(a1.fp_rate_benign) : "N/A"}</p>
    </div>

    <div class="panel metrics-table">
      <h2>Attack #2 — Transaction laundering (merchant network GNN)</h2>
      <table>
        <thead><tr><th>Slice</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr></thead>
        <tbody>${metricsRow("Held-out merchant nodes", a2)}</tbody>
      </table>
      ${!a2 ? `<p class="hint">GNN training did not converge on this run — see server logs.</p>` : ""}
    </div>

    <div class="panel">
      <h2>Mutator hardening curve (Task 10)</h2>
      ${mutationChart(data.mutation_rounds)}
    </div>
  `;
}

renderSessionList();
