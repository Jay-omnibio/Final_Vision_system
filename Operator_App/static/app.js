const state = {
  selectedCrops: new Set(),
  selectedSamples: new Set(),
  settings: {},
  cropUrls: new Map(),
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function text(value, fallback = "-") {
  return value === undefined || value === null || value === "" ? fallback : String(value);
}

function escapeHtml(value) {
  return text(value, "").replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function eventStatus(event) {
  if (event.status === "new") return "new";
  if (event.label === "unknown" || String(event.label || "").startsWith("new_")) return "unknown";
  return event.status || "normal";
}

function eventHtml(event, { selectable = false } = {}) {
  const status = eventStatus(event);
  if (event.crop_path && event.crop_url) state.cropUrls.set(event.crop_path, event.crop_url);
  const crop = event.crop_url ? `<img class="thumb" src="${escapeAttr(event.crop_url)}" alt="">` : `<div class="thumb"></div>`;
  const checkbox = selectable && event.crop_path
    ? `<label class="check"><input type="checkbox" data-crop="${escapeAttr(event.crop_path)}"> select</label>`
    : `<span class="status ${escapeAttr(status)}">${escapeHtml(status)}</span>`;
  return `
    <article class="event-card">
      ${crop}
      <div>
        <div class="label">${escapeHtml(event.label)}</div>
        <div class="meta">ID ${escapeHtml(event.track_id)} · frame ${escapeHtml(event.frame_index)} · score ${Number(event.score || 0).toFixed(3)}</div>
        <div class="meta">${escapeHtml(event.new_group || "")}</div>
      </div>
      ${checkbox}
    </article>
  `;
}

async function refreshStatus() {
  const status = await api("/api/status");
  const summary = document.getElementById("runtimeSummary");
  summary.textContent = status.running
    ? `Running ${status.mode} · pid ${status.pid} · configured ${status.configured_mode}`
    : `Stopped · configured ${status.configured_mode} · exit ${text(status.last_exit_code)}`;
  document.getElementById("runtimeLog").textContent = status.log_tail || "";
  document.getElementById("debugVideo").checked = Boolean(status.debug_video_default);
}

async function refreshEvents() {
  const data = await api("/api/events?limit=100");
  const latest = data.events.slice(-12).reverse();
  document.getElementById("eventList").innerHTML = latest.map(eventHtml).join("") || "<p>No events yet.</p>";
  document.getElementById("eventCards").innerHTML = data.events.reverse().map(event => eventHtml(event, { selectable: true })).join("") || "<p>No events yet.</p>";
  bindCropChecks();
}

async function refreshUnknowns() {
  const data = await api("/api/unknowns");
  document.getElementById("unknownGroups").innerHTML = data.groups.map(group => {
    const cards = group.events.slice(-8).reverse().map(event => eventHtml(event, { selectable: true })).join("");
    return `<article class="unknown-card"><div></div><div><div class="label">${escapeHtml(group.name)}</div><div class="meta">${Number(group.count || 0)} event(s)</div>${cards}</div><span class="status new">review</span></article>`;
  }).join("") || "<p>No unknown groups yet.</p>";
  bindCropChecks();
}

function bindCropChecks() {
  document.querySelectorAll("input[data-crop]").forEach(input => {
    input.checked = state.selectedCrops.has(input.dataset.crop);
    input.addEventListener("change", () => {
      if (input.checked) state.selectedCrops.add(input.dataset.crop);
      else state.selectedCrops.delete(input.dataset.crop);
      updateSelectedSummary();
    });
  });
  updateSelectedSummary();
}

function updateSelectedSummary() {
  const count = state.selectedCrops.size;
  document.getElementById("selectedSummary").textContent = count ? `${count} crop(s) selected.` : "No crops selected.";
  const previews = Array.from(state.selectedCrops).slice(0, 12).map(path => {
    const url = state.cropUrls.get(path) || `/api/crop?path=${encodeURIComponent(path)}`;
    return `<img class="thumb" src="${escapeAttr(url)}" alt="">`;
  });
  document.getElementById("selectedPreview").innerHTML = previews.join("");
}

function modelArtifactHtml(name, artifact) {
  const existsClass = artifact.exists ? "ok-text" : "warning";
  const details = [
    artifact.class_count !== undefined ? `classes ${artifact.class_count}` : "",
    artifact.subclass_count !== undefined ? `subclasses ${artifact.subclass_count}` : "",
    artifact.label_count !== undefined ? `labels ${artifact.label_count}` : "",
    artifact.embedding_count !== undefined ? `embeddings ${artifact.embedding_count}` : "",
    artifact.size_bytes !== undefined ? `${artifact.size_bytes} bytes` : "",
  ].filter(Boolean).join(" · ");
  return `
    <article class="event-card">
      <div class="thumb"></div>
      <div>
        <div class="label">${escapeHtml(name)}</div>
        <div class="meta">${escapeHtml(artifact.path)}</div>
        <div class="meta">${escapeHtml(details)}</div>
        ${artifact.warning ? `<div class="warning">${escapeHtml(artifact.warning)}</div>` : ""}
      </div>
      <span class="${existsClass}">${artifact.exists ? "ok" : "missing"}</span>
    </article>
  `;
}

async function refreshModelStatus() {
  const status = await api("/api/model-status");
  const warnings = status.warnings.length
    ? `<div class="warning">${status.warnings.map(escapeHtml).join("<br>")}</div>`
    : `<div class="ok-text">All required model files found.</div>`;
  const artifacts = Object.entries(status.artifacts).map(([name, artifact]) => modelArtifactHtml(name, artifact)).join("");
  document.getElementById("modelStatus").innerHTML = `
    <article class="event-card">
      <div class="thumb"></div>
      <div>
        <div class="label">${escapeHtml(status.runtime_mode)} · ${escapeHtml(status.dino_backend)}</div>
        <div class="meta">${status.using_active_model ? "Using active learned model" : "Using base model"}</div>
        ${warnings}
      </div>
      <span class="${status.warnings.length ? "warning" : "ok-text"}">${status.warnings.length ? "check" : "ready"}</span>
    </article>
    ${artifacts}
  `;
}

function sampleHtml(sample) {
  return `
    <article class="event-card">
      <img class="thumb" src="/api/teaching-image?path=${encodeURIComponent(sample.image_path)}" alt="" onerror="this.replaceWith(document.createElement('div'))">
      <div>
        <div class="label">${escapeHtml(sample.label)}</div>
        <div class="meta">${escapeHtml(sample.image_path)}</div>
      </div>
      <label class="check"><input type="checkbox" data-sample="${escapeAttr(sample.image_path)}"> delete</label>
    </article>
  `;
}

async function refreshTeachingSamples() {
  const data = await api("/api/teaching-samples");
  document.getElementById("teachingSamples").innerHTML = data.samples.map(sampleHtml).join("") || "<p>No teaching samples yet.</p>";
  bindSampleChecks();
}

function bindSampleChecks() {
  document.querySelectorAll("input[data-sample]").forEach(input => {
    input.checked = state.selectedSamples.has(input.dataset.sample);
    input.addEventListener("change", () => {
      if (input.checked) state.selectedSamples.add(input.dataset.sample);
      else state.selectedSamples.delete(input.dataset.sample);
    });
  });
}

async function refreshRebuildStatus() {
  const status = await api("/api/rebuild-status");
  if (status.output) document.getElementById("assignOutput").textContent = JSON.stringify(status, null, 2);
  if (status.running) setTimeout(refreshRebuildStatus, 1500);
  else {
    await Promise.all([refreshModelStatus(), loadSettings(), refreshTeachingSamples()]);
  }
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  const form = document.getElementById("settingsForm");
  for (const [key, value] of Object.entries(state.settings)) {
    const input = form.elements[key];
    if (!input) continue;
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = value ?? "";
  }
}

function formPayload(form) {
  const payload = {};
  for (const element of form.elements) {
    if (!element.name) continue;
    payload[element.name] = element.type === "checkbox" ? element.checked : element.value;
  }
  return payload;
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab, .panel").forEach(item => item.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.tab).classList.add("active");
    });
  });
}

function bindActions() {
  document.getElementById("startBtn").addEventListener("click", async () => {
    await api("/api/start", {
      method: "POST",
      body: JSON.stringify({ debug_video: document.getElementById("debugVideo").checked }),
    });
    await refreshStatus();
  });
  document.getElementById("stopBtn").addEventListener("click", async () => {
    await api("/api/stop", { method: "POST" });
    await refreshStatus();
  });
  document.getElementById("refreshEventsBtn").addEventListener("click", refreshEvents);
  document.getElementById("refreshUnknownsBtn").addEventListener("click", refreshUnknowns);
  document.getElementById("refreshModelBtn").addEventListener("click", refreshModelStatus);
  document.getElementById("refreshSamplesBtn").addEventListener("click", refreshTeachingSamples);
  document.getElementById("deleteSamplesBtn").addEventListener("click", async () => {
    if (!state.selectedSamples.size) return;
    const result = await api("/api/delete-samples", {
      method: "POST",
      body: JSON.stringify({ image_paths: Array.from(state.selectedSamples) }),
    });
    state.selectedSamples.clear();
    document.getElementById("modelOutput").textContent = JSON.stringify(result, null, 2);
    await refreshTeachingSamples();
  });
  document.getElementById("resetBaseBtn").addEventListener("click", async () => {
    const result = await api("/api/reset-base-model", { method: "POST" });
    document.getElementById("modelOutput").textContent = JSON.stringify(result, null, 2);
    await Promise.all([refreshModelStatus(), loadSettings()]);
  });
  document.getElementById("settingsForm").addEventListener("submit", async event => {
    event.preventDefault();
    const result = await api("/api/settings", { method: "POST", body: JSON.stringify(formPayload(event.currentTarget)) });
    document.getElementById("settingsOutput").textContent = JSON.stringify(result, null, 2);
    await loadSettings();
  });
  document.getElementById("assignForm").addEventListener("submit", async event => {
    event.preventDefault();
    const result = await api("/api/assign", {
      method: "POST",
      body: JSON.stringify({
        crop_paths: Array.from(state.selectedCrops),
        class_name: document.getElementById("className").value,
        object_name: document.getElementById("objectName").value,
      }),
    });
    document.getElementById("assignOutput").textContent = JSON.stringify(result, null, 2);
    state.selectedCrops.clear();
    await refreshTeachingSamples();
    updateSelectedSummary();
  });
  document.getElementById("rebuildBtn").addEventListener("click", async () => {
    document.getElementById("assignOutput").textContent = "Rebuilding gallery...";
    const result = await api("/api/rebuild-gallery", { method: "POST" });
    document.getElementById("assignOutput").textContent = JSON.stringify(result, null, 2);
    await refreshRebuildStatus();
  });
}

async function boot() {
  bindTabs();
  bindActions();
  await Promise.all([refreshStatus(), refreshEvents(), refreshUnknowns(), loadSettings(), refreshModelStatus(), refreshTeachingSamples()]);
  setInterval(() => {
    refreshStatus();
    refreshEvents();
  }, 3000);
}

boot().catch(error => {
  document.getElementById("runtimeSummary").textContent = error.message;
});
