const state = {
  selectedCrops: new Set(),
  settings: {},
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

function eventStatus(event) {
  if (event.status === "new") return "new";
  if (event.label === "unknown" || String(event.label || "").startsWith("new_")) return "unknown";
  return event.status || "normal";
}

function eventHtml(event, { selectable = false } = {}) {
  const status = eventStatus(event);
  const crop = event.crop_url ? `<img class="thumb" src="${event.crop_url}" alt="">` : `<div class="thumb"></div>`;
  const checkbox = selectable && event.crop_path
    ? `<label class="check"><input type="checkbox" data-crop="${event.crop_path}"> select</label>`
    : `<span class="status ${status}">${status}</span>`;
  return `
    <article class="event-card">
      ${crop}
      <div>
        <div class="label">${text(event.label)}</div>
        <div class="meta">ID ${text(event.track_id)} · frame ${text(event.frame_index)} · score ${Number(event.score || 0).toFixed(3)}</div>
        <div class="meta">${text(event.new_group || "")}</div>
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
    return `<article class="unknown-card"><div></div><div><div class="label">${group.name}</div><div class="meta">${group.count} event(s)</div>${cards}</div><span class="status new">review</span></article>`;
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
  });
  document.getElementById("rebuildBtn").addEventListener("click", async () => {
    document.getElementById("assignOutput").textContent = "Rebuilding gallery...";
    const result = await api("/api/rebuild-gallery", { method: "POST" });
    document.getElementById("assignOutput").textContent = JSON.stringify(result, null, 2);
    await loadSettings();
  });
}

async function boot() {
  bindTabs();
  bindActions();
  await Promise.all([refreshStatus(), refreshEvents(), refreshUnknowns(), loadSettings()]);
  setInterval(() => {
    refreshStatus();
    refreshEvents();
  }, 3000);
}

boot().catch(error => {
  document.getElementById("runtimeSummary").textContent = error.message;
});
