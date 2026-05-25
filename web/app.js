const $ = (id) => document.getElementById(id);
let jobId = null;

const el = {
  drop: $("drop"), file: $("file"), browse: $("browse"),
  progress: $("progress"), stageLabel: $("stage-label"),
  barFill: $("bar-fill"), progressMsg: $("progress-msg"),
  confirm: $("confirm"), confirmFrame: $("confirm-frame"),
  results: $("results"), resultsTitle: $("results-title"),
  clips: $("clips"), reveal: $("reveal"), again: $("again"),
};

const show = (node) => node.classList.remove("hidden");
const hide = (node) => node.classList.add("hidden");

const STAGE_LABELS = {
  probe: "Reading video", detect: "Finding the climber",
  segment: "Splitting attempts", cut: "Exporting clips",
};

// --- Drag & drop / file picker ----------------------------------------------
el.browse.addEventListener("click", () => el.file.click());
el.drop.addEventListener("click", (e) => {
  if (e.target === el.browse) return;
  el.file.click();
});
el.file.addEventListener("change", () => el.file.files[0] && start(el.file.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
el.drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) start(f);
});

el.again.addEventListener("click", () => location.reload());
el.reveal.addEventListener("click", () =>
  jobId && fetch(`/api/reveal/${jobId}`, { method: "POST" }));

// --- Flow -------------------------------------------------------------------
function start(file) {
  hide(el.drop); hide(el.confirm); hide(el.results);
  show(el.progress);
  setStage("probe", 0, "Uploading…");

  const form = new FormData();
  form.append("file", file);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      setStage("probe", e.loaded / e.total, `Uploading… ${Math.round(100 * e.loaded / e.total)}%`);
    }
  };
  xhr.onload = () => {
    if (xhr.status !== 200) return fail(`Upload failed (${xhr.status})`);
    jobId = JSON.parse(xhr.responseText).job_id;
    stream(`/api/analyze/${jobId}`);
  };
  xhr.onerror = () => fail("Upload failed");
  xhr.send(form);
}

function stream(url) {
  const es = new EventSource(url);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === "error") { es.close(); return fail(ev.message); }
    if (ev.type === "needs_confirmation") { es.close(); return askConfirm(ev); }
    if (ev.type === "done") { es.close(); return showResults(ev); }
    // progress event
    if (ev.stage) setStage(ev.stage, ev.pct ?? null, ev.message || "");
  };
  es.onerror = () => { es.close(); fail("Connection lost during processing."); };
}

function setStage(stage, pct, msg) {
  show(el.progress);
  el.stageLabel.textContent = STAGE_LABELS[stage] || "Working…";
  el.progressMsg.textContent = msg || "";
  if (pct === null || pct === undefined) {
    el.barFill.classList.add("indeterminate");
  } else {
    el.barFill.classList.remove("indeterminate");
    el.barFill.style.width = `${Math.round(pct * 100)}%`;
  }
}

// --- Climber confirmation ---------------------------------------------------
function askConfirm(ev) {
  hide(el.progress);
  show(el.confirm);
  el.confirmFrame.innerHTML = "";

  const wrap = document.createElement("div");
  wrap.className = "frame-wrap";
  const img = document.createElement("img");
  img.src = ev.frame_url;
  wrap.appendChild(img);

  img.onload = () => {
    const sx = img.clientWidth / ev.frame_w;
    const sy = img.clientHeight / ev.frame_h;
    ev.candidates.forEach((c) => {
      const hot = document.createElement("button");
      hot.className = "hotspot";
      hot.style.left = `${c.box[0] * sx}px`;
      hot.style.top = `${c.box[1] * sy}px`;
      hot.style.width = `${(c.box[2] - c.box[0]) * sx}px`;
      hot.style.height = `${(c.box[3] - c.box[1]) * sy}px`;
      hot.title = `Climber ${c.index}`;
      hot.textContent = c.index;
      hot.onclick = () => {
        hide(el.confirm);
        show(el.progress);
        setStage("cut", null, "Exporting clips…");
        stream(`/api/finalize/${jobId}?track_id=${c.track_id}`);
      };
      wrap.appendChild(hot);
    });
  };
  el.confirmFrame.appendChild(wrap);
}

// --- Results ----------------------------------------------------------------
function showResults(ev) {
  hide(el.progress); hide(el.confirm);
  show(el.results);
  const clips = ev.clips || [];
  el.resultsTitle.textContent = clips.length
    ? `${clips.length} attempt${clips.length > 1 ? "s" : ""}`
    : "No attempts found";
  el.clips.innerHTML = "";
  if (!clips.length) {
    el.clips.innerHTML = `<p class="muted">${ev.message || "Nothing detected — try adjusting thresholds in config.py."}</p>`;
    return;
  }
  clips.forEach((c) => {
    const card = document.createElement("div");
    card.className = "clip";
    const dur = c.duration ? ` · ${c.duration}s` : "";
    card.innerHTML = `
      <video src="${c.url}" controls preload="metadata"></video>
      <div class="clip-meta">
        <span>${c.name}${dur}</span>
        <a class="btn small" href="${c.url}" download>Download</a>
      </div>`;
    el.clips.appendChild(card);
  });
}

function fail(msg) {
  hide(el.progress); hide(el.confirm);
  show(el.results);
  el.resultsTitle.textContent = "Something went wrong";
  el.clips.innerHTML = `<p class="error">${msg}</p>`;
}
