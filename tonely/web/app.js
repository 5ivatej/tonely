// tonely front-end: upload → poll analysis → pre-plotted timeline → live playback.
// No framework, no build step: canvas 2D for every chart.

const $ = (s) => document.querySelector(s);
const STAGES = [
  ["decode", "Decoding audio"],
  ["acoustic", "Acoustic & prosodic features"],
  ["vad", "Detecting speech"],
  ["asr", "Transcribing words"],
  ["text", "Reading emotion in words"],
  ["ser", "Voice emotion model"],
  ["ser_dim", "Voice valence / arousal model"],
  ["clap", "Sound & music mood (CLAP)"],
  ["fusion", "Fusing signals"],
];
const SOURCE_SHORT = { acoustic: "Acoustics", text: "Words", ser: "Voice (categorical)", ser_dim: "Voice (dimensional)", clap: "Sound mood" };
const SOURCE_GLYPH = { acoustic: "A", text: "W", ser: "V", ser_dim: "V", clap: "S" };

const state = {
  R: null,          // analysis result
  file: null,
  url: null,
  zoom: 24,
  disp: null,       // smoothed display values
  lastFrame: 0,
  trail: [],
  hover: null,      // {canvas, t}
  activeSeg: -1,
  activeWord: null,
};

// ------------------------------------------------------------------ theme
const css = () => getComputedStyle(document.documentElement);
let COLORS = {};
function readColors() {
  const c = css();
  const get = (n) => c.getPropertyValue(n).trim();
  COLORS = {
    surface: get("--surface"), surface2: get("--surface-2"), ink: get("--ink"), ink2: get("--ink-2"),
    muted: get("--muted"), grid: get("--grid"), axis: get("--axis"), band: get("--band"),
    future: parseFloat(get("--future")) || 0.3,
  };
  for (const e of ["sadness", "surprise", "calm", "joy", "neutral", "disgust", "fear", "anger"]) COLORS[e] = get(`--e-${e}`);
}
function initTheme() {
  try {
    const t = localStorage.getItem("tonely-theme");
    if (t) document.documentElement.dataset.theme = t;
  } catch {}
  readColors();
  $("#btn-theme").onclick = () => {
    const cur = document.documentElement.dataset.theme
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("tonely-theme", next); } catch {}
    readColors();
    redrawAll();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); redrawAll(); });
}

// ------------------------------------------------------------------ utils
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const fmt = (s) => {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
};
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const pct = (x) => `${Math.round(x * 100)}%`;
function hexA(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function toast(msg, ms = 4000) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}

// sample a per-grid-point series at time t (linear interpolation)
function idxAt(t) {
  const g = state.R.grid;
  const f = clamp(t / g.hop - 0.5, 0, g.times.length - 1);
  const i = Math.floor(f);
  return [i, Math.min(i + 1, g.times.length - 1), f - i];
}
function at(arr, t) {
  if (!arr) return null;
  const [i, j, w] = idxAt(t);
  const a = arr[i], b = arr[j];
  if (a == null) return b;
  if (b == null) return a;
  return a + (b - a) * w;
}
function probsAt(t) {
  const P = state.R.fused.probs;
  const [i, j, w] = idxAt(t);
  return P[i].map((p, k) => p + (P[j][k] - p) * w);
}
const argmax = (a) => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);

// canvas setup with device-pixel-ratio scaling
function fit(cv) {
  const dpr = window.devicePixelRatio || 1;
  if (!cv.dataset.h) cv.dataset.h = cv.getAttribute("height") || "150";
  const w = cv.clientWidth;
  const h = cv.classList.contains("square") ? w : +cv.dataset.h;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    if (!cv.classList.contains("square")) cv.style.height = h + "px";
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, h];
}

// ------------------------------------------------------------------ upload
function initUpload() {
  const dz = $("#dropzone"), input = $("#file-input");
  input.onchange = () => input.files[0] && start(input.files[0]);
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  for (const ev of ["dragenter", "dragover"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) dz.addEventListener(ev, () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) start(f);
  });
  // allow dropping anywhere on the upload view
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    if (!$("#view-upload").hidden && e.dataTransfer.files[0]) start(e.dataTransfer.files[0]);
  });
  $("#btn-new").onclick = reset;
  $("#btn-export").onclick = exportJSON;
}

function show(view) {
  for (const v of ["upload", "progress", "player"]) $(`#view-${v}`).hidden = v !== view;
  $("#btn-new").hidden = view === "upload";
  $("#btn-export").hidden = view !== "player";
}

function reset() {
  const a = $("#audio");
  a.pause();
  if (state.url) URL.revokeObjectURL(state.url);
  Object.assign(state, { R: null, file: null, url: null, trail: [], disp: null, activeSeg: -1 });
  $("#file-input").value = "";
  show("upload");
}

async function start(file) {
  state.file = file;
  show("progress");
  $("#progress-file").textContent = file.name;
  renderStages(null, 0);
  const fd = new FormData();
  fd.append("file", file);
  let id;
  try {
    const r = await fetch("/api/jobs", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    id = (await r.json()).id;
  } catch (e) {
    toast(`Upload failed: ${e.message}`);
    return reset();
  }
  poll(id);
}

const seenStages = new Set();
function renderStages(current, progress) {
  if (current == null) seenStages.clear();
  else seenStages.add(current);
  $("#progress-bar").style.width = pct(progress);
  $("#progress-pct").textContent = pct(progress);
  const ol = $("#stage-list");
  ol.innerHTML = "";
  const curIdx = STAGES.findIndex(([id]) => id === current);
  STAGES.forEach(([id, label], i) => {
    const li = document.createElement("li");
    const cls = id === current ? "active" : (i < curIdx || (seenStages.has(id) && id !== current)) ? "done" : "pending";
    li.className = cls;
    li.innerHTML = `<span class="st"></span><span>${label}</span>`;
    ol.appendChild(li);
  });
}

async function poll(id) {
  for (;;) {
    let j;
    try {
      j = await (await fetch(`/api/jobs/${id}`)).json();
    } catch {
      await new Promise((r) => setTimeout(r, 1000));
      continue;
    }
    if (j.status === "error") {
      toast(`Analysis failed: ${j.error}`, 8000);
      return reset();
    }
    renderStages(j.stage, j.progress || 0);
    if (j.status === "done") break;
    await new Promise((r) => setTimeout(r, 350));
  }
  const R = await (await fetch(`/api/jobs/${id}/result`)).json();
  load(R);
}

function exportJSON() {
  const blob = new Blob([JSON.stringify(state.R, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = state.R.file.name.replace(/\.[^.]+$/, "") + ".tonely.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ------------------------------------------------------------------ player
function load(R) {
  state.R = R;
  state.trail = [];
  state.disp = { v: 0, a: 0, probs: R.emotions.map(() => 1 / R.emotions.length), src: {} };
  state.url = URL.createObjectURL(state.file);
  const audio = $("#audio");
  audio.src = state.url;
  show("player");
  $("#file-name").textContent = R.file.name;
  buildLegend();
  buildBars();
  buildSources();
  buildTranscript();
  buildStats();
  buildMoments();
  buildPipeline();
  requestAnimationFrame(() => { redrawAll(); tick(); });
}

function buildLegend() {
  $("#legend").innerHTML = state.R.emotions
    .map((e) => `<span><i style="background:${COLORS[e]}"></i>${e}</span>`).join("");
}

function buildBars() {
  $("#emo-bars").innerHTML = state.R.emotions.map((e) => `
    <li data-e="${e}"><span class="dot" style="background:${COLORS[e]}"></span><span class="name">${e}</span>
    <span class="track"><span class="fill" style="display:block;background:${COLORS[e]}"></span></span><span class="val">0%</span></li>`).join("");
}

function buildSources() {
  const ul = $("#sources");
  ul.innerHTML = state.R.fused.sources.map((s) => {
    const dim = s.valence.some((v) => v != null);
    return `
    <li data-src="${s.name}">
      <div class="src-head"><span class="src-name">${SOURCE_SHORT[s.name] || s.label}</span><span class="src-top"></span></div>
      <div class="src-rows">
        ${dim ? `<span>valence</span><div class="bipolar"><div class="bipolar-fill" data-k="v"></div></div>
        <span>arousal</span><div class="bipolar"><div class="bipolar-fill" data-k="a"></div></div>` : ""}
        <span>trust</span><div class="unipolar"><div class="unipolar-fill" data-k="w"></div></div>
      </div>
      <div class="muted small" style="margin-top:2px">${s.label}</div>
    </li>`;
  }).join("");
}

function buildTranscript() {
  const box = $("#transcript");
  const T = state.R.transcript;
  const lang = $("#lang");
  if (!T || !T.segments.length) {
    box.innerHTML = `<p class="empty-note">${T ? "No speech was recognised." : "No speech detected, so this read is based on sound, music and acoustics."}</p>`;
    lang.hidden = true;
    return;
  }
  lang.hidden = false;
  lang.textContent = `${T.language.toUpperCase()} · ${pct(T.language_probability)}`;
  box.innerHTML = T.segments.map((s, i) => {
    const words = s.words.length
      ? s.words.map((w, k) => `<span class="w" data-s="${i}" data-k="${k}">${escapeHTML(w.w)}</span>`).join("")
      : escapeHTML(s.text);
    const col = COLORS[s.emotion] || COLORS.axis;
    return `<div class="seg" data-i="${i}" style="border-left-color:${col}">
      <div class="meta"><span>${fmt(s.start)}</span><span>${s.emotion || ""}${s.emotion_p ? " " + pct(s.emotion_p) : ""}</span>
      <span>sentiment ${s.sentiment > 0 ? "+" : ""}${(s.sentiment ?? 0).toFixed(2)}</span></div>
      <div>${words}</div>${s.translation ? `<div class="tr">${escapeHTML(s.translation)}</div>` : ""}</div>`;
  }).join("");
  box.onclick = (e) => {
    const seg = e.target.closest(".seg");
    if (!seg) return;
    const s = T.segments[+seg.dataset.i];
    const w = e.target.closest(".w");
    seek(w ? s.words[+w.dataset.k].s : s.start);
  };
}
function escapeHTML(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function buildStats() {
  const R = state.R, S = R.fused.summary, A = R.acoustic.summary;
  const dom = S.dominant_emotion;
  const stat = (k, v, s, dot) => `<div class="stat"><div class="k">${k}</div><div class="v">${dot ? `<span class="dot" style="background:${COLORS[dot]}"></span>` : ""}${v}</div><div class="s">${s}</div></div>`;
  $("#stats").innerHTML = [
    stat("Overall mood", dom, `${pct(S.emotion_mix[dom])} average probability`, dom),
    stat("Sentiment", S.sentiment, `valence ${S.valence >= 0 ? "+" : ""}${S.valence.toFixed(2)} · ${pct(S.positivity)} positive / ${pct(S.negativity)} negative`),
    stat("Energy", S.arousal > 0.35 ? "high" : S.arousal > 0 ? "moderate" : S.arousal > -0.35 ? "low" : "very low", `arousal ${S.arousal >= 0 ? "+" : ""}${S.arousal.toFixed(2)}`),
    stat("Emotional arc", S.arc.shape.split(" (")[0], S.arc.shape.includes("(") ? S.arc.shape.match(/\((.*)\)/)[1] : `fit ${S.arc.fit}`),
    stat("Volatility", S.volatility > 1.5 ? "high" : S.volatility > 0.7 ? "medium" : "low", `${S.transitions} emotion shifts`),
    stat("Content", R.content.kind, `${A.key} · ${Math.round(A.tempo_bpm)} BPM`),
  ].join("");
  const share = Object.entries(S.time_share).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  $("#share").innerHTML = share.map(([e, v]) =>
    `<div style="flex:${v};background:${COLORS[e]}" title="${cap(e)} ${pct(v)}">${v > 0.08 ? `${e} ${pct(v)}` : ""}</div>`).join("");
}

function buildMoments() {
  const M = state.R.fused.moments;
  const icon = { peak: "▲", high: "＋", low: "－", shift: "↻" };
  $("#moments").innerHTML = M.length
    ? M.map((m) => `<li data-t="${m.t}"><span class="t">${fmt(m.t)}</span><span class="k">${icon[m.kind] || "•"}</span><span>${m.title}</span></li>`).join("")
    : `<p class="empty-note">No pronounced moments, the mood stays fairly even.</p>`;
  $("#moments").onclick = (e) => { const li = e.target.closest("li"); if (li) seek(+li.dataset.t); };
}

function buildPipeline() {
  const R = state.R;
  $("#pipeline").innerHTML = R.analyzers.map((a) =>
    `<tr><td>${a.label}</td><td class="${a.status === "ok" ? "ok" : "bad"}">${a.status === "ok" ? "✓ ok" : a.status}</td><td>${a.seconds}s</td></tr>`).join("")
    + `<tr><td>Device</td><td class="ok">${R.device}</td><td></td></tr>`;
}

// ------------------------------------------------------------------ playback
const audio = () => $("#audio");
function seek(t) {
  const a = audio();
  a.currentTime = clamp(t, 0, state.R.file.duration);
  state.trail = [];
  if (a.paused) { snapDisplay(); redrawAll(); }
}
function togglePlay() {
  const a = audio();
  if (a.paused) a.play().catch((e) => toast(`Can't play this file in the browser: ${e.message}`));
  else a.pause();
}
function initPlayer() {
  $("#btn-play").onclick = togglePlay;
  const a = audio();
  a.addEventListener("play", () => { $("#btn-play").classList.add("playing"); $("#btn-play").setAttribute("aria-label", "Pause"); });
  a.addEventListener("pause", () => { $("#btn-play").classList.remove("playing"); $("#btn-play").setAttribute("aria-label", "Play"); });
  a.addEventListener("ended", () => { $("#btn-play").classList.remove("playing"); });
  a.addEventListener("error", () => toast("Your browser can't decode this format for playback; the analysis is still shown."));
  document.addEventListener("keydown", (e) => {
    if (!state.R || $("#view-player").hidden || e.target.matches("input, textarea")) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.code === "ArrowRight") seek(a.currentTime + 5);
    if (e.code === "ArrowLeft") seek(a.currentTime - 5);
  });
  document.querySelectorAll("[data-zoom]").forEach((b) => b.onclick = () => {
    document.querySelectorAll("[data-zoom]").forEach((x) => x.classList.toggle("on", x === b));
    state.zoom = +b.dataset.zoom;
    redrawAll();
  });
  $("#tg-sources").onchange = () => drawCirc();
  bindSeek($("#cv-overview"), (x, w) => (x / w) * state.R.file.duration);
  bindSeek($("#cv-live"), (x, w) => liveTimeAt(x, w));
  bindSeek($("#cv-stream"), (x, w) => liveTimeAt(x, w));
  bindHover($("#cv-overview"), (x, w) => (x / w) * state.R.file.duration);
  bindHover($("#cv-live"), (x, w) => liveTimeAt(x, w));
  bindHover($("#cv-stream"), (x, w) => liveTimeAt(x, w));
  window.addEventListener("resize", () => redrawAll());
}
function bindSeek(cv, toT) {
  let drag = false;
  const go = (e) => { const r = cv.getBoundingClientRect(); seek(toT(e.clientX - r.left, r.width)); };
  cv.addEventListener("pointerdown", (e) => { drag = cv.id === "cv-overview"; go(e); });
  cv.addEventListener("pointermove", (e) => { if (drag) go(e); });
  window.addEventListener("pointerup", () => (drag = false));
}
function bindHover(cv, toT) {
  cv.addEventListener("pointermove", (e) => {
    const r = cv.getBoundingClientRect();
    const t = toT(e.clientX - r.left, r.width);
    if (t < 0 || t > state.R.file.duration) return hideTip();
    state.hover = { cv: cv.id, t };
    showTip(e.clientX, e.clientY, t);
    if (audio().paused) redrawAll();
  });
  cv.addEventListener("pointerleave", () => { state.hover = null; hideTip(); if (audio().paused) redrawAll(); });
}
function showTip(x, y, t) {
  const p = probsAt(t), e = state.R.emotions[argmax(p)];
  const v = at(state.R.fused.valence, t), a = at(state.R.fused.arousal, t);
  const tip = $("#tooltip");
  tip.innerHTML = `<div class="tt-t">${fmt(t)}</div>
    <div class="tt-e"><span class="dot" style="background:${COLORS[e]}"></span>${e} ${pct(p[argmax(p)])}</div>
    <div class="tt-r">valence ${v >= 0 ? "+" : ""}${v.toFixed(2)} · arousal ${a >= 0 ? "+" : ""}${a.toFixed(2)}</div>`;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = clamp(x + 14, 8, innerWidth - w - 8) + "px";
  tip.style.top = clamp(y - h - 12, 8, innerHeight - h - 8) + "px";
}
const hideTip = () => ($("#tooltip").hidden = true);

const LIVE_HEAD = 0.42; // playhead position across the live chart
function liveTimeAt(x, w) {
  return audio().currentTime + (x / w - LIVE_HEAD) * state.zoom;
}

// ------------------------------------------------------------------ animation loop
function snapDisplay() {
  const t = audio().currentTime;
  state.disp.v = at(state.R.fused.valence, t);
  state.disp.a = at(state.R.fused.arousal, t);
  state.disp.probs = probsAt(t);
}

function tick(now = performance.now()) {
  if (!state.R) return;
  const dt = Math.min(0.1, (now - (state.lastFrame || now)) / 1000);
  state.lastFrame = now;
  const a = audio();
  const t = a.currentTime;
  const R = state.R;

  // critically-damped-ish smoothing toward the interpolated targets so the
  // dot and bars glide instead of stepping at the analysis hop.
  const k = 1 - Math.exp(-dt / 0.12);
  const tv = at(R.fused.valence, t), ta = at(R.fused.arousal, t), tp = probsAt(t);
  const D = state.disp;
  D.v += (tv - D.v) * k;
  D.a += (ta - D.a) * k;
  D.probs = D.probs.map((p, i) => p + (tp[i] - p) * k);
  for (const s of R.fused.sources) {
    const cur = D.src[s.name] || (D.src[s.name] = { v: 0, a: 0, w: 0 });
    const sv = at(s.valence, t), sa = at(s.arousal, t);
    const sw = sv != null ? at(s.weight, t) : at(s.cat_weight, t);
    if (sv != null) cur.v += (sv - cur.v) * k;
    if (sa != null) cur.a += (sa - cur.a) * k;
    cur.w += ((sw ?? 0) - cur.w) * k;
    cur.dim = sv != null;
    cur.has = (sw ?? 0) > 0.03;
  }

  if (!a.paused) {
    const last = state.trail[state.trail.length - 1];
    if (!last || Math.abs(t - last.t) > 0.05) state.trail.push({ t, v: D.v, a: D.a });
    while (state.trail.length && t - state.trail[0].t > 8) state.trail.shift();
  }

  updateNow(t);
  drawLive();
  drawStream();
  drawCirc();
  drawOverview();
  updateBars();
  updateSources(t);
  updateTranscript(t);
  requestAnimationFrame(tick);
}

function redrawAll() {
  if (!state.R || $("#view-player").hidden) return;
  readColors();
  drawLive(); drawStream(); drawCirc(); drawOverview();
}

function updateNow(t) {
  const R = state.R, D = state.disp;
  const i = argmax(D.probs), e = R.emotions[i];
  $("#clock").textContent = `${fmt(t)} / ${fmt(R.file.duration)}`;
  const playing = !audio().paused || t > 0;
  $("#now-label").textContent = playing ? e : "Ready";
  $("#now-dot").style.background = playing ? COLORS[e] : COLORS.muted;
  $("#now-conf").textContent = playing ? `${pct(D.probs[i])} confident` : "";
  if (playing) {
    const sent = D.v > 0.15 ? "positive" : D.v < -0.15 ? "negative" : "neutral";
    const energy = D.a > 0.35 ? "high energy" : D.a > -0.1 ? "moderate energy" : "low energy";
    const second = [...D.probs.keys()].sort((x, y) => D.probs[y] - D.probs[x])[1];
    const tail = D.probs[second] > 0.18 ? ` · with a hint of ${R.emotions[second]}` : "";
    $("#now-sub").textContent = `${cap(sent)} · ${energy}${tail}`;
  }
  setBipolar($("#m-val"), D.v); $("#m-val-n").textContent = (D.v >= 0 ? "+" : "") + D.v.toFixed(2);
  setBipolar($("#m-aro"), D.a); $("#m-aro-n").textContent = (D.a >= 0 ? "+" : "") + D.a.toFixed(2);
  const inten = clamp(Math.hypot(D.v, D.a) / 1.2, 0, 1);
  $("#m-int").style.width = pct(inten); $("#m-int-n").textContent = pct(inten);
  $("#m-val").style.background = $("#m-aro").style.background = $("#m-int").style.background = COLORS[e];
}
function setBipolar(el, x) {
  x = clamp(x, -1, 1);
  el.style.left = x >= 0 ? "50%" : `${50 + x * 50}%`;
  el.style.width = `${Math.abs(x) * 50}%`;
}

function updateBars() {
  const D = state.disp, top = argmax(D.probs);
  document.querySelectorAll("#emo-bars li").forEach((li, i) => {
    li.classList.toggle("top", i === top);
    li.querySelector(".fill").style.width = pct(D.probs[i]);
    li.querySelector(".val").textContent = pct(D.probs[i]);
  });
}

function updateSources(t) {
  const R = state.R;
  for (const s of R.fused.sources) {
    const li = document.querySelector(`#sources [data-src="${s.name}"]`);
    if (!li) continue;
    const cur = state.disp.src[s.name];
    li.classList.toggle("off", !cur.has);
    if (cur.dim) {
      setBipolar(li.querySelector('[data-k="v"]'), cur.has ? cur.v : 0);
      setBipolar(li.querySelector('[data-k="a"]'), cur.has ? cur.a : 0);
    }
    li.querySelector('[data-k="w"]').style.width = pct(clamp(cur.w, 0, 1));
    const [i] = idxAt(t);
    const top = s.top ? s.top[i] : null;
    li.querySelector(".src-top").innerHTML = !cur.has && !top ? "silent here"
      : top ? `<span class="dot" style="background:${COLORS[top]}"></span>${top}` : "dimensional only";
  }
}

function updateTranscript(t) {
  const T = state.R.transcript;
  if (!T || !T.segments.length) return;
  let si = -1;
  for (let i = 0; i < T.segments.length; i++) {
    if (T.segments[i].start <= t + 0.05) si = i; else break;
  }
  if (si >= 0 && t > T.segments[si].end + 1.5) si = -1;
  if (si !== state.activeSeg) {
    document.querySelectorAll(".seg.active").forEach((x) => x.classList.remove("active"));
    if (si >= 0) {
      const el = document.querySelector(`.seg[data-i="${si}"]`);
      el.classList.add("active");
      const box = $("#transcript");
      const top = el.offsetTop - box.offsetTop - box.clientHeight / 3;
      box.scrollTo({ top, behavior: "smooth" });
    }
    state.activeSeg = si;
  }
  let wkey = null;
  if (si >= 0) {
    const W = T.segments[si].words;
    for (let k = 0; k < W.length; k++) if (W[k].s <= t && t <= W[k].e + 0.08) wkey = `${si}:${k}`;
  }
  if (wkey !== state.activeWord) {
    document.querySelectorAll(".w.on").forEach((x) => x.classList.remove("on"));
    if (wkey) {
      const [a, b] = wkey.split(":");
      document.querySelector(`.w[data-s="${a}"][data-k="${b}"]`)?.classList.add("on");
    }
    state.activeWord = wkey;
  }
}

// ------------------------------------------------------------------ charts
function drawLive() {
  const cv = $("#cv-live");
  const [ctx, W, H] = fit(cv);
  const R = state.R, F = R.fused, t0 = audio().currentTime, Z = state.zoom;
  const padL = 34, padR = 10, padT = 10, padB = 22;
  const iw = W - padL - padR, ih = H - padT - padB;
  const xOf = (t) => padL + ((t - t0) / Z + LIVE_HEAD) * iw;
  const yOf = (v) => padT + (1 - (v + 1) / 2) * ih;
  ctx.clearRect(0, 0, W, H);

  // grid
  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "middle";
  for (const v of [-1, -0.5, 0, 0.5, 1]) {
    ctx.strokeStyle = v === 0 ? COLORS.axis : COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, Math.round(yOf(v)) + 0.5); ctx.lineTo(W - padR, Math.round(yOf(v)) + 0.5); ctx.stroke();
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "right";
    ctx.fillText(v > 0 ? `+${v}` : `${v}`, padL - 6, yOf(v));
  }
  ctx.textAlign = "left";
  ctx.fillStyle = COLORS.muted;
  ctx.fillText("pleasant / energetic", padL + 4, yOf(1) + 8);
  ctx.fillText("unpleasant / calm", padL + 4, yOf(-1) - 8);

  // time ticks
  const step = Z <= 12 ? 2 : Z <= 30 ? 5 : 10;
  ctx.textAlign = "center";
  for (let s = Math.ceil((t0 - LIVE_HEAD * Z) / step) * step; s <= t0 + (1 - LIVE_HEAD) * Z; s += step) {
    if (s < 0 || s > R.file.duration) continue;
    const x = xOf(s);
    ctx.strokeStyle = COLORS.grid;
    ctx.beginPath(); ctx.moveTo(x + 0.5, padT); ctx.lineTo(x + 0.5, padT + ih); ctx.stroke();
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(fmt(s), x, H - 8);
  }

  const times = R.grid.times;
  const lo = Math.max(0, Math.floor((t0 - LIVE_HEAD * Z - 2) / R.grid.hop));
  const hi = Math.min(times.length - 1, Math.ceil((t0 + (1 - LIVE_HEAD) * Z + 2) / R.grid.hop));

  ctx.save();
  ctx.beginPath(); ctx.rect(padL, 0, iw, H); ctx.clip();

  // segments ribbon along the bottom
  for (const s of F.segments) {
    const x1 = Math.max(xOf(s.start), padL), x2 = Math.min(xOf(s.end), W - padR);
    if (x2 <= x1) continue;
    ctx.fillStyle = hexA(COLORS[s.emotion], s.end <= t0 ? 0.9 : 0.35);
    ctx.fillRect(x1 + 1, padT + ih - 4, Math.max(0, x2 - x1 - 2), 4);
  }

  // uncertainty band around valence
  ctx.fillStyle = COLORS.band;
  ctx.beginPath();
  for (let i = lo; i <= hi; i++) ctx.lineTo(xOf(times[i]), yOf(clamp(F.valence[i] + F.valence_sd[i], -1, 1)));
  for (let i = hi; i >= lo; i--) ctx.lineTo(xOf(times[i]), yOf(clamp(F.valence[i] - F.valence_sd[i], -1, 1)));
  ctx.closePath(); ctx.fill();

  // arousal: dashed, past solid ink-2 / future faded
  const drawSeries = (arr, style) => {
    for (const part of ["past", "future"]) {
      ctx.save();
      ctx.beginPath();
      if (part === "past") ctx.rect(0, 0, xOf(t0), H); else ctx.rect(xOf(t0), 0, W, H);
      ctx.clip();
      ctx.globalAlpha = part === "past" ? 1 : COLORS.future;
      style(arr);
      ctx.restore();
    }
  };
  drawSeries(F.arousal, (arr) => {
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = COLORS.ink2; ctx.lineWidth = 1.5; ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = lo; i <= hi; i++) ctx.lineTo(xOf(times[i]), yOf(arr[i]));
    ctx.stroke();
    ctx.setLineDash([]);
  });
  // valence: coloured per-step by the dominant emotion
  drawSeries(F.valence, (arr) => {
    ctx.lineWidth = 2.5; ctx.lineCap = "round";
    for (let i = lo; i < hi; i++) {
      const e = R.emotions[argmax(F.probs[i])];
      ctx.strokeStyle = COLORS[e];
      ctx.beginPath();
      ctx.moveTo(xOf(times[i]), yOf(arr[i]));
      ctx.lineTo(xOf(times[i + 1]), yOf(arr[i + 1]));
      ctx.stroke();
    }
  });

  // moments
  for (const m of F.moments) {
    if (m.kind === "shift") continue;
    const x = xOf(m.t);
    if (x < padL || x > W - padR) continue;
    ctx.fillStyle = COLORS.ink2;
    ctx.globalAlpha = m.t <= t0 ? 1 : 0.5;
    ctx.beginPath();
    ctx.moveTo(x, padT + 2); ctx.lineTo(x - 4, padT - 4 + 2); ctx.lineTo(x + 4, padT - 4 + 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.restore();

  // playhead
  const px = xOf(t0);
  ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(px, padT - 4); ctx.lineTo(px, padT + ih); ctx.stroke();
  const D = state.disp, e = R.emotions[argmax(D.probs)];
  for (const [val, isV] of [[D.a, false], [D.v, true]]) {
    ctx.beginPath();
    ctx.arc(px, yOf(val), isV ? 6 : 4.5, 0, Math.PI * 2);
    ctx.fillStyle = isV ? COLORS[e] : COLORS.ink2;
    ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = COLORS.surface; ctx.stroke();
  }
  // direct labels next to the playhead dots
  ctx.font = "600 11.5px system-ui, sans-serif";
  ctx.textAlign = "left";
  const vy = yOf(D.v), ay = yOf(D.a);
  const sep = Math.abs(vy - ay) < 14 ? (vy < ay ? -7 : 7) : 0;
  ctx.fillStyle = COLORS.ink;
  ctx.fillText("valence", px + 10, vy + sep);
  ctx.fillStyle = COLORS.ink2;
  ctx.fillText("arousal", px + 10, ay - sep);

  drawHoverLine(ctx, state.hover?.cv === "cv-live" ? xOf(state.hover.t) : null, padT, padT + ih);
}

function drawHoverLine(ctx, x, y1, y2) {
  if (x == null) return;
  ctx.strokeStyle = COLORS.muted; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
  ctx.beginPath(); ctx.moveTo(x + 0.5, y1); ctx.lineTo(x + 0.5, y2); ctx.stroke();
  ctx.setLineDash([]);
}

function drawStream() {
  const cv = $("#cv-stream");
  const [ctx, W, H] = fit(cv);
  const R = state.R, P = R.fused.probs, t0 = audio().currentTime, Z = state.zoom;
  const padL = 34, padR = 10, padT = 4, padB = 4;
  const iw = W - padL - padR, ih = H - padT - padB;
  const xOf = (t) => padL + ((t - t0) / Z + LIVE_HEAD) * iw;
  ctx.clearRect(0, 0, W, H);
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = COLORS.muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  ctx.fillText("100%", padL - 6, padT + 6);
  ctx.fillText("0", padL - 6, padT + ih - 4);

  const times = R.grid.times;
  const lo = Math.max(0, Math.floor((t0 - LIVE_HEAD * Z - 2) / R.grid.hop));
  const hi = Math.min(times.length - 1, Math.ceil((t0 + (1 - LIVE_HEAD) * Z + 2) / R.grid.hop));
  ctx.save();
  ctx.beginPath(); ctx.rect(padL, 0, iw, H); ctx.clip();
  const nE = R.emotions.length;
  const base = new Float64Array(hi - lo + 1);
  for (let k = 0; k < nE; k++) {
    const e = R.emotions[k];
    ctx.beginPath();
    for (let i = lo; i <= hi; i++) ctx.lineTo(xOf(times[i]), padT + ih - (base[i - lo] + P[i][k]) * ih);
    for (let i = hi; i >= lo; i--) ctx.lineTo(xOf(times[i]), padT + ih - base[i - lo] * ih);
    ctx.closePath();
    ctx.fillStyle = COLORS[e];
    ctx.fill();
    // 1.5px surface gap between bands
    ctx.strokeStyle = COLORS.surface; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = lo; i <= hi; i++) { base[i - lo] += P[i][k]; ctx.lineTo(xOf(times[i]), padT + ih - base[i - lo] * ih); }
    ctx.stroke();
  }
  // fade the future
  const px = xOf(t0);
  ctx.fillStyle = hexA(COLORS.surface, 1 - COLORS.future - 0.15);
  ctx.fillRect(px, 0, W - px, H);
  ctx.restore();
  ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();

  // direct label for the dominant band at the playhead
  const p = probsAt(t0), k = argmax(p);
  let below = 0;
  for (let j = 0; j < k; j++) below += p[j];
  const y = padT + ih - (below + p[k] / 2) * ih;
  ctx.font = "600 12px system-ui, sans-serif"; ctx.textAlign = "right";
  const label = `${R.emotions[k]} ${pct(p[k])}`;
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = hexA(COLORS.surface, 0.85);
  ctx.fillRect(px - tw - 14, y - 9, tw + 8, 18);
  ctx.fillStyle = COLORS.ink;
  ctx.fillText(label, px - 10, y);
  drawHoverLine(ctx, state.hover?.cv === "cv-stream" ? xOf(state.hover.t) : null, 0, H);
}

function drawCirc() {
  const cv = $("#cv-circ");
  const [ctx, W, H] = fit(cv);
  const R = state.R, D = state.disp;
  const m = 34, r = (Math.min(W, H) - 2 * m) / 2, cx = W / 2, cy = H / 2;
  const P = (v, a) => [cx + v * r, cy - a * r];
  ctx.clearRect(0, 0, W, H);

  // quadrant wash in the dominant emotion of each quadrant (very light)
  ctx.save();
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.clip();
  const quads = [["joy", 1, -1], ["anger", -1, -1], ["sadness", -1, 1], ["calm", 1, 1]];
  for (const [e, sx, sy] of quads) {
    const g = ctx.createRadialGradient(cx + sx * r * 0.7, cy + sy * r * 0.7, 0, cx + sx * r * 0.7, cy + sy * r * 0.7, r);
    g.addColorStop(0, hexA(COLORS[e], 0.14));
    g.addColorStop(1, hexA(COLORS[e], 0));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  }
  ctx.restore();

  ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
  for (const rr of [0.5, 1]) { ctx.beginPath(); ctx.arc(cx, cy, r * rr, 0, Math.PI * 2); ctx.stroke(); }
  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath(); ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy); ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r); ctx.stroke();

  ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = COLORS.muted; ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.fillText("energetic", cx, cy - r - 14);
  ctx.fillText("calm", cx, cy + r + 14);
  ctx.textAlign = "left"; ctx.fillText("unpleasant", cx - r + 4, cy - 10);
  ctx.textAlign = "right"; ctx.fillText("pleasant", cx + r - 4, cy - 10);

  // prototypes
  const top = R.emotions[argmax(D.probs)];
  for (const e of R.emotions) {
    const [pv, pa] = R.prototypes[e];
    const [x, y] = P(pv, pa);
    const on = e === top;
    ctx.beginPath(); ctx.arc(x, y, on ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS[e]; ctx.fill();
    ctx.font = `${on ? "600 " : ""}11.5px system-ui, sans-serif`;
    ctx.fillStyle = on ? COLORS.ink : COLORS.ink2;
    ctx.textAlign = pv >= 0 ? "left" : "right";
    ctx.fillText(e, x + (pv >= 0 ? 8 : -8), y);
  }

  // whole-track path, faint
  const F = R.fused;
  ctx.strokeStyle = hexA(COLORS.ink2, 0.12); ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < F.valence.length; i += 1) ctx.lineTo(...P(F.valence[i], F.arousal[i]));
  ctx.stroke();

  // source markers
  if ($("#tg-sources").checked) {
    for (const s of F.sources) {
      const cur = D.src[s.name];
      if (!cur || !cur.has || !cur.dim) continue;
      const [x, y] = P(clamp(cur.v, -1, 1), clamp(cur.a, -1, 1));
      ctx.globalAlpha = 0.35 + 0.65 * clamp(cur.w, 0, 1);
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.surface; ctx.fill();
      ctx.strokeStyle = COLORS.ink2; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = COLORS.ink2; ctx.font = "600 10px system-ui, sans-serif"; ctx.textAlign = "center";
      ctx.fillText(SOURCE_GLYPH[s.name] || "?", x, y + 0.5);
      ctx.globalAlpha = 1;
    }
  }

  // trail
  const T = state.trail, now = audio().currentTime;
  for (let i = 1; i < T.length; i++) {
    const age = clamp((now - T[i].t) / 8, 0, 1);
    ctx.strokeStyle = hexA(COLORS[top], (1 - age) * 0.8);
    ctx.lineWidth = 1 + (1 - age) * 3;
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(...P(T[i - 1].v, T[i - 1].a)); ctx.lineTo(...P(T[i].v, T[i].a)); ctx.stroke();
  }

  // uncertainty ellipse + current point
  const [x, y] = P(D.v, D.a);
  const sv = at(F.valence_sd, now) || 0.1, sa = at(F.arousal_sd, now) || 0.1;
  ctx.fillStyle = hexA(COLORS[top], 0.15);
  ctx.beginPath(); ctx.ellipse(x, y, Math.max(8, sv * r), Math.max(8, sa * r), 0, 0, Math.PI * 2); ctx.fill();
  const pulse = audio().paused ? 0 : (Math.sin(performance.now() / 180) + 1) * 2;
  ctx.beginPath(); ctx.arc(x, y, 9 + pulse, 0, Math.PI * 2);
  ctx.fillStyle = hexA(COLORS[top], 0.25); ctx.fill();
  ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.fillStyle = COLORS[top]; ctx.fill();
  ctx.lineWidth = 2.5; ctx.strokeStyle = COLORS.surface; ctx.stroke();
}

function drawOverview() {
  const cv = $("#cv-overview");
  const [ctx, W, H] = fit(cv);
  const R = state.R, F = R.fused, dur = R.file.duration, t0 = audio().currentTime;
  const xOf = (t) => (t / dur) * W;
  ctx.clearRect(0, 0, W, H);
  const wfH = H - 44, mid = wfH / 2 + 4;

  // waveform coloured by dominant emotion, future dimmed
  const peaks = R.waveform, n = peaks.length;
  const bw = W / n;
  for (let i = 0; i < n; i++) {
    const t = ((i + 0.5) / n) * dur;
    const [gi] = idxAt(t);
    const e = R.emotions[argmax(F.probs[gi])];
    const [mn, mx] = peaks[i];
    const x = i * bw;
    const h1 = Math.max(1, mx * (wfH / 2)), h2 = Math.max(1, -mn * (wfH / 2));
    ctx.fillStyle = hexA(COLORS[e], t <= t0 ? 1 : 0.4);
    ctx.fillRect(x, mid - h1, Math.max(1, bw - (bw > 2 ? 0.6 : 0)), h1 + h2);
  }

  // valence curve across the whole track
  const vy = (v) => mid - v * (wfH / 2) * 0.9;
  ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.75;
  ctx.beginPath();
  R.grid.times.forEach((t, i) => ctx.lineTo(xOf(t), vy(F.valence[i])));
  ctx.stroke(); ctx.globalAlpha = 1;

  // segment ribbon
  const ry = H - 30;
  for (const s of F.segments) {
    const x1 = xOf(s.start), x2 = xOf(s.end);
    ctx.fillStyle = COLORS[s.emotion];
    roundRect(ctx, x1 + 1, ry, Math.max(1, x2 - x1 - 2), 10, 3);
    ctx.fill();
    if (x2 - x1 > 64) {
      ctx.fillStyle = COLORS.ink2; ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(s.emotion, x1 + 3, ry + 13);
    }
  }
  // moments
  for (const m of F.moments) {
    if (m.kind === "shift") continue;
    const x = xOf(m.t);
    ctx.fillStyle = COLORS.ink2;
    ctx.beginPath(); ctx.moveTo(x, 8); ctx.lineTo(x - 4, 1); ctx.lineTo(x + 4, 1); ctx.fill();
  }
  // playhead
  const px = xOf(t0);
  ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H - 18); ctx.stroke();
  drawHoverLine(ctx, state.hover?.cv === "cv-overview" ? xOf(state.hover.t) : null, 0, H - 18);
}
function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// ------------------------------------------------------------------ boot
async function boot() {
  initTheme();
  initUpload();
  initPlayer();
  try {
    const info = await (await fetch("/api/info")).json();
    const chip = $("#device-chip");
    chip.textContent = `v${info.version} · ${info.device}`;
    chip.hidden = false;
  } catch {}
}
boot();
