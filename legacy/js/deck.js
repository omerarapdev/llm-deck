/* Deck engine: one slide at a time, the pinned query strip, the
 * "what we know so far" notes, the slide menu, theme, keyboard and swipe. */
(function () {
  "use strict";
  const { h, clear, chip } = window.UI;
  const M = window.TinyLM;
  const Widgets = (window.Widgets = window.Widgets || {});

  const PASSES = (window.PASSES = [
    { n: 0, name: "Start" },
    { n: 1, name: "The whole loop" },
    { n: 2, name: "Inside the model" },
    { n: 3, name: "Training at scale" },
    { n: 4, name: "Serving" },
    { n: 5, name: "Becoming an assistant" },
    { n: 6, name: "Wrap-up" },
  ]);

  const slides = Array.from(document.querySelectorAll(".slide"));
  const viewport = document.getElementById("slides");
  const stripTokens = document.getElementById("strip-tokens");
  const stripStage = document.getElementById("strip-stage");
  const counter = document.getElementById("nav-counter");
  const passLabel = document.getElementById("nav-pass");
  const progress = document.getElementById("nav-progress");
  const prevBtn = document.getElementById("nav-prev");
  const nextBtn = document.getElementById("nav-next");
  const notesBtn = document.getElementById("notes-btn");
  const notesCount = document.getElementById("notes-count");
  const notesPanel = document.getElementById("notes");
  const notesList = document.getElementById("notes-list");
  const menuBtn = document.getElementById("menu-btn");
  const menuPanel = document.getElementById("menu");
  const menuList = document.getElementById("menu-list");
  const themeBtn = document.getElementById("theme-btn");
  const scrim = document.getElementById("scrim");

  const store = {
    get(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } },
  };

  let current = -1;
  const inited = new Set();

  // ------------------------------------------------------------ strip
  function parseStrip(str) {
    const spec = { mode: "tokens", ids: false, focus: [], prefix: null, target: null, under: null };
    for (const part of (str || "").split(/\s+/).filter(Boolean)) {
      const [k, v] = part.split(":");
      if (k === "text" || k === "chars" || k === "tokens") spec.mode = k;
      else if (k === "ids") spec.ids = true;
      else if (k === "focus") spec.focus = v.split(",").map(Number);
      else if (k === "prefix") spec.prefix = Number(v);
      else if (k === "target") spec.target = Number(v);
      else if (k === "under") spec.under = v;
    }
    return spec;
  }

  const strip = {
    spec: null,
    render(spec) {
      this.spec = spec;
      clear(stripTokens);
      stripTokens.dataset.mode = spec.mode;
      if (spec.mode === "text") {
        stripTokens.appendChild(h("code", { class: "strip-text" }, M.data.running));
        return;
      }
      if (spec.mode === "chars") {
        Array.from(M.data.running).forEach((c) => stripTokens.appendChild(chip(c, { cls: "chip-char" })));
        return;
      }
      const toks = spec.tokens || M.running;
      const ids = toks.map(M.idOf);
      toks.forEach((t, i) => {
        let cls = "";
        if (spec.prefix !== null && i > spec.prefix) cls += " is-ghost";
        if (spec.focus.includes(i)) cls += " is-focus";
        if (spec.target === i) cls += " is-target";
        let el;
        if (spec.prefix !== null && i === spec.prefix) {
          el = h("span", { class: "chip chip-q is-target", "aria-label": "next token, unknown" }, h("span", { class: "chip-t" }, "?"));
        } else {
          el = chip(t, { id: spec.ids ? ids[i] : undefined, cls: cls.trim() });
        }
        if (spec.under && spec.under[i] !== undefined && spec.under[i] !== null) {
          el.appendChild(h("span", { class: "chip-under" }, spec.under[i]));
        }
        stripTokens.appendChild(el);
      });
    },
    // Widgets call set() to show live state; leaving the slide restores the default.
    set(patch) { this.render(Object.assign({}, this.base, patch)); },
    reset() { this.render(this.base); },
    stage(text) { stripStage.textContent = text || ""; },
  };

  // ------------------------------------------------------------ notes
  function renderNotes() {
    clear(notesList);
    let count = 0;
    for (let i = 0; i <= current; i++) {
      const f = slides[i].dataset.fact;
      if (!f) continue;
      count++;
      const li = h("li", { class: i === current ? "is-new" : "" });
      li.innerHTML = f;
      li.appendChild(h("button", { type: "button", class: "note-go", onclick: () => { closePanels(); go(i); } }, slides[i].dataset.title));
      notesList.appendChild(li);
    }
    if (!count) notesList.appendChild(h("li", { class: "empty" }, "Facts about our query collect here as the deck goes on."));
    notesCount.textContent = String(count);
  }

  // ------------------------------------------------------------ menu
  function buildMenu() {
    clear(menuList);
    for (const p of PASSES) {
      const items = slides.map((sl, i) => [sl, i]).filter(([sl]) => Number(sl.dataset.pass) === p.n);
      if (!items.length) continue;
      menuList.appendChild(h("li", { class: "menu-pass" },
        h("span", { class: "menu-pass-name" }, p.n >= 1 && p.n <= 5 ? `Pass ${p.n} · ${p.name}` : p.name),
        h("ol", { start: items[0][1] + 1 }, items.map(([sl, i]) => h("li", null,
          h("button", { type: "button", class: "menu-item", dataset: { i: String(i) }, onclick: () => { closePanels(); go(i); } },
            h("span", { class: "menu-num" }, String(i + 1)), sl.dataset.title))))));
    }
  }
  function markMenu() {
    menuList.querySelectorAll(".menu-item").forEach((b) => {
      const i = Number(b.dataset.i);
      b.classList.toggle("is-current", i === current);
      b.classList.toggle("is-seen", i < current);
      if (i === current) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current");
    });
  }

  function openPanel(panel, btn) {
    closePanels();
    panel.hidden = false;
    scrim.hidden = false;
    btn.setAttribute("aria-expanded", "true");
    const focusable = panel.querySelector(".is-current, button, [href]");
    if (focusable) focusable.focus({ preventScroll: false });
  }
  function closePanels() {
    [[notesPanel, notesBtn], [menuPanel, menuBtn]].forEach(([p, b]) => { p.hidden = true; b.setAttribute("aria-expanded", "false"); });
    scrim.hidden = true;
  }

  // ------------------------------------------------------------ glossary
  const TERMS = [
    ["token", "A unit of text from a fixed vocabulary. Our query is 9 tokens.", "bpe"],
    ["BPE", "Byte-pair encoding: builds a vocabulary by repeatedly merging the most frequent pair of neighbors.", "bpe"],
    ["embedding", "The learned vector that a token id selects from a table.", "embed"],
    ["attention", "Each position mixes in information from earlier positions, weighted by how well its query matches their keys.", "qkv"],
    ["query, key, value", "Three projections of a position's vector. Query·key decides where to look; values are what get mixed in.", "qkv"],
    ["head", "One independent attention computation. Different heads learn different jobs.", "heads"],
    ["residual stream", "The vector each position carries through the model; every sublayer adds its output to it.", "block"],
    ["MLP", "The per-position feed-forward part of a block: widen, apply a nonlinearity, narrow again.", "block"],
    ["RMSNorm", "Divides a vector by its root mean square, then multiplies by learned gains.", "block"],
    ["logit", "The raw score the model assigns a token before softmax.", "logits"],
    ["softmax", "Turns scores into probabilities: exponentiate each, divide by the total.", "logits"],
    ["temperature, top-k, top-p", "Decoding settings that reshape or truncate the next-token distribution before sampling.", "logits"],
    ["cross-entropy loss", "−ln of the probability the model gave the true next token.", "learn"],
    ["entropy, loss floor", "The data's own uncertainty: the lowest loss any model can reach.", "loss"],
    ["perplexity", "e raised to the loss: the effective number of equally likely choices.", "loss"],
    ["gradient", "How much the loss changes per unit change of each parameter.", "gradient"],
    ["backpropagation", "Computing every gradient by applying the chain rule backwards through the recorded computation.", "backprop"],
    ["learning rate", "The multiplier on each update step.", "optim"],
    ["Adam, AdamW", "Optimizers that scale each parameter's step by its own gradient history; AdamW applies weight decay separately.", "optim"],
    ["FLOPs", "Floating-point operations. Training costs about 6 × parameters × tokens.", "flops"],
    ["MFU", "Model FLOPs utilization: achieved FLOP/s divided by the hardware's peak.", "flops"],
    ["scaling laws", "Power-law fits of loss against parameters and training tokens, used to plan runs.", "scaling"],
    ["HBM", "The GPU's main memory: 80 GB read at 3.35 TB/s on an H100 SXM.", "gpu"],
    ["arithmetic intensity", "FLOPs per byte moved. Below about 295 on an H100, an operation is memory-bound.", "gpu"],
    ["BF16, FP8", "16- and 8-bit floating-point formats that save memory, bandwidth and time.", "precision"],
    ["data parallelism", "Every GPU holds the model, processes different data, and averages gradients.", "allreduce"],
    ["all-reduce", "A collective operation that leaves every GPU with the sum of all GPUs' data.", "allreduce"],
    ["ZeRO, FSDP", "Shard optimizer state, gradients and weights across data-parallel GPUs.", "zero"],
    ["tensor parallelism", "Split each layer's matrices across GPUs; partial results are summed.", "tp"],
    ["pipeline parallelism", "Split the layers across GPUs; micro-batches flow from stage to stage.", "pp"],
    ["prefill, decode", "Process the whole prompt in one pass, then generate one token per pass.", "prefill"],
    ["KV cache", "Stored keys and values of earlier tokens, reused at every decode step.", "kvcache"],
    ["continuous batching", "Adding and removing requests from the running batch at every decode step.", "batching"],
    ["PagedAttention", "Keeping the KV cache in fixed-size blocks with a block table per request.", "paged"],
    ["speculative decoding", "A small model drafts tokens and the large model checks them in one pass.", "speculative"],
    ["quantization", "Storing numbers in fewer bits, with scale factors to recover their range.", "quant"],
    ["SFT", "Supervised fine-tuning: next-token loss on curated example answers only.", "sft"],
    ["RLHF", "Reinforcement learning against a reward model trained on human preference comparisons.", "prefs"],
    ["DPO", "Direct preference optimization: learns from preference pairs without a separate reward model.", "prefs"],
    ["GRPO", "Reinforcement learning that scores each answer against its group's average instead of a learned critic.", "rl"],
  ];
  function buildGlossary() {
    const dl = document.getElementById("glossary");
    if (!dl) return;
    for (const [term, def, id] of TERMS) {
      const i = api.byId(id);
      dl.append(h("dt", null, term), h("dd", null, def, i >= 0 ? h("button", { type: "button", class: "note-go", onclick: () => { closePanels(); go(i); } }, `slide ${i + 1}: ${slides[i].dataset.title}`) : null));
    }
  }

  // ------------------------------------------------------------ progress
  function buildProgress() {
    clear(progress);
    for (const p of PASSES) {
      const n = slides.filter((sl) => Number(sl.dataset.pass) === p.n).length;
      if (!n) continue;
      progress.appendChild(h("span", { class: "seg-p", style: { flexGrow: String(n) }, dataset: { pass: String(p.n) }, title: p.name },
        h("span", { class: "seg-p-fill" })));
    }
  }
  function markProgress() {
    const pass = Number(slides[current].dataset.pass);
    progress.querySelectorAll(".seg-p").forEach((seg) => {
      const p = Number(seg.dataset.pass);
      const inPass = slides.map((sl, i) => [sl, i]).filter(([sl]) => Number(sl.dataset.pass) === p);
      const done = inPass.filter(([, i]) => i <= current).length;
      seg.firstChild.style.width = `${(done / inPass.length) * 100}%`;
      seg.classList.toggle("is-current", p === pass);
    });
  }

  // ------------------------------------------------------------ navigation
  function widgetFor(slide) {
    const el = slide.querySelector("[data-widget]");
    return el ? { el, w: Widgets[el.dataset.widget] } : null;
  }

  function go(i, { instant = false } = {}) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    if (i === current) return;
    const prev = current >= 0 ? slides[current] : null;
    const dir = i > current ? 1 : -1;
    if (prev) {
      const pw = widgetFor(prev);
      if (pw && pw.w && pw.w.leave) try { pw.w.leave(pw.el); } catch (e) { console.error(e); }
      prev.classList.remove("is-active");
      prev.setAttribute("aria-hidden", "true");
      prev.inert = true;
    }
    current = i;
    const sl = slides[i];
    sl.style.setProperty("--enter-dir", String(dir));
    sl.classList.toggle("no-anim", instant);
    sl.classList.add("is-active");
    sl.removeAttribute("aria-hidden");
    sl.inert = false;
    sl.scrollTop = 0;
    strip.base = parseStrip(sl.dataset.strip);
    strip.render(strip.base);
    strip.stage(sl.dataset.stage);
    document.body.dataset.pass = sl.dataset.pass;
    document.body.dataset.slide = sl.id;

    const cw = widgetFor(sl);
    if (cw && cw.w) {
      try {
        if (!inited.has(sl.id)) { cw.w.init(cw.el, api); inited.add(sl.id); }
        if (cw.w.enter) cw.w.enter(cw.el);
      } catch (e) {
        console.error(`widget ${cw.el.dataset.widget}`, e);
      }
    }
    const pass = Number(sl.dataset.pass);
    const pmeta = PASSES.find((p) => p.n === pass);
    passLabel.textContent = pass >= 1 && pass <= 5 ? `Pass ${pass} · ${pmeta.name}` : pmeta.name;
    counter.textContent = `${i + 1} / ${slides.length}`;
    prevBtn.disabled = i === 0;
    nextBtn.disabled = i === slides.length - 1;
    markProgress();
    markMenu();
    renderNotes();
    try { history.replaceState(null, "", "#" + sl.id); } catch (e) { /* sandboxed frames may refuse */ }
    if (i > 0) store.set("oq-last", sl.id);
    window.UI.hideTip();
  }

  const next = () => go(current + 1);
  const prev = () => go(current - 1);
  const api = { go, next, prev, strip, slides, index: () => current, byId: (id) => slides.findIndex((s) => s.id === id) };
  window.Deck = api;

  prevBtn.addEventListener("click", prev);
  nextBtn.addEventListener("click", next);
  notesBtn.addEventListener("click", () => (notesPanel.hidden ? openPanel(notesPanel, notesBtn) : closePanels()));
  menuBtn.addEventListener("click", () => (menuPanel.hidden ? openPanel(menuPanel, menuBtn) : closePanels()));
  document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closePanels));
  scrim.addEventListener("click", closePanels);
  document.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => {
    const t = b.dataset.go;
    go(t === "next" ? current + 1 : api.byId(t));
  }));

  // ------------------------------------------------------------ keyboard
  const typing = (el) => el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape") { closePanels(); return; }
    if (!menuPanel.hidden || !notesPanel.hidden) return;
    if (typing(document.activeElement)) return;
    if (["ArrowRight", "PageDown"].includes(e.key) || (e.key === " " && document.activeElement.tagName !== "BUTTON")) { e.preventDefault(); next(); }
    else if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); prev(); }
    else if (e.key === "Home") { e.preventDefault(); go(0); }
    else if (e.key === "End") { e.preventDefault(); go(slides.length - 1); }
    else if (e.key === "m") openPanel(menuPanel, menuBtn);
    else if (e.key === "n") openPanel(notesPanel, notesBtn);
  });

  // ------------------------------------------------------------ swipe
  let touch = null;
  viewport.addEventListener("touchstart", (e) => {
    if (e.touches.length !== 1) { touch = null; return; }
    const t = e.target;
    if (t.closest("input, textarea, select, button, [data-noswipe], .scroll-x")) { touch = null; return; }
    touch = { x: e.touches[0].clientX, y: e.touches[0].clientY, t: Date.now() };
  }, { passive: true });
  viewport.addEventListener("touchend", (e) => {
    if (!touch) return;
    const dx = e.changedTouches[0].clientX - touch.x, dy = e.changedTouches[0].clientY - touch.y;
    if (Math.abs(dx) > 56 && Math.abs(dx) > 1.6 * Math.abs(dy) && Date.now() - touch.t < 800) (dx < 0 ? next : prev)();
    touch = null;
  }, { passive: true });

  // ------------------------------------------------------------ theme
  function applyTheme(t) {
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
    const dark = t === "dark" || (!t && window.matchMedia("(prefers-color-scheme: dark)").matches);
    themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    themeBtn.dataset.dark = String(dark);
    document.dispatchEvent(new CustomEvent("themechange"));
  }
  themeBtn.addEventListener("click", () => {
    const dark = themeBtn.dataset.dark === "true";
    const t = dark ? "light" : "dark";
    store.set("oq-theme", t);
    applyTheme(t);
  });
  applyTheme(store.get("oq-theme"));

  // ------------------------------------------------------------ boot
  // number each pass's slides ("3 of 14"), skipping the pass openers
  for (const p of PASSES) {
    const list = slides.filter((sl) => Number(sl.dataset.pass) === p.n && !sl.classList.contains("opener"));
    list.forEach((sl, i) => {
      const span = Array.from(sl.querySelectorAll(".eyebrow span")).find((x) => /^\d+ of \d+$/.test(x.textContent.trim()));
      if (span) span.textContent = `${i + 1} of ${list.length}`;
    });
  }
  buildProgress();
  buildMenu();
  buildGlossary();
  window.UI.bindTips(document.body);
  slides.forEach((sl) => { sl.setAttribute("aria-hidden", "true"); sl.inert = true; });
  const fromHash = api.byId(decodeURIComponent((location.hash || "").slice(1)));
  go(fromHash >= 0 ? fromHash : 0, { instant: true });
  window.addEventListener("hashchange", () => {
    const j = api.byId(decodeURIComponent((location.hash || "").slice(1)));
    if (j >= 0 && j !== current) go(j);
  });
  const resume = document.getElementById("resume");
  const last = store.get("oq-last");
  if (resume && last && api.byId(last) > 0 && fromHash < 0) {
    const j = api.byId(last);
    resume.hidden = false;
    resume.querySelector("[data-resume-title]").textContent = `${j + 1}. ${slides[j].dataset.title}`;
    resume.querySelector("button").addEventListener("click", () => go(j));
  }
})();
