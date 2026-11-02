/* Deck engine. Content files describe slides as data:
 *
 *   Deck.part({ n: 2, title: "Learning from data" });
 *   Deck.chapter({ title: "Loss functions" });
 *   Deck.slide({ id, title, body, math, take, note, fig, layout, example, terms, src, check });
 *
 * Figures are rendered by components registered with Deck.fig(type, render).
 * Only the visible sheet (plus a small cache) is in the DOM at any time. */
(function () {
  "use strict";
  const { h, s, clear } = window.UI;

  const parts = [];
  const chapters = [];
  const slides = [];
  const figs = Object.create(null);
  const terms = [];
  const ids = new Set();
  let curPart = null, curChapter = null;

  const problems = []; // authoring mistakes found while loading; tools report them
  const Deck = (window.Deck = {
    parts, chapters, slides, figs, terms, problems,
    part(p) {
      curPart = Object.assign({ chapters: [], slides: [] }, p);
      parts.push(curPart);
      curChapter = null;
    },
    chapter(c) {
      if (!curPart) throw new Error("chapter before part: " + c.title);
      curChapter = Object.assign({ part: curPart, slides: [], code: `${curPart.n}.${curPart.chapters.length + 1}` }, c);
      curPart.chapters.push(curChapter);
      chapters.push(curChapter);
    },
    slide(sl) {
      if (!curPart) { problems.push(`sheet ${sl.id} comes before any Deck.part()`); return; }
      if (!sl.id || ids.has(sl.id)) {
        const orig = sl.id;
        let k = 2;
        while (ids.has(`${orig || "sheet"}-${k}`)) k++;
        sl.id = `${orig || "sheet"}-${k}`;
        problems.push(`duplicate or missing sheet id '${orig}' (renamed to '${sl.id}')`);
        console.error("Deck: " + problems[problems.length - 1]);
      }
      ids.add(sl.id);
      sl.part = curPart;
      // A part opener may come before the first chapter; it gets a chapter of its own
      // that is not listed in the part's contents.
      if (!curChapter) {
        if (sl.layout !== "part") problems.push(`sheet ${sl.id} comes before the first Deck.chapter() of part ${curPart.n}`);
        if (!curPart.lead) curPart.lead = { part: curPart, slides: [], code: `${curPart.n}.0`, title: curPart.title, lead: true };
        sl.chapter = curPart.lead;
      } else sl.chapter = curChapter;
      sl.index = slides.length;
      slides.push(sl);
      curPart.slides.push(sl);
      sl.chapter.slides.push(sl);
      for (const t of sl.terms || []) terms.push(Object.assign({ slide: sl }, t));
    },
    fig(type, render) { figs[type] = render; },
  });

  // ------------------------------------------------------------ helpers
  const pad3 = (n) => String(n).padStart(3, "0");
  const store = {
    get(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* unavailable */ } },
  };
  const html = (str) => window.UI.rich(str);

  // ------------------------------------------------------------ running-example strip
  // Slides that follow the SQL query show it as tokens; widgets can update it live.
  function parseStrip(str) {
    const spec = { mode: "tokens", ids: false, focus: [], prefix: null, target: null, under: null };
    for (const part of (str || "").split(/\s+/).filter(Boolean)) {
      const [k, v] = part.split(":");
      if (k === "text" || k === "chars" || k === "tokens") spec.mode = k;
      else if (k === "ids") spec.ids = true;
      else if (k === "focus") spec.focus = v.split(",").map(Number);
      else if (k === "prefix") spec.prefix = Number(v);
      else if (k === "target") spec.target = Number(v);
    }
    return spec;
  }
  function renderStrip(el, spec) {
    const M = window.TinyLM, chip = window.UI.chip;
    clear(el);
    el.appendChild(h("span", { class: "ex-label" }, "Example"));
    const row = h("span", { class: "ex-row" });
    el.appendChild(row);
    if (spec.mode === "text") { row.appendChild(h("code", { class: "ex-text" }, M.data.running)); return; }
    if (spec.mode === "chars") { Array.from(M.data.running).forEach((c) => row.appendChild(chip(c, { cls: "chip-char" }))); return; }
    const toks = spec.tokens || M.running;
    toks.forEach((t, i) => {
      let cls = "";
      if (spec.prefix !== null && spec.prefix !== undefined && i > spec.prefix) cls += " is-ghost";
      if ((spec.focus || []).includes(i)) cls += " is-focus";
      if (spec.target === i) cls += " is-target";
      let c;
      if (spec.prefix !== null && spec.prefix !== undefined && i === spec.prefix) c = h("span", { class: "chip chip-q is-target" }, h("span", { class: "chip-t" }, "?"));
      else c = chip(t, { id: spec.ids ? M.idOf(t) : undefined, cls: cls.trim() });
      if (spec.under && spec.under[i] != null) c.appendChild(h("span", { class: "chip-under" }, spec.under[i]));
      row.appendChild(c);
    });
  }

  // ------------------------------------------------------------ sheet rendering
  const view = () => document.getElementById("sheet-view");
  const cache = new Map(); // slide index -> { el, inst }
  const CACHE_MAX = 10;
  let current = -1;
  let figCounter = 0;

  function buildSheet(sl) {
    if (sl.layout === "part") return buildPartSheet(sl);
    const n = sl.index + 1, total = slides.length;
    const layout = sl.layout || (sl.fig ? "split" : "text");
    const sheet = h("article", { class: `sheet layout-${layout}`, id: `sheet-${sl.id}`, "aria-labelledby": `t-${sl.id}`, dataset: { part: String(sl.part.n) } });
    // margin: sheet number, section code, pencil note
    sheet.appendChild(h("aside", { class: "margin", "aria-label": "Sheet margin" },
      h("div", { class: "sheet-no", "aria-label": `Sheet ${n} of ${total}` }, pad3(n)),
      h("div", { class: "sheet-sec" }, sl.chapter.code),
      sl.note ? h("p", { class: "pencil" }, html(sl.note)) : null));
    const body = h("div", { class: "sheet-body" });
    sheet.appendChild(body);
    body.appendChild(h("header", { class: "sheet-head" },
      h("p", { class: "kicker" }, h("span", { class: "k-part" }, `Part ${sl.part.n}`), h("span", { class: "k-sep" }, "·"), h("span", null, sl.part.title), h("span", { class: "k-sep" }, "·"), h("span", null, sl.chapter.title)),
      h("h2", { class: "sheet-title", id: `t-${sl.id}` }, html(sl.title))));
    if (sl.example) {
      const strip = h("div", { class: "ex-strip", "aria-label": "Running example" });
      body.appendChild(strip);
      sl._strip = strip;
      renderStrip(strip, parseStrip(sl.example));
    }
    const cols = h("div", { class: "sheet-cols" });
    body.appendChild(cols);
    const prose = h("div", { class: "prose" });
    if (sl.body) prose.appendChild(html(sl.body));
    for (const m of [].concat(sl.math || [])) {
      const mm = typeof m === "string" ? { tex: m } : m;
      prose.appendChild(h("div", { class: "formula" }, window.UI.tex(mm.tex, true), mm.note ? h("p", { class: "formula-note" }, html(mm.note)) : null));
    }
    if (sl.take) prose.appendChild(h("div", { class: "take" }, h("span", { class: "take-tag" }, "Note"), h("p", null, html(sl.take))));
    if (sl.check && sl.fig) prose.appendChild(renderCheck(sl.check));
    if (sl.src && sl.src.length) prose.appendChild(h("p", { class: "src" }, "Source: ", sl.src.join("; ")));
    cols.appendChild(prose);
    let inst = null;
    const figSpec = sl.fig || (sl.check ? { type: "quiz", check: sl.check } : null);
    if (figSpec) {
      figCounter = sl.index + 1;
      const plate = h("figure", { class: "plate" + (figSpec.bare ? " bare" : "") });
      const caption = figSpec.caption ? h("figcaption", null, h("span", { class: "fig-no" }, `Fig. ${figCounter}`), h("span", { class: "fig-cap" }, html(figSpec.caption))) : h("figcaption", { class: "only-no" }, h("span", { class: "fig-no" }, `Fig. ${figCounter}`));
      plate.appendChild(caption);
      const holder = h("div", { class: "fig-body" });
      plate.appendChild(holder);
      cols.appendChild(plate);
      const render = figs[figSpec.type];
      if (!render) holder.appendChild(h("p", { class: "fig-missing" }, `Unknown figure type: ${figSpec.type}`));
      else {
        try { inst = render(holder, figSpec, api(sl)) || null; } catch (e) { console.error(`fig ${figSpec.type} on ${sl.id}`, e); holder.appendChild(h("p", { class: "fig-missing" }, "This figure failed to load.")); }
      }
    }
    sheet.appendChild(titleBlock(sl, n, total));
    return { el: sheet, inst };
  }

  // A part opener: big numeral, the part's promise, and its chapters with sheet ranges.
  function buildPartSheet(sl) {
    const n = sl.index + 1, total = slides.length, p = sl.part;
    const sheet = h("article", { class: "sheet layout-part", id: `sheet-${sl.id}`, "aria-labelledby": `t-${sl.id}`, dataset: { part: String(p.n) } });
    sheet.appendChild(h("aside", { class: "margin", "aria-label": "Sheet margin" },
      h("div", { class: "sheet-no" }, pad3(n)), h("div", { class: "sheet-sec" }, `Part ${p.n}`),
      sl.note ? h("p", { class: "pencil" }, html(sl.note)) : null));
    const body = h("div", { class: "sheet-body" });
    sheet.appendChild(body);
    body.appendChild(h("header", { class: "part-head" },
      h("div", { class: "part-numeral", "aria-hidden": "true" }, String(p.n).padStart(2, "0")),
      h("div", { class: "part-titles" },
        h("p", { class: "kicker" }, h("span", { class: "k-part" }, `Part ${p.n} of ${parts[parts.length - 1].n}`), h("span", { class: "k-sep" }, "·"), h("span", null, `${p.slides.length} sheets`)),
        h("h2", { class: "sheet-title", id: `t-${sl.id}` }, html(sl.title || p.title)))));
    const cols = h("div", { class: "part-cols" });
    body.appendChild(cols);
    const prose = h("div", { class: "prose" });
    if (sl.body) prose.appendChild(html(sl.body));
    if (sl.take) prose.appendChild(h("div", { class: "take" }, h("span", { class: "take-tag" }, "You will be able to"), h("p", null, html(sl.take))));
    cols.appendChild(prose);
    const list = h("ol", { class: "part-chapters" });
    for (const c of p.chapters) {
      if (!c.slides.length) continue;
      const first = c.slides[0], last = c.slides[c.slides.length - 1];
      list.appendChild(h("li", null, h("button", { type: "button", class: "pc-row", onclick: () => go(first.index) },
        h("span", { class: "pc-code" }, c.code), h("span", { class: "pc-t" }, html(c.title)),
        h("span", { class: "pc-lead", "aria-hidden": "true" }), h("span", { class: "pc-range" }, first === last ? pad3(first.index + 1) : `${pad3(first.index + 1)}–${pad3(last.index + 1)}`))));
    }
    cols.appendChild(h("nav", { class: "part-index", "aria-label": `Chapters in part ${p.n}` }, h("p", { class: "pi-head" }, "Contents of this part"), list));
    sheet.appendChild(titleBlock(sl, n, total));
    return { el: sheet, inst: null };
  }
  function titleBlock(sl, n, total) {
    return h("table", { class: "title-block", "aria-hidden": "true" }, h("tbody", null,
      h("tr", null, h("td", { colspan: "2" }, h("span", { class: "tb-l" }, "Subject"), h("span", { class: "tb-v" }, sl.chapter.title))),
      h("tr", null, h("td", null, h("span", { class: "tb-l" }, "Part"), h("span", { class: "tb-v" }, `${sl.part.n} · ${sl.part.short || sl.part.title}`)),
        h("td", null, h("span", { class: "tb-l" }, "Sheet"), h("span", { class: "tb-v" }, `${pad3(n)} / ${total}`)))));
  }

  function renderCheck(c) {
    const box = h("div", { class: "check" }, h("p", { class: "check-q" }, h("span", { class: "check-tag" }, "Check"), html(c.q)));
    const opts = h("div", { class: "check-opts", role: "group", "aria-label": "Answer options" });
    const why = h("p", { class: "check-why", hidden: true });
    c.options.forEach((o, i) => {
      const b = h("button", { type: "button", class: "opt" }, h("span", { class: "opt-l" }, "ABCDEFG"[i]), h("span", null, html(o)));
      b.addEventListener("click", () => {
        opts.querySelectorAll(".opt").forEach((x, j) => { x.classList.toggle("right", j === c.answer); x.classList.toggle("wrong", j === i && i !== c.answer); x.disabled = true; });
        why.hidden = false;
        why.innerHTML = (i === c.answer ? "<b>Right.</b> " : "<b>Not quite.</b> ") + (c.why || "");
      });
      opts.appendChild(b);
    });
    box.append(opts, why);
    return box;
  }

  // API handed to figure components
  function api(sl) {
    return {
      slide: sl,
      strip: {
        set(patch) { if (sl._strip) renderStrip(sl._strip, Object.assign(parseStrip(sl.example), patch)); },
        reset() { if (sl._strip) renderStrip(sl._strip, parseStrip(sl.example)); },
        stage() {},
      },
      go: (i) => go(i),
      byId: (id) => slides.findIndex((x) => x.id === id),
      slides: slides.map((x) => ({ dataset: { title: x.title, pass: String(x.part.n) } })),
    };
  }

  function go(i, opts = {}) {
    i = Math.max(0, Math.min(slides.length - 1, i));
    if (i === current && !opts.force) return;
    const host = view();
    const prev = cache.get(current);
    if (prev) {
      try { prev.inst && prev.inst.leave && prev.inst.leave(); } catch (e) { console.error(e); }
      prev.el.classList.remove("is-on");
      prev.el.remove();
    }
    const dir = i >= current ? 1 : -1;
    current = i;
    let entry = cache.get(i);
    if (!entry) {
      entry = buildSheet(slides[i]);
      cache.set(i, entry);
      if (cache.size > CACHE_MAX) {
        for (const k of cache.keys()) { if (k !== i) { cache.delete(k); break; } }
      }
    } else {
      cache.delete(i); cache.set(i, entry); // most recently used
    }
    entry.el.style.setProperty("--dir", String(dir));
    host.appendChild(entry.el);
    host.scrollTop = 0;
    requestAnimationFrame(() => entry.el.classList.add("is-on"));
    try { entry.inst && entry.inst.enter && entry.inst.enter(); } catch (e) { console.error(e); }
    const sl = slides[i];
    document.title = `${pad3(i + 1)} ${stripTags(sl.title)} · One Query, End to End`;
    try { history.replaceState(null, "", "#" + sl.id); } catch (e) { /* sandboxed */ }
    store.set("oq2-last", sl.id);
    updateChrome();
    window.UI.hideTip && window.UI.hideTip();
  }
  const stripTags = (s) => String(s).replace(/<[^>]+>/g, "");

  // ------------------------------------------------------------ chrome: ruler, buttons, register, terms
  let ruler, rulerCursor, rulerSvgW = 1000;
  function buildRuler() {
    const host = document.getElementById("ruler");
    const N = slides.length;
    const W = rulerSvgW, H = 30;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", class: "ruler-svg", role: "slider", "aria-label": "Position in the deck", tabindex: "0", "aria-valuemin": "1", "aria-valuemax": String(N) });
    svg.appendChild(s("line", { x1: 0, x2: W, y1: H - 1, y2: H - 1, class: "r-base" }));
    slides.forEach((sl, i) => {
      const x = ((i + 0.5) / N) * W;
      const first = i === 0 || slides[i - 1].part !== sl.part;
      const chFirst = i === 0 || slides[i - 1].chapter !== sl.chapter;
      const len = first ? 22 : chFirst ? 13 : 7;
      svg.appendChild(s("line", { x1: x, x2: x, y1: H - 1 - len, y2: H - 1, class: first ? "r-tick r-part" : chFirst ? "r-tick r-ch" : "r-tick" }));
    });
    rulerCursor = s("polygon", { class: "r-cursor", points: "0,0 0,0 0,0" });
    svg.appendChild(rulerCursor);
    host.appendChild(svg);
    const labels = h("div", { class: "ruler-labels", "aria-hidden": "true" });
    const starts = parts.filter((p) => p.slides.length).map((p) => [p.n, p.slides[0].index / N]);
    starts.forEach(([n, at], k) => {
      // skip a label that would collide with the next part's (Part 0 is only a few sheets long)
      if (k + 1 < starts.length && starts[k + 1][1] - at < 0.03) return;
      labels.appendChild(h("span", { style: { left: `${at * 100}%` } }, String(n)));
    });
    host.appendChild(labels);
    const pick = (e) => {
      const r = svg.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width;
      go(Math.floor(Math.max(0, Math.min(0.9999, x)) * N));
    };
    svg.addEventListener("click", pick);
    svg.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") { e.preventDefault(); go(current + 1); }
      if (e.key === "ArrowLeft") { e.preventDefault(); go(current - 1); }
    });
    ruler = svg;
  }
  function updateChrome() {
    const N = slides.length;
    const x = ((current + 0.5) / N) * rulerSvgW;
    if (ruler) {
      rulerCursor.setAttribute("points", `${x - 7},0 ${x + 7},0 ${x},10`);
      ruler.setAttribute("aria-valuenow", String(current + 1));
      ruler.setAttribute("aria-valuetext", `Sheet ${current + 1} of ${N}: ${stripTags(slides[current].title)}`);
    }
    const sl = slides[current];
    document.getElementById("nav-prev").disabled = current === 0;
    document.getElementById("nav-next").disabled = current === N - 1;
    document.getElementById("nav-prev-no").textContent = current > 0 ? pad3(current) : "";
    document.getElementById("nav-next-no").textContent = current < N - 1 ? pad3(current + 2) : "";
    document.getElementById("nav-where").textContent = `${sl.chapter.code} ${sl.chapter.title}`;
    document.querySelectorAll("#register .reg-row").forEach((r) => r.classList.toggle("is-current", Number(r.dataset.i) === current));
  }

  function buildRegister() {
    const list = document.getElementById("reg-list");
    for (const p of parts) {
      if (!p.slides.length) continue;
      const sec = h("section", { class: "reg-part" }, h("h3", null, h("span", { class: "reg-pn" }, `Part ${p.n}`), p.title));
      for (const c of (p.lead ? [p.lead] : []).concat(p.chapters)) {
        if (!c.slides.length) continue;
        sec.appendChild(h("h4", null, h("span", { class: "reg-code" }, c.code), c.lead ? "Opening" : c.title));
        const ol = h("ol", { class: "reg-sheets" });
        for (const sl of c.slides) {
          const q = [sl.title, c.title, p.title, sl.fig && sl.fig.caption, (sl.terms || []).map((t) => t.term).join(" ")].filter(Boolean).map(stripTags).join(" ").toLowerCase();
          ol.appendChild(h("li", null, h("button", { type: "button", class: "reg-row", dataset: { i: String(sl.index), q }, onclick: () => { closePanels(); go(sl.index); } },
            h("span", { class: "reg-no" }, pad3(sl.index + 1)), h("span", { class: "reg-t" }, html(sl.title)))));
        }
        sec.appendChild(ol);
      }
      list.appendChild(sec);
    }
    const q = document.getElementById("reg-q");
    const none = h("p", { class: "sub", hidden: true }, "No sheet title, caption or term matches.");
    list.appendChild(none);
    q.addEventListener("input", () => {
      const v = q.value.trim().toLowerCase();
      list.querySelectorAll(".reg-row").forEach((b) => { b.parentNode.hidden = v && !b.dataset.q.includes(v); });
      list.querySelectorAll("h4").forEach((h4) => {
        const ol = h4.nextElementSibling;
        h4.hidden = !!v && Array.from(ol.children).every((li) => li.hidden);
      });
      list.querySelectorAll(".reg-part").forEach((sec) => { sec.hidden = !!v && Array.from(sec.querySelectorAll("li")).every((li) => li.hidden); });
      none.hidden = !v || list.querySelector(".reg-part:not([hidden])") !== null;
    });
  }
  function buildTerms() {
    const dl = document.getElementById("terms-list");
    const sorted = terms.slice().sort((a, b) => a.term.localeCompare(b.term));
    for (const t of sorted) {
      dl.append(h("dt", null, t.term), h("dd", null, html(t.def), h("button", { type: "button", class: "term-go", onclick: () => { closePanels(); go(t.slide.index); } }, `sheet ${pad3(t.slide.index + 1)}`)));
    }
    document.getElementById("terms-count").textContent = String(terms.length);
  }

  const panels = () => ["register", "terms"].map((id) => document.getElementById(id));
  function openPanel(id) {
    closePanels();
    const p = document.getElementById(id);
    p.hidden = false;
    document.getElementById("scrim").hidden = false;
    const f = p.querySelector("input") || p.querySelector(".is-current") || p.querySelector("button");
    if (f) f.focus();
    if (id === "register") { const c = p.querySelector(".reg-row.is-current"); if (c) c.scrollIntoView({ block: "center" }); }
  }
  function closePanels() {
    panels().forEach((p) => { p.hidden = true; });
    document.getElementById("scrim").hidden = true;
  }

  // ------------------------------------------------------------ theme
  function applyTheme(t) {
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
    const dark = t === "dark" || (!t && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const b = document.getElementById("theme-btn");
    b.dataset.dark = String(dark);
    b.querySelector(".lbl").textContent = dark ? "Graph paper" : "Blueprint";
  }

  // ------------------------------------------------------------ boot
  Deck.start = function () {
    if (!slides.length) { view().textContent = "No sheets loaded."; return; }
    // each builder is independent: one failure must not take navigation down with it
    for (const [name, fn] of [["ruler", buildRuler], ["register", buildRegister], ["terms", buildTerms]]) {
      try { fn(); } catch (e) { console.error(`Deck: building the ${name} failed`, e); }
    }
    window.UI.bindTips(document.body);
    document.getElementById("nav-prev").addEventListener("click", () => go(current - 1));
    document.getElementById("nav-next").addEventListener("click", () => go(current + 1));
    document.getElementById("reg-btn").addEventListener("click", () => openPanel("register"));
    document.getElementById("terms-btn").addEventListener("click", () => openPanel("terms"));
    document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closePanels));
    document.getElementById("scrim").addEventListener("click", closePanels);
    const themeBtn = document.getElementById("theme-btn");
    themeBtn.addEventListener("click", () => { const t = themeBtn.dataset.dark === "true" ? "light" : "dark"; store.set("oq2-theme", t); applyTheme(t); });
    applyTheme(store.get("oq2-theme"));
    const typing = (el) => el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
    document.addEventListener("keydown", (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const open = panels().some((p) => !p.hidden);
      if (e.key === "Escape") { closePanels(); return; }
      if (open || typing(document.activeElement)) return;
      if (e.key === "ArrowRight" || e.key === "PageDown") { e.preventDefault(); go(current + 1); }
      else if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); go(current - 1); }
      else if (e.key === "Home") { e.preventDefault(); go(0); }
      else if (e.key === "End") { e.preventDefault(); go(slides.length - 1); }
      else if (e.key === "g" || e.key === "/") { e.preventDefault(); openPanel("register"); }
      else if (e.key === "t") { openPanel("terms"); }
    });
    let touch = null;
    const host = view();
    host.addEventListener("touchstart", (e) => {
      if (e.touches.length !== 1 || e.target.closest("input, textarea, select, button, [data-noswipe], .scroll-x, svg")) { touch = null; return; }
      touch = { x: e.touches[0].clientX, y: e.touches[0].clientY, t: Date.now() };
    }, { passive: true });
    host.addEventListener("touchend", (e) => {
      if (!touch) return;
      const dx = e.changedTouches[0].clientX - touch.x, dy = e.changedTouches[0].clientY - touch.y;
      if (Math.abs(dx) > 56 && Math.abs(dx) > 1.6 * Math.abs(dy) && Date.now() - touch.t < 800) go(current + (dx < 0 ? 1 : -1));
      touch = null;
    }, { passive: true });
    const fromHash = slides.findIndex((x) => x.id === decodeURIComponent((location.hash || "").slice(1)));
    go(fromHash >= 0 ? fromHash : 0, { force: true });
    window.addEventListener("hashchange", () => {
      const j = slides.findIndex((x) => x.id === decodeURIComponent((location.hash || "").slice(1)));
      if (j >= 0 && j !== current) go(j);
    });
    const last = store.get("oq2-last");
    const resume = document.getElementById("resume");
    if (resume && last && fromHash < 0) {
      const j = slides.findIndex((x) => x.id === last);
      if (j > 0) { resume.hidden = false; resume.textContent = `Resume at sheet ${pad3(j + 1)}`; resume.addEventListener("click", () => go(j)); }
    }
  };
  Deck.go = (i) => go(i);
  Deck.index = () => current;
  Deck.byId = (id) => slides.findIndex((x) => x.id === id);
})();
