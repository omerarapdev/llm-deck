/* Small UI toolkit shared by every slide: DOM helpers, number formatting,
 * token chips, color scales, controls, bar lists, heat strips, a line/scatter
 * chart, and a tooltip. No dependencies. */
(function () {
  "use strict";
  const SVGNS = "http://www.w3.org/2000/svg";
  // "?static" in the URL turns motion off, which makes automated screenshots deterministic.
  const STATIC = /[?&]static\b/.test(location.search);
  if (STATIC) document.documentElement.classList.add("static");
  const reduceMotion = () => STATIC || window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------ DOM
  function apply(el, attrs) {
    if (!attrs) return el;
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") el.setAttribute("class", v);
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "text") el.textContent = v;
      else if (k === "html") el.innerHTML = v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    return el;
  }
  function append(el, kids) {
    for (const k of kids.flat(Infinity)) {
      if (k === null || k === undefined || k === false) continue;
      el.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
    }
    return el;
  }
  const h = (tag, attrs, ...kids) => append(apply(document.createElement(tag), attrs), kids);
  const s = (tag, attrs, ...kids) => append(apply(document.createElementNS(SVGNS, tag), attrs), kids);
  const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };

  // ------------------------------------------------------------ formatting
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const fmt = {
    p: (x, d = 3) => (x >= 0.9995 && d === 3 ? "1.000" : x.toFixed(d)),
    f: (x, d = 2) => x.toFixed(d),
    pct: (x, d = 0) => (x * 100).toFixed(d) + "%",
    int: (x) => Math.round(x).toLocaleString("en-US"),
    sci(x, d = 2) {
      if (x === 0) return "0";
      const e = Math.floor(Math.log10(Math.abs(x)));
      const m = x / Math.pow(10, e);
      if (e >= -2 && e <= 3) return (+x.toPrecision(d + 1)).toLocaleString("en-US");
      return `${m.toFixed(d)} × 10${String(e).split("").map((c) => SUP[c]).join("")}`;
    },
    // decimal units (GB = 10^9 bytes), the way GPU memory is specified
    bytes(b, d = 1) {
      const u = ["B", "KB", "MB", "GB", "TB", "PB"];
      let i = 0;
      while (Math.abs(b) >= 1000 && i < u.length - 1) { b /= 1000; i++; }
      return `${i === 0 ? Math.round(b) : b.toFixed(b >= 100 ? 0 : d)} ${u[i]}`;
    },
    // binary units, for KV-cache style math where powers of two are natural
    ibytes(b, d = 1) {
      const u = ["B", "KiB", "MiB", "GiB", "TiB"];
      let i = 0;
      while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; }
      return `${i === 0 ? Math.round(b) : +b.toFixed(d)} ${u[i]}`;
    },
    signed: (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d),
    minus: (s) => String(s).replace(/^-/, "−"),
    time(sec) {
      if (sec < 1e-3) return `${(sec * 1e6).toFixed(1)} µs`;
      if (sec < 1) return `${(sec * 1e3).toFixed(sec < 0.01 ? 2 : 1)} ms`;
      if (sec < 120) return `${sec.toFixed(1)} s`;
      if (sec < 7200) return `${(sec / 60).toFixed(1)} min`;
      if (sec < 172800) return `${(sec / 3600).toFixed(1)} h`;
      return `${(sec / 86400).toFixed(1)} days`;
    },
  };

  // ------------------------------------------------------------ tokens
  const KEYWORDS = new Set(["SELECT", "FROM", "WHERE"]);
  const OPS = new Set(["=", ";"]);
  function roleOf(tok) {
    const t = tok.trim();
    if (KEYWORDS.has(t)) return "kw";
    if (OPS.has(t)) return "op";
    const D = window.MODEL_DATA;
    if (D && D.schema[t]) return "table";
    if (D && Object.values(D.schema).some((cols) => cols.includes(t))) return "col";
    if (/^[0-9]+$/.test(t)) return "val";
    return "piece";
  }
  // Leading spaces are part of tokens; draw them as a visible mark.
  function tokNode(tok) {
    const frag = document.createDocumentFragment();
    let rest = tok;
    while (rest.startsWith(" ")) {
      frag.appendChild(h("span", { class: "sp", "aria-hidden": "true" }));
      rest = rest.slice(1);
    }
    if (rest) frag.appendChild(document.createTextNode(rest));
    return frag;
  }
  const tokLabel = (tok) => (tok.startsWith(" ") ? "space + " + tok.slice(1) : tok);

  function chip(tok, opts = {}) {
    const el = h("span", {
      class: `chip role-${roleOf(tok)}${opts.cls ? " " + opts.cls : ""}`,
      "aria-label": tokLabel(tok) + (opts.id !== undefined ? `, id ${opts.id}` : ""),
      title: opts.title,
    }, h("span", { class: "chip-t" }, tokNode(tok)));
    if (opts.id !== undefined) el.appendChild(h("span", { class: "chip-id" }, String(opts.id)));
    return el;
  }

  // ------------------------------------------------------------ color
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  // sequential (0..1) on the accent hue; works in both themes through tokens
  const seq = (w) => `color-mix(in oklab, var(--accent) ${clamp(w, 0, 1) * 100}%, var(--panel))`;
  // diverging: negative blue, positive red, neutral panel at zero
  function div(v, maxAbs) {
    const w = clamp(Math.abs(v) / (maxAbs || 1), 0, 1);
    return `color-mix(in oklab, var(${v < 0 ? "--neg" : "--pos"}) ${(w * 92).toFixed(1)}%, var(--panel))`;
  }

  // ------------------------------------------------------------ tooltip
  let tipEl = null;
  function tip() {
    if (!tipEl) {
      tipEl = h("div", { class: "tip", role: "tooltip", hidden: true });
      document.body.appendChild(tipEl);
    }
    return tipEl;
  }
  function showTip(target, content) {
    const t = tip();
    clear(t);
    append(t, [content]);
    t.hidden = false;
    const r = target.getBoundingClientRect();
    const tr = t.getBoundingClientRect();
    let x = r.left + r.width / 2 - tr.width / 2;
    let y = r.top - tr.height - 8;
    if (y < 8) y = r.bottom + 8;
    x = clamp(x, 8, window.innerWidth - tr.width - 8);
    t.style.left = `${x}px`;
    t.style.top = `${y}px`;
  }
  const hideTip = () => { if (tipEl) tipEl.hidden = true; };
  // Elements with data-tip get hover/focus/tap tooltips.
  function bindTips(root) {
    root.addEventListener("pointerover", (e) => {
      const el = e.target.closest("[data-tip]");
      if (el && root.contains(el)) showTip(el, el.dataset.tip);
    });
    root.addEventListener("pointerout", (e) => {
      const el = e.target.closest("[data-tip]");
      if (el && !el.contains(e.relatedTarget)) hideTip();
    });
    root.addEventListener("focusin", (e) => {
      const el = e.target.closest("[data-tip]");
      if (el) showTip(el, el.dataset.tip);
    });
    root.addEventListener("focusout", hideTip);
  }

  // ------------------------------------------------------------ controls
  let uid = 0;
  const nextId = (p) => `${p}-${++uid}`;

  function slider({ id, label, min, max, step = 1, value, format = (v) => v, onInput, wide }) {
    id = id || nextId("rng");
    const out = h("output", { class: "ctl-val", for: id });
    const input = h("input", { type: "range", id, min, max, step, value, "aria-label": label });
    const el = h("div", { class: `ctl ctl-range${wide ? " wide" : ""}` },
      h("label", { for: id, class: "ctl-label" }, protectCase(h("span", { class: "ctl-text" }, label)), out), input);
    const sync = () => { out.textContent = format(+input.value); };
    input.addEventListener("input", () => { sync(); onInput && onInput(+input.value); });
    sync();
    el.input = input;
    el.set = (v, fire) => { input.value = v; sync(); if (fire && onInput) onInput(+input.value); };
    el.get = () => +input.value;
    return el;
  }

  function segmented({ id, label, options, value, onChange }) {
    id = id || nextId("seg");
    const btns = options.map((o) => h("button", {
      type: "button", class: "seg-btn", "aria-pressed": String(o.value === value), dataset: { v: String(o.value) },
      onclick: () => set(o.value, true),
    }, o.label));
    const el = h("div", { class: "ctl ctl-seg", role: "group", "aria-label": label, id },
      label ? h("span", { class: "ctl-label" }, protectCase(h("span", { class: "ctl-text" }, label))) : null, h("div", { class: "seg" }, btns));
    function set(v, fire) {
      value = v;
      btns.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i].value === v)));
      if (fire && onChange) onChange(v);
    }
    el.set = set;
    el.get = () => value;
    return el;
  }

  function button(label, onClick, { kind = "ghost", id, title, icon } = {}) {
    return h("button", { type: "button", class: `btn btn-${kind}`, id, title, "aria-label": title, onclick: onClick },
      icon ? h("span", { class: "btn-ico", "aria-hidden": "true" }, icon) : null, label);
  }

  // A play/pause loop helper that stops itself when the slide is left.
  function player(stepFn, { interval = 900, onState } = {}) {
    let timer = null;
    const api = {
      get playing() { return timer !== null; },
      play() {
        if (timer) return;
        timer = setInterval(() => { if (stepFn() === false) api.pause(); }, reduceMotion() ? Math.max(interval, 600) : interval);
        onState && onState(true);
      },
      pause() { if (timer) { clearInterval(timer); timer = null; onState && onState(false); } },
      toggle() { timer ? api.pause() : api.play(); },
    };
    return api;
  }

  // ------------------------------------------------------------ bars
  /* Horizontal bars. rows: [{key, label (node|string), value, max?, cls?, note?}].
   * Re-calling update with new rows animates widths. */
  function barList(container, { valueFmt = (v) => fmt.p(v), max = 1, labelWidth } = {}) {
    const root = h("div", { class: "bars" });
    if (labelWidth) root.style.setProperty("--bar-label", labelWidth);
    container.appendChild(root);
    const rowsByKey = new Map();
    function update(rows) {
      const keep = new Set();
      rows.forEach((r, idx) => {
        keep.add(r.key);
        let el = rowsByKey.get(r.key);
        if (!el) {
          el = h("div", { class: "bar-row" },
            h("div", { class: "bar-label" }), h("div", { class: "bar-track" }, h("div", { class: "bar-fill" })),
            h("div", { class: "bar-val" }));
          rowsByKey.set(r.key, el);
        }
        el.className = "bar-row" + (r.cls ? " " + r.cls : "");
        const lab = el.children[0];
        clear(lab);
        append(lab, [typeof r.label === "string" ? r.label : r.label]);
        const w = clamp((r.value / (r.max || max)) * 100, 0, 100);
        el.children[1].firstChild.style.width = `${w}%`;
        el.children[2].textContent = r.valueText !== undefined ? r.valueText : valueFmt(r.value);
        if (root.children[idx] !== el) root.insertBefore(el, root.children[idx] || null);
      });
      for (const [k, el] of rowsByKey) if (!keep.has(k)) { el.remove(); rowsByKey.delete(k); }
    }
    return { root, update };
  }

  // Top-k next-token rows for a probability vector.
  function topRows(probs, k, { target, vocab = window.TinyLM.vocab, chipOpts } = {}) {
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, k);
    if (target !== undefined && !order.some(([, i]) => i === target)) order.push([probs[target], target]);
    return order.map(([p, i]) => ({
      key: String(i), value: p, cls: i === target ? "is-target" : "",
      label: chip(vocab[i], chipOpts),
    }));
  }

  // ------------------------------------------------------------ heat strip
  function heatStrip(values, { maxAbs, label, cell = "", tipFmt = (v, i) => `dim ${i}: ${fmt.minus(v.toFixed(3))}` } = {}) {
    const m = maxAbs || Math.max(...values.map(Math.abs), 1e-9);
    return h("div", { class: `heat ${cell}`, role: "img", "aria-label": label || `vector of ${values.length} numbers` },
      values.map((v, i) => h("span", { class: "heat-c", style: { background: div(v, m) }, "data-tip": tipFmt(v, i) })));
  }

  // ------------------------------------------------------------ chart
  /* Minimal SVG chart with linear or log axes.
   * cfg: {width, height, x:{type, domain, ticks, label, fmt}, y:{...}, series:[{points, cls, label}],
   *       hlines:[{y, label, cls}], vlines:[{x, label, cls}], dots:[{x,y,label,cls}], hover:true} */
  function chart(container, cfg) {
    // On phones a 560-unit-wide chart would shrink its 11px text to ~7px; draw it narrower instead.
    const narrow = !cfg.fixedWidth && window.matchMedia && window.matchMedia("(max-width: 600px)").matches;
    const W = narrow ? Math.min(cfg.width || 640, 400) : cfg.width || 640, H = cfg.height || 300;
    const m = Object.assign({ l: 52, r: 16, t: 14, b: 40 }, cfg.margin || {});
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": cfg.aria || "chart" });
    container.appendChild(svg);
    const scale = (ax, a, b) => {
      const [d0, d1] = ax.domain;
      if (ax.type === "log") {
        const l0 = Math.log10(d0), l1 = Math.log10(d1);
        return (v) => a + ((Math.log10(Math.max(v, d0 * 1e-6)) - l0) / (l1 - l0)) * (b - a);
      }
      return (v) => a + ((v - d0) / (d1 - d0)) * (b - a);
    };
    const api = { svg };
    const clipId = nextId("clip");
    function draw() {
      clear(svg);
      svg.appendChild(s("defs", null, s("clipPath", { id: clipId }, s("rect", { x: m.l, y: m.t - 2, width: W - m.l - m.r, height: H - m.t - m.b + 4 })),
        s("clipPath", { id: clipId + "d" }, s("rect", { x: m.l - 8, y: m.t - 8, width: W - m.l - m.r + 16, height: H - m.t - m.b + 16 }))));
      const X = scale(cfg.x, m.l, W - m.r), Y = scale(cfg.y, H - m.b, m.t);
      api.X = X; api.Y = Y;
      const g = s("g", { class: "axes" });
      const xt = cfg.x.ticks || [], yt = cfg.y.ticks || [];
      const xf = cfg.x.fmt || ((v) => v), yf = cfg.y.fmt || ((v) => v);
      yt.forEach((v) => {
        g.appendChild(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }));
        g.appendChild(s("text", { x: m.l - 8, y: Y(v) + 4, class: "tick", "text-anchor": "end" }, yf(v)));
      });
      xt.forEach((v) => {
        g.appendChild(s("line", { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 5, class: "axis" }));
        g.appendChild(s("text", { x: X(v), y: H - m.b + 18, class: "tick", "text-anchor": "middle" }, xf(v)));
      });
      g.appendChild(s("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }));
      if (cfg.x.label) g.appendChild(protectCase(s("text", { x: (m.l + W - m.r) / 2, y: H - 6, class: "ax-label", "text-anchor": "middle" }, cfg.x.label)));
      if (cfg.y.label) g.appendChild(protectCase(s("text", { x: 14, y: (m.t + H - m.b) / 2, class: "ax-label", "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }, cfg.y.label)));
      svg.appendChild(g);
      for (const hl of cfg.hlines || []) {
        svg.appendChild(s("line", { x1: m.l, x2: W - m.r, y1: Y(hl.y), y2: Y(hl.y), class: `hline ${hl.cls || ""}` }));
        if (hl.label) svg.appendChild(s("text", { x: W - m.r - 4, y: Y(hl.y) - 6, class: `hl-label ${hl.cls || ""}`, "text-anchor": "end" }, hl.label));
      }
      for (const vl of cfg.vlines || []) {
        svg.appendChild(s("line", { x1: X(vl.x), x2: X(vl.x), y1: m.t, y2: H - m.b, class: `vline ${vl.cls || ""}` }));
        if (vl.label) {
          const right = X(vl.x) > W - m.r - 70;
          svg.appendChild(s("text", { x: X(vl.x) + (right ? -5 : 5), y: m.t + 12, class: `hl-label ${vl.cls || ""}`, "text-anchor": right ? "end" : "start" }, vl.label));
        }
      }
      for (const se of cfg.series || []) {
        const pts = se.points.filter((p) => p[1] !== null && isFinite(p[1]));
        if (!pts.length) continue;
        const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
        svg.appendChild(s("path", { d, class: `series ${se.cls || ""}`, fill: "none", "clip-path": cfg.clip === false ? null : `url(#${clipId})` }));
        if (se.label) {
          // label at the end of the line; flip inside the plot when the end is near the right edge
          const last = pts[pts.length - 1];
          const lx = X(last[0]), ly = Math.max(m.t + 10, Math.min(H - m.b - 4, Y(last[1])));
          const inside = lx > W - m.r - 70;
          svg.appendChild(s("text", { x: inside ? lx - 6 : lx + 6, y: ly + (inside ? -8 : 4) + (se.dy || 0), class: `s-label ${se.cls || ""}`, "text-anchor": inside ? "end" : "start" }, se.label));
        }
      }
      for (const dt of cfg.dots || []) {
        const c = s("circle", { cx: X(dt.x), cy: Y(dt.y), r: dt.r || 5, class: `dot ${dt.cls || ""}`, "clip-path": cfg.clip === false ? null : `url(#${clipId}d)` });
        if (dt.tip) c.setAttribute("data-tip", dt.tip);
        svg.appendChild(c);
        if (dt.label) svg.appendChild(s("text", { x: X(dt.x) + (dt.dx ?? 8), y: Y(dt.y) + (dt.dy ?? 4), class: `d-label ${dt.cls || ""}`, "text-anchor": dt.anchor || "start" }, dt.label));
      }
      if (cfg.after) cfg.after(svg, X, Y, { W, H, m });
    }
    api.update = (patch) => { Object.assign(cfg, patch); draw(); };
    api.cfg = cfg;
    draw();
    bindTips(svg);
    return api;
  }

  // log-scale tick helpers
  const decades = (a, b) => { const out = []; for (let e = Math.ceil(Math.log10(a)); e <= Math.floor(Math.log10(b)); e++) out.push(Math.pow(10, e)); return out; };

  // ------------------------------------------------------------ misc
  function tween(from, to, ms, onFrame, done) {
    if (reduceMotion() || ms <= 0) { onFrame(to); done && done(); return; }
    const t0 = performance.now();
    function frame(t) {
      const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
      onFrame(from + (to - from) * e);
      if (k < 1) requestAnimationFrame(frame); else done && done();
    }
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------ rich text
  // TeX via KaTeX, rendered to native MathML (no stylesheet or fonts needed).
  // Inline: \( ... \). Display: \[ ... \]. Use \lt and \gt inside TeX.
  function tex(src, display) {
    const t = document.createElement("template");
    t.innerHTML = texString(src, display);
    return t.content.firstChild;
  }
  // Browsers' MathML does not apply TeX's spacing rules: a minus after "(", "=" or ","
  // gets binary spacing, and an operator name (ln, log₂, max_k) loses the thin space
  // before its argument. Fix both on the generated MathML.
  const MATHNS = "http://www.w3.org/1998/Math/MathML";
  function fixMathML(markup) {
    const t = document.createElement("template");
    t.innerHTML = markup;
    for (const mo of t.content.querySelectorAll("mo")) {
      const c = mo.textContent.trim();
      if ((c === "−" || c === "+" || c === "±") && !mo.hasAttribute("lspace")) {
        const prev = mo.previousElementSibling;
        const prefix = !prev || (prev.localName === "mo" && prev.textContent.trim() !== "" && !/^[)\]}|⟩!′″]$/.test(prev.textContent.trim()));
        if (prefix) { mo.setAttribute("lspace", "0em"); mo.setAttribute("rspace", "0em"); }
      } else if (c === "\u2061") {
        mo.setAttribute("lspace", "0em");
        mo.setAttribute("rspace", "0em");
        // the name may be the base of a script: <msub><mrow><mi>log</mi><mo>⁡</mo></mrow><mn>2</mn></msub>
        const par = mo.parentElement;
        const scripted = par && par.localName === "mrow" && par.lastElementChild === mo && par.parentElement &&
          /^(msub|msup|msubsup|munder|mover|munderover)$/.test(par.parentElement.localName) && par.parentElement.firstElementChild === par;
        const host = scripted ? par.parentElement : mo;
        const next = host.nextElementSibling;
        if (!next || (next.localName === "mo" && /^[([{|‖⟨]/.test(next.textContent.trim()))) continue; // \exp(x): no space
        if (!scripted) mo.setAttribute("rspace", "0.17em");
        else {
          const sp = document.createElementNS(MATHNS, "mspace");
          sp.setAttribute("width", "0.17em");
          host.after(sp);
        }
      }
    }
    return t.innerHTML;
  }
  function texString(src, display) {
    if (!window.katex) return `<code class="tex-raw">${src.replace(/</g, "&lt;")}</code>`;
    return `<span class="${display ? "tex tex-d" : "tex"}">${fixMathML(window.katex.renderToString(src, { output: "mathml", displayMode: !!display, throwOnError: false, strict: "ignore" }))}</span>`;
  }
  const richString = (str) => String(str == null ? "" : str)
    .replace(/\\\[([\s\S]+?)\\\]/g, (m, t) => texString(t.trim(), true))
    .replace(/\\\(([\s\S]+?)\\\)/g, (m, t) => texString(t.trim(), false));
  // Uppercased labels would turn η into H and σ into Σ: wrap Greek runs so they keep their case.
  const GREEK = /[\u0370-\u03ff\u1f00-\u1fff]+/g;
  function protectCase(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const hits = [];
    while (walker.nextNode()) if (GREEK.test(walker.currentNode.nodeValue)) hits.push(walker.currentNode);
    GREEK.lastIndex = 0;
    for (const tn of hits) {
      if (tn.parentNode && tn.parentNode.closest && tn.parentNode.closest("math")) continue;
      const svg = tn.parentNode && tn.parentNode.namespaceURI === SVGNS;
      const frag = document.createDocumentFragment();
      let last = 0;
      tn.nodeValue.replace(GREEK, (m, at) => {
        if (at > last) frag.appendChild(document.createTextNode(tn.nodeValue.slice(last, at)));
        const w = svg ? document.createElementNS(SVGNS, "tspan") : document.createElement("span");
        w.setAttribute("class", "nocase");
        w.textContent = m;
        frag.appendChild(w);
        last = at + m.length;
        return m;
      });
      if (last < tn.nodeValue.length) frag.appendChild(document.createTextNode(tn.nodeValue.slice(last)));
      tn.parentNode.replaceChild(frag, tn);
    }
    return root;
  }
  function rich(str) {
    const t = document.createElement("template");
    t.innerHTML = richString(str);
    return t.content;
  }

  window.UI = {
    h, s, clear, append, fmt, chip, tokNode, tokLabel, roleOf, seq, div, clamp,
    slider, segmented, button, player, barList, topRows, heatStrip, chart, decades,
    showTip, hideTip, bindTips, tween, reduceMotion, nextId, tex, texString, rich, richString, protectCase,
  };
})();
