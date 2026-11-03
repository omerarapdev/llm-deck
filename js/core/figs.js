/* Core figure components. Each is registered with Deck.fig(type, render) and
 * rendered into a plate: render(el, spec, api) may return { enter, leave }.
 * Specs are plain data; where a formula is needed, content passes a function. */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip, slider, segmented, button, player, chart, heatStrip, div, seq } = window.UI;
  const F = window.Deck.fig;
  const html = (str) => window.UI.rich(str);
  // fixed decimals with a true minus sign; values that round to zero print as 0
  const fix = (v, d) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : v).toFixed(d);
  const num = (v, d = 3) => (v === undefined || v === null || Number.isNaN(v) ? "–" : fmt.minus(fix(+v, d)));

  // shared number formatter: "int" | "f0".."f6" | "sci" | "bytes" | "ibytes" | "time" | "pct" | "pct1" | "x" | function
  function format(v, how = "f3") {
    if (typeof how === "function") return how(v);
    if (v === undefined || v === null || Number.isNaN(v)) return "–";
    if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
    if (how === "int") return Math.round(v).toLocaleString("en-US");
    if (/^f\d$/.test(how)) return fmt.minus(fix(v, +how[1]));
    if (how === "sci") return fmt.sci(v, 2);
    if (how === "bytes") return fmt.bytes(v);
    if (how === "ibytes") return fmt.ibytes(v);
    if (how === "time") return fmt.time(v);
    if (how === "pct") return fmt.pct(v);
    if (how === "pct1") return fmt.pct(v, 1);
    if (how === "x") return `${v >= 100 ? Math.round(v).toLocaleString("en-US") : v.toFixed(v >= 10 ? 1 : 2)}×`;
    return String(v);
  }
  window.UI.format = format;

  const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), window.UI.protectCase(h("div", { class: "f-tile-l" }, l)));

  // ------------------------------------------------------------ html / table / compare / specs
  F("html", (el, sp) => { el.appendChild(html(sp.html)); });

  F("table", (el, sp) => {
    // wide tables scroll on phones instead of squeezing: minWidth (px), default 560 for 5+ columns
    const minW = sp.minWidth ?? (sp.cols && sp.cols.length >= 5 ? 560 : null);
    const t = h("table", { class: "f-table" + (sp.dense ? " dense" : ""), style: minW ? { minWidth: `${minW}px` } : null });
    if (sp.cols) t.appendChild(h("thead", null, h("tr", null, sp.cols.map((c, j) => h("th", { class: sp.align && sp.align[j] === "r" ? "r" : "" }, html(c))))));
    t.appendChild(h("tbody", null, sp.rows.map((r, i) => h("tr", { class: (sp.highlight || []).includes(i) ? "hi" : "" },
      r.map((c, j) => h(j === 0 && sp.rowHeads ? "th" : "td", { class: sp.align && sp.align[j] === "r" ? "r" : "" }, html(c)))))));
    el.appendChild(h("div", { class: "scroll-x" }, t));
    if (sp.note) el.appendChild(h("p", { class: "f-note" }, html(sp.note)));
  });

  F("compare", (el, sp) => {
    const col = (c) => h("div", { class: "f-cmp-col" }, h("h4", null, html(c.title)), h("ul", null, (c.items || []).map((x) => h("li", null, html(x)))));
    el.appendChild(h("div", { class: "f-cmp" }, col(sp.left), col(sp.right)));
    if (sp.note) el.appendChild(h("p", { class: "f-note" }, html(sp.note)));
  });

  F("specs", (el, sp) => {
    el.appendChild(h("div", { class: "f-specs" }, sp.items.map((it) => h("div", { class: "f-spec" },
      h("div", { class: "f-spec-v" }, html(it.v)), h("div", { class: "f-spec-l" }, html(it.label)), it.note ? h("div", { class: "f-spec-n" }, html(it.note)) : null))));
    if (sp.note) el.appendChild(h("p", { class: "f-note" }, html(sp.note)));
  });

  // ------------------------------------------------------------ quiz (used when a slide has a check but no figure)
  F("quiz", (el, sp) => {
    const c = sp.check;
    const opts = h("div", { class: "check-opts" });
    const why = h("p", { class: "check-why", hidden: true });
    el.appendChild(h("p", { class: "check-q big" }, html(c.q)));
    c.options.forEach((o, i) => {
      const b = h("button", { type: "button", class: "opt" }, h("span", { class: "opt-l" }, "ABCDEFG"[i]), h("span", null, html(o)));
      b.addEventListener("click", () => {
        opts.querySelectorAll(".opt").forEach((x, j) => { x.classList.toggle("right", j === c.answer); x.classList.toggle("wrong", j === i && i !== c.answer); x.disabled = true; });
        why.hidden = false;
        why.innerHTML = (i === c.answer ? "<b>Right.</b> " : "<b>Not quite.</b> ") + (c.why || "");
      });
      opts.appendChild(b);
    });
    el.append(opts, why);
  });

  // ------------------------------------------------------------ steps: a worked derivation, one line at a time
  F("steps", (el, sp) => {
    const list = h("ol", { class: "f-steps" });
    const items = sp.steps.map((st, i) => {
      const li = h("li", { class: "f-step" }, h("span", { class: "f-step-n" }, String(i + 1)), h("div", { class: "f-step-body" }, html(st.html || st)), st.note ? h("span", { class: "f-step-note" }, html(st.note)) : null);
      list.appendChild(li);
      return li;
    });
    let k = sp.start == null ? items.length - 1 : sp.start;
    const show = () => items.forEach((li, i) => { li.classList.toggle("is-cur", i === k); li.classList.toggle("is-later", i > k); });
    const prev = button("◂ Back", () => { k = Math.max(0, k - 1); show(); });
    const next = button("Next step ▸", () => { k = Math.min(items.length - 1, k + 1); show(); }, { kind: "primary" });
    const from = button("From the top", () => { k = 0; show(); });
    el.append(list, h("div", { class: "f-ctl" }, from, prev, next));
    show();
  });

  // ------------------------------------------------------------ code with line notes
  const KW = {
    python: /\b(def|return|for|in|if|else|elif|import|from|as|with|class|lambda|None|True|False|and|or|not|while|range|yield|pass)\b/g,
    cuda: /\b(__global__|__device__|__host__|__shared__|__syncthreads|int|float|double|void|const|for|if|else|return|struct|unsigned|size_t|auto|half|bool|dim3|threadIdx|blockIdx|blockDim|gridDim|warpSize|cudaMalloc|cudaMemcpy|cudaFree|cudaMemcpyHostToDevice|cudaMemcpyDeviceToHost|cudaDeviceSynchronize|template|typename|while)\b/g,
    c: /\b(int|float|double|void|const|for|if|else|return|struct|unsigned|size_t|while|char|long)\b/g,
  };
  // One pass over the line, so markup inserted for one token is never rescanned
  // (a second regex pass used to match "class" inside <span class="tk-n">).
  function highlight(line, lang) {
    const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const comment = lang === "text" ? null : lang === "python" ? "#[^\\n]*" : "\\/\\/[^\\n]*";
    const parts = [comment, '"(?:[^"\\\\]|\\\\.)*"', "'(?:[^'\\\\]|\\\\.)*'", "<<<|>>>", "\\b\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?f?\\b", "[A-Za-z_][A-Za-z0-9_]*"].filter(Boolean);
    const re = new RegExp(parts.join("|"), "g");
    const kw = KW[lang] ? new RegExp(`^(?:${KW[lang].source.replace(/^\\b\(|\)\\b$/g, "")})$`) : null;
    let out = "", last = 0, m;
    while ((m = re.exec(line))) {
      out += esc(line.slice(last, m.index));
      const t = m[0];
      let cls = null;
      if (comment && (t.startsWith("#") || t.startsWith("//"))) cls = "tk-c";
      else if (t[0] === '"' || t[0] === "'") cls = "tk-s";
      else if (t === "<<<" || t === ">>>") cls = "tk-k";
      else if (/^\d/.test(t)) cls = "tk-n";
      else if (kw && kw.test(t)) cls = "tk-k";
      out += cls ? `<span class="${cls}">${esc(t)}</span>` : esc(t);
      last = m.index + t.length;
    }
    out += esc(line.slice(last));
    return out || " ";
  }
  F("code", (el, sp) => {
    const lines = sp.code.replace(/\n$/, "").split("\n");
    const notes = sp.notes || {};
    const pre = h("div", { class: "f-code", role: "region", "aria-label": `${sp.lang || "code"} listing` });
    const rows = lines.map((ln, i) => {
      const r = h("div", { class: "f-line" + ((sp.highlight || []).includes(i + 1) ? " hl" : "") + (notes[i + 1] ? " has-note" : "") },
        h("span", { class: "f-ln" }, String(i + 1)), h("code", { html: highlight(ln, sp.lang || "python") }));
      pre.appendChild(r);
      return r;
    });
    el.appendChild(h("div", { class: "scroll-x" }, pre));
    const noteKeys = Object.keys(notes).map(Number).sort((a, b) => a - b);
    if (noteKeys.length) {
      const box = h("p", { class: "f-code-note", "aria-live": "polite" });
      let k = 0;
      const show = () => {
        rows.forEach((r, i) => r.classList.toggle("cur", i + 1 === noteKeys[k]));
        box.innerHTML = `<b>Line ${noteKeys[k]}.</b> ${notes[noteKeys[k]]}`;
      };
      rows.forEach((r, i) => { if (notes[i + 1]) r.addEventListener("click", () => { k = noteKeys.indexOf(i + 1); show(); }); });
      el.append(box, h("div", { class: "f-ctl" }, button("◂ Previous line", () => { k = Math.max(0, k - 1); show(); }), button("Next line ▸", () => { k = Math.min(noteKeys.length - 1, k + 1); show(); }, { kind: "primary" })));
      show();
    }
  });

  // ------------------------------------------------------------ plot: functions with live parameters
  F("plot", (el, sp) => {
    const p = {};
    (sp.params || []).forEach((q) => { p[q.name] = q.value; });
    const W = sp.width || 560, H = sp.height || 320;
    const host = h("div", { class: "f-plot" });
    el.appendChild(host);
    const ctl = h("div", { class: "f-ctl sliders" });
    const out = h("div", { class: "f-tiles" });
    const ch = chart(host, {
      width: W, height: H, fixedWidth: sp.fixedWidth, margin: { l: 46, r: 14, t: 12, b: 38 }, aria: sp.aria || "function plot",
      x: Object.assign({ domain: sp.x || [-5, 5], ticks: sp.xticks || ticks(sp.x || [-5, 5]), label: sp.xlabel }, sp.xaxis || {}),
      y: Object.assign({ domain: sp.y || [-5, 5], ticks: sp.yticks || ticks(sp.y || [-5, 5]), label: sp.ylabel }, sp.yaxis || {}),
    });
    function draw() {
      const [x0, x1] = sp.x || [-5, 5];
      const series = (sp.fns || []).map((f, k) => {
        const pts = [];
        const n = f.samples || 240;
        for (let i = 0; i <= n; i++) {
          const x = x0 + ((x1 - x0) * i) / n;
          let y;
          try { y = f.f(x, p); } catch (e) { y = NaN; }
          pts.push([x, Number.isFinite(y) ? y : null]);
        }
        return { points: pts.filter((q) => q[1] !== null), cls: f.cls || ["ink", "s2", "s1", "s3", "s4"][k] || "ink", label: f.label ? (typeof f.label === "function" ? f.label(p) : f.label) : undefined };
      });
      const dots = (sp.points || []).map((q) => ({ x: q[0], y: q[1], r: 4.5, cls: q[2] || "ink", tip: `(${num(q[0], 2)}, ${num(q[1], 2)})` }));
      const extra = sp.extra ? sp.extra(p) : {};
      ch.update(Object.assign({ series, dots, hlines: sp.hlines || [], vlines: sp.vlines || [] }, extra, {
        after(svg, X, Y, box) {
          if (sp.residuals) {
            const f = sp.fns[0].f;
            const clipId = window.UI.nextId("rclip");
            svg.appendChild(s("clipPath", { id: clipId }, s("rect", { x: box.m.l, y: box.m.t, width: box.W - box.m.l - box.m.r, height: box.H - box.m.t - box.m.b })));
            const g = s("g", { "clip-path": `url(#${clipId})` });
            svg.appendChild(g);
            (sp.points || []).forEach((q) => {
              const yy = f(q[0], p);
              if (sp.residualSquares) {
                const side = Math.abs(Y(yy) - Y(q[1]));
                // squares open toward the side with more room
                const left = X(q[0]) + side > box.W - box.m.r;
                g.appendChild(s("rect", { x: left ? X(q[0]) - side : X(q[0]), y: Math.min(Y(yy), Y(q[1])), width: side, height: side, class: "f-resid-sq" }));
              }
              g.appendChild(s("line", { x1: X(q[0]), x2: X(q[0]), y1: Y(q[1]), y2: Y(yy), class: "f-resid" }));
            });
          }
          if (extra.after) extra.after(svg, X, Y, p, box);
        },
      }));
      clear(out);
      (sp.readouts || []).forEach((r) => out.appendChild(tile(format(r.f(p), r.fmt || "f3"), html(r.label))));
    }
    (sp.params || []).forEach((q) => {
      ctl.appendChild(slider({ label: q.label || q.name, min: q.min, max: q.max, step: q.step || 0.01, value: q.value, format: (v) => format(v, q.fmt || "f2"), onInput: (v) => { p[q.name] = v; draw(); } }));
    });
    if (sp.params && sp.params.length) el.appendChild(ctl);
    if (sp.readouts && sp.readouts.length) el.appendChild(out);
    draw();
  });
  function ticks([a, b], n = 5) {
    const span = b - a, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => span / st <= n + 1) || mag * 10;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  window.UI.ticks = ticks;

  // ------------------------------------------------------------ chart: data series
  F("chart", (el, sp) => {
    const cfg = Object.assign({ width: 560, height: 300, margin: { l: 50, r: 16, t: 12, b: 38 } }, sp.chart || {});
    cfg.x = Object.assign({ ticks: sp.x && !sp.x.ticks && sp.x.type !== "log" ? ticks(sp.x.domain) : undefined }, sp.x || {});
    cfg.y = Object.assign({ ticks: sp.y && !sp.y.ticks && sp.y.type !== "log" ? ticks(sp.y.domain) : undefined }, sp.y || {});
    cfg.series = sp.series || [];
    cfg.dots = sp.dots || [];
    cfg.hlines = sp.hlines || [];
    cfg.vlines = sp.vlines || [];
    chart(el, cfg);
    if (sp.legend) el.appendChild(h("div", { class: "legend" }, sp.legend.map(([cls, label]) => h("span", { class: "key" }, h("span", { class: `sw sw-${cls}` }), label))));
  });

  // ------------------------------------------------------------ bars
  F("bars", (el, sp) => {
    const max = sp.max || Math.max(...sp.items.map((i) => Math.abs(i.value)), 1e-9);
    el.appendChild(h("div", { class: "f-bars" }, sp.items.map((it) => h("div", { class: "f-bar" + (it.hi ? " hi" : "") },
      h("span", { class: "f-bar-l" }, html(it.label)),
      h("span", { class: "f-bar-t" }, h("span", { class: "f-bar-f" + (it.value < 0 ? " neg" : ""), style: { width: `${(Math.abs(it.value) / max) * 100}%` } })),
      h("span", { class: "f-bar-v" }, it.text !== undefined ? html(it.text) : format(it.value, sp.fmt || "f3"))))));
    if (sp.note) el.appendChild(h("p", { class: "f-note" }, html(sp.note)));
  });

  // ------------------------------------------------------------ heat: a matrix of numbers
  F("heat", (el, sp) => {
    const M = sp.matrix, R = M.length, Cn = M[0].length;
    const maxAbs = sp.max || Math.max(...M.flat().map(Math.abs), 1e-9);
    const showNums = sp.numbers !== false && R * Cn <= 64;
    const t = h("table", { class: "f-heat" + (showNums ? " nums" : "") });
    if (sp.cols) t.appendChild(h("tr", null, h("th", null, ""), sp.cols.map((c) => h("th", null, html(c)))));
    M.forEach((row, i) => t.appendChild(h("tr", null, h("th", null, sp.rows ? html(sp.rows[i]) : ""), row.map((v, j) => h("td", {
      class: Math.abs(v) / maxAbs > 0.55 ? "hot" : null,
      style: { background: sp.signed === false ? seq(Math.abs(v) / maxAbs) : div(v, maxAbs) }, "data-tip": `${sp.rows ? sp.rows[i] : i} · ${sp.cols ? sp.cols[j] : j}: ${num(v, sp.dp ?? 3)}`,
    }, showNums ? num(v, sp.dp ?? 2) : "")))));
    el.appendChild(h("div", { class: "scroll-x" }, t));
    if (sp.note) el.appendChild(h("p", { class: "f-note" }, html(sp.note)));
  });

  // ------------------------------------------------------------ matmul: C = A × B, one cell at a time
  F("matmul", (el, sp) => {
    const A = sp.A, B = sp.B, m = A.length, k = A[0].length, n = B[0].length;
    const C = Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => A[i].reduce((acc, a, t) => acc + a * B[t][j], 0)));
    const dp = sp.dp ?? 0;
    let step = sp.start ?? 0; // number of cells revealed
    const mat = (M, name, cls) => {
      const t = h("table", { class: `f-mat ${cls}` });
      M.forEach((row, i) => t.appendChild(h("tr", null, row.map((v, j) => h("td", { dataset: { i: String(i), j: String(j) } }, v === null ? "" : num(v, dp))))));
      return h("div", { class: "f-mat-wrap" }, h("div", { class: "f-mat-name" }, html(name)), t);
    };
    const wrap = h("div", { class: "f-mm" });
    const aEl = mat(A, sp.aName || `A <span class="dim">${m}×${k}</span>`, "a");
    const bEl = mat(B, sp.bName || `B <span class="dim">${k}×${n}</span>`, "b");
    const cEl = mat(C.map((r) => r.map(() => null)), sp.cName || `C = A·B <span class="dim">${m}×${n}</span>`, "c");
    wrap.append(aEl, h("div", { class: "f-mm-op" }, "×"), bEl, h("div", { class: "f-mm-op" }, "="), cEl);
    el.appendChild(h("div", { class: "scroll-x" }, wrap));
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.appendChild(expl);
    const cells = (root) => root.querySelectorAll("td");
    function show() {
      const cur = step > 0 ? step - 1 : -1;
      const ci = cur >= 0 ? Math.floor(cur / n) : -1, cj = cur >= 0 ? cur % n : -1;
      cells(aEl).forEach((td) => td.classList.toggle("on", +td.dataset.i === ci));
      cells(bEl).forEach((td) => td.classList.toggle("on", +td.dataset.j === cj));
      cells(cEl).forEach((td) => {
        const idx = +td.dataset.i * n + +td.dataset.j;
        td.textContent = idx < step ? num(C[+td.dataset.i][+td.dataset.j], dp) : "";
        td.classList.toggle("on", idx === cur);
      });
      if (cur < 0) expl.innerHTML = `Each entry of C is a dot product: one row of A with one column of B. ${m}×${n} entries, each ${k} multiplications and ${k - 1} additions.`;
      else {
        const par = (v) => (v < 0 ? `(${num(v, dp)})` : num(v, dp));
        const terms = A[ci].map((a, t) => `${par(a)}·${par(B[t][cj])}`).join(" + ");
        expl.innerHTML = `C<sub>${ci + 1}${cj + 1}</sub> = row ${ci + 1} of A · column ${cj + 1} of B = ${terms} = <b>${num(C[ci][cj], dp)}</b>`;
      }
    }
    el.appendChild(h("div", { class: "f-ctl" },
      button("Reset", () => { step = 0; show(); }),
      button("Next entry ▸", () => { step = Math.min(m * n, step + 1); show(); }, { kind: "primary" }),
      button("Fill all", () => { step = m * n; show(); })));
    show();
  });

  // ------------------------------------------------------------ graph: forward and backward through a computation
  // Each operation gives f (forward), d (its local derivatives, one per input), tex (its formula,
  // from input names or values) and dtex (each local derivative as a formula of the input names);
  // dsub, where given, is that formula with the values put in. Leaves (input, param, const) have no f.
  const OPS = {
    input: {}, param: {}, const: {},
    add: { f: (a, b) => a + b, d: () => [1, 1], tex: ([a, b]) => `${a} + ${b}`, dtex: () => ["1", "1"] },
    sub: { f: (a, b) => a - b, d: () => [1, -1], tex: ([a, b]) => `${a} - ${b}`, dtex: () => ["1", "{-1}"] },
    mul: { f: (a, b) => a * b, d: (a, b) => [b, a], tex: ([a, b]) => `${a} \\times ${b}`, dtex: ([a, b]) => [b, a] },
    div: { f: (a, b) => a / b, d: (a, b) => [1 / b, -a / (b * b)], tex: ([a, b]) => `${a} / ${b}`, dtex: ([a, b]) => [`1/${b}`, `{-${a}}/${b}^2`], dsub: ([a, b]) => [`1/${b}`, `{-${a}}/${b}^2`] },
    square: { f: (a) => a * a, d: (a) => [2 * a], tex: ([a]) => `${a}^2`, dtex: ([a]) => [`2${a}`], dsub: ([a]) => [`2 \\times ${a}`] },
    exp: { f: (a) => Math.exp(a), d: (a) => [Math.exp(a)], tex: ([a]) => `e^{${a}}`, dtex: ([a]) => [`e^{${a}}`] },
    log: { f: (a) => Math.log(a), d: (a) => [1 / a], tex: ([a]) => `\\ln ${a}`, dtex: ([a]) => [`1/${a}`], dsub: ([a]) => [`1/${a}`] },
    neg: { f: (a) => -a, d: () => [-1], tex: ([a]) => `{-${a}}`, dtex: () => ["{-1}"] },
    sigmoid: { f: (a) => 1 / (1 + Math.exp(-a)), d: (a) => { const q = 1 / (1 + Math.exp(-a)); return [q * (1 - q)]; }, tex: ([a]) => `\\sigma(${a})`, dtex: ([a]) => [`\\sigma(${a})(1 - \\sigma(${a}))`] },
    tanh: { f: (a) => Math.tanh(a), d: (a) => [1 - Math.tanh(a) ** 2], tex: ([a]) => `\\tanh ${a}`, dtex: ([a]) => [`1 - \\tanh^2 ${a}`] },
    relu: { f: (a) => Math.max(0, a), d: (a) => [a > 0 ? 1 : 0], tex: ([a]) => `\\max(0, ${a})`, dtex: (n, [a]) => [a > 0 ? "1" : "0"] },
    sum: { f: (...a) => a.reduce((x, y) => x + y, 0), d: (...a) => a.map(() => 1), tex: (n) => n.join(" + "), dtex: (n) => n.map(() => "1") },
  };

  // The small TeX subset the node labels use, as SVG text: ^ raises, _ lowers, commands become symbols.
  function svgMath(tex) {
    const src = String(tex).replace(/\\times/g, "×").replace(/\\cdot/g, "·").replace(/\\(ln|tanh|max|exp)\s*/g, "$1 ")
      .replace(/\\sigma/g, "σ").replace(/\\partial\s*/g, "∂").replace(/\\operatorname\{([^}]*)\}/g, "$1").replace(/-/g, "−");
    const runs = [];
    let cur = "", i = 0;
    const flush = () => { if (cur) runs.push([cur, 0]); cur = ""; };
    while (i < src.length) {
      const c = src[i];
      if (c === "^" || c === "_") {
        flush();
        let g;
        if (src[i + 1] === "{") {
          let depth = 1, j = i + 2;
          while (j < src.length && depth) { if (src[j] === "{") depth++; else if (src[j] === "}") depth--; j++; }
          g = src.slice(i + 2, j - 1); i = j;
        } else { g = src[i + 1] || ""; i += 2; }
        runs.push([g.replace(/[{}]/g, ""), c === "^" ? 1 : -1]);
      } else { if (c !== "{" && c !== "}") cur += c; i++; }
    }
    flush();
    return runs;
  }
  // Scripts are set at 72% and shifted with dy, which counts in the tspan's own font size.
  function svgText(attrs, tex) {
    const t = s("text", attrs);
    let at = 0; // current baseline offset, in em of the main text
    for (const [str, k] of svgMath(tex)) {
      const want = k === 1 ? -0.32 : k === -1 ? 0.2 : 0;
      const dy = k ? (want - at) / 0.72 : want - at;
      t.appendChild(s("tspan", { class: k ? "f-script" : null, dy: dy ? `${+dy.toFixed(3)}em` : null }, str));
      at = want;
    }
    return t;
  }

  // Shared by every mode of the graph figure: number formats, TeX helpers, layout and drawing.
  function graphKit(sp, BH) {
    const dp = sp.dp ?? 3, BW = sp.bw || 108;
    const short = (v) => { const r = Number(fix(v, dp)); return String(r === 0 ? 0 : r); };
    const k = {
      dp, BW, BH,
      short,
      show: (v) => fmt.minus(short(v)),
      tn: (v) => (Number(fix(v, dp)) < 0 ? `{-${short(-v)}}` : short(v)),
      T: (src) => `\\(${src}\\)`,
      // a chain "a = b = c" as one formula per link, so a narrow line can wrap before each "="
      TT: (src) => src.split(/(?= = )/).map((p) => `\\(${p.trim()}\\)`).join(" "),
      // put trailing punctuation inside a closing inline formula, so a line cannot break before it
      end: (str, p) => (str.endsWith("\\)") ? `${str.slice(0, -2)}${p}\\)` : str + p),
      G: (name) => `\\operatorname{grad}(${name})`,
      dn: (n, a) => `\\partial ${n}/\\partial ${a}`,
    };
    k.tnp = (v) => (Number(fix(v, dp)) < 0 ? `(${k.tn(v)})` : k.tn(v));
    // Layout: node.x is the step from the inputs, node.y the lane. Across ("h") puts steps in
    // columns; down ("v") puts them in rows, for narrow plates.
    k.geometry = (o, list) => {
      const c0 = Math.min(...list.map((n) => (o === "h" ? n.x : n.y))), r0 = Math.min(...list.map((n) => (o === "h" ? n.y : n.x)));
      const col = (n) => (o === "h" ? n.x : n.y) - c0, row = (n) => (o === "h" ? n.y : n.x) - r0;
      const cw = BW + (o === "h" ? 72 : 10), rh = BH + (o === "h" ? 30 : 58), PAD = o === "h" ? 14 : 8;
      const W = Math.max(...list.map(col)) * cw + BW + 2 * PAD, H = Math.max(...list.map(row)) * rh + BH + 2 * PAD;
      return { o, W, H, P: (n) => [PAD + BW / 2 + col(n) * cw, PAD + BH / 2 + row(n) * rh] };
    };
    // start a drawing; labels widen it if they reach past the nodes (fit sets the final viewBox)
    k.begin = (svg, Gm) => {
      clear(svg);
      k.ext = { x0: 0, x1: Gm.W };
      svg.appendChild(k.defs());
    };
    k.fit = (svg, Gm) => {
      const { x0, x1 } = k.ext;
      svg.setAttribute("viewBox", `${x0} 0 ${x1 - x0} ${Gm.H}`);
      svg.style.maxWidth = `${x1 - x0}px`;
    };
    const uid = Math.random().toString(36).slice(2, 8);
    k.defs = () => s("defs", null, ["ink", "on", "bw"].map((c) => s("marker", { id: `ga-${uid}-${c}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 9, markerHeight: 9, markerUnits: "userSpaceOnUse", orient: "auto" },
      s("path", { d: "M0,1 L10,5 L0,9 z", class: `f-ah ${c}` }))));
    k.marker = (c) => `url(#ga-${uid}-${c})`;
    // the visible part of the edge a → b, between the two boxes
    k.seg = (Gm, a, b) => {
      const [ax, ay] = Gm.P(a), [bx, by] = Gm.P(b);
      const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
      const cut = (pad) => Math.min(ux ? (BW / 2 + pad) / Math.abs(ux) : Infinity, uy ? (BH / 2 + pad) / Math.abs(uy) : Infinity);
      const c0 = cut(1), c1 = cut(3);
      return { x1: ax + ux * c0, y1: ay + uy * c0, x2: bx - ux * c1, y2: by - uy * c1, nx: -uy, ny: ux };
    };
    // a forward edge, optionally with its backward arrow (on one side) and a label (on the other)
    // a forward edge, and its backward arrow (red, on the +normal side) while that edge is being crossed
    k.edge = (svg, Gm, a, b, { cls = "", bw = false } = {}) => {
      const e = k.seg(Gm, a, b);
      svg.appendChild(s("line", { x1: e.x1, y1: e.y1, x2: e.x2, y2: e.y2, class: `f-edge ${cls}`, "marker-end": k.marker(cls === "on" ? "on" : "ink") }));
      if (bw) {
        const o = 5.5;
        svg.appendChild(s("line", { x1: e.x2 + e.nx * o, y1: e.y2 + e.ny * o, x2: e.x1 + e.nx * o, y2: e.y1 + e.ny * o, class: "f-edge bw", "marker-end": k.marker("bw") }));
      }
    };
    // Where each edge's label goes: beside the edge, on whichever side and at whichever point along it
    // crosses no other line, no node and no earlier label. Every edge is placed, shown or not, so a
    // label keeps its place as more of them appear.
    const LH = 15, textW = (str) => str.length * 7.9 + 4;
    const hit = (a, b) => a.l < b.r && b.l < a.r && a.t < b.b && b.t < a.b;
    const crosses = (sg, r) => { // Liang–Barsky: does the segment enter rectangle r?
      let t0 = 0, t1 = 1;
      const dx = sg.x2 - sg.x1, dy = sg.y2 - sg.y1;
      const p = [-dx, dx, -dy, dy], q = [sg.x1 - r.l, r.r - sg.x1, sg.y1 - r.t, r.b - sg.y1];
      for (let i = 0; i < 4; i++) {
        if (p[i] === 0) { if (q[i] < 0) return false; continue; }
        const t = q[i] / p[i];
        if (p[i] < 0) { if (t > t1) return false; if (t > t0) t0 = t; } else { if (t < t0) return false; if (t < t1) t1 = t; }
      }
      return true;
    };
    k.placeLabels = (Gm, list, edges) => {
      const segs = edges.map((ed) => k.seg(Gm, ed.a, ed.b));
      const boxes = list.map((n) => { const [cx, cy] = Gm.P(n); return { l: cx - BW / 2 - 5, r: cx + BW / 2 + 5, t: cy - BH / 2 - 4, b: cy + BH / 2 + 4 }; });
      const taken = [];
      return edges.map((ed, i) => {
        if (!ed.text) return null;
        const e = segs[i], w = textW(ed.text);
        let best = null, bestCost = Infinity;
        for (const t of [0.5, 0.38, 0.62, 0.28, 0.72]) for (const side of [-1, 1]) {
          const lx = side * e.nx, ly = side * e.ny, off = side < 0 ? 11 : 20; // the backward arrow runs on the +1 side
          const anchor = lx > 0.35 ? "start" : lx < -0.35 ? "end" : "middle";
          const x = e.x1 + (e.x2 - e.x1) * t + lx * off, y = e.y1 + (e.y2 - e.y1) * t + ly * (anchor === "middle" ? off + 3 : off);
          const left = anchor === "start" ? x : anchor === "end" ? x - w : x - w / 2;
          const r = { l: left, r: left + w, t: y - LH / 2, b: y + LH / 2 };
          const cost = boxes.filter((bx) => hit(r, bx)).length * 4 + taken.filter((tr) => hit(r, tr)).length * 4 +
            segs.filter((sg, j) => j !== i && crosses(sg, r)).length * 3 + (crosses(e, r) ? 2 : 0) + (r.l < 0 || r.r > Gm.W ? 1 : 0);
          if (cost < bestCost) { best = { x, y, anchor, r }; bestCost = cost; }
        }
        taken.push(best.r);
        return best;
      });
    };
    k.label = (svg, at, text, cls = "") => {
      svg.appendChild(s("text", { x: at.x, y: at.y, class: `f-edge-l ${cls}`, "text-anchor": at.anchor, "dominant-baseline": "central" }, text));
      if (k.ext) { k.ext.x0 = Math.min(k.ext.x0, at.r.l - 2); k.ext.x1 = Math.max(k.ext.x1, at.r.r + 2); }
    };
    // a node box with two or three lines of text: label, value, and grad (or change)
    k.node = (svg, Gm, n, cls, lines) => {
      const [cx, cy] = Gm.P(n);
      const g = s("g", { class: `f-node ${cls}` });
      g.appendChild(s("rect", { x: cx - BW / 2, y: cy - BH / 2, width: BW, height: BH, rx: n.leaf ? 3 : 14 }));
      const ys = BH >= 60 ? [-17, 2, 21] : [-9, 10];
      lines.forEach((ln, i) => { if (ln && ln.tex !== "") g.appendChild(svgText({ x: cx, y: cy + ys[i], class: ln.cls, "text-anchor": "middle", "dominant-baseline": "central" }, ln.tex)); });
      svg.appendChild(g);
      return g;
    };
    k.slopeLabel = (v) => `×${v < 0 && Number(fix(v, dp)) < 0 ? `(${k.show(v)})` : k.show(v)}`;
    return k;
  }

  // The graph figure. Modes: "step" (forward, then backward; the default), "fw" (forward only),
  // "nudge" (raise one node by ε and recompute), "rule" (one operation's backward rule with sliders).
  F("graph", (el, sp) => {
    const mode = sp.mode || "step";
    if (mode === "rule") return graphRule(el, sp);
    const kit = graphKit(sp, mode === "fw" ? 50 : 66);
    const { show, tn, tnp, T, TT, G, dn, end } = kit;
    const nodes = sp.nodes.map((n) => Object.assign({}, n));
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
    for (const n of nodes) {
      const op = OPS[n.op];
      if (!op) throw new Error("unknown op " + n.op);
      n.leaf = !op.f;
      n.val = op.f ? op.f(...n.in.map((i) => byId[i].val)) : n.value;
    }
    const out = byId[sp.output || nodes[nodes.length - 1].id];
    const order = nodes.filter((n) => !n.leaf);
    const lab = (n) => n.label || n.id;
    const names = (n) => n.in.map((i) => lab(byId[i]));
    const vals = (n) => n.in.map((i) => byId[i].val);
    const formula = (n) => (n.leaf ? lab(n) : `${lab(n)} = ${OPS[n.op].tex(names(n))}`);
    // "∂n/∂a = formula = formula with values = value", without repeats
    const dChain = (n, t) => {
      const op = OPS[n.op], v = vals(n);
      const parts = [op.dtex(names(n), v)[t]];
      if (op.dsub) parts.push(op.dsub(v.map(tnp))[t]);
      parts.push(tn(op.d(...v)[t]));
      return [dn(lab(n), lab(byId[n.in[t]]))].concat(parts.filter((p, i) => i === 0 || p !== parts[i - 1])).join(" = ");
    };

    // backward steps: reverse topological order, one per edge; grad of a value used twice accumulates
    nodes.forEach((n) => { n.grad = 0; });
    out.grad = 1;
    const bwSteps = [];
    for (const n of order.slice().reverse()) {
      const local = OPS[n.op].d(...vals(n));
      n.in.forEach((iid, t) => {
        const a = byId[iid], before = a.grad;
        a.grad += n.grad * local[t];
        bwSteps.push({ from: n, to: a, t, local: local[t], upstream: n.grad, before, after: a.grad });
      });
    }
    bwSteps.forEach((b) => {
      const into = bwSteps.filter((x) => x.to === b.to);
      b.paths = into.length; b.path = into.indexOf(b) + 1;
    });
    const steps = [{ kind: "start" }].concat(order.map((n) => ({ kind: "fw", n })), mode === "fw" ? [] : bwSteps.map((b) => ({ kind: "bw", b })));
    let k = Math.min(sp.start ?? 0, steps.length - 1);

    const plain = (tex) => svgMath(tex).map((r) => r[0]).join("");
    const svg = s("svg", { class: "f-graph", role: "img", "aria-label": sp.aria || `computation graph: ${order.map((n) => plain(formula(n))).join("; ")}` });
    const wrap = h("div", { class: "f-graph-wrap" }, svg);
    const expl = h("p", { class: "f-graph-expl", "aria-live": "polite" });
    const ledger = h("div", { class: "f-graph-ledger scroll-x" });
    const box = h("div", { class: "f-graph-fig", style: { maxWidth: `${Math.max(kit.geometry("h", nodes).W, 640)}px` } }, wrap, expl);
    el.appendChild(box);
    let orient = null;
    const pick = () => {
      const w = wrap.clientWidth || (window.innerWidth < 700 ? 340 : 820);
      return w >= 0.8 * kit.geometry("h", nodes).W ? "h" : "v";
    };

    // ---------------------------------------------------------------- nudge: raise one node by ε
    if (mode === "nudge") {
      const ns = sp.nudge || {};
      const cands = (ns.nodes || nodes.filter((n) => n !== out).map((n) => n.id)).map((id) => byId[id]);
      let pickN = byId[ns.node] || cands[0], eps = ns.eps ?? 0.001;
      const sigA = (v) => v.toPrecision(5), sig = (v) => fmt.minus(sigA(v));
      const sigT = (v) => (v < 0 ? `{-${sigA(-v)}}` : sigA(v));
      const six = (v) => fmt.minus(String(Number(v.toFixed(6))));
      const rerun = (target, delta) => {
        const v = new Map();
        for (const n of nodes) v.set(n.id, (n.leaf ? n.val : OPS[n.op].f(...n.in.map((i) => v.get(i)))) + (n === target ? delta : 0));
        return v;
      };
      const after = (target) => {
        const set = new Set([target.id]);
        for (const n of nodes) if (!n.leaf && n.in.some((i) => set.has(i))) set.add(n.id);
        return set;
      };
      const ctl = h("div", { class: "f-ctl" },
        segmented({ label: "raise", value: pickN.id, options: cands.map((n) => ({ label: lab(n), value: n.id })), onChange: (v) => { pickN = byId[v]; draw(); } }),
        segmented({ label: "by ε", value: eps, options: (ns.epsOptions || [0.1, 0.01, 0.001, 0.0001]).map((e) => ({ label: String(e), value: e })), onChange: (v) => { eps = v; draw(); } }));
      box.append(ctl, ledger);
      const draw = () => {
        orient = pick();
        const Gm = kit.geometry(orient, nodes);
        kit.begin(svg, Gm);
        const v = rerun(pickN, eps), moved = after(pickN);
        for (const n of nodes) for (const iid of n.in || []) kit.edge(svg, Gm, byId[iid], n, { cls: moved.has(iid) && moved.has(n.id) ? "on" : "" });
        for (const n of nodes) {
          const m = moved.has(n.id);
          kit.node(svg, Gm, n, [n.leaf ? "leaf" : "", n.op === "param" ? "param" : "", n === out ? "out" : "", n === pickN ? "nudged" : m ? "moved" : "still"].join(" "), [
            { tex: formula(n), cls: "f-node-l" },
            { tex: m ? six(v.get(n.id)) : show(n.val), cls: "f-node-v" },
            m ? { tex: `${v.get(n.id) - n.val >= 0 ? "+" : ""}${sig(v.get(n.id) - n.val)}`, cls: "f-node-d" } : null,
          ]);
        }
        kit.fit(svg, Gm);
        const dy = v.get(out.id) - out.val, a = lab(pickN), y = lab(out);
        const dec = Math.min(11, Math.max(6, Math.ceil(-Math.log10(Math.abs(dy))) + 4)), yv = (val) => fmt.minus(val.toFixed(dec));
        expl.replaceChildren(html(`Raise ${T(a)} by ${T(`\\varepsilon = ${eps}`)}, from ${show(pickN.val)} to ${six(pickN.val + eps)}, and recompute only the nodes computed from it; every other node keeps its value. ` +
          `${T(y)} moves from ${yv(out.val)} to ${yv(v.get(out.id))}, so ` +
          `${T(`\\Delta ${y} = ${sigT(dy)}`)} and ${end(T(`\\Delta ${y}/\\varepsilon = ${sigT(dy / eps)}`), ".")}`));
        clear(ledger);
        const t = h("table", { class: "f-table dense" });
        t.appendChild(h("thead", null, h("tr", null, h("th", null, "raise"), h("th", { class: "r" }, html(T(`\\Delta ${y}`))), h("th", { class: "r" }, html(T(`\\Delta ${y}/\\varepsilon`))))));
        t.appendChild(h("tbody", null, cands.map((n) => {
          const d = rerun(n, eps).get(out.id) - out.val;
          return h("tr", { class: n === pickN ? "is-cur fw" : "" }, h("td", null, html(`${T(lab(n))} by ${eps}`)), h("td", { class: "r" }, sig(d)), h("td", { class: "r" }, h("b", null, sig(d / eps))));
        })));
        ledger.appendChild(t);
      };
      return mountGraph(wrap, draw, () => orient, pick);
    }

    // ---------------------------------------------------------------- step: forward, then backward
    const fwBtn = button("Forward step ▸", () => { if (k < order.length) { k++; draw(); } }, { kind: "primary" });
    const bwBtn = mode === "fw" ? null : button("Backward step ◂", () => { if (k < order.length) k = order.length; if (k < steps.length - 1) k++; draw(); }, { kind: "primary" });
    box.append(h("div", { class: "f-ctl" }, button("Reset", () => { k = 0; draw(); }), fwBtn, bwBtn, button("Show all", () => { k = steps.length - 1; draw(); })), ledger);
    const ledgerMode = sp.ledger ?? (mode === "fw" ? "fw" : "bw");

    function draw() {
      orient = pick();
      const Gm = kit.geometry(orient, nodes);
      kit.begin(svg, Gm);
      const cur = steps[k];
      const done = steps.slice(1, k + 1);
      const fwDone = new Set(done.filter((x) => x.kind === "fw").map((x) => x.n.id));
      const bwDone = done.filter((x) => x.kind === "bw").map((x) => x.b);
      const grads = new Map();
      if (bwDone.length) grads.set(out.id, 1);
      bwDone.forEach((b) => grads.set(b.to.id, b.after));
      const edges = [];
      for (const n of nodes) (n.in || []).forEach((iid, t) => edges.push({ a: byId[iid], b: n, t, text: mode === "fw" ? null : kit.slopeLabel(OPS[n.op].d(...vals(n))[t]) }));
      const at = kit.placeLabels(Gm, nodes, edges);
      const labels = [];
      edges.forEach((ed, i) => {
        const bwNow = cur.kind === "bw" && cur.b.from === ed.b && cur.b.to === ed.a && cur.b.t === ed.t;
        const crossed = bwDone.some((b) => b.from === ed.b && b.to === ed.a && b.t === ed.t);
        kit.edge(svg, Gm, ed.a, ed.b, { cls: cur.kind === "fw" && cur.n === ed.b ? "on" : "", bw: bwNow });
        if (crossed) labels.push([at[i], ed.text, bwNow ? "is-cur" : ""]);
      });
      for (const n of nodes) {
        const known = n.leaf || fwDone.has(n.id);
        const cls = [n.leaf ? "leaf" : "", n.op === "param" ? "param" : "", n === out ? "out" : "",
          cur.kind === "fw" && cur.n === n ? "fw-cur" : "", cur.kind === "bw" && cur.b.to === n ? "bw-cur" : "", cur.kind === "bw" && cur.b.from === n ? "bw-src" : ""].join(" ");
        const lines = [{ tex: formula(n), cls: "f-node-l" }, { tex: known ? show(n.val) : "?", cls: "f-node-v" }];
        if (mode !== "fw") lines.push({ tex: grads.has(n.id) ? `grad ${show(grads.get(n.id))}` : "", cls: "f-node-g" });
        kit.node(svg, Gm, n, cls, lines);
      }
      labels.forEach(([p, text, cls]) => kit.label(svg, p, text, cls));
      kit.fit(svg, Gm);
      expl.replaceChildren(html(explain(cur)));
      fwBtn.disabled = k >= order.length;
      if (bwBtn) bwBtn.disabled = k >= steps.length - 1;
      drawLedger(cur, fwDone, bwDone);
    }

    function explain(cur) {
      if (cur.kind === "start") return sp.intro || (mode === "fw"
        ? "Only the inputs are known. <b>Forward step</b> computes one node from the values on its incoming edges."
        : "Only the inputs are known. <b>Forward step</b> computes the nodes in order; <b>Backward step</b> then computes one grad per edge, from the output back.");
      if (cur.kind === "fw") {
        const n = cur.n, i = order.indexOf(n) + 1;
        const f = TT(`${formula(n)} = ${OPS[n.op].tex(vals(n).map(tnp))} = ${tn(n.val)}`);
        return `<b>Forward ${i}/${order.length}.</b> ` + (i === order.length ? `${end(f, ",")} the output.` : end(f, "."));
      }
      const b = cur.b, n = lab(b.from), a = lab(b.to), j = bwSteps.indexOf(b) + 1;
      const s0 = b.local, dt = OPS[b.from.op].dtex(names(b.from), vals(b.from))[b.t];
      let how;
      if (s0 === 1 && dt === "1") how = `a small change in ${T(a)} changes ${T(n)} by the same amount: ${T(`${dn(n, a)} = 1`)}`;
      else if (s0 === -1 && dt === "{-1}") how = `a small change in ${T(a)} changes ${T(n)} by the same amount in the opposite direction: ${T(`${dn(n, a)} = {-1}`)}`;
      else if (s0 === 0) how = `a small change in ${T(a)} does not change ${T(n)}: ${T(`${dn(n, a)} = 0`)}`;
      else how = `a small change in ${T(a)} changes ${T(n)} by ${T(dt)} times as much: ${TT(dChain(b.from, b.t))}`;
      const term = `${tn(b.upstream)} \\times ${tnp(b.local)}`;
      let res;
      if (b.paths > 1 && b.path > 1) res = `Use ${b.path} of ${b.paths} of ${T(a)} adds its term: ${TT(`${G(a)} = ${tn(b.before)} + ${term} = ${tn(b.after)}`)}`;
      else if (b.paths > 1) res = `${T(a)} is used ${b.paths} times, so ${T(G(a))} is a sum with one term per use. This use gives ${TT(`${G(n)} \\times ${dn(n, a)} = ${term} = ${tn(b.after)}`)}`;
      else res = TT(`${G(a)} = ${G(n)} \\times ${dn(n, a)} = ${term} = ${tn(b.after)}`);
      return `<b>Backward ${j}/${bwSteps.length}.</b> ${end(T(formula(b.from)), ",")} so ${end(how, ".")}<br>${end(res, ".")}`;
    }

    function drawLedger(cur, fwDone, bwDone) {
      clear(ledger);
      if (!ledgerMode) return;
      if (ledgerMode === "fw" || ledgerMode === "both") {
        const t = h("table", { class: "f-table dense fw" });
        const rows = order.filter((n) => fwDone.has(n.id));
        t.appendChild(h("thead", null, h("tr", null, h("th", null, "forward"), h("th", null, "computation"), h("th", { class: "r" }, "value"))));
        t.appendChild(h("tbody", null, rows.length ? rows.map((n) => h("tr", { class: cur.kind === "fw" && cur.n === n ? "is-cur fw" : "" },
          h("td", null, html(T(lab(n)))), h("td", null, html(T(`${OPS[n.op].tex(names(n))} = ${OPS[n.op].tex(vals(n).map(tnp))}`))), h("td", { class: "r" }, h("b", null, show(n.val)))))
          : h("tr", null, h("td", { colspan: 3, class: "f-ledger-empty" }, "Each forward step adds a row."))));
        ledger.appendChild(t);
      }
      if (ledgerMode === "bw" || ledgerMode === "both") {
        const t = h("table", { class: "f-table dense bw" });
        t.appendChild(h("thead", null, h("tr", null, h("th", null, "grad of"), h("th", null, "local derivative"), h("th", null, html(`grad of the node it feeds × local derivative`)))));
        const rows = bwDone.map((b) => {
          const n = lab(b.from), a = lab(b.to);
          const prod = b.path > 1 ? `${tn(b.before)} + ${tn(b.upstream)} \\times ${tnp(b.local)} = ${tn(b.after)}` : `${tn(b.upstream)} \\times ${tnp(b.local)} = ${tn(b.after)}`;
          return h("tr", { class: cur.kind === "bw" && cur.b === b ? "is-cur" : "" },
            h("td", null, html(T(G(a)))), h("td", null, html(T(dChain(b.from, b.t)))), h("td", null, html(T(prod) + (b.paths > 1 ? ` <span class="f-ledger-note">use ${b.path} of ${b.paths}</span>` : ""))));
        });
        if (bwDone.length) rows.unshift(h("tr", null, h("td", null, html(T(G(lab(out))))), h("td", { class: "f-ledger-note" }, "start"), h("td", null, html(T("1")))));
        t.appendChild(h("tbody", null, rows.length ? rows : h("tr", null, h("td", { colspan: 3, class: "f-ledger-empty" }, "Each backward step adds a row."))));
        ledger.appendChild(t);
      }
    }
    return mountGraph(wrap, draw, () => orient, pick);
  });

  // Draw now, and again whenever the plate's width calls for the other orientation.
  function mountGraph(wrap, draw, current, pick) {
    draw();
    if (typeof ResizeObserver === "function") new ResizeObserver(() => { if (wrap.clientWidth && pick() !== current()) draw(); }).observe(wrap);
    return { enter() { if (wrap.clientWidth && pick() !== current()) draw(); } };
  }

  // One operation's backward rule: its inputs, the grad arriving from the rest of the graph,
  // and what each input receives. Sliders move the inputs.
  function graphRule(el, sp) {
    const kit = graphKit(sp, 66);
    const { show, tn, tnp, T, TT, G, dn, end } = kit;
    let r = sp.start ?? 0, vals = null;
    const svg = s("svg", { class: "f-graph", role: "img", "aria-label": "one operation and the grads it sends back" });
    const wrap = h("div", { class: "f-graph-wrap" }, svg);
    const expl = h("p", { class: "f-graph-expl", "aria-live": "polite" });
    const sliders = h("div", { class: "f-ctl sliders" });
    const layoutNodes = (q) => {
      const ins = q.names.map((nm, i) => ({ id: `in${i}`, label: nm, leaf: true, x: 0, y: q.names.length === 1 ? 0.5 : i }));
      return ins.concat([{ id: "op", label: q.out, x: 1, y: 0.5 }], q.rest === false ? [] : [{ id: "rest", label: "rest", x: 2, y: 0.5, rest: true }]);
    };
    const ledger = h("div", { class: "f-graph-ledger scroll-x" });
    el.append(h("div", { class: "f-graph-fig", style: { maxWidth: `${Math.max(...sp.rules.map((q) => kit.geometry("h", layoutNodes(q)).W), 640)}px` } }, wrap, expl,
      h("div", { class: "f-ctl" }, segmented({ label: "operation", value: r, options: sp.rules.map((q, i) => ({ label: q.label, value: i })), onChange: (v) => { r = v; reset(); } })),
      sliders, ledger));
    let orient = null;
    const pick = () => {
      const w = wrap.clientWidth || (window.innerWidth < 700 ? 340 : 820);
      return w >= 0.8 * kit.geometry("h", layoutNodes(sp.rules[r])).W ? "h" : "v";
    };
    function reset() {
      const q = sp.rules[r];
      vals = q.values.slice();
      clear(sliders);
      q.names.forEach((nm, i) => {
        const [lo, hi, st] = q.ranges[i];
        sliders.appendChild(slider({ label: nm, min: lo, max: hi, step: st, value: vals[i], format: (v) => show(v), onInput: (v) => { vals[i] = v; draw(); } }));
      });
      draw();
    }
    function draw() {
      const q = sp.rules[r], op = OPS[q.op], list = layoutNodes(q);
      orient = pick();
      const Gm = kit.geometry(orient, list);
      kit.begin(svg, Gm);
      const outV = op.f(...vals), local = op.d(...vals), g = q.g;
      const ins = list.filter((n) => n.leaf), opN = list.find((n) => n.id === "op"), rest = list.find((n) => n.rest);
      const at = kit.placeLabels(Gm, list, ins.map((a, i) => ({ a, b: opN, text: kit.slopeLabel(local[i]) })));
      ins.forEach((a) => kit.edge(svg, Gm, a, opN, { bw: true }));
      if (rest) kit.edge(svg, Gm, opN, rest, { bw: true });
      ins.forEach((a, i) => kit.node(svg, Gm, a, "leaf bw-cur", [{ tex: a.label, cls: "f-node-l" }, { tex: show(vals[i]), cls: "f-node-v" }, { tex: `grad ${show(g * local[i])}`, cls: "f-node-g" }]));
      kit.node(svg, Gm, opN, rest ? "bw-src" : "bw-src out", [{ tex: `${q.out} = ${op.tex(q.names)}`, cls: "f-node-l" }, { tex: show(outV), cls: "f-node-v" }, { tex: `grad ${show(g)}`, cls: "f-node-g" }]);
      if (rest) kit.node(svg, Gm, rest, "rest", [{ tex: "rest of graph", cls: "f-node-l" }, { tex: `→ ${sp.output || "y"}`, cls: "f-node-v" }, null]);
      ins.forEach((a, i) => kit.label(svg, at[i], kit.slopeLabel(local[i]), "is-cur"));
      kit.fit(svg, Gm);
      const each = q.names.map((a, i) => TT(`${G(a)} = ${G(q.out)} \\times ${dn(q.out, a)} = ${tn(g)} \\times ${tnp(local[i])} = ${tn(g * local[i])}`));
      expl.replaceChildren(html(`${q.say ? q.say : ""}${each.map((e, i) => `<br>${end(e, i < each.length - 1 ? ";" : ".")}`).join("")}`));
      clear(ledger);
      const t = h("table", { class: "f-table dense bw" });
      t.appendChild(h("thead", null, h("tr", null, h("th", null, "grad of"), h("th", null, "local derivative"), h("th", null, `grad(${q.out}) × local derivative`))));
      const vn = q.names, dts = op.dtex(vn, vals), dsubs = op.dsub ? op.dsub(vals.map(tnp)) : null;
      t.appendChild(h("tbody", null, vn.map((a, i) => {
        const parts = [dts[i]].concat(dsubs ? [dsubs[i]] : [], [tn(local[i])]).filter((p, j, arr) => j === 0 || p !== arr[j - 1]);
        return h("tr", null, h("td", null, html(T(G(a)))), h("td", null, html(T(`${dn(q.out, a)} = ${parts.join(" = ")}`))), h("td", null, html(T(`${tn(g)} \\times ${tnp(local[i])} = ${tn(g * local[i])}`))));
      })));
      ledger.appendChild(t);
    }
    reset();
    return mountGraph(wrap, draw, () => orient, pick);
  }

  // ------------------------------------------------------------ network: a small MLP, forward pass with numbers
  const ACT = {
    relu: (x) => Math.max(0, x), sigmoid: (x) => 1 / (1 + Math.exp(-x)), tanh: Math.tanh, linear: (x) => x,
    gelu: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))), step: (x) => (x >= 0 ? 1 : 0),
  };
  window.UI.ACT = ACT;
  F("network", (el, sp) => {
    const sizes = sp.layers;
    const acts = sp.acts || sizes.slice(1).map((_, i) => (i === sizes.length - 2 ? "linear" : "relu"));
    let x = (sp.input || new Array(sizes[0]).fill(0)).slice();
    const Wd = 560, Hd = Math.max(220, Math.max(...sizes) * 62 + 30);
    const colX = (l) => 60 + (l * (Wd - 120)) / (sizes.length - 1);
    const rowY = (l, i) => Hd / 2 + (i - (sizes[l] - 1) / 2) * 62;
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "f-net", role: "img", "aria-label": `network with layers ${sizes.join(", ")}` });
    el.appendChild(svg);
    const out = h("p", { class: "f-net-expl", "aria-live": "polite" });
    el.appendChild(out);
    const ctl = h("div", { class: "f-ctl sliders" });
    function forward() {
      const vals = [x.slice()], pre = [null];
      for (let l = 0; l < sizes.length - 1; l++) {
        const W = sp.W[l], b = sp.b ? sp.b[l] : new Array(sizes[l + 1]).fill(0);
        const z = Array.from({ length: sizes[l + 1] }, (_, j) => vals[l].reduce((acc, v, i) => acc + v * W[j][i], 0) + b[j]);
        pre.push(z);
        vals.push(z.map(ACT[acts[l]]));
      }
      return { vals, pre };
    }
    let hover = null;
    function draw() {
      const { vals, pre } = forward();
      clear(svg);
      const maxW = Math.max(...sp.W.flat(2).map(Math.abs), 1e-9);
      for (let l = 0; l < sizes.length - 1; l++) for (let j = 0; j < sizes[l + 1]; j++) for (let i = 0; i < sizes[l]; i++) {
        const w = sp.W[l][j][i];
        const ln = s("line", { x1: colX(l) + 20, y1: rowY(l, i), x2: colX(l + 1) - 20, y2: rowY(l + 1, j), class: "f-wire" + (w < 0 ? " neg" : ""), style: `stroke-width:${0.6 + 3 * Math.abs(w) / maxW}`, "data-tip": `w = ${num(w, 2)} (from ${sp.names ? sp.names[l][i] : `unit ${i + 1}`} to ${sp.names ? sp.names[l + 1][j] : `unit ${j + 1}`})` });
        svg.appendChild(ln);
        if (sp.showWeights !== false && sizes[l] * sizes[l + 1] <= 12) {
          const t = 0.32;
          svg.appendChild(s("text", { x: colX(l) + 20 + (colX(l + 1) - colX(l) - 40) * t, y: rowY(l, i) + (rowY(l + 1, j) - rowY(l, i)) * t - 4, class: "f-wire-l", "text-anchor": "middle" }, num(w, 1)));
        }
      }
      for (let l = 0; l < sizes.length; l++) for (let i = 0; i < sizes[l]; i++) {
        const v = vals[l][i];
        const g = s("g", { class: "f-unit" + (l === 0 ? " in" : l === sizes.length - 1 ? " outu" : "") });
        g.appendChild(s("circle", { cx: colX(l), cy: rowY(l, i), r: 20, style: `fill:${seq(Math.min(1, Math.abs(v) / (sp.scale || 2)))}` }));
        g.appendChild(s("text", { x: colX(l), y: rowY(l, i) + 4, class: "f-unit-v", "text-anchor": "middle" }, num(v, 2)));
        if (sp.names) g.appendChild(s("text", { x: colX(l), y: rowY(l, i) - 26, class: "f-unit-n", "text-anchor": "middle" }, sp.names[l][i]));
        if (l > 0) g.setAttribute("data-tip", `z = ${num(pre[l][i], 3)} → ${acts[l - 1]}(z) = ${num(v, 3)}`);
        svg.appendChild(g);
      }
      sizes.forEach((n, l) => svg.appendChild(s("text", { x: colX(l), y: Hd - 6, class: "f-layer-l", "text-anchor": "middle" }, l === 0 ? "input" : l === sizes.length - 1 ? `output (${acts[l - 1]})` : `hidden (${acts[l - 1]})`)));
      const y = vals[vals.length - 1];
      out.innerHTML = sp.explain ? sp.explain(x, vals, pre) : `Output: <b>${y.map((v) => num(v, 3)).join(", ")}</b>. Hover a neuron to see its weighted sum z and activation.`;
    }
    (sp.inputSliders || []).forEach((q, i) => ctl.appendChild(slider({ label: q.label || `x${i + 1}`, min: q.min, max: q.max, step: q.step || 0.1, value: x[i], format: (v) => num(v, 2), onInput: (v) => { x[i] = v; draw(); } })));
    if (sp.inputSliders) el.appendChild(ctl);
    draw();
  });

  // ------------------------------------------------------------ gd: gradient descent in 1D or 2D
  F("gd", (el, sp) => {
    let lr = sp.lr, path, steps = 0;
    const host = h("div", { class: "f-gd" });
    el.appendChild(host);
    const info = h("p", { class: "f-gd-info", "aria-live": "polite" });
    const reset = () => { path = [sp.start.slice ? sp.start.slice() : [sp.start]]; steps = 0; draw(); };
    const one = () => {
      const p = path[path.length - 1];
      let nx;
      if (sp.dim === 2) { const g = sp.grad(p[0], p[1]); nx = [p[0] - lr * g[0], p[1] - lr * g[1]]; }
      else nx = [p[0] - lr * sp.df(p[0])];
      if (nx.some((v) => !Number.isFinite(v) || Math.abs(v) > 1e6)) return false;
      path.push(nx); steps++;
      return true;
    };
    let ch;
    function draw() {
      clear(host);
      const p = path[path.length - 1];
      if (sp.dim === 2) {
        const [x0, x1] = sp.x, [y0, y1] = sp.y, Wd = 460, Hd = 340, N = 46;
        const X = (v) => ((v - x0) / (x1 - x0)) * Wd, Y = (v) => Hd - ((v - y0) / (y1 - y0)) * Hd;
        const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "f-gd2", role: "img", "aria-label": "loss surface with the descent path" });
        let lo = Infinity, hi = -Infinity;
        const vals = [];
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          const xv = x0 + ((i + 0.5) / N) * (x1 - x0), yv = y0 + ((j + 0.5) / N) * (y1 - y0);
          const f = sp.f(xv, yv);
          vals.push([i, j, f]); lo = Math.min(lo, f); hi = Math.max(hi, f);
        }
        const tr = (f) => Math.pow((f - lo) / (hi - lo + 1e-12), 0.45);
        vals.forEach(([i, j, f]) => svg.appendChild(s("rect", { x: (i / N) * Wd, y: Hd - ((j + 1) / N) * Hd, width: Wd / N + 0.6, height: Hd / N + 0.6, style: `fill:${seq(1 - tr(f))}`, class: "f-gd-cell" })));
        svg.appendChild(s("polyline", { points: path.map((q) => `${X(q[0])},${Y(q[1])}`).join(" "), class: "f-gd-path" }));
        path.forEach((q, i) => svg.appendChild(s("circle", { cx: X(q[0]), cy: Y(q[1]), r: i === path.length - 1 ? 6 : 2.5, class: i === path.length - 1 ? "f-gd-ball" : "f-gd-dot" })));
        if (sp.min) svg.appendChild(s("text", { x: X(sp.min[0]), y: Y(sp.min[1]) + 5, class: "f-gd-min", "text-anchor": "middle" }, "×"));
        svg.appendChild(s("text", { x: 8, y: Hd - 8, class: "f-gd-ax" }, sp.xlabel || "w₁ →"));
        svg.appendChild(s("text", { x: 8, y: 16, class: "f-gd-ax" }, sp.ylabel || "↑ w₂"));
        host.appendChild(svg);
        const g = sp.grad(p[0], p[1]);
        const off = p[0] < x0 || p[0] > x1 || p[1] < y0 || p[1] > y1;
        info.innerHTML = `Step ${steps}: position (${num(p[0], 3)}, ${num(p[1], 3)}), loss <b>${num(sp.f(p[0], p[1]), 4)}</b>, gradient (${num(g[0], 3)}, ${num(g[1], 3)}).` +
          (off ? " <b>The point has left the plotted area.</b>" : "");
      } else {
        const [x0, x1] = sp.x;
        const pts = [];
        for (let i = 0; i <= 200; i++) { const xv = x0 + ((x1 - x0) * i) / 200; pts.push([xv, sp.f(xv)]); }
        const x = p[0], fx = sp.f(x), d = sp.df(x);
        const span = (x1 - x0) * 0.18;
        ch = chart(host, {
          width: 520, height: 300, margin: { l: 46, r: 14, t: 12, b: 36 }, aria: "function with the descent steps",
          x: { domain: sp.x, ticks: ticks(sp.x), label: sp.xlabel || "parameter" }, y: { domain: sp.y, ticks: ticks(sp.y), label: sp.ylabel || "loss" },
          series: [{ points: pts, cls: "ink" }, { points: [[x - span, fx - d * span], [x + span, fx + d * span]], cls: "s2 thin" }],
          dots: path.map((q, i) => ({ x: q[0], y: sp.f(q[0]), r: i === path.length - 1 ? 6 : 3, cls: i === path.length - 1 ? "" : "ink", tip: `step ${i}: ${sp.v || "x"} = ${num(q[0], 3)}` })),
        });
        const off = x < x0 || x > x1 || fx < sp.y[0] || fx > sp.y[1];
        const vn = sp.v || "x", ln = sp.lrName || "lr";
        info.innerHTML = `Step ${steps}: ${vn} = <b>${num(x, 4)}</b>, loss ${num(fx, 4)}, slope ${num(d, 4)}. Next: ${vn} − ${ln} × slope = ${num(x, 4)} − ${num(lr, 3)} × ${num(d, 4)} = <b>${num(x - lr * d, 4)}</b>.` +
          (off ? " <b>The point has left the plotted range.</b>" : "");
      }
    }
    el.appendChild(info);
    el.appendChild(h("div", { class: "f-ctl" },
      button("Step ▸", () => { one(); draw(); }, { kind: "primary" }),
      button("10 steps", () => { for (let i = 0; i < 10; i++) if (!one()) break; draw(); }),
      button("Reset", reset),
      slider({ label: "Learning rate", min: sp.lrMin ?? 0.001, max: sp.lrMax ?? 1, step: sp.lrStep ?? 0.001, value: lr, format: (v) => num(v, 3), onInput: (v) => { lr = v; reset(); } })));
    reset();
  });

  // ------------------------------------------------------------ diagram: boxes and arrows, drawn as a technical figure
  F("diagram", (el, sp) => {
    const U = sp.unit || 40, Wd = (sp.w || 12) * U, Hd = (sp.h || 7) * U;
    const uid = Math.random().toString(36).slice(2, 8);
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "f-dia", role: "img", "aria-label": sp.aria || "diagram" });
    svg.appendChild(s("defs", null,
      s("marker", { id: `da-${uid}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "f-ah" })),
      s("pattern", { id: `hatch-${uid}`, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, s("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "f-hatch" }))));
    for (const g of sp.groups || []) {
      svg.appendChild(s("rect", { x: g.x * U, y: g.y * U, width: g.w * U, height: g.h * U, class: "f-group" + (g.kind ? " " + g.kind : ""), rx: 2 }));
      if (g.label) svg.appendChild(s("text", { x: g.x * U + 6, y: g.y * U + 14, class: "f-group-l" }, g.label));
    }
    const nodes = Object.fromEntries((sp.nodes || []).map((n) => [n.id, n]));
    const box = (n) => ({ x: n.x * U, y: n.y * U, w: (n.w || 3) * U, h: (n.h || 1) * U });
    const edgePt = (b, tx, ty) => {
      const cx = b.x + b.w / 2, cy = b.y + b.h / 2, dx = tx - cx, dy = ty - cy;
      if (!dx && !dy) return [cx, cy];
      const sx = dx ? (b.w / 2) / Math.abs(dx) : Infinity, sy = dy ? (b.h / 2) / Math.abs(dy) : Infinity;
      const t = Math.min(sx, sy);
      return [cx + dx * t, cy + dy * t];
    };
    for (const e of sp.edges || []) {
      const a = box(nodes[e.from]), b = box(nodes[e.to]);
      const ac = [a.x + a.w / 2, a.y + a.h / 2], bc = [b.x + b.w / 2, b.y + b.h / 2];
      let d, mid;
      if (e.elbow) {
        const p1 = edgePt(a, bc[0], ac[1]), p2 = edgePt(b, bc[0], ac[1]);
        d = `M${p1[0]},${p1[1]} H${bc[0]} V${p2[1]}`; mid = [bc[0], ac[1]];
      } else {
        const p1 = edgePt(a, bc[0], bc[1]), p2 = edgePt(b, ac[0], ac[1]);
        d = `M${p1[0]},${p1[1]} L${p2[0]},${p2[1]}`; mid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
      }
      svg.appendChild(s("path", { d, class: "f-arrow" + (e.dashed ? " dashed" : "") + (e.kind ? " " + e.kind : ""), fill: "none", "marker-end": `url(#da-${uid})`, "marker-start": e.both ? `url(#da-${uid})` : null }));
      if (e.label) svg.appendChild(s("text", { x: mid[0] + (e.dx || 0), y: mid[1] - 6 + (e.dy || 0), class: "f-arrow-l", "text-anchor": "middle" }, e.label));
    }
    for (const n of sp.nodes || []) {
      const b = box(n);
      const g = s("g", { class: "f-box " + (n.kind || "") });
      g.appendChild(s("rect", { x: b.x, y: b.y, width: b.w, height: b.h, rx: n.round ? 10 : 2, style: n.kind === "hatch" ? `fill:url(#hatch-${uid})` : null }));
      const lines = String(n.label || "").split("\n");
      lines.forEach((ln, i) => g.appendChild(s("text", { x: b.x + b.w / 2, y: b.y + b.h / 2 + (i - (lines.length - 1) / 2) * 15 + (n.sub ? -5 : 5), class: "f-box-l", "text-anchor": "middle" }, ln)));
      if (n.sub) g.appendChild(s("text", { x: b.x + b.w / 2, y: b.y + b.h / 2 + 5 + (lines.length - 1) * 7.5 + 10, class: "f-box-s", "text-anchor": "middle" }, n.sub));
      if (n.tip) g.setAttribute("data-tip", n.tip);
      svg.appendChild(g);
    }
    for (const t of sp.notes || []) {
      if (t.to) svg.appendChild(s("path", { d: `M${t.x * U},${t.y * U} L${t.to[0] * U},${t.to[1] * U}`, class: "f-leader", fill: "none" }));
      svg.appendChild(s("text", { x: t.x * U, y: t.y * U - 4, class: "f-hand", "text-anchor": t.anchor || "start" }, t.text));
    }
    for (const d of sp.dims || []) {
      // dimension line: |<--- label --->|
      const [x1, y1, x2, y2] = d.at.map((v) => v * U);
      svg.appendChild(s("path", { d: `M${x1},${y1} L${x2},${y2}`, class: "f-dim", "marker-end": `url(#da-${uid})`, "marker-start": `url(#da-${uid})` }));
      svg.appendChild(s("text", { x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 5, class: "f-dim-l", "text-anchor": "middle" }, d.label));
    }
    el.appendChild(h("div", { class: "scroll-x" }, svg));
    if (sp.note) el.appendChild(h("p", { class: "f-note" }, html(sp.note)));
  });

  // ------------------------------------------------------------ calc: sliders in, numbers out
  F("calc", (el, sp) => {
    const v = {};
    const ctl = h("div", { class: "f-ctl sliders" });
    const out = h("div", { class: "f-tiles" });
    const note = h("p", { class: "f-note", "aria-live": "polite" });
    function draw() {
      clear(out);
      (sp.outputs || []).forEach((o) => { let r; try { r = o.f(v); } catch (e) { r = NaN; } out.appendChild(tile(format(r, o.fmt || "f3"), html(o.label))); });
      note.innerHTML = sp.note ? window.UI.richString(typeof sp.note === "function" ? sp.note(v) : sp.note) : "";
      if (sp.after) sp.after(extra, v);
    }
    const extra = h("div", { class: "f-calc-extra" });
    (sp.inputs || []).forEach((q) => {
      if (q.options) {
        v[q.id] = q.value ?? q.options[0].value;
        ctl.appendChild(segmented({ label: q.label, value: v[q.id], options: q.options, onChange: (x) => { v[q.id] = x; draw(); } }));
      } else if (q.log) {
        v[q.id] = q.value;
        // step "any": a snapped log value would make the label disagree with the outputs (70B shown as 71B)
        ctl.appendChild(slider({ label: q.label, min: Math.log10(q.min), max: Math.log10(q.max), step: q.step || "any", value: Math.log10(q.value), format: (x) => format(Math.pow(10, x), q.fmt || "sci"), onInput: (x) => { v[q.id] = Math.pow(10, x); draw(); } }));
      } else {
        v[q.id] = q.value;
        ctl.appendChild(slider({ label: q.label, min: q.min, max: q.max, step: q.step || 1, value: q.value, format: (x) => format(x, q.fmt || "f2"), onInput: (x) => { v[q.id] = x; draw(); } }));
      }
    });
    el.append(ctl, out, extra, note);
    draw();
  });

  // ------------------------------------------------------------ timeline: rows of busy intervals
  F("timeline", (el, sp) => {
    const rows = sp.rows, total = sp.total || Math.max(...sp.items.map((i) => i.start + i.len));
    const lab = sp.labelWidth || 90, rh = 30, Wd = sp.width || 600, Hd = rows.length * rh + 30;
    const X = (t) => lab + (t / total) * (Wd - lab - 10);
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "f-tl", role: "img", "aria-label": sp.aria || "timeline" });
    rows.forEach((r, i) => {
      svg.appendChild(s("text", { x: lab - 8, y: i * rh + rh * 0.62, class: "f-tl-r", "text-anchor": "end" }, r));
      svg.appendChild(s("line", { x1: lab, x2: Wd - 10, y1: (i + 1) * rh, y2: (i + 1) * rh, class: "f-tl-g" }));
    });
    for (const it of sp.items) {
      const x = X(it.start), w = Math.max(2, X(it.start + it.len) - x);
      const g = s("g", { class: "f-tl-i " + (it.kind || "") });
      g.appendChild(s("rect", { x: x + 1, y: it.row * rh + 4, width: w - 2, height: rh - 8, rx: 2 }));
      if (it.label && w > 18) g.appendChild(s("text", { x: x + w / 2, y: it.row * rh + rh * 0.62, class: "f-tl-l", "text-anchor": "middle" }, it.label));
      if (it.tip) g.setAttribute("data-tip", it.tip);
      svg.appendChild(g);
    }
    // axis: a custom label, or false for none (rows that are not time)
    const axis = sp.axis !== undefined ? sp.axis : sp.unit ? `time (${sp.unit}) →` : "time →";
    if (axis) svg.appendChild(s("text", { x: lab, y: Hd - 8, class: "f-tl-r" }, axis));
    el.appendChild(h("div", { class: "scroll-x" }, svg));
    if (sp.note) el.appendChild(h("p", { class: "f-note" }, html(sp.note)));
  });

  // ------------------------------------------------------------ vectors: drag two arrows, read their dot product
  F("vectors", (el, sp) => {
    let a = sp.a.slice(), b = sp.b.slice();
    const R = sp.range || 4, Wd = 360, Hd = 360;
    const X = (v) => Wd / 2 + (v / R) * (Wd / 2 - 16), Y = (v) => Hd / 2 - (v / R) * (Hd / 2 - 16);
    const inv = (px, py) => [((px - Wd / 2) / (Wd / 2 - 16)) * R, ((Hd / 2 - py) / (Hd / 2 - 16)) * R];
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "f-vec", role: "img", "aria-label": "two vectors on a plane", "data-noswipe": "" });
    const info = h("div", { class: "f-tiles" });
    el.append(svg, info);
    let drag = null;
    function draw() {
      clear(svg);
      const uid = "v" + Math.random().toString(36).slice(2, 7);
      svg.appendChild(s("defs", null, s("marker", { id: uid, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "f-ah-v" }))));
      for (let g = -R; g <= R; g++) {
        svg.appendChild(s("line", { x1: X(g), x2: X(g), y1: Y(-R), y2: Y(R), class: g === 0 ? "f-vec-axis" : "f-vec-grid" }));
        svg.appendChild(s("line", { x1: X(-R), x2: X(R), y1: Y(g), y2: Y(g), class: g === 0 ? "f-vec-axis" : "f-vec-grid" }));
      }
      const arrow = (v, cls, label) => {
        svg.appendChild(s("line", { x1: X(0), y1: Y(0), x2: X(v[0]), y2: Y(v[1]), class: `f-vec-a ${cls}`, "marker-end": `url(#${uid})` }));
        const flip = X(v[0]) > Wd * 0.62;
        svg.appendChild(s("text", { x: X(v[0]) + (flip ? -8 : 8), y: Y(v[1]) + (v[1] < 0 ? 18 : -8), class: `f-vec-l ${cls}`, "text-anchor": flip ? "end" : "start" }, `${label} (${num(v[0], 1)}, ${num(v[1], 1)})`));
      };
      if (sp.showSum) arrow([a[0] + b[0], a[1] + b[1]], "sum", "a+b");
      arrow(a, "va", "a"); arrow(b, "vb", "b");
      [[a, "a"], [b, "b"]].forEach(([v, k]) => {
        const hnd = s("circle", { cx: X(v[0]), cy: Y(v[1]), r: 12, class: "f-vec-h", tabindex: "0", "aria-label": `drag vector ${k}` });
        hnd.addEventListener("pointerdown", (e) => { drag = k; svg.setPointerCapture(e.pointerId); });
        svg.appendChild(hnd);
      });
      const dot = a[0] * b[0] + a[1] * b[1], na = Math.hypot(...a), nb = Math.hypot(...b);
      const cos = na && nb ? dot / (na * nb) : 0;
      clear(info);
      info.append(tile(`${num(a[0], 1)}·${num(b[0], 1)} + ${num(a[1], 1)}·${num(b[1], 1)} = <b>${num(dot, 2)}</b>`.replace(/<\/?b>/g, ""), "a · b"), tile(num(na, 2), "|a|"), tile(num(nb, 2), "|b|"), tile(num(cos, 3), "cos of the angle"));
    }
    svg.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const r = svg.getBoundingClientRect();
      const p = inv(((e.clientX - r.left) / r.width) * Wd, ((e.clientY - r.top) / r.height) * Hd).map((q) => Math.max(-R, Math.min(R, Math.round(q * 2) / 2)));
      if (drag === "a") a = p; else b = p;
      draw();
    });
    svg.addEventListener("pointerup", () => { drag = null; });
    draw();
  });

  // ------------------------------------------------------------ sequence: tokens with rows of values underneath
  F("sequence", (el, sp) => {
    const t = h("table", { class: "f-seq" });
    t.appendChild(h("tr", null, h("th", null, sp.head || ""), sp.tokens.map((tok, i) => h("td", { class: (sp.focus || []).includes(i) ? "on" : "" }, chip(tok)))));
    (sp.rows || []).forEach((r) => t.appendChild(h("tr", null, h("th", null, html(r.label)), r.values.map((v, i) => h("td", { class: (sp.focus || []).includes(i) ? "on" : "" }, html(typeof v === "number" ? num(v, r.dp ?? 3) : v))))));
    el.appendChild(h("div", { class: "scroll-x" }, t));
    if (sp.note) el.appendChild(h("p", { class: "f-note" }, html(sp.note)));
  });

  // ------------------------------------------------------------ tinylm: the interactive widgets built around the trained model
  F("tinylm", (el, sp, api) => {
    const T = window.WIDGET_TEMPLATES || {};
    if (!T[sp.widget]) { el.appendChild(h("p", { class: "fig-missing" }, `No template for widget ${sp.widget}`)); return null; }
    el.classList.add("tinylm");
    el.innerHTML = T[sp.widget];
    const w = window.Widgets[sp.widget];
    // widgets keep state on the object; give each sheet its own copy
    const inst = Object.create(w);
    inst.init(el, api);
    // the engine calls enter() once the sheet is on screen
    return { enter: () => inst.enter && inst.enter(el), leave: () => inst.leave && inst.leave(el) };
  });
})();
