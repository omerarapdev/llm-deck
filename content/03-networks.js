/* Part 3 — Neural networks: from one neuron to deep networks, backpropagation by
 * hand and in PyTorch, the techniques that make deep networks trainable, and a
 * network trained in the page. Every number the text states is recomputed in
 * tools/verify/part03.mjs (and, for PyTorch behavior, tools/verify/part03_torch.py).
 *
 * window.NN3 holds the pure numerical code behind the custom figures (a small MLP
 * with hand-written backward pass, Adam, the depth and residual experiments), so
 * the verification script can rerun exactly what the figures show. */
(function () {
  "use strict";
  const UI = window.UI;
  const { h, s, clear, fmt, chip, slider, segmented, button, chart } = UI;
  const M = window.TinyLM;
  const C = M.config;

  // ------------------------------------------------------------ small helpers
  // fixed decimals with a true minus sign; values that round to zero print as 0
  const fix = (v, d = 3) => {
    if (Number.isNaN(v)) return "–";
    if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
    return fmt.minus((Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : v).toFixed(d));
  };
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const sup = (n) => String(n).split("").map((c) => SUP[c]).join("");
  // 3.2 × 10⁻⁵ style, for values that span many decades
  const sci = (v, d = 1) => {
    if (v === 0) return "0";
    if (!Number.isFinite(v)) return "overflow";
    const e = Math.floor(Math.log10(Math.abs(v)));
    if (e >= -2 && e <= 3) return fix(v, Math.max(0, d + 1 - Math.max(0, e)));
    let m = v / Math.pow(10, e);
    if (Math.abs(+m.toFixed(d)) >= 10) return sci(Math.sign(v) * Math.pow(10, e + 1), d);
    return `${fmt.minus(m.toFixed(d))} × 10${sup(e)}`;
  };
  const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, UI.rich(l)));
  const int = (x) => Math.round(x).toLocaleString("en-US");
  const pow10 = (v) => `10${sup(Math.round(Math.log10(v)))}`;

  // ------------------------------------------------------------ numerical core (window.NN3)
  const NN3 = (window.NN3 = {});
  // standard normal from a uniform generator (Box–Muller, one value per call)
  NN3.gauss = (rand) => {
    let u = 0;
    while (u === 0) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const GELU_K = Math.sqrt(2 / Math.PI);
  NN3.ACT = {
    linear: { f: (z) => z, d: () => 1 },
    sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), d: (z, a) => a * (1 - a) },
    tanh: { f: Math.tanh, d: (z, a) => 1 - a * a },
    relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
    gelu: {
      f: (z) => 0.5 * z * (1 + Math.tanh(GELU_K * (z + 0.044715 * z * z * z))),
      d: (z) => {
        const t = Math.tanh(GELU_K * (z + 0.044715 * z * z * z));
        return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * GELU_K * (1 + 3 * 0.044715 * z * z);
      },
    },
    step: { f: (z) => (z >= 0 ? 1 : 0), d: (z) => (z === 0 ? NaN : 0) },
  };

  // The 2-3-1 network used in chapters 3.2 and 3.3. W1 is inputs × units (row convention, x W).
  NN3.net231 = {
    W1: [[0.5, -0.5, 1.0], [0.2, 0.5, -1.0]], b1: [0.1, -0.2, 0.3], W2: [0.6, -0.4, 0.9], b2: 0.05,
    x: [1, 2], y: 1,
  };
  // parameters as a flat list of 13 numbers, in the order the plates list them
  NN3.flat231 = (n) => [n.W1[0][0], n.W1[0][1], n.W1[0][2], n.W1[1][0], n.W1[1][1], n.W1[1][2], ...n.b1, ...n.W2, n.b2];
  NN3.names231 = ["W₁[1,1]", "W₁[1,2]", "W₁[1,3]", "W₁[2,1]", "W₁[2,2]", "W₁[2,3]", "b₁[1]", "b₁[2]", "b₁[3]", "W₂[1]", "W₂[2]", "W₂[3]", "b₂"];
  NN3.loss231 = (p, x = NN3.net231.x, y = NN3.net231.y) => {
    const z = [0, 1, 2].map((j) => x[0] * p[j] + x[1] * p[3 + j] + p[6 + j]);
    const hh = z.map((v) => (v > 0 ? v : 0));
    const yh = hh[0] * p[9] + hh[1] * p[10] + hh[2] * p[11] + p[12];
    return { z, h: hh, yh, L: (yh - y) * (yh - y) };
  };
  // backpropagation by hand, the five steps of the sheet
  NN3.grad231 = (p, x = NN3.net231.x, y = NN3.net231.y) => {
    const f = NN3.loss231(p, x, y);
    const dyh = 2 * (f.yh - y);
    const dW2 = f.h.map((v) => v * dyh);
    const dh = [p[9], p[10], p[11]].map((w) => dyh * w);
    const dz = dh.map((v, j) => (f.z[j] > 0 ? v : 0));
    const dW1 = [0, 1, 2].map((j) => x[0] * dz[j]).concat([0, 1, 2].map((j) => x[1] * dz[j]));
    return { f, dyh, dW2, dh, dz, grad: [...dW1, ...dz, ...dW2, dyh] };
  };
  // central or forward differences for each of the 13 parameters
  NN3.fd231 = (p, eps, central = true) => p.map((_, k) => {
    const up = p.slice(); up[k] += eps;
    const dn = p.slice(); dn[k] -= eps;
    const Lu = NN3.loss231(up).L;
    return central ? (Lu - NN3.loss231(dn).L) / (2 * eps) : (Lu - NN3.loss231(p).L) / eps;
  });

  // Piecewise-linear interpolation of f on [a, b] by k bumps (hat functions), built from k + 2 ReLUs.
  NN3.bumps = (f, a, b, k) => {
    let centers, hw;
    if (k === 1) { centers = [(a + b) / 2]; hw = (b - a) / 2; }
    else { hw = (b - a) / (k - 1); centers = Array.from({ length: k }, (_, i) => a + i * hw); }
    const relu = (v) => (v > 0 ? v : 0);
    const hat = (x, c) => (relu(x - c + hw) - 2 * relu(x - c) + relu(x - c - hw)) / hw;
    const g = (x) => centers.reduce((acc, c) => acc + f(c) * hat(x, c), 0);
    let maxErr = 0;
    for (let i = 0; i <= 3000; i++) { const x = a + ((b - a) * i) / 3000; maxErr = Math.max(maxErr, Math.abs(f(x) - g(x))); }
    return { centers, hw, hat, g, maxErr, units: k + 2 };
  };
  NN3.bumpTarget = { f: (x) => Math.sin(2 * x) + 0.5 * x, a: 0, b: 3 };

  // The tent t(x) = 2 ReLU(x) − 4 ReLU(x − 0.5), composed L times, at its 2^L + 1 breakpoints.
  NN3.tent = (x) => 2 * Math.max(0, x) - 4 * Math.max(0, x - 0.5);
  NN3.saw = (L) => {
    const n = Math.pow(2, L), pts = [];
    for (let j = 0; j <= n; j++) {
      let v = j / n;
      for (let l = 0; l < L; l++) v = NN3.tent(v);
      pts.push([j / n, v]);
    }
    return pts;
  };

  // ------------------------------------------------------------ Fig: code listing with line notes
  // Same markup and styles as the core "code" figure. The core highlighter runs its
  // keyword pass over HTML it has already produced, so in Python the keyword "class"
  // matches inside <span class="tk-n"> and breaks every line that contains a number.
  // This one tokenizes each line once.
  const PY_KW = new Set(["def", "return", "for", "in", "if", "else", "elif", "import", "from", "as", "with", "class", "lambda", "None", "True", "False", "and", "or", "not", "while", "range", "yield", "pass", "is"]);
  const escHtml = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function hlPython(line) {
    // listings write < and > as &lt; and &gt; so the source passes the HTML check
    line = line.replace(/&lt;/g, "<").replace(/&gt;/g, ">");
    let out = "", i = 0, m;
    while (i < line.length) {
      const rest = line.slice(i);
      if (rest[0] === "#") { out += `<span class="tk-c">${escHtml(rest)}</span>`; break; }
      if ((m = rest.match(/^("[^"]*"|'[^']*')/))) out += `<span class="tk-s">${escHtml(m[0])}</span>`;
      else if ((m = rest.match(/^[A-Za-z_]\w*/))) out += PY_KW.has(m[0]) ? `<span class="tk-k">${m[0]}</span>` : m[0];
      else if ((m = rest.match(/^\d+\.?\d*(?:e[-+]?\d+)?/))) out += `<span class="tk-n">${m[0]}</span>`;
      else { out += escHtml(rest[0]); i++; continue; }
      i += m[0].length;
    }
    return out || " ";
  }
  Deck.fig("nn-code", (el, sp) => {
    const lines = sp.code.replace(/\n$/, "").split("\n");
    const notes = sp.notes || {};
    const pre = h("div", { class: "f-code", role: "region", "aria-label": `${sp.lang || "python"} listing` });
    const rows = lines.map((ln, i) => {
      const row = h("div", { class: "f-line" + ((sp.highlight || []).includes(i + 1) ? " hl" : "") + (notes[i + 1] ? " has-note" : "") },
        h("span", { class: "f-ln" }, String(i + 1)), h("code", { html: hlPython(ln) }));
      pre.appendChild(row);
      return row;
    });
    el.appendChild(h("div", { class: "scroll-x" }, pre));
    const keys = Object.keys(notes).map(Number).sort((a, b) => a - b);
    if (!keys.length) return;
    const box = h("p", { class: "f-code-note", "aria-live": "polite" });
    let k = 0;
    const show = () => {
      rows.forEach((row, i) => row.classList.toggle("cur", i + 1 === keys[k]));
      clear(box);
      box.append(h("b", null, `Line ${keys[k]}. `), UI.rich(notes[keys[k]]));
    };
    rows.forEach((row, i) => { if (notes[i + 1]) row.addEventListener("click", () => { k = keys.indexOf(i + 1); show(); }); });
    el.append(box, h("div", { class: "f-ctl" },
      button("◂ Previous line", () => { k = Math.max(0, k - 1); show(); }),
      button("Next line ▸", () => { k = Math.min(keys.length - 1, k + 1); show(); }, { kind: "primary" })));
    show();
  });

  // ------------------------------------------------------------ Fig: XOR against a straight line
  // Sutherland–Hodgman: the part of a convex polygon where n·p > c
  function clipHalf(poly, n, c) {
    const out = [], val = (p) => n[0] * p[0] + n[1] * p[1] - c;
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i], q = poly[(i + 1) % poly.length], vp = val(p), vq = val(q);
      if (vp > 0) out.push(p);
      if ((vp > 0) !== (vq > 0)) { const t = vp / (vp - vq); out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
    }
    return out;
  }
  Deck.fig("nn-xor-line", (el, sp) => {
    const PTS = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
    const lo = -0.6, hi = 1.6, S = 300, pad = 34;
    const X = (v) => pad + ((v - lo) / (hi - lo)) * S, Y = (v) => pad + S - ((v - lo) / (hi - lo)) * S;
    let th = sp.theta ?? 45, c = sp.c ?? 0.5;
    const svg = s("svg", { viewBox: `0 0 ${S + 2 * pad} ${S + 2 * pad}`, class: "nn-xor", role: "img", "aria-label": "The four XOR points and a straight decision boundary" });
    const tiles = h("div", { class: "f-tiles" });
    function draw() {
      clear(svg);
      const r = (th * Math.PI) / 180, n = [Math.cos(r), Math.sin(r)];
      const box = [[lo, lo], [hi, lo], [hi, hi], [lo, hi]];
      const half = clipHalf(box, n, c);
      svg.appendChild(s("rect", { x: X(lo), y: Y(hi), width: S, height: S, class: "nn-frame" }));
      if (half.length > 2) svg.appendChild(s("polygon", { points: half.map((p) => `${X(p[0])},${Y(p[1])}`).join(" "), class: "nn-half" }));
      for (const v of [0, 1]) {
        svg.appendChild(s("line", { x1: X(v), x2: X(v), y1: Y(lo), y2: Y(hi), class: "nn-gridl" }));
        svg.appendChild(s("line", { x1: X(lo), x2: X(hi), y1: Y(v), y2: Y(v), class: "nn-gridl" }));
        svg.appendChild(s("text", { x: X(v), y: Y(lo) + 16, class: "nn-tick", "text-anchor": "middle" }, String(v)));
        svg.appendChild(s("text", { x: X(lo) - 8, y: Y(v) + 4, class: "nn-tick", "text-anchor": "end" }, String(v)));
      }
      svg.appendChild(s("text", { x: X(hi), y: Y(lo) + 30, class: "nn-axl", "text-anchor": "end" }, "x₁ →"));
      svg.appendChild(s("text", { x: X(lo) - 26, y: Y(hi) + 4, class: "nn-axl" }, "x₂"));
      // the boundary n·p = c, clipped to the box
      const seg = clipHalf(box, n, c - 1e-9).filter((p) => Math.abs(n[0] * p[0] + n[1] * p[1] - c) < 1e-6);
      if (seg.length >= 2) svg.appendChild(s("line", { x1: X(seg[0][0]), y1: Y(seg[0][1]), x2: X(seg[1][0]), y2: Y(seg[1][1]), class: "nn-bound" }));
      let ok = 0;
      for (const [a, b, cls] of PTS) {
        const sc = n[0] * a + n[1] * b - c, pred = sc > 0 ? 1 : 0, right = pred === cls;
        if (right) ok++;
        if (!right) svg.appendChild(s("circle", { cx: X(a), cy: Y(b), r: 15, class: "nn-wrong" }));
        svg.appendChild(s("circle", { cx: X(a), cy: Y(b), r: 8, class: `nn-pt c${cls}`, "data-tip": `(${a}, ${b}): XOR = ${cls}; score ${fix(sc, 3)} → predicts ${pred}` }));
        svg.appendChild(s("text", { x: X(a), y: Y(b) + (b ? -17 : 27), class: "nn-ptl", "text-anchor": "middle" }, `(${a}, ${b}) → ${cls}`));
      }
      clear(tiles);
      tiles.append(tile(fix(n[0], 2), "w₁ = cos θ"), tile(fix(n[1], 2), "w₂ = sin θ"), tile(fix(-c, 2), "b = −c"), tile(`${ok} of 4`, "correct"));
    }
    el.append(h("div", { class: "nn-xor-wrap" }, svg),
      h("div", { class: "f-ctl sliders" },
        slider({ label: "Line direction θ", min: 0, max: 359, step: 1, value: th, format: (v) => `${v}°`, onInput: (v) => { th = v; draw(); } }),
        slider({ label: "Offset c", min: -1.5, max: 1.5, step: 0.01, value: c, format: (v) => fix(v, 2), onInput: (v) => { c = v; draw(); } })),
      tiles,
      h("p", { class: "f-note" }, "The classifier predicts 1 where w₁x₁ + w₂x₂ + b > 0, the shaded side. Every setting gets at most 3 of the 4 points right."));
    draw();
  });

  // ------------------------------------------------------------ Fig: activation functions and their slopes
  Deck.fig("nn-act", (el, sp) => {
    const A = NN3.ACT;
    const NAMES = { step: "Step", sigmoid: "Sigmoid", tanh: "Tanh", relu: "ReLU", gelu: "GELU (tanh form)" };
    let which = sp.start || "sigmoid", x0 = sp.x0 ?? 1;
    const host = h("div", { class: "nn-chart" });
    const tiles = h("div", { class: "f-tiles" });
    const series = () => {
      const f = [], d = [];
      for (let i = 0; i <= 600; i++) {
        const x = -3 + (6 * i) / 600;
        f.push([x, A[which].f(x)]);
        const dv = A[which].d(x, A[which].f(x));
        d.push([x, Number.isFinite(dv) ? dv : null]);
      }
      if (which === "step") { // draw the jump as a vertical rise, the slope as 0 with a gap at 0
        const left = f.filter((p) => p[0] < 0), right = f.filter((p) => p[0] >= 0);
        return [{ points: left.concat([[0, 0], [0, 1]], right), cls: "s1" }, { points: d.filter((p) => p[1] !== null), cls: "s2 dashed" }];
      }
      return [{ points: f, cls: "s1" }, { points: d, cls: "s2 dashed" }];
    };
    const ch = chart(host, {
      width: 560, height: 300, margin: { l: 44, r: 14, t: 12, b: 36 }, aria: "activation function and its derivative",
      x: { domain: [-3, 3], ticks: [-3, -2, -1, 0, 1, 2, 3], label: "z", fmt: (v) => fmt.minus(String(v)) },
      y: { domain: [-1.2, 3], ticks: [-1, 0, 1, 2, 3], fmt: (v) => fmt.minus(String(v)) },
      series: series(),
    });
    const maxSlope = { step: "0", sigmoid: "0.25 at z = 0", tanh: "1 at z = 0", relu: "1 for z > 0", gelu: "1.13 at z = 1.42" };
    const YR = {
      step: [[-0.3, 1.3], [0, 0.5, 1]], sigmoid: [[-0.3, 1.3], [0, 0.5, 1]], tanh: [[-1.3, 1.3], [-1, -0.5, 0, 0.5, 1]],
      relu: [[-0.5, 3], [0, 1, 2, 3]], gelu: [[-0.5, 3], [0, 1, 2, 3]],
    };
    function draw() {
      const fx = A[which].f(x0), dx = A[which].d(x0, fx);
      ch.update({ y: { domain: YR[which][0], ticks: YR[which][1], fmt: (v) => fmt.minus(String(v)) }, series: series(), vlines: [{ x: x0, cls: "red" }], dots: [{ x: x0, y: fx, r: 5, cls: "s1", tip: `φ(${fix(x0, 1)}) = ${fix(fx, 4)}` }].concat(Number.isFinite(dx) ? [{ x: x0, y: dx, r: 5, cls: "s2", tip: `φ′(${fix(x0, 1)}) = ${fix(dx, 4)}` }] : []) });
      clear(tiles);
      const und = which === "relu" ? "undefined (PyTorch uses 0)" : "undefined";
      tiles.append(tile(fix(fx, 4), `φ(${fix(x0, 1)})`), tile(Number.isFinite(dx) ? fix(dx, 4) : und, `slope φ′(${fix(x0, 1)})`), tile(maxSlope[which], "largest slope"));
    }
    el.append(h("div", { class: "f-ctl" }, segmented({ label: "Function", value: which, options: Object.keys(NAMES).map((k) => ({ label: NAMES[k].replace(" (tanh form)", ""), value: k })), onChange: (v) => { which = v; draw(); } })),
      host,
      h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "φ(z)"), h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "slope φ′(z), dashed")),
      h("div", { class: "f-ctl sliders" }, slider({ label: "Read at z", min: -3, max: 3, step: 0.1, value: x0, format: (v) => fix(v, 1), onInput: (v) => { x0 = v; draw(); } })),
      tiles);
    draw();
  });

  // ------------------------------------------------------------ Fig: a sum of ReLU bumps traces a curve
  Deck.fig("nn-bumps", (el, sp) => {
    const { f, a, b } = NN3.bumpTarget;
    let k = sp.k ?? 4;
    const host = h("div", { class: "nn-chart" });
    const tiles = h("div", { class: "f-tiles" });
    const ch = chart(host, {
      width: 560, height: 290, margin: { l: 40, r: 14, t: 12, b: 36 }, aria: "target curve, bumps and their sum",
      x: { domain: [a, b], ticks: [0, 0.5, 1, 1.5, 2, 2.5, 3], label: "x" }, y: { domain: [-0.6, 2.2], ticks: [-0.5, 0, 0.5, 1, 1.5, 2], fmt: (v) => fmt.minus(String(v)) },
      series: [],
    });
    function draw() {
      const B = NN3.bumps(f, a, b, k), N = 360;
      const xs = Array.from({ length: N + 1 }, (_, i) => a + ((b - a) * i) / N);
      const bumps = B.centers.map((c) => ({ points: xs.map((x) => [x, f(c) * B.hat(x, c)]), cls: "s3 thin" }));
      ch.update({
        series: bumps.concat([{ points: xs.map((x) => [x, f(x)]), cls: "ink dashed" }, { points: xs.map((x) => [x, B.g(x)]), cls: "s2" }]),
        dots: B.centers.map((c) => ({ x: c, y: f(c), r: 4, cls: "s2", tip: `bump at x = ${fix(c, 3)}, height f(x) = ${fix(f(c), 3)}` })),
      });
      clear(tiles);
      tiles.append(tile(String(k), "bumps"), tile(String(B.units), "ReLU units (one hidden layer)"), tile(fix(B.maxErr, 3), "largest error on [0, 3]"));
    }
    el.append(host,
      h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-ink" }), "target sin 2x + 0.5x, dashed"), h("span", { class: "key" }, h("span", { class: "sw sw-s3" }), "each bump × its height"), h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "sum of the bumps")),
      h("div", { class: "f-ctl sliders" }, slider({ label: "Bumps k", min: 1, max: 8, step: 1, value: k, format: (v) => String(v), onInput: (v) => { k = v; draw(); } })),
      tiles);
    draw();
  });

  // ------------------------------------------------------------ Fig: depth composes tents into a sawtooth
  Deck.fig("nn-saw", (el, sp) => {
    let L = sp.L ?? 3;
    const host = h("div", { class: "nn-chart" });
    const tiles = h("div", { class: "f-tiles" });
    const ch = chart(host, {
      width: 560, height: 250, margin: { l: 40, r: 14, t: 12, b: 36 }, aria: "the tent function composed L times",
      x: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], label: "x" }, y: { domain: [0, 1.08], ticks: [0, 0.5, 1] }, series: [],
    });
    function draw() {
      const pts = NN3.saw(L);
      const prev = L > 1 ? NN3.saw(L - 1) : null;
      ch.update({ series: (prev ? [{ points: prev, cls: "faint" }] : []).concat([{ points: pts, cls: "s1" }]) });
      clear(tiles);
      tiles.append(tile(int(Math.pow(2, L)), "linear pieces"), tile(String(2 * L), `ReLUs: ${L} layer${L > 1 ? "s" : ""} of 2`), tile(int(Math.pow(2, L) - 1), "ReLUs one hidden layer needs"));
    }
    el.append(host,
      h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "t applied L times"), h("span", { class: "key" }, h("span", { class: "sw sw-ink" }), "L − 1 times, faint")),
      h("div", { class: "f-ctl sliders" }, slider({ label: "Layers L", min: 1, max: 6, step: 1, value: L, format: (v) => String(v), onInput: (v) => { L = v; draw(); } })),
      tiles);
    draw();
  });

  // ================================================================== Part 3
  Deck.part({ n: 3, title: "Neural networks", short: "Networks" });

  // ------------------------------------------------------------------ 3.1 Neurons
  Deck.chapter({ title: "Neurons" });

  Deck.slide({
    id: "nn-part",
    layout: "part",
    title: "Neural networks",
    body: `<p>Part 2 fitted models with a handful of parameters. This part builds the model family that scales to billions of them: layers of weighted sums with a nonlinear function between them. You compute a forward pass and every gradient of a small network by hand, check them numerically, let PyTorch compute them, and train a network in this page.</p>
<p>Depth makes these networks expressive and also hard to train. The later chapters measure why, and what our model does about it: initialization scaled to each layer's width, RMSNorm, residual connections, Adam, and a learning rate that warms up and then decays.</p>`,
    take: "Write a multi-layer network as matrix products, derive its gradients by backpropagation and check them, read a PyTorch training loop line by line, and say with numbers what each stabilization technique fixes.",
  });

  Deck.slide({
    id: "nn-xor",
    title: "A straight line cannot separate XOR",
    body: String.raw`<p>A <b>linear classifier</b> on two inputs computes a score \(s = w_1 x_1 + w_2 x_2 + b\) and predicts class 1 when \(s \gt 0\). The points where \(s = 0\) form a straight line, so the classifier splits the plane into two half-planes. A sigmoid on top (Part 2) turns the score into a probability but leaves the line where it is.</p>
<p><b>XOR</b> (exclusive or) is 1 when exactly one input is 1: (0, 0) → 0, (0, 1) → 1, (1, 0) → 1, (1, 1) → 0. Class 1 needs \(w_2 + b \gt 0\) and \(w_1 + b \gt 0\); adding them gives \(w_1 + w_2 + 2b \gt 0\). Class 0 at (0, 0) needs \(b \le 0\), so \(w_1 + w_2 + b \gt {-b} \ge 0\), and (1, 1) lands in class 1. No line gets all four points right; the best is 3 of 4.</p>
<p>The next four sheets add what a linear score lacks: a nonlinear function after the sum, and layers.</p>`,
    fig: { type: "nn-xor-line", caption: "Turn and shift the line. The shaded side is predicted 1; a red ring marks a wrong prediction.", theta: 45, c: 0.5 },
    terms: [
      { term: "Linear classifier", def: "A model that predicts a class from the sign of a weighted sum of its inputs plus a bias. Its decision boundary is a line (a hyperplane in more dimensions)." },
      { term: "XOR", def: "Exclusive or: 1 when exactly one of two binary inputs is 1. The smallest problem a linear classifier cannot solve." },
    ],
  });

  Deck.slide({
    id: "nn-neuron",
    title: "A neuron is a weighted sum followed by a nonlinear function",
    body: String.raw`<p>A <b>neuron</b> multiplies each input \(x_i\) by a <b>weight</b> \(w_i\), adds a <b>bias</b> \(b\), and passes the sum through an <b>activation function</b> \(\varphi\). The sum \(z\) is the <b>pre-activation</b>; \(a = \varphi(z)\) is the neuron's output, its activation.</p>
<p>With \(x = (1.0, 2.0),\) \(w = (0.6, {-0.2})\) and \(b = 0.1\): \(z = 0.6 \cdot 1.0 - 0.2 \cdot 2.0 + 0.1 = 0.3\). With the sigmoid \(\varphi(z) = 1/(1 + e^{-z})\), \(a = 0.574\).</p>
<p>The weighted sum alone is the linear classifier of the previous sheet. The activation function is what lets networks of neurons bend their boundaries: two sheets on, stacking neurons without it collapses back to one linear function.</p>`,
    math: String.raw`a = \varphi\Big(\sum_{i=1}^{n} w_i x_i + b\Big)`,
    fig: {
      type: "network", caption: "One sigmoid neuron with weights 0.6 and −0.2 and bias 0.1. Drag the inputs.",
      layers: [2, 1], acts: ["sigmoid"], W: [[[0.6, -0.2]]], b: [[0.1]], input: [1, 2], names: [["x₁", "x₂"], ["a"]],
      inputSliders: [{ label: "x₁", min: -3, max: 3 }, { label: "x₂", min: -3, max: 3 }],
      explain: (x, vals, pre) => `z = 0.6 × ${fix(x[0], 1)} + (−0.2) × ${fix(x[1], 1)} + 0.1 = <b>${fix(pre[1][0], 3)}</b>; a = σ(z) = <b>${fix(vals[1][0], 3)}</b>`,
    },
    terms: [
      { term: "Neuron", def: "A weighted sum of inputs plus a bias, passed through an activation function: \\(a = \\varphi(w \\cdot x + b)\\)." },
      { term: "Weight", def: "A parameter that multiplies one input of a neuron." },
      { term: "Bias", def: "A parameter added to a neuron's weighted sum; it shifts where the neuron switches." },
      { term: "Activation function", def: "The nonlinear function \\(\\varphi\\) applied to a neuron's weighted sum: sigmoid, tanh, ReLU, GELU." },
      { term: "Pre-activation", def: "A neuron's weighted sum plus bias, \\(z\\), before the activation function." },
    ],
  });

  Deck.slide({
    id: "nn-activations",
    title: "Activation functions are judged by their slopes",
    body: String.raw`<p>Every gradient that reaches a weight passes through the slope \(\varphi'(z)\) of the activation after it, so the slope matters as much as the shape.</p>
<ul>
<li><b>Step</b>, 1 if \(z \ge 0\) else 0: slope 0 everywhere except at 0, where it is undefined. Every gradient through it is 0, so gradient descent never moves the weights before it.</li>
<li><b>Sigmoid</b> \(\sigma(z)\): slope \(\sigma(z)(1 - \sigma(z))\), at most 0.25, and 0.0066 at \(z = 5\).</li>
<li><b>Tanh</b>: output in (−1, 1), slope \(1 - \tanh^2 z\), at most 1, and 0.0002 at \(z = 5\).</li>
<li><b>ReLU</b> \(\max(0, z)\): slope 1 for \(z \gt 0\), 0 for \(z \lt 0\). Cheap, and it does not shrink a gradient where it is active.</li>
<li><b>GELU</b> \(z\,\Phi(z)\), \(\Phi\) the standard normal CDF: a smooth ReLU that dips to −0.17 at \(z = {-0.75}\). Our model's MLP uses its tanh approximation.</li>
</ul>`,
    fig: { type: "nn-act", caption: "Each activation function (solid) and its slope (dashed). Read both at any z.", start: "sigmoid", x0: 1 },
    src: ["Hendrycks & Gimpel, Gaussian Error Linear Units (GELUs), arXiv:1606.08415, 2016"],
    terms: [
      { term: "Tanh", def: "The hyperbolic tangent, \\((e^z - e^{-z})/(e^z + e^{-z})\\): an S-shaped activation with outputs in (−1, 1) and slope at most 1." },
      { term: "GELU", def: "Gaussian error linear unit, \\(z\\,\\Phi(z)\\): a smooth version of ReLU. Our model uses the tanh approximation \\(0.5z(1 + \\tanh(\\sqrt{2/\\pi}(z + 0.044715z^3)))\\)." },
    ],
  });

  Deck.slide({
    id: "nn-collapse",
    title: "Without activations, two layers collapse into one",
    body: String.raw`<p>This part writes an input as a row vector \(x\) and a layer as \(xW + b\), where column \(j\) of \(W\) holds neuron \(j\)'s weights. That is how a batch sits in memory and how PyTorch writes it, <code>x @ W</code>. (With column vectors the same layer is \(W^\top x + b\).)</p>
<p>Stack two layers with nothing between them: \((xW_1 + b_1)W_2 + b_2 = x(W_1W_2) + (b_1W_2 + b_2)\). That is one layer with weights \(W_1W_2\). Any number of linear layers collapses the same way, so a deep linear network draws the same straight boundaries as one neuron.</p>
<p>Put a ReLU between the layers and the collapse fails. A linear map satisfies \(f({-x}) = {-f(x)}\). With the matrices in the plate, \(x = (1, {-1})\) gives (3, −1) but \(x = ({-1}, 1)\) gives (0, 0).</p>`,
    math: String.raw`\begin{aligned}(xW_1 + b_1)W_2 + b_2 &= x\,(W_1W_2) \\ &\quad + (b_1W_2 + b_2)\end{aligned}`,
    fig: {
      type: "steps", caption: "Two 2 × 2 layers, applied in turn and as one product",
      steps: [
        { html: String.raw`\(W_1 = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix},\ W_2 = \begin{pmatrix} 2 & 0 \\ 1 & {-1} \end{pmatrix}\)`, note: "x = (1, 2); no biases, to keep the numbers small" },
        { html: String.raw`\(\begin{aligned} xW_1 &= (1 \cdot 1 + 2 \cdot 0,\ 1 \cdot 2 + 2 \cdot 1) \\ &= (1, 4) \end{aligned}\)` },
        { html: String.raw`\(\begin{aligned} (xW_1)W_2 &= (2 + 4,\ 0 - 4) \\ &= (6, {-4}) \end{aligned}\)`, note: "two layers" },
        { html: String.raw`\(W_1W_2 = \begin{pmatrix} 4 & {-2} \\ 1 & {-1} \end{pmatrix}\)` },
        { html: String.raw`\(\begin{aligned} x(W_1W_2) &= (4 + 2,\ {-2} - 2) \\ &= (6, {-4}) \end{aligned}\)`, note: "one layer, same answer" },
        { html: String.raw`\(\begin{aligned} &\mathrm{ReLU}((1, {-1})W_1)\,W_2 \\ &= \mathrm{ReLU}((1, 1))\,W_2 = (3, {-1}) \end{aligned}\)`, note: "now with a ReLU between" },
        { html: String.raw`\(\begin{aligned} &\mathrm{ReLU}(({-1}, 1)W_1)\,W_2 \\ &= \mathrm{ReLU}(({-1}, {-1}))\,W_2 \\ &= (0, 0) \ne ({-3}, 1) \end{aligned}\)`, note: "not linear: no single matrix does this" },
      ],
    },
    take: "Depth without a nonlinearity adds parameters but no new functions.",
  });

  Deck.slide({
    id: "nn-xor-relu",
    title: "Two ReLU neurons solve XOR",
    body: String.raw`<p>A hidden layer of two ReLU neurons and one linear output neuron compute XOR exactly: \(h_1 = \mathrm{ReLU}(x_1 + x_2)\), \(h_2 = \mathrm{ReLU}(x_1 + x_2 - 1)\), \(y = h_1 - 2h_2\).</p>
<table><thead><tr><th>input</th><th class="r">\(h_1\)</th><th class="r">\(h_2\)</th><th class="r">\(y\)</th></tr></thead><tbody>
<tr><td>(0, 0)</td><td class="r">0</td><td class="r">0</td><td class="r">0</td></tr>
<tr><td>(0, 1)</td><td class="r">1</td><td class="r">0</td><td class="r">1</td></tr>
<tr><td>(1, 0)</td><td class="r">1</td><td class="r">0</td><td class="r">1</td></tr>
<tr><td>(1, 1)</td><td class="r">2</td><td class="r">1</td><td class="r">0</td></tr></tbody></table>
<p>The hidden layer moves the points: (0, 1) and (1, 0) both land on \((h_1, h_2) = (1, 0)\), and (1, 1) lands on (2, 1). In these new coordinates a straight line separates the classes, and the output neuron draws it.</p>`,
    fig: {
      type: "network", caption: "XOR from two ReLU units (biases 0 and −1) and a linear output. Set each input to 0 or 1.",
      layers: [2, 2, 1], acts: ["relu", "linear"], W: [[[1, 1], [1, 1]], [[1, -2]]], b: [[0, -1], [0]], input: [1, 1], names: [["x₁", "x₂"], ["h₁", "h₂"], ["y"]],
      inputSliders: [{ label: "x₁", min: 0, max: 1, step: 1 }, { label: "x₂", min: 0, max: 1, step: 1 }],
      explain: (x, vals) => `h₁ = ReLU(${x[0]} + ${x[1]}) = ${vals[1][0]}, h₂ = ReLU(${x[0]} + ${x[1]} − 1) = ${vals[1][1]}, y = h₁ − 2h₂ = <b>${fix(vals[2][0], 0)}</b> = XOR(${x[0]}, ${x[1]})`,
    },
    check: {
      q: String.raw`Change the output weight of \(h_2\) from −2 to −1. What does the network output for the input (1, 1)?`,
      options: ["0", "1", "2", "−1"], answer: 1,
      why: String.raw`For (1, 1), \(h_1 = 2\) and \(h_2 = 1\), so \(y = 2 - 1 = 1\): the same output as (0, 1) and (1, 0), and XOR fails. The weight −2 is what cancels \(h_1\)'s double count at (1, 1).`,
    },
  });

  // ------------------------------------------------------------------ 3.2 Layers and networks
  Deck.chapter({ title: "Layers and networks" });

  Deck.slide({
    id: "nn-layer",
    title: "A layer is one matrix multiply, a bias and an activation",
    body: String.raw`<p>A <b>layer</b> of \(m\) neurons reading \(n\) inputs stores its weights as an \(n \times m\) matrix \(W\) (column \(j\) belongs to neuron \(j\)) and its biases as a vector \(b\) of length \(m\). For a batch of \(B\) inputs stacked as the rows of \(X\), the whole layer is \(H = \varphi(XW + b)\): one matrix multiply produces all \(B \cdot m\) weighted sums, \(b\) is added to every row, and \(\varphi\) is applied to each entry.</p>
<p>The plate computes \(Z = XW + b\) for 3 inputs and the 3 hidden neurons used on the next sheets. The bias is written as a third row of \(W\) that multiplies a constant input 1. ReLU then gives \(H\) with rows (1.0, 0.3, 0), (0, 0.8, 0) and (0.9, 0, 3.3).</p>
<p>This is the operation GPUs are built for (Part 6). For one training batch of our model (1,024 tokens), the first MLP matrix alone is \(1{,}024 \times 16 \times 64 = 1{,}048{,}576\) multiply-adds in one call.</p>`,
    math: String.raw`H = \varphi(XW + b)`,
    fig: {
      type: "matmul", caption: "Three inputs through the 3-unit hidden layer: [X | 1] times [W; b]", dp: 2, start: 9,
      A: [[1, 2, 1], [-1, 1, 1], [2, -1, 1]], B: [[0.5, -0.5, 1.0], [0.2, 0.5, -1.0], [0.1, -0.2, 0.3]],
      aName: `[X | 1] <span class="dim">3×3</span>`, bName: `[W; b] <span class="dim">3×3</span>`, cName: `Z = XW + b <span class="dim">3×3</span>`,
    },
    terms: [{ term: "Layer", def: "A set of neurons that read the same inputs: \\(H = \\varphi(XW + b)\\), one matrix multiply for a whole batch." }],
  });

  Deck.slide({
    id: "nn-mlp-code",
    title: "A multi-layer perceptron is layers in sequence",
    body: String.raw`<p>A <b>multi-layer perceptron</b> (MLP) applies layers one after another; the layers between input and output are <b>hidden layers</b>. The listing builds the 2-3-1 network of the next sheet in PyTorch and runs the batch from the previous sheet through it.</p>
<p>Shapes are the whole design. Each weight matrix is (inputs × outputs), so the product chains only when inner sizes agree: (3 × 2)(2 × 3) gives 3 × 3, then (3 × 3)(3 × 1) gives 3 × 1, one output per input row: 0.53, −0.27 and 3.56.</p>
<p>Our model's MLP is the same pattern with GELU, no biases, and sizes 16 → 64 → 16. For one query its input is 8 × 16, one row per position, and its hidden activations are 8 × 64.</p>`,
    fig: {
      type: "nn-code", lang: "python", caption: "The 2-3-1 network in PyTorch, and our model's MLP block",
      code: `import torch
import torch.nn.functional as F
X  = torch.tensor([[1., 2.], [-1., 1.], [2., -1.]])  # 3 × 2
W1 = torch.tensor([[0.5, -0.5,  1.0],
                   [0.2,  0.5, -1.0]])               # 2 × 3
b1 = torch.tensor([0.1, -0.2, 0.3])                  # 3
W2 = torch.tensor([[0.6], [-0.4], [0.9]])            # 3 × 1
b2 = torch.tensor([0.05])                            # 1

H = torch.relu(X @ W1 + b1)   # (3×2) @ (2×3) -> 3 × 3
Y = H @ W2 + b2               # (3×3) @ (3×1) -> 3 × 1
print(Y.squeeze())  # tensor([ 0.5300, -0.2700,  3.5600])

# our model's MLP block for one query: 8 positions
act = F.gelu(h2 @ w1, approximate="tanh")  # 8 × 64
out = act @ w2                             # 8 × 16`,
      notes: {
        3: "A batch of 3 inputs, one per row. Writing <code>1.</code> makes the tensor float32, 4 bytes per number.",
        4: "W1 is (inputs × units). Column 1 holds unit 1's two weights, 0.5 and 0.2.",
        6: "One bias per hidden unit.",
        7: "W2 maps the 3 hidden units to 1 output.",
        10: "<code>X @ W1</code> is 3 × 3: every unit's weighted sum for every input at once. Adding <code>b1</code> (shape 3) broadcasts it to each row. ReLU zeroes the negatives: rows (1.0, 0.3, 0), (0, 0.8, 0), (0.9, 0, 3.3).",
        11: "Each output is a row of H dotted with W2, plus b2. Row 1: 1.0 × 0.6 + 0.3 × (−0.4) + 0 × 0.9 + 0.05 = 0.53.",
        12: "<code>squeeze()</code> drops the size-1 column to print 3 numbers.",
        15: "<code>h2</code> is the normalized residual stream, 8 positions × 16 numbers; <code>w1</code> is 16 × 64. GELU in its tanh form, as our model was trained.",
        16: "Back to 16 numbers per position; the block adds them to its input (the residual connection, chapter 3.5).",
      },
    },
    terms: [
      { term: "Multi-layer perceptron (MLP)", def: "Layers of neurons applied in sequence, each a matrix multiply, a bias and an activation. Every transformer block contains one." },
      { term: "Hidden layer", def: "A layer between the input and the output of a network; its activations are not compared with any target." },
    ],
    src: ["PyTorch 2.15 documentation; outputs reproduced by tools/verify/part03_torch.py"],
  });

  const N231 = NN3.net231;
  Deck.slide({
    id: "nn-forward",
    title: "A forward pass through a 2-3-1 network",
    body: String.raw`<p>The forward pass evaluates the network from input to output, layer by layer. For \(x = (1, 2)\):</p>
<ol>
<li>Pre-activations \(z = xW_1 + b_1\): \(z_1 = 0.5 + 0.4 + 0.1 = 1.0\), \(z_2 = {-0.5} + 1.0 - 0.2 = 0.3\), \(z_3 = 1.0 - 2.0 + 0.3 = {-0.7}\).</li>
<li>Hidden activations \(h = \mathrm{ReLU}(z) = (1.0, 0.3, 0)\). Unit 3 is off for this input.</li>
<li>Output \(\hat y = hW_2 + b_2 = 0.6 - 0.12 + 0 + 0.05 = 0.53\).</li>
</ol>
<p>That is 9 multiply-adds for 9 weights. Drag the inputs: unit 3 switches on when \(x_1 - x_2 \gt {-0.3}\), and the output changes slope there. The output is a piecewise-linear function of the input, with a kink wherever a hidden unit switches on or off.</p>`,
    fig: {
      type: "network", caption: "The 2-3-1 network: ReLU hidden layer, linear output. Hover a unit for its z.",
      layers: [2, 3, 1], acts: ["relu", "linear"], input: N231.x.slice(), names: [["x₁", "x₂"], ["h₁", "h₂", "h₃"], ["ŷ"]],
      W: [[0, 1, 2].map((j) => [N231.W1[0][j], N231.W1[1][j]]), [N231.W2.slice()]], b: [N231.b1.slice(), [N231.b2]],
      inputSliders: [{ label: "x₁", min: -2, max: 2 }, { label: "x₂", min: -2, max: 2 }],
      explain: (x, vals, pre) => `z = (${pre[1].map((v) => fix(v, 2)).join(", ")}), h = (${vals[1].map((v) => fix(v, 2)).join(", ")}), ŷ = <b>${fix(vals[2][0], 3)}</b>`,
    },
  });

  Deck.slide({
    id: "nn-relu-sum",
    title: "Enough ReLUs can trace any curve",
    body: String.raw`<p>Three ReLUs make a <b>bump</b> (the formula below): it is 0 outside \([c - s, c + s]\) and rises linearly to 1 at \(c\). Place bumps every \(s\), scale each by the target's value at its center, and add them: the sum passes through the target at every center and is linear in between.</p>
<p>For \(f(x) = \sin 2x + 0.5x\) on [0, 3], the largest error falls from 1.221 with one bump to 0.442 with four and 0.090 with eight. Neighboring bumps share kinks, so \(k\) bumps need only \(k + 2\) ReLU units in one hidden layer. Linear interpolation between points \(s\) apart errs by at most \(s^2 \max|f''|/8\), which is 0.092 for eight bumps here.</p>
<p>The general result is the universal approximation theorem: one hidden layer of enough units, with any non-polynomial activation, approximates any continuous function on a closed and bounded region. It promises that enough units exist; it says nothing about how many, or whether training finds them.</p>`,
    math: String.raw`\begin{aligned}\mathrm{bump}_c(x) &= \tfrac{1}{s}\,\mathrm{ReLU}(x - c + s) \\ &\quad - \tfrac{2}{s}\,\mathrm{ReLU}(x - c) \\ &\quad + \tfrac{1}{s}\,\mathrm{ReLU}(x - c - s)\end{aligned}`,
    fig: { type: "nn-bumps", caption: "k bumps, each built from ReLUs, added to approximate a curve", k: 4 },
    src: ["Cybenko, Approximation by superpositions of a sigmoidal function, 1989", "Leshno, Lin, Pinkus & Schocken, Multilayer feedforward networks with a nonpolynomial activation function can approximate any function, 1993"],
    terms: [{ term: "Universal approximation theorem", def: "A network with one hidden layer of enough units and a non-polynomial activation can approximate any continuous function on a closed and bounded region to any accuracy." }],
  });

  Deck.slide({
    id: "nn-params",
    title: "A layer from n inputs to m outputs has n·m + m parameters",
    example: "tokens ids",
    body: String.raw`<p>Each of the \(m\) neurons has one weight per input and one bias: \(nm + m\) parameters. The 2-3-1 network has \(2 \cdot 3 + 3 + 3 \cdot 1 + 1 = 13\).</p>
<p>Our model's MLP blocks have no biases: \(16 \times 64 = 1{,}024\) weights up and \(64 \times 16 = 1{,}024\) down, 2,048 per block and 4,096 in both blocks, 41% of the 9,968 parameters. The rest is attention (2,048), the token embeddings and the unembedding (1,808 each), positions (128) and the RMSNorm gains (80); Part 5 counts them term by term.</p>
<p>Parameters cost memory once and compute on every token: each weight of a matrix multiply takes part in one multiply-add per token in the forward pass (embedding tables are looked up instead), so the 4,096 MLP weights cost 4,096 multiply-adds per token.</p>`,
    fig: {
      type: "calc", caption: "Parameters of one layer",
      inputs: [
        { id: "n", label: "inputs n", min: 1, max: 128, step: 1, value: 16, fmt: "int" },
        { id: "m", label: "outputs m", min: 1, max: 128, step: 1, value: 64, fmt: "int" },
        { id: "bias", label: "biases", options: [{ label: "none (our model)", value: 0 }, { label: "one per output", value: 1 }], value: 0 },
      ],
      outputs: [
        { label: "weights n·m", f: (v) => v.n * v.m, fmt: "int" },
        { label: "biases", f: (v) => v.bias * v.m, fmt: "int" },
        { label: "parameters", f: (v) => v.n * v.m + v.bias * v.m, fmt: "int" },
        { label: "bytes in float32", f: (v) => 4 * (v.n * v.m + v.bias * v.m), fmt: "int" },
      ],
      note: (v) => `${v.n} × ${v.m}${v.bias ? ` + ${v.m}` : ""} = ${int(v.n * v.m + v.bias * v.m)} parameters. Our MLP's first matrix is 16 × 64, its second 64 × 16.`,
    },
  });

  Deck.slide({
    id: "nn-depth-width",
    title: "Depth multiplies linear pieces; width only adds them",
    body: String.raw`<p>A hidden layer of \(n\) ReLU units on one input bends the output at most \(n\) times, so the output has at most \(n + 1\) linear pieces. Composing layers multiplies pieces instead. The tent \(t(x) = 2\,\mathrm{ReLU}(x) - 4\,\mathrm{ReLU}(x - 0.5)\) maps [0, 1] onto itself with 2 pieces; \(t(t(x))\) has 4, and \(L\) layers of these 2 units give \(2^L\).</p>
<p>At \(L = 6\) that is 64 pieces from 12 ReLU units. One hidden layer needs at least 63 units for the same function, one per kink. Telgarsky turned this construction into a proof that some functions computed by deep, narrow networks need exponentially many units when computed by shallow ones.</p>
<p>The price of depth is training: each layer multiplies the backward signal by another matrix and another slope, and chapter 3.5 measures what that does. Real models are both deep and wide: our model has 2 blocks of width 16, Llama 3 405B has 126 layers of width 16,384.</p>`,
    fig: { type: "nn-saw", caption: "The tent composed with itself L times: a network of L layers with 2 ReLUs each", L: 3 },
    src: ["Telgarsky, Benefits of depth in neural networks, COLT 2016", "Dubey et al., The Llama 3 Herd of Models, 2024 (Table 3)"],
    check: {
      q: "A network has one input and one hidden layer of 10 ReLU units. What is the largest number of linear pieces its output can have?",
      options: ["10", "11", "20", "1,024"], answer: 1,
      why: String.raw`Each unit bends the output at one point, where its input weight times \(x\) plus its bias crosses 0. Ten bends split the line into at most 11 pieces. 1,024 pieces (\(2^{10}\)) take composition: 10 layers of the tent.`,
    },
  });

  // ------------------------------------------------------------ Fig: backprop against finite differences, all 13 parameters
  Deck.fig("nn-fdcheck", (el, sp) => {
    const p = NN3.flat231(NN3.net231), bp = NN3.grad231(p).grad;
    const EPS = [0.5, 0.1, 0.01, 1e-4, 1e-7];
    let central = true, eps = sp.eps ?? 1e-4;
    const tbody = h("tbody");
    const tiles = h("div", { class: "f-tiles" });
    const table = h("table", { class: "f-table dense nn-fd" },
      h("thead", null, h("tr", null, ["parameter", "|difference|", "backprop", "finite difference", "value"].map((c, j) => h("th", { class: j ? "r" : "" }, c)))), tbody);
    function draw() {
      const fd = NN3.fd231(p, eps, central);
      clear(tbody);
      let worst = 0, bad = 0;
      p.forEach((v, k) => {
        const d = Math.abs(fd[k] - bp[k]);
        worst = Math.max(worst, d);
        if (d > 1e-6) bad++;
        tbody.appendChild(h("tr", { class: d > 1e-6 ? "nn-bad" : "" },
          h("td", null, NN3.names231[k]), h("td", { class: "r" }, d === 0 ? "0" : sci(d, 1)), h("td", { class: "r" }, fix(bp[k], 6)),
          h("td", { class: "r" }, fix(fd[k], 6)), h("td", { class: "r" }, fix(v, 2))));
      });
      clear(tiles);
      tiles.append(tile(worst === 0 ? "0" : sci(worst, 1), "largest |difference|"), tile(`${bad} of 13`, "differ by more than 10⁻⁶"));
    }
    el.append(
      h("div", { class: "f-ctl" },
        segmented({ label: "Difference", value: central, options: [{ label: "central", value: true }, { label: "forward", value: false }], onChange: (v) => { central = v; draw(); } }),
        segmented({ label: "Nudge ε", value: eps, options: EPS.map((e) => ({ label: e >= 0.01 ? String(e) : pow10(e), value: e })), onChange: (v) => { eps = v; draw(); } })),
      h("div", { class: "scroll-x" }, table), tiles,
      h("p", { class: "f-note" }, "W₁[i,j] connects input i to hidden unit j. Central: (L(θ + ε) − L(θ − ε)) / 2ε. Forward: (L(θ + ε) − L(θ)) / ε. Rows in red disagree with backpropagation."));
    draw();
  });

  // ------------------------------------------------------------ Fig: p − y at the output, at each position of the query
  Deck.fig("nn-pgrad", (el, sp, api) => {
    const ids = M.runningIds, tr = M.forward(ids.slice(0, C.T)), V = M.vocab;
    let pos = sp.pos ?? 4;
    const tbody = h("tbody");
    const tiles = h("div", { class: "f-tiles" });
    const table = h("table", { class: "f-table dense" },
      h("thead", null, h("tr", null, ["next token", "p", "y", "p − y"].map((c, j) => h("th", { class: j ? "r" : "" }, c)))), tbody);
    function draw() {
      const pr = tr.probs[pos], tgt = ids[pos + 1];
      const order = pr.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 4).map((e) => e[1]);
      if (!order.includes(tgt)) order.push(tgt);
      const rest = pr.reduce((acc, v, i) => acc + (order.includes(i) ? 0 : v), 0);
      clear(tbody);
      order.forEach((i) => tbody.appendChild(h("tr", { class: i === tgt ? "hi" : "" },
        h("td", null, chip(V[i]), i === tgt ? h("span", { class: "nn-tgt" }, " target") : null),
        h("td", { class: "r" }, fix(pr[i], 4)), h("td", { class: "r" }, i === tgt ? "1" : "0"), h("td", { class: "r" }, fix(pr[i] - (i === tgt ? 1 : 0), 4)))));
      tbody.appendChild(h("tr", null, h("td", null, `other ${V.length - order.length} tokens together`), h("td", { class: "r" }, fix(rest, 4)), h("td", { class: "r" }, "0"), h("td", { class: "r" }, fix(rest, 4))));
      clear(tiles);
      tiles.append(tile(fix(-Math.log(pr[tgt]), 4), "loss −ln p(target)"), tile(fix(2 * (1 - pr[tgt]), 4), "sum of |p − y| over 113 logits"));
      if (api && api.strip) api.strip.set({ focus: [pos], target: pos + 1 });
    }
    el.append(h("div", { class: "f-ctl sliders" }, slider({ label: "Position", min: 0, max: 7, step: 1, value: pos, format: (v) => `${v}: after ${M.running[v].trim()}`, onInput: (v) => { pos = v; draw(); } })),
      h("div", { class: "scroll-x" }, table), tiles);
    draw();
    return { leave: () => api && api.strip && api.strip.reset() };
  });

  // ------------------------------------------------------------------ 3.3 Backpropagation
  Deck.chapter({ title: "Backpropagation" });

  Deck.slide({
    id: "nn-bp-tape",
    title: "The forward pass saves what each local derivative will need",
    body: String.raw`<p>Training needs \(\partial L/\partial \theta\) for every parameter. Part 1 showed how to get all of them for about the cost of two more forward passes: backpropagation, the chain rule run backward through the computational graph. This chapter does it for networks with every number, then counts its compute and memory.</p>
<p>Take one data point and the squared error \(L = (wx + b - y)^2\) with \(w = 0.8\), \(x = 1.5\), \(b = 0.2\), \(y = 2\): four operations. Each local derivative is evaluated at the forward pass's values, so the forward pass must keep them. A multiplication keeps both inputs (\(\partial m/\partial w = x\)), squaring keeps its input, since \(\partial L/\partial e = 2e,\) and addition and subtraction keep nothing, since their local derivatives are +1 and −1.</p>
<p>These saved values are why a training step needs far more memory than inference (last sheet of this chapter).</p>`,
    fig: {
      type: "table", caption: "The four operations of L = (wx + b − y)², as recorded", dense: true,
      cols: ["", "operation", "value", "local derivatives", "saved"], align: ["l", "l", "r", "l", "l"],
      rows: [
        ["1", String.raw`\(m = w \cdot x\)`, "1.2", String.raw`\(\partial m/\partial w = x = 1.5\), \(\partial m/\partial x = w = 0.8\)`, String.raw`\(w, x\)`],
        ["2", String.raw`\(z = m + b\)`, "1.4", String.raw`\(\partial z/\partial m = 1\), \(\partial z/\partial b = 1\)`, "nothing"],
        ["3", String.raw`\(e = z - y\)`, "−0.6", String.raw`\(\partial e/\partial z = 1\), \(\partial e/\partial y = {-1}\)`, "nothing"],
        ["4", String.raw`\(L = e^2\)`, "0.36", String.raw`\(\partial L/\partial e = 2e = {-1.2}\)`, String.raw`\(e\)`],
      ],
      note: "PyTorch records this list as the code runs and frees it after the backward pass (chapter 3.4).",
    },
  });

  Deck.slide({
    id: "nn-bp-graph",
    title: "The backward pass multiplies local derivatives from the loss back",
    layout: "wide",
    body: String.raw`<p>As in Part 1, the backward pass starts from \(\partial L/\partial L = 1\) and visits the operations in reverse. At each one it multiplies the gradient arriving from above by the local derivative and sends the product to each input. A value that feeds several operations adds up what comes back from each.</p>
<p>Here \(\partial L/\partial e = 2e = {-1.2}\). Addition and subtraction pass it on unchanged, so \(\partial L/\partial z = \partial L/\partial m = \partial L/\partial b = {-1.2}\). The multiplication sends \({-1.2} \times x = {-1.8}\) to \(w\) and \({-1.2} \times w = {-0.96}\) to \(x\).</p>
<p>Check with a finite difference: \(L(w + 0.001) = 0.35820225\) and \(L(w - 0.001) = 0.36180225\), so the slope is \({-0.0036} / 0.002 = {-1.8}\). Each edge of the graph is crossed once forward and once backward, so the backward pass costs a small constant times the forward pass: twice, for the matrix multiplies of a network (end of this chapter).</p>`,
    math: String.raw`\begin{aligned}\frac{\partial L}{\partial w} &= \frac{\partial L}{\partial e}\,\frac{\partial e}{\partial z}\,\frac{\partial z}{\partial m}\,\frac{\partial m}{\partial w} \\ &= ({-1.2})(1)(1)(1.5) = {-1.8}\end{aligned}`,
    fig: {
      type: "graph", caption: "L = (w·x + b − y)² with w = 0.8, x = 1.5, b = 0.2, y = 2. Reset, then step forward and backward.", dp: 2, start: 11,
      nodes: [
        { id: "w", op: "param", value: 0.8, x: 0, y: 0 }, { id: "x", op: "input", value: 1.5, x: 0, y: 1 }, { id: "b", op: "param", value: 0.2, x: 0, y: 2 }, { id: "y", op: "input", value: 2, x: 2, y: 2 },
        { id: "m", label: "m", op: "mul", in: ["w", "x"], x: 1, y: 0.5 }, { id: "z", label: "z", op: "add", in: ["m", "b"], x: 2, y: 1 },
        { id: "e", label: "e", op: "sub", in: ["z", "y"], x: 3, y: 1.5 }, { id: "L", label: "L", op: "square", in: ["e"], x: 4, y: 1.5 },
      ],
    },
  });

  Deck.slide({
    id: "nn-bp-neuron",
    title: "One sigmoid neuron, forward and backward by hand",
    body: String.raw`<p>The neuron from chapter 3.1, with target \(y = 1\) and squared error. Forward: \(z = 0.3\), \(a = \sigma(0.3) = 0.5744\), \(L = (a - 1)^2 = 0.1811\).</p>
<p>Backward, one local derivative per line. The sigmoid's slope is \(\sigma'(z) = \sigma(z)(1 - \sigma(z))\), so the backward pass needs only the saved output \(a\), not \(z\). Each weight's gradient is the gradient at \(z\) times that weight's input, so \(w_2\), whose input is 2.0, gets twice the gradient of \(w_1\).</p>
<p>One gradient step with learning rate 0.5 raises all three parameters and lowers the loss from 0.1811 to 0.0807.</p>`,
    fig: {
      type: "steps", caption: "x = (1.0, 2.0), w = (0.6, −0.2), b = 0.1, target y = 1",
      steps: [
        { html: String.raw`\(z = 0.6 \cdot 1.0 + ({-0.2}) \cdot 2.0 + 0.1 = 0.3\)`, note: "forward; save x" },
        { html: String.raw`\(a = \sigma(0.3) = 0.5744\)`, note: "save a" },
        { html: String.raw`\(\begin{aligned} L &= (a - y)^2 \\ &= (0.5744 - 1)^2 = 0.1811 \end{aligned}\)` },
        { html: String.raw`\(\partial L/\partial a = 2(a - y) = {-0.8511}\)`, note: "backward starts" },
        { html: String.raw`\(\begin{aligned} \partial L/\partial z &= {-0.8511} \times a(1 - a) \\ &= {-0.8511} \times 0.2445 = {-0.2081} \end{aligned}\)` },
        { html: String.raw`\(\begin{aligned} \partial L/\partial w_1 &= {-0.2081} \times 1.0 \\ \partial L/\partial b &= {-0.2081} \end{aligned}\)` },
        { html: String.raw`\(\partial L/\partial w_2 = {-0.2081} \times 2.0 = {-0.4161}\)`, note: "input 2.0: twice the gradient" },
        { html: String.raw`\(w_1 = 0.7040,\ w_2 = 0.0081,\ b = 0.2040\)`, note: "one step, η = 0.5: w − η ∂L/∂w" },
        { html: String.raw`\(z = 0.9242,\ a = 0.7159,\ L = 0.0807\)`, note: "the loss fell from 0.1811" },
      ],
    },
  });

  Deck.slide({
    id: "nn-bp-layer",
    title: "A layer's gradients: an outer product and a transposed matrix",
    body: String.raw`<p>For a layer \(y = xW + b\) with \(x\) of size \(1 \times n\), \(W\) of size \(n \times m\), and the gradient arriving from above \(\delta = \partial L/\partial y\) of size \(1 \times m\), three rules give every gradient, each with the shape of what it differentiates.</p>
<p>\(W_{ij}\) affects only \(y_j\), with slope \(x_i\), so \(\partial L/\partial W_{ij} = x_i \delta_j\): the <b>outer product</b> \(x^\top \delta\). Each \(x_i\) affects every \(y_j\), so its gradient sums over \(j\): \(\delta W^\top\). The bias gets \(\delta\).</p>
<p>For a batch, \(X\) is \(B \times n\) and \(\Delta\) is \(B \times m\). \(X^\top \Delta\) adds up the \(B\) outer products, and the bias gradient sums the rows of \(\Delta\). The backward pass of a layer is therefore two matrix multiplies, each the size of the forward one.</p>`,
    math: String.raw`\frac{\partial L}{\partial W} = x^\top \delta, \quad \frac{\partial L}{\partial x} = \delta\, W^\top, \quad \frac{\partial L}{\partial b} = \delta`,
    fig: {
      type: "steps", caption: "The three rules, derived entry by entry, then with numbers",
      steps: [
        { html: String.raw`\(y_j = \sum_i x_i W_{ij} + b_j\)`, note: "one output of the layer" },
        { html: String.raw`\(\begin{aligned} &\dfrac{\partial L}{\partial W_{ij}} = \delta_j\, x_i \;\Rightarrow\; \dfrac{\partial L}{\partial W} = x^\top \delta \\ &(n \times 1)(1 \times m) = n \times m \end{aligned}\)` },
        { html: String.raw`\(\begin{aligned} &\dfrac{\partial L}{\partial x_i} = \sum_j \delta_j W_{ij} \;\Rightarrow\; \dfrac{\partial L}{\partial x} = \delta\, W^\top \\ &(1 \times m)(m \times n) = 1 \times n \end{aligned}\)` },
        { html: String.raw`\(\dfrac{\partial L}{\partial b_j} = \delta_j \;\Rightarrow\; \dfrac{\partial L}{\partial b} = \delta\)` },
        { html: String.raw`\(\begin{aligned} &x = (1, 2),\ \delta = (0.5, {-1}, 2): \\ &x^\top\delta = \begin{pmatrix} 0.5 & {-1} & 2 \\ 1 & {-2} & 4 \end{pmatrix} \end{aligned}\)`, note: "W is the 2 × 3 matrix W₁" },
        { html: String.raw`\(\begin{aligned} \delta W_1^\top &= (0.25 + 0.5 + 2,\ 0.1 - 0.5 - 2) \\ &= (2.75, {-2.4}) \end{aligned}\)` },
      ],
    },
    check: {
      q: String.raw`\(W\) is 16 × 64 and the batch has 1,024 rows. What is the shape of \(\partial L/\partial W\)?`,
      options: ["1,024 × 64", "16 × 64", "1,024 × 16", "64 × 16"], answer: 1,
      why: String.raw`A gradient has the shape of what it differentiates. \(X^\top\) is 16 × 1,024 and \(\Delta\) is 1,024 × 64, so \(X^\top\Delta\) is 16 × 64: the 1,024 rows are summed away.`,
    },
    terms: [{ term: "Outer product", def: "The product of a column vector and a row vector, \\(x^\\top\\delta\\): a matrix whose entry \\((i, j)\\) is \\(x_i \\delta_j\\)." }],
  });

  Deck.slide({
    id: "nn-bp-relu",
    title: "Through a ReLU the gradient passes or stops",
    body: String.raw`<p>ReLU's slope is 1 where \(z \gt 0\) and 0 where \(z \lt 0\); PyTorch also uses 0 at exactly \(z = 0\). So the backward pass multiplies the incoming gradient element by element with a 0/1 mask saved from the forward pass.</p>
<p>In the 2-3-1 network at \(x = (1, 2)\), \(z = (1.0, 0.3, {-0.7})\) and the mask is (1, 1, 0). The gradient arriving at \(h\) is \(({-0.564}, 0.376, {-0.846})\) (the sheet <i>Backpropagation through the 2-3-1 network</i> shows where it comes from), and \(({-0.564}, 0.376, 0)\) gets through. Unit 3 did not contribute to the output, and its weights get no gradient from this input.</p>
<p>A unit whose \(z\) is negative for every input in the data never gets a gradient and never changes: a <b>dead ReLU</b>. One large update that pushes a bias far below zero can kill units this way. GELU's slope is not zero for negative \(z\) (it is −0.13 at \(z = {-1.5}\)), so a little gradient still gets through.</p>`,
    math: String.raw`\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \odot \mathbb{1}[z \gt 0]`,
    fig: {
      type: "table", caption: "The ReLU of the 2-3-1 network's hidden layer, forward and backward", dense: true,
      cols: ["unit", "z", "h = ReLU(z)", "mask", "∂L/∂h", "∂L/∂z"], align: ["l", "r", "r", "r", "r", "r"],
      rows: [["1", "1.0", "1.0", "1", "−0.564", "−0.564"], ["2", "0.3", "0.3", "1", "0.376", "0.376"], ["3", "−0.7", "0", "0", "−0.846", "0"]],
      highlight: [2],
      note: "The mask is 1 bit per unit per input. PyTorch keeps ReLU's output instead and tests it against 0, so no extra memory is needed.",
    },
    terms: [{ term: "Dead ReLU", def: "A ReLU unit whose pre-activation is negative for every input, so it outputs 0, passes no gradient, and stops learning." }],
    src: ["PyTorch 2.15 documentation; outputs reproduced by tools/verify/part03_torch.py"],
  });

  Deck.slide({
    id: "nn-bp-softmax",
    title: "At the output, softmax and cross-entropy send back p − y",
    example: "tokens ids focus:4 target:5",
    body: String.raw`<p>Part 2 derived the gradient of the cross-entropy loss with respect to the logits: \(\partial L/\partial z = p - y\), where \(p\) is the softmax output and \(y\) is 1 for the target token and 0 elsewhere. Every language model's backward pass starts here.</p>
<p>At position 4 our model has read <code>… WHERE</code> and must predict <code>id</code>. It gives <code>id</code> 0.3460, <code>email</code> 0.3448, <code>age</code> 0.3072 and 0.0020 to the other 110 tokens together. The gradient is −0.6540 on the <code>id</code> logit and +0.3448 and +0.3072 on the two rivals; the 113 entries sum to 0. Where the model is already certain (position 1, <code>FROM</code> at 1.000) the whole vector is nearly 0, and that position teaches almost nothing.</p>
<p>The layer rule takes over from there. The logits are \(fW_U\), with \(f\) the final 16-number vector, so \(\partial L/\partial W_U = f^\top(p - y)\) is a 16 × 113 outer product, and \(\partial L/\partial f = (p - y)W_U^\top\) carries the gradient into the network.</p>`,
    fig: { type: "nn-pgrad", caption: "The output gradient p − y of our trained model at each position of the query", pos: 4 },
  });


  Deck.slide({
    id: "nn-bp-2layer",
    title: "Backpropagation through the 2-3-1 network, checked numerically",
    body: String.raw`<p>The 2-3-1 network at \(x = (1, 2)\) with target \(y = 1\) and loss \(L = (\hat y - y)^2\). Forward: \(z = (1.0, 0.3, {-0.7}),\) \(h = (1.0, 0.3, 0),\) \(\hat y = 0.53,\) \(L = 0.2209.\) Backward, with the layer and ReLU rules:</p>
<ol>
<li>\(\partial L/\partial \hat y = 2(\hat y - y) = {-0.94}\), which is also \(\partial L/\partial b_2\).</li>
<li>\(\partial L/\partial W_2 = h^\top({-0.94}) = ({-0.94}, {-0.282}, 0)\).</li>
<li><span class="nn-nw">\(\partial L/\partial h = {-0.94}\,W_2^\top = ({-0.564}, 0.376, {-0.846})\);</span> the mask leaves \(\partial L/\partial z = ({-0.564}, 0.376, 0)\), which is also \(\partial L/\partial b_1\).</li>
<li>\(\partial L/\partial W_1 = x^\top \partial L/\partial z\), rows \(({-0.564}, 0.376, 0)\) and \(({-1.128}, 0.752, 0)\).</li>
</ol>
<p>The plate checks all 13 against central differences. At \(\varepsilon = 10^{-4}\) the largest disagreement is 4.4 × 10⁻¹³, rounding. Forward differences err in proportion to \(\varepsilon\) (1.4 × 10⁻⁴ at \(\varepsilon = 10^{-4}\)), and at \(\varepsilon = 0.5\) the nudges cross ReLU kinks: three gradients come out wrong because unit 2 switches off (W₁[1,2], W₁[2,2], b₁[2]) and one because unit 3 switches on (−0.181 for W₁[2,3] instead of 0).</p>`,
    fig: { type: "nn-fdcheck", caption: "Each gradient from one backward pass next to a finite difference that reruns the forward pass", eps: 1e-4 },
    take: "Checking backpropagation against central differences on a small case is the standard test of a hand-written backward pass.",
  });

  Deck.slide({
    id: "nn-bp-cost",
    title: "The backward pass costs about twice the forward pass",
    body: String.raw`<p>A linear layer \(Y = XW\) with \(X\) of size \(B \times n\) and \(W\) of size \(n \times m\) costs \(2Bnm\) FLOPs forward: one multiply and one add per weight per row. Its backward pass computes two products of the same size, \(\partial L/\partial X = \Delta W^\top\) and \(\partial L/\partial W = X^\top\Delta\): \(4Bnm\) FLOPs.</p>
<p>Per weight and per token that is 2 FLOPs forward and 4 backward, so training costs about 6 FLOPs per matrix weight per token, 3 times inference: the 6 in the \(6ND\) estimate of training compute (Part 7), with \(N\) the weights that take part in matrix multiplies. Our model has 7,952 of them; its 1,936 embedding entries are looked up and its 80 norm gains multiply element by element. So \(6 \times 7{,}952 \times 6{,}144{,}000 \approx 2.9 \times 10^{11}\) FLOPs, plus \(0.19 \times 10^{11}\) for attention's \(8 \times 8\) score and mixing products, \(\approx 3.1 \times 10^{11}\). Counting all 9,968 parameters gives \(3.7 \times 10^{11}\), 18% too many.</p>
<p>Two corrections, both small: the first layer can skip \(\partial L/\partial X\), since nothing below it needs a gradient, and element-wise operations such as ReLU cost \(O(Bm)\) per layer against \(O(Bnm)\) for the multiply.</p>`,
    math: String.raw`\underbrace{2Bnm}_{XW} \;+\; \underbrace{2Bnm}_{\Delta W^\top} \;+\; \underbrace{2Bnm}_{X^\top\Delta} \;=\; 6Bnm`,
    fig: {
      type: "calc", caption: "FLOPs of one linear layer, forward and backward",
      inputs: [
        { id: "B", label: "rows B (tokens)", min: 1, max: 4096, step: 1, value: 1024, fmt: "int" },
        { id: "n", label: "inputs n", min: 1, max: 128, step: 1, value: 16, fmt: "int" },
        { id: "m", label: "outputs m", min: 1, max: 128, step: 1, value: 64, fmt: "int" },
      ],
      outputs: [
        { label: "forward 2Bnm", f: (v) => 2 * v.B * v.n * v.m, fmt: "int" },
        { label: "backward 4Bnm", f: (v) => 4 * v.B * v.n * v.m, fmt: "int" },
        { label: "training step 6Bnm", f: (v) => 6 * v.B * v.n * v.m, fmt: "int" },
      ],
      note: (v) => `${int(v.n * v.m)} weights × ${int(v.B)} rows: 2 FLOPs each forward, 4 backward. The defaults are our model's first MLP matrix on one training batch.`,
    },
  });

  Deck.slide({
    id: "nn-bp-memory",
    title: "Training keeps every activation until the backward pass uses it",
    example: "tokens ids",
    body: String.raw`<p>The backward pass reads values saved by the forward pass: \(\partial L/\partial W = X^\top\Delta\) needs each layer's input \(X\), and an activation function's backward needs its input or its output. None can be freed before the backward pass has used it, so when the forward pass ends, every saved activation of every layer is in memory at once.</p>
<p>Our model's MLP block on one training batch (128 queries × 8 positions = 1,024 tokens) saves its input (1,024 × 16), the pre-activations GELU's backward needs (1,024 × 64) and the GELU output the second matrix's backward needs (1,024 × 64): 147,456 numbers, 589,824 bytes in float32. Its weights are 2,048 numbers, 8,192 bytes. The saved activations are 72 times larger.</p>
<p>Activation memory grows with batch size times sequence length; weight memory does not. Inference keeps none of this for a backward pass, though a transformer does keep each layer's keys and values for later tokens (the KV cache, Part 8). Part 7 counts activation memory for large models and trades recomputation for it.</p>`,
    fig: {
      type: "calc", caption: "What one MLP block keeps for its backward pass",
      inputs: [
        { id: "T", label: "tokens in the batch", min: 1, max: 4096, step: 1, value: 1024, fmt: "int" },
        { id: "d", label: "width d", min: 1, max: 128, step: 1, value: 16, fmt: "int" },
        { id: "f", label: "hidden size", min: 1, max: 512, step: 1, value: 64, fmt: "int" },
        { id: "k", label: "activation", options: [{ label: "GELU: pre-activation and activation saved", value: 2 }, { label: "ReLU: activation saved", value: 1 }], value: 2 },
      ],
      outputs: [
        { label: "saved numbers", f: (v) => v.T * (v.d + v.k * v.f), fmt: "int" },
        { label: "saved bytes, float32", f: (v) => 4 * v.T * (v.d + v.k * v.f), fmt: "int" },
        { label: "weight bytes, float32", f: (v) => 4 * 2 * v.d * v.f, fmt: "int" },
        { label: "activations ÷ weights", f: (v) => (v.T * (v.d + v.k * v.f)) / (2 * v.d * v.f), fmt: "x" },
      ],
      note: (v) => `${int(v.T)} × (${v.d} + ${v.k === 2 ? `2 × ${v.f}` : v.f}) numbers: the block's input, then ${v.k === 2 ? "GELU's input, which GELU's backward needs, and its output, which the second matrix's backward needs" : "ReLU's output, which serves both its own backward and the second matrix's"}.`,
    },
    check: {
      q: "You double the batch size. Which memory doubles?",
      options: ["The weights", "The saved activations", "The gradients of the weights", "None: memory is fixed per parameter"], answer: 1,
      why: "Weights and their gradients have one number per parameter whatever the batch. Saved activations have one row per token: 147,456 numbers at 1,024 tokens, 294,912 at 2,048.",
    },
    src: ["PyTorch 2.15 documentation; outputs reproduced by tools/verify/part03_torch.py"],
  });


  // ------------------------------------------------------------------ 3.4 Autograd and PyTorch
  Deck.chapter({ title: "Autograd and PyTorch" });

  Deck.slide({
    id: "nn-tensors",
    title: "A tensor has a shape, a dtype and a device",
    body: String.raw`<p>A <b>tensor</b> is an n-dimensional array with a <b>shape</b> (the size of each dimension), a <b>dtype</b> (the number format of every entry) and a <b>device</b> (CPU memory or a particular GPU's memory). Its size in bytes is the product of the shape times the bytes per entry: 4 for float32, 2 for bfloat16 (Part 7 covers number formats).</p>
<p>Every operation checks shapes. A matrix multiply needs the inner sizes to agree. Element-wise operations <b>broadcast</b>: shapes are compared from the last dimension, and each pair of sizes must be equal or one of them 1, a missing dimension counting as 1. That is how a bias of shape (3) is added to every row of a 3 × 3 matrix without being copied.</p>
<p>Our model is 20 tensors. The largest are the token embedding table (113 × 16) and the unembedding (16 × 113), 7,232 bytes each in float32. All 9,968 parameters take 39,872 bytes; Llama 3 405B's take 810 GB in bfloat16.</p>`,
    fig: {
      type: "nn-code", lang: "python", caption: "Shapes, dtypes and bytes, with the outputs PyTorch prints",
      code: `import torch
X = torch.tensor([[1., 2.], [-1., 1.], [2., -1.]])
X.shape                        # torch.Size([3, 2])
X.dtype                        # torch.float32
X.element_size() * X.numel()   # 24: 6 numbers × 4 bytes
W = torch.randn(2, 3)
(X @ W).shape                  # torch.Size([3, 3])
(X @ W + torch.zeros(3)).shape # torch.Size([3, 3])
X.to(torch.bfloat16).element_size()   # 2
X.to("cuda")                   # a copy in GPU memory
wte = torch.zeros(113, 16)     # our token embedding table
wte.numel() * wte.element_size()      # 7232`,
      notes: {
        2: "A tensor built from Python floats gets the default dtype, float32.",
        3: "The shape is (rows, columns): 3 inputs of 2 features.",
        5: "Bytes = number of entries × bytes per entry.",
        7: "Inner sizes agree (2 and 2), so (3 × 2) @ (2 × 3) is 3 × 3.",
        8: "A tensor of shape (3) broadcasts across the 3 rows: the same bias is added to each.",
        9: "bfloat16 halves the memory: 2 bytes per entry. Training large models stores most tensors this way (Part 7).",
        10: "Operations need all their inputs on the same device. <code>.to(\"cuda\")</code> copies the data from CPU memory into GPU memory; Part 6 covers what that copy costs.",
        12: "113 × 16 × 4 bytes = 7,232 bytes.",
      },
    },
    terms: [
      { term: "Tensor", def: "An n-dimensional array with a shape, a dtype and a device; the basic data structure of PyTorch." },
      { term: "dtype", def: "The number format of a tensor's entries, such as float32 (4 bytes) or bfloat16 (2 bytes)." },
      { term: "Broadcasting", def: "Applying an element-wise operation to tensors of different shapes. Compared from the last dimension, each pair of sizes must be equal or one of them 1; size-1 and missing dimensions are repeated without copying." },
    ],
    src: ["PyTorch 2.15 documentation; outputs reproduced by tools/verify/part03_torch.py", "Dubey et al., The Llama 3 Herd of Models, 2024 (405B parameters)"],
  });

  Deck.slide({
    id: "nn-autograd",
    title: "requires_grad records the graph; backward() runs it in reverse",
    body: String.raw`<p>PyTorch builds the computational graph while your code runs. A tensor created with <code>requires_grad=True</code> is a leaf that wants a gradient. Every operation on it returns a tensor whose <code>grad_fn</code> is the backward function of that operation, linked to the backward functions of its inputs: the graph, stored as pointers.</p>
<p><code>L.backward()</code> walks that graph from <code>L</code> to the leaves and stores each leaf's gradient in its <code>.grad</code>. The listing is the one-point example of chapter 3.3, and PyTorch returns the numbers computed there by hand: −1.8 for \(w\) and −1.2 for \(b\).</p>
<p>The graph is rebuilt on every forward pass and freed by <code>backward()</code>, along with the values it saved. So Python control flow (a loop whose length depends on the input, an if on a value) changes the graph from one step to the next with no extra work.</p>`,
    fig: {
      type: "nn-code", lang: "python", caption: "Autograd on L = (w·x + b − y)², with PyTorch's printed output",
      code: `import torch
w = torch.tensor(0.8, requires_grad=True)
b = torch.tensor(0.2, requires_grad=True)
x, y = torch.tensor(1.5), torch.tensor(2.0)

L = (w * x + b - y) ** 2   # tensor(0.3600, grad_fn=&lt;PowBackward0&gt;)
L.grad_fn.next_functions   # ((&lt;SubBackward0 ...&gt;, 0),)
L.backward()               # walk the graph from L to w and b
w.grad, b.grad             # (tensor(-1.8000), tensor(-1.2000))`,
      notes: {
        2: "A leaf tensor. <code>requires_grad=True</code> tells PyTorch to record every operation that uses it.",
        4: "Data, not parameters: no gradient is computed for x and y.",
        6: "Four operations run and four backward nodes are recorded. <code>L</code> holds a pointer to the last one, the square.",
        7: "Each backward node points to the nodes of its inputs. The subtraction's node points on to the addition's, and so on down to w and b.",
        8: "Runs each node's backward function in reverse order, multiplying by local derivatives: the same steps as the graph plate of chapter 3.3, <i>The backward pass multiplies local derivatives</i>.",
        9: "−1.2 × 1.5 = −1.8 for w; −1.2 for b.",
      },
    },
    terms: [{ term: "Autograd", def: "PyTorch's automatic differentiation: it records operations on tensors that require gradients and runs backpropagation through them when <code>backward()</code> is called." }],
    src: ["PyTorch 2.15 documentation; outputs reproduced by tools/verify/part03_torch.py"],
  });

  Deck.slide({
    id: "nn-loop",
    title: "The loop that trained our model",
    example: "tokens ids",
    body: String.raw`<p>Every training run is the same five steps in a loop: draw a batch, run the forward pass, compute the loss, run the backward pass, update the parameters. The listing is the loop of the script that trained our model, shortened: batch sampling and the model's forward function are defined elsewhere in the file.</p>
<p>Each step draws 128 fresh queries, which give 1,024 next-token predictions; the loss is their mean cross-entropy (Part 2). Over 6,000 steps each of the 9,968 parameters is updated 6,000 times. The order inside the loop matters: the old gradients must be cleared before <code>backward()</code> adds the new ones, and the update must come after both.</p>
<p>Scaling up changes how each line runs (Parts 6 and 7) but not the lines themselves.</p>`,
    fig: {
      type: "nn-code", lang: "python", caption: "The training loop of tools/build_model.py, shortened",
      code: `opt = torch.optim.AdamW(params, lr=0.01, betas=(0.9, 0.95),
                        weight_decay=0.0)
for step in range(6000):
    ids = sample_batch(128)                 # 128 × 9 token ids
    logits = forward(params, ids[:, :-1])   # 128 × 8 × 113
    loss = F.cross_entropy(logits.reshape(-1, 113),
                           ids[:, 1:].reshape(-1))
    for g in opt.param_groups:
        g["lr"] = lr_at(step)               # warmup, then decay
    opt.zero_grad()                         # drop old gradients
    loss.backward()                         # 9,968 new gradients
    opt.step()                              # update every parameter`,
      notes: {
        1: "AdamW with weight decay 0 is Adam (chapter 3.5). <code>params</code> is the list of our model's 20 weight tensors.",
        3: "6,000 steps. No epochs: the grammar has 270 queries and every batch is freshly sampled from it.",
        4: "128 random queries from the grammar, each 9 tokens.",
        5: "The model sees the first 8 tokens of each query and returns 113 logits at each of the 8 positions.",
        6: "Cross-entropy of the next token at each position, averaged over all 1,024 predictions. Inputs are 1,024 × 113 logits and 1,024 targets.",
        8: "The learning rate is set by hand each step: 100 warmup steps, then cosine decay from 0.01 to 0.001 (chapter 3.5).",
        10: "<code>backward()</code> adds into <code>.grad</code>, so last step's gradients are cleared first (last sheet of this chapter).",
        11: "One backward pass through the graph recorded by lines 5 and 6.",
        12: "Reads every <code>.grad</code> and changes every parameter in place.",
      },
    },
    src: ["tools/build_model.py, the script that trained our model (training loop and optimizer settings)"],
  });

  Deck.slide({
    id: "nn-step",
    title: "optimizer.step() updates each parameter in place",
    body: String.raw`<p><code>opt.step()</code> reads each parameter's <code>.grad</code> and overwrites the parameter. For plain stochastic gradient descent that is one subtraction per parameter, <code>p -= lr * p.grad</code>, run under <code>torch.no_grad()</code> so the update itself is not recorded in a graph.</p>
<p>On the one-point example (\(w = 0.8\), \(b = 0.2\), gradients −1.8 and −1.2) with learning rate 0.1: \(w\) becomes 0.98, \(b\) becomes 0.32, and the loss falls from 0.36 to 0.0441. <code>torch.optim.SGD</code> computes the same two numbers.</p>
<p>Other optimizers also keep <b>optimizer state</b> per parameter from step to step. Adam (chapter 3.5) keeps two running averages, so for our model AdamW holds 19,936 numbers next to the 9,968 parameters. Part 7 shows that optimizer state takes most of the 16 bytes per parameter that mixed-precision training stores.</p>`,
    fig: {
      type: "nn-code", lang: "python", caption: "One SGD step on the one-point example",
      code: `lr = 0.1
with torch.no_grad():         # not recorded in any graph
    w -= lr * w.grad          # 0.8 - 0.1 × (-1.8) = 0.98
    b -= lr * b.grad          # 0.2 - 0.1 × (-1.2) = 0.32

# the same step through the optimizer API:
#   opt = torch.optim.SGD([w, b], lr=0.1); opt.step()

L = (w * x + b - y) ** 2      # tensor(0.0441, grad_fn=&lt;PowBackward0&gt;)`,
      notes: {
        2: "Inside <code>no_grad()</code> operations are not recorded. Without it PyTorch refuses this in-place update (\"a leaf Variable that requires grad is being used in an in-place operation\"), and the out-of-place <code>w = w - lr * w.grad</code> would turn <code>w</code> into a non-leaf result of the graph whose <code>.grad</code> stays empty.",
        3: "In place: the tensor keeps its identity, so the optimizer and the model still point at it.",
        7: "<code>SGD.step()</code> runs the same subtraction for every parameter in its list, plus momentum or weight decay if asked for.",
        9: "(0.98 × 1.5 + 0.32 − 2)² = (−0.21)² = 0.0441, down from 0.36.",
      },
    },
    terms: [{ term: "Optimizer state", def: "Numbers an optimizer keeps per parameter between steps: none for plain SGD, one velocity for momentum, two running averages for Adam." }],
    src: ["PyTorch 2.15 documentation; outputs reproduced by tools/verify/part03_torch.py"],
  });

  Deck.slide({
    id: "nn-accum",
    title: "Gradients accumulate, which is why zero_grad() exists",
    body: String.raw`<p><code>backward()</code> adds into <code>.grad</code>; it never overwrites it. Run the one-point example's forward and backward twice and <code>w.grad</code> is −3.6, twice −1.8. A loop that forgets <code>zero_grad()</code> updates with the sum of every gradient so far.</p>
<p>The addition is useful on purpose. When a batch does not fit in memory, split it into \(k\) equal-size <b>micro-batches</b>, call <code>backward()</code> on each one's mean loss divided by \(k\), then step once. The accumulated gradient equals the gradient of the full batch's mean loss. For a batch of 128 split into 4 micro-batches of 32, PyTorch's two gradients differ by at most 7.5 × 10⁻⁹ in float32 (largest gradient entry 0.03) and by 1.2 × 10⁻¹⁷ in float64: rounding only.</p>
<p>The cost is time, not memory: four smaller passes run one after another. Part 7 uses this to reach batches of millions of tokens.</p>`,
    fig: {
      type: "nn-code", lang: "python", caption: "Accumulation by accident and on purpose",
      code: `((w * x + b - y) ** 2).backward()
((w * x + b - y) ** 2).backward()  # a new graph, same values
w.grad                             # tensor(-3.6000)

opt.zero_grad()                    # every .grad back to None
for i in range(4):                 # 4 micro-batches of 32
    xb, yb = X[32*i:32*(i+1)], Y[32*i:32*(i+1)]
    loss = F.cross_entropy(model(xb), yb) / 4
    loss.backward()                # .grad += this share
opt.step()                         # one update for all 128`,
      notes: {
        2: "The first <code>backward()</code> freed its graph, so the forward pass is run again.",
        3: "−1.8 + (−1.8): the second gradient was added to the first.",
        5: "By default (<code>set_to_none=True</code>), <code>zero_grad()</code> sets each <code>.grad</code> to None, and the next <code>backward()</code> allocates it fresh.",
        8: "Dividing by 4 turns the sum of 4 micro-batch means into the mean over all 128 examples.",
        9: "Only one micro-batch's activations are alive at a time: a quarter of the activation memory.",
        10: "The same update as one batch of 128, up to floating-point rounding.",
      },
    },
    check: {
      q: "A loop calls <code>opt.zero_grad()</code> once, before step 1, and never again. Which gradient does step 3 use?",
      options: ["The gradient of batch 3", "The sum of the gradients of batches 1, 2 and 3", "The mean of the gradients of batches 1, 2 and 3", "Zero"], answer: 1,
      why: "Each <code>backward()</code> adds into <code>.grad</code>. After the third one, <code>.grad</code> holds g₁ + g₂ + g₃: a sum, not a mean, so the effective step grows each iteration.",
    },
    terms: [{ term: "Micro-batch", def: "A slice of a batch processed by its own forward and backward pass; the gradients of all slices are accumulated before one optimizer step." }],
    src: ["PyTorch 2.15 documentation; outputs reproduced by tools/verify/part03_torch.py"],
  });


  // ================================================================== numerical core for chapters 3.5 and 3.6
  // A small dense MLP. W[l] is (n × m) row-major: W[l][i*m + j] connects input i to unit j.
  NN3.mlp = {
    create(sizes, act, init, seed = 1) {
      const rand = TinyLM.mulberry32(seed), W = [], b = [];
      for (let l = 0; l < sizes.length - 1; l++) {
        const n = sizes[l], m = sizes[l + 1], w = new Float64Array(n * m);
        for (let i = 0; i < n * m; i++) w[i] = typeof init === "number" ? init : NN3.gauss(rand) / Math.sqrt(n);
        W.push(w); b.push(new Float64Array(m));
      }
      return { sizes, act, W, b };
    },
    // X: Float64Array of B rows; returns every pre-activation and activation
    forward(net, X, B) {
      const { sizes, W, b } = net, A = NN3.ACT[net.act], hs = [X], zs = [null];
      for (let l = 0; l < sizes.length - 1; l++) {
        const n = sizes[l], m = sizes[l + 1], hh = hs[l], z = new Float64Array(B * m), w = W[l], bl = b[l];
        for (let r = 0; r < B; r++) {
          const ro = r * n, zo = r * m;
          for (let j = 0; j < m; j++) z[zo + j] = bl[j];
          for (let i = 0; i < n; i++) { const v = hh[ro + i]; if (v === 0) continue; const wo = i * m; for (let j = 0; j < m; j++) z[zo + j] += v * w[wo + j]; }
        }
        zs.push(z);
        hs.push(l === sizes.length - 2 ? z : z.map(A.f));
      }
      return { hs, zs };
    },
    // mean loss over the batch and its gradients; "mse": (ŷ − y)², "bce": logistic loss on the logit ŷ
    grads(net, X, Y, B, loss) {
      const { sizes, W } = net, A = NN3.ACT[net.act], { hs, zs } = NN3.mlp.forward(net, X, B), Lc = sizes.length - 1, out = hs[Lc];
      let L = 0, delta = new Float64Array(B);
      for (let r = 0; r < B; r++) {
        const zv = out[r];
        if (loss === "mse") { const e = zv - Y[r]; L += e * e; delta[r] = (2 * e) / B; }
        else { L += Y[r] ? NN3.softplus(-zv) : NN3.softplus(zv); delta[r] = (1 / (1 + Math.exp(-zv)) - Y[r]) / B; }
      }
      const gW = [], gb = [];
      for (let l = Lc - 1; l >= 0; l--) {
        const n = sizes[l], m = sizes[l + 1], hh = hs[l], gw = new Float64Array(n * m), gbl = new Float64Array(m);
        for (let r = 0; r < B; r++) {
          const ro = r * n, d0 = r * m;
          for (let j = 0; j < m; j++) gbl[j] += delta[d0 + j];
          for (let i = 0; i < n; i++) { const v = hh[ro + i]; if (v === 0) continue; const wo = i * m; for (let j = 0; j < m; j++) gw[wo + j] += v * delta[d0 + j]; }
        }
        gW[l] = gw; gb[l] = gbl;
        if (l > 0) {
          const nd = new Float64Array(B * n), z = zs[l], a = hs[l], w = W[l];
          for (let r = 0; r < B; r++) for (let i = 0; i < n; i++) {
            let acc = 0;
            const wo = i * m, d0 = r * m;
            for (let j = 0; j < m; j++) acc += delta[d0 + j] * w[wo + j];
            nd[r * n + i] = acc * A.d(z[r * n + i], a[r * n + i]);
          }
          delta = nd;
        }
      }
      return { loss: L / B, gW, gb };
    },
    loss(net, X, Y, B, loss) {
      const out = NN3.mlp.forward(net, X, B).hs[net.sizes.length - 1];
      let L = 0, ok = 0;
      for (let r = 0; r < B; r++) {
        const zv = out[r];
        if (loss === "mse") L += (zv - Y[r]) ** 2;
        else { L += Y[r] ? NN3.softplus(-zv) : NN3.softplus(zv); if ((zv > 0) === (Y[r] === 1)) ok++; }
      }
      return { loss: L / B, acc: ok / B };
    },
    sgd(net, g, lr) {
      net.W.forEach((w, l) => { for (let i = 0; i < w.length; i++) w[i] -= lr * g.gW[l][i]; });
      net.b.forEach((bb, l) => { for (let i = 0; i < bb.length; i++) bb[i] -= lr * g.gb[l][i]; });
    },
    adamState: (net) => ({ t: 0, m: [...net.W, ...net.b].map((a) => new Float64Array(a.length)), v: [...net.W, ...net.b].map((a) => new Float64Array(a.length)) }),
    adam(net, st, g, lr, b1 = 0.9, b2 = 0.95, eps = 1e-8) {
      st.t++;
      const P = [...net.W, ...net.b], G = [...g.gW, ...g.gb], c1 = 1 - Math.pow(b1, st.t), c2 = 1 - Math.pow(b2, st.t);
      P.forEach((p, k) => {
        const m = st.m[k], v = st.v[k], gg = G[k];
        for (let i = 0; i < p.length; i++) {
          m[i] = b1 * m[i] + (1 - b1) * gg[i];
          v[i] = b2 * v[i] + (1 - b2) * gg[i] * gg[i];
          p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
        }
      });
    },
  };
  NN3.softplus = (z) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));

  // XOR with the 2-3-1 tanh network: what equal, zero and random starting weights do
  NN3.xor = { X: Float64Array.from([0, 0, 0, 1, 1, 0, 1, 1]), Y: [0, 1, 1, 0] };
  NN3.symNet = (mode) => NN3.mlp.create([2, 3, 1], "tanh", mode === "half" ? 0.5 : mode === "zero" ? 0 : null, 1);
  NN3.symTrain = (net, steps, lr = 0.1) => {
    for (let s = 0; s < steps; s++) NN3.mlp.sgd(net, NN3.mlp.grads(net, NN3.xor.X, NN3.xor.Y, 4, "mse"), lr);
    return NN3.mlp.loss(net, NN3.xor.X, NN3.xor.Y, 4, "mse").loss;
  };

  // Standard deviation of pre-activations (forward) and of ∂L/∂z (backward) through L layers of
  // width N, for weights of standard deviation g/√N. One fixed set of random matrices for every g.
  NN3.GAINS = [0.5, 1, Math.SQRT2, 2, Math.sqrt(128)];
  NN3.depthStats = (() => {
    const N = 128, B = 64, L = 10, cache = new Map();
    let base = null;
    const std = (a) => { let m = 0; for (const v of a) m += v; m /= a.length; let q = 0; for (const v of a) q += (v - m) * (v - m); return Math.sqrt(q / a.length); };
    function draw() {
      const rand = TinyLM.mulberry32(2024);
      const Ws = Array.from({ length: L }, () => Float64Array.from({ length: N * N }, () => NN3.gauss(rand)));
      const X = Float64Array.from({ length: B * N }, () => NN3.gauss(rand));
      const G = Float64Array.from({ length: B * N }, () => NN3.gauss(rand));
      return { Ws, X, G };
    }
    function run(act, gain) {
      if (!base) base = draw();
      const { Ws, X, G } = base, A = NN3.ACT[act], sc = gain / Math.sqrt(N);
      let hcur = X, satCount = 0;
      const zs = [], as = [], fwd = [std(X)];
      for (let l = 0; l < L; l++) {
        const z = new Float64Array(B * N), W = Ws[l];
        for (let r = 0; r < B; r++) for (let i = 0; i < N; i++) { const v = hcur[r * N + i] * sc; if (v === 0) continue; const wo = i * N, zo = r * N; for (let j = 0; j < N; j++) z[zo + j] += v * W[wo + j]; }
        const a = z.map(A.f);
        if (act === "sigmoid") for (const v of a) if (v < 0.02 || v > 0.98) satCount++;
        zs.push(z); as.push(a); fwd.push(std(z)); hcur = a;
      }
      let gh = G;
      const bwd = new Array(L + 1).fill(NaN);
      for (let l = L - 1; l >= 0; l--) {
        const z = zs[l], a = as[l], gz = new Float64Array(B * N), W = Ws[l];
        for (let k = 0; k < B * N; k++) gz[k] = gh[k] * A.d(z[k], a[k]);
        bwd[l + 1] = std(gz);
        const nh = new Float64Array(B * N);
        for (let r = 0; r < B; r++) for (let i = 0; i < N; i++) { let acc = 0; const wo = i * N, go = r * N; for (let j = 0; j < N; j++) acc += gz[go + j] * W[wo + j]; nh[r * N + i] = acc * sc; }
        gh = nh;
      }
      return { fwd, bwd, sat: satCount / (L * B * N) };
    }
    return (act) => {
      if (!cache.has(act)) cache.set(act, NN3.GAINS.map((g) => run(act, g)));
      return cache.get(act);
    };
  })();

  // Gradient size through L blocks, plain h ← φ(hW) against residual h ← h + φ(hW)/√L, width 128.
  NN3.residStats = (() => {
    const N = 128, B = 32, LMAX = 50, cache = new Map();
    let base = null;
    const rms = (a) => { let q = 0; for (const v of a) q += v * v; return Math.sqrt(q / a.length); };
    function run(act, L, resid) {
      if (!base) {
        const rand = TinyLM.mulberry32(5);
        base = { Ws: Array.from({ length: LMAX }, () => Float64Array.from({ length: N * N }, () => NN3.gauss(rand))), X: Float64Array.from({ length: B * N }, () => NN3.gauss(rand)), G: Float64Array.from({ length: B * N }, () => NN3.gauss(rand)) };
      }
      const { Ws, X, G } = base, A = NN3.ACT[act], sc = 1 / Math.sqrt(N), br = resid ? 1 / Math.sqrt(L) : 1;
      let hcur = X;
      const zs = [], as = [];
      for (let l = 0; l < L; l++) {
        const z = new Float64Array(B * N), W = Ws[l];
        for (let r = 0; r < B; r++) for (let i = 0; i < N; i++) { const v = hcur[r * N + i] * sc; if (v === 0) continue; const wo = i * N, zo = r * N; for (let j = 0; j < N; j++) z[zo + j] += v * W[wo + j]; }
        const a = z.map(A.f), nh = new Float64Array(B * N);
        for (let k = 0; k < B * N; k++) nh[k] = (resid ? hcur[k] : 0) + br * a[k];
        zs.push(z); as.push(a); hcur = nh;
      }
      let gh = G;
      const gn = new Array(L + 1);
      gn[L] = rms(gh);
      for (let l = L - 1; l >= 0; l--) {
        const z = zs[l], a = as[l], gz = new Float64Array(B * N), W = Ws[l], nh = new Float64Array(B * N);
        for (let k = 0; k < B * N; k++) gz[k] = br * gh[k] * A.d(z[k], a[k]);
        for (let r = 0; r < B; r++) for (let i = 0; i < N; i++) { let acc = 0; const wo = i * N, go = r * N; for (let j = 0; j < N; j++) acc += gz[go + j] * W[wo + j]; nh[r * N + i] = acc * sc + (resid ? gh[r * N + i] : 0); }
        gh = nh; gn[l] = rms(gh);
      }
      return gn;
    }
    return (act, L) => {
      const key = act + L;
      if (!cache.has(key)) cache.set(key, { plain: run(act, L, false), resid: run(act, L, true) });
      return cache.get(key);
    };
  })();

  // Gradient descent with momentum (PyTorch's form: v ← βv + g, w ← w − ηv) on L = w₁² + k·w₂²
  NN3.valley = (w, k = 10) => w[0] * w[0] + k * w[1] * w[1];
  NN3.momentumPath = (eta, beta, steps, start = [-4, 1.5], k = 10) => {
    let w = start.slice(), v = [0, 0];
    const path = [w.slice()];
    for (let t = 0; t < steps; t++) {
      const g = [2 * w[0], 2 * k * w[1]];
      v = t === 0 ? g : [beta * v[0] + g[0], beta * v[1] + g[1]];
      w = [w[0] - eta * v[0], w[1] - eta * v[1]];
      path.push(w.slice());
      if (!Number.isFinite(NN3.valley(w, k)) || Math.abs(w[0]) > 1e6 || Math.abs(w[1]) > 1e6) break;
    }
    return path;
  };
  // steps until L < thr; null if not reached within max steps (see stepsLabel for why)
  NN3.stepsTo = (eta, beta, thr = 1e-3, k = 10, max = 5000) => {
    const path = NN3.momentumPath(eta, beta, max, [-4, 1.5], k);
    const i = path.findIndex((w) => NN3.valley(w, k) < thr);
    return i < 0 ? null : i;
  };
  NN3.stepsLabel = (eta, beta, max = 5000) => {
    const n = NN3.stepsTo(eta, beta, 1e-3, 10, max);
    if (n !== null) return String(n);
    const path = NN3.momentumPath(eta, beta, max), last = path[path.length - 1];
    return path.length <= max || !(NN3.valley(last) < 1e6) ? "diverges" : `> ${int(max)}`;
  };
  // Adam on the same valley, every intermediate kept (for the worked table)
  NN3.adamValley = (lr, steps, b1 = 0.9, b2 = 0.95, eps = 1e-8, start = [-4, 1.5]) => {
    let w = start.slice(), m = [0, 0], v = [0, 0];
    const rows = [];
    for (let t = 1; t <= steps; t++) {
      const g = [2 * w[0], 20 * w[1]];
      m = m.map((x, i) => b1 * x + (1 - b1) * g[i]);
      v = v.map((x, i) => b2 * x + (1 - b2) * g[i] * g[i]);
      const mh = m.map((x) => x / (1 - Math.pow(b1, t))), vh = v.map((x) => x / (1 - Math.pow(b2, t)));
      const d = mh.map((x, i) => (-lr * x) / (Math.sqrt(vh[i]) + eps));
      w = w.map((x, i) => x + d[i]);
      rows.push({ t, g, m: m.slice(), v: v.slice(), mh, vh, d, w: w.slice() });
    }
    return rows;
  };
  // SGD and RMSProp (ρ = 0.95, no bias correction) on L = ½(25a² + b²) from (−1, 1)
  NN3.bowl = (w) => 0.5 * (25 * w[0] * w[0] + w[1] * w[1]);
  NN3.bowlPath = (kind, eta, steps, rho = 0.95, eps = 1e-8) => {
    let w = [-1, 1], s = [0, 0];
    const rows = [{ w: w.slice(), s: s.slice() }];
    for (let t = 0; t < steps; t++) {
      const g = [25 * w[0], w[1]];
      if (kind === "sgd") w = [w[0] - eta * g[0], w[1] - eta * g[1]];
      else { s = s.map((v, i) => rho * v + (1 - rho) * g[i] * g[i]); w = w.map((v, i) => v - (eta * g[i]) / (Math.sqrt(s[i]) + eps)); }
      rows.push({ w: w.slice(), s: s.slice() });
      if (!(NN3.bowl(w) < 1e12)) break;
    }
    return rows;
  };
  NN3.bowlStepsTo = (kind, eta, thr = 1e-3, max = 5000) => {
    const rows = NN3.bowlPath(kind, eta, max), i = rows.findIndex((r) => NN3.bowl(r.w) < thr);
    if (i >= 0) return String(i);
    return rows.length <= max || !(NN3.bowl(rows[rows.length - 1].w) < 1e6) ? "diverges" : `> ${int(max)}`;
  };

  // our model's learning rate at a step, exactly as tools/build_model.py computes it
  NN3.lrAt = (step) => (step < C.warmup ? (C.lr * (step + 1)) / C.warmup : C.lr * (0.1 + 0.9 * 0.5 * (1 + Math.cos((Math.PI * (step - C.warmup)) / (C.steps - C.warmup)))));


  // ------------------------------------------------------------ Fig: symmetric starting weights
  Deck.fig("nn-sym", (el, sp) => {
    let mode = sp.mode || "half", net, step = 0;
    const tbody = h("tbody"), tiles = h("div", { class: "f-tiles" });
    const table = h("table", { class: "f-table dense" }, h("thead", null, h("tr", null, ["", "unit 1", "unit 2", "unit 3"].map((c, j) => h("th", { class: j ? "r" : "" }, c)))), tbody);
    const reset = () => { net = NN3.symNet(mode); step = 0; draw(); };
    const run = (k) => { NN3.symTrain(net, k); step += k; draw(); };
    function draw() {
      const W = net.W[0], b = net.b[0], W2 = net.W[1];
      const rows = [["weight from x₁", W[0], W[1], W[2]], ["weight from x₂", W[3], W[4], W[5]], ["bias", b[0], b[1], b[2]], ["weight to the output", W2[0], W2[1], W2[2]]];
      clear(tbody);
      rows.forEach((r) => tbody.appendChild(h("tr", null, h("td", null, r[0]), r.slice(1).map((v) => h("td", { class: "r" }, fix(v, 4))))));
      const L = NN3.mlp.loss(net, NN3.xor.X, NN3.xor.Y, 4, "mse").loss;
      const same = [0, 1, 2].every((j) => Math.abs(W[j] - W[0]) < 1e-12 && Math.abs(W[3 + j] - W[3]) < 1e-12 && Math.abs(b[j] - b[0]) < 1e-12 && Math.abs(W2[j] - W2[0]) < 1e-12);
      clear(tiles);
      tiles.append(tile(int(step), "steps of gradient descent"), tile(L < 1e-4 ? sci(L, 1) : fix(L, 4), "loss on the 4 XOR points"), tile(fix(net.b[1][0], 4), "output bias"), tile(same ? "yes" : "no", "all 3 units identical"));
    }
    el.append(
      h("div", { class: "f-ctl" }, segmented({ label: "Starting weights", value: mode, options: [{ label: "all 0.5", value: "half" }, { label: "all 0", value: "zero" }, { label: "random, std 1/√fan-in", value: "random" }], onChange: (v) => { mode = v; reset(); } })),
      h("div", { class: "scroll-x" }, table), tiles,
      h("div", { class: "f-ctl" }, button("Step ▸", () => run(1), { kind: "primary" }), button("100 steps", () => run(100)), button("1,000 steps", () => run(1000)), button("Reset", reset)));
    reset();
    if (sp.steps) run(sp.steps);
  });

  // ------------------------------------------------------------ Fig: signal size through 10 layers, forward or backward
  Deck.fig("nn-depth", (el, sp) => {
    const mode = sp.mode || "forward";
    let act = sp.act || "linear";
    const LABELS = ["g = 0.5", "g = 1", "g = √2", "g = 2", "σ = 1 (g = √128)"], CLS = ["s1", "s2", "s3", "s4", "s5"];
    const host = h("div", { class: "nn-chart" }), tiles = h("div", { class: "f-tiles nn-tiles5" });
    const ch = chart(host, {
      width: 560, height: 300, margin: { l: 54, r: 14, t: 12, b: 40 }, aria: mode === "forward" ? "standard deviation of pre-activations by layer" : "standard deviation of the gradient by layer",
      x: { domain: [0, 10], ticks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], label: mode === "forward" ? "layer" : "layer (the gradient flows from 10 down to 1)" },
      y: { type: "log", domain: [1e-3, 1e3] }, series: [],
    });
    function draw() {
      const st = NN3.depthStats(act);
      const series = st.map((r, i) => {
        const arr = mode === "forward" ? r.fwd : r.bwd;
        const pts = arr.map((v, l) => [l, v]).filter((p) => Number.isFinite(p[1]) && p[1] > 0);
        return { points: pts, cls: CLS[i] };
      });
      const all = series.flatMap((se) => se.points.map((p) => p[1]));
      let lo = Math.floor(Math.log10(Math.min(...all, 0.1))), hi = Math.ceil(Math.log10(Math.max(...all, 10)));
      const span = hi - lo, stepD = span > 12 ? 4 : span > 6 ? 2 : 1;
      lo = Math.floor(lo / stepD) * stepD; hi = Math.ceil(hi / stepD) * stepD;
      const tks = []; for (let e = lo; e <= hi; e += stepD) tks.push(Math.pow(10, e));
      ch.update({
        y: { type: "log", domain: [Math.pow(10, lo), Math.pow(10, hi)], ticks: tks, fmt: pow10 },
        series,
        dots: series.map((se, i) => { const p = se.points[se.points.length - 1]; return { x: p[0], y: p[1], r: 3.5, cls: CLS[i], tip: `${LABELS[i]}: ${sci(p[1], 2)} at layer ${p[0]}` }; }),
      });
      clear(tiles);
      st.forEach((r, i) => tiles.append(tile(mode === "forward" ? sci(r.fwd[10], 1) : sci(r.bwd[1] / r.bwd[10], 1), mode === "forward" ? `${LABELS[i]}: std at layer 10` : `${LABELS[i]}: layer 1 ÷ layer 10`)));
    }
    const note = h("p", { class: "f-note" });
    const setNote = () => {
      const st = NN3.depthStats(act);
      note.textContent = `Width 128, a batch of 64 random inputs, weights drawn once from a normal distribution and scaled to standard deviation g/√128. ${mode === "forward" ? "Each line is the standard deviation of the pre-activations z of one layer." : "A random gradient enters at layer 10; each line is the standard deviation of ∂L/∂z as it travels back."}` +
        (mode === "backward" && act === "sigmoid" ? ` The σ = 1 line does not vanish: its weights are 11 times those of the g = 1 line, which offsets the small slopes, but ${Math.round(100 * st[4].sat)}% of its sigmoid outputs are saturated, below 0.02 or above 0.98.` : "");
    };
    el.append(
      h("div", { class: "f-ctl" }, segmented({ label: "Activation", value: act, options: [{ label: "linear", value: "linear" }, { label: "tanh", value: "tanh" }, { label: "ReLU", value: "relu" }, { label: "sigmoid", value: "sigmoid" }], onChange: (v) => { act = v; draw(); setNote(); } })),
      host,
      h("div", { class: "legend" }, LABELS.map((l, i) => h("span", { class: "key" }, h("span", { class: `sw sw-${CLS[i]}` }), l))),
      tiles, note);
    draw(); setNote();
  });

  // ------------------------------------------------------------ Fig: which axis each normalization averages over
  Deck.fig("nn-normaxes", (el, sp, api) => {
    const ids = M.runningIds.slice(0, C.T), tr = M.forward(ids), X = tr.x0, R = X.length, D = X[0].length, eps = C.eps;
    let mode = sp.mode || "rms";
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };
    const rmsOf = (a) => Math.sqrt(mean(a.map((v) => v * v)));
    function apply(md) {
      if (md === "input") return X.map((r) => r.slice());
      if (md === "rms") return X.map((r) => { const q = Math.sqrt(mean(r.map((v) => v * v)) + eps); return r.map((v) => v / q); });
      if (md === "layer") return X.map((r) => { const m = mean(r), v = mean(r.map((x) => (x - m) ** 2)); return r.map((x) => (x - m) / Math.sqrt(v + eps)); });
      const cols = Array.from({ length: D }, (_, j) => X.map((r) => r[j]));
      const st = cols.map((c) => { const m = mean(c), v = mean(c.map((x) => (x - m) ** 2)); return [m, Math.sqrt(v + eps)]; });
      return X.map((r) => r.map((x, j) => (x - st[j][0]) / st[j][1]));
    }
    const wrap = h("div", { class: "scroll-x" }), note = h("p", { class: "f-note" });
    const NOTE = {
      input: "The residual stream entering layer 1: token embedding plus position embedding, before any normalization.",
      batch: "BatchNorm: every column (one feature, across the 8 positions) now has mean 0 and standard deviation 1. A row's values depend on the other rows.",
      layer: "LayerNorm: every row (one token's 16 features) now has mean 0 and standard deviation 1, so its RMS is 1 too.",
      rms: "RMSNorm: every row is divided by its root mean square, so its RMS is 1. Its mean is not moved to 0.",
    };
    function draw() {
      const Y = apply(mode), mx = 3;
      const t = h("table", { class: "f-heat nn-heat" });
      t.appendChild(h("tr", null, h("th", null, ""), h("th", { class: "nn-stat" }, "mean"), h("th", { class: "nn-stat" }, "RMS"), Array.from({ length: D }, (_, j) => h("th", null, String(j)))));
      Y.forEach((row, i) => t.appendChild(h("tr", { class: i === 4 ? "nn-focus" : "" }, h("th", { class: "nn-rowh" }, chip(M.running[i])),
        h("td", { class: "nn-stat" }, fix(mean(row), 2)), h("td", { class: "nn-stat" }, fix(rmsOf(row), 2)),
        row.map((v, j) => h("td", { style: { background: UI.div(v, mx) }, "data-tip": `${M.running[i].trim()}, feature ${j}: ${fix(v, 3)}` })))));
      const cols = Array.from({ length: D }, (_, j) => Y.map((r) => r[j]));
      t.appendChild(h("tr", { class: "nn-colstat" }, h("th", null, "column mean"), h("td", null, ""), h("td", null, ""), cols.map((c) => h("td", null, fix(mean(c), 1)))));
      t.appendChild(h("tr", { class: "nn-colstat" }, h("th", null, "column std"), h("td", null, ""), h("td", null, ""), cols.map((c) => h("td", null, fix(sd(c), 1)))));
      clear(wrap); wrap.appendChild(t);
      note.textContent = NOTE[mode];
    }
    el.append(h("div", { class: "f-ctl" }, segmented({ label: "Normalization", value: mode, options: [{ label: "none", value: "input" }, { label: "BatchNorm", value: "batch" }, { label: "LayerNorm", value: "layer" }, { label: "RMSNorm", value: "rms" }], onChange: (v) => { mode = v; draw(); } })), wrap, note);
    draw();
  });

  // ------------------------------------------------------------ Fig: RMSNorm on one position, every number
  Deck.fig("nn-rms", (el, sp, api) => {
    const ids = M.runningIds.slice(0, C.T), tr = M.forward(ids), g = Array.from(M.weights()["l0.n1"].data), eps = C.eps;
    let pos = sp.pos ?? 4;
    const tbody = h("tbody"), tiles = h("div", { class: "f-tiles" });
    const table = h("table", { class: "f-table dense nn-rms-t" }, h("thead", null, h("tr", null, ["i", "x", "x²", "x / RMS", "gain g", "output"].map((c, j) => h("th", { class: j ? "r" : "" }, c)))), tbody);
    function draw() {
      const x = tr.x0[pos], ms = x.reduce((a, v) => a + v * v, 0) / x.length, rr = Math.sqrt(ms + eps);
      const y = x.map((v, i) => (v / rr) * g[i]);
      const dev = Math.max(...y.map((v, i) => Math.abs(v - tr.layers[0].h1[pos][i])));
      clear(tbody);
      x.forEach((v, i) => tbody.appendChild(h("tr", null, h("td", null, String(i)), h("td", { class: "r" }, fix(v, 3)), h("td", { class: "r" }, fix(v * v, 3)), h("td", { class: "r" }, fix(v / rr, 3)), h("td", { class: "r" }, fix(g[i], 4)), h("td", { class: "r" }, fix(y[i], 3)))));
      clear(tiles);
      tiles.append(tile(fix(ms, 4), "mean of x²"), tile(fix(rr, 4), "RMS = √(mean + 10⁻⁵)"), tile(dev < 1e-12 ? "exactly" : sci(dev, 1), "matches the model's own h₁"));
      if (api && api.strip) api.strip.set({ focus: [pos] });
    }
    el.append(h("div", { class: "f-ctl sliders" }, slider({ label: "Position", min: 0, max: 7, step: 1, value: pos, format: (v) => `${v}: ${M.running[v].trim()}`, onInput: (v) => { pos = v; draw(); } })),
      h("div", { class: "scroll-x" }, table), tiles);
    draw();
    return { leave: () => api && api.strip && api.strip.reset() };
  });

  // ------------------------------------------------------------ Fig: gradient size with and without residual connections
  Deck.fig("nn-resid", (el, sp) => {
    let act = sp.act || "sigmoid", L = sp.L || 20;
    const host = h("div", { class: "nn-chart" }), tiles = h("div", { class: "f-tiles" });
    const ch = chart(host, {
      width: 560, height: 280, margin: { l: 54, r: 14, t: 12, b: 40 }, aria: "gradient size per layer, plain and residual",
      x: { domain: [0, 20], label: "layer (the gradient flows from the top down to 0)" }, y: { type: "log", domain: [1e-3, 10] }, series: [],
    });
    function draw() {
      const r = NN3.residStats(act, L);
      const pts = (a) => a.map((v, l) => [l, v]);
      const all = r.plain.concat(r.resid);
      let lo = Math.floor(Math.log10(Math.min(...all, 0.1))), hi = Math.ceil(Math.log10(Math.max(...all, 10)));
      const stepD = hi - lo > 12 ? 4 : hi - lo > 6 ? 2 : 1;
      lo = Math.floor(lo / stepD) * stepD; hi = Math.ceil(hi / stepD) * stepD;
      const tks = []; for (let e = lo; e <= hi; e += stepD) tks.push(Math.pow(10, e));
      const xt = L === 10 ? [0, 2, 4, 6, 8, 10] : L === 20 ? [0, 5, 10, 15, 20] : [0, 10, 20, 30, 40, 50];
      ch.update({
        x: { domain: [0, L], ticks: xt, label: "layer (the gradient flows from the top down to 0)" },
        y: { type: "log", domain: [Math.pow(10, lo), Math.pow(10, hi)], ticks: tks, fmt: pow10 },
        series: [{ points: pts(r.plain), cls: "s2" }, { points: pts(r.resid), cls: "s1" }],
        dots: [{ x: 0, y: r.plain[0], r: 4, cls: "s2", tip: `plain: ${sci(r.plain[0], 2)} at the input` }, { x: 0, y: r.resid[0], r: 4, cls: "s1", tip: `residual: ${sci(r.resid[0], 2)} at the input` }],
      });
      clear(tiles);
      tiles.append(tile(sci(r.plain[0] / r.plain[L], 1), "plain: input ÷ top"), tile(sci(r.resid[0] / r.resid[L], 1), "residual: input ÷ top"));
    }
    el.append(
      h("div", { class: "f-ctl" },
        segmented({ label: "Activation", value: act, options: [{ label: "sigmoid", value: "sigmoid" }, { label: "tanh", value: "tanh" }, { label: "ReLU", value: "relu" }], onChange: (v) => { act = v; draw(); } }),
        segmented({ label: "Blocks", value: L, options: [10, 20, 50].map((v) => ({ label: String(v), value: v })), onChange: (v) => { L = v; draw(); } })),
      host,
      h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "plain: h ← φ(hW)"), h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "residual: h ← h + φ(hW)/√L")),
      tiles,
      h("p", { class: "f-note" }, "Width 128, weights of standard deviation 1/√128, the same matrices for both networks. The line is the root mean square of ∂L/∂h at each layer after a random gradient enters at the top."));
    draw();
  });

  // ------------------------------------------------------------ Fig: gradient descent and momentum on the narrow valley
  Deck.fig("nn-mom", (el, sp) => {
    let eta = sp.eta ?? 0.09, beta = sp.beta ?? 0.5, k = sp.show ?? 3;
    const W0 = 460, H0 = 240, X = (v) => ((v + 5) / 10) * W0, Y = (v) => H0 / 2 - (v / 2) * (H0 / 2);
    const svg = s("svg", { viewBox: `0 0 ${W0} ${H0}`, class: "nn-mom", role: "img", "aria-label": "paths of gradient descent and momentum on L = w1² + 10 w2²" });
    const info = h("p", { class: "f-mm-expl", "aria-live": "polite" }), tiles = h("div", { class: "f-tiles" });
    const inBox = (p) => Math.abs(p[0]) <= 5 && Math.abs(p[1]) <= 2;
    function draw() {
      clear(svg);
      svg.appendChild(s("rect", { x: 0, y: 0, width: W0, height: H0, class: "nn-frame" }));
      [0.25, 1, 2.5, 5, 10, 16, 25].forEach((c) => svg.appendChild(s("ellipse", { cx: X(0), cy: Y(0), rx: (Math.sqrt(c) / 10) * W0, ry: (Math.sqrt(c / 10) / 4) * H0, class: "nn-contour" })));
      svg.appendChild(s("line", { x1: 0, x2: W0, y1: Y(0), y2: Y(0), class: "nn-gridl" }));
      svg.appendChild(s("line", { x1: X(0), x2: X(0), y1: 0, y2: H0, class: "nn-gridl" }));
      svg.appendChild(s("text", { x: W0 - 6, y: Y(0) - 6, class: "nn-tick", "text-anchor": "end" }, "w₁ (flat)"));
      svg.appendChild(s("text", { x: X(0) + 6, y: 14, class: "nn-tick" }, "w₂ (steep)"));
      const paths = [[NN3.momentumPath(eta, 0, k), "s1"], [NN3.momentumPath(eta, beta, k), "s2"]];
      for (const [path, cls] of paths) {
        const vis = path.filter(inBox);
        svg.appendChild(s("polyline", { points: vis.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" "), class: `nn-path ${cls}` }));
        vis.forEach((p, i) => svg.appendChild(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: i === vis.length - 1 ? 4.5 : 2.2, class: `nn-pathdot ${cls}`, "data-tip": `step ${path.indexOf(p)}: (${fix(p[0], 3)}, ${fix(p[1], 3)}), L = ${fix(NN3.valley(p), 4)}` })));
      }
      svg.appendChild(s("circle", { cx: X(-4), cy: Y(1.5), r: 4, class: "nn-start" }));
      svg.appendChild(s("text", { x: X(0), y: Y(0) + 5, class: "nn-min", "text-anchor": "middle" }, "×"));
      const g = paths[0][0], m = paths[1][0], last = (p) => p[p.length - 1];
      const where = (p) => (p.length <= k ? "diverged" : `(${fix(last(p)[0], 3)}, ${fix(last(p)[1], 3)}), L = ${fix(NN3.valley(last(p)), 4)}`);
      info.innerHTML = `After ${k} step${k === 1 ? "" : "s"} from (−4, 1.5): gradient descent at ${where(g)}; momentum at ${where(m)}.`;
      clear(tiles);
      tiles.append(tile(NN3.stepsLabel(eta, 0), "steps to L < 10⁻³, plain"), tile(NN3.stepsLabel(eta, beta), `steps to L < 10⁻³, β = ${fix(beta, 2)}`));
    }
    el.append(h("div", { class: "nn-mom-wrap" }, svg),
      h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "gradient descent"), h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "with momentum")),
      info,
      h("div", { class: "f-ctl" }, button("Step ▸", () => { k++; draw(); }, { kind: "primary" }), button("10 steps", () => { k += 10; draw(); }), button("Reset", () => { k = 0; draw(); })),
      h("div", { class: "f-ctl sliders" },
        slider({ label: "Momentum β", min: 0, max: 0.95, step: 0.05, value: beta, format: (v) => fix(v, 2), onInput: (v) => { beta = v; draw(); } }),
        slider({ label: "Learning rate η", min: 0.01, max: 0.12, step: 0.005, value: eta, format: (v) => fix(v, 3), onInput: (v) => { eta = v; draw(); } })),
      tiles);
    draw();
  });

  // ------------------------------------------------------------ Fig: our model's learning-rate schedule
  Deck.fig("nn-sched", (el, sp) => {
    let t = sp.step ?? 3050, zoom = false;
    const host = h("div", { class: "nn-chart" }), tiles = h("div", { class: "f-tiles" });
    const ch = chart(host, {
      width: 560, height: 260, margin: { l: 70, r: 16, t: 12, b: 40 }, aria: "learning rate by training step",
      x: { domain: [0, C.steps], label: "training step" }, y: { domain: [0, 0.011], ticks: [0, 0.002, 0.004, 0.006, 0.008, 0.01], fmt: (v) => v.toFixed(3), label: "learning rate" }, series: [],
    });
    let sl;
    function draw() {
      const x1 = zoom ? 300 : C.steps, n = zoom ? 300 : 600;
      const pts = []; for (let i = 0; i <= n; i++) { const st = Math.min(C.steps - 1, Math.round((x1 * i) / n)); pts.push([st, NN3.lrAt(st)]); }
      const xt = zoom ? [0, 50, 100, 150, 200, 250, 300] : [0, 1000, 2000, 3000, 4000, 5000, 6000];
      const shown = Math.min(t, x1);
      ch.update({
        x: { domain: [0, x1], ticks: xt, fmt: (v) => int(v), label: "training step" },
        series: [{ points: pts, cls: "s1" }], vlines: zoom ? [{ x: 100, label: "warmup ends at step 100" }] : [],
        dots: [{ x: shown, y: NN3.lrAt(shown), r: 5, cls: "s2", tip: `step ${int(shown)}: ${NN3.lrAt(shown).toFixed(6)}` }],
      });
      clear(tiles);
      tiles.append(tile(int(t), "step"), tile(NN3.lrAt(t).toFixed(5), "learning rate"), tile(`${fix((100 * NN3.lrAt(t)) / C.lr, 1)}%`, "of the peak 0.01"));
    }
    el.append(h("div", { class: "f-ctl" }, segmented({ label: "Steps shown", value: false, options: [{ label: "all 6,000", value: false }, { label: "first 300", value: true }], onChange: (v) => { zoom = v; sl.input.max = zoom ? 299 : C.steps - 1; if (zoom && t > 299) { t = 50; sl.set(50); } draw(); } })),
      host,
      h("div", { class: "f-ctl sliders" }, (sl = slider({ label: "Step", min: 0, max: C.steps - 1, step: 1, value: t, format: (v) => int(v), onInput: (v) => { t = v; draw(); } }))),
      tiles);
    draw();
  });

  // ------------------------------------------------------------ Fig: clipping by global norm vs each tensor alone
  Deck.fig("nn-clip", (el, sp) => {
    let ga = sp.ga ?? 1.5, gb = sp.gb ?? 2, c = sp.c ?? 1;
    const S = 300, R = 3, P = (v) => S / 2 + (v / R) * (S / 2 - 10), Q = (v) => S / 2 - (v / R) * (S / 2 - 10);
    const uid = UI.nextId("nnclip");
    const svg = s("svg", { viewBox: `0 0 ${S} ${S}`, class: "nn-clip", role: "img", "aria-label": "a gradient, the clipping circle and the two clipped gradients" });
    const tiles = h("div", { class: "f-tiles" });
    function arrow(x, y, cls) {
      svg.appendChild(s("line", { x1: P(0), y1: Q(0), x2: P(x), y2: Q(y), class: `nn-vec ${cls}`, "marker-end": `url(#${uid}-${cls})` }));
    }
    function draw() {
      clear(svg);
      svg.appendChild(s("defs", null, ["g", "s1", "s2"].map((cls) => s("marker", { id: `${uid}-${cls}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: `nn-ah ${cls}` })))));
      svg.appendChild(s("rect", { x: 0, y: 0, width: S, height: S, class: "nn-frame" }));
      for (let v = -R; v <= R; v += 0.5) {
        svg.appendChild(s("line", { x1: P(v), x2: P(v), y1: Q(-R), y2: Q(R), class: v === 0 ? "nn-axis" : "nn-gridl" }));
        svg.appendChild(s("line", { x1: P(-R), x2: P(R), y1: Q(v), y2: Q(v), class: v === 0 ? "nn-axis" : "nn-gridl" }));
      }
      svg.appendChild(s("circle", { cx: P(0), cy: Q(0), r: (c / R) * (S / 2 - 10), class: "nn-ball" }));
      svg.appendChild(s("rect", { x: P(-c), y: Q(c), width: P(c) - P(-c), height: Q(-c) - Q(c), class: "nn-box" }));
      const n = Math.hypot(ga, gb), sc = n > c ? c / n : 1;
      const pa = Math.sign(ga) * Math.min(Math.abs(ga), c), pb = Math.sign(gb) * Math.min(Math.abs(gb), c);
      arrow(ga, gb, "g"); arrow(ga * sc, gb * sc, "s1"); arrow(pa, pb, "s2");
      svg.appendChild(s("text", { x: P(R) - 4, y: Q(0) - 6, class: "nn-tick", "text-anchor": "end" }, "∂L/∂a"));
      svg.appendChild(s("text", { x: P(0) + 6, y: Q(R) + 12, class: "nn-tick" }, "∂L/∂b"));
      const ang = (x, y) => (Math.atan2(y, x) * 180) / Math.PI;
      clear(tiles);
      tiles.append(tile(fix(n, 2), "global norm ‖g‖"), tile(fix(sc, 3), "scale min(1, c/‖g‖)"),
        tile(`(${fix(ga * sc, 2)}, ${fix(gb * sc, 2)})`, "clipped by global norm"),
        tile(`(${fix(pa, 2)}, ${fix(pb, 2)})`, `each one alone: direction off by ${fix(Math.abs(ang(pa, pb) - ang(ga, gb)), 1)}°`));
    }
    el.append(h("div", { class: "nn-clip-wrap" }, svg),
      h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-ink" }), "gradient g"), h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "clipped by global norm"), h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "each parameter clipped alone")),
      h("div", { class: "f-ctl sliders" },
        slider({ label: "∂L/∂a", min: -3, max: 3, step: 0.1, value: ga, format: (v) => fix(v, 1), onInput: (v) => { ga = v; draw(); } }),
        slider({ label: "∂L/∂b", min: -3, max: 3, step: 0.1, value: gb, format: (v) => fix(v, 1), onInput: (v) => { gb = v; draw(); } }),
        slider({ label: "Threshold c", min: 0.25, max: 3, step: 0.25, value: c, format: (v) => fix(v, 2), onInput: (v) => { c = v; draw(); } })),
      tiles);
    draw();
  });


  // ------------------------------------------------------------------ 3.5 Making deep networks train
  Deck.chapter({ title: "Making deep networks train" });

  Deck.slide({
    id: "nn-init-symmetry",
    title: "Equal starting weights keep every hidden unit identical",
    body: String.raw`<p>Two hidden units that start with the same incoming weights, bias and outgoing weight compute the same value on every input. The backward pass then gives them the same gradients, so they take the same steps and stay equal forever. The layer acts as one unit, however wide it is. Random starting weights break this <b>symmetry</b>.</p>
<p>The plate trains the 2-3-1 network with tanh hidden units on XOR's four points (squared error, gradient descent, \(\eta = 0.1\)). From all weights 0.5, after 2,000 steps the three units are still identical (every first-layer weight is 1.7577) and the loss is 0.1753. It keeps falling but can never go below 1/6: the network computes one increasing or decreasing function of \(x_1 + x_2\), which cannot output 0, 1, 0. All zeros is worse: tanh(0) = 0, so every weight gradient is exactly 0, only the output bias moves (to 0.5) and the loss stays at 0.25. From random weights with standard deviation \(1/\sqrt{\text{fan-in}}\) the loss is below 10⁻⁷ after 500 steps.</p>
<p>Biases can start at 0: once the weights differ, the units do too.</p>`,
    fig: { type: "nn-sym", caption: "The 2-3-1 network on XOR after 2,000 steps from all-equal weights. Switch the start and train again.", mode: "half", steps: 2000 },
    terms: [{ term: "Symmetry (of hidden units)", def: "Units that start with identical weights get identical gradients and stay identical. Random initialization breaks the symmetry." }],
  });

  Deck.slide({
    id: "nn-init-depth",
    title: "Activations shrink or grow geometrically with depth",
    body: String.raw`<p>A unit with \(n\) inputs, its <b>fan-in</b>, sums \(n\) products of a weight and an input. With independent zero-mean weights of standard deviation \(\sigma\), the variance of that sum is \(n\sigma^2\) times the mean square of the inputs. Write \(\sigma = g/\sqrt{n}\): each linear layer multiplies the signal's standard deviation by \(g\), and 10 layers by \(g^{10}\).</p>
<p>The plate measures it in 10 layers of width 128. Linear layers: \(g = 0.5\) ends at 0.0009 and \(g = 2\) at 955. Unscaled weights, \(\sigma = 1\), are \(g = \sqrt{128} = 11.3\) and end at \(3.2 \times 10^{10}\). Only \(g = 1\), that is \(\sigma = 1/\sqrt{n}\), keeps the signal near 1 (0.93 after 10 layers).</p>
<p>ReLU sets about half its inputs to 0 and so halves the mean square it passes on. At \(g = 1\), nine ReLUs before layer 10 predict \((1/\sqrt2)^9 = 0.044\) there; this draw of weights gives 0.033, PyTorch at width 512 gives 0.047. ReLU needs \(g = \sqrt2\).</p>`,
    math: String.raw`\mathrm{Var}[z] = n\,\sigma^2\,\mathbb{E}[x^2]`,
    fig: { type: "nn-depth", caption: "Standard deviation of each layer's pre-activations for five weight scales", mode: "forward", act: "linear" },
    terms: [{ term: "Fan-in", def: "The number of inputs a unit sums: the first dimension of a layer's weight matrix in the x W convention." }],
  });

  const W0 = M.weights(0), stdOf = (name) => { const a = Array.from(W0[name].data), m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
  const sd3 = (...names) => names.map((n) => stdOf(n).toFixed(3)).join(", ");
  Deck.slide({
    id: "nn-init-schemes",
    title: "Xavier and He initialization set the scale from the layer's size",
    example: "tokens ids",
    body: String.raw`<p>A layer's <b>fan-out</b> is its number of units. A steady forward signal needs \(\mathrm{Var}[w] = 1/n_{in}\), with \(n_{in}\) the fan-in. The backward pass sums over the units each input feeds, so a steady gradient needs \(1/n_{out}\). Glorot and Bengio proposed the compromise \(2/(n_{in} + n_{out})\), now called <b>Xavier initialization</b>.</p>
<p>He et al. redid the derivation for ReLU, which halves the mean square: \(\mathrm{Var}[w] = 2/n_{in}\), <b>He initialization</b>. They report a 30-layer ReLU network that stalled with Xavier's scale and trained with theirs.</p>
<p>Our model uses \(1/\sqrt{n_{in}}\) for the matrices that read the residual stream and divides the two that write back into it, attention's output matrix and the MLP's second matrix, by a further \(\sqrt{2L} = 2\). That is GPT-2's rule for keeping a sum of many blocks from growing with depth. The plate lists each tensor with the standard deviation measured on the saved step-0 weights.</p>`,
    math: [String.raw`\text{Xavier: } \mathrm{Var}[w] = \frac{2}{n_{in} + n_{out}}`, String.raw`\text{He: } \mathrm{Var}[w] = \frac{2}{n_{in}}`],
    fig: {
      type: "table", caption: "Our model's initialization, and the standard deviations measured at step 0", dense: true,
      cols: ["tensor", "shape", "rule", "target std", "measured"], align: ["l", "l", "l", "r", "r"],
      rows: [
        ["token embeddings", "113 × 16", "fixed", "0.5", sd3("wte")],
        ["position embeddings", "8 × 16", "fixed", "0.5", sd3("wpe")],
        [String.raw`\(W_Q, W_K, W_V\), layer 1`, "16 × 16", String.raw`\(1/\sqrt{16}\)`, "0.25", sd3("l0.wq", "l0.wk", "l0.wv")],
        [String.raw`\(W_Q, W_K, W_V\), layer 2`, "16 × 16", String.raw`\(1/\sqrt{16}\)`, "0.25", sd3("l1.wq", "l1.wk", "l1.wv")],
        [String.raw`\(W_O\), layers 1 and 2`, "16 × 16", String.raw`\(1/\sqrt{16} \div 2\)`, "0.125", sd3("l0.wo", "l1.wo")],
        ["MLP first matrix, layers 1 and 2", "16 × 64", String.raw`\(1/\sqrt{16}\)`, "0.25", sd3("l0.w1", "l1.w1")],
        ["MLP second matrix, layers 1 and 2", "64 × 16", String.raw`\(1/\sqrt{64} \div 2\)`, "0.0625", sd3("l0.w2", "l1.w2")],
        ["unembedding", "16 × 113", "small", "0.02", sd3("wu")],
      ].map((row) => row.map((c, j) => (j === 1 ? c.replace(/ × /g, "\u00a0×\u00a0") : c))),
      note: "Measured over all entries of each matrix in the step-0 snapshot. A 16 × 16 matrix has 256 entries, so its measured value scatters a few percent around the target. The RMSNorm gains start at exactly 1.",
    },
    src: ["Glorot & Bengio, Understanding the difficulty of training deep feedforward neural networks, AISTATS 2010", "He, Zhang, Ren & Sun, Delving Deep into Rectifiers, ICCV 2015 (arXiv:1502.01852)", "Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2), 2019"],
    terms: [
      { term: "Fan-out", def: "The number of units in a layer, which is how many sums each input feeds." },
      { term: "Xavier initialization", def: "Weights with variance \\(2/(n_{in} + n_{out})\\), balancing the forward and backward signals (Glorot & Bengio, 2010)." },
      { term: "He initialization", def: "Weights with variance \\(2/n_{in}\\), which keeps the signal steady through ReLU layers (He et al., 2015)." },
    ],
  });

  Deck.slide({
    id: "nn-grad-depth",
    title: "Gradients vanish or explode through depth the same way",
    body: String.raw`<p>The backward pass multiplies by \(W^\top\) and by the activation's slope at every layer, so the gradient's size changes by a factor per layer, as the forward signal's does. The variance factor is \(n\sigma^2\) times the mean square of the slope.</p>
<p>The sigmoid's slope is at most 0.25. In the plate's 10-layer sigmoid network at \(g = 1\), the gradient reaching layer 1 is \(2.0 \times 10^{-6}\) of the gradient at layer 10: the last layers learn while the first barely move, which is called <b>vanishing gradients</b>. ReLU at \(g = \sqrt 2\) keeps the ratio at 0.94. ReLU at \(g = 2\) multiplies the gradient by 21 on its way down: <b>exploding gradients</b>, where a step that suits the last layer is far too large for the first.</p>
<p>This is why sigmoid networks more than a few layers deep were hard to train. Scaled initialization only fixes the start; normalization and residual connections (next sheets) keep the signal in range as the weights change.</p>`,
    fig: { type: "nn-depth", caption: "Standard deviation of the gradient ∂L/∂z at each layer as it travels back from layer 10", mode: "backward", act: "sigmoid" },
    src: ["Bengio, Simard & Frasconi, Learning long-term dependencies with gradient descent is difficult, IEEE Trans. Neural Networks, 1994", "Glorot & Bengio, AISTATS 2010"],
    terms: [
      { term: "Vanishing gradients", def: "Gradients that shrink by a factor at each layer of the backward pass, so early layers barely learn." },
      { term: "Exploding gradients", def: "Gradients that grow by a factor at each layer of the backward pass, so early layers take steps far too large." },
    ],
  });

  Deck.slide({
    id: "nn-norm-kinds",
    title: "BatchNorm, LayerNorm and RMSNorm normalize along different axes",
    example: "tokens ids focus:4",
    layout: "wide",
    body: String.raw`<p>A normalization layer rescales activations to a fixed size on every forward pass, then multiplies each feature by a learned gain. Initialization sets the scale once; the weights then change. In our model the vector at position 4 leaving layer 2 has RMS 0.64 at step 0, 4.10 at step 400 and 15.5 at step 6,000, 24 times larger, yet the RMSNorm after it still passes on a vector of RMS 1 before the gains. The three normalizations in use differ in what they average over.</p>
<ul>
<li><b>BatchNorm</b> (Ioffe & Szegedy, 2015): each feature is shifted to mean 0 and scaled to variance 1 across the batch, so one example's output depends on the others; inference uses averages saved during training.</li>
<li><b>LayerNorm</b> (Ba, Kiros & Hinton, 2016): each token's vector is shifted to mean 0 and scaled to variance 1 across its own features.</li>
<li><b>RMSNorm</b> (Zhang & Sennrich, 2019): each token's vector is divided by its root mean square, with no mean subtraction: one statistic per vector instead of two. Its authors report LayerNorm's quality with 7% to 64% less running time.</li>
</ul>
<p>Transformers use the per-token kinds. Our model applies RMSNorm before each attention and MLP sublayer and once more before the output: 5 RMSNorms with 16 gains each.</p>`,
    fig: { type: "nn-normaxes", caption: "Our query's 8 × 16 residual stream entering layer 1 (positions × features), normalized three ways. Blue is negative, red positive.", mode: "rms" },
    src: ["Ioffe & Szegedy, Batch Normalization, ICML 2015", "Ba, Kiros & Hinton, Layer Normalization, arXiv:1607.06450, 2016", "Zhang & Sennrich, Root Mean Square Layer Normalization, NeurIPS 2019"],
    terms: [
      { term: "BatchNorm", def: "Normalizes each feature to mean 0 and variance 1 across the examples of a batch, then applies a learned gain and bias." },
      { term: "LayerNorm", def: "Normalizes each example's vector to mean 0 and variance 1 across its own features, then applies a learned gain and bias." },
      { term: "RMSNorm", def: "Divides each vector by its root mean square \\(\\sqrt{\\tfrac{1}{d}\\sum x_i^2 + \\varepsilon}\\), then multiplies by a learned gain. No mean subtraction, no bias." },
    ],
  });

  Deck.slide({
    id: "nn-rmsnorm",
    title: "RMSNorm, worked on one vector of our model",
    example: "tokens ids focus:4",
    body: String.raw`<p>At position 4 our query's vector enters layer 1 as the embedding of <code>WHERE</code> plus the embedding of position 4: 16 numbers between −2.93 and 1.51. RMSNorm averages their squares, 1.6007, adds \(\varepsilon = 10^{-5}\) so that a zero vector cannot cause a division by zero, and takes the square root: the RMS is 1.2652. Dividing by it leaves a vector whose mean square is 1.0000.</p>
<p>Each entry is then multiplied by its learned gain. Training moved these 16 gains from 1 to values between 0.0025 and 0.82: feature 13 is all but switched off at the input of layer 1's attention, and feature 12 passes at 0.82.</p>
<p>The result is exactly the vector the model's own forward pass feeds to layer 1's attention, as the last tile confirms. Slide the position to see the other seven.</p>`,
    math: String.raw`\mathrm{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}}\; g_i`,
    fig: { type: "nn-rms", caption: "The 16 numbers of one position through layer 1's first RMSNorm", pos: 4 },
  });

  Deck.slide({
    id: "nn-residual",
    title: "A residual connection gives the gradient a path around every layer",
    body: String.raw`<p>A <b>residual connection</b> adds a block's input to its output: \(y = x + f(x)\) instead of \(y = f(x)\). Its derivative is \(I + \partial f/\partial x\), so the gradient reaching \(x\) is the gradient at \(y\) itself plus whatever flows back through \(f\). However small \(f\)'s slopes are, the identity term carries the gradient down unchanged.</p>
<p>The plate runs 20 blocks of width 128 both ways. With sigmoid blocks and no residual connections, the gradient at the input is \(2.3 \times 10^{-13}\) of the gradient at the top. With them, each block's output scaled by \(1/\sqrt{20}\) (the factor GPT-2 applies to its residual layers at initialization), the ratio is 1.0. ReLU blocks: \(8.8 \times 10^{-4}\) against 1.3.</p>
<p>He et al. introduced residual connections to train image networks 152 layers deep. Every transformer block has two, around attention and around the MLP; Part 5 follows the residual stream they form through our model.</p>`,
    math: String.raw`\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\Big(I + \frac{\partial f}{\partial x}\Big)`,
    fig: { type: "nn-resid", caption: "Size of the gradient at each layer, with and without residual connections", act: "sigmoid", L: 20 },
    src: ["He, Zhang, Ren & Sun, Deep Residual Learning for Image Recognition, CVPR 2016", "Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2), 2019"],
    terms: [{ term: "Residual connection", def: "Adding a block's input to its output, y = x + f(x), which gives the gradient an identity path around the block." }],
  });

  Deck.slide({
    id: "nn-momentum",
    title: "Momentum adds up steps that agree and cancels steps that alternate",
    body: String.raw`<p>On Part 2's valley \(L = w_1^2 + 10w_2^2\) from (−4, 1.5), gradient descent must keep \(\eta \lt 0.1\) for the steep \(w_2\). At \(\eta = 0.09\) the flat \(w_1\) then shrinks by only 18% per step, and reaching \(L \lt 10^{-3}\) takes 26 steps.</p>
<p><b>Momentum</b> keeps a decaying sum of past gradients, the <b>velocity</b> \(v \leftarrow \beta v + \nabla L\), in which each older gradient is multiplied by \(\beta\) once more per step, and steps along it: \(w \leftarrow w - \eta v\). Along \(w_1\) the gradients agree in sign and add up; along \(w_2\) they alternate and cancel. With \(\beta = 0.5\), the first three \(w_1\) gradients are −8, −6.56 and −4.66 but the velocities are −8, −10.56 and −9.94, so after 3 steps \(w_1 = {-1.44}\) instead of −2.21, and \(L \lt 10^{-3}\) takes 15 steps.</p>
<p>Too much momentum overshoots: \(\beta = 0.9\) needs 75 steps here. It pays off on narrower valleys: on \(w_1^2 + 100w_2^2\) at \(\eta = 0.0099\), \(\beta = 0.9\) needs 92 steps where plain descent needs 307.</p>`,
    math: String.raw`v \leftarrow \beta v + \nabla L, \qquad w \leftarrow w - \eta\, v`,
    fig: { type: "nn-mom", caption: "Gradient descent and momentum on L = w₁² + 10 w₂² from (−4, 1.5), first 3 steps", eta: 0.09, beta: 0.5, show: 3 },
    src: ["Polyak, Some methods of speeding up the convergence of iteration methods, 1964", "PyTorch documentation, torch.optim.SGD (momentum form v ← μv + g)"],
    terms: [
      { term: "Momentum", def: "An optimizer that steps along a decaying sum of past gradients, \\(v \\leftarrow \\beta v + g\\), so consistent directions speed up and oscillating ones cancel." },
      { term: "Velocity", def: "The decaying sum of past gradients that momentum steps along; each older gradient carries one more factor of \\(\\beta\\)." },
    ],
  });

  Deck.slide({
    id: "nn-rmsprop",
    title: "RMSProp gives each parameter its own step size",
    body: String.raw`<p>With one learning rate for all parameters, the steepest direction sets the limit and the flattest crawls. On the bowl \(\tfrac12(25a^2 + b^2)\), with curvatures 25 and 1, plain gradient descent (SGD) diverges above \(\eta = 0.08\), where the factor \(1 - 25\eta\) passes −1. At \(\eta = 0.07\) the flat coordinate shrinks by only 7% per step: from 1 to 0.484 after 10 steps.</p>
<p><b>RMSProp</b> keeps, for each parameter, a running mean of its squared gradient, \(s \leftarrow \rho s + (1 - \rho)g^2\), and divides that parameter's step by its root. Each parameter gets its own effective learning rate \(\eta/\sqrt{s}\): the learning rate over the root mean square of its recent gradients. With \(\rho = 0.95\) and \(\eta = 0.07\), after one step the rates are 0.0125 for \(a\) and 0.313 for \(b\), a ratio of 25, the ratio of the two gradients. Both coordinates then move by equal amounts along the diagonal: after 10 steps \(b = 0.050\), and \(L \lt 10^{-3}\) takes 16 steps against SGD's 43.</p>
<p>Starting \(s\) at 0 makes the first steps too large: step 1 is \(\eta/\sqrt{0.05} = 4.5\eta\). Adam (next sheet) corrects that and adds momentum.</p>`,
    math: [String.raw`s \leftarrow \rho s + (1 - \rho)\, g^2`, String.raw`w \leftarrow w - \eta\, \frac{g}{\sqrt{s} + \varepsilon}`],
    fig: { type: "nn-rms-sgd", caption: "SGD and RMSProp (ρ = 0.95) on ½(25a² + b²) from (−1, 1), first 10 steps", eta: 0.07, show: 10 },
    src: ["Tieleman & Hinton, Neural Networks for Machine Learning, Lecture 6.5 (rmsprop), Coursera, 2012"],
    terms: [{ term: "RMSProp", def: "An optimizer that divides each parameter's gradient by the root of a running mean of its squared gradients, giving each parameter its own effective learning rate \\(\\eta/\\sqrt{s}\\)." }],
  });

  const AD = NN3.adamValley(0.1, 3);
  const sgn4 = (v) => (v > 0 ? "+" : "") + fix(v, 4);
  const ADAM_TABLE = `<div class="scroll-x"><table class="nn-adamtab"><thead><tr><th>t</th><th></th><th class="r">g</th><th class="r">\\(\\hat m\\)</th><th class="r">\\(\\hat v\\)</th><th class="r">step</th></tr></thead><tbody>` +
    AD.map((r) => [0, 1].map((i) => `<tr${i ? "" : ' class="nn-t0"'}>${i ? "" : `<td rowspan="2">${r.t}</td>`}<td>\\(w_${i + 1}\\)</td><td class="r">${fix(r.g[i], 2)}</td><td class="r">${fix(r.mh[i], 3)}</td><td class="r">${fix(r.vh[i], 1)}</td><td class="r">${sgn4(r.d[i])}</td></tr>`).join("")).join("") + `</tbody></table></div>`;
  Deck.slide({
    id: "nn-adam",
    title: "Adam: momentum plus a per-parameter step size",
    example: "tokens ids",
    body: String.raw`<p><b>Adam</b> keeps two running averages per parameter: \(m\) of the gradient (momentum) and \(v\) of its square (RMSProp's scale). Both start at 0, so early on they are too small: after one step \(m = (1 - \beta_1)g\). Dividing by \(1 - \beta^t\), <b>bias correction</b>, removes exactly that factor; the step is \(\eta\,\hat m/(\sqrt{\hat v} + \varepsilon)\).</p>
<p>Three steps with our model's \(\beta_1 = 0.9\), \(\beta_2 = 0.95\), \(\varepsilon = 10^{-8}\) and \(\eta = 0.1\) on Part 2's valley from (−4, 1.5), whose two gradients differ by a factor of 3.75:</p>
${ADAM_TABLE}
<p>Both coordinates move by about \(\eta\) per step whatever their gradient. Without bias correction step 1 would be \(0.1 \cdot 0.1/\sqrt{0.05} = 0.045\). The plate runs Adam with its default \(\beta_2 = 0.999\) against SGD on the previous sheet's bowl. Its lower panel is our model's schedule, explained on the sheet <i>Our learning rate</i>.</p>`,
    math: [String.raw`\begin{aligned} m &\leftarrow \beta_1 m + (1 - \beta_1)\,g \\ v &\leftarrow \beta_2 v + (1 - \beta_2)\,g^2 \end{aligned}`, String.raw`w \leftarrow w - \eta\,\frac{m/(1 - \beta_1^t)}{\sqrt{v/(1 - \beta_2^t)} + \varepsilon}`],
    fig: { type: "tinylm", widget: "optim", caption: "SGD and Adam (β₁ = 0.9, β₂ = 0.999) on the bowl ½(25a² + b²), and our model's learning-rate schedule" },
    check: {
      q: "Two parameters have steady gradients of 0.001 and 1,000. After 100 Adam steps with η = 0.01, how far has each moved?",
      options: ["0.001 and 1,000", "Both about 1", "Both about 0.01", "0.1 and 100"], answer: 1,
      why: String.raw`With a constant gradient, \(\hat m = g\) and \(\hat v = g^2\), so each step is \(\eta\, g/|g| = 0.01\) whatever the gradient's size: 100 steps move both by about 1. Plain gradient descent at the same \(\eta\) would move them by 0.001 and 1,000.`,
    },
    src: ["Kingma & Ba, Adam: A Method for Stochastic Optimization, ICLR 2015"],
    terms: [
      { term: "Adam", def: "An optimizer that keeps running averages of each parameter's gradient (m) and squared gradient (v), corrects their startup bias, and steps by \\(\\eta\\,\\hat m/(\\sqrt{\\hat v} + \\varepsilon)\\)." },
      { term: "Bias correction", def: "Dividing Adam's running averages by \\(1 - \\beta^t\\), which undoes their shrinkage toward the initial 0 in the first steps." },
    ],
  });

  Deck.slide({
    id: "nn-adamw",
    title: "AdamW applies weight decay outside the adaptive step",
    body: String.raw`<p>Weight decay (Part 2) shrinks every weight toward 0 a little each step; written as a penalty \(\tfrac{\lambda}{2}\lVert w \rVert^2\) on the loss, it adds \(\lambda w\) to each gradient (Part 2's penalty \(\lambda \sum c_j^2\) has gradient \(2\lambda c_j\); PyTorch folds the 2 into \(\lambda\)). Inside Adam that sum is then divided by \(\sqrt{\hat v}\), so the decay a weight actually gets depends on its gradient history, not just on \(\lambda\).</p>
<p>Two weights start at 1, with \(\eta = 0.01\) and \(\lambda = 0.1\). One is never used by the loss (gradient 0): with L2 inside Adam its whole gradient is the decay term, which Adam normalizes to a step of about \(\eta\), so after 100 steps it is at 0.125 (the note below shows the same for a 100 times smaller \(\lambda\)). The other has a steady gradient of 10: adding \(\lambda w\) changes its Adam step by nothing, because Adam divides out the gradient's size.</p>
<p><b>AdamW</b> applies decay to the weight directly, \(w \leftarrow w - \eta\lambda w\) in PyTorch's form, then takes the Adam step on the loss gradient alone. Both weights shrink by the same 0.1% per step: the unused one is at 0.905 after 100 steps (\(0.999^{100}\)). Our model used AdamW with \(\lambda = 0\), which is Adam; GPT-3 used \(\lambda = 0.1\).</p>`,
    fig: {
      type: "table", caption: "A weight starting at 1, with η = 0.01, β = (0.9, 0.95), computed with torch.optim", dense: true,
      cols: ["weight", "Adam, no decay", "Adam + L2, λ = 0.1", "AdamW, λ = 0.1"], align: ["l", "r", "r", "r"],
      rows: [
        ["unused (gradient 0), after 1 step", "1.000", "0.990", "0.999"],
        ["unused (gradient 0), after 100 steps", "1.000", "0.125", "0.905"],
        ["steady gradient 10, after 1 step", "0.990", "0.990", "0.989"],
      ],
      highlight: [1],
      note: "With L2 inside Adam, λ = 0.001 also leaves the unused weight at 0.125 after 100 steps: the decay rate is set by η, not by λ.",
    },
    src: ["Loshchilov & Hutter, Decoupled Weight Decay Regularization, ICLR 2019", "PyTorch documentation, torch.optim.AdamW", "Brown et al., Language Models are Few-Shot Learners (GPT-3), 2020, Appendix B"],
    terms: [{ term: "AdamW", def: "Adam with weight decay applied directly to the weights, \\(w \\leftarrow w - \\eta\\lambda w\\), instead of being added to the gradient (Loshchilov & Hutter, 2019)." }],
  });

  Deck.slide({
    id: "nn-schedule",
    title: "Our learning rate: 100 warmup steps, then cosine decay to 0.001",
    example: "tokens ids",
    body: String.raw`<p>A <b>learning-rate schedule</b> changes \(\eta\) during training. Ours starts with <b>warmup</b>, a linear ramp from 0.0001 at step 0 to 0.01 at step 99, then follows a <b>cosine decay</b> from 0.01 at step 100 to 0.001 at step 6,000. Halfway through the decay, at step 3,050, it is 0.0055.</p>
<p>Warmup protects the first steps. Adam's first update moves every parameter by about \(\eta\) whatever its gradient, using statistics from a single batch. At step 0 our model's \(\eta\) is 0.0001, so no weight moves by more than 0.0001; without warmup each would move by about 0.01, a hundred times more, on that one batch's evidence. Decay serves the end: each batch's gradient is noisy, and smaller late steps average the noise out so the loss settles lower.</p>
<p>The same shape at scale: Llama 3 405B warmed up over 8,000 steps to a peak of \(8 \times 10^{-5}\) and decayed to \(8 \times 10^{-7}\) over 1,200,000 steps.</p>`,
    math: String.raw`\begin{aligned} \eta_t &= 0.01\,\big(0.1 + 0.45\,(1 + \cos\theta_t)\big) \\ \theta_t &= \pi\,\tfrac{t - 100}{5{,}900}, \quad 100 \le t \le 6{,}000 \end{aligned}`,
    fig: { type: "nn-sched", caption: "The learning rate of our model at every step, computed as tools/build_model.py computes it", step: 3050 },
    src: ["Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017", "Goyal et al., Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, 2017", "Dubey et al., The Llama 3 Herd of Models, 2024 (section 3.4.1)"],
    terms: [
      { term: "Learning-rate schedule", def: "A rule that sets the learning rate at each training step." },
      { term: "Warmup", def: "Raising the learning rate from near 0 to its peak over the first steps of training." },
      { term: "Cosine decay", def: "Lowering the learning rate along half a cosine curve, from the peak to a floor, over the rest of training." },
    ],
  });

  Deck.slide({
    id: "nn-clip",
    title: "Gradient clipping caps the length of one step",
    body: String.raw`<p>A rare bad batch or an exploding layer can produce a gradient many times its usual size, and the step it causes can throw the weights far from where training was heading. <b>Gradient clipping</b> by global norm measures all gradients together as one vector, \(\lVert g \rVert = \sqrt{\sum_i g_i^2}\) over every parameter of the model, and if that exceeds a threshold \(c\), multiplies every gradient by \(c/\lVert g \rVert\). The direction is kept; only the length is capped.</p>
<p>Two parameters with gradients 1.5 and 2 have global norm 2.5. With \(c = 1\) both are scaled by 0.4, to 0.6 and 0.8. Clipping each one separately to at most 1 would give 1 and 1, a direction 8.1° off. PyTorch's <code>clip_grad_norm_</code> returns the norm before clipping, which is also the number to log: a rising gradient norm is often the first sign of instability.</p>
<p>GPT-3 clipped the global norm at 1.0. Our model trained without clipping; after step 300 its loss, averaged over 10 steps, never rose more than 0.007 above its lowest value of the previous 100 steps.</p>`,
    math: String.raw`g \leftarrow g \cdot \min\Big(1, \frac{c}{\lVert g \rVert}\Big)`,
    fig: { type: "nn-clip", caption: "A gradient of two parameters, the clipping threshold, and two ways to clip", ga: 1.5, gb: 2, c: 1 },
    src: ["Pascanu, Mikolov & Bengio, On the difficulty of training recurrent neural networks, ICML 2013", "Brown et al., Language Models are Few-Shot Learners (GPT-3), 2020, Appendix B", "PyTorch documentation, torch.nn.utils.clip_grad_norm_"],
    terms: [{ term: "Gradient clipping", def: "Scaling all gradients by \\(c/\\lVert g \\rVert\\) when their global norm \\(\\lVert g \\rVert\\) exceeds a threshold \\(c\\), which caps the size of one step without changing its direction." }],
  });

  Deck.slide({
    id: "nn-recipe",
    title: "What each technique fixes, measured in this chapter",
    layout: "wide",
    body: String.raw`<p>Each technique in this chapter answers one failure that a plain deep network trained by plain gradient descent runs into. The table pairs each with the number that showed the failure on these sheets. Our model uses all of them except clipping and weight decay. GPT-3 used all of them with two differences: LayerNorm where ours has RMSNorm, and initial weights with a fixed standard deviation of 0.02 instead of one set by the fan-in, the weights of residual layers further scaled by \(1/\sqrt{N}\).</p>`,
    fig: {
      type: "table", caption: "The techniques of chapter 3.5, each with the failure it removes", dense: true,
      note: "GPT-3: same initialization and pre-normalization as GPT-2 (weights drawn with standard deviation 0.02, as in GPT-1; residual layers scaled by 1/√N), Adam with β = (0.9, 0.95), global-norm clipping at 1.0, warmup, cosine decay, weight decay 0.1.",
      cols: ["technique", "failure without it", "measured on these sheets"],
      rows: [
        ["random initialization", "units that start equal stay equal", "XOR loss never below 1/6 from equal weights (0.1753 at step 2,000); below 10⁻⁷ from random ones"],
        [String.raw`weight scale \(g/\sqrt{\text{fan-in}}\), \(g = 1\) or \(\sqrt2\) for ReLU`, "the signal is multiplied by g per layer", "after 10 linear layers: 0.0009 at g = 0.5, 955 at g = 2, 0.93 at g = 1"],
        ["RMSNorm before each sublayer", "the scale drifts as the weights change", "position 4 leaving layer 2: RMS 0.64 at step 0, 15.5 at step 6,000; RMS 1 again after the norm, before the gains"],
        ["residual connections", "gradients shrink by a factor per layer", "20 sigmoid layers: 2.3 × 10⁻¹³ without, 1.0 with"],
        ["momentum", "slow along flat directions, zig-zag across steep ones", "26 steps to L < 10⁻³ without, 15 with β = 0.5"],
        ["Adam", "one step size for every parameter", "steps of 0.1 for gradients of −8 and 30 alike"],
        ["AdamW", "decay depends on gradient history", "unused weight after 100 steps: 0.125 with L2 in Adam, 0.905 with AdamW"],
        ["warmup and decay", "unreliable first steps, noisy last ones", "η from 0.0001 to 0.01 by step 99, then down to 0.001 at step 6,000"],
        ["gradient clipping", "one huge step undoes many good ones", "norm 2.5 scaled to 1, direction kept"],
      ],
    },
    src: ["Brown et al., Language Models are Few-Shot Learners (GPT-3), 2020, section 2.1 and Appendix B", "Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2), 2019", "Radford et al., Improving Language Understanding by Generative Pre-Training (GPT-1), 2018: initialization N(0, 0.02)"],
    check: {
      q: "A 30-layer sigmoid network trains, but its first layers' weights barely change while the last layers' do. Which change addresses the cause?",
      options: ["Clip the global gradient norm at 1.0", "Lower the learning rate", "Add residual connections around the layers", "Lengthen the warmup"], answer: 2,
      why: String.raw`The first layers' gradients vanish: each sigmoid layer multiplies the backward signal by a slope of at most 0.25 (2.0 × 10⁻⁶ after 10 layers in this chapter's plate). Residual connections give the gradient an identity path around each layer (2.3 × 10⁻¹³ became 1.0 over 20 layers). Clipping and a lower rate only shrink steps, and warmup only changes the first steps.`,
    },
  });



  // ------------------------------------------------------------ Fig: SGD and RMSProp on the elongated bowl
  Deck.fig("nn-rms-sgd", (el, sp) => {
    let eta = sp.eta ?? 0.07, k = sp.show ?? 10;
    const R0 = 1.3, W0 = 460, H0 = 300, X = (v) => W0 / 2 + (v / R0) * (W0 / 2), Y = (v) => H0 / 2 - (v / R0) * (H0 / 2);
    const svg = s("svg", { viewBox: `0 0 ${W0} ${H0}`, class: "nn-mom", role: "img", "aria-label": "paths of SGD and RMSProp on the bowl ½(25a² + b²)" });
    const info = h("p", { class: "f-mm-expl", "aria-live": "polite" }), tiles = h("div", { class: "f-tiles" });
    const inBox = (p) => Math.abs(p[0]) <= R0 && Math.abs(p[1]) <= R0;
    function draw() {
      clear(svg);
      svg.appendChild(s("rect", { x: 0, y: 0, width: W0, height: H0, class: "nn-frame" }));
      [0.05, 0.25, 0.75, 1.5, 3, 6, 12].forEach((c) => svg.appendChild(s("ellipse", { cx: X(0), cy: Y(0), rx: (Math.sqrt((2 * c) / 25) / R0) * (W0 / 2), ry: (Math.sqrt(2 * c) / R0) * (H0 / 2), class: "nn-contour" })));
      svg.appendChild(s("line", { x1: 0, x2: W0, y1: Y(0), y2: Y(0), class: "nn-gridl" }));
      svg.appendChild(s("line", { x1: X(0), x2: X(0), y1: 0, y2: H0, class: "nn-gridl" }));
      svg.appendChild(s("text", { x: W0 - 6, y: Y(0) - 6, class: "nn-tick", "text-anchor": "end" }, "a (steep, curvature 25)"));
      svg.appendChild(s("text", { x: X(0) + 6, y: 14, class: "nn-tick" }, "b (flat, curvature 1)"));
      const sg = NN3.bowlPath("sgd", eta, k), rp = NN3.bowlPath("rms", eta, k);
      for (const [rows, cls] of [[sg, "s1"], [rp, "s2"]]) {
        const pts = rows.map((r) => r.w).filter(inBox);
        svg.appendChild(s("polyline", { points: pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" "), class: `nn-path ${cls}` }));
        pts.forEach((p, i) => svg.appendChild(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: i === pts.length - 1 ? 4.5 : 2.2, class: `nn-pathdot ${cls}`, "data-tip": `(${fix(p[0], 3)}, ${fix(p[1], 3)}), L = ${fix(NN3.bowl(p), 4)}` })));
      }
      svg.appendChild(s("circle", { cx: X(-1), cy: Y(1), r: 4, class: "nn-start" }));
      svg.appendChild(s("text", { x: X(0), y: Y(0) + 5, class: "nn-min", "text-anchor": "middle" }, "×"));
      const where = (rows) => { const last = rows[rows.length - 1].w; return rows.length <= k ? "diverged" : `(${fix(last[0], 3)}, ${fix(last[1], 3)})`; };
      info.innerHTML = `After ${k} step${k === 1 ? "" : "s"} from (−1, 1): SGD at ${where(sg)}; RMSProp at ${where(rp)}.`;
      const sl = rp[rp.length - 1].s, eff = sl.map((v) => (v > 0 ? eta / Math.sqrt(v) : NaN));
      clear(tiles);
      tiles.append(tile(k ? fix(eff[0], 4) : "–", "RMSProp rate η/√s for a"), tile(k ? fix(eff[1], 4) : "–", "RMSProp rate η/√s for b"),
        tile(NN3.bowlStepsTo("sgd", eta), "SGD: steps to L < 10⁻³"), tile(NN3.bowlStepsTo("rms", eta), "RMSProp: steps to L < 10⁻³"));
    }
    el.append(h("div", { class: "nn-mom-wrap" }, svg),
      h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "SGD, one rate η"), h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "RMSProp, ρ = 0.95")),
      info,
      h("div", { class: "f-ctl" }, button("Step ▸", () => { k++; draw(); }, { kind: "primary" }), button("10 steps", () => { k += 10; draw(); }), button("Reset", () => { k = 0; draw(); })),
      h("div", { class: "f-ctl sliders" }, slider({ label: "Learning rate η", min: 0.01, max: 0.1, step: 0.005, value: eta, format: (v) => fix(v, 3), onInput: (v) => { eta = v; draw(); } })),
      tiles);
    draw();
  });

  // ================================================================== chapter 3.6: a network trained in the page
  // Two classes of 2-D points. Deterministic: each set comes from its own seeded generator.
  NN3.data = {
    spirals(n = 100, noise = 0.02, seed = 3) {
      const rand = TinyLM.mulberry32(seed), X = [], Y = [];
      for (let c = 0; c < 2; c++) for (let i = 0; i < n; i++) {
        const t = i / n, r = 0.08 + 0.92 * t, ang = 1.75 * t * 2 * Math.PI + c * Math.PI;
        X.push(r * Math.sin(ang) + noise * NN3.gauss(rand), r * Math.cos(ang) + noise * NN3.gauss(rand)); Y.push(c);
      }
      return { X: Float64Array.from(X), Y, B: 2 * n };
    },
    moons(n = 15, noise = 0.25, seed = 21) {
      const rand = TinyLM.mulberry32(seed), X = [], Y = [];
      for (let c = 0; c < 2; c++) for (let i = 0; i < n; i++) {
        const th = (Math.PI * (i + 0.5)) / n;
        const x = c ? 1 - Math.cos(th) : Math.cos(th), y = c ? 0.5 - Math.sin(th) : Math.sin(th);
        X.push((x - 0.5) * 0.8 + noise * NN3.gauss(rand), (y - 0.25) * 0.8 + noise * NN3.gauss(rand)); Y.push(c);
      }
      return { X: Float64Array.from(X), Y, B: 2 * n };
    },
  };
  NN3.TRAIN_SEED = 11;
  // One training run, as the plates run it: tanh hidden layers, one logit, logistic loss, full batch, Adam (0.9, 0.95).
  NN3.trainer = (data, width, depth, lr, test = null) => {
    const net = NN3.mlp.create([2, ...Array(depth).fill(width), 1], "tanh", null, NN3.TRAIN_SEED), st = NN3.mlp.adamState(net);
    const run = { net, st, steps: 0, curve: [], testCurve: [] };
    run.step = () => {
      const g = NN3.mlp.grads(net, data.X, data.Y, data.B, "bce");
      run.curve.push([run.steps, g.loss]);
      NN3.mlp.adam(net, st, g, lr);
      run.steps++;
      if (test && run.steps % 10 === 0) run.testCurve.push([run.steps, NN3.mlp.loss(net, test.X, test.Y, test.B, "bce").loss]);
    };
    run.eval = () => NN3.mlp.loss(net, data.X, data.Y, data.B, "bce");
    run.evalTest = () => (test ? NN3.mlp.loss(net, test.X, test.Y, test.B, "bce") : null);
    return run;
  };
  NN3.trainTo = (data, width, depth, lr, steps, test = null) => {
    const run = NN3.trainer(data, width, depth, lr, test);
    while (run.steps < steps) run.step();
    return run;
  };

  // ------------------------------------------------------------ Fig: train an MLP in the page
  Deck.fig("nn-train", (el, sp) => {
    const kind = sp.data || "spirals";
    const data = kind === "spirals" ? NN3.data.spirals() : NN3.data.moons(15, 0.25, 21);
    const test = kind === "moons" ? NN3.data.moons(500, 0.25, 99) : null;
    let width = sp.width ?? 16, depth = sp.depth ?? 2, lr = sp.lr ?? 0.01, run, raf = 0, target = 0;
    const lo = kind === "spirals" ? [-1.15, -1.15] : [-1.7, -1.3], hi = kind === "spirals" ? [1.15, 1.15] : [1.7, 1.3];
    const G = 40, S = 300, Hs = Math.round((S * (hi[1] - lo[1])) / (hi[0] - lo[0]));
    const PX = (v) => ((v - lo[0]) / (hi[0] - lo[0])) * S, PY = (v) => Hs - ((v - lo[1]) / (hi[1] - lo[1])) * Hs;
    const svg = s("svg", { viewBox: `0 0 ${S} ${Hs}`, class: "nn-field", role: "img", "aria-label": "decision boundary of the network and the training points" });
    const cells = [], grid = new Float64Array(G * G * 2);
    for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) {
      const cx = lo[0] + ((a + 0.5) / G) * (hi[0] - lo[0]), cy = lo[1] + ((b + 0.5) / G) * (hi[1] - lo[1]);
      grid[2 * (a * G + b)] = cx; grid[2 * (a * G + b) + 1] = cy;
      const r = s("rect", { x: (a / G) * S, y: Hs - ((b + 1) / G) * Hs, width: S / G + 0.4, height: Hs / G + 0.4, class: "nn-cell" });
      cells.push(r); svg.appendChild(r);
    }
    for (let i = 0; i < data.B; i++) svg.appendChild(s("circle", { cx: PX(data.X[2 * i]), cy: PY(data.X[2 * i + 1]), r: kind === "spirals" ? 2.6 : 4, class: `nn-dpt c${data.Y[i]}`, "data-tip": `class ${data.Y[i]} at (${fix(data.X[2 * i], 2)}, ${fix(data.X[2 * i + 1], 2)})` }));
    svg.appendChild(s("rect", { x: 0, y: 0, width: S, height: Hs, class: "nn-fieldframe" }));
    const chartHost = h("div", { class: "nn-chart nn-train-chart" }), tiles = h("div", { class: "f-tiles" });
    const ch = chart(chartHost, {
      width: 320, height: 250, margin: { l: 66, r: 12, t: 10, b: 36 }, aria: "loss by training step",
      x: { domain: [0, 200], ticks: [0, 100, 200], label: "step" }, y: { type: "log", domain: [1e-4, 1], ticks: [1e-4, 1e-3, 1e-2, 1e-1, 1], fmt: pow10, label: "loss" }, series: [],
    });
    function drawField() {
      const out = NN3.mlp.forward(run.net, grid, G * G).hs[run.net.sizes.length - 1];
      for (let k = 0; k < G * G; k++) {
        const p = 1 / (1 + Math.exp(-out[k])), w = Math.min(1, Math.abs(p - 0.5) * 2);
        cells[k].style.fill = `color-mix(in oklab, var(${p > 0.5 ? "--s2" : "--s1"}) ${(8 + 44 * w).toFixed(1)}%, var(--plate))`;
      }
    }
    function draw() {
      drawField();
      const n = run.steps, xmax = Math.max(200, Math.ceil(n / 200) * 200);
      const tk = xmax <= 1000 ? Array.from({ length: xmax / 200 + 1 }, (_, i) => i * 200) : Array.from({ length: 6 }, (_, i) => Math.round((i * xmax) / 5));
      const ser = [{ points: run.curve.map(([x, y]) => [x, Math.max(y, 1e-300)]), cls: "s3" }];
      if (test) ser.push({ points: run.testCurve.map(([x, y]) => [x, Math.max(y, 1e-300)]), cls: "s4" });
      const vals = run.curve.concat(run.testCurve).map((q) => q[1]).filter((v) => v > 0);
      let lo10 = Math.max(-10, Math.floor(Math.log10(Math.min(1e-4, ...vals)))), hi10 = Math.max(0, Math.ceil(Math.log10(Math.max(1, ...vals))));
      const stp = hi10 - lo10 > 6 ? 2 : 1;
      lo10 = Math.floor(lo10 / stp) * stp;
      const yt = []; for (let e = lo10; e <= hi10; e += stp) yt.push(Math.pow(10, e));
      ch.update({ x: { domain: [0, xmax], ticks: tk, fmt: (v) => int(v), label: "step" }, y: { type: "log", domain: [Math.pow(10, lo10), Math.pow(10, hi10) * (hi10 === 0 ? 1.4 : 1)], ticks: yt, fmt: pow10, label: "loss" }, series: ser });
      const tr = run.eval();
      clear(tiles);
      tiles.append(tile(int(n), "steps"), tile(tr.loss < 1e-3 ? sci(tr.loss, 1) : fix(tr.loss, 3), "training loss"), tile(`${fix(100 * tr.acc, 1)}%`, "training accuracy"));
      if (test) {
        const te = run.evalTest(), best = run.testCurve.reduce((b, q) => (q[1] < b[1] ? q : b), [0, Infinity]);
        tiles.append(tile(fix(te.loss, 3), "test loss (1,000 new points)"), tile(`${fix(100 * te.acc, 1)}%`, "test accuracy"), tile(best[1] < Infinity ? `${fix(best[1], 3)} at ${int(best[0])}` : "–", "lowest test loss, at step"));
      }
    }
    const stop = () => { if (raf) cancelAnimationFrame(raf); raf = 0; };
    function trainTo(t) {
      target = Math.max(target, t);
      if (UI.reduceMotion()) { while (run.steps < target) run.step(); draw(); return; }
      if (raf) return;
      const frame = () => {
        const chunk = Math.max(1, Math.ceil((target - run.steps) / 12));
        for (let i = 0; i < Math.min(chunk, 25) && run.steps < target; i++) run.step();
        draw();
        raf = run.steps < target ? requestAnimationFrame(frame) : 0;
      };
      raf = requestAnimationFrame(frame);
    }
    function reset() { stop(); run = NN3.trainer(data, width, depth, lr, test); target = 0; draw(); }
    const legend = h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "class 0"), h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "class 1"),
      h("span", { class: "key" }, h("span", { class: "sw sw-s3" }), "training loss"), test ? h("span", { class: "key" }, h("span", { class: "sw sw-s4" }), "test loss") : null);
    el.append(
      h("div", { class: "nn-train" }, h("div", { class: "nn-field-wrap" }, svg), chartHost),
      legend,
      h("div", { class: "f-ctl" },
        button("Step ▸", () => { stop(); target = run.steps + 1; run.step(); draw(); }),
        button("Train 200 steps", () => trainTo(Math.max(run.steps, target) + 200), { kind: "primary" }),
        button("Reset", reset)),
      h("div", { class: "f-ctl" },
        segmented({ label: "Width", value: width, options: [2, 4, 8, 16, 32].map((v) => ({ label: String(v), value: v })), onChange: (v) => { width = v; reset(); } }),
        segmented({ label: "Hidden layers", value: depth, options: [1, 2, 3].map((v) => ({ label: String(v), value: v })), onChange: (v) => { depth = v; reset(); } }),
        segmented({ label: "Learning rate", value: lr, options: [0.001, 0.01, 0.1, 1].map((v) => ({ label: String(v), value: v })), onChange: (v) => { lr = v; reset(); } })),
      tiles);
    reset();
    if (sp.steps) { while (run.steps < sp.steps) run.step(); target = run.steps; draw(); }
    return { leave: stop };
  });


  // ------------------------------------------------------------------ 3.6 A network trained in the page
  Deck.chapter({ title: "A network trained in the page" });

  const P36 = (w, d) => 2 * w + w + (d - 1) * (w * w + w) + w + 1; // parameters of 2 → w (× d layers) → 1
  Deck.slide({
    id: "nn-train-demo",
    title: "A two-layer network learns two spirals in this page",
    body: String.raw`<p>The plate trains a multi-layer perceptron on 200 points from two interleaved spirals, 100 per class. Each point's 2 coordinates are the input; two tanh hidden layers of 16 units feed one output logit; the loss is Part 2's logistic loss (sigmoid plus cross-entropy). The forward and backward passes are chapter 3.3's hand-written code extended to any widths, checked against finite differences by the verification script.</p>
<p>Weights start at standard deviation \(1/\sqrt{\text{fan-in}}\) from a fixed seed, biases at 0. Each step is one Adam update (\(\eta = 0.01\), our model's \(\beta_1 = 0.9\), \(\beta_2 = 0.95\)) on all 200 points; the network has 337 parameters.</p>
<p>The plate starts after 200 steps: loss 0.409, 84.5% of the points on the correct side. Press <b>Train 200 steps</b>. After 400 steps the loss is 0.015 and all 200 points are classified correctly; by 600 it is 0.0002. The shading is the network's probability of class 1 at each point of the plane.</p>`,
    fig: { type: "nn-train", caption: "A 2-16-16-1 tanh network learning two spirals. Change the width, the depth or the learning rate and train again.", data: "spirals", width: 16, depth: 2, lr: 0.01, steps: 200 },
  });

  const L36 = [
    ["width 2, 1 layer", P36(2, 1), "0.635", "56.0%"], ["width 4, 1 layer", P36(4, 1), "0.538", "52.0%"], ["width 8, 1 layer", P36(8, 1), "0.377", "77.5%"],
    ["width 16, 1 layer", P36(16, 1), "0.280", "88.0%"], ["width 32, 1 layer", P36(32, 1), "0.054", "100%"],
    ["width 8, 2 layers", P36(8, 2), "0.0034", "100%"], ["width 8, 3 layers", P36(8, 3), "5.9 × 10⁻⁹", "100%"],
    ["width 16, 2 layers, η = 0.001", P36(16, 2), "0.441", "77.5%"], ["width 16, 2 layers, η = 0.01", P36(16, 2), "1.1 × 10⁻⁷", "100%"],
    ["width 16, 2 layers, η = 0.1", P36(16, 2), "8.6 × 10⁻¹⁰", "100%"], ["width 16, 2 layers, η = 1", P36(16, 2), "0.693", "50.0%"],
  ];
  Deck.slide({
    id: "nn-train-knobs",
    title: "Width, depth and learning rate, measured",
    body: String.raw`<p>The table trains the same network on the same 200 points for 1,000 steps, changing one setting at a time; the previous plate reproduces any row.</p>
<p><b>Width.</b> With one hidden layer, 2 units end at 56% accuracy, barely above the 50% of guessing; 32 units reach 100%. <b>Depth.</b> Two layers of 8 units (105 parameters) reach a loss of 0.0034 by step 1,000, one layer of 32 (129 parameters) 0.054. Both classify every point by then, and one layer also approaches zero loss given more steps (width 16: 5.4 × 10⁻⁸ at step 4,000), so here depth bought faster training. The tent of chapter 3.2 is where depth buys functions a shallow network needs far more units for.</p>
<p><b>Learning rate.</b> At 0.001 the network is at 77.5% after 1,000 steps; at 0.1 it fits all 200 points by step 200. At 1, Adam's first steps of about 1 per weight saturate the tanh units: after 10 steps 78% of first-layer outputs are beyond ±0.99, where the slope is nearly 0. From step 300 on every point gets the same logit, which Adam's steps keep swinging between −0.69 and +0.69: the loss cycles between 0.693 and 0.752 and the accuracy stays at 50%.</p>`,
    fig: {
      type: "table", caption: "Two spirals, 1,000 Adam steps from the same starting seed (η = 0.01 unless stated)", dense: true,
      cols: ["network", "parameters", "loss", "accuracy"], align: ["l", "r", "r", "r"],
      rows: L36.map((r) => [r[0], int(r[1]), r[2], r[3]]), highlight: [8],
      note: "At η = 1 the loss cycles between 0.693 and 0.752 from step 300 on; step 1,000 falls near the bottom of the cycle.",
    },
    check: {
      q: "At learning rate 1, from step 300 on, the loss never falls below 0.693. Why that number?",
      options: ["The network gives every point the same output, and on two equal classes a constant output scores at best ln 2 = 0.693", "Half the points are classified correctly with confidence", "The loss is diverging", "0.693 is the lowest loss any network can reach on these points"], answer: 0,
      why: String.raw`From step 300 on all 200 logits are equal, so the network outputs one probability \(p\) for every point. With 100 points per class the loss is \(-(\ln p + \ln(1 - p))/2\), smallest at \(p = 0.5\), where it is \(\ln 2 = 0.693\). Any other shared \(p\) costs more: the swings of the logit to ±0.69 give 0.752.`,
    },
  });

  Deck.slide({
    id: "nn-overfit",
    title: "A large network memorizes 30 noisy points",
    body: String.raw`<p>Part 2 showed overfitting with polynomials; networks do the same. The plate's training set is 30 points, 15 per class, from two interlocking half-moons with noise of standard deviation 0.25. A separate test set of 1,000 points comes from the same distribution.</p>
<p>Two hidden layers of 32 units, 1,185 parameters, fit the 30 points exactly: after 1,000 steps the training loss is \(2 \times 10^{-9}\), and the boundary bends around single noisy points. On the test set the accuracy is 81.9% and the loss 2.59: 150 of the 181 wrong predictions have probability above 0.99, and each costs more than \(\ln 100 = 4.6\). The test loss was lowest, 0.352, at step 30. Four units in one layer (17 parameters) do better, 0.281 at step 370, and then overfit too.</p>
<p>The cures are more data, a smaller model, or stopping early (Part 2). Our model cannot overfit this way: it trained on fresh samples of the same 270 queries it is tested on, and its loss, 0.6942, sits just above the 0.6937 floor that no model can beat.</p>`,
    fig: { type: "nn-train", caption: "A 2-32-32-1 network on 30 noisy points after 1,000 steps; the test curve is measured on 1,000 new points.", data: "moons", width: 32, depth: 2, lr: 0.01, steps: 1000 },
  });

  Deck.slide({
    id: "nn-same-loop",
    title: "The same loop, scaled up, trains every language model",
    example: "tokens ids",
    layout: "wide",
    body: String.raw`<p>Every training run in this set runs the loop of chapter 3.4: draw a batch, run the forward pass, compute the loss, backpropagate, update with Adam, adjust the learning rate. The spirals plate of this chapter is its smallest case, with one fixed batch of all 200 points and a constant learning rate. The table lines up three runs.</p>
<p>What grows is everything around the loop. The data stops being points in a plane and becomes text, which must first be turned into numbers: tokens and embeddings (Part 4). The network becomes a transformer (Part 5). A step of Llama 3 405B on 16 million tokens is about \(4 \times 10^{19}\) floating-point operations (6 × 405 billion × 16 million), run on GPUs (Part 6) and spread over thousands of them whose gradients are summed every step (Part 7).</p>`,
    fig: {
      type: "table", caption: "Three training runs of the same loop", rowHeads: true,
      cols: ["", "this page (spirals)", "our model", "Llama 3 405B"],
      rows: [
        ["parameters", "337", "9,968", "405 billion"],
        ["examples per step", "200 points", "128 queries × 8 positions = 1,024 predictions", "4M tokens at first, 16M later"],
        ["steps", "hundreds", "6,000", "cosine schedule over 1,200,000"],
        ["loss", "logistic, 2 classes", "cross-entropy over 113 tokens", "cross-entropy over a 128K vocabulary"],
        ["optimizer", "Adam, β = (0.9, 0.95)", "AdamW, β = (0.9, 0.95), decay 0", "AdamW"],
        ["peak learning rate", "0.01", "0.01", "8 × 10⁻⁵"],
        ["warmup, then decay", "none", "100 steps, cosine to 0.001", "8,000 steps, cosine to 8 × 10⁻⁷"],
        ["hardware", "this browser tab", "one CPU thread", "up to 16K H100 GPUs"],
      ],
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (sections 3.3 and 3.4.1, Table 3)", "tools/build_model.py (our model's training script)"],
    check: {
      q: "Going from this page's network to Llama 3 405B, which of these stays the same?",
      options: ["The number of examples per step", "The way gradients are computed", "The number of parameters per layer", "The hardware"], answer: 1,
      why: "Every gradient is still the chain rule run backward through the recorded operations: backpropagation, at about twice the cost of the forward pass. The rest grows by orders of magnitude: 200 points to 16 million tokens per step, 337 parameters to 405 billion.",
    },
  });

})();
