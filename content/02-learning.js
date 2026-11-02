/* Part 2 — Learning from data: loss functions, gradient descent, classification,
 * information, generalization. Chapters 2.1–2.3 use one dataset of five points
 * throughout; 2.4–2.5 use the trained model; 2.6 uses noisy samples of sin(πx). */
(function () {
  "use strict";
  const UI = window.UI;
  const { h, s, clear, chip, slider, segmented, button, chart } = UI;
  const M = window.TinyLM;

  // ------------------------------------------------------------ shared helpers
  const F = (v, d) => UI.format(v, "f" + d); // fixed decimals with a true minus sign
  const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, UI.rich(l)));
  const sci = (v, d = 1) => {
    if (v === 0) return "0";
    const e = Math.floor(Math.log10(Math.abs(v)));
    return `${UI.fmt.minus((v / 10 ** e).toFixed(d))} × 10${String(e).split("").map((c) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(c)]).join("")}`;
  };
  const nf = (v, d) => (Math.abs(v) >= 1e5 ? sci(v, 2) : F(v, d));
  const clipRect = (svg, box) => {
    const id = UI.nextId("lclip");
    svg.appendChild(s("clipPath", { id }, s("rect", { x: box.m.l, y: box.m.t - 1, width: box.W - box.m.l - box.m.r, height: box.H - box.m.t - box.m.b + 2 })));
    const g = s("g", { "clip-path": `url(#${id})` });
    svg.appendChild(g);
    return g;
  };
  const inside = (v, [a, b]) => v >= a && v <= b;
  // tick labels for log axes: 10⁻⁴, 0.01, 1, 1,000
  const sup = (e) => String(e).split("").map((c) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(c)]).join("");
  const p10 = (v) => (v >= 0.01 ? v.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "10" + sup(Math.round(Math.log10(v))));
  const pathD = (pts, X, Y) => pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  const PANEL = { width: 300, height: 270, margin: { l: 52, r: 12, t: 10, b: 42 } };
  // single wide charts use a narrower viewBox at phone width so their text stays readable
  const NARROW = !!(window.matchMedia && window.matchMedia("(max-width: 760px)").matches);
  const WIDE = NARROW ? 400 : 560;

  // ------------------------------------------------------------ the five points (chapters 2.1–2.3)
  const DX = [1, 2, 3, 4, 5], DY = [3, 3, 6, 7, 8], N5 = 5;
  const loss5 = (w, b) => DX.reduce((a, x, i) => a + (w * x + b - DY[i]) ** 2, 0) / N5;
  const grad5 = (w, b) => {
    let gw = 0, gb = 0;
    DX.forEach((x, i) => { const e = w * x + b - DY[i]; gw += (2 * e * x) / N5; gb += (2 * e) / N5; });
    return [gw, gb];
  };
  // data panel: the five points, a line y = w x + b, its residuals (and optionally squares), clipped to the plot
  function drawData(svg, X, Y, box, w, b, opt = {}) {
    const g = clipRect(svg, box);
    const [x0, x1] = opt.xd || [0, 6];
    if (opt.best) g.appendChild(s("line", { x1: X(x0), y1: Y(opt.best[0] * x0 + opt.best[1]), x2: X(x1), y2: Y(opt.best[0] * x1 + opt.best[1]), class: "learn-best" }));
    DX.forEach((x, i) => {
      const yh = w * x + b;
      if (opt.squares) {
        const side = Math.abs(Y(yh) - Y(DY[i]));
        const left = X(x) + side > box.W - box.m.r;
        g.appendChild(s("rect", { x: left ? X(x) - side : X(x), y: Math.min(Y(yh), Y(DY[i])), width: side, height: side, class: "learn-sq" }));
      }
      g.appendChild(s("line", { x1: X(x), x2: X(x), y1: Y(DY[i]), y2: Y(yh), class: "learn-resid" }));
    });
    g.appendChild(s("line", { x1: X(x0), y1: Y(w * x0 + b), x2: X(x1), y2: Y(w * x1 + b), class: "learn-fit" }));
    DX.forEach((x, i) => {
      const yh = w * x + b;
      svg.appendChild(s("circle", { cx: X(x), cy: Y(DY[i]), r: 4.5, class: "learn-pt", "data-tip": `point ${i + 1}: (${x}, ${DY[i]})\nprediction ${F(yh, 2)}, error ${F(yh - DY[i], 2)}` }));
    });
  }

  // ------------------------------------------------------------ fig: L(w) for a fixed b, beside the line on the data
  Deck.fig("learn-curve", (el, sp) => {
    el.classList.add("learn-fig");
    let w = sp.w ?? 1, b = sp.b ?? 0;
    const bestW = () => (95 - 15 * b) / 55;
    const two = h("div", { class: "learn-two" });
    const lp = h("div", { class: "learn-panel" }, h("p", { class: "learn-ptitle" }, "The line on the data"));
    const rp = h("div", { class: "learn-panel" }, h("p", { class: "learn-ptitle" }, "Loss as a function of w"));
    two.append(lp, rp);
    el.appendChild(two);
    const out = h("div", { class: "f-tiles" });
    const chL = chart(lp, Object.assign({}, PANEL, {
      aria: "five data points and the line y = w x + b", x: { domain: [0, 6], ticks: [0, 1, 2, 3, 4, 5, 6], label: "x" }, y: { domain: [0, 12], ticks: [0, 2, 4, 6, 8, 10, 12], label: "y" },
      after: (svg, X, Y, box) => drawData(svg, X, Y, box, w, b, { squares: true }),
    }));
    const chR = chart(rp, Object.assign({}, PANEL, {
      aria: "the loss L as a function of w for the chosen b", x: { domain: [0, 3], ticks: [0, 0.5, 1, 1.5, 2, 2.5, 3], label: "w" }, y: { domain: [0, 40], ticks: [0, 10, 20, 30, 40], label: "L(w)" },
      after(svg, X, Y, box) {
        const g = clipRect(svg, box);
        const pts = [];
        for (let i = 0; i <= 240; i++) { const ww = (3 * i) / 240; pts.push([ww, loss5(ww, b)]); }
        g.appendChild(s("path", { d: pathD(pts, X, Y), class: "learn-curve" }));
        const bw = bestW(), bl = loss5(bw, b);
        if (inside(bw, [0, 3])) {
          g.appendChild(s("line", { x1: X(bw), x2: X(bw), y1: Y(0), y2: Y(bl), class: "learn-guide" }));
          svg.appendChild(s("circle", { cx: X(bw), cy: Y(bl), r: 3.5, class: "learn-mindot", "data-tip": `lowest point for b = ${F(b, 1)}: w = ${F(bw, 3)}, L = ${F(bl, 3)}` }));
        }
        const L = loss5(w, b);
        if (L <= 40) svg.appendChild(s("circle", { cx: X(w), cy: Y(L), r: 6, class: "learn-cur", "data-tip": `w = ${F(w, 2)}, L = ${F(L, 3)}` }));
      },
    }));
    function draw() {
      chL.update({}); chR.update({});
      clear(out);
      const bw = bestW();
      out.append(tile(F(loss5(w, b), 3), "L(w, b)"), tile(F(grad5(w, b)[0], 2), "slope dL/dw"), tile(F(bw, 3), "best w for this b"), tile(F(loss5(bw, b), 3), "L at that w"));
    }
    el.appendChild(h("div", { class: "f-ctl sliders" },
      slider({ label: "slope w", min: 0, max: 3, step: 0.01, value: w, format: (v) => F(v, 2), onInput: (v) => { w = v; draw(); } }),
      slider({ label: "intercept b (held fixed)", min: -2, max: 4, step: 0.1, value: b, format: (v) => F(v, 1), onInput: (v) => { b = v; draw(); } })));
    el.appendChild(out);
    draw();
  });

  // ------------------------------------------------------------ fig: the loss surface L(w, b) as exact contours, with the line beside it
  // L(w,b) − L* = dᵀ A d with A = [[Σx²/n, Σx/n], [Σx/n, 1]] = [[11, 3], [3, 1]] and d = (w − 1.4, b − 1.2).
  const A5 = [[11, 3], [3, 1]], OPT5 = [1.4, 1.2], LSTAR5 = 0.32;
  function eig2(A) {
    const a = A[0][0], c = A[0][1], d = A[1][1];
    const mid = (a + d) / 2, rad = Math.sqrt(((a - d) / 2) ** 2 + c * c);
    const l1 = mid + rad, l2 = mid - rad;
    let v = Math.abs(c) > 1e-12 ? [c, l1 - a] : a >= d ? [1, 0] : [0, 1];
    const n = Math.hypot(v[0], v[1]);
    v = [v[0] / n, v[1] / n];
    return { l1, l2, v1: v, v2: [-v[1], v[0]] };
  }
  function drawContours(g, svg, X, Y, { A, center, Lmin, levels, labels, xd, yd }) {
    const E = eig2(A);
    for (const L of levels) {
      const k = L - Lmin;
      if (k <= 0) continue;
      const r1 = Math.sqrt(k / E.l1), r2 = Math.sqrt(k / E.l2), pts = [];
      for (let t = 0; t <= 360; t++) {
        const th = (2 * Math.PI * t) / 360, c1 = r1 * Math.cos(th), c2 = r2 * Math.sin(th);
        pts.push([center[0] + c1 * E.v1[0] + c2 * E.v2[0], center[1] + c1 * E.v1[1] + c2 * E.v2[1]]);
      }
      g.appendChild(s("path", { d: pathD(pts, X, Y) + "Z", class: "learn-contour", "data-tip": `L = ${L}` }));
      if (labels.includes(L)) {
        const p = [center[0] + r1 * E.v1[0], center[1] + r1 * E.v1[1]];
        if (inside(p[0], xd) && inside(p[1], yd)) svg.appendChild(s("text", { x: X(p[0]) + 4, y: Y(p[1]) + 4, class: "learn-clabel" }, String(L)));
      }
    }
    return E;
  }
  Deck.fig("learn-surface", (el, sp) => {
    el.classList.add("learn-fig");
    const mode = sp.mode || "explore";
    const WD = [-0.5, 3.5], BD = [-1.5, 4];
    const start = sp.start || [0, 0];
    let w, b, path, steps, diverged, lr = sp.lr || 0.01;
    const reset = () => { w = start[0]; b = start[1]; path = [[w, b]]; steps = 0; diverged = false; };
    reset();
    const two = h("div", { class: "learn-two" });
    const lp = h("div", { class: "learn-panel" }, h("p", { class: "learn-ptitle" }, mode === "explore" ? "Loss surface L(w, b) · click or drag" : "Loss surface L(w, b) and the path"));
    const rp = h("div", { class: "learn-panel" }, h("p", { class: "learn-ptitle" }, "The line on the data"));
    two.append(lp, rp);
    el.appendChild(two);
    const info = h("p", { class: "f-gd-info", "aria-live": "polite" });
    const out = h("div", { class: "f-tiles" });
    const chL = chart(lp, Object.assign({}, PANEL, {
      aria: "contour map of the mean squared error over slope w and intercept b",
      x: { domain: WD, ticks: [-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], label: "w (slope)", fmt: (v) => (Number.isInteger(v) ? String(v) : "") },
      y: { domain: BD, ticks: [-1, 0, 1, 2, 3, 4], label: "b (intercept)", fmt: (v) => F(v, 0) },
      after(svg, X, Y, box) {
        const g = clipRect(svg, box);
        drawContours(g, svg, X, Y, { A: A5, center: OPT5, Lmin: LSTAR5, levels: [0.5, 1, 2, 4, 8, 16, 32, 64], labels: [1, 4, 16], xd: WD, yd: BD });
        svg.appendChild(s("text", { x: X(OPT5[0]), y: Y(OPT5[1]) + 5, class: "learn-min", "text-anchor": "middle", "data-tip": "minimum: w = 1.4, b = 1.2, L = 0.32" }, "×"));
        if (mode === "descent") {
          const shown = path.length > 3000 ? path.slice(path.length - 3000) : path;
          g.appendChild(s("path", { d: pathD(shown, X, Y), class: "learn-path" }));
          shown.slice(-200).forEach((q) => g.appendChild(s("circle", { cx: X(q[0]), cy: Y(q[1]), r: 2, class: "learn-pathdot" })));
        }
        if (inside(w, WD) && inside(b, BD)) svg.appendChild(s("circle", { cx: X(w), cy: Y(b), r: 6, class: "learn-cur", "data-tip": `w = ${F(w, 3)}, b = ${F(b, 3)}, L = ${F(loss5(w, b), 4)}` }));
      },
    }));
    const chR = chart(rp, Object.assign({}, PANEL, {
      aria: "the five points and the current line", x: { domain: [0, 6], ticks: [0, 1, 2, 3, 4, 5, 6], label: "x" }, y: { domain: [0, 12], ticks: [0, 2, 4, 6, 8, 10, 12], label: "y" },
      after: (svg, X, Y, box) => drawData(svg, X, Y, box, w, b, { best: OPT5 }),
    }));
    let wS, bS;
    function draw() {
      chL.update({}); chR.update({});
      clear(out);
      const L = loss5(w, b), g = grad5(w, b);
      if (mode === "explore") {
        out.append(tile(F(w, 2), "slope w"), tile(F(b, 2), "intercept b"), tile(F(L, 3), "L(w, b)"), tile("0.320", "minimum, at (1.4, 1.2)"));
        info.innerHTML = `At (${F(w, 2)}, ${F(b, 2)}) the loss is <b>${F(L, 3)}</b>. The cross marks the lowest point, L = 0.32 at w = 1.4, b = 1.2. The dashed line in the data panel is that best line.`;
      } else {
        out.append(tile(steps.toLocaleString("en-US"), "steps"), tile(diverged ? "diverged" : nf(L, 4), "L(w, b)"), tile(diverged ? "–" : F(g[0], 3), "\\(\\partial L/\\partial w\\)"), tile(diverged ? "–" : F(g[1], 3), "\\(\\partial L/\\partial b\\)"));
        info.innerHTML = diverged
          ? `Diverged: after ${steps} steps the loss is ${nf(L, 1)}, more than 30 times the starting 33.4. At η = ${F(lr, 3)} each step overshoots the steep direction by more than the last (the limit is 2/23.66 = 0.0845).`
          : `Step ${steps}: (w, b) = (${F(w, 3)}, ${F(b, 3)}), loss <b>${F(L, 4)}</b>. Next: w − η·∂L/∂w = ${F(w, 3)} − ${F(lr, 3)} × ${F(g[0], 3)} = <b>${F(w - lr * g[0], 3)}</b>, and b becomes <b>${F(b - lr * g[1], 3)}</b>.`;
      }
    }
    function one() {
      if (diverged) return false;
      const g = grad5(w, b);
      const nw = w - lr * g[0], nb = b - lr * g[1];
      if (!Number.isFinite(nw) || !Number.isFinite(nb)) { diverged = true; return false; }
      w = nw; b = nb; path.push([w, b]); steps++;
      if (loss5(w, b) > 1000) { diverged = true; return false; } // 30 times the starting loss: the run has left the bowl
      return true;
    }
    if (mode === "explore") {
      const svg = chL.svg;
      svg.classList.add("learn-pick");
      svg.setAttribute("data-noswipe", "");
      let drag = false;
      const pick = (e) => {
        const r = svg.getBoundingClientRect();
        if (!r.width) return;
        const px = ((e.clientX - r.left) / r.width) * PANEL.width, py = ((e.clientY - r.top) / r.height) * PANEL.height;
        const m = PANEL.margin;
        const nw = WD[0] + ((px - m.l) / (PANEL.width - m.l - m.r)) * (WD[1] - WD[0]);
        const nb = BD[0] + ((PANEL.height - m.b - py) / (PANEL.height - m.t - m.b)) * (BD[1] - BD[0]);
        w = Math.round(Math.max(WD[0], Math.min(WD[1], nw)) * 100) / 100;
        b = Math.round(Math.max(BD[0], Math.min(BD[1], nb)) * 100) / 100;
        wS.set(w); bS.set(b);
        draw();
      };
      svg.addEventListener("pointerdown", (e) => { drag = true; svg.setPointerCapture(e.pointerId); pick(e); });
      svg.addEventListener("pointermove", (e) => { if (drag) pick(e); });
      svg.addEventListener("pointerup", () => { drag = false; });
      wS = slider({ label: "slope w", min: WD[0], max: WD[1], step: 0.01, value: w, format: (v) => F(v, 2), onInput: (v) => { w = v; draw(); } });
      bS = slider({ label: "intercept b", min: BD[0], max: BD[1], step: 0.01, value: b, format: (v) => F(v, 2), onInput: (v) => { b = v; draw(); } });
      el.append(h("div", { class: "f-ctl sliders" }, wS, bS), out, info);
    } else {
      const many = (n) => { for (let i = 0; i < n; i++) if (!one()) break; draw(); };
      el.append(out, info, h("div", { class: "f-ctl" },
        button("Step ▸", () => many(1), { kind: "primary" }), button("10 steps", () => many(10)), button("100 steps", () => many(100)), button("Reset", () => { reset(); draw(); }),
        slider({ label: "Learning rate η", min: 0.001, max: 0.09, step: 0.001, value: lr, format: (v) => F(v, 3), onInput: (v) => { lr = v; reset(); draw(); } })));
    }
    draw();
  });

  // ------------------------------------------------------------ fig: one movable point; best slope under MSE and under MAE (model y = w x)
  const slopeMSE = (ys) => DX.reduce((a, x, i) => a + x * ys[i], 0) / 55;
  // Σ|w xᵢ − yᵢ| = Σ xᵢ |w − yᵢ/xᵢ|: minimized by the weighted median of the slopes yᵢ/xᵢ with weights xᵢ.
  // The weights are integers summing to 15 (odd), so the median is unique.
  const slopeMAE = (ys) => {
    const r = DX.map((x, i) => [ys[i] / x, x]).sort((p, q) => p[0] - q[0]);
    let c = 0;
    for (const [sl, wt] of r) { c += wt; if (c > 7.5) return sl; }
    return r[r.length - 1][0];
  };
  Deck.fig("learn-outlier", (el, sp) => {
    el.classList.add("learn-fig");
    let k = sp.point ?? 4;
    let ys = DY.slice();
    const base = { mse: slopeMSE(DY), mae: slopeMAE(DY) };
    const host = h("div");
    el.appendChild(host);
    const out = h("div", { class: "f-tiles" });
    const ch = chart(host, {
      width: WIDE, height: 300, margin: { l: 46, r: 14, t: 12, b: 38 }, aria: "five points, one movable, with the best line under squared and under absolute error",
      x: { domain: [0, 6], ticks: [0, 1, 2, 3, 4, 5, 6], label: "x" }, y: { domain: [-6, 26], ticks: [-5, 0, 5, 10, 15, 20, 25], label: "y" },
      after(svg, X, Y, box) {
        const g = clipRect(svg, box);
        const wm = slopeMSE(ys), wa = slopeMAE(ys);
        g.appendChild(s("line", { x1: X(0), y1: Y(0), x2: X(6), y2: Y(6 * wm), class: "learn-mse-line" }));
        g.appendChild(s("line", { x1: X(0), y1: Y(0), x2: X(6), y2: Y(6 * wa), class: "learn-mae-line" }));
        if (ys[k] !== DY[k]) g.appendChild(s("circle", { cx: X(DX[k]), cy: Y(DY[k]), r: 4.5, class: "learn-pt hollow", "data-tip": `original position (${DX[k]}, ${DY[k]})` }));
        DX.forEach((x, i) => svg.appendChild(s("circle", { cx: X(x), cy: Y(ys[i]), r: i === k ? 6 : 4.5, class: "learn-pt" + (i === k ? " moved" : ""), "data-tip": `point ${i + 1}: (${x}, ${F(ys[i], 1)})` })));
      },
    });
    host.appendChild(h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "best slope under squared error (MSE)"), h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "best slope under absolute error (MAE)")));
    let yS;
    function draw() {
      ch.update({});
      clear(out);
      const wm = slopeMSE(ys), wa = slopeMAE(ys);
      const d = (v, b0) => `${v - b0 >= 0 ? "+" : "−"}${F(Math.abs(v - b0), 3)}`;
      out.append(tile(F(wm, 3), `MSE slope (${d(wm, base.mse)})`), tile(F(wa, 3), `MAE slope (${d(wa, base.mae)})`), tile(`(${DX[k]}, ${F(ys[k], 1)})`, `point ${k + 1}`));
    }
    const pickSeg = segmented({ label: "Move point", value: k, options: DX.map((x, i) => ({ value: i, label: String(i + 1) })), onChange: (v) => { k = v; ys = DY.slice(); yS.set(ys[k]); draw(); } });
    yS = slider({ label: "its y", min: -5, max: 25, step: 0.5, value: ys[k], format: (v) => F(v, 1), onInput: (v) => { ys[k] = v; draw(); } });
    el.append(h("div", { class: "f-ctl sliders" }, pickSeg, yS), out);
    draw();
  });

  // ------------------------------------------------------------ fig: learning rates on L(w) = (w − 3)², 10 steps from w = −1
  Deck.fig("learn-lr", (el, sp) => {
    el.classList.add("learn-fig");
    let eta = sp.lr ?? 0.1;
    const N = 10, XD = [-3, 9];
    const iterate = () => { const ws = [-1]; for (let t = 0; t < N; t++) ws.push(ws[t] - eta * 2 * (ws[t] - 3)); return ws; };
    const two = h("div", { class: "learn-two" });
    const lp = h("div", { class: "learn-panel" }, h("p", { class: "learn-ptitle" }, "10 steps on L(w) = (w − 3)²"));
    const rp = h("div", { class: "learn-panel" }, h("p", { class: "learn-ptitle" }, "Distance to the minimum |w − 3| (log scale)"));
    two.append(lp, rp);
    el.appendChild(two);
    const out = h("div", { class: "f-tiles" });
    const note = h("p", { class: "f-gd-info", "aria-live": "polite" });
    const chL = chart(lp, Object.assign({}, PANEL, {
      aria: "the parabola (w − 3) squared with the path of 10 gradient steps", x: { domain: XD, ticks: [-3, 0, 3, 6, 9], label: "w" }, y: { domain: [0, 37], ticks: [0, 10, 20, 30], label: "L(w)" },
      after(svg, X, Y, box) {
        const g = clipRect(svg, box);
        const pts = [];
        for (let i = 0; i <= 240; i++) { const ww = XD[0] + ((XD[1] - XD[0]) * i) / 240; pts.push([ww, (ww - 3) ** 2]); }
        g.appendChild(s("path", { d: pathD(pts, X, Y), class: "learn-curve" }));
        const ws = iterate();
        g.appendChild(s("path", { d: pathD(ws.map((ww) => [ww, (ww - 3) ** 2]), X, Y), class: "learn-path" }));
        ws.forEach((ww, t) => {
          if (!inside(ww, XD)) return;
          svg.appendChild(s("circle", { cx: X(ww), cy: Y((ww - 3) ** 2), r: t === 0 ? 5 : t === N ? 6 : 3.2, class: t === N ? "learn-cur" : "learn-pathdot", "data-tip": `step ${t}: w = ${F(ww, 4)}, L = ${F((ww - 3) ** 2, 4)}` }));
        });
      },
    }));
    const chR = chart(rp, Object.assign({}, PANEL, {
      aria: "distance to the minimum after each step on a log scale",
      x: { domain: [0, 10], ticks: [0, 2, 4, 6, 8, 10], label: "step" },
      y: { domain: [1e-4, 1e3], type: "log", ticks: [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000], fmt: p10 },
      after(svg, X, Y, box) {
        const ws = iterate();
        const g = clipRect(svg, box);
        const pts = ws.map((ww, t) => [t, Math.max(Math.abs(ww - 3), 1e-4)]);
        g.appendChild(s("path", { d: pathD(pts, X, Y), class: "learn-path" }));
        ws.forEach((ww, t) => {
          const e = Math.abs(ww - 3);
          svg.appendChild(s("circle", { cx: X(t), cy: Y(Math.max(e, 1e-4)), r: 3.2, class: e < 1e-4 ? "learn-pt hollow" : "learn-pathdot", "data-tip": `step ${t}: |w − 3| = ${e === 0 ? "0 (exact)" : e < 1e-4 ? sci(e, 2) : F(e, 4)}` }));
        });
      },
    }));
    let seg;
    function draw() {
      chL.update({}); chR.update({});
      const ws = iterate(), f = 1 - 2 * eta, e10 = Math.abs(ws[N] - 3);
      const off = ws.map((ww, t) => (inside(ww, XD) ? null : t)).filter((t) => t !== null);
      const verdict = f === 0 ? "lands on 3 in one step" : f > 0 ? "converges from one side" : f > -1 ? "overshoots, converges" : f === -1 ? "bounces forever" : "diverges";
      clear(out);
      out.append(tile(F(f, 2), "factor \\(1 - 2\\eta\\)"), tile(e10 === 0 ? "0" : e10 < 1e-3 ? sci(e10, 1) : F(e10, 3), "|w − 3| after 10 steps"), tile(verdict, "behaviour"));
      note.innerHTML = `Each step multiplies the distance to the minimum by 1 − 2η = <b>${F(f, 2)}</b>, starting from 4. After 10 steps: 4 × |${F(f, 2)}|<sup>10</sup> = <b>${e10 === 0 ? "0" : e10 < 1e-3 ? sci(e10, 1) : F(e10, 3)}</b>.` + (off.length ? ` Steps ${off[0]}–${off[off.length - 1]} are off the first chart.` : "");
    }
    const sl = slider({ label: "Learning rate η", min: 0.01, max: 1.2, step: 0.01, value: eta, format: (v) => F(v, 2), onInput: (v) => { eta = v; seg.set(null); draw(); } });
    seg = segmented({ label: "Presets", value: eta, options: [0.02, 0.1, 0.5, 0.9, 1, 1.05].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { eta = v; sl.set(v); draw(); } });
    el.append(h("div", { class: "f-ctl sliders" }, sl, seg), out, note);
    draw();
  });

  // ------------------------------------------------------------ fig: every minibatch gradient at (0, 0) for a batch size
  const perEx = (w, b) => DX.map((x, i) => { const e = w * x + b - DY[i]; return [2 * e * x, 2 * e]; });
  function subsets(n, k) {
    const out = [];
    const rec = (start, acc) => { if (acc.length === k) { out.push(acc.slice()); return; } for (let i = start; i < n; i++) { acc.push(i); rec(i + 1, acc); acc.pop(); } };
    rec(0, []);
    return out;
  }
  Deck.fig("learn-batches", (el, sp) => {
    el.classList.add("learn-fig");
    let B = sp.B ?? 2;
    const g = perEx(0, 0), full = [-38, -10.8];
    const host = h("div");
    el.appendChild(host);
    const out = h("div", { class: "f-tiles" });
    const ests = () => subsets(5, B).map((S) => ({ S, gw: S.reduce((a, i) => a + g[i][0], 0) / B, gb: S.reduce((a, i) => a + g[i][1], 0) / B }));
    const ch = chart(host, {
      width: WIDE, height: 280, margin: { l: 50, r: 16, t: 14, b: 40 }, aria: "gradient estimates from every minibatch, and the full gradient",
      x: { domain: [-85, 0], ticks: [-80, -70, -60, -50, -40, -30, -20, -10, 0], label: "estimate of ∂L/∂w" }, y: { domain: [-18, 0], ticks: [-18, -15, -12, -9, -6, -3, 0], label: "estimate of ∂L/∂b" },
      after(svg, X, Y) {
        ests().forEach((e) => svg.appendChild(s("circle", { cx: X(e.gw), cy: Y(e.gb), r: 5, class: "learn-est", "data-tip": `minibatch {${e.S.map((i) => i + 1).join(", ")}}: (${F(e.gw, 1)}, ${F(e.gb, 1)})` })));
        svg.appendChild(s("circle", { cx: X(full[0]), cy: Y(full[1]), r: 8, class: "learn-full", "data-tip": "full gradient (−38, −10.8)" }));
        svg.appendChild(s("line", { x1: X(full[0]), y1: Y(full[1]) - 9, x2: X(full[0]), y2: Y(-4.6), class: "learn-leader" }));
        svg.appendChild(s("text", { x: X(full[0]), y: Y(-3.6), class: "learn-tag", "text-anchor": "middle" }, "full gradient (−38, −10.8) = average of all minibatches"));
      },
    });
    function draw() {
      ch.update({});
      const E = ests(), mw = E.reduce((a, e) => a + e.gw, 0) / E.length, mb = E.reduce((a, e) => a + e.gb, 0) / E.length;
      const sd = Math.sqrt(E.reduce((a, e) => a + (e.gw - mw) ** 2, 0) / E.length);
      clear(out);
      out.append(tile(String(E.length), `minibatches of ${B}`), tile(`(${F(mw, 1)}, ${F(mb, 1)})`, "their average"), tile(F(sd, 2), "spread of \\(\\partial L/\\partial w\\) (std)"), tile(`${B} of 5`, "examples per step"));
    }
    el.append(h("div", { class: "f-ctl" }, segmented({ label: "Batch size B", value: B, options: [1, 2, 3, 4, 5].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { B = v; draw(); } })), out);
    draw();
  });

  // ------------------------------------------------------------ fig: minibatch noise vs batch size, by sampling
  const NOISE_K = 2000, NOISE_BS = [1, 2, 4, 8, 16, 32, 64, 128];
  const NOISE_AT = { start: [0, 0], min: [1.4, 1.2] };
  // K minibatches of B examples drawn independently (with replacement); returns the K estimates of ∂L/∂w
  function noiseSample(which, B) {
    const gw = perEx(...NOISE_AT[which]).map((q) => q[0]);
    const r = M.mulberry32(1000 + B + (which === "min" ? 500 : 0));
    const est = new Float64Array(NOISE_K);
    for (let k = 0; k < NOISE_K; k++) { let sum = 0; for (let j = 0; j < B; j++) sum += gw[Math.floor(r() * 5)]; est[k] = sum / B; }
    return est;
  }
  const meanSd = (a) => { let m = 0; for (const v of a) m += v; m /= a.length; let q = 0; for (const v of a) q += (v - m) ** 2; return [m, Math.sqrt(q / a.length)]; };
  Deck.fig("learn-noise", (el, sp) => {
    el.classList.add("learn-fig");
    let which = sp.at || "start", B = sp.B ?? 1;
    const cache = new Map();
    const get = (wh, b) => { const key = wh + b; if (!cache.has(key)) { const est = noiseSample(wh, b); cache.set(key, { est, ms: meanSd(est) }); } return cache.get(key); };
    const sigma = (wh) => meanSd(perEx(...NOISE_AT[wh]).map((q) => q[0]));
    const two = h("div", { class: "learn-two" });
    const lp = h("div", { class: "learn-panel" }, h("p", { class: "learn-ptitle" }, "2,000 minibatch estimates of ∂L/∂w"));
    const rp = h("div", { class: "learn-panel" }, h("p", { class: "learn-ptitle" }, "Their spread against batch size"));
    two.append(lp, rp);
    el.appendChild(two);
    const out = h("div", { class: "f-tiles" });
    const dom = () => (which === "start" ? { x: [-85, 5], bins: 45, ticks: [-80, -60, -40, -20, 0], sd: [1, 40], sdt: [1, 3, 10, 30] } : { x: [-4.5, 4.5], bins: 45, ticks: [-4, -2, 0, 2, 4], sd: [0.1, 4], sdt: [0.1, 0.3, 1, 3] });
    const chL = chart(lp, Object.assign({}, PANEL, {
      aria: "histogram of minibatch gradient estimates", x: { domain: [-85, 5], label: "estimate" }, y: { domain: [0, 0.4], ticks: [0, 0.1, 0.2, 0.3, 0.4], label: "fraction", fmt: (v) => F(v, 1) },
      after(svg, X, Y, box) {
        const D = dom(), { est } = get(which, B), [lo, hi] = D.x, bw = (hi - lo) / D.bins, cnt = new Array(D.bins).fill(0);
        for (const v of est) { const i = Math.min(D.bins - 1, Math.max(0, Math.floor((v - lo) / bw))); cnt[i]++; }
        const g = clipRect(svg, box);
        cnt.forEach((c, i) => { if (!c) return; const f = c / est.length; g.appendChild(s("rect", { x: X(lo + i * bw) + 0.5, y: Y(Math.min(f, 0.4)), width: Math.max(1, X(lo + (i + 1) * bw) - X(lo + i * bw) - 1), height: Y(0) - Y(Math.min(f, 0.4)), class: "learn-bar", "data-tip": `${F(lo + i * bw, 2)} to ${F(lo + (i + 1) * bw, 2)}: ${c} of 2,000` })); });
        const tg = sigma(which)[0];
        svg.appendChild(s("line", { x1: X(tg), x2: X(tg), y1: Y(0), y2: Y(0.4), class: "learn-truth" }));
        svg.appendChild(s("text", { x: X(tg) + (tg > (lo + hi) / 2 ? -5 : 5), y: Y(0.37), class: "learn-tag", "text-anchor": tg > (lo + hi) / 2 ? "end" : "start" }, `full gradient ${F(tg, 0)}`));
      },
    }));
    const chR = chart(rp, Object.assign({}, PANEL, {
      aria: "standard deviation of the estimates against batch size, log-log", x: { domain: [1, 128], type: "log", ticks: NOISE_BS, label: "batch size B" }, y: { domain: [1, 40], type: "log", ticks: [1, 3, 10, 30], label: "std of estimates" },
      after(svg, X, Y, box) {
        const sg = sigma(which)[1], g = clipRect(svg, box);
        g.appendChild(s("path", { d: pathD([[1, sg], [128, sg / Math.sqrt(128)]], X, Y), class: "learn-theory" }));
        NOISE_BS.forEach((bb) => { const sd = get(which, bb).ms[1]; svg.appendChild(s("circle", { cx: X(bb), cy: Y(sd), r: bb === B ? 6 : 4, class: bb === B ? "learn-cur" : "learn-pt", "data-tip": `B = ${bb}: measured ${F(sd, 3)}, σ/√B = ${F(sg / Math.sqrt(bb), 3)}` })); });
      },
    }));
    function draw() {
      const D = dom();
      chL.update({ x: { domain: D.x, ticks: D.ticks, label: "estimate of ∂L/∂w" } });
      chR.update({ y: { domain: D.sd, type: "log", ticks: D.sdt, label: "std of estimates" } });
      const [tg, sg] = sigma(which), [m, sd] = get(which, B).ms;
      clear(out);
      out.append(tile(F(tg, 2), "full gradient"), tile(F(m, 2), "mean of 2,000 estimates"), tile(F(sd, 3), "their std (measured)"), tile(F(sg / Math.sqrt(B), 3), `\\(\\sigma/\\sqrt{B} = ${F(sg, 2)}/\\sqrt{${B}}\\)`));
    }
    el.append(h("div", { class: "f-ctl" },
      segmented({ label: "At the point", value: which, options: [{ value: "start", label: "start (0, 0)" }, { value: "min", label: "minimum (1.4, 1.2)" }], onChange: (v) => { which = v; draw(); } }),
      segmented({ label: "Batch size B", value: B, options: NOISE_BS.map((v) => ({ value: v, label: String(v) })), onChange: (v) => { B = v; draw(); } })), out);
    draw();
  });

  // ------------------------------------------------------------ fig: three training curves on the five points
  const CURVE_STEPS = 600;
  function trainRun({ B, eta, seed, cosine }) {
    const r = M.mulberry32(seed);
    let w = 0, b = 0;
    const L = [loss5(w, b)];
    for (let t = 0; t < CURVE_STEPS; t++) {
      const idx = B >= 5 ? [0, 1, 2, 3, 4] : Array.from({ length: B }, () => Math.floor(r() * 5));
      let gw = 0, gb = 0;
      for (const i of idx) { const e = w * DX[i] + b - DY[i]; gw += (2 * e * DX[i]) / idx.length; gb += (2 * e) / idx.length; }
      const lr = cosine ? eta * 0.5 * (1 + Math.cos((Math.PI * t) / CURVE_STEPS)) : eta;
      w -= lr * gw; b -= lr * gb;
      L.push(loss5(w, b));
    }
    return L;
  }
  const RUNS = [
    { key: "gd", label: "gradient descent, all 5 points, η = 0.05", cls: "s1", B: 5, eta: 0.05, seed: 1 },
    { key: "sgd", label: "SGD, B = 1, η = 0.03", cls: "s2", B: 1, eta: 0.03, seed: 1 },
    { key: "cos", label: "SGD, B = 1, η from 0.03 to 0 (cosine)", cls: "s3", B: 1, eta: 0.03, seed: 1, cosine: true },
  ];
  Deck.fig("learn-curves", (el, sp) => {
    el.classList.add("learn-fig");
    let t = sp.t ?? 100;
    const curves = RUNS.map((r) => trainRun(r));
    const host = h("div");
    el.appendChild(host);
    const out = h("div", { class: "f-tiles" });
    const ch = chart(host, {
      width: WIDE, height: 300, margin: { l: 50, r: 16, t: 12, b: 40 }, aria: "loss against step for three training runs, log scale",
      x: { domain: [0, CURVE_STEPS], ticks: [0, 100, 200, 300, 400, 500, 600], label: "step" },
      y: { domain: [0.25, 40], type: "log", ticks: [0.3, 1, 3, 10, 30], label: "loss L (log scale)" },
      hlines: [{ y: 0.32 }],
      series: RUNS.map((r, i) => ({ points: curves[i].map((v, k) => [k, v]), cls: r.cls + (r.key === "gd" ? "" : " thin") })).reverse(),
    });
    host.appendChild(h("div", { class: "legend" }, RUNS.map((r) => h("span", { class: "key" }, h("span", { class: `sw sw-${r.cls}` }), r.label)), h("span", { class: "key" }, h("span", { class: "learn-dash" }), "the minimum, 0.32")));
    function draw() {
      ch.update({ vlines: [{ x: t, label: `step ${t}`, cls: "cursor" }] });
      clear(out);
      RUNS.forEach((r, i) => out.appendChild(tile(F(curves[i][t], 3), r.key === "gd" ? "gradient descent" : r.key === "sgd" ? "SGD, constant \\(\\eta\\)" : "SGD, cosine \\(\\eta\\)")));
    }
    el.append(h("div", { class: "f-ctl sliders" }, slider({ label: "read the curves at step", min: 0, max: CURVE_STEPS, step: 1, value: t, format: (v) => String(v), onInput: (v) => { t = v; draw(); } })), out);
    draw();
  });

  // the 8 positions of a query by role: what the model has just read, and what it must predict
  const ROLE_IN = ["<code>SELECT</code>", "<i>column</i>", "<code>FROM</code>", "<i>table</i>", "<code>WHERE</code>", "<i>column</i>", "<code>=</code>", "<i>digit</i>"];
  const ROLE_NEXT = ["<i>column</i>", "<code>FROM</code>", "<i>table</i>", "<code>WHERE</code>", "<i>column</i>", "<code>=</code>", "<i>digit</i>", "<code>;</code>"];
  const ROLE_NEXT_TEXT = ["column", "FROM", "table", "WHERE", "column", "=", "digit", ";"];

  // ------------------------------------------------------------ fig: the trained model's logits after "… WHERE" and their softmax
  const COLS = [" id", " email", " age", " name"];
  Deck.fig("learn-logits", (el, sp) => {
    el.classList.add("learn-fig");
    const tr = M.forward(M.runningIds.slice(0, M.config.T));
    const z0 = tr.logits[4], ids = COLS.map((t) => M.idOf(t));
    const otherMax = Math.max(...z0.filter((_, i) => !ids.includes(i)));
    let over = "all", zName = z0[ids[3]];
    const list = h("div", { class: "learn-lrows", role: "table", "aria-label": "logits and probabilities of the four column tokens" });
    const out = h("div", { class: "f-tiles" });
    const probs = () => {
      const z = z0.slice(); z[ids[3]] = zName;
      if (over === "all") return M.softmax(z);
      const p4 = M.softmax(ids.map((i) => z[i])), p = new Array(z.length).fill(0);
      ids.forEach((i, k) => { p[i] = p4[k]; });
      return p;
    };
    function draw() {
      const p = probs();
      clear(list);
      list.appendChild(h("div", { class: "learn-lrow head" }, h("span", null, "token"), h("span", { class: "r" }, "logit z"), h("span", null, "probability p"), h("span", { class: "r" }, "p")));
      ids.forEach((id, k) => {
        const z = k === 3 ? zName : z0[id];
        list.appendChild(h("div", { class: "learn-lrow" + (k === 0 ? " target" : "") },
          h("span", null, chip(COLS[k], { id })), h("span", { class: "r num" }, F(z, 3)),
          h("span", { class: "learn-ltrack" }, h("span", { class: "learn-lfill", style: { width: `${(p[id] * 100).toFixed(2)}%` } })),
          h("span", { class: "r num", "data-tip": `p = ${p[id].toPrecision(6)}` }, F(p[id], 4))));
      });
      const rest = 1 - ids.reduce((a, i) => a + p[i], 0);
      list.appendChild(h("div", { class: "learn-lrow dim" }, h("span", null, "other 109 tokens"), h("span", { class: "r num" }, over === "all" ? `≤ ${F(otherMax, 2)}` : "–"),
        h("span", { class: "learn-ltrack" }, h("span", { class: "learn-lfill", style: { width: `${(Math.max(rest, 0) * 100).toFixed(2)}%` } })), h("span", { class: "r num" }, F(Math.max(rest, 0), 4))));
      clear(out);
      const three = p[ids[0]] + p[ids[1]] + p[ids[2]];
      out.append(tile(F(three, 4), "<code>id</code> + <code>email</code> + <code>age</code>"), tile(F(p[ids[3]], 4), "<code>name</code>"), tile(F(p[ids[2]] / p[ids[0]], 3), "p(<code>age</code>) / p(<code>id</code>)"));
    }
    let zS;
    el.append(list, h("div", { class: "f-ctl sliders" },
      segmented({ label: "Softmax over", value: over, options: [{ value: "all", label: "all 113 tokens" }, { value: "four", label: "these 4 only" }], onChange: (v) => { over = v; draw(); } }),
      (zS = slider({ label: "logit of name (try raising it)", min: -2, max: 10, step: 0.01, value: zName, format: (v) => F(v, 2), onInput: (v) => { zName = v; draw(); } })),
      button("Model's value", () => { zName = z0[ids[3]]; zS.set(zName); draw(); })), out);
    draw();
  });

  // ------------------------------------------------------------ fig: perplexity per position over training, against the grammar's floor
  Deck.fig("learn-ppl", (el, sp) => {
    el.classList.add("learn-fig");
    const rows = M.data.exactCurve, floor = M.data.floor, next = ROLE_NEXT_TEXT;
    let k = sp.index ?? rows.length - 1;
    const W = WIDE, H = 290, m = { l: 52, r: 14, t: 16, b: 46 }, cw = (W - m.l - m.r) / 8;
    const Y = (v) => H - m.b - (Math.log2(Math.max(v, 1)) / 7) * (H - m.t - m.b);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart learn-ppl-svg", role: "img", "aria-label": "perplexity at each of the 8 positions against the grammar's floor" });
    el.appendChild(svg);
    const out = h("div", { class: "f-tiles" });
    function draw() {
      clear(svg);
      const [step, mean, per] = rows[k];
      [1, 2, 4, 8, 16, 32, 64, 128].forEach((v) => {
        svg.appendChild(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }));
        svg.appendChild(s("text", { x: m.l - 8, y: Y(v) + 4, class: "tick", "text-anchor": "end" }, String(v)));
      });
      svg.appendChild(s("line", { x1: m.l, x2: W - m.r, y1: Y(1), y2: Y(1), class: "axis" }));
      svg.appendChild(s("text", { x: 14, y: (m.t + H - m.b) / 2, class: "ax-label", "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }, "perplexity (log scale)"));
      per.forEach((L, i) => {
        const x0 = m.l + i * cw, ppl = Math.exp(L), fl = Math.exp(floor[i]);
        svg.appendChild(s("rect", { x: x0 + cw * 0.2, y: Y(ppl), width: cw * 0.6, height: Math.max(0.5, Y(1) - Y(ppl)), class: "learn-pbar", "data-tip": `position ${i}, predicting the ${next[i]}, averaged over all 270 queries:\nloss ${F(L, 4)} nats, perplexity ${F(ppl, 3)}\nfloor ${F(floor[i], 4)} nats, perplexity ${F(fl, 3)}` }));
        svg.appendChild(s("line", { x1: x0 + cw * 0.08, x2: x0 + cw * 0.92, y1: Y(fl), y2: Y(fl), class: "learn-ftick" }));
        svg.appendChild(s("text", { x: x0 + cw / 2, y: Y(Math.max(ppl, fl)) - 6, class: "learn-pval", "text-anchor": "middle" }, ppl >= 9.995 ? F(ppl, 1) : F(ppl, 2)));
        svg.appendChild(s("text", { x: x0 + cw / 2, y: H - m.b + 18, class: "learn-ptok", "text-anchor": "middle" }, next[i]));
      });
      svg.appendChild(s("text", { x: (m.l + W - m.r) / 2, y: H - 6, class: "ax-label", "text-anchor": "middle" }, "what is predicted (averages over all 270 queries)"));
      clear(out);
      out.append(tile(step.toLocaleString("en-US"), "training step"), tile(F(mean, 4), "mean loss (nats)"), tile(F(Math.exp(mean), 3), "perplexity \\(e^{\\text{loss}}\\)"), tile(F(Math.exp(floor.reduce((a, b) => a + b, 0) / 8), 3), "floor's perplexity"));
    }
    el.append(h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "model, exact over all 270 queries"), h("span", { class: "key" }, h("span", { class: "learn-dash red" }), "grammar's floor")),
      h("div", { class: "f-ctl sliders" }, slider({ label: "training step", min: 0, max: rows.length - 1, step: 1, value: k, format: (v) => rows[v][0].toLocaleString("en-US"), onInput: (v) => { k = v; draw(); } })), out);
    draw();
  });

  // ------------------------------------------------------------ 2.6 data: 10 training and 101 held-out points of sin(πx) + noise (std 0.2), seed 4
  const POLY = (() => {
    const r = M.mulberry32(4);
    const gauss = () => { let u = 0; while (u === 0) u = r(); const v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const f = (x) => Math.sin(Math.PI * x);
    const tx = Array.from({ length: 10 }, (_, i) => -1 + (2 * i) / 9), ty = tx.map((x) => f(x) + 0.2 * gauss());
    const hx = Array.from({ length: 101 }, (_, i) => -1 + i / 50), hy = hx.map((x) => f(x) + 0.2 * gauss());
    return { f, tx, ty, hx, hy };
  })();
  // least squares by Householder QR: minimizes ‖A c − y‖
  function lstsq(A, y) {
    const m = A.length, k = A[0].length, R = A.map((row) => row.slice()), b = y.slice();
    for (let j = 0; j < k; j++) {
      let nrm = 0;
      for (let i = j; i < m; i++) nrm += R[i][j] ** 2;
      nrm = Math.sqrt(nrm);
      if (!nrm) continue;
      const alpha = R[j][j] > 0 ? -nrm : nrm, v = new Array(m).fill(0);
      for (let i = j; i < m; i++) v[i] = R[i][j];
      v[j] -= alpha;
      const vv = v.reduce((a, c) => a + c * c, 0);
      if (!vv) continue;
      for (let c = j; c < k; c++) { let d = 0; for (let i = j; i < m; i++) d += v[i] * R[i][c]; const f = (2 * d) / vv; for (let i = j; i < m; i++) R[i][c] -= f * v[i]; }
      let d = 0;
      for (let i = j; i < m; i++) d += v[i] * b[i];
      const f = (2 * d) / vv;
      for (let i = j; i < m; i++) b[i] -= f * v[i];
    }
    const c = new Array(k).fill(0);
    for (let j = k - 1; j >= 0; j--) { let sum = b[j]; for (let q = j + 1; q < k; q++) sum -= R[j][q] * c[q]; c[j] = sum / R[j][j]; }
    return c;
  }
  const vander = (xs, d) => xs.map((x) => Array.from({ length: d + 1 }, (_, p) => x ** p));
  const peval = (c, x) => { let v = 0; for (let p = c.length - 1; p >= 0; p--) v = v * x + c[p]; return v; };
  const mseOf = (pred, xs, ys) => xs.reduce((a, x, i) => a + (pred(x) - ys[i]) ** 2, 0) / xs.length;
  const fitDeg = (d) => lstsq(vander(POLY.tx, d), POLY.ty);
  // L2-regularized fit of degree 9: minimizes (1/n)‖X c − y‖² + λ‖c‖² (all 10 coefficients)
  const fitRidge = (lam) => lstsq(vander(POLY.tx, 9).concat(Array.from({ length: 10 }, (_, p) => Array.from({ length: 10 }, (_, q) => (p === q ? Math.sqrt(10 * lam) : 0)))), POLY.ty.concat(new Array(10).fill(0)));
  // gradient descent on the degree-9 MSE from c = 0 with step η: closed form through the eigenvectors of the Hessian H = (2/n) XᵀX,
  // c_t = Σ_k (1 − (1 − η h_k)^t) (u_kᵀ c*) u_k, with c* the exact least-squares fit.
  const ES_ETA = 0.3;
  let esCache = null;
  function earlyStop() {
    if (esCache) return esCache;
    const X = vander(POLY.tx, 9), n = 10;
    const A = Array.from({ length: 10 }, (_, i) => Array.from({ length: 10 }, (_, j) => (2 / n) * X.reduce((a, row) => a + row[i] * row[j], 0)));
    const V = Array.from({ length: 10 }, (_, i) => Array.from({ length: 10 }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 100; sweep++) { // cyclic Jacobi
      let off = 0;
      for (let i = 0; i < 10; i++) for (let j = i + 1; j < 10; j++) off += A[i][j] ** 2;
      if (off < 1e-32) break;
      for (let p = 0; p < 10; p++) for (let q = p + 1; q < 10; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < 10; k++) { const a1 = A[k][p], a2 = A[k][q]; A[k][p] = c * a1 - sn * a2; A[k][q] = sn * a1 + c * a2; }
        for (let k = 0; k < 10; k++) { const a1 = A[p][k], a2 = A[q][k]; A[p][k] = c * a1 - sn * a2; A[q][k] = sn * a1 + c * a2; }
        for (let k = 0; k < 10; k++) { const v1 = V[k][p], v2 = V[k][q]; V[k][p] = c * v1 - sn * v2; V[k][q] = sn * v1 + c * v2; }
      }
    }
    const hs = A.map((row, i) => row[i]), cs = fitDeg(9), proj = hs.map((_, k) => V.reduce((a, row, i) => a + row[k] * cs[i], 0));
    const at = (t) => { const c = new Array(10).fill(0); hs.forEach((hk, k) => { const g = (1 - Math.pow(1 - ES_ETA * hk, t)) * proj[k]; for (let i = 0; i < 10; i++) c[i] += V[i][k] * g; }); return c; };
    esCache = { hs, at };
    return esCache;
  }
  const lamOf = (v) => (v <= -10 ? 0 : 10 ** v);
  const lamFmt = (v) => (v <= -10 ? "0" : 10 ** v >= 0.1 ? F(10 ** v, 2) : sci(10 ** v, 1));
  const tOf = (v) => Math.round(10 ** v);
  Deck.fig("learn-poly", (el, sp) => {
    el.classList.add("learn-fig");
    const mode = sp.mode;
    let model = sp.model || "true", deg = sp.degree ?? 9, logt = sp.logt ?? 0, loglam = sp.loglam ?? -10;
    const current = () => {
      if (mode === "split") {
        if (model === "true") return { pred: POLY.f };
        if (model === "nn") return { pred: (x) => { let bi = 0; POLY.tx.forEach((t, i) => { if (Math.abs(t - x) < Math.abs(POLY.tx[bi] - x)) bi = i; }); return POLY.ty[bi]; } };
        const c = fitDeg(1); return { pred: (x) => peval(c, x), c };
      }
      const c = mode === "degree" ? fitDeg(deg) : mode === "early" ? earlyStop().at(tOf(logt)) : fitRidge(lamOf(loglam));
      return { pred: (x) => peval(c, x), c };
    };
    const host = h("div", { class: mode === "split" ? "" : "learn-two" });
    el.appendChild(host);
    const lp = h("div", { class: "learn-panel" }), rp = h("div", { class: "learn-panel" });
    host.appendChild(lp);
    if (mode !== "split") { host.appendChild(rp); lp.prepend(h("p", { class: "learn-ptitle" }, "The fit")); }
    const out = h("div", { class: "f-tiles" });
    const sizes = mode === "split" ? { width: WIDE, height: 300, margin: { l: 50, r: 14, t: 12, b: 40 } } : PANEL;
    const chL = chart(lp, Object.assign({}, sizes, {
      aria: "training points, held-out points and the model's curve", x: { domain: [-1, 1], ticks: [-1, -0.5, 0, 0.5, 1], label: "x" }, y: { domain: [-3, 3], ticks: [-3, -2, -1, 0, 1, 2, 3], label: "y" },
      after(svg, X, Y, box) {
        const g = clipRect(svg, box), cur = current(), pts = [], tru = [];
        for (let i = 0; i <= 400; i++) { const x = -1 + i / 200; pts.push([x, cur.pred(x)]); tru.push([x, POLY.f(x)]); }
        POLY.hx.forEach((x, i) => g.appendChild(s("circle", { cx: X(x), cy: Y(POLY.hy[i]), r: 2.3, class: "learn-pt held" })));
        g.appendChild(s("path", { d: pathD(tru, X, Y), class: "learn-truth-curve" }));
        g.appendChild(s("path", { d: pathD(pts, X, Y), class: "learn-curve" }));
        POLY.tx.forEach((x, i) => svg.appendChild(s("circle", { cx: X(x), cy: Y(POLY.ty[i]), r: 4.5, class: "learn-pt", "data-tip": `training point (${F(x, 3)}, ${F(POLY.ty[i], 3)})` })));
      },
    }));
    lp.appendChild(h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "learn-dot" }), "10 training points"), h("span", { class: "key" }, h("span", { class: "learn-dot hollow" }), "101 held-out points"), h("span", { class: "key" }, h("span", { class: "learn-dash" }), "sin(πx)")));
    // the loss against the knob
    let chR = null, series = null;
    if (mode !== "split") {
      const knob = mode === "degree" ? Array.from({ length: 10 }, (_, d) => d) : mode === "early" ? Array.from({ length: 181 }, (_, k) => k / 20) : Array.from({ length: 101 }, (_, k) => -10 + k / 10);
      const cOf = (v) => (mode === "degree" ? fitDeg(v) : mode === "early" ? earlyStop().at(tOf(v)) : fitRidge(lamOf(v)));
      series = knob.map((v) => { const c = cOf(v); return [v, mseOf((x) => peval(c, x), POLY.tx, POLY.ty), mseOf((x) => peval(c, x), POLY.hx, POLY.hy)]; });
      const xd = mode === "degree" ? [0, 9] : mode === "early" ? [0, 9] : [-10, 0];
      const xt = mode === "degree" ? [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] : mode === "early" ? [0, 3, 6, 9] : [-10, -8, -6, -4, -2, 0];
      const xf = mode === "degree" ? (v) => String(v) : mode === "early" ? (v) => (v === 0 ? "1" : "10" + sup(v)) : (v) => (v === -10 ? "0" : v === 0 ? "1" : "10" + sup(v));
      const xl = mode === "degree" ? "degree" : mode === "early" ? "gradient steps (log scale)" : "λ (log scale; left end is 0)";
      rp.appendChild(h("p", { class: "learn-ptitle" }, "Training and held-out MSE"));
      chR = chart(rp, Object.assign({}, PANEL, {
        aria: "training and held-out loss against the setting", x: { domain: xd, ticks: xt, label: xl, fmt: xf }, y: { domain: [0, 0.6], ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], label: "MSE", fmt: (v) => F(v, 1) },
        series: [{ points: series.map((q) => [q[0], q[1]]), cls: "s1" }, { points: series.map((q) => [q[0], q[2]]), cls: "s2" }],
        hlines: [{ y: 0.0331 }],
      }));
      rp.appendChild(h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "training"), h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "held-out"), h("span", { class: "key" }, h("span", { class: "learn-dash" }), "sin(πx) on held-out, 0.033")));
    }
    function draw() {
      chL.update({});
      const cur = current(), tr = mseOf(cur.pred, POLY.tx, POLY.ty), ho = mseOf(cur.pred, POLY.hx, POLY.hy);
      clear(out);
      if (mode === "split") {
        out.append(tile(F(tr, 3), "training MSE"), tile(F(ho, 3), "held-out MSE"), tile(F(ho - tr, 3), "gap (held-out − training)"));
        return;
      }
      const knobV = mode === "degree" ? deg : mode === "early" ? logt : loglam;
      chR.update({ vlines: [{ x: knobV, cls: "cursor" }], dots: [{ x: knobV, y: Math.min(tr, 0.6), r: 4, cls: "s1" }, { x: knobV, y: Math.min(ho, 0.6), r: 4, cls: "s2" }] });
      const nrm = Math.hypot(...cur.c);
      if (mode === "degree") out.append(tile(String(deg + 1), "parameters"), tile(F(tr, 3), "training MSE"), tile(F(ho, 3), "held-out MSE"), tile(F(Math.max(...cur.c.map(Math.abs)), 1), "largest |coefficient|"));
      else if (mode === "early") out.append(tile(tOf(logt).toLocaleString("en-US"), "steps"), tile(F(tr, 3), "training MSE"), tile(F(ho, 3), "held-out MSE"), tile(F(nrm, 1), "\\(\\lVert \\mathbf c \\rVert\\)"));
      else out.append(tile(lamFmt(loglam), "\\(\\lambda\\)"), tile(F(nrm, 1), "\\(\\lVert \\mathbf c \\rVert\\)"), tile(F(tr, 3), "training MSE"), tile(F(ho, 3), "held-out MSE"));
    }
    if (mode === "split") el.append(h("div", { class: "f-ctl" }, segmented({ label: "Model", value: model, options: [{ value: "true", label: "sin(πx) itself" }, { value: "nn", label: "memorizer: nearest training point" }, { value: "line", label: "straight line" }], onChange: (v) => { model = v; draw(); } })), out);
    else if (mode === "degree") el.append(h("div", { class: "f-ctl sliders" }, slider({ label: "polynomial degree", min: 0, max: 9, step: 1, value: deg, format: (v) => String(v), onInput: (v) => { deg = v; draw(); } })), out);
    else if (mode === "early") el.append(h("div", { class: "f-ctl sliders" }, slider({ label: "gradient steps (log scale), learning rate 0.3", min: 0, max: 9, step: 0.05, value: logt, format: (v) => tOf(v).toLocaleString("en-US"), onInput: (v) => { logt = v; draw(); } })), out);
    else el.append(h("div", { class: "f-ctl sliders" }, slider({ label: "regularization strength (log scale)", min: -10, max: 0, step: 0.1, value: loglam, format: lamFmt, onInput: (v) => { loglam = v; draw(); } })), out);
    draw();
  });
  // training and held-out MSE for degrees 0–9 (the underfitting sheet's chart)
  const DEGREE_CURVE = Array.from({ length: 10 }, (_, d) => { const c = fitDeg(d); return [d, mseOf((x) => peval(c, x), POLY.tx, POLY.ty), mseOf((x) => peval(c, x), POLY.hx, POLY.hy)]; });

  // ============================================================ Part 2
  Deck.part({ n: 2, title: "Learning from data", short: "Learning" });

  // ------------------------------------------------------------ 2.1
  Deck.chapter({ title: "Models and parameters" });

  Deck.slide({
    id: "learn-part",
    layout: "part",
    title: "Learning from data",
    body: String.raw`<p>A model is a function with adjustable numbers, and training chooses those numbers so that its outputs match data. This part makes each word of that sentence precise: what “match” means (a loss function), how the numbers are adjusted (gradient descent), how the same machinery predicts categories (softmax and cross-entropy), what the best possible loss is (entropy), and how to tell learning from memorizing (held-out data).</p>
<p>Chapters 2.1 to 2.3 use one dataset of five points throughout, small enough to check every number by hand. Chapters 2.4 and 2.5 switch to the trained model and its real predictions after <code>WHERE</code>.</p>`,
    take: "Write a loss for a prediction task, take its gradient, choose a learning rate that converges, and explain why the best possible loss on our grammar is 0.6937 nats per token, not zero.",
  });

  Deck.slide({
    id: "learn-model",
    title: "A model is a function with adjustable numbers",
    example: "tokens ids",
    body: String.raw`<p>A <b>model</b> is a function from an input to a <b>prediction</b>, written so that some of the numbers inside it can be changed. Those numbers are its <b>parameters</b>. The line \(\hat y = w x + b\) is a model with two parameters, the slope \(w\) and the intercept \(b\). With \(w = 2,\ b = 0\) it predicts \(\hat y = 6\) at \(x = 3\); with \(w = 1.4,\ b = 1.2\) it predicts 5.4.</p>
<p>The rule stays fixed and only the parameters change. The hat marks a prediction: \(\hat y\) is what the model says, \(y\) is the true value from the data.</p>
<p>Our transformer is the same thing at a larger size: a fixed computation (Part 5) with 9,968 parameters, which maps 8 token ids to 8 probability distributions over 113 tokens. Training it means choosing those 9,968 numbers.</p>`,
    fig: {
      type: "table", caption: "Models at four sizes: the structure is the same",
      cols: ["Model", "Input", "Output", "Parameters"], align: ["l", "l", "l", "r"],
      rows: [
        [String.raw`\(\hat y = w x + b\)`, "one number \\(x\\)", "one number \\(\\hat y\\)", "2"],
        ["polynomial of degree 9 (chapter 2.6)", "one number", "one number", "10"],
        ["our transformer", "8 token ids", "8 × 113 probabilities", "9,968"],
        ["GPT-3", "up to 2,048 token ids", "a distribution over its vocabulary at each position", "175.0 billion"],
      ],
      highlight: [2],
      note: "GPT-3's size and context length are from Table 2.1 and Section 2.1 of its paper.",
    },
    src: ["Brown et al., Language Models are Few-Shot Learners, 2020 (Table 2.1, Section 2.1)"],
    terms: [
      { term: "Model", def: "A function from an input to a prediction with adjustable numbers inside it, its parameters." },
      { term: "Parameter", def: "A number inside a model that training adjusts: \\(w\\) and \\(b\\) of a line, or the 9,968 weights of our transformer." },
      { term: "Prediction", def: "The model's output for one input, written \\(\\hat y\\); the true value from the data is \\(y\\)." },
    ],
  });

  Deck.slide({
    id: "learn-data",
    title: "Five points and a line",
    body: String.raw`<p>Chapters 2.1 to 2.3 use one <b>dataset</b>, a list of <b>examples</b>, each an input with its known answer:</p>
<table><thead><tr><th>\(x\)</th><th class="r">1</th><th class="r">2</th><th class="r">3</th><th class="r">4</th><th class="r">5</th></tr></thead><tbody><tr><th>\(y\)</th><td class="r">3</td><td class="r">3</td><td class="r">6</td><td class="r">7</td><td class="r">8</td></tr></tbody></table>
<p>The model \(\hat y = w x + b\) makes one prediction per example, \(\hat y_i = w x_i + b\): the height of the line above \(x_i\). Move the sliders and the line pivots and shifts.</p>
<p>No line passes through all five points. The best one, \(\hat y = 1.4x + 1.2\) (derived at the end of chapter 2.2), still misses every point, by 0.2 to 1.0. Learning means finding that \(w\) and \(b\) from the data alone.</p>`,
    fig: {
      type: "plot", width: WIDE, height: NARROW ? 300 : 320, caption: "The five points and the line \\(\\hat y = w x + b\\)",
      x: [0, 6], y: [0, 12], xlabel: "x", ylabel: "y",
      params: [{ name: "w", label: "slope w", min: 0, max: 3, step: 0.01, value: 2 }, { name: "b", label: "intercept b", min: -2, max: 4, step: 0.01, value: 0 }],
      fns: [{ f: (x, p) => p.w * x + p.b }],
      points: DX.map((x, i) => [x, DY[i]]),
      readouts: [
        { label: "predictions \\(\\hat y_1 \\dots \\hat y_5\\)", f: (p) => DX.map((x) => p.w * x + p.b), fmt: (v) => v.map((q) => F(q, 2)).join(", ") },
      ],
    },
    terms: [
      { term: "Dataset", def: "A list of examples used to fit a model." },
      { term: "Example", def: "One input together with its known answer, such as the point \\((4, 7)\\) or one query's next token." },
    ],
  });

  Deck.slide({
    id: "learn-hyper",
    title: "Inputs, parameters and hyperparameters are set differently",
    body: String.raw`<p>Three kinds of numbers appear in every training setup, and each is set in a different way.</p>
<ul><li><b>Inputs</b> come from the data and change with every example: \(x\) for the line, the token ids for our model.</li>
<li><b>Parameters</b> are learned: \(w\) and \(b\), or our model's 9,968 weights. Training changes them.</li>
<li><b>Hyperparameters</b> are chosen by the engineer before training and are not changed by gradient descent: the learning rate, the batch size, the number of steps, the size of the model.</li></ul>
<p>Choosing hyperparameters is a search of its own, done by training several times and comparing (chapter 2.6). Our model used a learning rate of 0.01, reached after 100 warmup steps and then lowered along a cosine curve to 0.001 (Part 3), with batches of 128 queries and 6,000 steps. GPT-3 used batches of 3.2 million tokens and a learning rate of 0.6 × 10⁻⁴.</p>`,
    fig: {
      type: "table", caption: "The three kinds of numbers in three training setups",
      cols: ["", "The line", "Our model", "GPT-3"],
      rows: [
        ["Inputs", "\\(x\\)", "8 token ids", "up to 2,048 token ids"],
        ["Parameters", "\\(w,\\ b\\)", "9,968 weights", "175.0 billion weights"],
        ["Hyperparameters", "learning rate, number of steps", "learning rate 0.01, 128 queries per batch, 6,000 steps, 2 layers of width 16", "learning rate 0.6 × 10⁻⁴, 3.2M tokens per batch, 300B tokens, 96 layers of width 12,288"],
      ],
      rowHeads: true,
    },
    src: ["Brown et al., Language Models are Few-Shot Learners, 2020 (Table 2.1)"],
    check: {
      q: "Which of these is a hyperparameter of our model?",
      options: ["The 16 numbers of the embedding of <code>users</code>", "The learning rate 0.01", "The token id 63 of <code>users</code>", "The probability 0.3460 it gives <code>id</code> after <code>WHERE</code>"],
      answer: 1,
      why: "The learning rate is chosen before training and never changed by gradient descent. The embedding is a parameter (learned), the token id is an input, and 0.3460 is an output.",
    },
    terms: [
      { term: "Input", def: "The data a model reads for one example: \\(x\\) for the line, token ids for our model." },
      { term: "Hyperparameter", def: "A setting chosen before training and not learned by gradient descent: learning rate, batch size, number of steps, model size." },
    ],
  });

  Deck.slide({
    id: "learn-loop",
    title: "Training repeats three steps: predict, measure, adjust",
    body: String.raw`<p>Learning means choosing parameters that make the predictions match the data. Every method in this set, from the line to GPT-3, runs the same loop:</p>
<ol><li><b>Predict</b>: run the model on the inputs.</li>
<li><b>Measure</b>: compute the <b>loss</b>, one number that says how far the predictions are from the answers (chapter 2.2).</li>
<li><b>Adjust</b>: change each parameter a little in the direction that lowers the loss (chapter 2.3).</li></ol>
<p>One pass on the five points, from \(w = 0,\ b = 0\): every prediction is 0 and the loss is 33.4. The gradient of the loss is (−38, −10.8), so increasing either parameter lowers it. A step of 0.01 times the gradient, in the opposite direction, gives \(w = 0.38,\ b = 0.108\), and the loss falls to 19.64. <b>Training</b> repeats this until the loss stops falling.</p>`,
    fig: {
      type: "diagram", caption: "One turn of the training loop on the five points, with its real numbers", w: 15.4, h: 6.4,
      nodes: [
        { id: "p", x: 0.3, y: 0.7, w: 5.0, h: 1.8, label: "1 Predict", sub: "w = 0, b = 0: every ŷ = 0", kind: "blue" },
        { id: "m", x: 10.1, y: 0.7, w: 5.0, h: 1.8, label: "2 Measure", sub: "L = mean (ŷ − y)² = 33.4" },
        { id: "a", x: 10.1, y: 4.0, w: 5.0, h: 1.8, label: "3 Adjust", sub: "gradient (−38, −10.8)", kind: "hi" },
        { id: "n", x: 0.3, y: 4.0, w: 5.0, h: 1.8, label: "New parameters", sub: "w = 0.38, b = 0.108" },
      ],
      edges: [
        { from: "p", to: "m", label: "ŷ = 0, 0, 0, 0, 0" },
        { from: "m", to: "a", label: "slope of L", dx: 44 },
        { from: "a", to: "n", label: "minus 0.01 × gradient" },
        { from: "n", to: "p", label: "repeat: L = 19.64", dashed: true, dx: 66 },
      ],
    },
    check: {
      q: "The gradient at (0, 0) is (−38, −10.8) and the step size is 0.01. What are the new parameters?",
      options: ["(0.38, 0.108)", "(−0.38, −0.108)", "(38, 10.8)", "(0.0038, 0.00108)"],
      answer: 0,
      why: "Each parameter moves against its gradient: \\(w = 0 - 0.01 \\times ({-38}) = 0.38\\) and \\(b = 0 - 0.01 \\times ({-10.8}) = 0.108\\). Moving with the gradient gives the negative answer and raises the loss; forgetting the step size gives (38, 10.8).",
    },
    terms: [
      { term: "Loss", def: "One number that measures how far a model's predictions are from the data; training makes it small." },
      { term: "Training", def: "Repeating predict, measure, adjust until the loss stops falling." },
    ],
  });

  // ------------------------------------------------------------ 2.2
  Deck.chapter({ title: "Loss functions" });

  Deck.slide({
    id: "learn-error",
    title: "The error of one prediction has a size and a sign",
    body: String.raw`<p>Take the line \(\hat y = 2x\) (\(w = 2,\ b = 0\)) and the point \((4, 7)\). The prediction is 8, and the <b>error</b>, also called the <b>residual</b>, is \(e = \hat y - y = 8 - 7 = 1\): the line is 1 too high. At \((1, 3)\) the prediction is 2 and \(e = {-1}\): 1 too low.</p>
<p>The sign says which way the prediction is off; the size says by how much. Over the five points this line's errors are −1, 1, 0, 1, 2.</p>
<p>A loss function has to turn such a list into one number. Two requirements follow from the table: +1 and −1 must count as equally bad, and the loss must be 0 only when every error is 0. The next sheet shows that the plain sum of the errors fails both.</p>`,
    fig: {
      type: "table", caption: "Errors of the line \\(\\hat y = 2x\\) on the five points",
      cols: ["\\(x\\)", "\\(y\\)", "\\(\\hat y = 2x\\)", "\\(e = \\hat y - y\\)", "the line is"], align: ["r", "r", "r", "r", "l"],
      rows: [
        ["1", "3", "2", "−1", "1 too low"], ["2", "3", "4", "1", "1 too high"], ["3", "6", "6", "0", "exact"],
        ["4", "7", "8", "1", "1 too high"], ["5", "8", "10", "2", "2 too high"],
      ],
      highlight: [3],
    },
    terms: [{ term: "Error (residual)", def: "For one example, \\(e = \\hat y - y\\): positive when the prediction is too high, negative when too low." }],
  });

  Deck.slide({
    id: "learn-squared",
    title: "Squaring the errors makes every miss count",
    body: String.raw`<p>Adding raw errors fails because positive and negative errors cancel. The flat line \(\hat y = 5.4\) misses all five points, by 2.4, 2.4, −0.6, −1.6 and −2.6, and these sum to exactly 0. So does the best line, \(\hat y = 1.4x + 1.2\): the sum cannot tell them apart.</p>
<p>The <b>squared error</b> \(e^2\) has three properties that matter:</p>
<ul><li>It is never negative, so misses cannot cancel: the flat line's squared errors sum to 21.2, the best line's to 1.6.</li>
<li>It grows faster than the miss: an error of 2 costs 4, an error of 1 costs 1, so one large miss costs more than two small ones.</li>
<li>It is smooth: its derivative \(2e\) exists everywhere and shrinks to 0 with the error, which gradient descent needs (chapter 2.3).</li></ul>
<p>In the plate each red square has the miss as its side, so its area is proportional to the squared error.</p>`,
    fig: {
      type: "plot", width: WIDE, height: NARROW ? 300 : 320, caption: "Squared errors drawn as squares; start from the flat line \\(\\hat y = 5.4\\)",
      x: [0, 6], y: [0, 12], xlabel: "x", ylabel: "y",
      params: [{ name: "w", label: "slope w", min: -1, max: 3, step: 0.01, value: 0 }, { name: "b", label: "intercept b", min: -2, max: 8, step: 0.01, value: 5.4 }],
      fns: [{ f: (x, p) => p.w * x + p.b }],
      points: DX.map((x, i) => [x, DY[i]]), residuals: true, residualSquares: true,
      readouts: [
        { label: "sum of errors \\(\\sum e_i\\)", f: (p) => DX.reduce((a, x, i) => a + p.w * x + p.b - DY[i], 0), fmt: "f2" },
        { label: "sum of squared errors \\(\\sum e_i^2\\)", f: (p) => DX.reduce((a, x, i) => a + (p.w * x + p.b - DY[i]) ** 2, 0), fmt: "f2" },
      ],
    },
    terms: [{ term: "Squared error", def: "\\(e^2 = (\\hat y - y)^2\\) for one example: never negative, smooth, and four times larger for a miss twice as big." }],
  });

  Deck.slide({
    id: "learn-mse",
    title: "Mean squared error averages the squared misses",
    body: String.raw`<p>The <b>mean squared error</b> (MSE) of a model on \(n\) examples is the average of their squared errors. A <b>loss function</b> like this one depends on the parameters: the data is fixed, so for the line the MSE is a function of \(w\) and \(b\).</p>
<p>For \(\hat y = 2x\) the errors are −1, 1, 0, 1, 2, their squares 1, 1, 0, 1, 4, the sum 7 and the MSE \(7/5 = 1.4\). The flat line \(\hat y = 5.4\) scores \(21.2/5 = 4.24\). The best line, \(\hat y = 1.4x + 1.2\), scores 0.32, the lowest any line can reach on these points.</p>
<p>Dividing by \(n\) makes losses comparable across dataset sizes: an MSE of 1.4 is a squared miss of 1.4 per point on average. Its square root, 1.18, is in the units of \(y\).</p>`,
    math: String.raw`L(w, b) = \frac{1}{n} \sum_{i=1}^{n} \left(w x_i + b - y_i\right)^2`,
    fig: {
      type: "table", caption: "MSE of two lines, worked point by point",
      cols: ["\\(x\\)", "\\(y\\)", "\\(\\hat y = 2x\\)", "\\(e^2\\)", "\\(\\hat y = 1.4x + 1.2\\)", "\\(e^2\\)"], align: ["r", "r", "r", "r", "r", "r"],
      rows: [
        ["1", "3", "2", "1", "2.6", "0.16"], ["2", "3", "4", "1", "4.0", "1.00"], ["3", "6", "6", "0", "5.4", "0.36"],
        ["4", "7", "8", "1", "6.8", "0.04"], ["5", "8", "10", "4", "8.2", "0.04"],
        ["", "", "sum", "7", "sum", "1.60"], ["", "", "<b>MSE</b>", "<b>1.4</b>", "<b>MSE</b>", "<b>0.32</b>"],
      ],
      highlight: [6],
    },
    terms: [
      { term: "Mean squared error (MSE)", def: "The average of the squared errors over a dataset: \\(\\frac1n \\sum_i (\\hat y_i - y_i)^2\\)." },
      { term: "Loss function", def: "A function of the parameters that measures how far the model's predictions are from the data. Training minimizes it." },
    ],
  });

  Deck.slide({
    id: "learn-mae",
    title: "Absolute error resists a single bad point",
    body: String.raw`<p>The <b>mean absolute error</b> (MAE) averages \(|e|\) instead of \(e^2\). It counts a miss of 10 as 10 times a miss of 1, where squared error counts it as 100 times. So one far-off point, an <b>outlier</b>, pulls the MSE fit much harder than the MAE fit.</p>
<p>To compare the two exactly, fit only the slope of \(\hat y = w x\) (\(b = 0\)). On the original data the MSE-best slope is 1.727 and the MAE-best slope is 1.750. Drag point 5 from \(y = 8\) up to \(y = 20\): the MSE slope rises to 2.818, while the MAE slope rises only to 2.000, reached at \(y = 10\), and stays there however far the point goes.</p>
<p>The reason is the derivative. For this model the derivative of \(\sum_i |e_i|\) with respect to \(w\) is \(\sum_i x_i\, \mathrm{sign}(e_i)\): it depends on which side of the line each point lies, not on how far. The cost: \(|e|\) has a corner at 0 where its derivative jumps from −1 to +1, and the MAE line need not be unique (for this model it is). MSE is smooth and, for a line, has an exact formula for its minimum (the last sheet of this chapter).</p>`,
    fig: { type: "learn-outlier", caption: "Move one point: the best slope of \\(\\hat y = wx\\) under each loss", point: 4 },
    terms: [
      { term: "Mean absolute error (MAE)", def: "The average of \\(|\\hat y_i - y_i|\\) over a dataset. Less sensitive than MSE to a few large errors." },
      { term: "Outlier", def: "An example far from the pattern of the others; squared error lets it dominate the fit." },
    ],
  });

  Deck.slide({
    id: "learn-curve",
    title: "Hold b fixed and the loss is a parabola in w",
    body: String.raw`<p>The loss is a function of the parameters; \(x\) and \(y\) are fixed by the data. Hold \(b = 0\) and the MSE depends on \(w\) alone. With \(\sum x_i^2 = 55\), \(\sum x_i y_i = 95\) and \(\sum y_i^2 = 167\), expanding the square gives a parabola:</p>
\[L(w) = \tfrac{1}{5}\left(55\,w^2 - 2 \cdot 95\,w + 167\right) = 11\,w^2 - 38\,w + 33.4\]
<p>At \(w = 1\) the loss is 6.4 and the slope \(dL/dw = 22w - 38\) is −16: increasing \(w\) lowers the loss. The bottom is where the slope is 0, at \(w = 38/22 = 1.727\), with \(L = 0.582\).</p>
<p>Change \(b\) and the parabola moves. At \(b = 1.2\) its bottom is at \(w = 1.4\) with \(L = 0.32\); at \(b = {-2}\) it is at \(w = 2.273\). The best \(w\) depends on \(b\), so the two have to be found together.</p>`,
    fig: { type: "learn-curve", caption: "The line with its squared errors, and the loss for every \\(w\\) at the chosen \\(b\\)", w: 1, b: 0 },
  });

  Deck.slide({
    id: "learn-surface",
    title: "With two parameters the loss is a surface",
    body: String.raw`<p>With \(w\) and \(b\) both free, the loss is a function of two inputs, a surface over the \((w, b)\) plane. The contour map draws it from above: each curve joins the points where \(L\) has one value (Part 1 draws level curves the same way).</p>
<p>For a line under MSE the <b>loss surface</b> is a bowl with a single lowest point, at \(w = 1.4,\ b = 1.2\), where \(L = 0.32\). Its contours are ellipses, and they are long and thin. Along the long axis \(w\) and \(b\) change in the ratio −1 to 3.61, so the line pivots about \(x = 3.61\), near the middle of the data, so the predictions barely change and the loss rises slowly. Across it the line tilts and the loss rises fast. The curvatures in the two directions differ by a factor of 70.</p>
<p>Click the map or move the sliders. The data panel draws the line for the chosen \((w, b)\) next to the best line (dashed).</p>`,
    fig: { type: "learn-surface", mode: "explore", start: [2, 0], caption: "Contours of \\(L(w, b)\\) at 0.5, 1, 2, 4 … 64, with the minimum marked ×" },
    terms: [{ term: "Loss surface", def: "The loss as a function of all the parameters; for two parameters it can be drawn as a contour map." }],
  });

  Deck.slide({
    id: "learn-normal",
    title: "For a line the minimum has a formula",
    body: String.raw`<p>At the bottom of the bowl both partial derivatives are zero. For the line they are linear in \(w\) and \(b\), so setting them to zero gives two linear equations, the <b>normal equations</b>, which solve exactly (step through the plate). Their solution is the <b>least-squares</b> fit: the line with the smallest MSE.</p>
<p>For a neural network this fails. Its loss contains products of parameters and nonlinear functions, so the 9,968 equations “gradient = 0” for our model are nonlinear, have many solutions, and have no formula. The general method is search: start somewhere and repeatedly step downhill. That is gradient descent, the subject of the next chapter, and it is how every model in this set is trained.</p>`,
    fig: {
      type: "steps", caption: "Setting both partial derivatives to zero", start: 0,
      steps: [
        { html: String.raw`\[L(w, b) = \tfrac{1}{5} \sum_{i=1}^{5} \left(w x_i + b - y_i\right)^2\]`, note: "the MSE of the line" },
        { html: String.raw`\[\frac{\partial L}{\partial w} = \tfrac{2}{5} \sum_i \left(w x_i + b - y_i\right) x_i = 0 \;\Rightarrow\; 55\,w + 15\,b = 95\]`, note: "uses Σx² = 55, Σx = 15, Σxy = 95" },
        { html: String.raw`\[\frac{\partial L}{\partial b} = \tfrac{2}{5} \sum_i \left(w x_i + b - y_i\right) = 0 \;\Rightarrow\; 15\,w + 5\,b = 27\]`, note: "uses Σy = 27" },
        { html: String.raw`\[b = 5.4 - 3w \;\Rightarrow\; 55\,w + 15\,(5.4 - 3w) = 95 \;\Rightarrow\; 10\,w = 14\]` },
        { html: String.raw`\[w = 1.4, \qquad b = 5.4 - 3 \times 1.4 = 1.2\]` },
        { html: String.raw`\[L(1.4,\ 1.2) = \tfrac{1}{5}\left(0.16 + 1 + 0.36 + 0.04 + 0.04\right) = 0.32\]`, note: "no line does better" },
      ],
    },
    check: {
      q: "With \\(b\\) held at 0, the best slope under MSE is \\(w = \\sum x_i y_i / \\sum x_i^2\\). What is it for our five points?",
      options: ["1.4", "1.727", "1.75", "2"],
      answer: 1,
      why: "\\(95 / 55 = 1.727\\). 1.4 is the best slope when \\(b\\) is free (and equals 1.2); 1.75 is the MAE-best slope through the origin; 2 is the first guess \\(\\hat y = 2x\\).",
    },
    terms: [
      { term: "Normal equations", def: "The linear equations obtained by setting the gradient of the MSE of a linear model to zero; their solution is the exact least-squares fit." },
      { term: "Least squares", def: "Fitting by minimizing the sum (or mean) of squared errors." },
    ],
  });

  // ------------------------------------------------------------ 2.3
  Deck.chapter({ title: "Gradient descent" });

  Deck.slide({
    id: "learn-update",
    title: "Step against the slope: w ← w − η · dL/dw",
    body: String.raw`<p>Near a point \(w\), the loss is close to its tangent line (Part 1): \(L(w + \Delta) \approx L(w) + L'(w)\,\Delta\). To make the loss smaller, choose \(\Delta\) with the sign opposite to \(L'(w)\). The simplest choice is a fixed multiple of the slope. That is <b>gradient descent</b>, the update in the box below.</p>
<p>The positive number \(\eta\) (eta) is the <b>learning rate</b>. With this step the tangent predicts a decrease of \(\eta\,L'(w)^2\), which is never negative. The prediction is accurate only for small steps, because the tangent ignores the curvature.</p>
<p>On the five points with \(b = 0\): at \(w = 0\) the slope is −38. With \(\eta = 0.01\) the step is \(+0.38\) and the tangent predicts a drop of 14.44, to 18.96. The real loss at \(w = 0.38\) is 20.548, a drop of 12.852. The difference, \(11 \times 0.38^2 = 1.588\), is the curvature term the tangent leaves out.</p>`,
    math: String.raw`w \leftarrow w - \eta\, \frac{dL}{dw}`,
    fig: {
      type: "steps", caption: "Deriving the update, then checking it on \\(L(w) = 11w^2 - 38w + 33.4\\)", start: 0,
      steps: [
        { html: String.raw`\[L(w + \Delta) \approx L(w) + L'(w)\,\Delta\]`, note: "the tangent line at w" },
        { html: String.raw`\[\Delta = {-\eta\, L'(w)} \;\Rightarrow\; L(w + \Delta) \approx L(w) - \eta\, L'(w)^2\]`, note: "the predicted decrease is ≥ 0" },
        { html: String.raw`\[L'(w) = 22w - 38, \qquad L'(0) = {-38}\]`, note: "five points, b = 0" },
        { html: String.raw`\[\eta = 0.01:\quad \Delta = 0.38, \quad 33.4 - 0.01 \times 38^2 = 33.4 - 14.44 = 18.96\]` },
        { html: String.raw`\[L(0.38) = 20.548 = 18.96 + 11 \times 0.38^2\]`, note: "the curvature term" },
      ],
    },
    terms: [
      { term: "Gradient descent", def: "Repeatedly moving the parameters a small step against the gradient of the loss: \\(w \\leftarrow w - \\eta\\, \\partial L/\\partial w\\)." },
      { term: "Learning rate", def: "The step-size multiplier \\(\\eta\\) in gradient descent; a hyperparameter." },
    ],
  });

  Deck.slide({
    id: "learn-three",
    title: "Three steps by hand",
    body: String.raw`<p>The simplest loss to descend is the squared error of our first point, \((1, 3)\), alone, for the model \(\hat y = w x\): \(L(w) = (w \cdot 1 - 3)^2 = (w - 3)^2\). Its slope is \(2(w - 3)\) and its minimum is at \(w = 3\). Start at \(w = {-1}\) with \(\eta = 0.1\):</p>
<table><thead><tr><th class="r">step</th><th class="r">\(w\)</th><th class="r">\(L(w)\)</th><th class="r">slope</th><th class="r">\(w - 0.1 \times\) slope</th></tr></thead><tbody>
<tr><td class="r">0</td><td class="r">−1</td><td class="r">16</td><td class="r">−8</td><td class="r">−0.2</td></tr>
<tr><td class="r">1</td><td class="r">−0.2</td><td class="r">10.24</td><td class="r">−6.4</td><td class="r">0.44</td></tr>
<tr><td class="r">2</td><td class="r">0.44</td><td class="r">6.5536</td><td class="r">−5.12</td><td class="r">0.952</td></tr>
<tr><td class="r">3</td><td class="r">0.952</td><td class="r">4.1943</td><td class="r">−4.096</td><td class="r">1.3616</td></tr></tbody></table>
<p>The distance to the minimum goes 4, 3.2, 2.56, 2.048: it shrinks by the same factor, 0.8, every step, so after 10 steps it is \(4 \times 0.8^{10} = 0.43\). The plate runs the same steps; press Step three times, then 10 steps.</p>`,
    fig: {
      type: "gd", caption: "Gradient descent on \\(L(w) = (w - 3)^2\\) from \\(w = {-1}\\)",
      f: (w) => (w - 3) ** 2, df: (w) => 2 * (w - 3), start: -1, lr: 0.1, x: [-2, 8], y: [0, 18], lrMin: 0.01, lrMax: 1, lrStep: 0.01, xlabel: "w", ylabel: "L(w)", v: "w", lrName: "η",
    },
  });

  Deck.slide({
    id: "learn-lr",
    title: "The learning rate decides between crawling, converging and diverging",
    body: String.raw`<p>For \(L(w) = (w - 3)^2\) the update is \(w_{t+1} = w_t - 2\eta\,(w_t - 3)\). Subtract 3 from both sides: each step multiplies the distance to the minimum by the same factor (the box below), so everything depends on \(1 - 2\eta\):</p>
<ul><li>\(\eta = 0.02\): factor 0.96; after 10 steps 66% of the distance remains. Too small.</li>
<li>\(\eta = 0.5\): factor 0; the first step lands exactly on 3.</li>
<li>\(\eta = 0.9\): factor −0.8; each step overshoots to the other side, but the distance still shrinks.</li>
<li>\(\eta = 1.05\): factor −1.1; each overshoot is larger than the last, and after 10 steps the distance has grown 2.6 times. For any \(\eta > 1\), \(|1 - 2\eta| > 1\) and the steps diverge.</li></ul>
<p>The limit comes from the curvature. For a parabola with second derivative \(L''\) the factor is \(1 - \eta L''\), so the steps converge only if \(\eta \lt 2/L''\). Here \(L'' = 2\). For all five points with \(b = 0\), \(L'' = 22\) and the limit drops to \(\eta \lt 0.0909\).</p>`,
    math: String.raw`w_{t+1} - 3 = (1 - 2\eta)\,(w_t - 3)`,
    fig: { type: "learn-lr", caption: "Ten steps from \\(w = {-1}\\) at any learning rate; the distance panel is on a log scale", lr: 0.1 },
  });

  Deck.slide({
    id: "learn-gd2",
    title: "Descending in w and b at once",
    body: String.raw`<p>With two parameters the gradient has two components, and the update applies the same rule to each, with the same \(\eta\). The two components are in the box below, with \(e_i = w x_i + b - y_i\).</p>
<p>From (0, 0) the gradient is (−38, −10.8), and with \(\eta = 0.01\) the first step goes to \((0.38, 0.108)\), loss 19.64, the turn of the loop in chapter 2.1. Ten steps reach \((1.494, 0.441)\) with loss 0.566, near the floor of the valley. Then progress slows: after 100 steps the loss is 0.373, and it takes 514 steps to get within 1% of the minimum, 0.32.</p>
<p>Raise \(\eta\) to 0.08 and the path zig-zags across the valley but gets within 1% in 64 steps. At 0.085 it diverges. The next sheet explains both.</p>`,
    math: String.raw`\frac{\partial L}{\partial w} = \frac{2}{n} \sum_{i} e_i\, x_i, \qquad \frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i} e_i`,
    fig: { type: "learn-surface", mode: "descent", start: [0, 0], lr: 0.01, caption: "Gradient descent on the five points from \\((w, b) = (0, 0)\\)" },
  });

  Deck.slide({
    id: "learn-valley",
    title: "A narrow valley limits the learning rate",
    body: String.raw`<p>On \(L = w_1^2 + 10\,w_2^2\) the two coordinates descend independently: each step multiplies \(w_1\) by \(1 - 2\eta\) and \(w_2\) by \(1 - 20\eta\). The steep direction sets the limit: \(w_2\) diverges unless \(\eta \lt 2/20 = 0.1\).</p>
<p>At \(\eta = 0.09\), just under the limit, \(w_2\) is multiplied by −0.8 per step: it overshoots and zig-zags across the valley. \(w_1\) is multiplied by 0.82 and creeps along the valley floor; 10 steps take it only from −4 to −0.55. Lowering \(\eta\) stops the zig-zag and makes \(w_1\) even slower.</p>
<p>The ratio of the largest to the smallest curvature, here 10, measures how hard the problem is for gradient descent. On the five points, \(L - 0.32 = 11\,dw^2 + 6\,dw\,db + db^2\), so the Hessian is \(\left[\begin{smallmatrix} 22 & 6 \\ 6 & 2 \end{smallmatrix}\right]\) with eigenvalues \(12 \pm \sqrt{136}\): curvatures 23.66 and 0.338, a ratio of 70. That is why its descent slowed after reaching the valley and why \(\eta\) above \(2/23.66 = 0.0845\) diverged. Part 3 shows how momentum and Adam move faster along the flat direction.</p>`,
    fig: {
      type: "gd", dim: 2, caption: "\\(L = w_1^2 + 10\\,w_2^2\\) from (−4, 1.5). The slider goes past the limit of 0.1; above it \\(w_2\\) diverges and the point soon leaves the plot",
      f: (a, b) => a * a + 10 * b * b, grad: (a, b) => [2 * a, 20 * b], start: [-4, 1.5], lr: 0.09, x: [-5, 5], y: [-2, 2], min: [0, 0], lrMin: 0.005, lrMax: 0.12, lrStep: 0.005,
    },
  });

  Deck.slide({
    id: "learn-sgd",
    title: "A minibatch gradient is a noisy, unbiased estimate",
    body: String.raw`<p>The full gradient averages one gradient per example: \(\nabla L = \frac{1}{n}\sum_i \nabla \ell_i\), where \(\ell_i = e_i^2\) is the loss on example \(i\). For five points that is cheap; for billions of tokens it is not. <b>Stochastic gradient descent</b> (SGD) estimates the gradient from a random <b>minibatch</b> of \(B\) examples and steps with that estimate.</p>
<p>At (0, 0) the five values of \(\partial \ell_i / \partial w\) are −6, −12, −36, −56 and −80. The 10 possible minibatches of 2 give 10 estimates of \(\partial L / \partial w\), from −9 to −68 (two coincide at −46), and their average is exactly −38, the full gradient. An estimate whose average over all possible draws equals the true value is <b>unbiased</b>.</p>
<p>Three counts with numbers: the <b>batch size</b> \(B\) is the number of examples per step; a <b>step</b> is one parameter update; an <b>epoch</b> is one pass over the dataset, 5 steps at \(B = 1\). Our model took 6,000 steps of 128 freshly sampled queries each, 768,000 queries in all.</p>`,
    fig: { type: "learn-batches", caption: "Every possible minibatch gradient at \\((0, 0)\\) and their average", B: 2 },
    terms: [
      { term: "Stochastic gradient descent (SGD)", def: "Gradient descent that uses the gradient of a random minibatch instead of the whole dataset at each step." },
      { term: "Minibatch", def: "The random subset of examples whose average gradient is used for one step." },
      { term: "Batch size", def: "The number of examples in a minibatch, \\(B\\). Our model used 128 queries." },
      { term: "Epoch", def: "One pass over the whole training set." },
      { term: "Unbiased estimate", def: "An estimate whose average over all possible random draws equals the true value." },
    ],
  });

  Deck.slide({
    id: "learn-noise",
    title: "Gradient noise shrinks as one over the square root of the batch size",
    body: String.raw`<p>Draw the \(B\) examples of a minibatch independently. The estimate is then an average of \(B\) independent per-example gradients, so its <b>variance</b> (the square of its standard deviation) is the per-example variance divided by \(B\), and its standard deviation is \(\sigma / \sqrt{B}\): Part 1's law of large numbers, applied to gradients. Quadrupling the batch halves the noise. The previous sheet drew without replacement from \(n = 5\) examples, which shrinks the spread by a further \(\sqrt{(n - B)/(n - 1)}\): 16.86 instead of 19.47 at \(B = 2\), and 0 at \(B = n\). With \(n\) in the billions that factor is 1.</p>
<p>At \((0, 0)\) the five per-example values of \(\partial \ell_i / \partial w\) have \(\sigma = 27.54\). The plate draws 2,000 minibatches for each batch size. At \(B = 1\) the estimates have standard deviation 27.28; at \(B = 128\), 2.42, against \(27.54/\sqrt{128} = 2.43\).</p>
<p>At the minimum \((1.4, 1.2)\) the full gradient is 0 but the per-example gradients are −0.8, 4, −3.6, −1.6 and 2, with \(\sigma = 2.69\), so SGD at a fixed learning rate keeps being pushed around the minimum. That is one reason schedules lower the learning rate at the end; ours decays from 0.01 to 0.001.</p>`,
    fig: { type: "learn-noise", caption: "Minibatches drawn with replacement, 2,000 per batch size; the dashed line is \\(\\sigma/\\sqrt{B}\\)", at: "start", B: 1 },
  });

  Deck.slide({
    id: "learn-curves",
    title: "A training curve falls fast, then flattens",
    body: String.raw`<p>A <b>training curve</b> plots the loss against the step. The plate shows three runs on the five points from \((0, 0)\):</p>
<ul><li>Full-batch gradient descent, \(\eta = 0.05\): 33.4 to 0.41 in 5 steps, then a long tail along the valley. It is within 1% of the minimum after 102 steps.</li>
<li>SGD with \(B = 1\), \(\eta = 0.03\): it falls as fast, then never settles. Over the last 100 steps it averages 0.44 and ranges from 0.32 to 1.07.</li>
<li>The same SGD with \(\eta\) lowered along a cosine to 0 at step 600: it ends at 0.321.</li></ul>
<p>When to stop: after a budget of steps fixed in advance (our model: 6,000, with its learning rate decayed), or when the loss on held-out data stops improving (chapter 2.6). The rule “stop when 10 steps improve the loss by less than 0.1%” stops gradient descent at step 143, loss 0.3208. On the constant-rate SGD curve it fires at step 12, at 0.69, because the loss happens to be higher than 10 steps earlier. A noisy curve has to be averaged before such a rule means anything.</p>`,
    fig: { type: "learn-curves", caption: "Loss against step for three runs, log scale; the slider reads all three at one step", t: 100 },
    check: {
      q: "SGD with \\(B = 1\\) and constant \\(\\eta = 0.03\\) hovers around 0.44 although the minimum is 0.32. Which change lowers that plateau?",
      options: ["A larger learning rate", "A smaller learning rate or a larger batch", "More steps at the same learning rate", "Starting closer to the minimum"],
      answer: 1,
      why: "The plateau is set by gradient noise: each step moves the parameters by about \\(\\eta\\,\\sigma/\\sqrt{B}\\) even at the minimum. With \\(\\eta = 0.01\\) the average over the last 100 steps drops from 0.44 to 0.34. More steps or a better start leave the same noise in place.",
    },
    terms: [{ term: "Training curve", def: "The loss plotted against the training step." }],
  });
  // ------------------------------------------------------------ 2.4
  Deck.chapter({ title: "Classification" });

  Deck.slide({
    id: "learn-classes",
    title: "Classification predicts one of a fixed set of categories",
    example: "tokens ids focus:4 target:5",
    body: String.raw`<p>The line predicts a number. <b>Classification</b> predicts which of \(K\) categories, called <b>classes</b>, an input belongs to. Our model is a classifier with \(K = 113\): after <code>SELECT name FROM users WHERE</code> it must pick one of 113 tokens, and the token that really followed, <code>id</code>, is class 58, its token id.</p>
<p>The classifiers in this set output a probability for every class, so a model can say “<code>id</code>, <code>email</code> or <code>age</code>, one third each”, which is exactly right for our grammar here: the filter column can be any column of <code>users</code> except <code>name</code>.</p>
<p>The target is written as a <b>one-hot vector</b> \(\mathbf y\): 113 entries, all 0 except \(y_{58} = 1\). Two pieces are needed: a way to turn the model's 113 output numbers into a probability distribution (the next three sheets), and a loss that compares that distribution with \(\mathbf y\) (the three after).</p>`,
    fig: {
      type: "table", caption: "Predicting a number and predicting a class",
      cols: ["", "Regression (chapter 2.2)", "Classification (this chapter)"],
      rows: [
        ["Target", "a number, \\(y = 7\\) at \\(x = 4\\)", "a class: <code>id</code>, number 58 of 113"],
        ["Model output", "one number \\(\\hat y\\)", "113 probabilities that sum to 1"],
        ["Loss for one example", "squared error \\((\\hat y - y)^2\\)", "cross-entropy \\(-\\ln\\, p_{\\text{target}}\\)"],
        ["Our example", "the line on five points", "the trained model after <code>WHERE</code>"],
      ],
      rowHeads: true,
    },
    terms: [
      { term: "Classification", def: "Predicting which of a fixed set of categories an input belongs to, as a probability for each category." },
      { term: "Class", def: "One of the categories a classifier chooses between; for our model, one of the 113 tokens." },
      { term: "One-hot vector", def: "A vector with a 1 at the position of the true class and 0 everywhere else." },
    ],
  });

  Deck.slide({
    id: "learn-sigmoid",
    title: "With two classes, the sigmoid turns one score into a probability",
    body: String.raw`<p>For two classes a model can output one score \(z\), any real number, called a <b>logit</b>. The sigmoid function (Part 1) maps it to a probability:</p>
\[\sigma(z) = \frac{1}{1 + e^{-z}}\]
<p>Class 1 gets \(p = \sigma(z)\) and class 0 gets the rest, \(1 - \sigma(z) = \sigma({-z})\).</p>
<p>Worked: \(\sigma(0) = 1/(1 + 1) = 0.5\). \(\sigma(2) = 1/(1 + e^{-2}) = 1/(1 + 0.1353) = 0.881\), and \(\sigma({-2}) = 0.119 = 1 - 0.881\). A logit of 5 gives 0.9933. The output approaches 1 but never reaches it, so a finite logit always leaves some probability on the other class.</p>
<p>The logit is the log of the odds, \(z = \ln\, \frac{p}{1 - p}\): at \(p = 0.881\) the odds are 7.39 to 1, and \(\ln\, 7.39 = 2\).</p>`,
    fig: {
      type: "plot", width: WIDE, height: NARROW ? 300 : 320, caption: "\\(\\sigma(z)\\), with the three worked values marked", x: [-6, 6], y: [0, 1], xlabel: "logit z", ylabel: "p = σ(z)", yticks: [0, 0.25, 0.5, 0.75, 1],
      params: [{ name: "z", label: "logit z", min: -6, max: 6, step: 0.01, value: 2 }],
      fns: [{ f: (x) => 1 / (1 + Math.exp(-x)) }],
      extra: (p) => ({ dots: [[-2, 0], [0, 0], [2, 0]].map(([x]) => ({ x, y: 1 / (1 + Math.exp(-x)), r: 4, cls: "ink", tip: `σ(${x}) = ${F(1 / (1 + Math.exp(-x)), 4)}` })).concat([{ x: p.z, y: 1 / (1 + Math.exp(-p.z)), r: 6.5, tip: `σ(${F(p.z, 2)}) = ${F(1 / (1 + Math.exp(-p.z)), 4)}` }]) }),
      readouts: [
        { label: "\\(\\sigma(z)\\), class 1", f: (p) => 1 / (1 + Math.exp(-p.z)), fmt: "f4" },
        { label: "\\(1 - \\sigma(z)\\), class 0", f: (p) => 1 / (1 + Math.exp(p.z)), fmt: "f4" },
        { label: "odds \\(e^{z}\\)", f: (p) => Math.exp(p.z), fmt: "f3" },
      ],
    },
    terms: [{ term: "Logit", def: "A model's unnormalized score for a class, any real number; softmax or the sigmoid turns logits into probabilities." }],
  });

  Deck.slide({
    id: "learn-softmax",
    title: "Softmax turns K scores into K probabilities",
    body: String.raw`<p>For \(K\) classes the model outputs \(K\) logits \(z_1 \dots z_K\). <b>Softmax</b> exponentiates each and divides by the total:</p>
\[p_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}\]
<p>Every exponential is positive, so every \(p_i > 0\), and dividing by the sum makes them add to 1. Worked on the logits (2, 1, 0, −1): the exponentials are 7.389, 2.718, 1 and 0.368, their sum is 11.475, and the probabilities are 0.644, 0.237, 0.087 and 0.032. A logit 1 higher means a probability \(e = 2.718\) times higher.</p>
<p>Adding the same constant \(c\) to every logit multiplies every \(e^{z_i}\) by \(e^c\), which cancels in the division, so the probabilities do not change (move the last slider). Implementations subtract the largest logit before exponentiating for this reason (Part 1). With \(K = 2\), softmax is the sigmoid of the difference: \(p_1 = \sigma(z_1 - z_2)\).</p>`,
    fig: {
      type: "calc", caption: "Softmax of four logits; the last slider adds the same constant to all four",
      inputs: [
        { id: "z1", label: "logit z₁", min: -4, max: 4, step: 0.1, value: 2, fmt: "f1" }, { id: "z2", label: "logit z₂", min: -4, max: 4, step: 0.1, value: 1, fmt: "f1" },
        { id: "z3", label: "logit z₃", min: -4, max: 4, step: 0.1, value: 0, fmt: "f1" }, { id: "z4", label: "logit z₄", min: -4, max: 4, step: 0.1, value: -1, fmt: "f1" },
        { id: "c", label: "add c to every logit", min: -10, max: 10, step: 0.5, value: 0, fmt: "f1" },
      ],
      outputs: [1, 2, 3, 4].map((i) => ({ label: `\\(p_${i}\\)`, f: (v) => { const z = [v.z1, v.z2, v.z3, v.z4].map((q) => q + v.c), mx = Math.max(...z), e = z.map((q) => Math.exp(q - mx)); return e[i - 1] / e.reduce((a, b) => a + b, 0); }, fmt: "f3" })).concat([
        { label: "sum", f: (v) => { const z = [v.z1, v.z2, v.z3, v.z4].map((q) => q + v.c), mx = Math.max(...z), e = z.map((q) => Math.exp(q - mx)); return e.reduce((a, b) => a + b, 0) / e.reduce((a, b) => a + b, 0); }, fmt: "f3" }]),
      note: (v) => { const z = [v.z1, v.z2, v.z3, v.z4].map((q) => q + v.c), e = z.map((q) => Math.exp(q)), S = e.reduce((a, b) => a + b, 0); return `Exponentials: ${e.map((q) => UI.format(q, q < 1000 ? "f3" : "sci")).join(", ")}; their sum ${UI.format(S, S < 1000 ? "f3" : "sci")}. Each probability is one exponential divided by the sum.`; },
    },
    terms: [{ term: "Softmax", def: "\\(p_i = e^{z_i} / \\sum_j e^{z_j}\\): turns \\(K\\) logits into \\(K\\) positive probabilities that sum to 1." }],
  });

  Deck.slide({
    id: "learn-logits",
    title: "Our model's scores after WHERE",
    example: "tokens ids focus:4 target:5",
    body: String.raw`<p>Run the trained model on <code>SELECT name FROM users WHERE</code> and read position 4. It outputs 113 logits. The four column names of <code>users</code> get <code>id</code> 7.621, <code>email</code> 7.617, <code>age</code> 7.502 and <code>name</code> 1.994.</p>
<p>Softmax over all 113 gives <code>id</code> 0.3460, <code>email</code> 0.3448, <code>age</code> 0.3072 and <code>name</code> 0.0012; the other 109 tokens share the remaining 0.0008. The grammar's answer is 1/3, 1/3, 1/3 and 0: the filter column must differ from <code>name</code>, and the three others are equally likely.</p>
<p>Differences between logits become ratios of probabilities. <code>age</code> is 0.119 below <code>id</code>, so it gets \(e^{-0.119} = 0.888\) times the probability. <code>name</code> is 5.63 below, a factor of \(e^{-5.63} = 0.0036\). Raise the logit of <code>name</code> in the plate: its probability grows and all the others shrink, because they share one total.</p>`,
    fig: { type: "learn-logits", caption: "Logits and probabilities at position 4, computed from the trained weights" },
  });

  Deck.slide({
    id: "learn-se-probs",
    title: "Squared error on a probability stalls when the model is confidently wrong",
    body: String.raw`<p>Squared error can be applied to a probability. With two classes, \(p = \sigma(z)\) and target class 1, take \(L = (p - 1)^2\). It is a poor loss for training, and its gradient with respect to the logit shows why:</p>
\[\frac{dL}{dz} = 2\,(p - 1)\,\sigma'(z) = 2\,(p - 1)\, p\,(1 - p)\]
<p>The factor \(p(1 - p)\) is the sigmoid's own slope, and it vanishes at both ends. When the model is right and sure, that is harmless. When it is wrong and sure, the gradient disappears: at \(z = {-5}\) the true class gets \(p = 0.0067\) and the gradient is only −0.0132; at \(z = {-10}\) it is \(-9.1 \times 10^{-5}\). The more confidently wrong the model is, the less the loss pushes it.</p>
<p>At \(z = 0\), where the model is unsure, the gradient is −0.25, about 19 times larger than at \(z = {-5}\). Gradient descent would barely move the examples it gets most wrong.</p>`,
    fig: {
      type: "plot", width: WIDE, height: NARROW ? 300 : 320, caption: "Squared error \\((\\sigma(z) - 1)^2\\) and its gradient with respect to the logit, for target class 1", x: [-10, 6], y: [-0.35, 1.05], xlabel: "logit z of the true class", ylabel: "value", yticks: [-0.25, 0, 0.25, 0.5, 0.75, 1],
      params: [{ name: "z", label: "logit z", min: -10, max: 6, step: 0.05, value: -5 }],
      fns: [{ f: (x) => (1 / (1 + Math.exp(-x)) - 1) ** 2, cls: "ink" }, { f: (x) => { const q = 1 / (1 + Math.exp(-x)); return 2 * (q - 1) * q * (1 - q); }, cls: "s2" }],
      hlines: [{ y: 0 }],
      extra: (p) => { const q = 1 / (1 + Math.exp(-p.z)); return { dots: [{ x: p.z, y: (q - 1) ** 2, r: 5.5, cls: "ink" }, { x: p.z, y: 2 * (q - 1) * q * (1 - q), r: 5.5 }] }; },
      readouts: [
        { label: "\\(p = \\sigma(z)\\)", f: (p) => 1 / (1 + Math.exp(-p.z)), fmt: "f4" },
        { label: "loss \\((p - 1)^2\\)", f: (p) => (1 / (1 + Math.exp(-p.z)) - 1) ** 2, fmt: "f4" },
        { label: "gradient \\(dL/dz\\) (red)", f: (p) => { const q = 1 / (1 + Math.exp(-p.z)); return 2 * (q - 1) * q * (1 - q); }, fmt: (v) => (Math.abs(v) < 1e-3 ? sci(v, 2) : UI.format(v, "f4")) },
      ],
    },
  });

  Deck.slide({
    id: "learn-xent",
    title: "Cross-entropy: minus the log of the probability of the right answer",
    body: String.raw`<p>Treat the model's probabilities as a claim about the data and ask how probable the data is under it. The <b>likelihood</b> of a dataset is the product of the probabilities the model gave the correct class of each example; training should make it large. A product of many numbers below 1 underflows (Part 1), so take the log, which turns it into a sum, and negate it to get a loss: the <b>negative log-likelihood</b>. Its average over examples is the <b>cross-entropy loss</b>, \(-\ln\, p\) per example, with \(p\) the probability of the correct class.</p>
<table><thead><tr><th>\(p\)</th><th class="r">0.9</th><th class="r">0.5</th><th class="r">0.1</th><th class="r">0.01</th></tr></thead><tbody><tr><th>\(-\ln\, p\)</th><td class="r">0.105</td><td class="r">0.693</td><td class="r">2.303</td><td class="r">4.605</td></tr></tbody></table>
<p>For these four examples the likelihood is \(0.9 \times 0.5 \times 0.1 \times 0.01 = 4.5 \times 10^{-4}\), and \(-\ln\,(4.5 \times 10^{-4}) = 7.706\), the sum of the four losses. A sure, wrong prediction costs without limit as \(p \to 0\). With the sigmoid, the gradient with respect to the logit is \(p - 1\): at \(z = {-5}\) it is −0.993, 75 times the squared-error gradient there.</p>`,
    fig: {
      type: "plot", width: WIDE, height: NARROW ? 300 : 320, caption: "Cross-entropy \\(-\\ln\\, p\\) and, in red, squared error \\((1 - p)^2\\), as functions of the probability of the right answer", x: [0, 1], y: [0, 5], xlabel: "p of the correct class", ylabel: "loss",
      params: [{ name: "p", label: "p of the correct class", min: 0.005, max: 1, step: 0.005, value: 0.1, fmt: "f3" }],
      fns: [{ f: (x) => -Math.log(x), cls: "ink" }, { f: (x) => (1 - x) ** 2, cls: "s2" }],
      extra: (p) => ({ dots: [0.9, 0.5, 0.1, 0.01].map((q) => ({ x: q, y: -Math.log(q), r: 3.5, cls: "ink", tip: `−ln ${q} = ${F(-Math.log(q), 3)}` })).concat([{ x: p.p, y: Math.min(-Math.log(p.p), 5), r: 6 }]) }),
      readouts: [{ label: "cross-entropy \\(-\\ln\\, p\\)", f: (p) => -Math.log(p.p), fmt: "f3" }, { label: "squared error \\((1 - p)^2\\)", f: (p) => (1 - p.p) ** 2, fmt: "f3" }],
    },
    check: {
      q: "After <code>WHERE</code> our model gives <code>id</code> probability 0.3460, and <code>id</code> is what followed. What is the cross-entropy loss on this prediction?",
      options: ["0.346", "1.061", "0.654", "−1.061"],
      answer: 1,
      why: "\\(-\\ln\\, 0.3460 = 1.061\\). 0.654 is \\(1 - p\\), a miss measured the squared-error way; −1.061 drops the minus sign, which would make better predictions cost more.",
    },
    terms: [
      { term: "Likelihood", def: "The probability a model assigns to the observed data: the product of the probabilities it gave each example's correct answer." },
      { term: "Negative log-likelihood", def: "\\(-\\ln\\,\\) of the likelihood: the sum of \\(-\\ln\\, p\\) over examples." },
      { term: "Cross-entropy loss", def: "The average of \\(-\\ln\\, p_{\\text{correct}}\\) over examples; the loss our model and every LLM is trained on." },
    ],
  });

  Deck.slide({
    id: "learn-pminusy",
    title: "The gradient of softmax and cross-entropy is p − y",
    example: "tokens ids focus:4 target:5",
    body: String.raw`<p>For one example with correct class \(c\), the loss as a function of the logits is \(L = {-\ln\, p_c}\). Its gradient with respect to every logit has the simplest possible form, derived in the plate:</p>
\[\frac{\partial L}{\partial z_i} = p_i - y_i\]
<p>where \(\mathbf y\) is the one-hot target. The components sum to 0, because \(\sum_i p_i = 1 = \sum_i y_i\). If the logits were themselves the parameters, as on the next sheet, one step of gradient descent would raise the correct logit by \(\eta\,(1 - p_c)\) and lower every other logit \(z_i\) by \(\eta\, p_i\): the more probability a wrong class holds, the harder it is pushed down. In a network the logits are computed from the weights, and this vector is passed back to them (Part 3).</p>
<p>On our model after <code>WHERE</code>, with target <code>id</code>, the gradient is \(0.3460 - 1 = {-0.6540}\) for <code>id</code>, +0.3448 for <code>email</code>, +0.3072 for <code>age</code>, +0.0012 for <code>name</code> and +0.0008 in total for the other 109 tokens.</p>`,
    fig: {
      type: "steps", caption: "Deriving \\(\\partial L / \\partial z_i\\) for softmax followed by cross-entropy", start: 0,
      steps: [
        { html: String.raw`\[L = {-\ln\, p_c} = {-\ln\, \frac{e^{z_c}}{\sum_j e^{z_j}}} = {-z_c} + \ln\, \sum_j e^{z_j}\]`, note: "the log of a quotient" },
        { html: String.raw`\[\frac{\partial}{\partial z_i}\left({-z_c}\right) = \begin{cases} {-1} & i = c \\ 0 & i \ne c \end{cases} \;=\; {-y_i}\]` },
        { html: String.raw`\[\frac{\partial}{\partial z_i} \ln\, \sum_j e^{z_j} = \frac{1}{\sum_j e^{z_j}} \cdot e^{z_i} = p_i\]`, note: "chain rule: 1/S times dS/dz_i" },
        { html: String.raw`\[\frac{\partial L}{\partial z_i} = p_i - y_i\]` },
        { html: String.raw`\[\text{id: } 0.3460 - 1 = {-0.6540}, \quad \text{email: } 0.3448, \quad \text{age: } 0.3072\]`, note: "the real model" },
      ],
    },
  });

  Deck.slide({
    id: "learn-gradient",
    title: "Gradient descent on four logits learns the data's frequencies",
    example: "tokens ids focus:4 target:5",
    body: String.raw`<p>Take only the four logits for <code>id</code>, <code>name</code>, <code>email</code> and <code>age</code>, start them all at 0 (probability 0.250 each, softmax over these four), and train them with the rule \(z \leftarrow z - \eta\,(p - y)\). Each step draws the true next token from the grammar: <code>id</code>, <code>email</code> or <code>age</code> with probability 1/3 each, never <code>name</code>.</p>
<p>From the reset, step 1 draws <code>email</code>. The gradient \(p - y\) for (<code>id</code>, <code>name</code>, <code>email</code>, <code>age</code>) is (0.25, 0.25, −0.75, 0.25), so at learning rate 0.5 the <code>email</code> logit rises by 0.375 and the others fall by 0.125. <code>name</code> is never the target, so its logit falls on every step: after 100 steps at rate 0.5 its probability is 0.016.</p>
<p>The other three do not settle at exactly 1/3: each step uses one sample, a minibatch of 1 (chapter 2.3). After 100 steps they are 0.227, 0.401 and 0.356, and the expected loss is 1.143 against the floor \(\ln\, 3 = 1.099\). At rate 5 they chase the last sample: after each of the first 300 steps, the token just drawn holds at least 0.97.</p>`,
    fig: { type: "tinylm", widget: "gradient", caption: "Four logits trained one sampled token at a time; the chart is the expected loss under the grammar's 1/3, 0, 1/3, 1/3" },
    check: {
      q: "At step 1 the gradient for the logit of <code>name</code> is +0.25. What does a step at learning rate 0.5 do to that logit?",
      options: ["Raises it by 0.125", "Lowers it by 0.125", "Lowers it by 0.25", "Leaves it alone, since <code>name</code> was not drawn"],
      answer: 1,
      why: "\\(z \\leftarrow z - \\eta\\,(p - y) = 0 - 0.5 \\times 0.25 = {-0.125}\\). Every class that is not the target is pushed down by \\(\\eta\\) times its own probability, whether or not it was drawn.",
    },
  });

  // ------------------------------------------------------------ 2.5
  Deck.chapter({ title: "Information" });

  Deck.slide({
    id: "learn-entropy",
    title: "Entropy is the average surprise of a distribution",
    body: String.raw`<p>The <b>surprise</b> of an outcome with probability \(p\) is \(-\ln\, p\): 0 for a certain outcome, large for a rare one. The <b>entropy</b> of a distribution is its average surprise, each outcome weighted by its probability:</p>
\[H(p) = {-\sum_i p_i \ln\, p_i}\]
<ul><li>A fair coin: \(H = {-2} \times 0.5\, \ln\, 0.5 = \ln\, 2 = 0.693\).</li>
<li>A fair 9-sided die, which is the digit after <code>=</code> in our grammar: \(H = \ln\, 9 = 2.197\).</li>
<li>A certain outcome: \(H = {-1} \cdot \ln\, 1 = 0\), as after <code>SELECT name</code>, where <code>FROM</code> must follow.</li></ul>
<p>With the natural log the unit is the <b>nat</b>; with \(\log_2\) it is the <b>bit</b>, and a fair coin is 1 bit = 0.693 nats. A uniform distribution over \(N\) outcomes has entropy \(\ln\, N\), the largest any distribution on \(N\) outcomes can have. For our 113 tokens that is \(\ln\, 113 = 4.727\).</p>`,
    fig: {
      type: "plot", width: WIDE, height: NARROW ? 300 : 320, caption: "Entropy of a coin that lands heads with probability \\(p\\)", x: [0, 1], y: [0, 0.8], xlabel: "p(heads)", ylabel: "entropy (nats)", yticks: [0, 0.2, 0.4, 0.6, 0.8],
      params: [{ name: "p", label: "p(heads)", min: 0, max: 1, step: 0.01, value: 0.5 }],
      fns: [{ f: (x) => (x <= 0 || x >= 1 ? 0 : -x * Math.log(x) - (1 - x) * Math.log(1 - x)), samples: 400 }],
      hlines: [{ y: Math.LN2, label: "ln 2 = 0.693" }],
      extra: (p) => { const H = p.p <= 0 || p.p >= 1 ? 0 : -p.p * Math.log(p.p) - (1 - p.p) * Math.log(1 - p.p); return { dots: [{ x: p.p, y: H, r: 6 }] }; },
      readouts: [
        { label: "entropy, nats", f: (p) => (p.p <= 0 || p.p >= 1 ? 0 : -p.p * Math.log(p.p) - (1 - p.p) * Math.log(1 - p.p)), fmt: "f3" },
        { label: "entropy, bits", f: (p) => (p.p <= 0 || p.p >= 1 ? 0 : (-p.p * Math.log(p.p) - (1 - p.p) * Math.log(1 - p.p)) / Math.LN2), fmt: "f3" },
      ],
    },
    terms: [
      { term: "Surprise", def: "\\(-\\ln\\, p\\) for an outcome of probability \\(p\\): 0 if certain, large if rare." },
      { term: "Entropy", def: "The average surprise of a distribution, \\(H(p) = {-\\sum_i p_i \\ln\\, p_i}\\): the lowest average loss any model can reach on data drawn from it." },
      { term: "Nat", def: "The unit of information with the natural log; 1 bit = ln 2 = 0.693 nats." },
      { term: "Bit", def: "The unit of information with \\(\\log_2\\): the entropy of a fair coin." },
    ],
  });

  Deck.slide({
    id: "learn-kl",
    title: "Cross-entropy is entropy plus the KL divergence",
    example: "tokens ids focus:4 target:5",
    note: "The letters change here: in chapter 2.4, <b>p</b> was the model's output. From now on <b>p</b> is the data's distribution and <b>q</b> the model's.",
    body: String.raw`<p>Let \(p\) be the true distribution of the next token and \(q\) the model's. The model's expected loss is the <b>cross-entropy</b> \(H(p, q) = {-\sum_i p_i \ln\, q_i}\): surprise measured with \(q\), averaged over outcomes drawn from \(p\). It splits into two parts:</p>
\[H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q), \qquad D_{\mathrm{KL}}(p \,\|\, q) = \sum_i p_i \ln\, \frac{p_i}{q_i} \ge 0\]
<p>The <b>KL divergence</b> is 0 only when \(q = p\), so the expected loss can never go below the entropy of the data. The entropy is the floor; the KL divergence is the part the model can still remove.</p>
<p>After <code>SELECT name FROM users WHERE</code>, \(p\) gives 1/3 each to <code>id</code>, <code>email</code> and <code>age</code>, so \(H(p) = \ln\, 3 = 1.0986\). Our model's \(q\) of 0.3460, 0.3448 and 0.3072 gives \(H(p, q) = 1.1021\) and \(D_{\mathrm{KL}} = 0.0035\) nats. The plate starts at those values; move them.</p>`,
    fig: {
      type: "calc", caption: "The model's probabilities after <code>WHERE</code> against the grammar's 1/3, 1/3, 1/3; tokens not listed get the remainder",
      inputs: [
        { id: "qi", label: "q(id)", min: 0, max: 1, step: 0.0001, value: 0.3460, fmt: "f4" }, { id: "qe", label: "q(email)", min: 0, max: 1, step: 0.0001, value: 0.3448, fmt: "f4" },
        { id: "qa", label: "q(age)", min: 0, max: 1, step: 0.0001, value: 0.3072, fmt: "f4" }, { id: "qn", label: "q(name)", min: 0, max: 1, step: 0.0001, value: 0.0012, fmt: "f4" },
      ],
      outputs: [
        { label: "\\(H(p)\\), the floor", f: () => Math.log(3), fmt: "f4" },
        { label: "\\(H(p, q)\\)", f: (v) => (v.qi + v.qe + v.qa + v.qn > 1 + 1e-9 ? NaN : -(Math.log(v.qi) + Math.log(v.qe) + Math.log(v.qa)) / 3), fmt: "f4" },
        { label: "\\(D_{\\mathrm{KL}}(p \\,\\|\\, q)\\)", f: (v) => (v.qi + v.qe + v.qa + v.qn > 1 + 1e-9 ? NaN : -(Math.log(v.qi) + Math.log(v.qe) + Math.log(v.qa)) / 3 - Math.log(3)), fmt: "f4" },
        { label: "left for other tokens", f: (v) => 1 - v.qi - v.qe - v.qa - v.qn, fmt: "f4" },
      ],
      note: (v) => (v.qi + v.qe + v.qa + v.qn > 1 + 1e-9 ? "The four probabilities add to more than 1; lower one of them." : `H(p, q) = −(ln ${UI.format(v.qi, "f4")} + ln ${UI.format(v.qe, "f4")} + ln ${UI.format(v.qa, "f4")}) / 3. The probability on <code>name</code> enters only by taking mass from the other three.`),
    },
    terms: [
      { term: "Cross-entropy H(p, q)", def: "\\(-\\sum_i p_i \\ln\\, q_i\\): the expected loss of a model \\(q\\) on data from \\(p\\). Equals \\(H(p)\\) plus the KL divergence." },
      { term: "KL divergence", def: "\\(D_{\\mathrm{KL}}(p\\,\\|\\,q) = \\sum_i p_i \\ln\\, (p_i/q_i) \\ge 0\\): how much more loss a model \\(q\\) pays than the floor \\(H(p)\\)." },
    ],
  });

  Deck.slide({
    id: "learn-floor",
    title: "Our grammar's entropy sets a floor of 0.6937 nats per token",
    example: "tokens",
    body: String.raw`<p>Each query gives the model 8 predictions. For our grammar the true next-token distribution is known at every position, so its entropy, the lowest possible average loss there, can be computed exactly. Averaged over the 270 queries with their probabilities:</p>
<ul><li>After <code>SELECT</code>: 9 possible columns. <code>id</code> is in every table and has probability 5/18; each of the others has 1/12 or 1/9. Entropy 2.0865 nats.</li>
<li>After <code>FROM</code>: the table is certain unless the column is <code>id</code>, which leaves 3 tables: 0.3025.</li>
<li>After <code>WHERE</code>: 3 remaining columns, or 2 for <code>products</code>: 0.9635.</li>
<li>After <code>=</code>: 9 equally likely digits, \(\ln\, 9 = 2.1972\).</li>
<li>The other four positions are certain: 0.</li></ul>
<p>The mean over the 8 positions is \(5.5497 / 8 = 0.6937\) nats per token. No model can do better on this data. Our trained model reaches 0.6942 (Part 4 shows it position by position).</p>`,
    fig: {
      type: "table", caption: "The grammar's entropy at each of the 8 positions: averages over all 270 queries, each weighted by its probability", dense: true,
      cols: ["Position", "After", "Predicts", "Entropy (nats)"], align: ["r", "l", "l", "r"],
      rows: [0, 1, 2, 3, 4, 5, 6, 7].map((i) => [String(i), ROLE_IN[i], ROLE_NEXT[i], UI.format(M.data.floor[i], "f4")]),
      note: "Sum 5.5497 nats over 8 predictions: a floor of 0.6937 nats per token. Italic words stand for whichever column, table or digit the query has.",
    },
  });

  Deck.slide({
    id: "learn-perplexity",
    title: "Perplexity converts a loss into a number of equally likely choices",
    example: "tokens",
    body: String.raw`<p><b>Perplexity</b> is \(e\) raised to the average loss in nats. A uniform distribution over \(N\) outcomes has entropy \(\ln\, N\) and perplexity \(N\), so a perplexity reads as “as uncertain as a choice among this many equally likely options”.</p>
<p>After <code>=</code> the grammar's entropy is \(\ln\, 9\), perplexity exactly 9: nine equally likely digits. After <code>SELECT</code> it is \(e^{2.0865} = 8.06\): nine columns that are not equally likely. After <code>WHERE</code>, 2.62. Over all 8 positions the floor of 0.6937 nats gives \(e^{0.6937} = 2.001\): on average the grammar leaves as much uncertainty per token as a fair coin. The overall perplexity is the geometric mean of the per-position ones.</p>
<p>Training moved our model from a near-uniform guess to the floor. Its loss was 4.722 at step 0, perplexity 112.4 (a uniform guess over 113 tokens scores 113), and 0.6942 at step 6,000, perplexity 2.002. Drag the slider through training.</p>`,
    fig: { type: "learn-ppl", caption: "Perplexity at each position, averaged over all 270 queries, during training; the red ticks are the grammar's floor" },
    check: {
      q: "After <code>=</code> the floor is \\(\\ln\\, 9 = 2.197\\) nats. A model averages 2.303 nats there. Which statement is right?",
      options: ["Its perplexity there is 10: as uncertain as a uniform choice among 10 options, 0.105 nats above the floor", "Its perplexity there is 2.303", "It beats the floor", "Its KL divergence from the data there is 2.303 nats"],
      answer: 0,
      why: "\\(e^{2.303} = 10.0 > 9\\), and the gap to the floor is the KL divergence, \\(\\ln\\, 10 - \\ln\\, 9 = 0.105\\) nats, not the loss itself; no model can average below the floor. The extra 0.105 nats can come from probability on tokens that cannot follow, or from uneven probability on the nine digits: 0.304 on one digit and 0.087 on each of the others also averages \\(\\ln\\, 10\\).",
    },
    terms: [{ term: "Perplexity", def: "\\(e^{\\text{loss}}\\) for a loss in nats: the number of equally likely choices that would give the same loss." }],
  });

  // ------------------------------------------------------------ 2.6
  Deck.chapter({ title: "Generalization" });

  Deck.slide({
    id: "learn-heldout",
    title: "Held-out data measures what the model learned",
    body: String.raw`<p>The <b>training set</b> is the data the gradient is computed on. A <b>held-out set</b> is data from the same source that training never touches. The loss that matters is the loss on new data, and only held-out data can measure it.</p>
<p>The data for this chapter: 10 training points \(y = \sin(\pi x) + \text{noise}\), with \(x\) evenly spaced from −1 to 1 and noise of standard deviation 0.2, and 101 held-out points from the same process. Even the true function \(\sin(\pi x)\) has MSE 0.052 on the training points and 0.033 on the held-out ones: that is the noise, which no model can predict.</p>
<p>A model can drive the training loss to zero without learning anything general. The memorizer in the plate returns the \(y\) of the nearest training point: training MSE 0, held-out MSE 0.103, three times the true function's. The <b>generalization gap</b>, held-out loss minus training loss, is 0.103 for the memorizer and −0.019 for the true function.</p>`,
    fig: { type: "learn-poly", mode: "split", model: "nn", caption: "Ten training points (filled), 101 held-out points (hollow) and a model's predictions" },
    terms: [
      { term: "Training set", def: "The examples whose loss gradient drives the parameter updates." },
      { term: "Held-out set", def: "Examples from the same source that training never uses, kept to measure the loss on new data." },
      { term: "Generalization gap", def: "Held-out loss minus training loss." },
    ],
  });

  Deck.slide({
    id: "learn-overfit",
    title: "A degree-9 polynomial memorizes 10 points",
    body: String.raw`<p>A polynomial of degree \(d\), \(\hat y = c_0 + c_1 x + \dots + c_d x^d\), has \(d + 1\) parameters and is still linear in them, so its least-squares fit has a formula, the normal equations of chapter 2.2 with more unknowns. Fit degrees 0 to 9 to the 10 training points:</p>
<ul><li>Degree 1: training MSE 0.274, held-out 0.257. A line cannot bend.</li>
<li>Degree 3: 0.064 and 0.037, close to the noise.</li>
<li>Degree 9: 10 parameters for 10 points, so the curve passes through every point. Training MSE 0, held-out 0.556, 15 times worse than degree 3.</li></ul>
<p>This is <b>overfitting</b>: the extra parameters fit the noise in the training points, and the curve swings between them to do so, with coefficients up to 157 in size and values up to 2.5 in size, where no data point exceeds 1.33. More parameters lowered the training loss every time; the held-out loss fell, then rose.</p>`,
    fig: { type: "learn-poly", mode: "degree", degree: 9, caption: "Least-squares polynomial fits of degree 0 to 9" },
    terms: [{ term: "Overfitting", def: "Fitting the noise in the training data: training loss keeps falling while held-out loss rises." }],
  });

  Deck.slide({
    id: "learn-underfit",
    title: "Too simple a model underfits",
    body: String.raw`<p><b>Underfitting</b> is the opposite failure: the model cannot represent the pattern, so the training loss itself stays high. Degrees 0 to 2 cannot make the two bends of \(\sin(\pi x)\) on [−1, 1]: their training MSE is 0.274 or more, about 7 times the noise variance of 0.04, and their held-out loss is just as bad.</p>
<p>Plotting both losses against degree gives the usual picture. The training loss falls with every added parameter. The held-out loss falls while the model is too simple, is lowest at degree 3 (0.037), changes little up to degree 6, then climbs as the model fits the noise, to 0.556 at degree 9.</p>
<p>The two numbers give the diagnosis. High training loss means underfitting: add capacity. Low training loss with much higher held-out loss means overfitting: add data, reduce capacity, stop earlier or regularize (the next sheets).</p>`,
    fig: {
      type: "chart", caption: "Training and held-out MSE against the degree of the fitted polynomial; the dashed line is the held-out MSE of sin(πx) itself, 0.033",
      chart: { width: WIDE, height: 300, margin: { l: 50, r: 16, t: 12, b: 38 } },
      x: { domain: [0, 9], ticks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label: "degree" }, y: { domain: [0, 0.6], ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], label: "MSE", fmt: (v) => UI.format(v, "f1") },
      series: [{ points: DEGREE_CURVE.map((q) => [q[0], q[1]]), cls: "s1" }, { points: DEGREE_CURVE.map((q) => [q[0], q[2]]), cls: "s2" }],
      dots: DEGREE_CURVE.flatMap((q) => [{ x: q[0], y: q[1], r: 4, cls: "s1", tip: `degree ${q[0]}: training ${F(q[1], 4)}` }, { x: q[0], y: q[2], r: 4, cls: "s2", tip: `degree ${q[0]}: held-out ${F(q[2], 4)}` }]),
      hlines: [{ y: 0.0331 }],
      legend: [["s1", "training MSE"], ["s2", "held-out MSE"]],
    },
    terms: [{ term: "Underfitting", def: "A model too simple for the pattern: its training loss stays high." }],
  });

  Deck.slide({
    id: "learn-early",
    title: "A validation set tells you when to stop",
    body: String.raw`<p>Held-out data used to make decisions (the degree, the learning rate, when to stop) is a <b>validation set</b>. Choosing by it makes its loss slightly optimistic, so a final <b>test set</b> is kept aside and used once.</p>
<p>Gradient descent itself can overfit. Train the degree-9 polynomial from all-zero coefficients with \(\eta = 0.3\). The held-out loss falls to its lowest value, 0.041, around step 500, then climbs while the training loss keeps falling; by \(10^{9}\) steps descent has reached the fit through every training point (held-out 0.556).</p>
<p><b>Early stopping</b> keeps the parameters from the step with the lowest validation loss: here 0.041, nearly as good as the degree-3 fit's 0.037. It works because of where the wiggles live. The fit through every point gets most of its size from the two flattest directions of the loss: components of 198 and 125, at curvatures \(1.5 \times 10^{-7}\) and \(2.9 \times 10^{-6}\). Each step removes a fraction \(\eta h\) of what remains along a direction of curvature \(h\), so these take about \(1/(\eta h) = 2.3 \times 10^{7}\) and \(1.2 \times 10^{6}\) steps. At step 500 the coefficients have length 3.7, against 234.8.</p>`,
    fig: { type: "learn-poly", mode: "early", logt: 2.7, caption: "Gradient descent on the degree-9 polynomial: the fit after \\(t\\) steps, and both losses against \\(t\\)" },
    terms: [
      { term: "Validation set", def: "Held-out data used to choose hyperparameters and when to stop." },
      { term: "Test set", def: "Held-out data used once, at the end, to report the final loss." },
      { term: "Early stopping", def: "Keeping the parameters from the training step with the lowest validation loss." },
    ],
  });

  Deck.slide({
    id: "learn-regularize",
    title: "L2 regularization pulls the parameters toward zero",
    body: String.raw`<p><b>Regularization</b> changes training to prefer simpler parameters. <b>L2 regularization</b> adds \(\lambda\) times the sum of the squared parameters to the loss:</p>
\[L_{\text{reg}}(\mathbf c) = \frac{1}{n} \sum_{i} (\hat y_i - y_i)^2 + \lambda \sum_{j} c_j^2\]
<p>Its gradient adds \(2\lambda c_j\) to each parameter's gradient, so under plain gradient descent every step also multiplies each \(c_j\) by \(1 - 2\eta\lambda\). Shrinking the weights by a fixed factor each step is called <b>weight decay</b>. The two are the same under plain gradient descent but not under Adam, which rescales each parameter's gradient, the L2 term included; AdamW applies the decay directly (Part 3). On the degree-9 fit, \(\lambda = 0\) gives coefficients of length \(\lVert \mathbf c \rVert = 234.8\) and held-out MSE 0.556. \(\lambda = 10^{-3}\) shrinks them to 4.4; the training MSE rises to 0.039 and the held-out MSE falls to 0.045. Too much, \(\lambda = 1\), underfits: held-out 0.416.</p>
<p>The penalty costs a little training fit and buys a smoother curve. \(\lambda\) is a hyperparameter, chosen on the validation set.</p>`,
    fig: { type: "learn-poly", mode: "ridge", loglam: -3, caption: "Degree-9 fit with L2 regularization of strength \\(\\lambda\\); the left end of the slider is \\(\\lambda = 0\\)" },
    check: {
      q: "On the degree-9 fit, \\(\\lambda = 10^{-3}\\) raised the training MSE from 0 to 0.039 and lowered the held-out MSE from 0.556 to 0.045. What does that show?",
      options: ["Regularization trades a little training fit for much better generalization", "The regularized model is worse, because its training loss rose", "λ should be raised until the training loss is 0 again", "The held-out set must be mislabeled"],
      answer: 0,
      why: "The goal is low held-out loss. Raising \\(\\lambda\\) only raises the training loss further (at \\(\\lambda = 1\\) it is 0.353); the useful range here is around \\(10^{-3}\\).",
    },
    src: ["Loshchilov & Hutter, Decoupled Weight Decay Regularization, ICLR 2019"],
    terms: [
      { term: "Regularization", def: "Any change to training that prefers simpler parameters to reduce overfitting." },
      { term: "L2 regularization", def: "Adding \\(\\lambda \\sum_j w_j^2\\) to the loss. Under plain gradient descent it multiplies each weight by \\(1 - 2\\eta\\lambda\\) per step." },
      { term: "Weight decay", def: "Multiplying every weight by a factor slightly below 1 at each step. The same as L2 regularization under plain gradient descent, but not under Adam; AdamW applies it directly." },
    ],
  });

  Deck.slide({
    id: "learn-onepass",
    title: "Loss on data not yet trained on is a held-out loss",
    body: String.raw`<p>Large language models overfit little for a reason other than regularization: most of their data is seen once. GPT-3 trained on 300 billion tokens. It drew 180 billion of them from its largest source, 410 billion tokens of filtered web text, which is 0.44 of one pass. Smaller, higher-quality sources were repeated up to 3.4 times, which its authors describe as accepting a small amount of overfitting.</p>
<p>When every batch is new data, the loss on a batch measured before the update is an unbiased estimate of the held-out loss, because the model has not been trained on that batch. In expectation the training curve is then the held-out curve, and no systematic gap opens.</p>
<p>Our model is the opposite case: its 768,000 training queries were drawn from only 270 distinct queries, so it saw each <code>users</code> or <code>orders</code> query about 2,370 times and each <code>products</code> query about 4,741 times (2,844 on average). Part 5 tests what that repetition does, with a copy of the model trained without 27 of the queries.</p>`,
    fig: {
      type: "table", caption: "GPT-3's training data: how many times each source was used during 300 billion tokens",
      cols: ["Source", "Tokens (billions)", "Share of the 300B", "Passes"], align: ["l", "r", "r", "r"],
      rows: [
        ["Common Crawl (filtered)", "410", "60%", "0.44"], ["WebText2", "19", "22%", "2.9"], ["Books1", "12", "8%", "1.9"],
        ["Books2", "55", "8%", "0.43"], ["Wikipedia", "3", "3%", "3.4"],
      ],
      highlight: [0],
      note: "From Table 2.2 of the GPT-3 paper. 60% of 300 billion is 180 billion tokens, 0.44 of Common Crawl's 410 billion.",
    },
    src: ["Brown et al., Language Models are Few-Shot Learners, 2020 (Section 2.2, Table 2.2)"],
    check: {
      q: "A model trains for one pass over fresh data. Over its last 100 batches, with each batch's loss measured just before the update on that batch, the average loss is 2.10. What do you expect on a held-out set from the same source?",
      options: ["About 2.10", "Much higher, because training loss always underestimates", "Much lower, because the model keeps improving", "It cannot be estimated without a held-out run"],
      answer: 0,
      why: "Each batch was new to the model when its loss was measured, so that loss is already a held-out measurement. A systematic gap can open only when data repeats, as for our model, which saw each query thousands of times.",
    },
  });
})();
