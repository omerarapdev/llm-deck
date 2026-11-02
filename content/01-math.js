/* Part 1 — The math toolkit: functions, derivatives, the chain rule, gradients,
 * vectors, matrices, probability, exponentials and logarithms. Every number the
 * text states is recomputed in tools/verify/part01.mjs. */
(function () {
  "use strict";
  const UI = window.UI;
  const { h, s, clear, chip, slider, segmented, button, chart } = UI;
  const M = window.TinyLM;
  const rich = (str) => UI.rich(str);

  // ------------------------------------------------------------ number helpers
  // fixed decimals with a true minus sign; values that round to zero print as 0
  function fix(v, d = 3) {
    if (Number.isNaN(v)) return "NaN";
    if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
    const t = (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : v).toFixed(d);
    return t.replace(/^-/, "−");
  }
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const sup = (n) => String(n).split("").map((c) => SUP[c]).join("");
  // "10⁻⁸" for tick labels on log axes
  const pow10 = (v) => `10${sup(Math.round(Math.log10(v)))}`;
  // "3.18 × 10⁻³"
  function sci(v, d = 2) {
    if (v === 0) return "0";
    if (!Number.isFinite(v)) return fix(v);
    let e = Math.floor(Math.log10(Math.abs(v)));
    let m = v / Math.pow(10, e);
    if (Math.abs(+m.toFixed(d)) >= 10) { m /= 10; e += 1; }
    return `${fix(m, d)} × 10${sup(e)}`;
  }
  const tickMinus = (v) => String(v).replace("-", "−");
  const tokText = (t) => t.trim();

  // ------------------------------------------------------------ plot helpers
  // A <g> clipped to the plot area, for marks drawn in a plot's `after` hook.
  function clipG(svg, box) {
    const id = UI.nextId("mclip");
    svg.appendChild(s("clipPath", { id }, s("rect", { x: box.m.l, y: box.m.t, width: box.W - box.m.l - box.m.r, height: box.H - box.m.t - box.m.b })));
    const g = s("g", { "clip-path": `url(#${id})` });
    svg.appendChild(g);
    return g;
  }
  const inside = (box, px, py) => px >= box.m.l - 0.5 && px <= box.W - box.m.r + 0.5 && py >= box.m.t - 0.5 && py <= box.H - box.m.b + 0.5;

  /* A draggable point for `plot` figures. Dragging moves the slider with index
   * `idx` (in the order of `params`), so the slider stays the single source of
   * truth and keyboard users keep the slider. */
  function dragHandle(svg, X, Y, box, xdom, idx, px, py) {
    if (!inside(box, X(px), Y(py))) return;
    const hnd = s("circle", { cx: X(px), cy: Y(py), r: 13, class: "math-handle" });
    svg.appendChild(hnd);
    if (!svg._mathDrag) {
      const st = (svg._mathDrag = { on: false, idx: 0 });
      svg.setAttribute("data-noswipe", "");
      svg.addEventListener("pointermove", (e) => {
        if (!st.on) return;
        const r = svg.getBoundingClientRect();
        const vx = ((e.clientX - r.left) / r.width) * box.W;
        const v = xdom[0] + ((vx - box.m.l) / (box.W - box.m.l - box.m.r)) * (xdom[1] - xdom[0]);
        const body = svg.closest(".fig-body");
        const input = body && body.querySelectorAll(".f-ctl input[type=range]")[st.idx];
        if (!input) return;
        input.value = String(v);
        input.dispatchEvent(new Event("input"));
        e.preventDefault();
      });
      const stop = () => { st.on = false; };
      svg.addEventListener("pointerup", stop);
      svg.addEventListener("pointercancel", stop);
    }
    hnd.addEventListener("pointerdown", (e) => {
      svg._mathDrag.on = true;
      svg._mathDrag.idx = idx;
      try { svg.setPointerCapture(e.pointerId); } catch (err) { /* not capturable */ }
      e.preventDefault();
    });
  }
  // label text with a halo, kept inside the plot area
  function plabel(g, box, x, y, text, anchor = "start", cls = "") {
    const w = text.length * 6.6;
    let xx = x;
    if (anchor === "start") xx = Math.min(x, box.W - box.m.r - w - 2);
    if (anchor === "end") xx = Math.max(x, box.m.l + w + 2);
    const yy = Math.max(box.m.t + 11, Math.min(y, box.H - box.m.b - 4));
    g.appendChild(s("text", { x: xx, y: yy, class: "math-lab " + cls, "text-anchor": anchor }, text));
  }
  // a marked point with dashed guides down to the x axis and across to the y axis
  function guides(g, X, Y, box, x, y) {
    const yb = box.H - box.m.b, xl = box.m.l;
    g.appendChild(s("path", { d: `M${X(x)},${yb} V${Y(y)} H${xl}`, class: "math-guide" }));
  }

  // ============================================================ custom figures

  // Small multiples: one mini chart per function, with marked values and a note.
  Deck.fig("math-gallery", (el, sp) => {
    const grid = h("div", { class: "math-gal" });
    for (const it of sp.items) {
      const cell = h("div", { class: "math-gal-c" });
      cell.appendChild(h("div", { class: "math-gal-t" }, rich(it.title)));
      const host = h("div", { class: "math-gal-plot" });
      cell.appendChild(host);
      const pts = [];
      const n = 200;
      for (let i = 0; i <= n; i++) {
        const x = it.x[0] + ((it.x[1] - it.x[0]) * i) / n;
        const y = it.f(x);
        if (Number.isFinite(y)) pts.push([x, y]);
      }
      chart(host, {
        width: 230, height: 150, margin: { l: 30, r: 8, t: 8, b: 22 }, aria: it.aria || "function plot",
        x: { domain: it.x, ticks: it.xt, fmt: tickMinus }, y: { domain: it.y, ticks: it.yt, fmt: tickMinus },
        series: [{ points: pts, cls: "s1" }],
        hlines: (it.hl || []).map((y) => ({ y, cls: "" })),
        dots: it.marks.map(([x, y, tip]) => ({ x, y, r: 3.8, cls: "ink", tip })),
      });
      cell.appendChild(h("p", { class: "math-gal-n" }, rich(it.note)));
      grid.appendChild(cell);
    }
    el.appendChild(grid);
  });

  /* A pipeline of functions shown as a table: each stage's rule, its value at the
   * current input, and (optionally) its local derivative; the product of local
   * derivatives is compared with a central difference of the whole chain. */
  Deck.fig("math-pipe", (el, sp) => {
    const orders = sp.orders || [{ label: "", stages: sp.stages }];
    let k = 0, x = sp.x.value;
    const dp = sp.dp ?? 3;
    const table = h("table", { class: "f-table math-pipe" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.appendChild(h("div", { class: "scroll-x" }, table));
    el.appendChild(expl);
    const ctl = h("div", { class: "f-ctl sliders" });
    ctl.appendChild(slider({ label: sp.x.label || "input x", min: sp.x.min, max: sp.x.max, step: sp.x.step, value: x, format: (v) => fix(v, sp.x.dp ?? 2), onInput: (v) => { x = v; draw(); } }));
    if (orders.length > 1) ctl.appendChild(segmented({ label: "order", value: 0, options: orders.map((o, i) => ({ label: o.label, value: i })), onChange: (v) => { k = v; draw(); } }));
    el.appendChild(ctl);
    function draw() {
      const st = orders[k].stages;
      const vals = [x];
      st.forEach((g) => vals.push(g.f(vals[vals.length - 1])));
      clear(table);
      const cols = ["", "rule", "value"].concat(sp.slopes ? ["local derivative"] : []);
      table.appendChild(h("thead", null, h("tr", null, cols.map((c, j) => h("th", { class: j >= 2 ? "r" : "" }, c)))));
      const tb = h("tbody");
      tb.appendChild(h("tr", null, h("td", null, rich(String.raw`\(x\)`)), h("td", null, "input"), h("td", { class: "r" }, fix(x, sp.x.dp ?? 2)), sp.slopes ? h("td", null, "") : null));
      st.forEach((g, i) => {
        const prev = i === 0 ? "x" : st[i - 1].v;
        const cells = [h("td", null, rich(`\\(${g.v}\\)`)), h("td", null, rich(`\\(${g.tex}\\)`)), h("td", { class: "r" }, fix(vals[i + 1], g.dp ?? dp))];
        if (sp.slopes) {
          const tex = rich(`\\(d${g.v}/d${prev} = ${g.dtex}\\)`);
          cells.push(h("td", { class: "r" }, tex, g.dconst ? null : " = ", g.dconst ? null : h("b", null, fix(g.df(vals[i]), g.ddp ?? dp))));
        }
        tb.appendChild(h("tr", null, cells));
      });
      table.appendChild(tb);
      const out = st[st.length - 1].v;
      if (sp.slopes) {
        const loc = st.map((g, i) => g.df(vals[i]));
        const prod = loc.reduce((a, b) => a * b, 1);
        const whole = (t) => st.reduce((u, g) => g.f(u), t);
        const hh = 1e-5, num = (whole(x + hh) - whole(x - hh)) / (2 * hh);
        expl.innerHTML = `<b>d${out}/dx</b> = ${loc.map((v, i) => fix(v, st[i].ddp ?? dp)).join(" × ")} = <b>${fix(prod, 4)}</b>. Central difference of the whole chain (h = 10⁻⁵): <b>${fix(num, 4)}</b>.`;
      } else {
        expl.innerHTML = orders[k].explain ? orders[k].explain(x, vals) : "";
      }
    }
    draw();
  });

  /* f(x, y) = x² + 3y² as a wireframe surface and as a contour map; a slider picks
   * the level c that is drawn on both. */
  Deck.fig("math-surface", (el, sp) => {
    const f = (x, y) => x * x + 3 * y * y;
    let c = sp.level ?? 4;
    const XR = 2, YR = 1.2;
    // --- surface: rotate about the vertical axis, tilt, then fit into the box
    const phi = (32 * Math.PI) / 180, el0 = (26 * Math.PI) / 180, zs = 0.36;
    const proj = (x, y, z) => {
      const X = x * Math.cos(phi) + y * Math.sin(phi);
      const D = -x * Math.sin(phi) + y * Math.cos(phi);
      return [X, -(D * Math.sin(el0) + z * zs * Math.cos(el0))];
    };
    const lines = [];
    for (let i = 0; i <= 8; i++) { const y = -YR + (2 * YR * i) / 8; const L = []; for (let j = 0; j <= 40; j++) { const x = -XR + (2 * XR * j) / 40; L.push(proj(x, y, f(x, y))); } lines.push(L); }
    for (let j = 0; j <= 10; j++) { const x = -XR + (2 * XR * j) / 10; const L = []; for (let i = 0; i <= 24; i++) { const y = -YR + (2 * YR * i) / 24; L.push(proj(x, y, f(x, y))); } lines.push(L); }
    // fit the surface, its floor and the axis labels into the box
    const LX = [0, -YR - 0.42], LY = [XR + 0.42, 0]; // where the x and y labels sit on the floor
    const all = lines.flat().concat([[-XR, -YR], [XR, -YR], [XR, YR], [-XR, YR], LX, LY].map(([x, y]) => proj(x, y, 0)), [proj(0, 0, 9.6)]);
    const minX = Math.min(...all.map((p) => p[0])), maxX = Math.max(...all.map((p) => p[0]));
    const minY = Math.min(...all.map((p) => p[1])), maxY = Math.max(...all.map((p) => p[1]));
    const W1 = 320, H1 = 240, pad = 16;
    const k1 = Math.min((W1 - 2 * pad) / (maxX - minX), (H1 - 2 * pad - 8) / (maxY - minY));
    const T = ([X, Y]) => [pad + (X - minX) * k1 + ((W1 - 2 * pad) - (maxX - minX) * k1) / 2, pad + 8 + (Y - minY) * k1];
    const P = (x, y, z) => T(proj(x, y, z));
    const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
    const sv1 = s("svg", { viewBox: `0 0 ${W1} ${H1}`, class: "math-surf-svg", role: "img", "aria-label": "Surface of f(x, y) = x² + 3y² with the cut at height c" });
    // --- contour map, equal scale on both axes
    const W2 = 320, H2 = 240, mx = [-2.45, 2.45], my = [-1.6, 1.6];
    const k2 = (W2 - 16) / (mx[1] - mx[0]);
    const cx = W2 / 2, cy = 14 + ((my[1] - my[0]) * k2) / 2;
    const MX = (x) => cx + x * k2, MY = (y) => cy - y * k2;
    const sv2 = s("svg", { viewBox: `0 0 ${W2} ${H2}`, class: "math-surf-svg", role: "img", "aria-label": "Contour map of f(x, y) = x² + 3y²" });
    const tiles = h("div", { class: "f-tiles" });
    const wrap = h("div", { class: "math-surf" },
      h("div", { class: "math-surf-cell" }, h("p", { class: "math-surf-h" }, "Surface: height f(x, y)"), sv1),
      h("div", { class: "math-surf-cell" }, h("p", { class: "math-surf-h" }, "Contour map: seen from above"), sv2));
    el.appendChild(wrap);
    el.appendChild(h("div", { class: "f-ctl sliders" }, slider({ label: "level c", min: 0.25, max: 4, step: 0.25, value: c, format: (v) => fix(v, 2), onInput: (v) => { c = v; draw(); } })));
    el.appendChild(tiles);
    const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, rich(l)));
    function ellipse(z, n = 120) {
      const pts = [];
      for (let i = 0; i <= n; i++) { const t = (2 * Math.PI * i) / n; pts.push(P(Math.sqrt(z) * Math.cos(t), Math.sqrt(z / 3) * Math.sin(t), z)); }
      return pts;
    }
    function draw() {
      clear(sv1);
      // the cutting plane at height c, then the wireframe, then the cut
      const corners = [[-XR, -YR], [XR, -YR], [XR, YR], [-XR, YR], [-XR, -YR]].map(([x, y]) => P(x, y, c));
      sv1.appendChild(s("path", { d: path(corners) + "Z", class: "math-plane" }));
      lines.forEach((L) => sv1.appendChild(s("path", { d: path(L.map(T)), class: "math-wire" })));
      sv1.appendChild(s("path", { d: path(ellipse(c)), class: "math-lev" }));
      // floor rectangle z = 0 with the x and y directions labelled on its near edges
      const fl = [[-XR, -YR], [XR, -YR], [XR, YR], [-XR, YR], [-XR, -YR]].map(([x, y]) => P(x, y, 0));
      sv1.insertBefore(s("path", { d: path(fl) + "Z", class: "math-floor" }), sv1.firstChild);
      const xa = P(LX[0], LX[1], 0), ya = P(LY[0], LY[1], 0);
      sv1.appendChild(s("text", { x: xa[0], y: xa[1] + 4, class: "math-lab", "text-anchor": "middle" }, "x"));
      sv1.appendChild(s("text", { x: ya[0], y: ya[1] + 4, class: "math-lab", "text-anchor": "middle" }, "y"));
      const o = P(0, 0, 0), top = P(0, 0, 9.4);
      sv1.appendChild(s("path", { d: `M${o[0]},${o[1]} L${top[0]},${top[1]}`, class: "math-axis3" }));
      sv1.appendChild(s("text", { x: top[0] + 5, y: top[1] + 4, class: "math-lab" }, "f ↑"));
      sv1.appendChild(s("text", { x: 6, y: 14, class: "math-lab red" }, `cut at height c = ${fix(c, 2)}`));
      // map
      clear(sv2);
      sv2.appendChild(s("rect", { x: MX(mx[0]), y: MY(my[1]), width: (mx[1] - mx[0]) * k2, height: (my[1] - my[0]) * k2, class: "math-mapbg" }));
      sv2.appendChild(s("line", { x1: MX(mx[0]), x2: MX(mx[1]), y1: MY(0), y2: MY(0), class: "math-axis" }));
      sv2.appendChild(s("line", { x1: MX(0), x2: MX(0), y1: MY(my[0]), y2: MY(my[1]), class: "math-axis" }));
      for (const z of [0.5, 1, 2, 3, 4]) {
        sv2.appendChild(s("ellipse", { cx, cy, rx: Math.sqrt(z) * k2, ry: Math.sqrt(z / 3) * k2, class: "math-cont", "data-tip": `level curve f = ${z}` }));
      }
      sv2.appendChild(s("ellipse", { cx, cy, rx: Math.sqrt(c) * k2, ry: Math.sqrt(c / 3) * k2, class: "math-lev" }));
      // label the selected level just outside the ellipse, below and to the right
      const lt = -Math.PI / 3, lx = MX(Math.sqrt(c) * Math.cos(lt)) + 6, ly = MY(Math.sqrt(c / 3) * Math.sin(lt)) + 14;
      sv2.appendChild(s("text", { x: lx, y: ly, class: "math-lab red" }, `f = ${fix(c, 2)}`));
      for (const [x, y] of [[1, 1], [2, 0]]) sv2.appendChild(s("circle", { cx: MX(x), cy: MY(y), r: 4, class: "math-pt ink", "data-tip": `f(${x}, ${y}) = ${f(x, y)}` }));
      for (const t of [-2, -1, 1, 2]) sv2.appendChild(s("text", { x: MX(t), y: MY(0) + 13, class: "math-tick", "text-anchor": "middle" }, tickMinus(t)));
      for (const t of [-1, 1]) sv2.appendChild(s("text", { x: MX(0) - 5, y: MY(t) + 4, class: "math-tick", "text-anchor": "end" }, tickMinus(t)));
      sv2.appendChild(s("text", { x: MX(mx[1]) - 3, y: MY(0) - 5, class: "math-lab", "text-anchor": "end" }, "x"));
      sv2.appendChild(s("text", { x: MX(0) - 6, y: MY(my[1]) + 12, class: "math-lab", "text-anchor": "end" }, "y"));
      clear(tiles);
      tiles.append(tile(fix(c, 2), "level c"), tile(`±${fix(Math.sqrt(c), 3)}`, String.raw`x reaches \(\pm\sqrt{c}\)`), tile(`±${fix(Math.sqrt(c / 3), 3)}`, String.raw`y reaches \(\pm\sqrt{c/3}\)`));
    }
    UI.bindTips(sv2);
    draw();
  });


  /* Contour map of f(x, y) = x² + 3y² with a draggable point, in three modes:
   *  partial      slices through the point along x and along y, with their slopes
   *  gradient     the gradient field, the gradient at the point and the level curve through it
   *  directional  a direction u at angle θ and the slope of f along it, against θ */
  Deck.fig("math-contour", (el, sp) => {
    const f = (x, y) => x * x + 3 * y * y;
    const grad = (x, y) => [2 * x, 6 * y];
    const mode = sp.mode || "gradient";
    let P = (sp.point || [1, 1]).slice(), th = sp.angle ?? 30;
    const XR = [-3, 3], YR = [-2, 2], k = 70, pad = 10;
    const W = (XR[1] - XR[0]) * k + 2 * pad, H = (YR[1] - YR[0]) * k + 2 * pad;
    const MX = (x) => pad + (x - XR[0]) * k, MY = (y) => pad + (YR[1] - y) * k;
    const inv = (px, py) => [XR[0] + (px - pad) / k, YR[1] - (py - pad) / k];
    const snap = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.round(v * 10) / 10));
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "math-cmap", role: "img", "aria-label": "Contour map of f(x, y) = x² + 3y² with a draggable point", "data-noswipe": "" });
    el.appendChild(svg);
    const below = h("div", { class: "math-cmap-below" });
    el.appendChild(below);
    const tiles = h("div", { class: "f-tiles" });
    const ctl = h("div", { class: "f-ctl sliders" });
    const sx = slider({ label: "x", min: XR[0], max: XR[1], step: 0.1, value: P[0], format: (v) => fix(v, 1), onInput: (v) => { P[0] = v; draw(); } });
    const sy = slider({ label: "y", min: YR[0], max: YR[1], step: 0.1, value: P[1], format: (v) => fix(v, 1), onInput: (v) => { P[1] = v; draw(); } });
    ctl.append(sx, sy);
    let sth = null;
    if (mode === "directional") { sth = slider({ label: "direction θ (degrees)", min: 0, max: 359, step: 1, value: th, format: (v) => `${fix(v, 0)}°`, onInput: (v) => { th = v; draw(); } }); ctl.appendChild(sth); }
    el.append(ctl, tiles);
    const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, rich(l)));
    const clipId = UI.nextId("cmclip");
    function arrow(g, x1, y1, x2, y2, cls, mk) { g.appendChild(s("line", { x1: MX(x1), y1: MY(y1), x2: MX(x2), y2: MY(y2), class: "math-arrow " + cls, "marker-end": mk })); }
    // slices for the "partial" mode
    let sliceA = null, sliceB = null;
    if (mode === "partial") {
      const a = h("div", { class: "math-slice" }), b = h("div", { class: "math-slice" });
      below.append(a, b);
      const cfg = (dom, lab) => ({ width: 300, height: 200, margin: { l: 40, r: 10, t: 22, b: 42 }, x: { domain: dom, ticks: UI.ticks(dom), label: lab, fmt: tickMinus }, y: { domain: [0, 22], ticks: [0, 5, 10, 15, 20] }, series: [] });
      sliceA = chart(a, cfg(XR, "x (y held fixed)"));
      sliceB = chart(b, cfg(YR, "y (x held fixed)"));
    }
    let dirChart = null;
    if (mode === "directional") {
      const a = h("div", { class: "math-slice wide" });
      below.appendChild(a);
      dirChart = chart(a, { width: 600, height: 170, margin: { l: 44, r: 12, t: 12, b: 34 }, x: { domain: [0, 360], ticks: [0, 90, 180, 270, 360], label: "direction θ (degrees)" }, y: { domain: [-14, 14], ticks: [-10, 0, 10], fmt: tickMinus, label: "slope along u" }, series: [] });
    }
    function draw() {
      const [x0, y0] = P, g0 = grad(x0, y0), c0 = f(x0, y0), gl = Math.hypot(g0[0], g0[1]);
      clear(svg);
      const mk = { s1: arrowHead(svg, "s1"), s2: arrowHead(svg, "s2"), ink: arrowHead(svg, "ink"), red: arrowHead(svg, "red") };
      svg.appendChild(s("clipPath", { id: clipId }, s("rect", { x: pad, y: pad, width: W - 2 * pad, height: H - 2 * pad })));
      const g = s("g", { "clip-path": `url(#${clipId})` });
      svg.appendChild(g);
      g.appendChild(s("rect", { x: pad, y: pad, width: W - 2 * pad, height: H - 2 * pad, class: "math-mapbg" }));
      g.appendChild(s("line", { x1: MX(XR[0]), x2: MX(XR[1]), y1: MY(0), y2: MY(0), class: "math-axis" }));
      g.appendChild(s("line", { x1: MX(0), x2: MX(0), y1: MY(YR[0]), y2: MY(YR[1]), class: "math-axis" }));
      for (const c of [2, 4, 6, 8, 10, 12]) {
        g.appendChild(s("ellipse", { cx: MX(0), cy: MY(0), rx: Math.sqrt(c) * k, ry: Math.sqrt(c / 3) * k, class: "math-cont" }));
        const lx = -Math.sqrt(c) * Math.SQRT1_2, ly = Math.sqrt(c / 3) * Math.SQRT1_2;
        g.appendChild(s("text", { x: MX(lx) - 3, y: MY(ly) - 3, class: "math-tick", "text-anchor": "end" }, String(c)));
      }
      if (mode === "gradient") {
        for (const gx of [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) for (const gy of [-1.5, -0.5, 0.5, 1.5]) {
          const d = grad(gx, gy);
          arrow(g, gx, gy, gx + 0.05 * d[0], gy + 0.05 * d[1], "ink thin", mk.ink);
        }
      }
      if (c0 > 1e-12) g.appendChild(s("ellipse", { cx: MX(0), cy: MY(0), rx: Math.sqrt(c0) * k, ry: Math.sqrt(c0 / 3) * k, class: "math-lev" }));
      let angle = NaN;
      if (mode === "partial") {
        g.appendChild(s("line", { x1: MX(XR[0]), x2: MX(XR[1]), y1: MY(y0), y2: MY(y0), class: "math-slice-l s1" }));
        g.appendChild(s("line", { x1: MX(x0), x2: MX(x0), y1: MY(YR[0]), y2: MY(YR[1]), class: "math-slice-l s2" }));
      }
      if (mode === "gradient" && gl > 1e-12) {
        // tangent of the level curve from its parametrization (x, y) = (√c cos t, √(c/3) sin t)
        const t = Math.atan2(y0 / Math.sqrt(c0 / 3), x0 / Math.sqrt(c0));
        const T = [-Math.sqrt(c0) * Math.sin(t), Math.sqrt(c0 / 3) * Math.cos(t)], tl = Math.hypot(T[0], T[1]);
        const u = [T[0] / tl, T[1] / tl];
        g.appendChild(s("line", { x1: MX(x0 - 0.9 * u[0]), y1: MY(y0 - 0.9 * u[1]), x2: MX(x0 + 0.9 * u[0]), y2: MY(y0 + 0.9 * u[1]), class: "math-tan" }));
        angle = (Math.acos(Math.min(1, Math.abs(g0[0] * u[0] + g0[1] * u[1]) / gl)) * 180) / Math.PI;
        const gs = 0.12;
        arrow(g, x0, y0, x0 + gs * g0[0], y0 + gs * g0[1], "s1", mk.s1);
        // right-angle mark in screen space
        const gu = [g0[0] / gl, g0[1] / gl], q = 0.14;
        const p1 = [x0 + q * u[0], y0 + q * u[1]], p2 = [p1[0] + q * gu[0], p1[1] + q * gu[1]], p3 = [x0 + q * gu[0], y0 + q * gu[1]];
        g.appendChild(s("path", { d: `M${MX(p1[0])},${MY(p1[1])} L${MX(p2[0])},${MY(p2[1])} L${MX(p3[0])},${MY(p3[1])}`, class: "math-rt" }));
        const tip = [x0 + gs * g0[0], y0 + gs * g0[1]];
        g.appendChild(s("text", { x: MX(tip[0]) + 7, y: MY(tip[1]) + 4, class: "math-lab blue" }, "∇f"));
      }
      let du = NaN, dnum = NaN;
      if (mode === "directional") {
        const r = (th * Math.PI) / 180, u = [Math.cos(r), Math.sin(r)];
        du = g0[0] * u[0] + g0[1] * u[1];
        const hh = 1e-4;
        dnum = (f(x0 + hh * u[0], y0 + hh * u[1]) - f(x0 - hh * u[0], y0 - hh * u[1])) / (2 * hh);
        if (gl > 1e-12) arrow(g, x0, y0, x0 + 0.12 * g0[0], y0 + 0.12 * g0[1], "s1", mk.s1);
        arrow(g, x0, y0, x0 + u[0], y0 + u[1], "s2", mk.s2);
        g.appendChild(s("text", { x: MX(x0 + 1.12 * u[0]), y: MY(y0 + 1.12 * u[1]) + 4, class: "math-lab", "text-anchor": "middle" }, "u"));
        if (gl > 1e-12) { const tip = [x0 + 0.12 * g0[0], y0 + 0.12 * g0[1]]; g.appendChild(s("text", { x: MX(tip[0]) + 7, y: MY(tip[1]) + 4, class: "math-lab blue" }, "∇f")); }
        const ut = s("circle", { cx: MX(x0 + u[0]), cy: MY(y0 + u[1]), r: 11, class: "math-handle" });
        ut.addEventListener("pointerdown", (e) => { drag = "u"; try { svg.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } e.preventDefault(); });
        g.appendChild(ut);
      }
      g.appendChild(s("circle", { cx: MX(x0), cy: MY(y0), r: 5.5, class: "math-pt" }));
      const hp = s("circle", { cx: MX(x0), cy: MY(y0), r: 13, class: "math-handle", tabindex: "0", "aria-label": "point; drag or use the sliders" });
      hp.addEventListener("pointerdown", (e) => { drag = "p"; try { svg.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } e.preventDefault(); });
      g.appendChild(hp);
      svg.appendChild(s("text", { x: MX(XR[1]) - 4, y: MY(0) - 5, class: "math-lab", "text-anchor": "end" }, "x"));
      svg.appendChild(s("text", { x: MX(0) + 6, y: MY(YR[1]) + 13, class: "math-lab" }, "y"));
      // readouts
      clear(tiles);
      const pt = `(${fix(x0, 1)}, ${fix(y0, 1)})`;
      if (mode === "partial") {
        const hh = 1e-4;
        tiles.append(
          tile(fix(g0[0], 2), String.raw`\(\partial f/\partial x = 2x\)`),
          tile(fix((f(x0 + hh, y0) - f(x0 - hh, y0)) / (2 * hh), 4), String.raw`\(\partial f/\partial x,\) central difference`),
          tile(fix(g0[1], 2), String.raw`\(\partial f/\partial y = 6y\)`),
          tile(fix((f(x0, y0 + hh) - f(x0, y0 - hh)) / (2 * hh), 4), String.raw`\(\partial f/\partial y,\) central difference`));
        const tan = (cx, cv, sl) => (t) => cv + sl * (t - cx);
        const ptsA = [], ptsB = [];
        for (let i = 0; i <= 120; i++) { const t = XR[0] + ((XR[1] - XR[0]) * i) / 120; ptsA.push([t, f(t, y0)]); }
        for (let i = 0; i <= 120; i++) { const t = YR[0] + ((YR[1] - YR[0]) * i) / 120; ptsB.push([t, f(x0, t)]); }
        const tA = tan(x0, c0, g0[0]), tB = tan(y0, c0, g0[1]);
        sliceA.update({ series: [{ points: ptsA, cls: "s1" }, { points: [[XR[0], tA(XR[0])], [XR[1], tA(XR[1])]], cls: "ink thin dashed" }], dots: [{ x: x0, y: c0, r: 5, cls: "", tip: `f(${fix(x0, 1)}, ${fix(y0, 1)}) = ${fix(c0, 2)}` }],
          after(sv) { sv.appendChild(s("text", { x: 44, y: 14, class: "math-lab" }, `slope ∂f/∂x = ${fix(g0[0], 2)}`)); } });
        sliceB.update({ series: [{ points: ptsB, cls: "s2" }, { points: [[YR[0], tB(YR[0])], [YR[1], tB(YR[1])]], cls: "ink thin dashed" }], dots: [{ x: y0, y: c0, r: 5, cls: "", tip: `f(${fix(x0, 1)}, ${fix(y0, 1)}) = ${fix(c0, 2)}` }],
          after(sv) { sv.appendChild(s("text", { x: 44, y: 14, class: "math-lab" }, `slope ∂f/∂y = ${fix(g0[1], 2)}`)); } });
      } else if (mode === "gradient") {
        tiles.append(
          tile(fix(c0, 2), String.raw`\(f\) at ${pt}`),
          tile(`(${fix(g0[0], 1)}, ${fix(g0[1], 1)})`, String.raw`gradient \(\nabla f = (2x, 6y)\)`),
          tile(fix(gl, 3), "length of the gradient"),
          tile(Number.isNaN(angle) ? "–" : `${fix(angle, 1)}°`, "angle to the level curve"));
      } else {
        const r = (th * Math.PI) / 180;
        tiles.append(
          tile(`(${fix(Math.cos(r), 3)}, ${fix(Math.sin(r), 3)})`, String.raw`\(\mathbf{u} = (\cos\theta, \sin\theta)\)`),
          tile(fix(du, 3), String.raw`\(\nabla f \cdot \mathbf{u}\) with \(\nabla f = \) ` + `(${fix(g0[0], 1)}, ${fix(g0[1], 1)})`),
          tile(fix(dnum, 3), "slope along u, central difference"),
          tile(fix(gl, 3), "largest slope = length of ∇f"));
        const pts = [];
        for (let a = 0; a <= 360; a += 2) { const rr = (a * Math.PI) / 180; pts.push([a, g0[0] * Math.cos(rr) + g0[1] * Math.sin(rr)]); }
        dirChart.update({ series: [{ points: pts, cls: "s2" }], hlines: [{ y: 0, cls: "" }, { y: gl, label: `max ${fix(gl, 2)}` }, { y: -gl, label: "" }], vlines: [{ x: th, cls: "cursor" }], dots: [{ x: th, y: du, r: 5, cls: "", tip: `θ = ${th}°: slope ${fix(du, 3)}` }] });
      }
    }
    let drag = null;
    svg.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const r = svg.getBoundingClientRect();
      const [vx, vy] = inv(((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H);
      if (drag === "p") { P = [snap(vx, XR[0], XR[1]), snap(vy, YR[0], YR[1])]; sx.set(P[0]); sy.set(P[1]); }
      else { th = Math.round((((Math.atan2(vy - P[1], vx - P[0]) * 180) / Math.PI) + 360) % 360) % 360; sth.set(th); }
      draw();
      e.preventDefault();
    });
    const stop = () => { drag = null; };
    svg.addEventListener("pointerup", stop);
    svg.addEventListener("pointercancel", stop);
    svg.addEventListener("keydown", (e) => {
      const m = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[e.key];
      if (!m) return;
      P = [snap(P[0] + m[0], XR[0], XR[1]), snap(P[1] + m[1], YR[0], YR[1])]; sx.set(P[0]); sy.set(P[1]); draw();
      e.preventDefault(); e.stopPropagation();
    });
    UI.bindTips(svg);
    draw();
  });

  /* Partial derivatives of the loss of our query with respect to the 16 numbers of one
   * token's vector, estimated by central differences on the real model. */
  Deck.fig("math-modelgrad", (el, sp) => {
    const T = M.config.T, d = M.config.d, ids = M.runningIds.slice(0, T);
    const stepsAvail = sp.steps || [400, M.finalStep];
    let step = sp.step ?? M.finalStep, pos = sp.pos ?? 1;
    const cache = new Map();
    const lossOf = (W) => {
      const tr = M.forward(ids, undefined, { weights: W });
      let sum = 0;
      for (let i = 0; i < T; i++) sum -= Math.log(tr.probs[i][M.runningIds[i + 1]]);
      return sum / T;
    };
    function partials(st, tok) {
      const key = `${st}:${tok}`;
      if (cache.has(key)) return cache.get(key);
      const W = M.weights(st), hh = 1e-4;
      const data = Float64Array.from(W.wte.data);
      const Wp = Object.assign({}, W, { wte: { shape: W.wte.shape, data } });
      const g = [];
      for (let j = 0; j < d; j++) {
        const kk = tok * d + j, v0 = data[kk];
        data[kk] = v0 + hh; const lp = lossOf(Wp);
        data[kk] = v0 - hh; const lm = lossOf(Wp);
        data[kk] = v0;
        g.push((lp - lm) / (2 * hh));
      }
      const res = { L: lossOf(W), g, w: Array.from(W.wte.data.subarray(tok * d, (tok + 1) * d)) };
      cache.set(key, res);
      return res;
    }
    const ctl = h("div", { class: "f-ctl" });
    ctl.appendChild(segmented({ label: "training step", value: step, options: stepsAvail.map((v) => ({ label: v === M.finalStep ? `${v.toLocaleString("en-US")} (final)` : v.toLocaleString("en-US"), value: v })), onChange: (v) => { step = v; draw(); } }));
    ctl.appendChild(segmented({ label: "token", value: pos, options: ids.map((id, i) => ({ label: M.vocab[id].trim(), value: i })), onChange: (v) => { pos = v; draw(); } }));
    const Wd = 560, Hd = 210, ml = 58, mr = 10, mt = 16, mb = 30;
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "math-bars-svg", role: "img", "aria-label": "16 partial derivatives of the loss" });
    const tiles = h("div", { class: "f-tiles" });
    el.append(ctl, svg, tiles);
    const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, rich(l)));
    function draw() {
      const tok = ids[pos], res = partials(step, tok);
      const mx = Math.max(...res.g.map(Math.abs), 1e-12);
      // a round scale just above the largest bar
      const mag = Math.pow(10, Math.floor(Math.log10(mx))), top = [1, 2, 5, 10].map((q) => q * mag).find((q) => q >= mx);
      clear(svg);
      const y0 = mt + (Hd - mt - mb) / 2, sc = (Hd - mt - mb) / 2 / top, bw = (Wd - ml - mr) / d;
      for (const v of [top, 0, -top]) {
        svg.appendChild(s("line", { x1: ml, x2: Wd - mr, y1: y0 - v * sc, y2: y0 - v * sc, class: v === 0 ? "math-axis" : "math-gridl" }));
        svg.appendChild(s("text", { x: ml - 6, y: y0 - v * sc + 4, class: "math-tick", "text-anchor": "end" }, v === 0 ? "0" : (v > 0 ? "" : "−") + String(+Math.abs(v).toPrecision(2))));
      }
      res.g.forEach((v, j) => {
        const x = ml + j * bw + bw * 0.18, w = bw * 0.64, hgt = Math.abs(v) * sc;
        svg.appendChild(s("rect", { x, y: v >= 0 ? y0 - hgt : y0, width: w, height: Math.max(hgt, 0.8), class: v >= 0 ? "math-bar pos" : "math-bar neg", "data-tip": `entry ${j + 1}: value ${fix(res.w[j], 4)}, ∂L/∂w = ${fix(v, 6)}` }));
        svg.appendChild(s("text", { x: x + w / 2, y: Hd - mb + 16, class: "math-tick", "text-anchor": "middle" }, String(j + 1)));
      });
      svg.appendChild(s("text", { x: ml, y: Hd - 2, class: "math-tick" }, `entry of the vector of “${M.vocab[tok].trim()}” (id ${tok})`));
      clear(tiles);
      const jmax = res.g.reduce((b, v, j) => (Math.abs(v) > Math.abs(res.g[b]) ? j : b), 0);
      tiles.append(
        tile(fix(res.L, 4), "loss on our query (nats)"),
        tile(fix(Math.abs(res.g[jmax]), 4), `largest |∂L/∂w| (entry ${jmax + 1})`),
        tile(String(2 * d), "forward passes for these 16"),
        tile((2 * M.config.params).toLocaleString("en-US"), "for all 9,968 parameters"));
    }
    UI.bindTips(svg);
    draw();
  });


  /* Two draggable 2-D vectors.
   *  add  a, b, the sum a + b drawn tip to tail, and c·a for a scalar c
   *  dot  a, b, the angle between them, the projection of b on a, and the dot product */
  Deck.fig("math-vec", (el, sp) => {
    const mode = sp.mode || "add";
    let a = sp.a.slice(), b = sp.b.slice(), c = sp.c ?? 2;
    const R = sp.range || 5, Wd = 380, Hd = 380, pad = 14, u = (Wd - 2 * pad) / (2 * R);
    const X = (v) => Wd / 2 + v * u, Y = (v) => Hd / 2 - v * u;
    const inv = (px, py) => [(px - Wd / 2) / u, (Hd / 2 - py) / u];
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "math-vec", role: "img", "aria-label": "two vectors on a grid; drag their tips", "data-noswipe": "" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const tiles = h("div", { class: "f-tiles" });
    el.append(svg, expl);
    if (mode === "add") el.appendChild(h("div", { class: "f-ctl sliders" }, slider({ label: "scalar c", min: -2, max: 2, step: 0.5, value: c, format: (v) => fix(v, 1), onInput: (v) => { c = v; draw(); } })));
    el.appendChild(tiles);
    const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, rich(l)));
    const vstr = (v) => `(${fix(v[0], 1)}, ${fix(v[1], 1)})`;
    let drag = null;
    function draw() {
      clear(svg);
      const mk = { s1: arrowHead(svg, "s1"), s2: arrowHead(svg, "s2"), s3: arrowHead(svg, "s3"), s4: arrowHead(svg, "s4") };
      for (let g = -R; g <= R; g++) {
        svg.appendChild(s("line", { x1: X(g), x2: X(g), y1: Y(-R), y2: Y(R), class: g === 0 ? "math-axis" : "math-gridl" }));
        svg.appendChild(s("line", { x1: X(-R), x2: X(R), y1: Y(g), y2: Y(g), class: g === 0 ? "math-axis" : "math-gridl" }));
      }
      const arr = (from, to, cls) => { if (Math.hypot(to[0] - from[0], to[1] - from[1]) > 1e-9) svg.appendChild(s("line", { x1: X(from[0]), y1: Y(from[1]), x2: X(to[0]), y2: Y(to[1]), class: "math-arrow " + cls, "marker-end": mk[cls.split(" ")[0]] })); };
      // short name labels just past each tip, along the arrow; values go in the readouts below
      const lab = (v, text, cls = "") => {
        const L = Math.hypot(v[0], v[1]);
        if (L < 1e-9) return;
        const dx = v[0] / L, dy = v[1] / L;
        let x = X(v[0]) + 13 * dx, y = Y(v[1]) - 13 * dy + 4;
        x = Math.max(8, Math.min(Wd - 8, x)); y = Math.max(14, Math.min(Hd - 6, y));
        svg.appendChild(s("text", { x, y, class: "math-lab " + cls, "text-anchor": dx > 0.3 ? "start" : dx < -0.3 ? "end" : "middle" }, text));
      };
      const sum = [a[0] + b[0], a[1] + b[1]];
      if (mode === "add") {
        const ca = [c * a[0], c * a[1]];
        arr([0, 0], ca, "s4 thin");
        arr(a, sum, "s2 dashed");
        arr([0, 0], sum, "s3");
        arr([0, 0], a, "s1");
        arr([0, 0], b, "s2");
        lab(sum, "a + b");
        if (Math.abs(c - 1) > 1e-9 && Math.abs(c) > 1e-9) lab(ca, `${fix(c, 1)}·a`);
      } else {
        const ab = a[0] * b[0] + a[1] * b[1], na = Math.hypot(a[0], a[1]), nb = Math.hypot(b[0], b[1]);
        if (na > 1e-9 && nb > 1e-9) {
          const t = ab / (na * na), foot = [t * a[0], t * a[1]];
          svg.appendChild(s("line", { x1: X(-R * a[0] / na), y1: Y(-R * a[1] / na), x2: X(R * a[0] / na), y2: Y(R * a[1] / na), class: "math-gridl dark" }));
          svg.appendChild(s("line", { x1: X(b[0]), y1: Y(b[1]), x2: X(foot[0]), y2: Y(foot[1]), class: "math-proj" }));
          svg.appendChild(s("line", { x1: X(0), y1: Y(0), x2: X(foot[0]), y2: Y(foot[1]), class: "math-projseg" }));
          // the angle between a and b
          const al = Math.atan2(a[1], a[0]);
          let dl = Math.atan2(b[1], b[0]) - al;
          while (dl > Math.PI) dl -= 2 * Math.PI;
          while (dl <= -Math.PI) dl += 2 * Math.PI;
          const rr = 30 / u, pts = [];
          for (let i = 0; i <= 24; i++) { const q = al + (dl * i) / 24; pts.push(`${X(rr * Math.cos(q))},${Y(rr * Math.sin(q))}`); }
          svg.appendChild(s("polyline", { points: pts.join(" "), class: "math-angle" }));
          const mid = al + dl / 2;
          svg.appendChild(s("text", { x: X((48 / u) * Math.cos(mid)), y: Y((48 / u) * Math.sin(mid)) + 4, class: "math-lab", "text-anchor": "middle" }, "θ"));
        }
        arr([0, 0], a, "s1");
        arr([0, 0], b, "s2");
      }
      // a's label sits beside the middle of the arrow (its tip may be covered by c·a)
      const La = Math.hypot(a[0], a[1]);
      if (La > 1e-9) svg.appendChild(s("text", { x: X(a[0] / 2) + 13 * (a[1] / La), y: Y(a[1] / 2) + 13 * (a[0] / La) + 4, class: "math-lab", "text-anchor": "middle" }, "a"));
      lab(b, "b");
      [["a", a], ["b", b]].forEach(([key, v]) => {
        const hd = s("circle", { cx: X(v[0]), cy: Y(v[1]), r: 13, class: "math-handle", tabindex: "0", "aria-label": `tip of vector ${key}; drag it` });
        hd.addEventListener("pointerdown", (e) => { drag = key; try { svg.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } e.preventDefault(); });
        svg.appendChild(hd);
      });
      clear(tiles);
      if (mode === "add") {
        expl.innerHTML = `a = ${vstr(a)}, b = ${vstr(b)}. &nbsp; a + b = (${fix(a[0], 1)} + ${fix(b[0], 1)}, ${fix(a[1], 1)} + ${fix(b[1], 1)}) = <b>${vstr(sum)}</b>. &nbsp; ${fix(c, 1)} · a = <b>${vstr([c * a[0], c * a[1]])}</b>.`;
        tiles.append(tile(fix(Math.hypot(a[0], a[1]), 3), "length of a"), tile(fix(Math.hypot(c * a[0], c * a[1]), 3), "length of c · a"), tile(fix(Math.abs(c), 1), "ratio of the lengths, |c|"));
      } else {
        const ab = a[0] * b[0] + a[1] * b[1], na = Math.hypot(a[0], a[1]), nb = Math.hypot(b[0], b[1]);
        const cs = na > 1e-9 && nb > 1e-9 ? ab / (na * nb) : NaN;
        const ang = Number.isNaN(cs) ? NaN : (Math.acos(Math.max(-1, Math.min(1, cs))) * 180) / Math.PI;
        expl.innerHTML = `a = ${vstr(a)}, b = ${vstr(b)}. &nbsp; a · b = ${fix(a[0], 1)}·${fix(b[0], 1)} + ${fix(a[1], 1)}·${fix(b[1], 1)} = <b>${fix(ab, 2)}</b> &nbsp; and &nbsp; |a| |b| cos θ = ${fix(na, 3)} × ${fix(nb, 3)} × ${Number.isNaN(cs) ? "–" : fix(cs, 4)} = <b>${Number.isNaN(cs) ? "–" : fix(na * nb * cs, 2)}</b>`;
        tiles.append(tile(fix(na, 3), "|a|"), tile(fix(nb, 3), "|b|"), tile(Number.isNaN(cs) ? "–" : fix(cs, 4), "cos θ"), tile(Number.isNaN(ang) ? "–" : `${fix(ang, 1)}°`, "angle θ"));
      }
    }
    svg.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const r = svg.getBoundingClientRect();
      const v = inv(((e.clientX - r.left) / r.width) * Wd, ((e.clientY - r.top) / r.height) * Hd).map((q) => Math.max(-R, Math.min(R, Math.round(q * 2) / 2)));
      if (drag === "a") a = v; else b = v;
      draw();
      e.preventDefault();
    });
    const stop = () => { drag = null; };
    svg.addEventListener("pointerup", stop);
    svg.addEventListener("pointercancel", stop);
    svg.addEventListener("keydown", (e) => {
      const m = { ArrowLeft: [-0.5, 0], ArrowRight: [0.5, 0], ArrowUp: [0, 0.5], ArrowDown: [0, -0.5] }[e.key];
      const key = document.activeElement && document.activeElement.getAttribute("aria-label");
      if (!m || !key) return;
      const which = key.includes("vector a") ? "a" : "b";
      const v = (which === "a" ? a : b).map((q, i) => Math.max(-R, Math.min(R, q + m[i])));
      if (which === "a") a = v; else b = v;
      draw();
      const again = svg.querySelector(`[aria-label="tip of vector ${which}; drag it"]`);
      if (again) again.focus();
      e.preventDefault(); e.stopPropagation();
    });
    draw();
  });

  /* Cosine similarity between the trained vectors of two tokens. */
  Deck.fig("math-cos", (el, sp) => {
    const W = M.weights(), d = M.config.d;
    const vec = (t) => M.row(W.wte, M.idOf(t));
    const groups = sp.groups;
    let A = sp.a, B = sp.b;
    const mkSel = (val, on) => {
      const sel = h("select", { class: "sel", "aria-label": "token" }, groups.map(([name, toks]) => h("optgroup", { label: name }, toks.map((t) => h("option", { value: t, selected: t === val ? true : null }, `${t.trim()}  (id ${M.idOf(t)})`)))));
      sel.addEventListener("change", () => on(sel.value));
      return sel;
    };
    const ctl = h("div", { class: "f-ctl" },
      h("label", { class: "ctl" }, h("span", { class: "ctl-label" }, "token a"), mkSel(A, (v) => { A = v; draw(); })),
      h("label", { class: "ctl" }, h("span", { class: "ctl-label" }, "token b"), mkSel(B, (v) => { B = v; draw(); })));
    const rows = h("div", { class: "math-cos-rows" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const tiles = h("div", { class: "f-tiles" });
    el.append(ctl, rows, expl, tiles);
    const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, rich(l)));
    function draw() {
      const a = vec(A), b = vec(B), pr = a.map((v, i) => v * b[i]);
      const mx = Math.max(...a.map(Math.abs), ...b.map(Math.abs));
      const dot = pr.reduce((x, y) => x + y, 0), na = Math.sqrt(a.reduce((x, v) => x + v * v, 0)), nb = Math.sqrt(b.reduce((x, v) => x + v * v, 0));
      clear(rows);
      const row = (label, vals, m, tipName) => h("div", { class: "math-cos-row" }, h("span", { class: "math-cos-l" }, label), UI.heatStrip(vals, { maxAbs: m, label: tipName, tipFmt: (v, i) => `${tipName}, entry ${i + 1}: ${fix(v, 3)}` }));
      rows.append(row(chip(A), a, mx, `vector of “${A.trim()}”`), row(chip(B), b, mx, `vector of “${B.trim()}”`), row(h("span", { class: "math-cos-prod" }, "a₁b₁ … a₁₆b₁₆"), pr, Math.max(...pr.map(Math.abs), 1e-9), "product"));
      expl.innerHTML = `cos = a · b / (|a| |b|) = ${fix(dot, 3)} / (${fix(na, 3)} × ${fix(nb, 3)}) = <b>${fix(dot / (na * nb), 4)}</b>`;
      clear(tiles);
      tiles.append(tile(fix(dot, 3), "a · b"), tile(fix(na, 3), "|a|"), tile(fix(nb, 3), "|b|"), tile(fix(dot / (na * nb), 4), "cosine similarity"));
    }
    draw();
  });


  /* A 2 × 2 matrix acting on the plane: the grid, the unit square and the point (1, 1)
   * are mapped by A; drag the images of (1, 0) and (0, 1), which are A's columns. */
  Deck.fig("math-grid", (el, sp) => {
    let A = sp.A.map((r) => r.slice());
    const R = 3, Wd = 380, Hd = 380, pad = 10, u = (Wd - 2 * pad) / (2 * R);
    const X = (v) => Wd / 2 + v * u, Y = (v) => Hd / 2 - v * u;
    const inv = (px, py) => [(px - Wd / 2) / u, (Hd / 2 - py) / u];
    const map = (p) => [A[0][0] * p[0] + A[0][1] * p[1], A[1][0] * p[0] + A[1][1] * p[1]];
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "math-vec", role: "img", "aria-label": "the plane transformed by a 2 by 2 matrix", "data-noswipe": "" });
    const mat = h("div", { class: "math-mat-show" });
    const top = h("div", { class: "math-grid-top" }, svg, mat);
    const tiles = h("div", { class: "f-tiles" });
    const sl = [];
    const names = [["a", 0, 0], ["b", 0, 1], ["c", 1, 0], ["d", 1, 1]];
    const ctlS = h("div", { class: "f-ctl sliders math-grid-sl" });
    names.forEach(([nm, i, j]) => { const e = slider({ label: nm, min: -2, max: 2, step: 0.01, value: A[i][j], format: (v) => fix(v, 2), onInput: (v) => { A[i][j] = v; draw(); } }); sl.push(e); ctlS.appendChild(e); });
    const presets = sp.presets;
    const seg = segmented({ label: "examples", value: 0, options: presets.map((q, k) => ({ label: q.label, value: k })), onChange: (k) => { A = presets[k].A.map((r) => r.slice()); syncSliders(); draw(); } });
    el.append(top, h("div", { class: "f-ctl" }, seg), ctlS, tiles);
    const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, rich(l)));
    const syncSliders = () => names.forEach(([, i, j], k) => sl[k].set(A[i][j]));
    const clipId = UI.nextId("gclip");
    const vs = (v) => `(${fix(v[0], 2)}, ${fix(v[1], 2)})`;
    let drag = null;
    function draw() {
      clear(svg);
      const mk = { s1: arrowHead(svg, "s1"), s2: arrowHead(svg, "s2") };
      svg.appendChild(s("clipPath", { id: clipId }, s("rect", { x: pad, y: pad, width: Wd - 2 * pad, height: Hd - 2 * pad })));
      const g = s("g", { "clip-path": `url(#${clipId})` });
      svg.appendChild(g);
      for (let k = -R; k <= R; k++) {
        g.appendChild(s("line", { x1: X(k), x2: X(k), y1: Y(-R), y2: Y(R), class: "math-gridl" }));
        g.appendChild(s("line", { x1: X(-R), x2: X(R), y1: Y(k), y2: Y(k), class: "math-gridl" }));
      }
      // images of the grid lines x = k and y = k
      for (let k = -8; k <= 8; k++) {
        const p1 = map([k, -12]), p2 = map([k, 12]), q1 = map([-12, k]), q2 = map([12, k]);
        g.appendChild(s("line", { x1: X(p1[0]), y1: Y(p1[1]), x2: X(p2[0]), y2: Y(p2[1]), class: k === 0 ? "math-tgrid axis" : "math-tgrid" }));
        g.appendChild(s("line", { x1: X(q1[0]), y1: Y(q1[1]), x2: X(q2[0]), y2: Y(q2[1]), class: k === 0 ? "math-tgrid axis" : "math-tgrid" }));
      }
      const sq = [[0, 0], [1, 0], [1, 1], [0, 1]].map(map);
      g.appendChild(s("polygon", { points: sq.map((q) => `${X(q[0])},${Y(q[1])}`).join(" "), class: "math-usq" }));
      const e1 = map([1, 0]), e2 = map([0, 1]), p11 = map([1, 1]);
      g.appendChild(s("line", { x1: X(0), y1: Y(0), x2: X(e1[0]), y2: Y(e1[1]), class: "math-arrow s1", "marker-end": mk.s1 }));
      g.appendChild(s("line", { x1: X(0), y1: Y(0), x2: X(e2[0]), y2: Y(e2[1]), class: "math-arrow s2", "marker-end": mk.s2 }));
      g.appendChild(s("circle", { cx: X(p11[0]), cy: Y(p11[1]), r: 5, class: "math-pt ink", "data-tip": `A(1, 1) = ${vs(p11)}` }));
      const lab = (v, t) => { const L = Math.hypot(v[0], v[1]) || 1; g.appendChild(s("text", { x: X(v[0]) + 14 * v[0] / L, y: Y(v[1]) - 14 * v[1] / L + 4, class: "math-lab", "text-anchor": v[0] / L > 0.3 ? "start" : v[0] / L < -0.3 ? "end" : "middle" }, t)); };
      lab(e1, "A(1, 0)"); lab(e2, "A(0, 1)");
      g.appendChild(s("text", { x: X(p11[0]) + 8, y: Y(p11[1]) + 16, class: "math-lab" }, "A(1, 1)"));
      [["e1", e1], ["e2", e2]].forEach(([key, v]) => {
        const hd = s("circle", { cx: X(v[0]), cy: Y(v[1]), r: 13, class: "math-handle" });
        hd.addEventListener("pointerdown", (e) => { drag = key; try { svg.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } e.preventDefault(); });
        g.appendChild(hd);
      });
      clear(mat);
      mat.appendChild(rich(String.raw`\[\mathbf{A} = \begin{pmatrix} ${fix(A[0][0], 2)} & ${fix(A[0][1], 2)} \\ ${fix(A[1][0], 2)} & ${fix(A[1][1], 2)} \end{pmatrix}\]`.replace(/−/g, "-")));
      clear(tiles);
      tiles.append(tile(vs(e1), String.raw`\(\mathbf{A}(1, 0)\): column 1`), tile(vs(e2), String.raw`\(\mathbf{A}(0, 1)\): column 2`), tile(vs(p11), String.raw`\(\mathbf{A}(1, 1)\)`), tile(fix(A[0][0] * A[1][1] - A[0][1] * A[1][0], 3), "signed area of the square after A: ad − bc"));
    }
    svg.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const r = svg.getBoundingClientRect();
      const v = inv(((e.clientX - r.left) / r.width) * Wd, ((e.clientY - r.top) / r.height) * Hd).map((q) => Math.max(-2, Math.min(2, Math.round(q * 10) / 10)));
      const col = drag === "e1" ? 0 : 1;
      A[0][col] = v[0]; A[1][col] = v[1];
      syncSliders(); draw();
      e.preventDefault();
    });
    const stop = () => { drag = null; };
    svg.addEventListener("pointerup", stop);
    svg.addEventListener("pointercancel", stop);
    UI.bindTips(svg);
    draw();
  });

  /* A matrix and its transpose; hovering an entry highlights its partner. */
  Deck.fig("math-transpose", (el, sp) => {
    const A = sp.A, m = A.length, n = A[0].length;
    const At = Array.from({ length: n }, (_, i) => Array.from({ length: m }, (_, j) => A[j][i]));
    const cells = [];
    const mat = (Mx, name, tr) => {
      const t = h("table", { class: "f-mat a" });
      Mx.forEach((row, i) => t.appendChild(h("tr", null, row.map((v, j) => {
        const td = h("td", { tabindex: "0", "data-tip": `${tr ? "Aᵀ" : "A"}${i + 1}${j + 1} = ${fix(v, 0)}` }, fix(v, 0));
        const key = tr ? `${j},${i}` : `${i},${j}`;
        cells.push([key, td]);
        const on = () => cells.forEach(([k2, c]) => c.classList.toggle("on", k2 === key));
        td.addEventListener("pointerenter", on); td.addEventListener("focus", on);
        return td;
      }))));
      return h("div", { class: "f-mat-wrap" }, h("div", { class: "f-mat-name" }, rich(name)), t);
    };
    el.appendChild(h("div", { class: "scroll-x" }, h("div", { class: "f-mm" }, mat(A, String.raw`\(\mathbf{A}\) <span class="dim">${m}×${n}</span>`, false), h("div", { class: "f-mm-op" }, "→"), mat(At, String.raw`\(\mathbf{A}^\top\) <span class="dim">${n}×${m}</span>`, true))));
    el.appendChild(h("div", { class: "math-tr-eq" }, rich(sp.after)));
    cells[0][1].dispatchEvent(new Event("pointerenter"));
  });

  /* The output layer of our model as one matrix product, with the real numbers:
   * X (final vectors, 8 × 16) times W_U (16 × 113) gives Z (scores, 8 × 113). */
  Deck.fig("math-batchmm", (el, sp) => {
    const ids = M.runningIds.slice(0, M.config.T);
    const tr = M.forward(ids), Xm = tr.final, Zm = tr.logits;
    const Wu = M.weights().wu, V = M.config.V, d = M.config.d, T = ids.length;
    const Wm = Array.from({ length: d }, (_, i) => Array.from(Wu.data.subarray(i * V, (i + 1) * V)));
    const cw = 4, ch = 13, xw = 9, wh = 5;
    const L = 62, gap = 14, topY = 26;
    const xX = L, xW = L + d * xw + gap, yW = topY, yZ = topY + d * wh + 20;
    const Wd = xW + V * cw + 6, Hd = yZ + T * ch + 24;
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "math-bmm", role: "img", "aria-label": "X times W_U equals Z with the real numbers of our model" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.append(h("div", { class: "scroll-x" }, svg), expl);
    const mx = (Mx) => Math.max(...Mx.flat().map(Math.abs));
    const mX = mx(Xm), mW = mx(Wm), mZ = mx(Zm);
    let sel = sp.row ?? 4, col = null;
    const vocab = M.vocab;
    function draw() {
      clear(svg);
      const block = (Mx, x0, y0, w, hh, m, kind) => Mx.forEach((row, i) => row.forEach((v, j) => {
        const r = s("rect", { x: x0 + j * w, y: y0 + i * hh, width: w, height: hh, style: `fill:${UI.div(v, m)}`, class: "math-bmm-c" });
        if (kind === "Z") { r.setAttribute("data-tip", `Z[${i + 1}, ${j + 1}] = row ${i + 1} of X · column ${j + 1} of W_U = ${fix(v, 3)}  (score of “${vocab[j].trim() || "space"}” after “${vocab[ids[i]].trim()}”)`); r.addEventListener("pointerenter", () => { sel = i; col = j; draw(); }); }
        if (kind === "X") r.addEventListener("pointerenter", () => { sel = i; col = null; draw(); });
        svg.appendChild(r);
      }));
      block(Wm, xW, yW, cw, wh, mW, "W");
      block(Xm, xX, yZ, xw, ch, mX, "X");
      block(Zm, xW, yZ, cw, ch, mZ, "Z");
      // frames and highlights
      const frame = (x, y, w, hh, cls) => svg.appendChild(s("rect", { x, y, width: w, height: hh, class: cls }));
      frame(xW, yW, V * cw, d * wh, "math-bmm-f"); frame(xX, yZ, d * xw, T * ch, "math-bmm-f"); frame(xW, yZ, V * cw, T * ch, "math-bmm-f");
      frame(xX - 1, yZ + sel * ch, d * xw + 2, ch, "math-bmm-h");
      frame(xW - 1, yZ + sel * ch, V * cw + 2, ch, "math-bmm-h");
      if (col !== null) { frame(xW + col * cw, yW - 1, cw, d * wh + 2, "math-bmm-h"); frame(xW + col * cw, yZ + sel * ch, cw, ch, "math-bmm-h2"); }
      ids.forEach((id, i) => svg.appendChild(s("text", { x: xX - 6, y: yZ + i * ch + ch - 3, class: "math-bmm-t" + (i === sel ? " on" : ""), "text-anchor": "end" }, vocab[id].trim())));
      svg.appendChild(s("text", { x: xX, y: yZ - 6, class: "math-lab" }, "X  8 × 16"));
      svg.appendChild(s("text", { x: xW, y: yW - 7, class: "math-lab" }, "W_U  16 × 113"));
      svg.appendChild(s("text", { x: xW, y: yZ + T * ch + 16, class: "math-lab" }, "Z = X W_U  8 × 113: one score per position and vocabulary token"));
      svg.appendChild(s("text", { x: xX + (d * xw) / 2, y: yW + (d * wh) / 2 + 4, class: "math-lab dim", "text-anchor": "middle" }, "row × column"));
      // explanation: the top three scores of the selected row
      const z = Zm[sel], top3 = z.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]).slice(0, 3), p = tr.probs[sel];
      const cell = col !== null ? ` Hovered: column ${col + 1} (“${vocab[col].trim() || "space"}”), score ${fix(z[col], 3)}.` : "";
      expl.innerHTML = `Row ${sel + 1} (input “${vocab[ids[sel]].trim()}”): 16 numbers times W_U give 113 scores. Highest: ${top3.map(([v, j]) => `“${vocab[j].trim()}” ${fix(v, 2)} (p = ${fix(p[j], 3)})`).join(", ")}.${cell}`;
    }
    UI.bindTips(svg);
    draw();
  });


  /* Paired horizontal bars: two probabilities per outcome (e.g. the grammar's and the model's). */
  Deck.fig("math-pairs", (el, sp) => {
    const items = sp.items, mxv = sp.max || Math.max(...items.flatMap((it) => [it.a, it.b]));
    const rowH = 30, L = 92, R = 128, Wd = 560, Hd = items.length * rowH + 30;
    const X = (v) => L + (v / mxv) * (Wd - L - R);
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "math-pairs", role: "img", "aria-label": sp.aria || "paired probabilities" });
    items.forEach((it, i) => {
      const y = 6 + i * rowH;
      svg.appendChild(s("text", { x: L - 8, y: y + 15, class: "math-bmm-t big", "text-anchor": "end" }, it.label));
      svg.appendChild(s("rect", { x: L, y: y + 2, width: Math.max(1, X(it.a) - L), height: 10, class: "math-pa", "data-tip": `${it.label}: ${sp.aName} ${it.aText || fix(it.a, 4)}` }));
      svg.appendChild(s("rect", { x: L, y: y + 13, width: Math.max(1, X(it.b) - L), height: 10, class: "math-pb", "data-tip": `${it.label}: ${sp.bName} ${fix(it.b, 4)}` }));
      svg.appendChild(s("text", { x: Wd - R + 8, y: y + 15, class: "math-tick" }, `${it.aText || fix(it.a, 4)} · ${fix(it.b, 4)}`));
    });
    svg.appendChild(s("line", { x1: L, x2: L, y1: 2, y2: Hd - 22, class: "math-axis" }));
    el.appendChild(h("div", { class: "scroll-x" }, svg));
    el.appendChild(h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw math-sw-a" }), sp.aName), h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), sp.bName)));
    UI.bindTips(svg);
  });

  /* Sampling with the cumulative distribution: lay the probabilities end to end on [0, 1),
   * draw u uniformly, return the outcome whose segment contains u. */
  Deck.fig("math-cdf", (el, sp) => {
    const out = sp.outcomes, K = out.length;
    const seg = [];
    let acc = 0;
    out.forEach((o) => { seg.push([acc, acc + o.p]); acc += o.p; });
    let u = sp.u ?? 0.3;
    let counts = new Array(K).fill(0), n = 0, rng = M.mulberry32(sp.seed ?? 7);
    const Wd = 600, Hd = 168, x0 = 14, x1 = Wd - 14, by = 70, bh = 34;
    const X = (v) => x0 + v * (x1 - x0);
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "math-cdf", role: "img", "aria-label": "outcomes laid end to end on the interval from 0 to 1" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const sv2 = s("svg", { viewBox: `0 0 ${Wd} 150`, class: "math-cdf", role: "img", "aria-label": "share of draws per outcome against its probability" });
    const su = slider({ label: "u", min: 0, max: 0.999, step: 0.001, value: u, format: (v) => fix(v, 3), onInput: (v) => { u = v; draw(); } });
    const draws = (k) => { for (let i = 0; i < k; i++) { u = rng(); const j = pick(u); counts[j]++; n++; } su.set(Math.min(0.999, +u.toFixed(3))); draw(); };
    // u printed with enough decimals that the printed value lies in the printed segment
    const showU = (v, a, b) => { for (let d = 3; d < 9; d++) { const t = +v.toFixed(d); if (t >= +a.toFixed(4) && t < +b.toFixed(4)) return fix(v, d); } return fix(v, 9); };
    const ctl = h("div", { class: "f-ctl" }, button("Draw 1", () => draws(1), { kind: "primary" }), button("Draw 100", () => draws(100)), button("Draw 1,000", () => draws(1000)), button("Reset counts", () => { counts = new Array(K).fill(0); n = 0; rng = M.mulberry32(sp.seed ?? 7); draw(); }));
    el.append(h("div", { class: "scroll-x" }, svg), expl, h("div", { class: "f-ctl sliders" }, su), ctl, h("div", { class: "scroll-x" }, sv2));
    function pick(v) { for (let j = 0; j < K; j++) if (v < seg[j][1]) return j; return K - 1; }
    function draw() {
      clear(svg);
      const cur = pick(u);
      seg.forEach(([a, b], j) => {
        svg.appendChild(s("rect", { x: X(a), y: by, width: X(b) - X(a), height: bh, class: "math-seg" + (j % 2 ? " alt" : "") + (j === cur ? " on" : ""), "data-tip": `${out[j].label}: [${fix(a, 4)}, ${fix(b, 4)}), length ${out[j].text || fix(out[j].p, 4)}` }));
        const cx = (X(a) + X(b)) / 2, narrow = X(b) - X(a) < out[j].label.length * 7 + 6;
        svg.appendChild(s("text", { x: cx, y: by - 8, class: "math-lab" + (j === cur ? " red" : ""), "text-anchor": narrow ? "start" : "middle", transform: narrow ? `rotate(-35 ${cx} ${by - 8})` : null }, out[j].label));
        svg.appendChild(s("line", { x1: X(b), x2: X(b), y1: by + bh, y2: by + bh + 6, class: "math-axis" }));
        if (j < K - 1) svg.appendChild(s("text", { x: X(b), y: by + bh + 18, class: "math-tick", "text-anchor": "middle" }, fix(b, 3)));
      });
      svg.appendChild(s("text", { x: X(0), y: by + bh + 18, class: "math-tick", "text-anchor": "start" }, "0"));
      svg.appendChild(s("text", { x: X(1), y: by + bh + 18, class: "math-tick", "text-anchor": "end" }, "1"));
      svg.appendChild(s("line", { x1: X(u), x2: X(u), y1: by - 4, y2: by + bh + 26, class: "math-u" }));
      svg.appendChild(s("path", { d: `M${X(u) - 6},${by + bh + 36} L${X(u) + 6},${by + bh + 36} L${X(u)},${by + bh + 27} Z`, class: "math-u-tri" }));
      const [a, b] = seg[cur], ut = showU(u, a, b);
      svg.appendChild(s("text", { x: Math.min(Math.max(X(u), 40), Wd - 40), y: by + bh + 50, class: "math-lab red", "text-anchor": "middle" }, `u = ${ut}`));
      expl.innerHTML = `u = ${ut} lies in [${fix(a, 4)}, ${fix(b, 4)}), the segment of <b>${out[cur].label}</b>, so the draw is <b>${out[cur].label}</b>.`;
      // shares after n draws, against the probabilities
      clear(sv2);
      const shares = counts.map((c) => (n ? c / n : 0)), ymax = Math.max(0.32, ...shares);
      const cw = (Wd - 20) / K, top = 112, Y = (v) => top - (v / ymax) * 90;
      sv2.appendChild(s("line", { x1: 10, x2: Wd - 10, y1: top, y2: top, class: "math-axis" }));
      out.forEach((o, j) => {
        const cx = 10 + j * cw + cw / 2, share = shares[j];
        sv2.appendChild(s("rect", { x: cx - 15, y: Y(o.p), width: 14, height: top - Y(o.p), class: "math-pa", "data-tip": `${o.label}: probability ${o.text || fix(o.p, 4)}` }));
        sv2.appendChild(s("rect", { x: cx + 1, y: Y(share), width: 14, height: Math.max(0, top - Y(share)), class: "math-pb", "data-tip": `${o.label}: ${counts[j]} of ${n} draws = ${fix(share, 4)}` }));
        sv2.appendChild(s("text", { x: cx, y: top + 14, class: "math-tick", "text-anchor": "middle" }, o.label));
        sv2.appendChild(s("text", { x: cx, y: top + 27, class: "math-tick", "text-anchor": "middle" }, String(counts[j])));
      });
      sv2.appendChild(s("text", { x: 12, y: 12, class: "math-lab" }, `${n.toLocaleString("en-US")} draw${n === 1 ? "" : "s"}. Outline: probability. Filled: share of draws. Bottom: counts.`));
    }
    UI.bindTips(svg); UI.bindTips(sv2);
    draws(sp.initial ?? 0);
    u = sp.u ?? 0.3; su.set(u); draw();
  });

  // ============================================================ Part 1
  Deck.part({ n: 1, title: "The math toolkit", short: "Math" });
  // The engine needs a chapter before any sheet; opening chapter 1.1 here keeps the
  // chapter codes 1.1–1.8 (a separate "Opener" chapter would shift them to 1.2–1.9).
  Deck.chapter({ title: "Functions" });

  Deck.slide({
    id: "math-part",
    layout: "part",
    title: "The math toolkit",
    body: String.raw`<p>Training a model comes down to one question, asked for each of its 9,968 parameters: if this one number changed a little, how would the error of the model's predictions change? The answer is a derivative, and the chain rule computes it through the many steps of the model. The model itself is built from matrix products on vectors, its output is a probability distribution, and its error is measured with a logarithm.</p>
<p>This part builds each of those tools from school algebra. Every rule is checked on numbers you can redo by hand, and most plates let you move an input and watch the result change.</p>`,
    take: "Compute a derivative by rule and check it numerically, push derivatives through a chain of operations, multiply matrices and count their FLOPs, and compute the probability of a query and its logarithm without underflow.",
  });

  // ------------------------------------------------------------ 1.1 Functions (chapter opened above)
  Deck.slide({
    id: "math-function",
    title: "A function assigns one output to each input",
    body: String.raw`<p>A <b>function</b> is a rule that takes an input and returns exactly one output. \(f(x) = (x - 3)^2\) says: subtract 3, then square. At \(x = 1\) it returns \((1 - 3)^2 = 4;\) at \(x = 3\) it returns 0.</p>
<p>In code it is a pure function, <code>def f(x): return (x - 3)**2</code>: the same input always gives the same output.</p>
<table><tr><th>x</th><td class="r">0</td><td class="r">1</td><td class="r">2</td><td class="r">3</td><td class="r">4</td><td class="r">5</td><td class="r">6</td></tr><tr><th>f(x)</th><td class="r">9</td><td class="r">4</td><td class="r">1</td><td class="r">0</td><td class="r">1</td><td class="r">4</td><td class="r">9</td></tr></table>
<p>A table lists some input–output pairs. The <b>graph</b> of the function draws every pair \((x, f(x))\) as a point; together they form a curve. This one is a parabola, lowest at \(x = 3.\)</p>
<p>Our model is a function too. Its input is up to 8 token ids, its output is 113 probabilities for each position, and its 9,968 parameters are fixed numbers inside the rule.</p>`,
    note: "Part 2 uses this exact curve as its first loss, L(w) = (w − 3)².",
    fig: {
      type: "plot", caption: String.raw`<span>\(f(x) = (x - 3)^2.\) The black dots are the table. Drag the red point, or use the slider, to evaluate any other input.</span>`,
      x: [-1, 7], y: [-1, 17], xlabel: "x", ylabel: "f(x)",
      xaxis: { fmt: tickMinus }, yaxis: { fmt: tickMinus },
      params: [{ name: "x0", label: "input x", min: -1, max: 7, step: 0.1, value: 1, fmt: "f1" }],
      fns: [{ f: (x) => (x - 3) ** 2, cls: "ink" }],
      points: [[0, 9], [1, 4], [2, 1], [3, 0], [4, 1], [5, 4], [6, 9]],
      extra: (p) => ({
        after(svg, X, Y, pp, box) {
          const y0 = (p.x0 - 3) ** 2, g = clipG(svg, box);
          guides(g, X, Y, box, p.x0, y0);
          g.appendChild(s("circle", { cx: X(p.x0), cy: Y(y0), r: 6, class: "math-pt" }));
          plabel(g, box, X(p.x0) + 10, Y(y0) - 8, `f(${fix(p.x0, 1)}) = ${fix(y0, 2)}`);
          dragHandle(svg, X, Y, box, [-1, 7], 0, p.x0, y0);
        },
      }),
      readouts: [
        { label: "input x", f: (p) => p.x0, fmt: "f1" },
        { label: "x − 3", f: (p) => p.x0 - 3, fmt: "f1" },
        { label: "output f(x) = (x − 3)²", f: (p) => (p.x0 - 3) ** 2, fmt: "f2" },
      ],
    },
    terms: [
      { term: "Function", def: "A rule that returns exactly one output for each input, such as \\(f(x) = (x - 3)^2.\\)" },
      { term: "Graph of a function", def: "The set of points \\((x, f(x)),\\) drawn as a curve." },
    ],
  });

  Deck.slide({
    id: "math-linear",
    title: "A linear function is a line with a slope and an intercept",
    body: String.raw`<p>The function \(y = wx + b\) draws a straight line. The <b>intercept</b> \(b\) is its value at \(x = 0.\) The <b>slope</b> \(w\) is how much \(y\) changes when \(x\) grows by 1.</p>
<p>With \(w = 2\) and \(b = 1,\) the inputs \(x = 0, 1, 2, 3\) give \(y = 1, 3, 5, 7.\) Each step of 1 in \(x\) adds 2 to \(y,\) anywhere on the line. Between any two points the slope is the rise over the run: \((7 - 3)/(3 - 1) = 2.\)</p>
<p>A negative slope makes the line fall, and \(w = 0\) makes it flat at height \(b.\) With \(w = {-0.5}\) and \(b = 3\) the line drops 0.5 per step and crosses \(y = 0\) at \(x = 6.\)</p>
<p>Lines matter twice in this set. Part 2's first model is <i>ŷ</i> = <i>wx</i> + <i>b</i>, with \(w\) and \(b\) as its parameters. And near any point where a function has a derivative, its graph is close to a line whose slope is that derivative (chapter 1.2).</p>`,
    fig: {
      type: "plot", caption: String.raw`<span>\(y = wx + b.\) The triangle marks a run of 1 from \(x = 1\) and the rise \(w\) it produces.</span>`,
      x: [-2, 8], y: [-6, 10], xlabel: "x", ylabel: "y",
      xaxis: { fmt: tickMinus }, yaxis: { fmt: tickMinus },
      params: [
        { name: "w", label: "slope w", min: -3, max: 3, step: 0.1, value: 2, fmt: "f1" },
        { name: "b", label: "intercept b", min: -4, max: 4, step: 0.5, value: 1, fmt: "f1" },
      ],
      fns: [{ f: (x, p) => p.w * x + p.b, cls: "s1" }],
      extra: (p) => ({
        dots: [{ x: 0, y: p.b, r: 5, cls: "ink", tip: `intercept: (0, ${fix(p.b, 1)})` }],
        after(svg, X, Y, pp, box) {
          const g = clipG(svg, box), y1 = p.w + p.b, y2 = 2 * p.w + p.b;
          g.appendChild(s("path", { d: `M${X(1)},${Y(y1)} H${X(2)} V${Y(y2)}`, class: "math-tri" }));
          plabel(g, box, X(1.5), Y(y1) + (p.w >= 0 ? 14 : -6), "run 1", "middle");
          plabel(g, box, X(2) + 6, (Y(y1) + Y(y2)) / 2 + 4, `rise ${fix(p.w, 1)}`);
        },
      }),
      readouts: [
        { label: "y at x = 0 (intercept b)", f: (p) => p.b, fmt: "f1" },
        { label: "rise per unit run (slope w)", f: (p) => p.w, fmt: "f1" },
        { label: "crosses y = 0 at x =", f: (p) => (p.w === 0 ? (p.b === 0 ? "everywhere" : "never") : fix(-p.b / p.w, 2)), fmt: (v) => v },
      ],
    },
    terms: [
      { term: "Slope", def: "How much \\(y\\) changes per unit increase of \\(x\\) along a line: rise over run. For \\(y = wx + b\\) it is \\(w.\\)" },
      { term: "Intercept", def: "The value of a line at \\(x = 0;\\) \\(b\\) in \\(y = wx + b.\\)" },
    ],
  });

  Deck.slide({
    id: "math-compose",
    title: "Composing functions feeds one output into the next",
    body: String.raw`<p>Two functions can run in sequence: the output of the first becomes the input of the second. With \(g(x) = 3x + 1\) and \(f(u) = u^2,\) the <b>composition</b> \(f(g(x))\) means: compute \(u = g(x),\) then return \(f(u).\)</p>
<p>At \(x = 2\): \(u = 3 \cdot 2 + 1 = 7,\) then \(f(7) = 49.\) Written out, \(f(g(x)) = (3x + 1)^2.\)</p>
<p>Order matters. \(g(f(x))\) squares first and then scales: \(g(f(2)) = 3 \cdot 4 + 1 = 13,\) not 49.</p>
<p>A neural network is a long composition. Each step takes the previous step's output as its input: our model looks up 16 numbers per token, sends them through 2 layers of normalizations, matrix products, nonlinear functions and additions, and ends with 113 probabilities. Chapter 1.3 shows how to differentiate a composition, and that rule is how training gets its derivatives.</p>`,
    fig: {
      type: "math-pipe", caption: "A two-stage pipeline. Each row's input is the value in the row above it.",
      x: { label: "input x", min: -2, max: 3, step: 0.1, value: 2, dp: 1 },
      dp: 2,
      orders: [
        { label: "f(g(x))", stages: [{ v: "u", tex: "g(x) = 3x + 1", f: (x) => 3 * x + 1 }, { v: "y", tex: "f(u) = u^2", f: (u) => u * u }],
          explain: (x, v) => `f(g(${fix(x, 1)})) = f(${fix(v[1], 1)}) = <b>${fix(v[2], 2)}</b>` },
        { label: "g(f(x))", stages: [{ v: "u", tex: "f(x) = x^2", f: (x) => x * x }, { v: "y", tex: "g(u) = 3u + 1", f: (u) => 3 * u + 1 }],
          explain: (x, v) => `g(f(${fix(x, 1)})) = g(${fix(v[1], 2)}) = <b>${fix(v[2], 2)}</b>` },
      ],
    },
    terms: [{ term: "Composition", def: "Running functions in sequence, \\(f(g(x))\\): the output of \\(g\\) is the input of \\(f.\\)" }],
  });

  Deck.slide({
    id: "math-surface",
    title: "A function of two inputs is a surface and a contour map",
    body: String.raw`<p>A function can take several inputs. \(f(x, y) = x^2 + 3y^2\) takes two and returns one number: \(f(1, 1) = 1 + 3 = 4,\) \(f(2, 0) = 4,\) \(f(0, 0) = 0.\)</p>
<p>Its graph is a <b>surface</b>: above each point \((x, y)\) of the plane, a height \(f(x, y).\) Its lowest point is \((0, 0),\) and it rises faster along \(y\) than along \(x\) because of the factor 3.</p>
<p>A <b>contour map</b> draws the same function flat. Each <b>level curve</b> joins the points where \(f\) has one value \(c\): it is where a horizontal cut at height \(c\) meets the surface. Here \(x^2 + 3y^2 = c\) is an ellipse that reaches \(x = {\pm\sqrt{c}}\) and \(y = {\pm\sqrt{c/3}};\) at \(c = 4\) that is \(\pm 2\) and \(\pm 1.155,\) and both \((1, 1)\) and \((2, 0)\) lie on it.</p>
<p>Part 2 draws loss functions of two parameters as contour maps, and chapter 1.4 reads the gradient off them.</p>`,
    fig: { type: "math-surface", caption: String.raw`<span>\(f(x, y) = x^2 + 3y^2.\) The slider moves the cut; the same level curve is red in both views. The dots are \((1, 1)\) and \((2, 0).\)</span>`, level: 4 },
    terms: [
      { term: "Contour map", def: "A drawing of a function of two inputs as level curves on the plane, one curve per output value." },
      { term: "Level curve", def: "The set of inputs where a function of two inputs has one fixed value \\(c;\\) for \\(x^2 + 3y^2 = c,\\) an ellipse." },
    ],
  });

  Deck.slide({
    id: "math-recurring",
    title: "Five functions recur through this set",
    body: String.raw`<p>Five functions of one input do most of the work in the parts that follow. The plate plots each over the range where its shape shows; hover a dot for its exact value.</p>
<ul>
<li>\(x^2\) is never negative, and large inputs dominate: \(3^2 = 9\) but \(10^2 = 100.\)</li>
<li>The <b>exponential</b> \(e^x\) raises \(e = 2.71828\) to the power \(x.\) It is positive for every \(x,\) \(e^0 = 1,\) and each step of 1 in \(x\) multiplies it by 2.718.</li>
<li>The <b>natural logarithm</b> \(\ln x\) undoes the exponential: \(\ln e^x = x.\) It is defined only for \(x \gt 0,\) with \(\ln 1 = 0\) and \(\ln 0.1 = {-2.303}.\)</li>
<li>The <b>ReLU</b>, \(\max(0, x),\) returns \(x\) when it is positive and 0 otherwise.</li>
<li>The <b>sigmoid</b>, \(\sigma(x) = 1/(1 + e^{-x}),\) lies strictly between 0 and 1: \(\sigma(0) = 0.5,\) \(\sigma(2) = 0.881,\) \(\sigma({-2}) = 0.119.\)</li>
</ul>`,
    fig: {
      type: "math-gallery", caption: "The five functions, each with where it appears later",
      items: [
        { title: String.raw`\(x^2\)`, f: (x) => x * x, x: [-3, 3], y: [-1, 9], xt: [-2, 0, 2], yt: [0, 4, 8], marks: [[-2, 4, "(−2)² = 4"], [0, 0, "0² = 0"], [1, 1, "1² = 1"], [2, 4, "2² = 4"]], note: "Squared error, the first loss (Part 2)." },
        { title: String.raw`\(e^x\)`, f: (x) => Math.exp(x), x: [-3, 2.5], y: [-1, 9], xt: [-2, 0, 2], yt: [0, 4, 8], hl: [0], marks: [[-1, Math.exp(-1), "e⁻¹ = 0.368"], [0, 1, "e⁰ = 1"], [1, Math.E, "e¹ = 2.718"], [2, Math.exp(2), "e² = 7.389"]], note: "Turns scores into positive weights in softmax (Part 2)." },
        { title: String.raw`\(\ln x\)`, f: (x) => (x > 0 ? Math.log(x) : NaN), x: [0, 8], y: [-3, 2.5], xt: [0, 2, 4, 6, 8], yt: [-2, 0, 2], hl: [0], marks: [[0.1, Math.log(0.1), "ln 0.1 = −2.303"], [1, 0, "ln 1 = 0"], [Math.E, 1, "ln e = 1"], [Math.exp(2), 2, "ln 7.389 = 2.000"]], note: "Log-probabilities and the cross-entropy loss (Part 2)." },
        { title: String.raw`\(\max(0, x)\)`, f: (x) => Math.max(0, x), x: [-3, 3], y: [-1, 3], xt: [-2, 0, 2], yt: [0, 1, 2, 3], marks: [[-2, 0, "max(0, −2) = 0"], [0, 0, "max(0, 0) = 0"], [2, 2, "max(0, 2) = 2"]], note: "The ReLU activation of a neuron (Part 3)." },
        { title: String.raw`\(\sigma(x) = \frac{1}{1 + e^{-x}}\)`, f: (x) => 1 / (1 + Math.exp(-x)), x: [-6, 6], y: [-0.2, 1.2], xt: [-6, -3, 0, 3, 6], yt: [0, 0.5, 1], hl: [0, 1], marks: [[-2, 1 / (1 + Math.exp(2)), "σ(−2) = 0.119"], [0, 0.5, "σ(0) = 0.5"], [2, 1 / (1 + Math.exp(-2)), "σ(2) = 0.881"]], note: "One score to a probability between 0 and 1 (Part 2)." },
      ],
    },
    terms: [
      { term: "Exponential", def: "\\(e^x,\\) the number \\(e = 2.71828\\) raised to the power \\(x.\\) Positive for every \\(x;\\) \\(e^{a+b} = e^a e^b.\\)" },
      { term: "Natural logarithm", def: "\\(\\ln x,\\) the inverse of \\(e^x\\): \\(\\ln e^x = x.\\) Defined for \\(x \\gt 0;\\) \\(\\ln 1 = 0.\\)" },
      { term: "ReLU", def: "The function \\(\\max(0, x)\\): passes positive inputs and replaces negative ones with 0." },
      { term: "Sigmoid", def: "\\(\\sigma(x) = 1/(1 + e^{-x}),\\) which maps any number into the interval between 0 and 1; \\(\\sigma(0) = 0.5.\\)" },
    ],
    check: {
      q: String.raw`With \(g(x) = x + 2\) and \(f(u) = \ln u,\) what is \(f(g({-1}))\)?`,
      options: ["0", "1", String.raw`\(\ln({-1}) + 2,\) undefined`, "−1"],
      answer: 0,
      why: String.raw`Inside first: \(g({-1}) = {-1} + 2 = 1,\) then \(f(1) = \ln 1 = 0.\) Applying \(\ln\) to \({-1}\) first is the wrong order, and \(\ln\) of a negative number is undefined.`,
    },
  });

  // ------------------------------------------------------------ 1.2 Derivatives
  Deck.chapter({ title: "Derivatives" });

  const sqr = (x) => x * x;
  // an arrowhead marker for marks drawn inside a plot (the svg is cleared on every draw)
  function arrowHead(svg, cls = "") {
    const id = UI.nextId("mah");
    svg.appendChild(s("defs", null, s("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "math-ah " + cls }))));
    return `url(#${id})`;
  }
  const sig4 = (h) => String(+h.toPrecision(4));
  const hOf = (k) => Math.pow(10, -k / 10);

  Deck.slide({
    id: "math-secant",
    title: "The average rate of change is the slope between two points",
    body: String.raw`<p>On a curve the steepness changes from place to place, so a slope needs two points. Between the inputs \(x_1\) and \(x_2\) the <b>average rate of change</b> is the rise over the run:</p>
\[\frac{f(x_2) - f(x_1)}{x_2 - x_1}\]
<p>It is the slope of the <b>secant</b>, the straight line through those two points of the curve. For \(f(x) = x^2\) between \(x_1 = 3\) and \(x_2 = 5\) it is \((25 - 9)/(5 - 3) = 16/2 = 8.\) Between 3 and 4 it is \((16 - 9)/1 = 7.\)</p>
<p>For \(x^2\) the secant slope always equals \(x_1 + x_2,\) because \(x_2^2 - x_1^2 = (x_2 - x_1)(x_2 + x_1).\) Every slider position can be checked by adding the two inputs. The formula breaks down only at \(x_1 = x_2,\) where rise and run are both 0.</p>`,
    fig: {
      type: "plot", caption: String.raw`<span>\(f(x) = x^2\) and the secant through two of its points. Drag either point.</span>`,
      x: [-1, 6], y: [-2, 36], xlabel: "x", ylabel: "f(x)", xaxis: { fmt: tickMinus }, yaxis: { fmt: tickMinus },
      params: [
        { name: "a", label: "x₁", min: -1, max: 6, step: 0.1, value: 3, fmt: "f1" },
        { name: "b", label: "x₂", min: -1, max: 6, step: 0.1, value: 5, fmt: "f1" },
      ],
      fns: [
        { f: sqr, cls: "ink" },
        { f: (x, p) => { if (Math.abs(p.b - p.a) < 1e-9) return NaN; const m = (sqr(p.b) - sqr(p.a)) / (p.b - p.a); return sqr(p.a) + m * (x - p.a); }, cls: "s1" },
      ],
      extra: (p) => ({
        after(svg, X, Y, pp, box) {
          const g = clipG(svg, box);
          if (Math.abs(p.b - p.a) > 1e-9) g.appendChild(s("path", { d: `M${X(p.a)},${Y(sqr(p.a))} H${X(p.b)} V${Y(sqr(p.b))}`, class: "math-tri dashed" }));
          [[p.a, "x₁"], [p.b, "x₂"]].forEach(([x, l]) => {
            g.appendChild(s("circle", { cx: X(x), cy: Y(sqr(x)), r: 5.5, class: "math-pt" }));
            plabel(g, box, X(x) - 9, Y(sqr(x)) - 7, l, "end");
          });
          dragHandle(svg, X, Y, box, [-1, 6], 0, p.a, sqr(p.a));
          dragHandle(svg, X, Y, box, [-1, 6], 1, p.b, sqr(p.b));
        },
      }),
      readouts: [
        { label: String.raw`\(f(x_1)\)`, f: (p) => sqr(p.a), fmt: "f2" },
        { label: String.raw`\(f(x_2)\)`, f: (p) => sqr(p.b), fmt: "f2" },
        { label: "secant slope: rise / run", f: (p) => (Math.abs(p.b - p.a) < 1e-9 ? "none: x₁ = x₂" : fix((sqr(p.b) - sqr(p.a)) / (p.b - p.a), 3)), fmt: (v) => v },
        { label: String.raw`\(x_1 + x_2\)`, f: (p) => p.a + p.b, fmt: "f1" },
      ],
    },
    terms: [
      { term: "Average rate of change", def: "Rise over run between two inputs: \\((f(x_2) - f(x_1))/(x_2 - x_1).\\)" },
      { term: "Secant", def: "The straight line through two points of a curve; its slope is the average rate of change between them." },
    ],
  });

  Deck.slide({
    id: "math-limit",
    title: "Shrinking the gap gives the slope at a single point",
    body: String.raw`<p>Fix one point at \(x = 3\) and put the other at \(3 + h.\) For \(f(x) = x^2\) the secant slope is \(((3 + h)^2 - 9)/h = (6h + h^2)/h = 6 + h.\)</p>
<table><tr><th>h</th><td class="r">1</td><td class="r">0.1</td><td class="r">0.01</td><td class="r">0.001</td></tr><tr><th>slope</th><td class="r">7</td><td class="r">6.1</td><td class="r">6.01</td><td class="r">6.001</td></tr></table>
<p>As \(h\) shrinks the slope approaches 6, and getting there never needs \(h = 0\) itself. That limiting value is the <b>derivative</b> of \(f\) at 3, written \(f'(3) = 6,\) or \(\frac{df}{dx} = 6\) at \(x = 3.\)</p>
\[f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}\]
<p>The same algebra at any \(x\) gives \(((x + h)^2 - x^2)/h = 2x + h,\) so the derivative of \(x^2\) is \(2x\): 2 at \(x = 1,\) 6 at 3, −4 at −2. The derivative is itself a function: one slope for every input.</p>
<p>The limit must exist and be the same whether \(h\) shrinks from above or from below. For \(\max(0, x)\) at 0 the secant slopes are 1 from the right and 0 from the left, so the ReLU has no derivative at 0 (Part 3 deals with this).</p>`,
    fig: {
      type: "plot", caption: String.raw`<span>The secant from \(x = 3\) to \(3 + h\) (blue) and the line of slope 6 (dashed). Shrink \(h.\)</span>`,
      x: [1.5, 4.5], y: [0, 21], xlabel: "x", ylabel: "f(x)",
      params: [{ name: "k", label: "gap h", min: 0, max: 30, step: 1, value: 0, fmt: (v) => sig4(hOf(v)) }],
      fns: [
        { f: sqr, cls: "ink" },
        { f: (x) => 9 + 6 * (x - 3), cls: "s2 thin dashed" },
        { f: (x, p) => { const h = hOf(p.k); return 9 + ((sqr(3 + h) - 9) / h) * (x - 3); }, cls: "s1" },
      ],
      extra: (p) => ({
        after(svg, X, Y, pp, box) {
          const g = clipG(svg, box), h = hOf(p.k);
          g.appendChild(s("circle", { cx: X(3), cy: Y(9), r: 5.5, class: "math-pt ink" }));
          g.appendChild(s("circle", { cx: X(3 + h), cy: Y(sqr(3 + h)), r: 4.5, class: "math-pt" }));
          plabel(g, box, X(3) + 8, Y(9) + 16, "(3, 9)");
          if (h > 0.08) plabel(g, box, X(3 + h) - 8, Y(sqr(3 + h)) - 8, "(3 + h, (3 + h)²)", "end");
        },
      }),
      readouts: [
        { label: "gap h", f: (p) => sig4(hOf(p.k)), fmt: (v) => v },
        { label: String.raw`secant slope \(((3 + h)^2 - 9)/h\)`, f: (p) => { const h = hOf(p.k); return (sqr(3 + h) - 9) / h; }, fmt: "f6" },
        { label: "distance from 6", f: (p) => { const h = hOf(p.k); return (sqr(3 + h) - 9) / h - 6; }, fmt: "f6" },
      ],
    },
    terms: [{ term: "Derivative", def: "The slope of a function at a point: the limit of the secant slope \\((f(x + h) - f(x))/h\\) as \\(h\\) shrinks to 0, when that limit exists and is the same from both sides. Written \\(f'(x)\\) or \\(\\frac{df}{dx}.\\)" }],
  });

  Deck.slide({
    id: "math-tangent",
    title: "The derivative is the slope of the tangent line",
    body: String.raw`<p>The line through \((x_0, f(x_0))\) with slope \(f'(x_0)\) is the <b>tangent line</b>:</p>
\[y = f(x_0) + f'(x_0)\,(x - x_0)\]
<p>For \(f(x) = x^2\) at \(x_0 = 3\) it is \(y = 9 + 6(x - 3).\) It touches the curve at that point and has the curve's direction there.</p>
<p>Near \(x_0\) the tangent is an accurate approximation of the function. At \(x = 3.1\) the line gives \(9 + 6 \cdot 0.1 = 9.6\) and the curve gives \(3.1^2 = 9.61.\) The gap is 0.01, the square of the distance 0.1. For this function the gap is exactly \((x - x_0)^2,\) so it shrinks much faster than the distance: 0.0001 at a distance of 0.01.</p>
<p>This is what makes derivatives useful for training. The derivative predicts how much the output changes for a small change of the input, without trying the change.</p>`,
    fig: {
      type: "plot", caption: String.raw`<span>\(f(x) = x^2\) and its tangent line at \(x_0.\) Drag the point along the curve.</span>`,
      x: [-3, 5], y: [-6, 25], xlabel: "x", ylabel: "f(x)", xaxis: { fmt: tickMinus }, yaxis: { fmt: tickMinus },
      params: [{ name: "x0", label: "x₀", min: -2.5, max: 4.5, step: 0.1, value: 3, fmt: "f1" }],
      fns: [{ f: sqr, cls: "ink" }, { f: (x, p) => sqr(p.x0) + 2 * p.x0 * (x - p.x0), cls: "s2" }],
      extra: (p) => ({
        after(svg, X, Y, pp, box) {
          const g = clipG(svg, box);
          g.appendChild(s("circle", { cx: X(p.x0), cy: Y(sqr(p.x0)), r: 6, class: "math-pt" }));
          plabel(g, box, X(p.x0) + 12, Y(sqr(p.x0)) + 18, `slope ${fix(2 * p.x0, 1)}`);
          dragHandle(svg, X, Y, box, [-3, 5], 0, p.x0, sqr(p.x0));
        },
      }),
      readouts: [
        { label: String.raw`slope \(f'(x_0) = 2x_0\)`, f: (p) => 2 * p.x0, fmt: "f1" },
        { label: String.raw`tangent at \(x_0 + 0.1\)`, f: (p) => sqr(p.x0) + 2 * p.x0 * 0.1, fmt: "f3" },
        { label: String.raw`curve at \(x_0 + 0.1\)`, f: (p) => sqr(p.x0 + 0.1), fmt: "f3" },
        { label: "gap", f: (p) => sqr(p.x0 + 0.1) - (sqr(p.x0) + 2 * p.x0 * 0.1), fmt: "f4" },
      ],
    },
    terms: [{ term: "Tangent line", def: "The line through \\((x_0, f(x_0))\\) with slope \\(f'(x_0)\\): \\(y = f(x_0) + f'(x_0)(x - x_0).\\) The best straight-line approximation of \\(f\\) near \\(x_0.\\)" }],
  });

  Deck.slide({
    id: "math-rules",
    title: "Three rules give the derivative of any polynomial",
    body: String.raw`<p>Taking limits every time is slow. Three rules, each proved once from the limit definition, cover every polynomial:</p>
<ul>
<li><b>Power rule:</b> \(\frac{d}{dx} x^n = n\,x^{n-1}.\) So \((x^2)' = 2x,\) \((x^3)' = 3x^2,\) \((x)' = 1,\) and a constant has derivative 0.</li>
<li><b>Constant factor:</b> \((c\,f)' = c\,f'.\) Scaling a function scales its slope.</li>
<li><b>Sum rule:</b> \((f + g)' = f' + g'.\) The slopes of added functions add.</li>
</ul>
<p>Applied to \(f(x) = 3x^2 + 2x,\) the sum rule splits it, the constant factor rule pulls out 3 and 2, and the power rule finishes: \(f'(x) = 6x + 2.\) At \(x = 1\) that is 8.</p>
<p>Check with a small step: \(f(1.001) = 5.008003\) and \(f(1) = 5,\) so \((5.008003 - 5)/0.001 = 8.003.\) The extra 0.003 is \(3h,\) a term that vanishes as \(h\) shrinks.</p>`,
    fig: {
      type: "steps", caption: String.raw`<span>The derivative of \(3x^2 + 2x,\) one rule per line</span>`,
      steps: [
        { html: String.raw`\[f(x) = 3x^2 + 2x\]`, note: "the function" },
        { html: String.raw`\[f'(x) = (3x^2)' + (2x)'\]`, note: "sum rule" },
        { html: String.raw`\[= 3\,(x^2)' + 2\,(x)'\]`, note: "constant factor" },
        { html: String.raw`\[= 3 \cdot 2x + 2 \cdot 1 = 6x + 2\]`, note: "power rule" },
        { html: String.raw`\[f'(1) = 6 + 2 = 8\]`, note: "evaluate at x = 1" },
        { html: String.raw`\[\frac{f(1.001) - f(1)}{0.001} = \frac{5.008003 - 5}{0.001} = 8.003\]`, note: "numeric check: off by 3h" },
      ],
    },
    terms: [{ term: "Power rule", def: "\\(\\frac{d}{dx} x^n = n\\,x^{n-1}.\\) With the constant-factor and sum rules it differentiates every polynomial." }],
  });

  Deck.slide({
    id: "math-exp-ln",
    title: "The exponential is its own derivative; ln x has derivative 1/x",
    body: String.raw`<p>At every point the slope of the exponential equals its value; only constant multiples of \(e^x\) have that property.</p>
\[\frac{d}{dx}\, e^x = e^x \qquad \frac{d}{dx} \,\ln x = \frac{1}{x}\]
<p>Check at \(x = 1\): \((e^{1.001} - e^{1})/0.001 = 2.7196,\) against \(e = 2.7183.\) At \(x = 0\) the slope is \(e^0 = 1.\) This is why the base \(e = 2.71828\) is used throughout: any other base \(a\) gives \((a^x)' = a^x \,\ln a,\) with an extra factor.</p>
<p>The logarithm's slope \(1/x\) is steep near 0 and flattens as \(x\) grows. Check at \(x = 2\): \((\ln 2.001 - \ln 2)/0.001 = 0.4999,\) against \(1/2.\) At \(x = 0.1\) the slope is 10.</p>
<p>Both rules appear in every loss computation from Part 2 on: a probability is built with \(e^x\) and scored with \(\ln,\) and training multiplies their derivatives together.</p>`,
    fig: {
      type: "plot", caption: String.raw`<span>\(e^x\) (blue) and \(\ln x\) (red) with their tangent lines at \(x_0\) (dashed). Drag either point.</span>`,
      x: [-1.5, 3], y: [-3, 5.5], xlabel: "x", ylabel: "y", xaxis: { fmt: tickMinus }, yaxis: { fmt: tickMinus },
      params: [{ name: "x0", label: "x₀", min: 0.1, max: 1.6, step: 0.01, value: 1, fmt: "f2" }],
      fns: [
        { f: (x) => Math.exp(x), cls: "s1" },
        { f: (x) => (x > 0 ? Math.log(x) : NaN), cls: "s2", samples: 480 },
        { f: (x, p) => Math.exp(p.x0) * (1 + x - p.x0), cls: "s1 thin dashed" },
        { f: (x, p) => Math.log(p.x0) + (x - p.x0) / p.x0, cls: "s2 thin dashed" },
      ],
      extra: (p) => ({
        after(svg, X, Y, pp, box) {
          const g = clipG(svg, box);
          g.appendChild(s("circle", { cx: X(p.x0), cy: Y(Math.exp(p.x0)), r: 5.5, class: "math-pt s1" }));
          g.appendChild(s("circle", { cx: X(p.x0), cy: Y(Math.log(p.x0)), r: 5.5, class: "math-pt s2" }));
          plabel(g, box, X(-1.4), Y(Math.exp(-1.4)) - 8, "eˣ");
          plabel(g, box, X(2.6), Y(Math.log(2.6)) + 18, "ln x");
          dragHandle(svg, X, Y, box, [-1.5, 3], 0, p.x0, Math.exp(p.x0));
          dragHandle(svg, X, Y, box, [-1.5, 3], 0, p.x0, Math.log(p.x0));
        },
      }),
      readouts: [
        { label: String.raw`\(e^{x_0}\)`, f: (p) => Math.exp(p.x0), fmt: "f4" },
        { label: String.raw`slope of \(e^x\) (central difference)`, f: (p) => (Math.exp(p.x0 + 1e-5) - Math.exp(p.x0 - 1e-5)) / 2e-5, fmt: "f4" },
        { label: String.raw`\(1/x_0\)`, f: (p) => 1 / p.x0, fmt: "f4" },
        { label: String.raw`slope of \(\ln x\) (central difference)`, f: (p) => (Math.log(p.x0 + 1e-5) - Math.log(p.x0 - 1e-5)) / 2e-5, fmt: "f4" },
      ],
    },
  });

  // finite differences of e^x at x = 1, in float64 and emulated float32 (every operation rounded)
  const F32 = Math.fround;
  const FD = {
    f32: (h) => { const x = F32(1), hh = F32(h); return F32(F32(F32(Math.exp(F32(x + hh))) - F32(Math.exp(x))) / hh); },
    c32: (h) => { const x = F32(1), hh = F32(h); return F32(F32(F32(Math.exp(F32(x + hh))) - F32(Math.exp(F32(x - hh)))) / F32(2 * hh)); },
    f64: (h) => (Math.exp(1 + h) - Math.exp(1)) / h,
    c64: (h) => (Math.exp(1 + h) - Math.exp(1 - h)) / (2 * h),
  };
  const H_GRID = Array.from({ length: 241 }, (_, i) => Math.pow(10, -8 + (8 * i) / 240));
  const errSeries = (fn) => H_GRID.map((h) => [h, Math.max(Math.abs(fn(h) - Math.E), 1e-14)]);

  Deck.slide({
    id: "math-finite-diff",
    title: "Finite differences estimate a derivative; the central one is more accurate",
    body: String.raw`<p>When a function is available only as code, its derivative can be estimated from a few evaluations. The <b>forward difference</b> is the secant slope from \(x\) to \(x + h;\) the <b>central difference</b> steps \(h\) to both sides:</p>
\[\frac{f(x + h) - f(x)}{h} \qquad \frac{f(x + h) - f(x - h)}{2h}\]
<p>For \(e^x\) at \(x = 1,\) whose true derivative is \(e = 2.718282:\)</p>
<table><tr><th></th><th class="r">estimate, h = 0.1</th><th class="r">error</th><th class="r">error, h = 0.05</th></tr><tr><th>forward</th><td class="r">2.858842</td><td class="r">0.140560</td><td class="r">0.069104</td></tr><tr><th>central</th><td class="r">2.722815</td><td class="r">0.004533</td><td class="r">0.001133</td></tr></table>
<p>Halving \(h\) about halves the forward error and divides the central error by 4. The forward secant tilts with the curve's bend, which gives an error proportional to \(h;\) in the central difference the bends on the two sides cancel, leaving an error proportional to \(h^2.\)</p>
<p>Part 3 uses central differences to check that backpropagation computes the right gradients.</p>`,
    fig: {
      type: "plot", caption: String.raw`<span>\(e^x\) near \(x = 1.\) Red: forward secant. Blue: central secant. Dashed: the tangent, slope \(e.\) Set \(h = 0.1\) to reproduce the table.</span>`,
      x: [0.3, 1.7], y: [1, 5.6], xlabel: "x", ylabel: "eˣ",
      params: [{ name: "h", label: "step h", min: 0.01, max: 0.6, step: 0.01, value: 0.3, fmt: "f2" }],
      fns: [
        { f: (x) => Math.exp(x), cls: "ink" },
        { f: (x) => Math.E * x, cls: "ink thin dashed" },
        { f: (x, p) => Math.E + FD.f64(p.h) * (x - 1), cls: "s2" },
        { f: (x, p) => Math.exp(1 - p.h) + FD.c64(p.h) * (x - (1 - p.h)), cls: "s1" },
      ],
      extra: (p) => ({
        after(svg, X, Y, pp, box) {
          const g = clipG(svg, box);
          [[1 - p.h, "s1"], [1, "ink"], [1 + p.h, "s2"]].forEach(([x, c]) => g.appendChild(s("circle", { cx: X(x), cy: Y(Math.exp(x)), r: 4.5, class: "math-pt " + c })));
        },
      }),
      readouts: [
        { label: "forward estimate", f: (p) => FD.f64(p.h), fmt: "f6" },
        { label: "forward error", f: (p) => FD.f64(p.h) - Math.E, fmt: "f6" },
        { label: "central estimate", f: (p) => FD.c64(p.h), fmt: "f6" },
        { label: "central error", f: (p) => FD.c64(p.h) - Math.E, fmt: "f6" },
      ],
    },
    terms: [
      { term: "Forward difference", def: "The estimate \\((f(x + h) - f(x))/h\\) of a derivative; its error shrinks in proportion to \\(h.\\)" },
      { term: "Central difference", def: "The estimate \\((f(x + h) - f(x - h))/(2h)\\) of a derivative; its error shrinks in proportion to \\(h^2.\\)" },
    ],
  });

  Deck.slide({
    id: "math-roundoff",
    title: "Too small a step loses the derivative to rounding",
    body: String.raw`<p>The formulas suggest making \(h\) as small as possible. In floating point that fails. A float32 number carries about 7 significant digits: the gap between 1 and the next float32 number, the <b>machine epsilon</b>, is \(2^{-23} \approx 1.19 \times 10^{-7}.\) So \(f(x + h)\) and \(f(x)\) each carry a rounding error near \(10^{-7}\) of their size, and dividing their difference by a small \(h\) magnifies that error by \(1/h.\)</p>
<p>The plate computes both estimates for \(e^x\) at \(x = 1\) in float32, for \(h\) from \(10^{-8}\) to 1. Right of the dip the error falls as \(h\) shrinks; left of it, rounding error grows as \(1/h.\) The forward difference does best near \(h = 2 \times 10^{-4},\) where its error is about \(10^{-3};\) the central difference does best near \(h = 3 \times 10^{-3},\) with an error near \(10^{-5}.\) Below \(h = 2^{-24} \approx 6 \times 10^{-8},\) \(1 + h\) rounds to exactly 1 and the forward estimate is 0.</p>
<p>In float64 (about 16 digits) the dashed curves bottom out four to six orders of magnitude lower, which is why gradient checks run in float64.</p>`,
    fig: {
      type: "chart", caption: String.raw`<span>Error of the two estimates of the slope of \(e^x\) at \(x = 1\) against the step \(h.\) Both axes are logarithmic. Solid: float32. Dashed: float64.</span>`,
      x: { domain: [1e-8, 1], type: "log", ticks: [1e-8, 1e-6, 1e-4, 1e-2, 1], fmt: pow10, label: "step h" },
      y: { domain: [1e-12, 1e3], type: "log", ticks: [1e-12, 1e-9, 1e-6, 1e-3, 1, 1e3], fmt: pow10, label: "absolute error" },
      series: [
        { points: errSeries(FD.f64), cls: "s2 thin dashed" },
        { points: errSeries(FD.c64), cls: "s1 thin dashed" },
        { points: errSeries(FD.f32), cls: "s2" },
        { points: errSeries(FD.c32), cls: "s1" },
      ],
      vlines: [{ x: Math.pow(2, -24), label: "1 + h = 1" }],
      legend: [["s2", "forward difference"], ["s1", "central difference"]],
    },
    terms: [{ term: "Machine epsilon", def: "The gap between 1 and the next representable number: \\(2^{-23} \\approx 1.19 \\times 10^{-7}\\) in float32, \\(2^{-52} \\approx 2.2 \\times 10^{-16}\\) in float64." }],
    src: ["IEEE Std 754-2019, IEEE Standard for Floating-Point Arithmetic (binary32: 24-bit significand)"],
  });

  Deck.slide({
    id: "math-minimum",
    title: "The sign of the derivative says which way is down",
    body: String.raw`<p>If \(f'(x) \lt 0,\) the function falls as \(x\) increases, so a small step to the right lowers it. If \(f'(x) \gt 0,\) a step to the left lowers it. Either way, stepping against the sign of the derivative goes downhill.</p>
<p>For \(f(x) = (x - 3)^2 + 1\) the derivative is \(f'(x) = 2(x - 3).\) At \(x = 1,\) \(f' = {-4},\) so increase \(x\): \(f(1.1) = 4.61\) is below \(f(1) = 5.\) At \(x = 5,\) \(f' = 4,\) so decrease \(x.\) At \(x = 3,\) \(f' = 0\) and the function has its <b>minimum</b>, \(f(3) = 1.\)</p>
<p>Where a function has a derivative, the derivative at a minimum is 0, but a zero derivative alone does not prove a minimum: \(x^3\) has \(f'(0) = 0\) at a flat point that is neither a low nor a high. The next sheet shows how to tell a minimum from a maximum.</p>
<p>Gradient descent (Part 2) is this rule repeated: compute the derivative, step against it, and stop when it is near 0.</p>`,
    fig: {
      type: "plot", caption: String.raw`<span>\(f(x) = (x - 3)^2 + 1.\) The arrow points the way that lowers \(f\): against the sign of \(f'.\)</span>`,
      x: [-1, 7], y: [0, 18], xlabel: "x", ylabel: "f(x)", xaxis: { fmt: tickMinus },
      params: [{ name: "x0", label: "x", min: -0.5, max: 6.5, step: 0.05, value: 1, fmt: "f2" }],
      fns: [{ f: (x) => sqr(x - 3) + 1, cls: "ink" }, { f: (x, p) => sqr(p.x0 - 3) + 1 + 2 * (p.x0 - 3) * (x - p.x0), cls: "s2 thin" }],
      extra: (p) => ({
        after(svg, X, Y, pp, box) {
          const g = clipG(svg, box), y0 = sqr(p.x0 - 3) + 1, d = 2 * (p.x0 - 3);
          if (Math.abs(d) > 1e-9) {
            const dir = d > 0 ? -1 : 1, len = Math.min(1.4, 0.35 + Math.abs(d) * 0.2);
            g.appendChild(s("line", { x1: X(p.x0), y1: Y(y0) - 18, x2: X(p.x0 + dir * len), y2: Y(y0) - 18, class: "math-arrow red", "marker-end": arrowHead(svg, "red") }));
            plabel(g, box, X(p.x0 + dir * len) + (dir > 0 ? 6 : -6), Y(y0) - 14, "lower", dir > 0 ? "start" : "end", "red");
          } else plabel(g, box, X(p.x0), Y(y0) - 16, "f′ = 0: minimum", "middle", "red");
          g.appendChild(s("circle", { cx: X(p.x0), cy: Y(y0), r: 6, class: "math-pt" }));
          dragHandle(svg, X, Y, box, [-1, 7], 0, p.x0, y0);
        },
      }),
      readouts: [
        { label: "f(x)", f: (p) => sqr(p.x0 - 3) + 1, fmt: "f4" },
        { label: "f′(x) = 2(x − 3)", f: (p) => 2 * (p.x0 - 3), fmt: "f2" },
        { label: "step that lowers f", f: (p) => { const d = 2 * (p.x0 - 3); return Math.abs(d) < 1e-9 ? "none: minimum" : d < 0 ? "increase x" : "decrease x"; }, fmt: (v) => v },
      ],
    },
    terms: [{ term: "Minimum", def: "An input where a function is at least as low as at every nearby input (a local minimum); the global minimum is the lowest value of all. Where the function has a derivative, it is 0 at a minimum." }],
  });

  Deck.slide({
    id: "math-second",
    title: "The second derivative measures how the slope bends",
    body: String.raw`<p>The derivative of the derivative is the <b>second derivative</b>, \(f''(x).\) It says how fast the slope changes: positive means the slope is increasing and the curve bends upward, negative means it bends downward. Its size measures how sharply the curve bends, which optimization texts call the <b>curvature</b>.</p>
<p>For \(f(x) = x^3 - 3x,\) \(f'(x) = 3x^2 - 3\) and \(f''(x) = 6x.\) The slope is 0 at \(x = {-1}\) and at \(x = 1.\) At \(x = 1,\) \(f'' = 6 \gt 0\): the curve bends up, so this is a local minimum, \(f(1) = {-2},\) though not the lowest value overall: \(f({-3}) = {-18}.\) At \(x = {-1},\) \(f'' = {-6}\): a local maximum, \(f({-1}) = 2.\) When \(f'' = 0\) too, the test says nothing: \(x^3\) and \(x^4\) both have \(f'(0) = f''(0) = 0,\) and only \(x^4\) has a minimum there.</p>
<p>Curvature also limits how far a step can go. On \((x - 3)^2 + 1,\) \(f'' = 2\) everywhere. Part 2 shows that a step of \(\eta\) times the derivative there multiplies the distance to the minimum by \(1 - 2\eta,\) so any \(\eta \gt 1\) overshoots further than it started. For any quadratic the limit is \(\eta \lt 2/f'':\) the sharper the bend, the smaller the step.</p>`,
    fig: {
      type: "plot", caption: String.raw`<span>\(f(x) = x^3 - 3x\) and its tangent at \(x_0.\) Left of 0, \(f'' \lt 0\) and the curve bends down; right of 0 it bends up.</span>`,
      x: [-2.5, 2.5], y: [-5, 5], xlabel: "x", ylabel: "f(x)", xaxis: { fmt: tickMinus }, yaxis: { fmt: tickMinus },
      params: [{ name: "x0", label: "x₀", min: -2.2, max: 2.2, step: 0.05, value: 1, fmt: "f2" }],
      fns: [{ f: (x) => x ** 3 - 3 * x, cls: "ink" }, { f: (x, p) => p.x0 ** 3 - 3 * p.x0 + (3 * sqr(p.x0) - 3) * (x - p.x0), cls: "s2 thin" }],
      vlines: [{ x: 0 }],
      extra: (p) => ({
        dots: [{ x: -1, y: 2, r: 4.5, cls: "ink", tip: "local maximum: f(−1) = 2, f″(−1) = −6" }, { x: 1, y: -2, r: 4.5, cls: "ink", tip: "local minimum: f(1) = −2, f″(1) = 6" }],
        after(svg, X, Y, pp, box) {
          const g = clipG(svg, box), y0 = p.x0 ** 3 - 3 * p.x0;
          plabel(g, box, X(-0.15), box.m.t + 14, "f″ < 0: bends down", "end", "dim");
          plabel(g, box, X(0.15), box.m.t + 14, "f″ > 0: bends up", "start", "dim");
          plabel(g, box, X(-1), Y(2) - 10, "local max", "middle");
          plabel(g, box, X(1), Y(-2) + 20, "local min", "middle");
          g.appendChild(s("circle", { cx: X(p.x0), cy: Y(y0), r: 6, class: "math-pt" }));
          dragHandle(svg, X, Y, box, [-2.5, 2.5], 0, p.x0, y0);
        },
      }),
      readouts: [
        { label: String.raw`\(f'(x_0) = 3x_0^2 - 3\)`, f: (p) => 3 * sqr(p.x0) - 3, fmt: "f3" },
        { label: String.raw`\(f''(x_0) = 6x_0\)`, f: (p) => 6 * p.x0, fmt: "f2" },
        { label: "the curve", f: (p) => (Math.abs(p.x0) < 1e-9 ? "changes bend" : p.x0 > 0 ? "bends up" : "bends down"), fmt: (v) => v },
      ],
    },
    terms: [
      { term: "Second derivative", def: "The derivative of the derivative, \\(f''(x):\\) how fast the slope changes. Positive where the curve bends up." },
      { term: "Curvature", def: "How sharply a curve bends; for a function of one input, measured here by the second derivative." },
    ],
    check: {
      q: String.raw`For \(f(x) = x^2 - 4x,\) which statement is true at \(x = 1\)?`,
      options: [
        String.raw`\(f'(1) = {-2},\) so increasing \(x\) lowers \(f\)`,
        String.raw`\(f'(1) = 2,\) so increasing \(x\) lowers \(f\)`,
        String.raw`\(f'(1) = {-3},\) so decreasing \(x\) lowers \(f\)`,
        String.raw`\(f'(1) = 0,\) so \(x = 1\) is the minimum`,
      ],
      answer: 0,
      why: String.raw`By the power and sum rules \(f'(x) = 2x - 4,\) so \(f'(1) = {-2}.\) A negative slope means \(f\) falls to the right: \(f(1.1) = {-3.19}\) is below \(f(1) = {-3}.\) The value \({-3}\) is \(f(1),\) not the slope, and the minimum is where \(2x - 4 = 0,\) at \(x = 2.\)`,
    },
  });

  // ------------------------------------------------------------ 1.3 The chain rule
  Deck.chapter({ title: "The chain rule" });

  Deck.slide({
    id: "math-chain-rule",
    title: "The chain rule: the slope of a composition is a product of slopes",
    body: String.raw`<p>If \(y = f(u)\) and \(u = g(x),\) a small change \(\Delta x\) changes \(u\) by about \(g'(x)\,\Delta x\) (the tangent-line approximation), and that change moves \(y\) by about \(f'(u)\) times as much. So the slopes multiply. This is the <b>chain rule</b>:</p>
\[\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = f'(g(x))\, g'(x)\]
<p>Take \(y = (3x + 1)^2,\) so \(u = 3x + 1\) and \(y = u^2.\) The chain rule gives \(2u \cdot 3 = 6(3x + 1) = 18x + 6.\) Expanding first gives the same: \(y = 9x^2 + 6x + 1,\) and the power and sum rules give \(18x + 6.\)</p>
<p>At \(x = 2\): \(u = 7,\) \(dy/du = 14,\) \(du/dx = 3,\) so \(dy/dx = 42.\) Numeric check: \(((3 \cdot 2.001 + 1)^2 - 49)/0.001 = 42.009.\)</p>
<p>Expanding works here. For a composition of hundreds of functions it does not, and the chain rule is the only practical route.</p>`,
    fig: {
      type: "plot", caption: String.raw`<span>\(y = (3x + 1)^2\) and its tangent. The readouts multiply the two local slopes and compare with a central difference.</span>`,
      x: [-1.5, 2.5], y: [-8, 72], xlabel: "x", ylabel: "y", xaxis: { fmt: tickMinus }, yaxis: { fmt: tickMinus },
      params: [{ name: "x0", label: "x", min: -1.4, max: 2.4, step: 0.1, value: 2, fmt: "f1" }],
      fns: [{ f: (x) => sqr(3 * x + 1), cls: "ink" }, { f: (x, p) => sqr(3 * p.x0 + 1) + 6 * (3 * p.x0 + 1) * (x - p.x0), cls: "s2 thin" }],
      extra: (p) => ({
        after(svg, X, Y, pp, box) {
          const g = clipG(svg, box), y0 = sqr(3 * p.x0 + 1);
          g.appendChild(s("circle", { cx: X(p.x0), cy: Y(y0), r: 6, class: "math-pt" }));
          dragHandle(svg, X, Y, box, [-1.5, 2.5], 0, p.x0, y0);
        },
      }),
      readouts: [
        { label: String.raw`\(u = 3x + 1\)`, f: (p) => 3 * p.x0 + 1, fmt: "f1" },
        { label: String.raw`\(dy/du = 2u\)`, f: (p) => 2 * (3 * p.x0 + 1), fmt: "f1" },
        { label: String.raw`\(dy/du \cdot du/dx\)`, f: (p) => 2 * (3 * p.x0 + 1) * 3, fmt: "f3" },
        { label: "central difference", f: (p) => (sqr(3 * (p.x0 + 1e-5) + 1) - sqr(3 * (p.x0 - 1e-5) + 1)) / 2e-5, fmt: "f3" },
      ],
    },
    terms: [{ term: "Chain rule", def: "The derivative of a composition is the product of the derivatives of its stages: \\(\\frac{dy}{dx} = \\frac{dy}{du} \\cdot \\frac{du}{dx}.\\)" }],
  });

  Deck.slide({
    id: "math-chain-three",
    title: "A longer chain multiplies one local slope per stage",
    body: String.raw`<p>The rule extends to any number of stages. Each stage contributes the slope of its own function at its own input, its <b>local derivative</b>, and the slope of the whole chain is their product.</p>
<p>The plate chains three stages: \(u = 3x + 1,\) then \(v = u^2,\) then \(y = \ln v.\) At \(x = 1\) the forward values are \(u = 4,\) \(v = 16\) and \(y = \ln 16 = 2.7726,\) and the local derivatives are 3, \(2u = 8\) and \(1/v = 0.0625.\) Their product is \(3 \times 8 \times 0.0625 = 1.5.\)</p>
<p>Directly: \(y = \ln\big((3x + 1)^2\big) = 2 \,\ln(3x + 1),\) so \(dy/dx = 6/(3x + 1) = 6/4 = 1.5.\)</p>
<p>Each local derivative needs only its own stage's input, which the forward computation already produced. That is the recipe used for every model in this set: run the chain forward and keep every intermediate value, then multiply the local derivatives.</p>`,
    fig: {
      type: "math-pipe", caption: "Three stages, their values at x and their local derivatives. The product of the local derivatives matches the slope of the whole chain at every x.",
      x: { label: "input x", min: 0, max: 3, step: 0.05, value: 1, dp: 2 },
      dp: 4, slopes: true,
      stages: [
        { v: "u", tex: "3x + 1", f: (x) => 3 * x + 1, df: () => 3, dtex: "3", ddp: 0, dconst: true },
        { v: "v", tex: "u^2", f: (u) => u * u, df: (u) => 2 * u, dtex: "2u", ddp: 3 },
        { v: "y", tex: "\\ln v", f: (v) => Math.log(v), df: (v) => 1 / v, dtex: "1/v", ddp: 4 },
      ],
    },
    terms: [{ term: "Local derivative", def: "The derivative of one stage of a computation with respect to its own input, evaluated at the value that stage received." }],
  });

  Deck.slide({
    id: "math-graph",
    title: "A computation is a graph, evaluated forward and then backward",
    body: String.raw`<p>Write every elementary operation as a node and every value it uses as an incoming edge. The result is a <b>computational graph</b>. The plate draws \(y = \ln((wx + b)^2)\) with \(w = 3,\) \(b = 1\) and \(x = 1\): the chain of the last sheet, with the 3 and the 1 named as parameters, the way a model's numbers are.</p>
<p>The <b>forward pass</b> evaluates the nodes from inputs to output: \(m = 3,\) \(u = 4,\) \(v = 16,\) \(y = 2.7726.\) The <b>backward pass</b> starts from slope 1 at the output and walks the edges in reverse, multiplying by each node's local derivative: \(1 \times 0.0625 \times 8 = 0.5\) reaches \(u.\)</p>
<p>From \(u\) the derivative passes along both incoming edges. It reaches \(b\) as 0.5 and \(m\) as 0.5. From \(m = w \cdot x\) it reaches \(w\) as \(0.5 \times x = 0.5\) and \(x\) as \(0.5 \times w = 1.5,\) the slope of the last sheet. One backward pass gives the derivative of the output with respect to every input.</p>
<p>The plate writes grad(v) for the derivative of \(y\) with respect to a node \(v,\) and \(\partial v/\partial u\) for the local slope from \(u\) to \(v;\) \(\partial\) marks a derivative with respect to one of several inputs (chapter 1.4).</p>`,
    fig: {
      type: "graph", caption: String.raw`<span>\(y = \ln((wx + b)^2)\) at \(w = 3,\) \(x = 1,\) \(b = 1,\) after the forward pass. Step backward, or reset and step forward.</span>`,
      dp: 4, gx: 122, gy: 82, bw: 98, start: 4,
      nodes: [
        { id: "w", op: "param", value: 3, x: 0, y: 0 },
        { id: "x", op: "input", value: 1, x: 0, y: 1 },
        { id: "b", op: "param", value: 1, x: 0, y: 2 },
        { id: "m", label: "m", op: "mul", in: ["w", "x"], x: 1, y: 0.5 },
        { id: "u", label: "u", op: "add", in: ["m", "b"], x: 2, y: 1 },
        { id: "v", label: "v", op: "square", in: ["u"], x: 3, y: 1 },
        { id: "y", label: "y", op: "log", in: ["v"], x: 4, y: 1 },
      ],
    },
    terms: [
      { term: "Computational graph", def: "A computation drawn as nodes (operations) and edges (the values they use). Evaluated forward for values and backward for derivatives." },
      { term: "Forward pass", def: "Evaluating a computational graph from its inputs to its output, keeping every intermediate value." },
      { term: "Backward pass", def: "Walking a computational graph from the output back to the inputs, multiplying local derivatives, to get the derivative of the output with respect to every input." },
    ],
  });

  Deck.slide({
    id: "math-two-paths",
    title: "When a value is used twice, the paths add",
    body: String.raw`<p>A value can feed several operations. In \(y = x \cdot x,\) \(x\) reaches the product along two edges. A small change in \(x\) moves both factors and the two effects add: each edge contributes the other factor, so \(dy/dx = x + x = 2x,\) the power rule's answer.</p>
<p>In general, the derivative of the output with respect to a value is a <b>sum over paths</b>: for each path from the value to the output, multiply the local derivatives along it, then add the paths.</p>
<p>The plate uses \(y = x \cdot e^x.\) One path goes straight into the product and contributes the other factor, \(e^x.\) The other passes through \(s = e^x\) and contributes \(x \cdot e^x.\) At \(x = 1\) that is \(e + e = 2e = 5.4366.\) The product rule, \((fg)' = f'g + fg',\) is this sum over two paths.</p>
<p>Numeric check: \((1.001\,e^{1.001} - 0.999\,e^{0.999})/0.002 = 5.4366.\) Backpropagation implements the sum with <code>+=</code>: each path adds its share into the derivative stored for the shared value.</p>`,
    fig: {
      type: "graph", caption: String.raw`<span>\(y = x \cdot e^x\) at \(x = 1.\) Step backward: the derivative stored at \(x\) grows by one path at a time.</span>`,
      dp: 4, gx: 190, gy: 92, bw: 110, start: 2,
      nodes: [
        { id: "x", op: "input", value: 1, x: 0, y: 1 },
        { id: "s", label: "s", op: "exp", in: ["x"], x: 1, y: 0 },
        { id: "y", label: "y", op: "mul", in: ["s", "x"], x: 2, y: 1 },
      ],
    },
    terms: [{ term: "Sum over paths", def: "The derivative of an output with respect to a value used more than once: multiply local derivatives along each path from the value to the output, then add the paths." }],
  });

  Deck.slide({
    id: "math-chain-why",
    title: "A neural network is one long chain, and training runs it backward",
    body: String.raw`<p>Our model computes each next-token probability through one long composition. It looks up 16 numbers for each token and adds the 16 numbers of its position. Each of its 2 layers then normalizes, multiplies by matrices, mixes information across positions, adds, and applies a nonlinear function. A final normalization and multiplication give 113 scores, which become probabilities. Every step is a node of one computational graph whose inputs include all 9,968 parameters (Parts 3 to 5 define each step).</p>
<p>Training needs the derivative of the loss with respect to every parameter. Finite differences need one extra forward pass per parameter: 9,969 forward passes in all with forward differences, 19,936 with central ones. Running the chain rule backward through the graph gives all 9,968 derivatives exactly, for about the cost of two more forward passes (Part 3 counts it). For Llama 3 8B, with about 8 billion parameters, that is billions of forward passes against three.</p>
<p>This backward pass through a network is <b>backpropagation</b>. Part 3 builds it layer by layer.</p>`,
    fig: {
      type: "diagram", caption: "Our model as one chain. Forward (left): values flow down. Backward (right, red): derivatives flow up from the loss, and each block with parameters hands back one derivative per parameter.", w: 14, h: 12.7,
      nodes: [
        { id: "ids", x: 0.3, y: 0.3, w: 5.4, h: 1.2, label: "token ids", sub: "8 inputs" },
        { id: "emb", x: 0.3, y: 2.1, w: 5.4, h: 1.2, label: "embed + position", sub: "1,936 params", kind: "blue" },
        { id: "l0", x: 0.3, y: 3.9, w: 5.4, h: 1.2, label: "layer 0", sub: "3,104 params", kind: "blue" },
        { id: "l1", x: 0.3, y: 5.7, w: 5.4, h: 1.2, label: "layer 1", sub: "3,104 params", kind: "blue" },
        { id: "out", x: 0.3, y: 7.5, w: 5.4, h: 1.2, label: "norm, × W_U", sub: "1,824 params", kind: "blue" },
        { id: "sm", x: 0.3, y: 9.3, w: 5.4, h: 1.2, label: "softmax", sub: "113 probabilities" },
        { id: "loss", x: 0.3, y: 11.1, w: 5.4, h: 1.2, label: "loss", sub: "average of −ln p", kind: "hi" },
        { id: "bloss", x: 8.0, y: 11.1, w: 5.7, h: 1.2, label: "∂loss/∂loss = 1", sub: "start", kind: "ghost" },
        { id: "bsm", x: 8.0, y: 9.3, w: 5.7, h: 1.2, label: "∂loss/∂scores", sub: "113 numbers per position", kind: "ghost" },
        { id: "bout", x: 8.0, y: 7.5, w: 5.7, h: 1.2, label: "1,824 derivatives", sub: "norm gains and W_U", kind: "ghost" },
        { id: "bl1", x: 8.0, y: 5.7, w: 5.7, h: 1.2, label: "3,104 derivatives", sub: "layer 1 weights", kind: "ghost" },
        { id: "bl0", x: 8.0, y: 3.9, w: 5.7, h: 1.2, label: "3,104 derivatives", sub: "layer 0 weights", kind: "ghost" },
        { id: "bemb", x: 8.0, y: 2.1, w: 5.7, h: 1.2, label: "1,936 derivatives", sub: "token and position vectors", kind: "ghost" },
      ],
      edges: [
        { from: "ids", to: "emb" }, { from: "emb", to: "l0" }, { from: "l0", to: "l1" }, { from: "l1", to: "out" }, { from: "out", to: "sm" }, { from: "sm", to: "loss" },
        { from: "loss", to: "bloss", kind: "red", dashed: true },
        { from: "bloss", to: "bsm", kind: "red", dashed: true }, { from: "bsm", to: "bout", kind: "red", dashed: true },
        { from: "bout", to: "bl1", kind: "red", dashed: true }, { from: "bl1", to: "bl0", kind: "red", dashed: true }, { from: "bl0", to: "bemb", kind: "red", dashed: true },
      ],
      notes: [{ x: 8.0, y: 0.9, text: "1,936 + 3,104 + 3,104 + 1,824 = 9,968" }],
    },
    terms: [{ term: "Backpropagation", def: "The backward pass of the chain rule through a neural network: it computes the derivative of the loss with respect to every parameter for about the cost of two forward passes." }],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Llama 3 8B: 8 billion parameters)"],
    check: {
      q: String.raw`\(y = (2x + 1)^3.\) What is \(dy/dx\) at \(x = 1\)?`,
      options: ["54", "27", "18", "9"],
      answer: 0,
      why: String.raw`With \(u = 2x + 1 = 3\) and \(y = u^3\): \(dy/du = 3u^2 = 27\) and \(du/dx = 2,\) so \(dy/dx = 27 \times 2 = 54.\) Forgetting the inner slope 2 gives 27; 9 is \(u^2\) alone.`,
    },
  });

  // ------------------------------------------------------------ 1.4 Partial derivatives and the gradient
  Deck.chapter({ title: "Partial derivatives and the gradient" });

  const bowl = (x, y) => x * x + 3 * y * y;

  Deck.slide({
    id: "math-partial",
    title: "A partial derivative varies one input and holds the others fixed",
    body: String.raw`<p>For a function of several inputs, ask how it changes when one input moves and the others stay put. That is the <b>partial derivative</b>, written \(\partial f/\partial x.\) Holding \(y\) fixed turns \(f(x, y)\) into an ordinary function of \(x,\) and the rules of chapter 1.2 apply.</p>
<p>For \(f(x, y) = x^2 + 3y^2\): with \(y\) fixed, \(3y^2\) is a constant with derivative 0, so \(\partial f/\partial x = 2x.\) With \(x\) fixed, \(\partial f/\partial y = 6y.\)</p>
<p>At \((1, 1)\): \(\partial f/\partial x = 2\) and \(\partial f/\partial y = 6.\) Check: \(f(1.001, 1) = 4.002001,\) so \((4.002001 - 4)/0.001 = 2.001;\) and \(f(1, 1.001) = 4.006003\) gives 6.003.</p>
<p>On the plate, the blue line through the point is the slice where only \(x\) varies, and the red line the slice where only \(y\) varies. Each partial derivative is the slope of its slice, drawn below the map.</p>`,
    fig: { type: "math-contour", mode: "partial", point: [1, 1], caption: String.raw`<span>\(f(x, y) = x^2 + 3y^2\) with level curves at 2, 4, 6, 8, 10 and 12. Drag the point; the two slices through it are plotted below.</span>` },
    terms: [{ term: "Partial derivative", def: "The derivative of a function of several inputs with respect to one of them, the others held fixed. Written \\(\\partial f/\\partial x.\\)" }],
  });

  Deck.slide({
    id: "math-gradient",
    title: "The gradient collects all the partial derivatives into one vector",
    body: String.raw`<p>List the partial derivatives in order and you have the <b>gradient</b>, written \(\nabla f\) ("nabla f"):</p>
\[\nabla f(x, y) = \left( \frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y} \right) = (2x,\ 6y)\]
<p>At \((1, 1)\) it is \((2, 6);\) at \((2, {-1})\) it is \((4, {-6});\) at the lowest point, \((0, 0),\) it is \((0, 0).\) It has one entry per input and it changes from point to point: the gradient is itself a function, from points to lists of slopes.</p>
<p>A list of numbers like this is a vector, and chapter 1.5 works with vectors directly. The plate computes the gradient at any point twice, from the formulas and from central differences with \(h = 10^{-4},\) and the two agree at every slider position.</p>`,
    fig: {
      type: "calc", caption: String.raw`<span>The gradient of \(f(x, y) = x^2 + 3y^2,\) by formula and by central differences</span>`,
      inputs: [
        { id: "x", label: "x", min: -3, max: 3, step: 0.1, value: 1, fmt: "f1" },
        { id: "y", label: "y", min: -2, max: 2, step: 0.1, value: 1, fmt: "f1" },
      ],
      outputs: [
        { label: String.raw`\(f(x, y)\)`, f: (v) => bowl(v.x, v.y), fmt: "f3" },
        { label: String.raw`\(\nabla f = (2x, 6y)\)`, f: (v) => `(${fix(2 * v.x, 2)}, ${fix(6 * v.y, 2)})`, fmt: (t) => t },
        { label: String.raw`\(\partial f/\partial x,\) central difference`, f: (v) => (bowl(v.x + 1e-4, v.y) - bowl(v.x - 1e-4, v.y)) / 2e-4, fmt: "f6" },
        { label: String.raw`\(\partial f/\partial y,\) central difference`, f: (v) => (bowl(v.x, v.y + 1e-4) - bowl(v.x, v.y - 1e-4)) / 2e-4, fmt: "f6" },
      ],
      note: (v) => `∂f/∂x: (f(${fix(v.x + 1e-4, 4)}, ${fix(v.y, 1)}) − f(${fix(v.x - 1e-4, 4)}, ${fix(v.y, 1)})) / 0.0002. For a quadratic the central difference is exact up to rounding.`,
    },
    terms: [{ term: "Gradient", def: "The vector of all partial derivatives of a function, \\(\\nabla f = (\\partial f/\\partial x_1, \\ldots, \\partial f/\\partial x_n);\\) it points in the direction of steepest ascent." }],
  });

  Deck.slide({
    id: "math-directional",
    title: "The slope in any direction is the gradient times that direction",
    body: String.raw`<p>Step a small distance \(t\) from \((x, y)\) in the direction \(\mathbf{u} = (u_1, u_2),\) a direction of length 1. The inputs move by \(t u_1\) and \(t u_2,\) and the function changes by about the sum of the two partial effects:</p>
\[\Delta f \approx \frac{\partial f}{\partial x}\, t\,u_1 + \frac{\partial f}{\partial y}\, t\,u_2\]
<p>Dividing by \(t\) gives the <b>directional derivative</b>, the slope along \(\mathbf{u}\): \(D_{\mathbf{u}} f = \frac{\partial f}{\partial x} u_1 + \frac{\partial f}{\partial y} u_2.\) Chapter 1.5 names this sum of products the dot product, \(\nabla f \cdot \mathbf{u}.\)</p>
<p>At \((1, 1),\) with \(\nabla f = (2, 6)\) and \(\mathbf{u} = (\cos\theta, \sin\theta)\): along \(x,\) at \(\theta = 0^\circ,\) the slope is 2, and along \(y,\) at \(90^\circ,\) it is 6. It peaks at \(\theta = 71.57^\circ,\) the gradient's own direction, where it equals \(\sqrt{2^2 + 6^2} = 6.325,\) the gradient's length. At \(161.57^\circ\) it is 0: that direction runs along the level curve.</p>`,
    fig: { type: "math-contour", mode: "directional", point: [1, 1], angle: 30, caption: String.raw`<span>Red arrow: the direction \(\mathbf{u}\) (drag its tip or use the slider). Blue: \(\nabla f.\) Below: the slope along \(\mathbf{u}\) for every direction.</span>` },
    terms: [{ term: "Directional derivative", def: "The slope of a function along a direction \\(\\mathbf{u}\\) of length 1: \\(\\nabla f \\cdot \\mathbf{u}.\\) Largest along the gradient, where it equals the gradient's length." }],
  });

  Deck.slide({
    id: "math-steepest",
    title: "The gradient points uphill, at right angles to the level curves",
    body: String.raw`<p>The last sheet's curve peaks where \(\mathbf{u}\) points along \(\nabla f,\) at a slope equal to the gradient's length; chapter 1.5 shows why this holds at every point (the slope is \(|\nabla f| \cos\theta\)). So the gradient's direction is the <b>direction of steepest ascent</b>: of all directions you could step from a point, a small step along \(\nabla f\) raises \(f\) the most. The opposite direction, \({-\nabla f},\) lowers \(f\) the most. Plain gradient descent (Part 2) steps that way; Adam, which trained our model, rescales each entry of the step first (Part 3).</p>
<p>The gradient is also perpendicular to the level curve through its point: along the level curve \(f\) does not change, and the slope along a direction is 0 only at right angles to \(\nabla f.\)</p>
<p>At \((1, 1)\) the gradient \((2, 6)\) points mostly along \(y,\) where \(f\) climbs 3 times faster per unit than along \(x.\) The level curve there, \(x^2 + 3y^2 = 4,\) runs in the direction \((3, {-1}),\) at a right angle to \((2, 6).\) Drag the point: the arrows always cross the ellipses at right angles, and they are longest where the ellipses crowd together.</p>`,
    fig: { type: "math-contour", mode: "gradient", point: [1, 1], caption: String.raw`<span>Gray arrows: the gradient at grid points (scaled by 0.05). Blue: the gradient at the point (scaled by 0.12). Red: the level curve through the point and its tangent.</span>` },
    terms: [{ term: "Direction of steepest ascent", def: "The direction in which a small step raises a function the most: the direction of its gradient. The opposite direction is that of steepest descent." }],
  });

  Deck.slide({
    id: "math-many",
    title: "Our model's gradient has 9,968 entries",
    example: "tokens ids",
    body: String.raw`<p>Nothing changes with more inputs. A function of \(n\) inputs has \(n\) partial derivatives, and its gradient is a list of \(n\) numbers. Each answers one question: if this input moves a little, how fast does the output move?</p>
<p>The loss of our model on one query is the average of \({-\ln p}\) over its 8 predictions (Part 2 defines it). With the query held fixed, it is a function of the 9,968 parameters, so its gradient has 9,968 entries. The plate estimates 16 of them, the partial derivatives with respect to the 16 numbers of one token's vector, by central differences: change one number by \(\pm 10^{-4},\) rerun the model, divide.</p>
<p>For <code>name</code> after training (step 6,000) the largest of the 16 is 0.0007 in size. At step 400, early in training, the largest is 0.0750, about 100 times as large. Each number is the answer to its question for this query; the sign says which way to move that one parameter to lower the loss.</p>
<p>These 16 estimates cost 32 forward passes; all 9,968 would cost 19,936. Backpropagation (Part 3) gets them all from one backward pass.</p>`,
    fig: { type: "math-modelgrad", caption: "Partial derivatives of the loss on our query with respect to one token's 16 numbers, by central differences on the real model. Hover a bar for its value.", steps: [400, 6000], step: 6000, pos: 1 },
    check: {
      q: String.raw`\(f(x, y) = x^2 y.\) What is \(\nabla f\) at \((2, 3)\)?`,
      options: [String.raw`\((12, 4)\)`, String.raw`\((4, 12)\)`, String.raw`\((12, 0)\)`, "16"],
      answer: 0,
      why: String.raw`\(\partial f/\partial x\) holds \(y = 3\) fixed: \(2xy = 12.\) \(\partial f/\partial y\) holds \(x = 2\) fixed: \(x^2 = 4.\) So \(\nabla f = (12, 4).\) \((4, 12)\) swaps the entries; 16 adds them, but the gradient keeps one entry per input.`,
    },
  });

  // ------------------------------------------------------------ 1.5 Vectors
  Deck.chapter({ title: "Vectors" });

  const WTE = () => M.weights().wte;
  const COLS16 = Array.from({ length: 16 }, (_, j) => String(j + 1));

  Deck.slide({
    id: "math-vector",
    title: "A vector is a list of numbers; our model stores each token as 16 of them",
    layout: "wide",
    example: "tokens ids focus:1",
    body: String.raw`<p>A <b>vector</b> is an ordered list of numbers, its <b>entries</b>. The number of entries is its <b>dimension</b>. \(\mathbf{a} = (3, 1)\) has dimension 2 and can be drawn as an arrow from the origin to the point \((3, 1).\) Entry \(i\) is written \(a_i\): here \(a_1 = 3\) and \(a_2 = 1.\)</p>
<p>Our model stores each of its 113 tokens as a vector of 16 numbers, learned in training. The plate shows the vector of <code>name</code>, token id 93, from the trained weights: \(a_1 = {-0.302},\) \(a_6 = 1.416,\) \(a_{10} = {-1.455}.\) No one chose these values; training moved them until the model predicted well. Entries count from 1 here, as in mathematics; token ids and positions count from 0, as in the code, so <code>name</code> is at position 1, the second token.</p>
<p>Sixteen dimensions cannot be drawn as one arrow, but nothing in the arithmetic depends on drawing. Every operation in this chapter is defined entry by entry and works the same for 2 entries, for 16, or for the 4,096 numbers per token in Llama 3 8B.</p>`,
    fig: { type: "heat", caption: "The 16 entries of the trained vector of “name” (row 93 of the 113 × 16 table of token vectors). Blue is negative, red positive.", rows: ["name"], cols: COLS16, matrix: [Array.from(M.row(M.weights().wte, 93))], dp: 3 },
    terms: [
      { term: "Vector", def: "An ordered list of numbers, \\(\\mathbf{a} = (a_1, \\ldots, a_n);\\) drawn in 2-D as an arrow from the origin." },
      { term: "Dimension", def: "The number of entries of a vector: 2 for \\((3, 1),\\) 16 for a token vector of our model." },
    ],
    src: ["Meta, llama-models repository, models/sku_list.py: Llama 3 8B dim 4096", "Dubey et al., The Llama 3 Herd of Models, 2024"],
  });

  Deck.slide({
    id: "math-vec-ops",
    title: "Vectors add entry by entry and scale by a number",
    example: "tokens ids focus:1",
    body: String.raw`<p>Two vectors of the same dimension add entry by entry: \((2, 1) + (1, 3) = (3, 4).\) Drawn, \(\mathbf{b}\) is placed at the tip of \(\mathbf{a},\) and the sum runs from the origin to where \(\mathbf{b}\) ends.</p>
<p>Multiplying by a number, a <b>scalar</b>, scales every entry: \(2 \cdot (2, 1) = (4, 2)\) is twice as long in the same direction, \({-1} \cdot (2, 1) = ({-2}, {-1})\) points the opposite way, and \(0.5 \cdot (2, 1) = (1, 0.5)\) is half as long.</p>
<p>Our model's first computation is a 16-dimensional addition: each token's vector plus the vector of its position. For <code>name</code> at position 1, entries 2 to 4 are \(0.634 + ({-0.118}) = 0.516,\) \(0.373 + ({-0.141}) = 0.232\) and \({-0.313} + ({-0.458}) = {-0.771},\) and so on for all 16.</p>
<p>A training step is a vector operation too. Plain gradient descent (Part 2) subtracts a small number times the gradient from the parameters, entry by entry, for all 9,968 entries at once; Adam, which trained our model, first rescales each entry (Part 3).</p>`,
    fig: { type: "math-vec", mode: "add", a: [2, 1], b: [1, 3], c: 2, caption: "Blue a, red b, green a + b, with b moved to the tip of a (dashed); yellow c · a lies along a. Drag the tips; they snap to halves." },
    terms: [{ term: "Scalar", def: "A single number, as opposed to a vector; multiplying a vector by a scalar multiplies every entry." }],
  });

  Deck.slide({
    id: "math-dot",
    title: "The dot product multiplies matching entries and adds them up",
    body: String.raw`<p>The <b>dot product</b> of two vectors of the same dimension is the sum of the products of matching entries:</p>
\[\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i\]
<p>\((2, 1) \cdot (1, 3) = 2 \cdot 1 + 1 \cdot 3 = 5.\) The result is one number. It costs \(n\) multiplications and \(n - 1\) additions, and most of a model's arithmetic is dot products (Part 6 counts it).</p>
<p>Geometrically, \(\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}|\,|\mathbf{b}| \,\cos\theta,\) where \(|\mathbf{a}|\) is the length of the arrow and \(\theta\) the angle between the two. Here \(|\mathbf{a}| = \sqrt{5} = 2.236\) and \(|\mathbf{b}| = \sqrt{10} = 3.162,\) so \(\cos\theta = 5/\sqrt{50} = 0.7071\) and \(\theta = 45^\circ.\)</p>
<p>The sign tells the angle: positive below \(90^\circ,\) zero at exactly \(90^\circ\) (perpendicular), negative beyond. For a direction \(\mathbf{u}\) of length 1, the slope of chapter 1.4 is \(\nabla f \cdot \mathbf{u} = |\nabla f| \,\cos\theta\): largest, \(|\nabla f|,\) along the gradient and 0 at right angles to it, along the level curve, as \((2, 6) \cdot (3, {-1}) = 0\) showed.</p>`,
    fig: { type: "math-vec", mode: "dot", a: [2, 1], b: [1, 3], caption: "The dot product two ways: as a sum of products and as |a| |b| cos θ. The dashed line drops b onto the line of a; drag the tips." },
    terms: [{ term: "Dot product", def: "\\(\\mathbf{a} \\cdot \\mathbf{b} = \\sum_i a_i b_i,\\) one number; equal to \\(|\\mathbf{a}|\\,|\\mathbf{b}| \\,\\cos\\theta.\\)" }],
  });

  Deck.slide({
    id: "math-cosine",
    title: "Length and cosine similarity compare two vectors",
    example: "tokens ids",
    body: String.raw`<p>The <b>length</b> (or <b>norm</b>) of a vector is \(|\mathbf{a}| = \sqrt{\mathbf{a} \cdot \mathbf{a}},\) Pythagoras in \(n\) dimensions. The vector of <code>7</code> has length 4.614; that of <code>name</code>, 2.874.</p>
<p>Dividing the dot product by both lengths removes size and keeps direction. The result is the <b>cosine similarity</b>, between −1 (opposite directions) and 1 (the same direction):</p>
\[\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}|\,|\mathbf{b}|}\]
<p>In our trained model <code>7</code> and <code>3</code> have cosine similarity 0.99, and every pair of the nine digits lies between 0.87 and 0.99; no other token comes closer than 0.67 to any digit. Every digit appears in exactly the same contexts in the grammar, and training pointed their vectors the same way. <code>7</code> and <code>name</code> have −0.35. Part 4 reads the whole table of token vectors this way.</p>`,
    fig: {
      type: "math-cos", caption: "Two trained token vectors, their entry-by-entry products, and the cosine similarity. Pick any two tokens.", a: " 7", b: " 3",
      groups: [
        ["our query", ["SELECT", " name", " FROM", " users", " WHERE", " id", " =", " 7", ";"]],
        ["columns", [" email", " age", " user_id", " total", " status", " title", " price"]],
        ["tables", [" orders", " products"]],
        ["digits", [" 1", " 2", " 3", " 4", " 5", " 6", " 8", " 9"]],
      ],
    },
    terms: [
      { term: "Norm", def: "The length of a vector, \\(|\\mathbf{a}| = \\sqrt{\\mathbf{a} \\cdot \\mathbf{a}} = \\sqrt{\\sum_i a_i^2}.\\)" },
      { term: "Cosine similarity", def: "\\(\\mathbf{a} \\cdot \\mathbf{b} / (|\\mathbf{a}|\\,|\\mathbf{b}|),\\) the cosine of the angle between two vectors: 1 for the same direction, 0 for perpendicular, −1 for opposite." },
    ],
  });

  Deck.slide({
    id: "math-stack",
    title: "Vectors stacked as rows form a matrix",
    example: "tokens ids",
    body: String.raw`<p>Stack the 9 token vectors of our query, one per row, and you get a grid of 9 rows and 16 columns: a <b>matrix</b> of shape 9 × 16, holding 144 numbers. Row \(i\) is the vector of token \(i;\) column \(j\) holds entry \(j\) of every token.</p>
<p>The model's whole table of token vectors is one such matrix: 113 rows, one per vocabulary token, by 16 columns, 1,808 numbers. Looking up a token's vector means reading its row; <code>name</code> is the row with index 93 (ids count from 0).</p>
<p>Stacking pays off in speed. The same operation applied to every row, such as multiplying each token's vector by the same weights, becomes one operation on the whole matrix, and chapter 1.6 shows it is a single matrix product.</p>`,
    fig: { type: "heat", caption: "The 9 token vectors of our query as a 9 × 16 matrix. Hover a cell for its value.", rows: M.running.map((t) => t.trim()), cols: COLS16, matrix: M.runningIds.map((id) => Array.from(M.row(M.weights().wte, id))), dp: 3 },
    terms: [{ term: "Matrix", def: "A grid of numbers with \\(m\\) rows and \\(n\\) columns, its shape \\(m \\times n;\\) for example 9 token vectors of 16 numbers stacked as rows." }],
    check: {
      q: String.raw`\(\mathbf{a} = (1, 2)\) and \(\mathbf{b} = (4, {-2}).\) What is \(\mathbf{a} \cdot \mathbf{b},\) and what does it say?`,
      options: ["0: the vectors are perpendicular", "(4, −4): one product per entry", "8: they point the same way", "1: the cosine similarity is 1"],
      answer: 0,
      why: String.raw`\(1 \cdot 4 + 2 \cdot ({-2}) = 4 - 4 = 0.\) The dot product is one number, the sum of the products; \((4, {-4})\) forgets to add them. A dot product of 0 means \(\cos\theta = 0,\) an angle of \(90^\circ.\)`,
    },
  });

  // ------------------------------------------------------------ 1.6 Matrices
  Deck.chapter({ title: "Matrices" });

  const A23 = [[1, 2, 0], [0, 1, 3]];
  const c30 = Math.cos(Math.PI / 6), s30 = Math.sin(Math.PI / 6);

  Deck.slide({
    id: "math-matrix",
    title: "A matrix is a grid of numbers with a shape",
    body: String.raw`<p>A matrix of shape \(m \times n\) has \(m\) rows and \(n\) columns. Entry \(A_{ij}\) sits in row \(i,\) column \(j.\) In</p>
\[\mathbf{A} = \begin{pmatrix} 1 & 2 & 0 \\ 0 & 1 & 3 \end{pmatrix}\]
<p>the shape is 2 × 3, \(A_{12} = 2\) and \(A_{23} = 3.\) A vector with \(n\) entries is an \(n \times 1\) matrix (a column) or a \(1 \times n\) matrix (a row).</p>
<p>Almost all of our model's 9,968 parameters live in matrices, listed on the plate. The product of the two numbers in a shape is the number of entries, and the entries add up to 9,968.</p>
<p>In memory a matrix is one flat array. Row-major order, the convention of C and of the model code in this set, stores row 1, then row 2, and so on: entry \((i, j)\) of an \(m \times n\) matrix sits at offset \((i - 1)\,n + (j - 1).\)</p>`,
    fig: {
      type: "table", caption: "Every parameter of our model, by matrix", align: ["l", "r", "r", "l"],
      cols: ["Name", "Shape", "Entries", "Role"],
      rows: [
        ["<code>wte</code>", "113 × 16", "1,808", "one vector per token"],
        ["<code>wpe</code>", "8 × 16", "128", "one vector per position"],
        ["<code>wq wk wv wo</code>, 2 layers", "8 × (16 × 16)", "2,048", "attention (Part 5)"],
        ["<code>w1</code>, 2 layers", "2 × (16 × 64)", "2,048", "MLP, first matrix (Part 3)"],
        ["<code>w2</code>, 2 layers", "2 × (64 × 16)", "2,048", "MLP, second matrix"],
        ["<code>wu</code>", "16 × 113", "1,808", "one score per vocabulary token"],
        ["5 normalization gains", "5 × 16", "80", "vectors, not matrices (Part 3)"],
        ["", "", "<b>9,968</b>", ""],
      ],
      highlight: [5],
    },
    terms: [{ term: "Shape", def: "The numbers of rows and columns of a matrix, written \\(m \\times n.\\)" }],
  });

  Deck.slide({
    id: "math-matvec",
    title: "A matrix times a vector is one dot product per row",
    body: String.raw`<p>Multiplying an \(m \times n\) matrix by a vector of \(n\) entries gives a vector of \(m\) entries. Entry \(i\) of the result is the dot product of row \(i\) with the vector:</p>
\[(\mathbf{A}\mathbf{x})_i = \sum_{j=1}^{n} A_{ij}\, x_j\]
<p>With the matrix of the last sheet and \(\mathbf{x} = (1, 2, 1)\): row 1 gives \(1 \cdot 1 + 2 \cdot 2 + 0 \cdot 1 = 5\) and row 2 gives \(0 \cdot 1 + 1 \cdot 2 + 3 \cdot 1 = 5,\) so \(\mathbf{A}\mathbf{x} = (5, 5).\)</p>
<p>The sizes must match: each row needs one entry of \(\mathbf{x}\) per column. A 2 × 3 matrix accepts only 3-entry vectors and returns 2-entry vectors, so a matrix is a function from \(n\) numbers to \(m\) numbers.</p>
<p>Every layer of our model works this way. Each of its 16 × 16 attention matrices turns a 16-number vector into another 16 numbers with 256 multiplications.</p>`,
    fig: { type: "matmul", caption: "A 2 × 3 matrix times a 3-entry vector: step through the two dot products", A: A23, B: [[1], [2], [1]], aName: String.raw`\(\mathbf{A}\) <span class="dim">2×3</span>`, bName: String.raw`\(\mathbf{x}\) <span class="dim">3×1</span>`, cName: String.raw`\(\mathbf{A}\mathbf{x}\) <span class="dim">2×1</span>` },
  });

  Deck.slide({
    id: "math-transform",
    title: "A 2 × 2 matrix moves every point of the plane",
    body: String.raw`<p>A 2 × 2 matrix maps each point \((x, y)\) to \((ax + by,\ cx + dy).\) Applied to every point of a grid, it moves the whole plane: straight lines stay straight, parallel lines stay parallel, and the origin stays put.</p>
<p>The columns say where the axes go. \((1, 0)\) lands on the first column, \((a, c),\) and \((0, 1)\) on the second, \((b, d).\) Every other point follows: \((1, 1)\) lands on \((a + b, c + d).\)</p>
<ul>
<li><b>Scaling</b> by \(\begin{pmatrix} 2 & 0 \\ 0 & 0.5 \end{pmatrix}\) doubles \(x\) and halves \(y.\)</li>
<li><b>Rotation</b> by \(\theta\) is \(\begin{pmatrix} \,\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix};\) at \(30^\circ\) it sends \((1, 0)\) to \((0.866, 0.5).\)</li>
<li><b>Shear</b> by \(\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}\) slides each point sideways by its height: \((1, 1) \mapsto (2, 1).\)</li>
</ul>
<p>The number \(ad - bc\) is the signed area of the image of the unit square: 1 for each example above, negative when the plane is flipped, and 0 for the flatten example, which maps the whole plane onto one line. Each layer of a network applies matrices like these, in 16 or more dimensions instead of 2.</p>`,
    fig: {
      type: "math-grid", caption: "Gray: the original grid. Blue lines: the grid after A. Yellow: the image of the unit square. Drag the arrow tips (the columns of A) or pick an example.",
      A: [[1, 1], [0, 1]],
      presets: [
        { label: "shear", A: [[1, 1], [0, 1]] },
        { label: "scale", A: [[2, 0], [0, 0.5]] },
        { label: "rotate 30°", A: [[c30, -s30], [s30, c30]] },
        { label: "identity", A: [[1, 0], [0, 1]] },
        { label: "flatten", A: [[1, 2], [0.5, 1]] },
      ],
    },
  });

  Deck.slide({
    id: "math-matmul",
    title: "A matrix product pairs every row of one with every column of the other",
    body: String.raw`<p>The product \(\mathbf{C} = \mathbf{A}\mathbf{B}\) of an \(m \times k\) matrix and a \(k \times n\) matrix is \(m \times n.\) Entry \(C_{ij}\) is the dot product of row \(i\) of \(\mathbf{A}\) with column \(j\) of \(\mathbf{B}\):</p>
\[C_{ij} = \sum_{t=1}^{k} A_{it}\, B_{tj} \qquad (m \times k)(k \times n) = m \times n\]
<p>The inner sizes must agree (both \(k\)) and disappear; the outer sizes give the shape. On the plate the first column of \(\mathbf{B}\) is the \(\mathbf{x}\) of the matrix–vector sheet, so the first column of \(\mathbf{C}\) is \(\mathbf{A}\mathbf{x} = (5, 5)\): \(C_{11} = 1 \cdot 1 + 2 \cdot 2 + 0 \cdot 1 = 5.\)</p>
<p>A matrix product is a batch of matrix–vector products that share one matrix, one for each column of \(\mathbf{B}.\) Order matters: \(\mathbf{A}\mathbf{B}\) and \(\mathbf{B}\mathbf{A}\) usually differ, and often only one of them has matching shapes. Here \(\mathbf{B}\mathbf{A}\) exists but is \((3 \times 2)(2 \times 3) = 3 \times 3.\)</p>`,
    fig: { type: "matmul", caption: "A 2 × 3 matrix times a 3 × 2 matrix, one entry at a time", A: A23, B: [[1, 0], [2, 1], [1, 2]] },
    terms: [{ term: "Matrix product", def: "\\(C_{ij} = \\sum_t A_{it} B_{tj}:\\) row \\(i\\) of \\(\\mathbf{A}\\) dotted with column \\(j\\) of \\(\\mathbf{B};\\) shapes \\((m \\times k)(k \\times n) = m \\times n.\\)" }],
  });

  Deck.slide({
    id: "math-flops",
    title: "Multiplying m × k by k × n costs 2mnk floating-point operations",
    layout: "wide",
    example: "tokens",
    body: String.raw`<p>Each of the \(mn\) entries of the product is a dot product of length \(k\): \(k\) multiplications and \(k\) additions into a running sum. That is \(mnk\) <b>multiply–adds</b>, counted as \(2mnk\) <b>FLOPs</b> (floating-point operations). Strictly the first addition is not needed; the convention ignores the difference.</p>
<p>Our model's output layer multiplies the final vectors of 8 positions (8 × 16) by the output matrix (16 × 113): \(2 \cdot 8 \cdot 113 \cdot 16 = 28{,}928\) FLOPs, giving 113 scores at each position. The same layer in Llama 3 8B maps 4,096 numbers to 128,256 vocabulary scores: \(2 \cdot 4{,}096 \cdot 128{,}256 = 1.05 \times 10^9\) FLOPs for every token.</p>
<p>The numbers read and written grow more slowly: \(mk + kn + mn\) entries for the three matrices. With one token at a time the Llama layer does about 2 FLOPs per number it touches; with 4,096 tokens at once, about 4,000. Part 6 shows why GPUs need the second kind.</p>`,
    fig: {
      type: "table", caption: "FLOPs against the numbers a product reads and writes", align: ["l", "r", "r", "r", "r"],
      cols: ["Product", "m × k · k × n", "FLOPs 2mnk", "Entries mk + kn + mn", "FLOPs per entry"],
      rows: [
        ["our output layer", "8 × 16 · 16 × 113", "28,928", "2,840", "10.2"],
        ["Llama 3 8B output, 1 token", "1 × 4,096 · 4,096 × 128,256", "1.05 × 10⁹", "5.25 × 10⁸", "2.0"],
        ["same, 4,096 tokens", "4,096 × 4,096 · 4,096 × 128,256", "4.30 × 10¹²", "1.07 × 10⁹", "4,032"],
        ["square, 4,096³", "4,096 × 4,096 · 4,096 × 4,096", "1.37 × 10¹¹", "5.03 × 10⁷", "2,731"],
      ],
      highlight: [0],
    },
    terms: [
      { term: "FLOP", def: "One floating-point operation, an addition or a multiplication. A matrix product of \\((m \\times k)(k \\times n)\\) costs \\(2mnk\\) FLOPs." },
      { term: "Multiply–add", def: "One multiplication followed by one addition into a running sum; a length-\\(k\\) dot product is \\(k\\) of them." },
    ],
    src: ["Meta, llama-models repository, models/sku_list.py: Llama 3 8B dim 4096, vocab_size 128256"],
  });

  Deck.slide({
    id: "math-transpose",
    title: "The transpose swaps rows and columns; the identity changes nothing",
    body: String.raw`<p>The <b>transpose</b> \(\mathbf{A}^\top\) turns rows into columns: \((A^\top)_{ij} = A_{ji},\) and an \(m \times n\) matrix becomes \(n \times m.\) Transposing twice gives back \(\mathbf{A},\) and the transpose of a product reverses the order, \((\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top,\) so that the shapes still match.</p>
<p>A dot product is a matrix product with a transpose: \(\mathbf{a}^\top \mathbf{b}\) is \((1 \times n)(n \times 1) = 1 \times 1,\) one number.</p>
<p>The <b>identity matrix</b> \(\mathbf{I}\) has 1 on the diagonal and 0 elsewhere. Each of its rows picks out one entry, so \(\mathbf{I}\mathbf{x} = \mathbf{x}\) and \(\mathbf{A}\mathbf{I} = \mathbf{A}.\)</p>
<p>Code often stores a layer's matrix transposed. PyTorch's <code>nn.Linear</code> keeps its weight as out × in and computes \(\mathbf{y} = \mathbf{x}\mathbf{W}^\top + \mathbf{b}\) for each input row \(\mathbf{x};\) our model's code stores in × out and computes \(\mathbf{x}\mathbf{W}.\) Same numbers, different layout. The backward pass through a layer multiplies by the transpose (Part 3).</p>`,
    fig: {
      type: "math-transpose", caption: "Hover an entry: it moves from row i, column j to row j, column i", A: A23,
      after: String.raw`\[\mathbf{A}\mathbf{A}^\top = \begin{pmatrix} 5 & 2 \\ 2 & 10 \end{pmatrix} \qquad \mathbf{I}_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \qquad \mathbf{A}\,\mathbf{I}_3 = \mathbf{A}\]`,
    },
    terms: [
      { term: "Transpose", def: "\\(\\mathbf{A}^\\top,\\) the matrix with rows and columns swapped: \\((A^\\top)_{ij} = A_{ji}.\\)" },
      { term: "Identity matrix", def: "\\(\\mathbf{I},\\) with 1 on the diagonal and 0 elsewhere; \\(\\mathbf{I}\\mathbf{x} = \\mathbf{x}.\\)" },
    ],
    src: ["PyTorch documentation, torch.nn.Linear: y = xAᵀ + b, weight of shape (out_features, in_features)"],
  });

  Deck.slide({
    id: "math-batch",
    title: "One matrix product processes every position at once",
    example: "tokens ids focus:4",
    body: String.raw`<p>Our model turns the final 16-number vector at each position into 113 scores by multiplying it by the 16 × 113 output matrix \(\mathbf{W}_U.\) For one query that is 8 separate vector–matrix products.</p>
<p>Stack the 8 vectors as the rows of an 8 × 16 matrix \(\mathbf{X}\) and the 8 products become one: \(\mathbf{Z} = \mathbf{X}\mathbf{W}_U,\) of shape \((8 \times 16)(16 \times 113) = 8 \times 113.\) Row \(i\) of \(\mathbf{Z}\) depends only on row \(i\) of \(\mathbf{X},\) so the result is the same, and so is the cost: 28,928 FLOPs.</p>
<p>A training step does this for 128 queries at once: \(\mathbf{X}\) has \(128 \times 8 = 1{,}024\) rows and the product costs 3,702,784 FLOPs, still one matrix product (a single <code>@</code> in the training code). The plate shows the real matrices of our query; hover a score to see the row and column that produced it.</p>
<p>What one product buys is speed. Products with thousands of rows and columns in every dimension come close to a GPU's peak arithmetic rate; ours, at about 28 FLOPs per number read or written, is far too small for that (Part 6).</p>`,
    fig: { type: "math-batchmm", caption: "Our query's output layer with its real numbers: X (left) times W_U (top) gives Z. Red outlines: one row of X and the scores it produces.", row: 4 },
    check: {
      q: "An 8 × 16 matrix times a 16 × 16 matrix: what is the shape of the result, and how many FLOPs does it take?",
      options: ["8 × 16, 4,096 FLOPs", "16 × 16, 4,096 FLOPs", "8 × 16, 2,048 FLOPs", "8 × 8, 2,048 FLOPs"],
      answer: 0,
      why: String.raw`\((8 \times 16)(16 \times 16)\) keeps the outer sizes: 8 × 16. FLOPs \(= 2mnk = 2 \cdot 8 \cdot 16 \cdot 16 = 4{,}096;\) 2,048 counts only the multiplications.`,
    },
  });

  // ------------------------------------------------------------ 1.7 Probability
  Deck.chapter({ title: "Probability" });

  // the grammar's distribution of the token after SELECT, and the model's
  const AFTER_SELECT = [[" id", 5 / 18, "5/18"], [" name", 1 / 12, "1/12"], [" email", 1 / 12, "1/12"], [" age", 1 / 12, "1/12"], [" user_id", 1 / 12, "1/12"], [" total", 1 / 12, "1/12"], [" status", 1 / 12, "1/12"], [" title", 1 / 9, "1/9"], [" price", 1 / 9, "1/9"]];
  const TR = M.forward(M.runningIds.slice(0, M.config.T));
  const P_SELECT = TR.probs[0];
  const MODEL_P = M.runningIds.slice(1).map((id, i) => TR.probs[i][id]);

  Deck.slide({
    id: "math-distribution",
    title: "A probability distribution: numbers of at least 0 that sum to 1",
    example: "text",
    body: String.raw`<p>A <b>probability distribution</b> over a finite set of outcomes gives each outcome a number \(p \ge 0,\) and the numbers sum to 1. The probability of a set of outcomes is the sum of theirs.</p>
<p>Our grammar starts by picking a table, each with probability 1/3. A <code>users</code> query then has probability \(1/3 \cdot 1/4 \cdot 1/3 \cdot 1/9 = 1/324,\) and there are 108 of them, so together they have \(108/324 = 1/3.\) The 54 <code>products</code> queries have 1/162 each, also 1/3 in total. Over all 270 queries the probabilities sum to \(1/3 + 1/3 + 1/3 = 1.\)</p>
<p>A model's output is a distribution of this kind: 113 numbers of at least 0, one per token, summing to 1. After <code>SELECT</code> our model gives the 9 column names 0.99984 of the total; the other 104 tokens share the remaining 0.00016.</p>`,
    fig: {
      type: "table", caption: "The grammar's 270 queries, table by table", align: ["l", "r", "r", "r", "r"],
      cols: ["Table", "p(table)", "Queries", "p(each query)", "Total"],
      rows: [
        ["<code>users</code>", "1/3", "108", "1/324", "108/324 = 1/3"],
        ["<code>orders</code>", "1/3", "108", "1/324", "108/324 = 1/3"],
        ["<code>products</code>", "1/3", "54", "1/162", "54/162 = 1/3"],
        ["all", "1", "270", "", "<b>1</b>"],
      ],
      note: "A users query: table 1/3, output column 1/4 (of 4), filter column 1/3 (of the other 3), digit 1/9.",
    },
    terms: [{ term: "Probability distribution", def: "An assignment of a number \\(p \\ge 0\\) to each outcome of a set, with the numbers summing to 1." }],
  });

  Deck.slide({
    id: "math-conditional",
    title: "Conditional probability: what the prefix says about the next token",
    example: "tokens prefix:1",
    body: String.raw`<p>The <b>conditional probability</b> \(p(B \mid A)\) is the probability of \(B\) once \(A\) is known to have happened: \(p(B \mid A) = p(A \text{ and } B)/p(A).\)</p>
<p>Every query starts with <code>SELECT</code>, so after it comes the output column. <code>name</code> comes next only when the table is <code>users</code> (1/3) and that table's column pick is <code>name</code> (1/4): \(p(\texttt{name} \mid \texttt{SELECT}) = 1/3 \cdot 1/4 = 1/12 = 0.0833.\) <code>id</code> belongs to all three tables: \(1/12 + 1/12 + 1/9 = 5/18 = 0.2778.\) The nine possible columns sum to 1.</p>
<p>Conditioning can make a token certain. After <code>SELECT name FROM</code> only <code>users</code> has a <code>name</code> column, so \(p(\texttt{users} \mid \texttt{SELECT name FROM}) = 1.\)</p>
<p>The trained model's probabilities after <code>SELECT</code> sit close to these fractions: 0.0854 for <code>name</code>, 0.2806 for <code>id</code>.</p>`,
    fig: {
      type: "math-pairs", caption: "The token after SELECT: the grammar's exact probability and the trained model's", aName: "grammar", bName: "trained model",
      items: AFTER_SELECT.map(([t, p, txt]) => ({ label: t.trim(), a: p, aText: txt, b: P_SELECT[M.idOf(t)] })),
    },
    terms: [{ term: "Conditional probability", def: "\\(p(B \\mid A) = p(A \\text{ and } B)/p(A):\\) the probability of \\(B\\) once \\(A\\) is known." }],
  });

  Deck.slide({
    id: "math-seq-prob",
    title: "The probability of a sequence is a product of next-token probabilities",
    layout: "wide",
    example: "tokens",
    body: String.raw`<p>The <b>chain rule of probability</b> splits a joint probability into conditionals: \(p(a, b, c) = p(a)\, p(b \mid a)\, p(c \mid a, b).\) Applied to a sequence of tokens, the probability of the whole is the product of each token's probability given all the tokens before it:</p>
\[p(t_1, \ldots, t_n) = \prod_{i=1}^{n} p(t_i \mid t_1, \ldots, t_{i-1})\]
<p>For our 9-token query the grammar gives \(1 \cdot 1/12 \cdot 1 \cdot 1 \cdot 1\) \(\cdot\; 1/3 \cdot 1 \cdot 1/9 \cdot 1\) \(= 1/324 = 0.003086.\) Only three tokens involve a choice: the output column, the filter column and the digit.</p>
<p>The trained model's product is 0.003182. Its first factor is 1 by construction: every query starts with <code>SELECT</code>, and the model only predicts what follows. So a model that predicts the next token is a complete model of whole queries: multiplying its predictions gives the probability of any of them.</p>`,
    fig: {
      type: "sequence", caption: "Each token's probability given the tokens before it, and the running product", tokens: M.running,
      rows: [
        { label: "grammar", values: ["1", "1/12", "1", "1", "1", "1/3", "1", "1/9", "1"] },
        { label: "model", values: ["1 (given)"].concat(MODEL_P), dp: 4 },
        { label: "model product", values: [1].concat(MODEL_P).map((_, i, a) => a.slice(0, i + 1).reduce((x, y) => x * y, 1)), dp: 6 },
      ],
      focus: [1, 5, 7],
    },
    terms: [{ term: "Chain rule of probability", def: "\\(p(t_1, \\ldots, t_n) = \\prod_i p(t_i \\mid t_1, \\ldots, t_{i-1}):\\) a joint probability is a product of conditionals." }],
  });

  Deck.slide({
    id: "math-expectation",
    title: "The expectation is the probability-weighted average",
    body: String.raw`<p>If a quantity \(X\) takes the value \(x\) with probability \(p(x),\) its <b>expectation</b> is \(E[X] = \sum_x p(x)\, x\): each value weighted by how often it occurs.</p>
<p>The digit in our grammar is uniform over 1 to 9: \(E[v] = (1 + 2 + \cdots + 9)/9 = 45/9 = 5.\) The number of columns of the chosen table is 4, 4 or 3, each with probability 1/3: \(E = (4 + 4 + 3)/3 = 11/3 = 3.667,\) a value no single table has.</p>
<p>Expectation is linear: \(E[aX + b] = a\,E[X] + b,\) and \(E[X + Y] = E[X] + E[Y]\) always. The number of possible filter columns is one less than the number of columns, so its expectation is \(11/3 - 1 = 8/3 = 2.667.\)</p>
<p>Part 2 defines the model's loss as an expectation of this kind: the average over all 270 queries, each weighted by its probability. For the trained model it is 0.6942 per token, measured in nats (chapter 1.8).</p>`,
    fig: {
      type: "calc", caption: "Reweight the three tables (the weights are divided by their sum) and watch the expected number of columns move between 3 and 4",
      inputs: [
        { id: "u", label: "weight of users", min: 0, max: 3, step: 0.1, value: 1, fmt: "f1" },
        { id: "o", label: "weight of orders", min: 0, max: 3, step: 0.1, value: 1, fmt: "f1" },
        { id: "q", label: "weight of products", min: 0, max: 3, step: 0.1, value: 1, fmt: "f1" },
      ],
      outputs: [
        { label: "p(users), p(orders), p(products)", f: (v) => { const S = v.u + v.o + v.q; return S > 0 ? `${fix(v.u / S, 3)}, ${fix(v.o / S, 3)}, ${fix(v.q / S, 3)}` : "–"; }, fmt: (t) => t },
        { label: "E[columns] = 4p₁ + 4p₂ + 3p₃", f: (v) => { const S = v.u + v.o + v.q; return S > 0 ? (4 * v.u + 4 * v.o + 3 * v.q) / S : NaN; }, fmt: "f3" },
        { label: "E[filter columns] = E[columns] − 1", f: (v) => { const S = v.u + v.o + v.q; return S > 0 ? (4 * v.u + 4 * v.o + 3 * v.q) / S - 1 : NaN; }, fmt: "f3" },
      ],
      note: (v) => (v.u + v.o + v.q > 0 ? "" : "All weights are 0: no distribution to average over."),
    },
    terms: [{ term: "Expectation", def: "\\(E[X] = \\sum_x p(x)\\, x,\\) the average of a quantity weighted by probability. Linear: \\(E[aX + b] = aE[X] + b.\\)" }],
  });

  Deck.slide({
    id: "math-sampling",
    title: "Sampling turns a uniform random number into a draw",
    example: "tokens prefix:1",
    body: String.raw`<p>To draw from a distribution, lay its probabilities end to end on the interval from 0 to 1, so that each outcome owns a segment as long as its probability. The right end of outcome \(k\)'s segment is the <b>cumulative distribution</b>: the sum of the probabilities of outcomes 1 to \(k.\)</p>
<p>Then draw \(u\) uniformly from \([0, 1),\) which every programming language provides, and return the outcome whose segment contains \(u.\) An outcome with probability 5/18 owns 5/18 of the interval, so it is returned 5/18 of the time.</p>
<p>For the columns after <code>SELECT</code> in the plate's order, <code>id</code> owns \([0, 0.2778)\) and <code>name</code> owns \([0.2778, 0.3611),\) so \(u = 0.3\) returns <code>name</code>.</p>
<p>This is how a model generates text: compute the 113 probabilities, draw \(u,\) pick the token, append it, repeat (Part 4). The loop is <code>cum += p[i]; if (u &lt; cum) return i</code>.</p>`,
    fig: {
      type: "math-cdf", caption: "The nine columns after SELECT laid end to end. Move u, or draw: each draw is a uniform u, mapped through the segments.", u: 0.3, seed: 7, initial: 1000,
      outcomes: AFTER_SELECT.map(([t, p, txt]) => ({ label: t.trim(), p, text: txt })),
    },
    terms: [{ term: "Cumulative distribution", def: "For outcomes in a fixed order, the running sum of their probabilities; sampling returns the first outcome whose running sum exceeds a uniform \\(u.\\)" }],
  });

  // law of large numbers: running means of uniform digits for three fixed seeds
  const LLN_N = 10000, SIGMA9 = Math.sqrt(80 / 12);
  const llnRun = (seed) => {
    const rnd = M.mulberry32(seed), pts = [];
    let sum = 0;
    for (let n = 1; n <= LLN_N; n++) {
      sum += 1 + Math.floor(rnd() * 9);
      if (n <= 100 || n % Math.ceil(n / 100) === 0 || n === LLN_N) pts.push([n, sum / n]);
    }
    return pts;
  };
  const bandPts = (sgn) => Array.from({ length: 121 }, (_, i) => { const n = Math.pow(10, (4 * i) / 120); return [n, 5 + (sgn * SIGMA9) / Math.sqrt(n)]; });

  Deck.slide({
    id: "math-lln",
    title: "Independent draws: the average converges to the expectation",
    body: String.raw`<p>Two events are <b>independent</b> if knowing one does not change the probability of the other: \(p(A \text{ and } B) = p(A)\,p(B).\) In our grammar the digit is independent of everything before it: \(p(v = 7 \mid \texttt{users}, \texttt{name}, \texttt{id}) = 1/9,\) the same as with no information.</p>
<p>The <b>law of large numbers</b>: the average of \(n\) independent draws of \(X\) approaches \(E[X]\) as \(n\) grows. How fast follows from another fact: the <b>variances</b> \(\sigma^2 = E[(X - E[X])^2]\) of independent draws add, so a sum of \(n\) draws has variance \(n\sigma^2\) and the average has <b>standard deviation</b> \(\sigma/\sqrt{n}.\) For the digit \(\sigma = 2.582,\) so the average is typically off by 0.82 after 10 draws, 0.26 after 100 and 0.026 after 10,000.</p>
<p>The plate draws digits with three fixed random seeds. After 10 draws the averages are 5.9, 5.1 and 4.1; after 10,000 all three are within 0.015 of 5. Training relies on this: the average loss over a batch of 128 random queries is a noisy estimate of the expected loss, and Part 2 shows how its noise shrinks with the batch size.</p>`,
    fig: {
      type: "chart", caption: String.raw`<span>Running average of uniform digits 1 to 9 for three seeds, against the number of draws (logarithmic). Dashed: \(5 \pm \sigma/\sqrt{n}.\)</span>`,
      x: { domain: [1, LLN_N], type: "log", ticks: [1, 10, 100, 1000, 10000], fmt: (v) => v.toLocaleString("en-US"), label: "draws n" },
      y: { domain: [1, 9], ticks: [1, 3, 5, 7, 9], label: "running average" },
      series: [
        { points: bandPts(1), cls: "ink thin dashed" }, { points: bandPts(-1), cls: "ink thin dashed" },
        { points: llnRun(1), cls: "s1" }, { points: llnRun(2), cls: "s2" }, { points: llnRun(3), cls: "s3" },
      ],
      hlines: [{ y: 5, label: "E[v] = 5" }],
      legend: [["s1", "seed 1"], ["s2", "seed 2"], ["s3", "seed 3"]],
    },
    terms: [
      { term: "Independent events", def: "Events with \\(p(A \\text{ and } B) = p(A)\\,p(B):\\) knowing one does not change the probability of the other." },
      { term: "Law of large numbers", def: "The average of \\(n\\) independent draws approaches the expectation as \\(n\\) grows." },
      { term: "Variance", def: "\\(\\sigma^2 = E[(X - E[X])^2],\\) the average squared distance from the expectation. Variances of independent draws add, so the average of \\(n\\) draws has variance \\(\\sigma^2/n.\\)" },
      { term: "Standard deviation", def: "\\(\\sigma,\\) the square root of the variance: the typical distance of a draw from the expectation. The average of \\(n\\) independent draws has standard deviation \\(\\sigma/\\sqrt{n}.\\)" },
    ],
    check: {
      q: "What probability does the grammar give the query <code>SELECT title FROM products</code> <code>WHERE price = 3;</code>",
      options: ["1/162", "1/324", "1/270", "1/54"],
      answer: 0,
      why: String.raw`1/3 (products) × 1/3 (title, one of 3 columns) × 1/2 (price, one of the other 2) × 1/9 (the digit 3) = 1/162. 1/324 is the probability of a users or orders query; 1/270 would hold only if all 270 queries were equally likely.`,
    },
  });

  // ------------------------------------------------------------ 1.8 Exponentials and logarithms
  Deck.chapter({ title: "Exponentials and logarithms" });

  const LN_MODEL = MODEL_P.map((q) => Math.log(q));

  Deck.slide({
    id: "math-exp-log",
    title: "The exponential multiplies by e per unit step; ln undoes it",
    body: String.raw`<p>\(e^x\) turns addition into multiplication: \(e^{a + b} = e^a e^b.\) Each step of 1 in \(x\) multiplies the value by \(e = 2.71828,\) so it grows fast: \(e^{1} = 2.718,\) \(e^{5} = 148.4,\) \(e^{10} = 22{,}026.\) Negative \(x\) gives small positive values, \(e^{-10} = 0.0000454,\) and the value never reaches 0.</p>
<p>The natural logarithm answers the reverse question: \(\ln y\) is the power of \(e\) that gives \(y.\) So \(\ln e^x = x\) and \(e^{\ln y} = y,\) with \(\ln 1 = 0,\) \(\ln 2 = 0.6931\) and \(\ln 10 = 2.3026.\) It is defined only for \(y \gt 0\) and is negative below 1: \(\ln 0.1 = {-2.3026}.\) Its graph is the graph of \(e^x\) mirrored in the line \(y = x.\)</p>
<p>Logarithms in other bases differ by a constant factor: \(\log_2 y = \ln y / \ln 2.\) Information is measured in nats with \(\ln\) and in bits with \(\log_2;\) one bit is \(\ln 2 = 0.6931\) nats (Part 2).</p>`,
    fig: {
      type: "plot", caption: String.raw`<span>\(e^x\) (blue) and \(\ln x\) (red) are mirror images in the line \(y = x\) (dashed). The point \((x_0, e^{x_0})\) mirrors to \((e^{x_0}, x_0).\)</span>`,
      // 400 units wide, so the narrow-screen cap leaves it alone; 340 × 245 inner units give equal scales on x and y
      x: [-3, 8.4], y: [-3, 5.2], width: 400, height: 295, xlabel: "x", ylabel: "y", xaxis: { fmt: tickMinus }, yaxis: { fmt: tickMinus },
      params: [{ name: "x0", label: "x₀", min: -2.5, max: 1.6, step: 0.01, value: 1, fmt: "f2" }],
      fns: [{ f: (x) => Math.exp(x), cls: "s1" }, { f: (x) => (x > 0 ? Math.log(x) : NaN), cls: "s2", samples: 600 }, { f: (x) => x, cls: "ink thin dashed" }],
      extra: (p) => ({
        after(svg, X, Y, pp, box) {
          const g = clipG(svg, box), ex = Math.exp(p.x0);
          g.appendChild(s("line", { x1: X(p.x0), y1: Y(ex), x2: X(ex), y2: Y(p.x0), class: "math-guide" }));
          g.appendChild(s("circle", { cx: X(p.x0), cy: Y(ex), r: 5.5, class: "math-pt s1" }));
          g.appendChild(s("circle", { cx: X(ex), cy: Y(p.x0), r: 5.5, class: "math-pt s2" }));
          plabel(g, box, X(-2.9), Y(Math.exp(-2.9)) - 10, "eˣ");
          plabel(g, box, X(7.4), Y(Math.log(7.4)) + 18, "ln x");
          dragHandle(svg, X, Y, box, [-3, 8.4], 0, p.x0, ex);
        },
      }),
      readouts: [
        { label: String.raw`\(e^{x_0}\)`, f: (p) => Math.exp(p.x0), fmt: "f4" },
        { label: String.raw`\(\ln(e^{x_0})\)`, f: (p) => Math.log(Math.exp(p.x0)), fmt: "f4" },
        { label: String.raw`\(e^{x_0 + 1}/e^{x_0}\)`, f: (p) => Math.exp(p.x0 + 1) / Math.exp(p.x0), fmt: "f5" },
      ],
    },
  });

  Deck.slide({
    id: "math-log-sum",
    title: "Logarithms turn products into sums",
    layout: "wide",
    example: "tokens",
    body: String.raw`<p>Because \(e^{a+b} = e^a e^b,\) the logarithm of a product is the sum of the logarithms: \(\ln(ab) = \ln a + \ln b.\) Also \(\ln(1/a) = {-\ln a}\) and \(\ln a^k = k \,\ln a.\) So the log-probability of our query is a sum of 9 terms, one per token:</p>
\[\ln p(\text{query}) = \sum_{i=1}^{9} \,\ln p(t_i \mid t_1, \ldots, t_{i-1})\]
<p>From the grammar: \(0 + \ln(1/12) + 0 + 0 + 0\) \(+ \ln(1/3) + 0 + \ln(1/9) + 0\) \(= {-2.4849} - 1.0986 - 2.1972\) \(= {-5.7807},\) which is \(\ln(1/324).\) Certain tokens contribute exactly 0.</p>
<p>From the model: \({-2.4605} - 1.0613 - 2.2283\) plus six terms that round to 0 (the largest, for <code>users</code>, is −0.000008), in total −5.7501, and \(e^{-5.7501} = 0.003182,\) the product of the last chapter. Divided by the 8 predicted positions and negated, it is 0.7188 nats: the loss on our query from chapter 1.4.</p>`,
    fig: {
      type: "sequence", caption: "The log-probability of each token given the tokens before it, and the running sum", tokens: M.running,
      rows: [
        { label: "ln p, grammar", values: ["0", "−2.4849", "0", "0", "0", "−1.0986", "0", "−2.1972", "0"] },
        { label: "ln p, model", values: ["0 (given)"].concat(LN_MODEL), dp: 4 },
        { label: "running sum", values: [0].concat(LN_MODEL).map((_, i, a) => a.slice(0, i + 1).reduce((x, y) => x + y, 0)), dp: 4 },
      ],
      focus: [1, 5, 7],
    },
  });

  // repeated multiplication in float32 (every product rounded) and float64
  const powProduct = (p, n, f32) => { let x = f32 ? Math.fround(1) : 1; const q = f32 ? Math.fround(p) : p; for (let i = 0; i < n; i++) x = f32 ? Math.fround(x * q) : x * q; return x; };
  const firstZero = (p, f32) => { let x = f32 ? Math.fround(1) : 1; const q = f32 ? Math.fround(p) : p; for (let i = 1; i <= 5000; i++) { x = f32 ? Math.fround(x * q) : x * q; if (x === 0) return i; } return NaN; };
  const logSum32 = (p, n) => { let acc = Math.fround(0); const l = Math.fround(Math.log(p)); for (let i = 0; i < n; i++) acc = Math.fround(acc + l); return acc; };

  Deck.slide({
    id: "math-underflow",
    title: "Multiplying many probabilities underflows; adding their logs does not",
    body: String.raw`<p>Floating-point numbers have a smallest positive value. In float32 the smallest normal number is \(2^{-126} \approx 1.18 \times 10^{-38};\) below it precision drops step by step (subnormal numbers) down to \(2^{-149} \approx 1.4 \times 10^{-45}.\) Results at or below half of that, \(2^{-150} \approx 7.0 \times 10^{-46},\) round to 0. This loss of precision below the smallest normal number, ending at 0, is <b>underflow</b>.</p>
<p>A sequence of 1,000 tokens, each with probability 0.1, has probability \(10^{-1000}.\) Multiplied out in float32, the running product falls below \(2^{-126}\) after 38 factors. The 45th factor gives \(9.8 \times 10^{-46},\) which still rounds up to \(2^{-149};\) the 46th gives exactly 0, and every later factor leaves it there. In float64, whose smallest normal number is \(2^{-1022} \approx 2.2 \times 10^{-308},\) it reaches 0 after 324 factors.</p>
<p>The sum of the logs has no such problem: \(1{,}000 \times \,\ln 0.1 = {-2{,}302.585},\) an ordinary number, and float32 accumulates it to \({-2{,}302.592}.\) This is why language-model training sums log-probabilities instead of multiplying probabilities.</p>`,
    fig: {
      type: "calc", caption: "The product of n equal probabilities, multiplied out in float32 and in float64, against the sum of their logs",
      inputs: [
        { id: "n", label: "number of tokens n", min: 1, max: 1000, step: 1, value: 1000, fmt: "int" },
        { id: "p", label: "probability of each", min: 0.01, max: 0.99, step: 0.01, value: 0.1, fmt: "f2" },
      ],
      outputs: [
        { label: "exact product pⁿ", f: (v) => { const e = v.n * Math.log10(v.p), k = Math.floor(e); return `${fix(Math.pow(10, e - k), 2)} × 10${sup(k)}`; }, fmt: (t) => t },
        { label: "product in float32", f: (v) => powProduct(v.p, v.n, true), fmt: (x) => (x === 0 ? "0 (underflow)" : sci(x, 2)) },
        { label: "product in float64", f: (v) => powProduct(v.p, v.n, false), fmt: (x) => (x === 0 ? "0 (underflow)" : sci(x, 2)) },
        { label: "sum of logs n · ln p, float32", f: (v) => logSum32(v.p, v.n), fmt: "f3" },
      ],
      note: (v) => { const z32 = firstZero(v.p, true), z64 = firstZero(v.p, false); return `float32 reaches exactly 0 after ${Number.isNaN(z32) ? "more than 5,000" : z32.toLocaleString("en-US")} factors; float64 after ${Number.isNaN(z64) ? "more than 5,000" : z64.toLocaleString("en-US")}.`; },
    },
    terms: [{ term: "Underflow", def: "A result below the smallest normal number (\\(2^{-126}\\) in float32) loses precision; at or below \\(2^{-150} \\approx 7.0 \\times 10^{-46}\\) it rounds to 0." }],
    src: ["IEEE Std 754-2019, IEEE Standard for Floating-Point Arithmetic (binary32: emin = −126, 24-bit significand; binary64: emin = −1022, 53-bit significand)"],
  });

  Deck.slide({
    id: "math-lse",
    title: "Log-sum-exp: subtract the largest score before exponentiating",
    body: String.raw`<p>Softmax (Part 2) turns scores \(z_j\) into probabilities \(e^{z_j}/\sum_k e^{z_k},\) and its logarithm needs \(\ln\sum_k e^{z_k},\) the <b>log-sum-exp</b>. Computed directly it overflows: the largest float64 is about \(1.8 \times 10^{308},\) which \(e^z\) passes at \(z = 709.78;\) in float32 the limit is \(z = 88.72.\) For the scores 1000, 1001 and 1002 the direct formula gives \(\infty/\infty,\) which is not a number.</p>
<p>Subtracting a constant \(c\) from every score changes nothing, because \(e^{z_j - c}/\sum_k e^{z_k - c}\) has the factor \(e^{-c}\) above and below. With \(c = m = \max_k z_k\) the largest exponential is \(e^0 = 1\) and none can overflow:</p>
\[\ln\sum_k e^{z_k} = m + \ln\sum_k e^{z_k - m}\]
<p>Worked: \(1000, 1001, 1002 \to {-2}, {-1}, 0;\) \(e^{-2} + e^{-1} + e^{0} = 0.1353 + 0.3679 + 1 = 1.5032.\) The probabilities are 0.0900, 0.2447 and 0.6652, and the log-sum-exp is \(1002 + \ln 1.5032 = 1002.4076.\)</p>
<p>Very negative scores fail the other way: for −1000, −1001 and −1002 every \(e^z\) underflows to 0 (float64 does below \(z \approx {-745.1}\)), so the direct formula gives \(0/0\) and \(\ln 0 = {-\infty}.\) Subtracting the maximum fixes this too, because one term becomes \(e^0 = 1.\)</p>`,
    fig: {
      type: "calc", caption: "Three scores, shifted together. The direct formula fails once a score, or the sum of the exponentials, passes 1.8 × 10³⁰⁸, and when every exponential underflows to 0; the stable one never does.",
      inputs: [
        { id: "c", label: "shift added to all scores", min: -1000, max: 1000, step: 10, value: 1000, fmt: "int" },
        { id: "a", label: "score 1 − shift", min: -3, max: 3, step: 0.5, value: 0, fmt: "f1" },
        { id: "b", label: "score 2 − shift", min: -3, max: 3, step: 0.5, value: 1, fmt: "f1" },
        { id: "d", label: "score 3 − shift", min: -3, max: 3, step: 0.5, value: 2, fmt: "f1" },
      ],
      outputs: [
        { label: "scores", f: (v) => [v.c + v.a, v.c + v.b, v.c + v.d].map((z) => fix(z, 1)).join(", "), fmt: (t) => t },
        { label: "direct: e^z / Σ e^z", f: (v) => { const e = [v.a, v.b, v.d].map((q) => Math.exp(v.c + q)), S = e[0] + e[1] + e[2]; if (!Number.isFinite(S)) return e.some((x) => !Number.isFinite(x)) ? "fails: e^z overflowed to ∞" : "fails: the sum overflowed to ∞"; if (S === 0) return "fails: every e^z is 0, so 0/0"; return e.map((x) => fix(x / S, 4)).join(", "); }, fmt: (t) => t },
        { label: "stable: subtract the max first", f: (v) => { const z = [v.a, v.b, v.d], m = Math.max(...z), e = z.map((q) => Math.exp(q - m)), S = e[0] + e[1] + e[2]; return e.map((x) => fix(x / S, 4)).join(", "); }, fmt: (t) => t },
        { label: "log-sum-exp, direct", f: (v) => Math.log([v.a, v.b, v.d].reduce((acc, q) => acc + Math.exp(v.c + q), 0)), fmt: (x) => (Number.isFinite(x) ? fix(x, 4) : x > 0 ? "∞ (overflow)" : "−∞ (ln 0)") },
        { label: "log-sum-exp, stable", f: (v) => { const z = [v.c + v.a, v.c + v.b, v.c + v.d], m = Math.max(...z); return m + Math.log(z.reduce((acc, q) => acc + Math.exp(q - m), 0)); }, fmt: "f4" },
      ],
    },
    terms: [{ term: "Log-sum-exp", def: "\\(\\ln\\sum_k e^{z_k},\\) computed stably as \\(m + \\ln\\sum_k e^{z_k - m}\\) with \\(m = \\max_k z_k.\\)" }],
    src: ["IEEE Std 754-2019 (largest binary64 ≈ 1.8 × 10³⁰⁸, largest binary32 ≈ 3.4 × 10³⁸)"],
    check: {
      q: String.raw`Scores 5, 5 and 5 go through softmax. What are the probabilities and the log-sum-exp?`,
      options: [String.raw`1/3 each; \(5 + \ln 3 = 6.0986\)`, "1/3 each; 15", "1 each; 5", String.raw`0.5, 0.25, 0.25; \(6.0986\)`],
      answer: 0,
      why: String.raw`Subtract the largest score, 5: \(e^0 = 1\) three times, sum 3. Each probability is \(1/3,\) and the log-sum-exp is \(5 + \ln 3 = 6.0986.\) It is not the sum of the scores (15), and the three \(e^0 = 1\) only become probabilities after dividing by their sum; equal scores always give equal probabilities.`,
    },
  });
})();
