/* Part 5 — The transformer: the full architecture computed on our query with the
 * trained weights, then the variants used by Llama-class models.
 * Every model number on these sheets is recomputed in tools/verify/part05.mjs. */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip, slider, segmented, button, chart, heatStrip, seq, div } = window.UI;
  const M = window.TinyLM;
  const IDS = M.runningIds.slice(0, 8);
  const TOKS = M.running;
  const TR = M.forward(IDS);
  const WTS = M.weights();
  const short = (t) => t.trim();
  // fixed decimals with a true minus sign; values that round to zero print as 0
  const fix = (v, d) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : v).toFixed(d);
  const num = (v, d = 3) => fmt.minus(fix(+v, d));
  const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, window.UI.rich(l)));
  const HEADS = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const norm2 = (v) => Math.sqrt(v.reduce((a, b) => a + b * b, 0));
  const r2 = (m) => m.map((row) => row.map((v) => v));

  // ------------------------------------------------------------------ custom figures

  // Fig: the causal mask, step by step, for one head of the trained model.
  // Stage 0: all 64 scores q·k/√8 (later positions included). Stage 1: −∞ above the
  // diagonal. Stage 2: the softmax weights, which the trace stores.
  function fullScores(l, hh) {
    const L = TR.layers[l];
    return L.q.map((qv) => L.k.map((kv) => {
      let d = 0;
      for (let c = 0; c < 8; c++) d += qv[hh * 8 + c] * kv[hh * 8 + c];
      return d / Math.sqrt(8);
    }));
  }
  Deck.fig("tf-mask", (el, sp) => {
    let head = sp.head ?? 0, stage = sp.stage ?? 0, row = sp.row ?? 4;
    const ctl = h("div", { class: "f-ctl" },
      segmented({ label: "Head", value: head, options: HEADS.map(([l, hh], i) => ({ value: i, label: `L${l + 1} H${hh + 1}` })), onChange: (v) => { head = v; draw(); } }),
      segmented({ label: "Stage", value: stage, options: [{ value: 0, label: "1 · scores" }, { value: 1, label: "2 · add mask" }, { value: 2, label: "3 · softmax" }], onChange: (v) => { stage = v; draw(); } }));
    const grid = h("div", { class: "scroll-x" });
    const expl = h("p", { class: "f-mm-expl tf-expl", "aria-live": "polite" });
    el.append(ctl, grid, expl);
    function draw() {
      const [l, hh] = HEADS[head];
      const S = fullScores(l, hh), A = TR.layers[l].heads[hh].att;
      const smax = Math.max(...S.flat().map(Math.abs));
      const t = h("table", { class: "tf-mask-t", role: "grid", "aria-label": `Layer ${l + 1} head ${hh + 1}, stage ${stage + 1}` });
      t.appendChild(h("tr", null, h("th", { class: "tf-mask-corner" }, "reads →"),
        TOKS.slice(0, 8).map((tk, j) => h("th", { class: "tf-mask-col" }, h("span", { class: "tf-mask-j" }, String(j)), short(tk))),
        stage === 2 ? h("th", { class: "tf-mask-col" }, h("span", { class: "tf-mask-j" }, "row"), "sum") : null));
      for (let i = 0; i < 8; i++) {
        const trEl = h("tr", { class: i === row ? "is-row" : "", tabindex: "0", "aria-label": `row ${i}, ${short(TOKS[i])}` });
        trEl.addEventListener("click", () => { row = i; draw(); });
        trEl.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); row = i; draw(); } });
        trEl.appendChild(h("th", { class: "tf-mask-rowh" }, h("span", { class: "tf-mask-j" }, String(i)), short(TOKS[i])));
        for (let j = 0; j < 8; j++) {
          const fut = j > i, pair = `${short(TOKS[i])} (${i}) → ${short(TOKS[j])} (${j})`;
          let txt, cls = "tf-mask-c", style = null, tip;
          if (stage === 0) {
            txt = num(S[i][j], 2); style = { background: div(S[i][j], smax) }; tip = `score ${pair}: ${num(S[i][j], 3)}`;
            if (fut) cls += " fut";
          } else if (fut) {
            txt = stage === 1 ? "−∞" : "0"; cls += " masked"; tip = stage === 1 ? `${pair}: masked, −∞` : `${pair}: weight 0 (masked)`;
          } else if (stage === 1) {
            txt = num(S[i][j], 2); style = { background: div(S[i][j], smax) }; tip = `score ${pair}: ${num(S[i][j], 3)}`;
          } else {
            txt = num(A[i][j], 2); style = { background: seq(A[i][j]) }; tip = `weight ${pair}: ${num(A[i][j], 3)}`;
          }
          trEl.appendChild(h("td", { class: cls, style, "data-tip": tip }, txt));
        }
        if (stage === 2) trEl.appendChild(h("td", { class: "tf-mask-sum" }, num(A[i].reduce((a, b) => a + b, 0), 2)));
        t.appendChild(trEl);
      }
      clear(grid);
      grid.appendChild(t);
      const i = row, tok = `<code>${short(TOKS[i])}</code>`;
      const best = (arr, n) => { let b = 0; for (let j = 1; j < n; j++) if (arr[j] > arr[b]) b = j; return b; };
      const bAll = best(S[i], 8), bVis = best(S[i], i + 1);
      const name = (j) => `<code>${short(TOKS[j])}</code> (${j})`;
      if (stage === 0) {
        expl.innerHTML = `Row ${i}, ${tok}: all 8 scores can be computed. The largest is ${num(S[i][bAll], 2)} on ${name(bAll)}` +
          (bAll > i ? `, a later position.` : `, which ${tok} is allowed to see.`) + ` Select a row to inspect it.`;
      } else if (stage === 1) {
        expl.innerHTML = i === 7 ? `Row 7, ${tok}: it may see every position, so the mask leaves all 8 scores in place.`
          : `Row ${i}, ${tok}: the ${7 - i} later position${7 - i === 1 ? "" : "s"} get${7 - i === 1 ? "s" : ""} −∞ and ${i + 1} score${i === 0 ? "" : "s"} remain${i === 0 ? "s" : ""}. The largest remaining is ${num(S[i][bVis], 2)} on ${name(bVis)}.`;
      } else {
        const free = M.softmax(S[i]);
        const fut = free.slice(i + 1).reduce((a, b) => a + b, 0);
        expl.innerHTML = `Row ${i}, ${tok}: the ${i + 1} weight${i === 0 ? "" : "s"} sum${i === 0 ? "s" : ""} to 1; the largest is ${num(A[i][bVis], 3)} on ${name(bVis)}.` +
          (i === 7 ? ` Nothing was masked in this row.` : ` Without the mask, softmax over all 8 scores would put ${num(fut, 3)} on positions ${i + 1}–7.`);
      }
    }
    draw();
  });

  // Fig: why scores are divided by √d. 250 seeded queries, each against 8 keys, all
  // entries independent N(0, 1). Same draws for every d, so the plate is reproducible.
  function scaleExperiment(d, seed = 11, rows = 250, keys = 8) {
    const rand = M.mulberry32(seed);
    let spare = null;
    const normal = () => {
      if (spare !== null) { const z = spare; spare = null; return z; }
      let u = 0;
      while (u === 0) u = rand();
      const v = rand(), rr = Math.sqrt(-2 * Math.log(u));
      spare = rr * Math.sin(2 * Math.PI * v);
      return rr * Math.cos(2 * Math.PI * v);
    };
    const dots = [];
    let maxRaw = 0, maxScaled = 0, sat = 0;
    for (let r = 0; r < rows; r++) {
      const q = Array.from({ length: d }, normal);
      const sc = [];
      for (let j = 0; j < keys; j++) {
        const k = Array.from({ length: d }, normal);
        let dot = 0;
        for (let c = 0; c < d; c++) dot += q[c] * k[c];
        sc.push(dot); dots.push(dot);
      }
      const pr = Math.max(...M.softmax(sc)), ps = Math.max(...M.softmax(sc.map((x) => x / Math.sqrt(d))));
      maxRaw += pr; maxScaled += ps; if (pr > 0.99) sat++;
    }
    const mean = dots.reduce((a, b) => a + b, 0) / dots.length;
    const sd = Math.sqrt(dots.reduce((a, b) => a + (b - mean) ** 2, 0) / dots.length);
    return { d, dots, sd, sdScaled: sd / Math.sqrt(d), maxRaw: maxRaw / rows, maxScaled: maxScaled / rows, sat: sat / rows };
  }
  Deck.fig("tf-scale", (el, sp) => {
    const DS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let di = Math.max(0, DS.indexOf(sp.d ?? 128));
    const cache = new Map();
    const run = (d) => { if (!cache.has(d)) cache.set(d, scaleExperiment(d, sp.seed ?? 11)); return cache.get(d); };
    const host = h("div", { class: "tf-scale-plot" });
    const over = h("p", { class: "f-note" });
    const tiles = h("div", { class: "f-tiles" });
    const sl = slider({ label: "Head size", min: 0, max: DS.length - 1, step: 1, value: di, format: (i) => String(DS[i]), onInput: (i) => { di = i; draw(); } });
    el.append(host, h("div", { class: "legend" },
      h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "q · k (unscaled)"),
      h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "q · k / √d")), over, h("div", { class: "f-ctl sliders" }, sl), tiles);
    const X0 = -30, X1 = 30;
    const ch = chart(host, {
      width: 560, height: 230, margin: { l: 46, r: 14, t: 10, b: 36 }, aria: "Histograms of unscaled and scaled dot products",
      x: { domain: [X0, X1], ticks: [-30, -20, -10, 0, 10, 20, 30], label: "score" }, y: { domain: [0, 800], ticks: [0, 200, 400, 600, 800], label: "count per unit" },
      series: [],
    });
    const hist = (vals) => {
      const bins = new Array(X1 - X0).fill(0);
      let out = 0;
      for (const v of vals) { const b = Math.floor(v - X0); if (b < 0 || b >= bins.length) out++; else bins[b]++; }
      const pts = [[X0, 0]];
      bins.forEach((c, b) => { pts.push([X0 + b, c], [X0 + b + 1, c]); });
      pts.push([X1, 0]);
      return { pts, out };
    };
    function draw() {
      const d = DS[di], e = run(d);
      const raw = hist(e.dots), scaled = hist(e.dots.map((x) => x / Math.sqrt(d)));
      ch.update({ series: [{ points: raw.pts, cls: "s2" }, { points: scaled.pts, cls: "s1" }] });
      over.textContent = raw.out ? `${raw.out.toLocaleString("en-US")} of 2,000 unscaled scores fall outside ±30 and are not drawn.` : "All 2,000 scores fall inside ±30.";
      clear(tiles);
      tiles.append(
        tile(num(e.sd, 2), String.raw`std of \(q \cdot k\)`), tile(num(Math.sqrt(d), 2), String.raw`\(\sqrt{d}\)`), tile(num(e.sdScaled, 3), String.raw`std of \(q \cdot k / \sqrt{d}\)`),
        tile(num(e.maxRaw, 3), "largest weight, unscaled (mean)"), tile(num(e.maxScaled, 3), "largest weight, scaled (mean)"),
        tile(`${Math.round(e.sat * 100)}%`, "rows with a weight above 0.99, unscaled"));
    }
    draw();
  });


  // Fig: the MLP of one position. GELU drawn as a curve, the 64 hidden units of the
  // chosen position and layer as dots (pre-activation z, GELU(z)), from the trace.
  Deck.fig("tf-mlp", (el, sp) => {
    let pos = sp.pos ?? 4, layer = sp.layer ?? 0;
    const host = h("div", { class: "tf-mlp-plot" });
    const tiles = h("div", { class: "f-tiles" });
    el.append(host,
      h("div", { class: "f-ctl" },
        segmented({ label: "Layer", value: layer, options: [{ value: 0, label: "1" }, { value: 1, label: "2" }], onChange: (v) => { layer = v; draw(); } }),
        segmented({ label: "Position", value: pos, options: TOKS.slice(0, 8).map((t, i) => ({ value: i, label: short(t) })), onChange: (v) => { pos = v; draw(); } })),
      tiles);
    const ch = chart(host, { width: 560, height: 250, margin: { l: 46, r: 14, t: 10, b: 38 }, aria: "GELU and the hidden units of one position", x: { domain: [-4, 4] }, y: { domain: [-1, 4] }, series: [] });
    function draw() {
      const pre = TR.layers[layer].pre[pos], act = TR.layers[layer].act[pos];
      const lo = Math.min(-4, Math.floor(Math.min(...pre)) - 1), hi = Math.max(4, Math.ceil(Math.max(...pre)) + 1);
      const curve = [];
      for (let k = 0; k <= 300; k++) { const x = lo + ((hi - lo) * k) / 300; curve.push([x, M.gelu(x)]); }
      ch.update({
        x: { domain: [lo, hi], ticks: window.UI.ticks([lo, hi]), label: "pre-activation z" },
        y: { domain: [-1, hi], ticks: window.UI.ticks([-1, hi]), label: "GELU(z)" },
        series: [{ points: curve, cls: "ink thin" }],
        dots: pre.map((z, u) => ({ x: z, y: act[u], r: 3.6, cls: act[u] > 0.1 ? "s2" : "s1", tip: `unit ${u}: z = ${num(z, 3)}, GELU(z) = ${num(act[u], 3)}` })),
      });
      clear(tiles);
      tiles.append(
        tile(String(pre.filter((z) => z > 0).length), "units with z > 0"),
        tile(String(act.filter((a) => a > 0.1).length), "units with GELU(z) > 0.1"),
        tile(`${num(Math.min(...pre), 2)} to ${num(Math.max(...pre), 2)}`, "range of z"),
        tile(num(norm2(TR.layers[layer].mlpOut[pos]), 2), "length of the MLP output"));
    }
    draw();
  });

  // Forward pass with sublayers switched off: on = [L1 attention, L1 MLP, L2 attention, L2 MLP].
  // A switched-off sublayer adds nothing to the stream; everything else is unchanged.
  function ablate(ids, on) {
    const T = ids.length, d = 16, hd = 8;
    let x = ids.map((tid, t) => M.row(WTS.wte, tid).map((v, c) => v + WTS.wpe.data[t * d + c]));
    for (let l = 0; l < 2; l++) {
      const P = (k) => WTS[`l${l}.${k}`];
      if (on[2 * l]) {
        const h1 = x.map((v) => M.rmsnorm(v, P("n1")));
        const q = h1.map((v) => M.vecMat(v, P("wq"))), k = h1.map((v) => M.vecMat(v, P("wk"))), v = h1.map((u) => M.vecMat(u, P("wv")));
        const cat = Array.from({ length: T }, () => new Array(d).fill(0));
        for (let hh = 0; hh < 2; hh++) for (let i = 0; i < T; i++) {
          const sc = [];
          for (let j = 0; j <= i; j++) { let z = 0; for (let c = 0; c < hd; c++) z += q[i][hh * hd + c] * k[j][hh * hd + c]; sc.push(z / Math.sqrt(hd)); }
          const a = M.softmax(sc);
          for (let j = 0; j <= i; j++) for (let c = 0; c < hd; c++) cat[i][hh * hd + c] += a[j] * v[j][hh * hd + c];
        }
        const o = cat.map((u) => M.vecMat(u, P("wo")));
        x = x.map((u, t) => u.map((z, c) => z + o[t][c]));
      }
      if (on[2 * l + 1]) {
        const m = x.map((u) => M.vecMat(M.vecMat(M.rmsnorm(u, P("n2")), P("w1")).map(M.gelu), P("w2")));
        x = x.map((u, t) => u.map((z, c) => z + m[t][c]));
      }
    }
    return x.map((u) => M.softmax(M.vecMat(M.rmsnorm(u, WTS.nf), WTS.wu)));
  }
  Deck.fig("tf-ablate", (el, sp) => {
    const NAMES = ["Layer 1 attention", "Layer 1 MLP", "Layer 2 attention", "Layer 2 MLP"];
    let on = (sp.on || [1, 1, 1, 1]).slice();
    const QS = M.allQueries().map((q) => ({ p: q.p, ids: M.encode(q.text).map(M.idOf) }));
    const cache = new Map();
    const evalOn = (key) => {
      if (!cache.has(key)) {
        const o = key.split("").map(Number), per = new Array(8).fill(0);
        for (const q of QS) { const pr = ablate(q.ids.slice(0, 8), o); for (let i = 0; i < 8; i++) per[i] += -Math.log(pr[i][q.ids[i + 1]]) * q.p; }
        cache.set(key, { ours: ablate(IDS, o), per, mean: per.reduce((a, b) => a + b, 0) / 8 });
      }
      return cache.get(key);
    };
    const btns = NAMES.map((n, k) => h("button", { type: "button", class: "btn btn-ghost tf-tog", "aria-pressed": "true", onclick: () => { on[k] = 1 - on[k]; draw(); } }, n));
    const table = h("div", { class: "scroll-x" });
    const tiles = h("div", { class: "f-tiles" });
    el.append(h("div", { class: "f-ctl tf-tog-row" }, btns, button("All on", () => { on = [1, 1, 1, 1]; draw(); })), table, tiles);
    const full = evalOn("1111");
    const bar = (p, cls) => h("span", { class: "tf-abl-p" }, h("span", { class: "w-track" }, h("span", { class: `w-fill ${cls || ""}`, style: { width: `${(p * 100).toFixed(1)}%` } })), h("span", { class: "num" }, fmt.p(p)));
    function draw() {
      btns.forEach((b, k) => b.setAttribute("aria-pressed", String(!!on[k])));
      const res = evalOn(on.join(""));
      const t = h("table", { class: "tf-abl" });
      t.appendChild(h("tr", null, h("th", null, "next token"), h("th", null, "p, all sublayers"), h("th", null, "p, your switches"), h("th", { class: "r" }, "loss here, all queries")));
      for (let i = 0; i < 8; i++) {
        const tgt = M.runningIds[i + 1];
        t.appendChild(h("tr", { class: Math.abs(res.per[i] - full.per[i]) > 0.01 ? "is-hit" : "" },
          h("td", null, chip(TOKS[i + 1])), h("td", null, bar(full.ours[i][tgt], "")), h("td", null, bar(res.ours[i][tgt], "b")),
          h("td", { class: "r num", "data-tip": `full model: ${num(full.per[i], 4)}` }, num(res.per[i], 3))));
      }
      clear(table);
      table.appendChild(t);
      clear(tiles);
      tiles.append(tile(num(res.mean, 4), "exact loss, your switches"), tile(num(full.mean, 4), "exact loss, all sublayers"), tile(`${num(res.mean - full.mean, 4)}`, "change"));
    }
    draw();
  });


  // Layer-1 attention output (after W_O) at the last position for a given order of our
  // 8 tokens, with or without the position embeddings added.
  function layer1Last(order, usePos) {
    const X = order.map((i, t) => M.row(WTS.wte, IDS[i]).map((v, c) => v + (usePos ? WTS.wpe.data[t * 16 + c] : 0)));
    const P = (k) => WTS[`l0.${k}`];
    const hn = X.map((v) => M.rmsnorm(v, P("n1")));
    const q = M.vecMat(hn[7], P("wq")), k = hn.map((v) => M.vecMat(v, P("wk"))), v = hn.map((u) => M.vecMat(u, P("wv")));
    const cat = new Array(16).fill(0), att = [];
    for (let hh = 0; hh < 2; hh++) {
      const a = M.softmax(k.map((kv) => { let z = 0; for (let c = 0; c < 8; c++) z += q[hh * 8 + c] * kv[hh * 8 + c]; return z / Math.sqrt(8); }));
      att.push(a);
      a.forEach((w, j) => { for (let c = 0; c < 8; c++) cat[hh * 8 + c] += w * v[j][hh * 8 + c]; });
    }
    return { out: M.vecMat(cat, P("wo")), att };
  }
  Deck.fig("tf-perm", (el, sp) => {
    const rand = M.mulberry32(sp.seed ?? 3);
    const BASE = [0, 1, 2, 3, 4, 5, 6, 7];
    let order = BASE.slice(), usePos = false;
    const ref = { false: layer1Last(BASE, false), true: layer1Last(BASE, true) };
    const orderRow = h("div", { class: "scroll-x" });
    const vecs = h("div", { class: "tf-perm-vecs" });
    const tiles = h("div", { class: "f-tiles" });
    const posSeg = segmented({ label: "Position embeddings", value: 0, options: [{ value: 0, label: "left out" }, { value: 1, label: "added" }], onChange: (v) => { usePos = !!v; draw(); } });
    el.append(h("div", { class: "f-ctl" },
      button("Shuffle positions 0–6", () => {
        const a = order.slice(0, 7);
        for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
        order = a.concat([7]); draw();
      }, { kind: "primary" }),
      button("Original order", () => { order = BASE.slice(); draw(); }), posSeg), orderRow, vecs, tiles);
    function draw() {
      const cur = layer1Last(order, usePos), base = ref[usePos];
      const diff = Math.max(...cur.out.map((v, c) => Math.abs(v - base.out[c])));
      const t = h("table", { class: "f-seq tf-perm-t" });
      t.appendChild(h("tr", null, h("th", null, "position"), order.map((_, i) => h("td", { class: i === 7 ? "on" : "" }, String(i)))));
      t.appendChild(h("tr", null, h("th", null, "token"), order.map((o, i) => h("td", { class: i === 7 ? "on" : "" }, chip(TOKS[o])))));
      t.appendChild(h("tr", null, h("th", null, "head 1 weight"), cur.att[0].map((w, i) => h("td", { class: i === 7 ? "on" : "", "data-tip": num(w, 4) }, num(w, 3)))));
      clear(orderRow); orderRow.appendChild(t);
      const m = Math.max(...base.out.map(Math.abs), ...cur.out.map(Math.abs));
      clear(vecs);
      vecs.append(
        h("div", { class: "tf-perm-v" }, h("span", { class: "tf-perm-l" }, "output at position 7, original order"), heatStrip(base.out, { maxAbs: m })),
        h("div", { class: "tf-perm-v" }, h("span", { class: "tf-perm-l" }, "output at position 7, this order"), heatStrip(cur.out, { maxAbs: m })));
      clear(tiles);
      tiles.append(
        tile(diff === 0 ? "0" : diff < 1e-9 ? fmt.sci(diff, 1) : num(diff, 3), "largest change in the 16 outputs"),
        tile(order.join(" "), "order of the original positions"));
    }
    draw();
  });

  // Fig: RoPE in two dimensions. q is rotated by m·θ, k by n·θ; their dot product
  // depends only on (m − n)·θ.
  Deck.fig("tf-rope", (el, sp) => {
    let m = sp.m ?? 3, n = sp.n ?? 1, th = sp.theta ?? 0.5;
    const q0 = sp.q || [0.6, 0.8], k0 = sp.k || [1, 0];
    const rot = (v, a) => [Math.cos(a) * v[0] - Math.sin(a) * v[1], Math.sin(a) * v[0] + Math.cos(a) * v[1]];
    const Wd = 340, R = 1.3, X = (v) => Wd / 2 + (v / R) * (Wd / 2 - 14), Y = (v) => Wd / 2 - (v / R) * (Wd / 2 - 14);
    const svg = s("svg", { viewBox: `0 0 ${Wd} ${Wd}`, class: "tf-rope-svg", role: "img", "aria-label": "Query and key rotated by their positions" });
    const tiles = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl tf-expl", "aria-live": "polite" });
    const sm = slider({ label: "Query position m", min: 0, max: 12, step: 1, value: m, format: (v) => String(v), onInput: (v) => { m = v; draw(); } });
    const sn = slider({ label: "Key position n", min: 0, max: 12, step: 1, value: n, format: (v) => String(v), onInput: (v) => { n = v; draw(); } });
    const st = slider({ label: "θ (radians per position)", min: 0.05, max: 1.5, step: 0.05, value: th, format: (v) => v.toFixed(2), onInput: (v) => { th = v; draw(); } });
    const shift = button("Shift both by +1", () => { if (m < 12 && n < 12) { m++; n++; sm.set(m); sn.set(n); draw(); } }, { kind: "primary" });
    el.append(h("div", { class: "tf-rope-wrap" }, svg, h("div", { class: "tf-rope-side" }, tiles, expl)),
      h("div", { class: "f-ctl sliders" }, sm, sn, st), h("div", { class: "f-ctl" }, shift, button("Reset", () => { m = sp.m ?? 3; n = sp.n ?? 1; th = sp.theta ?? 0.5; sm.set(m); sn.set(n); st.set(th); draw(); })));
    const uid = window.UI.nextId("tfr");
    function arrow(v, cls, label) {
      svg.appendChild(s("line", { x1: X(0), y1: Y(0), x2: X(v[0]), y2: Y(v[1]), class: `tf-rope-a ${cls}`, "marker-end": `url(#${uid}-${cls})` }));
      if (label) {
        const lx = Math.max(44, Math.min(Wd - 44, X(1.16 * v[0]))), ly = Math.max(14, Math.min(Wd - 6, Y(1.16 * v[1]) + 4));
        svg.appendChild(s("text", { x: lx, y: ly, class: "tf-rope-l", "text-anchor": "middle" }, label));
      }
    }
    function draw() {
      clear(svg);
      svg.appendChild(s("defs", null, ["q", "k", "q0", "k0"].map((c) => s("marker", { id: `${uid}-${c}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: `tf-rope-ah ${c}` })))));
      svg.appendChild(s("line", { x1: X(-R), x2: X(R), y1: Y(0), y2: Y(0), class: "tf-rope-axis" }));
      svg.appendChild(s("line", { x1: X(0), x2: X(0), y1: Y(-R), y2: Y(R), class: "tf-rope-axis" }));
      svg.appendChild(s("circle", { cx: X(0), cy: Y(0), r: X(1) - X(0), class: "tf-rope-circ" }));
      const q = rot(q0, m * th), k = rot(k0, n * th);
      const aq = Math.atan2(q[1], q[0]), ak = Math.atan2(k[1], k[0]);
      // arc between the rotated vectors, radius 0.35
      let d = aq - ak; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
      const rr = 0.35, a0 = ak, a1 = ak + d;
      svg.appendChild(s("path", { d: `M${X(rr * Math.cos(a0))},${Y(rr * Math.sin(a0))} A${(X(rr) - X(0)).toFixed(2)},${(X(rr) - X(0)).toFixed(2)} 0 0 ${d > 0 ? 0 : 1} ${X(rr * Math.cos(a1))},${Y(rr * Math.sin(a1))}`, class: "tf-rope-arc" }));
      arrow(q0, "q0", null); arrow(k0, "k0", null);
      arrow(q, "q", `q at m = ${m}`);
      arrow(k, "k", `k at n = ${n}`);
      const dot = q[0] * k[0] + q[1] * k[1];
      clear(tiles);
      tiles.append(
        tile(`(${num(q[0], 3)}, ${num(q[1], 3)})`, String.raw`\(R(m\theta)\,\mathbf{q}\)`),
        tile(`(${num(k[0], 3)}, ${num(k[1], 3)})`, String.raw`\(R(n\theta)\,\mathbf{k}\)`),
        tile(num(dot, 3), "score q · k after rotation"),
        tile(num((m - n) * th, 2), String.raw`\((m - n)\,\theta\) in radians`));
      expl.innerHTML = `m − n = ${m - n}. Unrotated, q · k = ${num(q0[0] * k0[0] + q0[1] * k0[1], 3)}. Every (m, n) with the same difference gives ${num(dot, 3)} at this θ; shift both positions to check.`;
      shift.disabled = m >= 12 || n >= 12;
    }
    draw();
  });


  // Fig: grouped-query attention in Llama 3 70B (64 query heads, head size 128, 80 layers,
  // d = 8,192): which key/value head each query head reads, and what the choice costs.
  Deck.fig("tf-gqa", (el, sp) => {
    const NQ = 64, LAY = 80, HD = 128, DM = 8192;
    let G = sp.kv ?? 8, ctx = sp.ctx ?? 131072;
    const host = h("div", { class: "scroll-x" });
    const tiles = h("div", { class: "f-tiles" });
    el.append(h("div", { class: "f-ctl" },
      segmented({ label: "Key/value heads", value: G, options: [64, 32, 16, 8, 4, 2, 1].map((g) => ({ value: g, label: g === 64 ? "64 (MHA)" : g === 1 ? "1 (MQA)" : g === 8 ? "8 (Llama 3)" : String(g) })), onChange: (v) => { G = v; draw(); } }),
      segmented({ label: "Context", value: ctx, options: [8192, 32768, 131072].map((c) => ({ value: c, label: `${c / 1024}K` })), onChange: (v) => { ctx = v; draw(); } })),
      host, tiles);
    function draw() {
      const Wd = 600, Hd = 158, x0 = 10, cw = (Wd - 2 * x0) / NQ, per = NQ / G;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "tf-gqa-svg", role: "img", "aria-label": `${NQ} query heads sharing ${G} key/value heads` });
      svg.appendChild(s("text", { x: x0, y: 13, class: "tf-gqa-t" }, "64 query heads"));
      svg.appendChild(s("text", { x: x0, y: Hd - 4, class: "tf-gqa-t" }, `${G} key/value head${G === 1 ? "" : "s"}, each shared by ${per} query head${per === 1 ? "" : "s"}`));
      for (let g = 0; g < G; g++) {
        const kx = x0 + (g + 0.5) * per * cw, kw = Math.max(5, per * cw - 3);
        for (let j = g * per; j < (g + 1) * per; j++) {
          const qx = x0 + (j + 0.5) * cw;
          svg.appendChild(s("line", { x1: qx, y1: 46, x2: kx, y2: 104, class: `tf-gqa-ln g${g % 2}` }));
          svg.appendChild(s("rect", { x: x0 + j * cw + 0.6, y: 22, width: cw - 1.2, height: 22, class: `tf-gqa-q g${g % 2}`, "data-tip": `query head ${j + 1} reads key/value head ${g + 1}` }));
        }
        svg.appendChild(s("rect", { x: kx - kw / 2, y: 106, width: kw, height: 26, class: `tf-gqa-kv g${g % 2}`, "data-tip": `key/value head ${g + 1}: K and V of size 128 for every past token` }));
      }
      clear(host);
      host.appendChild(svg);
      const perTok = 2 * LAY * G * HD * 2;
      clear(tiles);
      tiles.append(
        tile(String(per), "query heads per key/value head"),
        tile((2 * DM * G * HD).toLocaleString("en-US"), String.raw`\(\mathbf{W}_K\) and \(\mathbf{W}_V\) weights per layer`),
        tile(`${perTok.toLocaleString("en-US")} B`, "cache per token, 80 layers, BF16"),
        tile(`${(perTok * ctx / 1e9).toFixed(1)} GB`, `cache for one ${ctx / 1024}K-token sequence`));
    }
    draw();
  });

  // Fig: causal versus sliding-window masks. A window of W keeps itself and the W − 1
  // positions before it. After L layers the last position can depend on L·(W − 1) + 1 positions.
  Deck.fig("tf-swa", (el, sp) => {
    let T = sp.T ?? 16, Wn = sp.W ?? 4, Ly = sp.L ?? 2;
    const grids = h("div", { class: "tf-swa-grids" });
    const strip = h("div", { class: "tf-swa-strip-wrap" });
    const tiles = h("div", { class: "f-tiles" });
    el.append(grids, strip,
      h("div", { class: "f-ctl sliders" },
        slider({ label: "Tokens T", min: 4, max: 32, step: 1, value: T, format: (v) => String(v), onInput: (v) => { T = v; draw(); } }),
        slider({ label: "Window W", min: 1, max: 32, step: 1, value: Wn, format: (v) => String(v), onInput: (v) => { Wn = v; draw(); } }),
        slider({ label: "Layers L", min: 1, max: 6, step: 1, value: Ly, format: (v) => String(v), onInput: (v) => { Ly = v; draw(); } })),
      tiles);
    const grid = (title, keep) => {
      const size = 180, cs = size / T;
      const svg = s("svg", { viewBox: `0 0 ${size + 2} ${size + 2}`, class: "tf-swa-svg", role: "img", "aria-label": `${title} attention mask for ${T} tokens` });
      let n = 0;
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        const on = keep(i, j);
        if (on) n++;
        svg.appendChild(s("rect", { x: 1 + j * cs, y: 1 + i * cs, width: cs, height: cs, class: on ? "tf-swa-on" : "tf-swa-off" }));
      }
      return { el: h("figure", { class: "tf-swa-fig" }, h("figcaption", null, title), svg), n };
    };
    function draw() {
      const w = Math.min(Wn, T);
      const a = grid("Causal", (i, j) => j <= i), b = grid(`Window W = ${Wn}`, (i, j) => j <= i && j > i - w);
      clear(grids);
      grids.append(a.el, b.el);
      const reach = Math.min(T, Ly * (w - 1) + 1), first = T - reach;
      clear(strip);
      strip.append(h("p", { class: "tf-swa-cap" }, `Inputs the last position can depend on after ${Ly} layer${Ly === 1 ? "" : "s"} of windowed attention:`),
        h("div", { class: "tf-swa-strip" }, Array.from({ length: T }, (_, j) => h("span", { class: j >= first ? "on" : "", "data-tip": `position ${j}` }))));
      clear(tiles);
      tiles.append(
        tile(a.n.toLocaleString("en-US"), "scored pairs, causal"),
        tile(b.n.toLocaleString("en-US"), "scored pairs, window"),
        tile(`${T} vs ${w}`, "keys kept per layer, causal vs window"),
        tile(`${reach} of ${T}`, `positions reached after ${Ly} layer${Ly === 1 ? "" : "s"}`));
    }
    draw();
  });

  // Mean attention weights over all 270 queries, grouped by the selected column.
  const ROLE_COLS = ["id", "name", "email", "age", "user_id", "total", "status", "title", "price"];
  const ROLE_PAIRS = [[0, 0, 4, 1], [0, 0, 4, 3], [0, 1, 4, 1], [0, 1, 4, 3], [0, 1, 2, 1], [1, 1, 2, 1]]; // layer, head, row, column
  // computed on first use (270 forward passes), so page load does not pay for it
  let roleMean = null;
  const ROLE_MEAN = () => roleMean || (roleMean = (() => {
    const acc = ROLE_COLS.map(() => ROLE_PAIRS.map(() => 0)), w = ROLE_COLS.map(() => 0);
    for (const q of M.allQueries()) {
      const t = M.forward(M.encode(q.text).map(M.idOf).slice(0, 8)), r = ROLE_COLS.indexOf(q.c1);
      w[r] += q.p;
      ROLE_PAIRS.forEach(([l, hh, i, j], c) => { acc[r][c] += q.p * t.layers[l].heads[hh].att[i][j]; });
    }
    return acc.map((row, r) => row.map((v) => v / w[r]));
  })());

  // ------------------------------------------------------------------ sheets
  Deck.part({ n: 5, title: "The transformer", short: "Transformer" });
  // The engine needs a current chapter before any sheet, so the part opener sits at
  // the start of chapter 5.1; chapter codes then match CURRICULUM.md.
  Deck.chapter({ title: "Why attention" });

  Deck.slide({
    id: "tf-part",
    layout: "part",
    title: "The transformer",
    body: String.raw`<p>This part computes the whole transformer on our query with the trained weights: the embeddings, attention, the MLP, the residual stream and the output layer. Every number is read from the model as it runs in this page, so each step can be checked by hand.</p>
<p>It then follows the training run of this model, measures what it learned, and makes the changes that turn it into a Llama-class architecture (rotary positions, SwiGLU, grouped-query attention). It ends with the mixture-of-experts layers of Mixtral and DeepSeek-V3.</p>`,
    take: "Compute every intermediate of a transformer forward pass, count its parameters and FLOPs term by term, and read a Llama 3 configuration line by line.",
  });

  // ================================================================== 5.1 (chapter opened above)

  Deck.slide({
    id: "tf-need",
    title: "Each prediction needs specific earlier tokens",
    example: "tokens ids focus:1,3 target:4",
    body: String.raw`<p>At position 2 the model reads <code>FROM</code> and must predict the table. The vector it starts with there is the embedding of <code>FROM</code> plus the embedding of position 2: the same 16 numbers in all 270 queries, because every query puts <code>FROM</code> at position 2. That vector cannot say which table comes next. The answer is at position 1: <code>name</code> exists only in <code>users</code>, <code>total</code> only in <code>orders</code>.</p>
<p>At position 4, <code>WHERE</code>, the prediction needs two earlier tokens: the table (position 3), which fixes the allowed columns, and the selected column (position 1), which the filter may not repeat. For our query the trained model gives <code>id</code>, <code>email</code> and <code>age</code> 0.346, 0.345 and 0.307, and <code>name</code> 0.001.</p>
<p>The distances differ, 1 position back for the table and 3 for the column, and the useful positions depend on the content. A fixed rule such as “read the previous token” cannot supply both.</p>`,
    fig: { type: "tinylm", widget: "need", caption: "<span>Edit the query: the trained model's predictions after <code>FROM</code> and after <code>WHERE</code> follow the earlier tokens</span>" },
    take: "The input vector at a position does not determine its prediction. The model needs to read specific earlier positions, chosen by content.",
  });

  Deck.slide({
    id: "tf-rnn",
    title: "A recurrent network reads one token at a time",
    example: "tokens ids",
    body: String.raw`<p>Before transformers, the best language models were <b>recurrent neural networks</b> (RNNs), mostly the LSTM and GRU variants. An RNN keeps one state vector and updates it once per token, \(\mathbf{h}_t = \tanh(\mathbf{W}\mathbf{h}_{t-1} + \mathbf{U}\mathbf{x}_t)\), and the prediction at position t reads only \(\mathbf{h}_t\).</p>
<p>Two properties follow from that formula. Step t needs the result of step t − 1, so our 8 positions take 8 dependent steps and a 128K-token document takes 131,072; the positions of one sequence cannot be computed in parallel. And every earlier token must survive inside one fixed-size vector: <code>name</code> at position 1 reaches <code>WHERE</code> at position 4 only through 3 more updates.</p>
<p>Vaswani et al. (2017) compared layer types on exactly these counts. A self-attention layer connects any two positions in one step and has no sequential dependency inside a layer. It pays with work that grows as n² instead of n.</p>`,
    fig: {
      type: "table", caption: "Layer types compared (n = sequence length, d = vector size), with our query's n = 8, d = 16",
      cols: ["", "Self-attention", "Recurrent"], align: ["l", "r", "r"], rowHeads: true,
      rows: [
        ["Work per layer", String.raw`\(O(n^2 \cdot d)\)`, String.raw`\(O(n \cdot d^2)\)`],
        ["… at n = 8, d = 16", "n²·d = 1,024", "n·d² = 2,048"],
        ["Sequential steps per layer", String.raw`\(O(1)\)`, String.raw`\(O(n)\)`],
        ["… at n = 8", "1", "8"],
        ["Longest path between two positions", String.raw`\(O(1)\)`, String.raw`\(O(n)\)`],
      ],
      note: "Order-of-growth rows from Vaswani et al. (2017), Table 1; the evaluated rows keep only those leading terms. A self-attention layer also has its Q, K, V and output projections, 4·n·d² = 8,192 multiply-adds here. LSTMs and GRUs are RNNs with gated updates and the same sequential structure.",
    },
    terms: [{ term: "Recurrent neural network (RNN)", def: "A sequence model that keeps one state vector and updates it once per token, so step t depends on step t − 1." }],
    src: ["Vaswani et al., Attention Is All You Need, NeurIPS 2017, section 1 and Table 1"],
  });

  Deck.slide({
    id: "tf-attn-idea",
    title: "Attention: a weighted average over earlier positions",
    example: "tokens ids focus:4",
    body: String.raw`<p><b>Self-attention</b> gives every position a direct read of every earlier position. Position i computes a weight \(a_{ij}\) for each position j ≤ i, all positive and summing to 1, and outputs the weighted average of vectors computed from those positions.</p>
<p>The weights are computed from the vectors themselves, so they change with the content. The plate shows the weights that the two attention heads of layer 1 give from position 4, <code>WHERE</code>. Head 2 puts 1.00 on <code>users</code> and less than 0.005 anywhere else, so its output at <code>WHERE</code> equals the vector it reads from <code>users</code> to 3 decimals. Head 1 puts 0.98 on <code>name</code>.</p>
<p>Chapter 5.2 derives where the weights come from (queries and keys) and what is averaged (values). The shape of the computation is the point here: every position reads any earlier position in one step, with weights decided by the data.</p>`,
    math: [String.raw`\mathrm{out}_i = \sum_{j \le i} a_{ij}\, \mathbf{v}_j`, String.raw`a_{ij} \ge 0, \qquad \sum_{j \le i} a_{ij} = 1`],
    fig: {
      type: "sequence", caption: "<span>Attention weights from position 4, <code>WHERE</code>, in layer 1 of the trained model; later positions are not visible</span>",
      tokens: TOKS.slice(0, 8), focus: [4],
      rows: [
        { label: "head 1", values: TR.layers[0].heads[0].att[4].map((v, j) => (j > 4 ? "–" : v)), dp: 2 },
        { label: "head 2", values: TR.layers[0].heads[1].att[4].map((v, j) => (j > 4 ? "–" : v)), dp: 2 },
      ],
      note: "Each row sums to 1 over positions 0 to 4.",
    },
    terms: [{ term: "Self-attention", def: "A layer in which every position outputs a weighted average of vectors from the positions it may see, with weights computed from the data." }],
  });

  Deck.slide({
    id: "tf-plan",
    title: "The plan of our transformer",
    layout: "wide",
    example: "tokens ids",
    body: String.raw`<p>Each of the 8 positions carries a vector of 16 numbers from bottom to top. Two lookup tables turn token ids and positions into the first vectors. Two blocks follow; each adds an attention result (positions read each other) and then an MLP result (each position transformed on its own). A final RMSNorm and a 16 × 113 matrix turn each vector into 113 scores, and softmax turns the scores into a distribution over the next token.</p>
<p>Shapes for one query: 8 ids, then 8 × 16 through both blocks, then 8 × 113 logits and probabilities. Row i of the output is the prediction for token i + 1, so one pass yields 8 predictions.</p>`,
    fig: {
      type: "diagram", caption: "Our model from token ids to next-token probabilities, with the inside of a block below", w: 27.8, h: 7.7, unit: 40,
      aria: "Data flow of the model: ids, embedding plus position, block 1, block 2, final norm, unembedding, softmax; inside a block, norm and attention add to the stream, then norm and MLP add to it",
      groups: [{ x: 3.6, y: 3.05, w: 22.9, h: 4.45, label: "inside each block" }],
      nodes: [
        { id: "ids", x: 0.2, y: 0.6, w: 2.0, h: 1.5, label: "Token ids", sub: "8 integers" },
        { id: "emb", x: 3.2, y: 0.6, w: 3.8, h: 1.5, label: "Embed + position", sub: "W_E[id] + W_P[t]" },
        { id: "b1", x: 8.2, y: 0.6, w: 2.8, h: 1.5, label: "Block 1", sub: "attention, MLP", kind: "blue" },
        { id: "b2", x: 12.2, y: 0.6, w: 2.8, h: 1.5, label: "Block 2", sub: "attention, MLP", kind: "blue" },
        { id: "nf", x: 16.2, y: 0.6, w: 3.0, h: 1.5, label: "Final RMSNorm", sub: "gain: 16" },
        { id: "wu", x: 20.4, y: 0.6, w: 3.0, h: 1.5, label: "Unembedding", sub: "W_U: 16 × 113" },
        { id: "sm", x: 24.9, y: 0.6, w: 2.7, h: 1.5, label: "Softmax", sub: "next-token p", kind: "hi" },
        { id: "xin", x: 4.0, y: 5.9, w: 1.6, h: 1.0, label: "x" },
        { id: "n1", x: 6.2, y: 3.85, w: 2.4, h: 1.0, label: "RMSNorm" },
        { id: "att", x: 9.2, y: 3.85, w: 3.0, h: 1.0, label: "Attention" },
        { id: "a1", x: 12.6, y: 5.9, w: 1.0, h: 1.0, label: "+", round: true },
        { id: "n2", x: 14.4, y: 3.85, w: 2.4, h: 1.0, label: "RMSNorm" },
        { id: "mlp", x: 17.4, y: 3.85, w: 3.0, h: 1.0, label: "MLP" },
        { id: "a2", x: 20.8, y: 5.9, w: 1.0, h: 1.0, label: "+", round: true },
        { id: "xout", x: 23.8, y: 5.9, w: 2.2, h: 1.0, label: "x, updated" },
      ],
      edges: [
        { from: "ids", to: "emb", label: "8" }, { from: "emb", to: "b1", label: "8 × 16" }, { from: "b1", to: "b2", label: "8 × 16" },
        { from: "b2", to: "nf", label: "8 × 16" }, { from: "nf", to: "wu", label: "8 × 16" }, { from: "wu", to: "sm", label: "8 × 113" },
        { from: "xin", to: "a1" }, { from: "xin", to: "n1" }, { from: "n1", to: "att" }, { from: "att", to: "a1" },
        { from: "a1", to: "a2" }, { from: "a1", to: "n2" }, { from: "n2", to: "mlp" }, { from: "mlp", to: "a2" }, { from: "a2", to: "xout" },
      ],
      notes: [{ x: 5.9, y: 7.35, text: "the residual stream: sublayers add to it" }],
    },
    check: {
      q: "Our model reads 8 tokens. How many next-token distributions does one forward pass produce, and how many numbers is that?",
      options: ["1 distribution, 113 numbers", "8 distributions, 904 numbers", "8 distributions, 128 numbers", "9 distributions, 1,017 numbers"],
      answer: 1,
      why: "Every position predicts the token after it, so the output is 8 rows of 113 probabilities: 904 numbers. 128 = 8 × 16 is the size of the hidden vectors, not of the output.",
    },
  });

  // ================================================================== 5.2
  Deck.chapter({ title: "Self-attention step by step" });

  Deck.slide({
    id: "tf-input",
    title: "The input is token embedding plus position embedding",
    layout: "wide",
    example: "tokens ids focus:4",
    body: String.raw`<p>The first vectors come from two lookup tables. Row 52 of the token table \(\mathbf{W}_E\) (113 × 16) is the embedding of <code>WHERE</code>; row 4 of the position table \(\mathbf{W}_P\) (8 × 16) is the embedding of position 4. Their sum is the vector position 4 starts with, \(\mathbf{x}_4 = \mathbf{W}_E[52] + \mathbf{W}_P[4]\).</p>
<p>The plate shows all three for our query. Check the first entry: −0.694 + 0.003 = −0.691. Stacking the 8 positions gives \(\mathbf{X}\), an 8 × 16 matrix. Every later step transforms it, and the stream keeps this 8 × 16 shape until the output layer.</p>`,
    fig: {
      type: "heat", caption: "Token embedding, position embedding and their sum at position 4, all 16 entries (hover for 3 decimals)",
      rows: [String.raw`\(\mathbf{W}_E[52]\) <code>WHERE</code>`, String.raw`\(\mathbf{W}_P[4]\)`, String.raw`\(\mathbf{x}_4\) = sum`],
      cols: Array.from({ length: 16 }, (_, c) => String(c)),
      matrix: [TR.tokEmb[4], TR.posEmb[4], TR.x0[4]], dp: 2, max: 4.5,
      note: "Blue is negative, red positive. Our model multiplies nothing at this step: both tables are read by index.",
    },
  });

  Deck.slide({
    id: "tf-qkv-maps",
    title: "Queries, keys and values are three projections of one vector",
    example: "tokens ids focus:4",
    note: "Layers and heads are numbered from 1, positions from 0, as in the strip.",
    body: String.raw`<p>Attention starts from \(\hat{\mathbf{x}}_i = \mathrm{RMSNorm}(\mathbf{x}_i)\): the vector divided by its root mean square, then multiplied entry by entry by 16 learned gains (chapter 5.4 works one through). Three learned 16 × 16 matrices map it to three new vectors: a <b>query</b> \(\mathbf{q}_i = \hat{\mathbf{x}}_i \mathbf{W}_Q\), a <b>key</b> \(\mathbf{k}_i = \hat{\mathbf{x}}_i \mathbf{W}_K\) and a <b>value</b> \(\mathbf{v}_i = \hat{\mathbf{x}}_i \mathbf{W}_V\).</p>
<p>Each 16-number result is split into 2 heads of 8 numbers, so each head uses its own 16 → 8 slice of the three maps. A position's query is compared with the keys of the positions it may read; the values are what gets averaged. This query is a vector and has nothing to do with the SQL query.</p>
<p>The plate shows layer 1, head 2: the query of <code>WHERE</code> and the key and value of <code>users</code>. For all 8 positions at once, \(\mathbf{Q} = \hat{\mathbf{X}}\mathbf{W}_Q\) is one (8 × 16)(16 × 16) matrix product.</p>`,
    fig: {
      type: "heat", caption: "<span>Layer 1, head 2: columns 8 to 15 of the query at <code>WHERE</code> and of the key and value at <code>users</code></span>",
      rows: [String.raw`\(\mathbf{q}\) at <code>WHERE</code>`, String.raw`\(\mathbf{k}\) at <code>users</code>`, String.raw`\(\mathbf{v}\) at <code>users</code>`],
      cols: Array.from({ length: 8 }, (_, c) => String(c + 8)),
      matrix: [TR.layers[0].q[4].slice(8, 16), TR.layers[0].k[3].slice(8, 16), TR.layers[0].v[3].slice(8, 16)], dp: 2, max: 6.5,
      note: "The next sheet takes the dot product of the first two rows.",
    },
    terms: [{ term: "Query, key, value", def: "Three linear projections of each position's normalized vector. Scores compare one position's query with other positions' keys; the weighted values form the output." }],
  });

  Deck.slide({
    id: "tf-scores",
    title: "Scores: the query of WHERE against every key",
    example: "tokens ids focus:4",
    body: String.raw`<p>A head decides how much position i reads from position j with a <b>score</b>: the dot product of i's query with j's key, divided by \(\sqrt{8}\) (the next sheet explains the division). A large score means a good match.</p>
<p>In layer 1, head 2, the query of <code>WHERE</code> and the key of <code>users</code> have dot product 40.28; divided by \(\sqrt 8\) that is a score of 14.24. The other visible positions score −3.28 (<code>SELECT</code>), −1.05 (<code>name</code>), −1.36 (<code>FROM</code>) and −10.27 (<code>WHERE</code> itself). Positions 5 to 7 come later and are not scored.</p>
<p>Pick another head or position in the plate. Every score is a dot product of 8 numbers: 8 multiplications and 7 additions.</p>`,
    math: String.raw`s_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{8}}, \qquad j \le i`,
    fig: { type: "tinylm", widget: "qkv", caption: "The query of one position against the keys of positions 0 to 7 (trained weights, our query)" },
    terms: [{ term: "Attention score", def: "The dot product of one position's query with another position's key, divided by the square root of the head size." }],
  });

  Deck.slide({
    id: "tf-sqrt-d",
    title: "Dividing by √d keeps the scores at unit scale",
    body: String.raw`<p>If the entries of \(\mathbf{q}\) and \(\mathbf{k}\) are independent with mean 0 and variance 1, the dot product \(\sum_{c=1}^{d} q_c k_c\) is a sum of d terms of variance 1. Its variance is d and its standard deviation \(\sqrt d\) (Vaswani et al., footnote 4). Dividing by \(\sqrt d\) brings the standard deviation back to 1 for any head size.</p>
<p>The plate draws 250 queries, each against 8 keys, from a seeded random generator. At d = 128, the head size of Llama 3, the unscaled scores have standard deviation 11.40. Softmax over 8 of them puts on average 0.919 of the weight on one key, and 56% of rows put more than 0.99 on one key. Scaled, the standard deviation is 1.008 and the largest weight averages 0.366.</p>
<p>A saturated softmax also stops learning. The derivative of a weight p with respect to its own score is p(1 − p): 0.0099 at p = 0.99, against 0.25 at p = 0.5.</p>`,
    math: [String.raw`\mathrm{Var}(\mathbf{q}\cdot\mathbf{k}) = d`, String.raw`\mathrm{Var}\left(\frac{\mathbf{q}\cdot\mathbf{k}}{\sqrt d}\right) = 1`],
    fig: { type: "tf-scale", caption: "2,000 seeded dot products (250 queries × 8 keys) at the head size you choose", d: 128, seed: 11 },
    src: ["Vaswani et al., Attention Is All You Need, 2017, section 3.2.1 and footnote 4", "Dubey et al., The Llama 3 Herd of Models, 2024, Table 3 (head size 4,096 / 32 = 128)"],
  });

  Deck.slide({
    id: "tf-mask",
    title: "The causal mask hides later positions",
    example: "tokens ids focus:4",
    body: String.raw`<p>In training, all 8 positions are computed in one pass and position i is graded on predicting token i + 1. If position 4 could read position 5, the answer, <code>id</code>, would be in its input. At generation time position 5 does not exist yet, so a model that learned to copy it would fail.</p>
<p>The <b>causal mask</b> adds −∞ to every score with j &gt; i before the softmax. Since \(e^{-\infty} = 0\), later positions get weight exactly 0 and the remaining weights still sum to 1.</p>
<p>The plate steps through it for one head. Scores for later positions can be computed, and nothing in training kept them small: in layer 1, head 1, <code>WHERE</code> scores position 5 at 5.69, higher than any earlier position. Without the mask, that row would put 0.896 of its weight on <code>id</code>, the token it has to predict.</p>`,
    math: [String.raw`\mathbf{A}_h = \mathrm{softmax}\left(\frac{\mathbf{Q}_h\mathbf{K}_h^\top}{\sqrt{8}} + \mathbf{M}\right)`, String.raw`M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j \gt i \end{cases}`],
    fig: { type: "tf-mask", caption: "One head of the trained model on our query: scores, masked scores, weights. Select a row.", head: 0, stage: 0, row: 4 },
    terms: [{ term: "Causal mask", def: "A matrix added to the attention scores that is 0 where j ≤ i and −∞ where j > i, so no position can read a later one." }],
  });

  Deck.slide({
    id: "tf-mix",
    title: "Softmax turns scores into weights, and the weights mix the values",
    example: "tokens ids focus:4",
    body: String.raw`<p>Softmax over the visible scores of position i gives the <b>attention weights</b> \(a_{ij} = e^{s_{ij}} / \sum_{m \le i} e^{s_{im}}\), positive and summing to 1. The head's output at i is the weighted sum of the value vectors, \(\mathbf{o}_i = \sum_{j \le i} a_{ij}\mathbf{v}_j\): 8 numbers.</p>
<p>At <code>WHERE</code>, layer 1, head 2 scores <code>users</code> at 14.24 and nothing else above −1.05, a gap of 15.29. Since \(e^{-15.29} \approx 2 \times 10^{-7}\), the weight on <code>users</code> is 1.000 and the output is the value vector of <code>users</code>. Head 1 of the same layer puts 0.98 on <code>name</code>.</p>
<p>So layer 1's attention output at <code>WHERE</code> is built from the value of <code>users</code> (head 2) and, with weight 0.98, the value of <code>name</code> (head 1). Chapter 5.3 shows how \(\mathbf{W}_O\) combines the two before the result is added to the stream.</p>`,
    fig: { type: "tinylm", widget: "mix", caption: "Scores, weights and value vectors for one head and position; the output is their weighted sum" },
    terms: [{ term: "Attention weights", def: "The softmax of one position's scores over the positions it may read: positive numbers that sum to 1." }],
  });

  Deck.slide({
    id: "tf-matrix-form",
    title: "One attention layer in six steps, with every shape",
    example: "tokens ids",
    body: String.raw`<p>Written for all positions and both heads at once, one attention layer of our model is the computation in the plate. The projections multiply activations by weights; the scores and the mixing multiply activations by activations.</p>
<p>Per head, \(\mathbf{Q}_h\mathbf{K}_h^\top\) is \((T \times d_h)(d_h \times T) = T \times T\): a score for every pair of positions, 64 dot products of length 8. The mask and the softmax act row by row. \(\mathbf{A}_h\mathbf{V}_h\) is \((T \times T)(T \times d_h) = T \times d_h\): an 8-number output per position. Concatenating the 2 heads gives T × d = 8 × 16, and \(\mathbf{W}_O\) maps it back into the stream. Steps 1, 3, 5 and 6 are matrix products; steps 2 and 4 are not.</p>
<p>Steps 3 and 5 compute all 64 pairs, of which the mask then discards 28; kernels that skip the masked pairs compute 36.</p>`,
    math: [String.raw`\mathrm{head}_h = \mathrm{softmax}\left(\frac{\mathbf{Q}_h\mathbf{K}_h^\top}{\sqrt{d_h}} + \mathbf{M}\right)\mathbf{V}_h`, String.raw`\mathrm{Attn}(\hat{\mathbf{X}}) = [\mathrm{head}_1, \mathrm{head}_2]\,\mathbf{W}_O`],
    fig: {
      type: "table", caption: "One attention layer of our model on one query: T = 8 positions, d = 16, 2 heads of size d<sub>h</sub> = 8", dense: true,
      cols: ["Step", "Computation", "Shapes", "Multiply-adds"], align: ["l", "l", "l", "r"],
      rows: [
        ["1 Project", String.raw`\(\mathbf{Q},\mathbf{K},\mathbf{V} = \hat{\mathbf{X}}\mathbf{W}_{Q,K,V}\)`, String.raw`\((T{\times}d)(d{\times}d) \to T{\times}d\), three times`, "6,144"],
        ["2 Split", String.raw`\(\mathbf{Q}_h\): columns \(8(h-1)\) to \(8h-1\)`, String.raw`\(T{\times}d_h\) per head`, "0"],
        ["3 Score", String.raw`\(\mathbf{Q}_h\mathbf{K}_h^\top/\sqrt 8\)`, String.raw`\((T{\times}d_h)(d_h{\times}T) \to T{\times}T\) per head`, "1,024"],
        ["4 Mask, softmax", String.raw`\(\mathbf{A}_h = \mathrm{softmax}(\mathbf{S}_h + \mathbf{M})\)`, String.raw`\(T{\times}T\) per head, row by row`, "elementwise"],
        ["5 Mix", String.raw`\(\mathbf{O}_h = \mathbf{A}_h\mathbf{V}_h\)`, String.raw`\((T{\times}T)(T{\times}d_h) \to T{\times}d_h\) per head`, "1,024"],
        ["6 Project", String.raw`\([\mathbf{O}_1, \mathbf{O}_2]\,\mathbf{W}_O\)`, String.raw`\((T{\times}d)(d{\times}d) \to T{\times}d\)`, "2,048"],
        ["", "", "Total", "<b>10,240</b>"],
      ],
      note: "A multiply-add is 2 FLOPs. Steps 1 and 6 grow with T·d², steps 3 and 5 with T²·d.",
    },
  });

  Deck.slide({
    id: "tf-att-matrix",
    title: "The full attention matrix of layer 1, head 1",
    example: "tokens ids focus:4",
    body: String.raw`<p>The plate is the weight matrix \(\mathbf{A}_1\) of layer 1 for our query: row i is position i reading, column j is position j being read. Each row sums to 1, and above the diagonal the mask forces 0.</p>
<p>Row 4, <code>WHERE</code>, puts 0.98 on <code>name</code>, the column the filter may not repeat. Rows 2 and 3 split between <code>SELECT</code> (0.60 and 0.61) and <code>FROM</code> (0.39 and 0.35). The token after the digit is always <code>;</code>, so row 7 needs no particular earlier token; it spreads its weight between 0.08 and 0.16 over all 8 positions.</p>
<p>Every query computes such a matrix for each of the 4 heads. Chapter 5.3 compares them.</p>`,
    fig: {
      type: "heat", caption: "Attention weights of layer 1, head 1 on our query (rows read, columns are read)", signed: false, dp: 2, max: 1.6,
      rows: TOKS.slice(0, 8).map((t, i) => `${i}: <code>${short(t)}</code>`),
      cols: TOKS.slice(0, 8).map((t, j) => `${j}`),
      matrix: TR.layers[0].heads[0].att.map((row, i) => row.map((v, j) => (j > i ? null : v))),
      note: "– marks a masked entry (weight 0). Hover a cell for 3 decimals.",
    },
  });

  Deck.slide({
    id: "tf-att-cost",
    title: "Attention's cost grows with the square of the context",
    example: "tokens ids",
    body: String.raw`<p>Per layer, the scores take T·T·d multiply-adds summed over heads: each of the T² pairs is a dot product of length \(d_h\), for H heads, and \(H d_h = d\). The mixing takes the same. The weight matrices take T times the layer's count of matrix weights \(P\) (3,072 for ours; its 32 norm gains act elementwise). The ratio of attention's T² work to the weight work is therefore \(2Td/P\), which grows linearly with T.</p>
<p>Our layer at T = 8: 2 × 8 × 16 / 3,072 = 0.083, so the T² terms add 8% to the weight work. A Llama 3 8B layer has P = 218,103,808 and d = 4,096: the ratio is 0.31 at T = 8,192 and 4.92 at T = 131,072. At long context, attention is most of the compute, and each head's 8 × 8 matrix has become 131,072 × 131,072.</p>`,
    fig: {
      type: "calc", caption: "FLOPs of one layer for one sequence of T tokens: the T² attention terms against the weight matrices",
      inputs: [
        { id: "m", label: "Layer", options: [{ label: "ours, d = 16", value: 0 }, { label: "Llama 3 8B, d = 4,096", value: 1 }], value: 1 },
        { id: "T", label: "Context T (tokens)", options: [8, 1024, 8192, 32768, 131072].map((t) => ({ label: t.toLocaleString("en-US"), value: t })), value: 131072 },
      ],
      outputs: [
        { label: String.raw`attention T² terms, \(4T^2d\)`, f: (v) => 4 * v.T * v.T * [16, 4096][v.m], fmt: "sci" },
        { label: String.raw`weight matrices, \(2TP\)`, f: (v) => 2 * v.T * [3072, 218103808][v.m], fmt: "sci" },
        { label: String.raw`ratio \(2Td/P\)`, f: (v) => (2 * v.T * [16, 4096][v.m]) / [3072, 218103808][v.m], fmt: "f3" },
      ],
      note: (v) => `P = ${[3072, 218103808][v.m].toLocaleString("en-US")} matrix weights per layer. With the causal mask only T(T + 1)/2 of the T² pairs are needed; kernels that skip the rest halve the first number.`,
    },
    src: ["Dubey et al. (Llama Team, Meta), The Llama 3 Herd of Models, arXiv:2407.21783, 2024, Table 3 (8B shapes)"],
  });

  Deck.slide({
    id: "tf-attn-check",
    title: "Check: the shapes and scale of attention",
    body: String.raw`<p>One attention layer, in order: normalize each position, project to Q, K and V, split into heads, score every pair, add the mask, take the softmax of each row, mix the values, concatenate the heads, project with \(\mathbf{W}_O\), add the result to the stream.</p>
<p>In our model every intermediate is 8 × 16, or 8 × 8 per head. In a Llama 3 8B layer at T = 8,192 the same steps produce Q of 8,192 × 4,096, K and V of 8,192 × 1,024 (8 key/value heads, chapter 5.7) and, per head, 8,192 × 8,192 scores. The question uses those sizes.</p>`,
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024, Table 3"],
    check: {
      q: String.raw`A Llama 3 8B layer has 32 query heads of size \(d_h = 128\) and processes T = 8,192 tokens. How many entries does one head's matrix of attention weights have in that layer, and what is the standard deviation of an unscaled q · k for random vectors with unit-variance entries?`,
      options: ["8,192 entries; 128", "67,108,864 entries; about 11.3", "1,048,576 entries; about 11.3", "67,108,864 entries; 128"],
      answer: 1,
      why: "The weights form a T × T matrix per head: 8,192² = 67,108,864 entries, about half of them set to 0 by the mask; a kernel that skips the masked entries computes T(T + 1)/2 = 33,558,528. A sum of 128 unit-variance products has variance 128, so its standard deviation is √128 = 11.3; dividing by √128 brings it back to 1. 1,048,576 is 8,192 × 128, the size of one head's Q.",
    },
  });

  // ================================================================== 5.3
  Deck.chapter({ title: "Multi-head attention" });

  Deck.slide({
    id: "tf-heads-split",
    title: "Heads split the vector into independent attention computations",
    example: "tokens ids focus:4",
    body: String.raw`<p><b>Multi-head attention</b> runs H attention computations side by side on slices of the same projections. With d = 16 and H = 2, head 1 uses columns 0–7 of Q, K and V and head 2 uses columns 8–15. Each head has its own scores and its own softmax, so each can put its weight somewhere different.</p>
<p>That matters at <code>WHERE</code>, which needs two earlier tokens. One softmax row either concentrates on one position or splits its weight, and an even split would deliver the average of two value vectors. With two heads, layer 1 reads both: head 1 puts 0.98 on <code>name</code> and head 2 puts 1.00 on <code>users</code>.</p>
<p>Splitting costs nothing extra. H heads of size d/H compute T²·(d/H) multiply-adds each, T²·d in total, the same as one head of size d. Llama 3 8B splits d = 4,096 into 32 heads of 128.</p>`,
    fig: {
      type: "diagram", caption: "One attention layer of our model as two heads", w: 14.8, h: 9.3, unit: 40,
      aria: "Q, K and V of size 8 by 16 split into two heads of 8 columns; each head computes its own attention; the outputs are concatenated and multiplied by W_O",
      nodes: [
        { id: "xh", x: 0.1, y: 4.1, w: 1.7, h: 1.3, label: "x̂", sub: "8 × 16" },
        { id: "qkv", x: 2.5, y: 4.1, w: 2.7, h: 1.3, label: "Q, K, V", sub: "8 × 16 each" },
        { id: "h1", x: 5.9, y: 1.1, w: 5.4, h: 1.7, label: "head 1: columns 0–7", sub: "softmax(Q₁K₁ᵀ/√8 + M) V₁", kind: "s1" },
        { id: "h2", x: 5.9, y: 6.6, w: 5.4, h: 1.7, label: "head 2: columns 8–15", sub: "softmax(Q₂K₂ᵀ/√8 + M) V₂", kind: "s2" },
        { id: "cat", x: 12.0, y: 4.1, w: 2.7, h: 1.3, label: "concat · W_O", sub: "→ 8 × 16" },
      ],
      edges: [{ from: "xh", to: "qkv" }, { from: "qkv", to: "h1" }, { from: "qkv", to: "h2" }, { from: "h1", to: "cat", label: "8 × 8" }, { from: "h2", to: "cat", label: "8 × 8" }],
      notes: [{ x: 6.0, y: 0.85, text: "WHERE → name: 0.98" }, { x: 6.0, y: 9.05, text: "WHERE → users: 1.00" }],
    },
    terms: [{ term: "Multi-head attention", def: "Several attention computations (heads) run in parallel on slices of Q, K and V; their outputs are concatenated and projected by W_O." }],
    src: ["Vaswani et al., Attention Is All You Need, 2017, section 3.2.2", "Dubey et al., The Llama 3 Herd of Models, 2024, Table 3"],
  });

  Deck.slide({
    id: "tf-heads",
    title: "The attention patterns of our four heads",
    example: "tokens ids focus:4",
    body: String.raw`<p>The plate shows all four 8 × 8 weight matrices for a query you choose; rows are positions reading, columns the positions read.</p>
<p>In layer 1, head 2, four positions read the token just before them: <code>name</code> reads <code>SELECT</code> (1.00), <code>users</code> reads <code>FROM</code> (0.98), <code>WHERE</code> reads <code>users</code> (1.00), <code>id</code> reads <code>WHERE</code> (0.97). Layer 1, head 1 has <code>WHERE</code> read <code>name</code> (0.98). Layer 2, head 2 has <code>FROM</code> read <code>name</code> (0.95) and <code>id</code> read <code>name</code> (0.88).</p>
<p>Switch the plate to the <code>orders</code> query, <code>SELECT total FROM orders</code> <code>WHERE status = 3;</code>: <code>WHERE</code> reads <code>total</code> (layer 1, head 1, 0.99) and <code>orders</code> (layer 1, head 2, 1.00), and <code>FROM</code> reads <code>total</code> (layer 2, head 2, 0.95). Nobody assigned these roles; gradient descent found them. The last sheet of this chapter shows where they change.</p>`,
    fig: { type: "tinylm", widget: "heads", caption: "Attention weights of every head of the trained model; select a cell to read it" },
  });

  Deck.slide({
    id: "tf-concat-wo",
    title: String.raw`The heads are concatenated, then projected by \(\mathbf{W}_O\)`,
    layout: "wide",
    example: "tokens ids focus:4",
    body: String.raw`<p>The two 8-number head outputs are placed side by side, 16 numbers, and multiplied by the output matrix \(\mathbf{W}_O\) (16 × 16). Rows 0–7 of \(\mathbf{W}_O\) only ever multiply head 1's output and rows 8–15 only head 2's, so the product is a sum of one term per head: \([\mathbf{o}_1, \mathbf{o}_2]\,\mathbf{W}_O = \mathbf{o}_1\mathbf{W}_O^{(1)} + \mathbf{o}_2\mathbf{W}_O^{(2)}\).</p>
<p>The plate shows both terms at <code>WHERE</code> in layer 1 and their sum, the attention output that is added to the stream. Head 1's term has length 10.86, head 2's 5.93, their sum 13.81. Every output entry mixes both heads: entry 0 is 6.45 + 0.57 = 7.02.</p>`,
    fig: {
      type: "heat", caption: "<span>Layer 1 at <code>WHERE</code>: each head's contribution through \\(\\mathbf{W}_O\\), and the attention output</span>", dp: 2,
      rows: [String.raw`head 1: \(\mathbf{o}_1\mathbf{W}_O^{(1)}\)`, String.raw`head 2: \(\mathbf{o}_2\mathbf{W}_O^{(2)}\)`, "attention output"],
      cols: Array.from({ length: 16 }, (_, c) => String(c)),
      matrix: [0, 1].map((hh) => {
        const out = new Array(16).fill(0), Wo = WTS["l0.wo"];
        for (let rr = 0; rr < 8; rr++) for (let c = 0; c < 16; c++) out[c] += TR.layers[0].concat[4][8 * hh + rr] * Wo.data[(8 * hh + rr) * 16 + c];
        return out;
      }).concat([TR.layers[0].attnOut[4]]), max: 14,
      note: "Row 3 is the sum of rows 1 and 2, entry by entry.",
    },
  });

  Deck.slide({
    id: "tf-attn-params",
    title: "An attention layer has 4d² parameters, whatever the head count",
    example: "tokens ids",
    body: String.raw`<p>\(\mathbf{W}_Q\), \(\mathbf{W}_K\), \(\mathbf{W}_V\) and \(\mathbf{W}_O\) are each d × d: 4 × 16 × 16 = 1,024 parameters per layer, 2,048 for our two layers. The head count does not appear in that product. Heads slice these matrices into column groups; they do not add to them. With 1, 2, 4, 8 or 16 heads our layer has the same 1,024 parameters and only the head size \(d_h = d/H\) changes.</p>
<p>For d = 4,096 the formula gives 4 × 4,096² = 67,108,864. Llama 3 8B has fewer attention parameters per layer, 41,943,040, because its keys and values use 8 heads instead of 32 (chapter 5.7).</p>`,
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024, Table 3"],
    fig: {
      type: "calc", caption: "Attention parameters per layer for a width d and a head count H",
      inputs: [
        { id: "d", label: "Width d", options: [{ label: "16 (ours)", value: 16 }, { label: "4,096", value: 4096 }], value: 16 },
        { id: "H", label: "Heads H", options: [1, 2, 4, 8, 16, 32].map((x) => ({ label: String(x), value: x })), value: 2 },
      ],
      outputs: [
        { label: String.raw`head size \(d_h = d/H\)`, f: (v) => (v.d % v.H ? NaN : v.d / v.H), fmt: "int" },
        { label: String.raw`one head's slice of \(\mathbf{W}_Q\)`, f: (v) => (v.d % v.H ? NaN : (v.d * v.d) / v.H), fmt: "int" },
        { label: String.raw`\(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O\) together`, f: (v) => (v.d % v.H ? NaN : 4 * v.d * v.d), fmt: "int" },
      ],
      note: (v) => (v.d % v.H ? `d = ${v.d} cannot be split into ${v.H} heads: the head size would be ${v.d / v.H}.` : `4 × ${v.d.toLocaleString("en-US")}² = ${(4 * v.d * v.d).toLocaleString("en-US")} for any H that divides d.`),
    },
  });

  Deck.slide({
    id: "tf-head-roles",
    title: "Which head reads what depends on the column",
    example: "tokens ids focus:4",
    body: String.raw`<p>Averaged over all 270 queries, the patterns of the last sheets hold for most columns and change for a few. The plate shows six attention weights for each selected column.</p>
<p>At <code>WHERE</code>, layer 1, head 1 reads the selected column with weight 0.98 or more, except for <code>title</code> (0.25) and <code>price</code> (0.03); for those it reads the table instead (0.74 and 0.95). Head 2 reads the table with 0.999 or more, except for <code>price</code>, where it reads the column (1.00). For <code>price</code> the two heads swap jobs.</p>
<p>At <code>FROM</code>, where the table must be predicted from the column, layer 1, head 2 reads the column for <code>id</code>, <code>title</code>, <code>price</code>, <code>user_id</code>, <code>total</code> and <code>status</code> (0.995 or more). Layer 2, head 2 reads it for the six <code>users</code> and <code>orders</code> columns other than <code>id</code> (0.91 to 0.95). Every column is read at <code>FROM</code> by at least one head.</p>`,
    fig: {
      type: "heat", caption: "Mean attention weight over the queries with each selected column", signed: false, dp: 2, max: 1.6,
      rows: ROLE_COLS.map((c) => `<code>${c}</code>`),
      cols: ["L1 H1<br>WHERE→col", "L1 H1<br>WHERE→table", "L1 H2<br>WHERE→col", "L1 H2<br>WHERE→table", "L1 H2<br>FROM→col", "L2 H2<br>FROM→col"],
      get matrix() { return ROLE_MEAN(); },
      note: "col = position 1, the selected column; table = position 3. Each mean weights queries by their probability.",
    },
    check: {
      q: "For queries starting <code>SELECT price FROM products WHERE</code>, which layer-1 head reads the table from position 4?",
      options: ["Head 2, as for every other column", "Head 1, with weight 0.95", "Neither head", "Both heads, about 0.5 each"],
      answer: 1,
      why: "For price, head 2 puts 1.00 on the column instead, and head 1 reads the table with 0.95. A head's job can depend on the tokens, so naming one role per head is an approximation.",
    },
  });

  // ================================================================== 5.4
  Deck.chapter({ title: "The transformer block" });

  Deck.slide({
    id: "tf-residual",
    title: "The residual stream: each sublayer adds to x",
    example: "tokens ids focus:4",
    body: String.raw`<p>Each position carries one 16-number vector from the embeddings to the output. Every sublayer reads it, computes something, and adds the result back. This running sum is the <b>residual stream</b>.</p>
<p>Nothing overwrites the stream, so what the embeddings put there is still part of the sum at the top, and every later sublayer can read what earlier ones wrote. The gradient has the same direct path back to every sublayer (Part 3 measures this).</p>
<p>At <code>WHERE</code> the length of the stream grows from 5.1 to 15.8 after layer 1's attention, 22.8 after its MLP and 62.0 after layer 2. Our model stacks 2 blocks; Llama 3 70B stacks 80.</p>`,
    math: [String.raw`\mathbf{x} \leftarrow \mathbf{x} + \mathrm{Attn}(\mathrm{RMSNorm}(\mathbf{x}))`, String.raw`\mathbf{x} \leftarrow \mathbf{x} + \mathrm{MLP}(\mathrm{RMSNorm}(\mathbf{x}))`],
    fig: { type: "tinylm", widget: "block", caption: "The stream at one position through both blocks, and what each sublayer adds" },
    terms: [{ term: "Residual stream", def: "The vector each position carries through the model. Every attention and MLP sublayer adds its output to it instead of replacing it." }],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024, Table 3 (80 layers in 70B)"],
  });

  Deck.slide({
    id: "tf-rmsnorm",
    title: "Pre-norm: each sublayer reads a normalized copy of the stream",
    example: "tokens ids focus:4",
    body: String.raw`<p>The stream grows as sublayers add to it, and unevenly: entering layer 2 its length is 22.84 at <code>WHERE</code> and 591.03 at <code>SELECT</code>, a 26-fold spread. Each sublayer therefore reads RMSNorm of the stream (Part 3 defines it), and the stream itself stays unnormalized. This placement is <b>pre-norm</b>, used by GPT-2 and LLaMA. The original transformer normalized after each addition (post-norm); Xiong et al. (2020) show that post-norm needs a learning-rate warmup to train stably and pre-norm does not, and LLaMA gives training stability as its reason.</p>
<p>Worked at <code>WHERE</code> entering layer 2: the mean of the 16 squares is 32.60, so RMS = √(32.60 + 10⁻⁵) = 5.710. Dividing by it gives a vector with RMS 1, whose entries 0, 6 and 7 are 1.654, −2.829 and 0.692. The learned gains 0.760, 0.821 and 0.893 turn them into 1.257, −2.323 and 0.618, which is what attention reads.</p>`,
    math: [String.raw`\mathrm{RMSNorm}(\mathbf{x}) = \mathbf{g} \odot \frac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})}`, { tex: String.raw`\mathrm{RMS}(\mathbf{x}) = \sqrt{\tfrac{1}{d}\textstyle\sum_{c} x_c^2 + \epsilon}`, note: "ε = 10⁻⁵ in our model and in Llama 3." }],
    fig: {
      type: "sequence", caption: "Vector lengths entering layer 2, before and after RMSNorm",
      tokens: TOKS.slice(0, 8), focus: [4],
      rows: [
        { label: "stream", values: TR.layers[0].post.map(norm2), dp: 2 },
        { label: "x / RMS", values: TR.layers[0].post.map((v) => norm2(v.map((z) => z / Math.sqrt(v.reduce((a, b) => a + b * b, 0) / 16 + 1e-5)))), dp: 3 },
        { label: "× gains", values: TR.layers[1].h1.map(norm2), dp: 2 },
      ],
      note: "A vector of 16 entries with RMS 1 has length √16 = 4. The gains then shrink or stretch each entry; gain 4 is 0.000, so entry 4 is switched off in this layer.",
    },
    terms: [{ term: "Pre-norm", def: "Placing the normalization at the input of each sublayer, x + f(norm(x)), so the residual stream itself is never normalized." }],
    src: ["Zhang and Sennrich, Root Mean Square Layer Normalization, NeurIPS 2019", "Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2), 2019, section 2.3", "Touvron et al., LLaMA, arXiv:2302.13971, 2023, section 2.2", "Vaswani et al., 2017, section 3.1", "Xiong et al., On Layer Normalization in the Transformer Architecture, arXiv:2002.04745, 2020"],
  });

  Deck.slide({
    id: "tf-mlp",
    title: "The MLP transforms each position on its own: 16 → 64 → 16",
    example: "tokens ids focus:4",
    body: String.raw`<p>The second sublayer is a two-layer network applied to each position separately (Part 3 builds it): \(\mathrm{MLP}(\hat{\mathbf{x}}) = \mathrm{GELU}(\hat{\mathbf{x}}\mathbf{W}_1)\,\mathbf{W}_2\), with \(\mathbf{W}_1\) of 16 × 64 and \(\mathbf{W}_2\) of 64 × 16. It cannot move information between positions; attention did that. It holds 2,048 of each layer's 3,104 parameters, 66%.</p>
<p>The plate plots GELU and the 64 hidden pre-activations of the position you pick. At <code>WHERE</code> in layer 1 they range from −6.45 to 3.53: 5 of 64 are positive and only 3 produce more than 0.1. At <code>id</code> in the same layer 39 are positive. GELU passes large positive inputs almost unchanged and sends large negative ones to almost 0, so each position uses its own subset of the 64 units.</p>`,
    math: [String.raw`\mathrm{GELU}(z) \approx \tfrac{z}{2}\,\bigl(1 + \tanh u\bigr)`, String.raw`u = \sqrt{2/\pi}\,\bigl(z + 0.044715\,z^3\bigr)`],
    fig: { type: "tf-mlp", caption: "GELU and the 64 hidden units of one position (red: output above 0.1)", pos: 4, layer: 0 },
    src: ["Hendrycks and Gimpel, Gaussian Error Linear Units (GELUs), arXiv:1606.08415 (tanh form used by our model)"],
  });

  Deck.slide({
    id: "tf-block-diagram",
    title: "One block, with the shape of every tensor",
    layout: "wide",
    example: "tokens ids",
    body: String.raw`<p>One block of our model for one query of 8 tokens. The upper row is the attention sublayer, read left to right; the lower row is the MLP sublayer, read right to left from where the attention sum comes down. The horizontal line under each row is the residual stream, which the sublayer's output is added into.</p>
<p>Parameters in a block: two RMSNorm gains of 16, four 16 × 16 attention matrices (1,024) and the MLP's 16 × 64 and 64 × 16 (2,048), 3,104 in all. Everything else in the drawing is an activation, computed anew for every query.</p>`,
    fig: {
      type: "diagram", caption: "One transformer block of our model (T = 8, d = 16, 2 heads of 8, MLP width 64); the lower row runs right to left", w: 27.8, h: 8.8, unit: 40,
      aria: "Attention sublayer, left to right: RMSNorm, Q K V projections, scores with mask, softmax, mixing, concat and W_O, added to the stream. The sum flows down into the MLP sublayer, right to left: RMSNorm, W1, GELU, W2, added to the stream.",
      nodes: [
        { id: "x", x: 0.2, y: 2.45, w: 1.6, h: 1.1, label: "x", sub: "8 × 16" },
        { id: "n1", x: 2.4, y: 0.3, w: 3.2, h: 1.3, label: "RMSNorm", sub: "gain g₁: 16" },
        { id: "qkv", x: 6.4, y: 0.3, w: 3.4, h: 1.3, label: "× W_Q, W_K, W_V", sub: "3 × (16×16) → 3 × 8×16" },
        { id: "sc", x: 10.6, y: 0.3, w: 3.2, h: 1.3, label: "QₕKₕᵀ/√8 + M", sub: "2 heads × 8×8" },
        { id: "smx", x: 14.6, y: 0.3, w: 2.8, h: 1.3, label: "softmax", sub: "2 × 8×8, rows" },
        { id: "av", x: 18.2, y: 0.3, w: 2.8, h: 1.3, label: "× Vₕ", sub: "2 × 8×8" },
        { id: "wo", x: 21.8, y: 0.3, w: 3.4, h: 1.3, label: "concat, × W_O", sub: "8×16 · 16×16" },
        { id: "a1", x: 25.9, y: 2.45, w: 1.1, h: 1.1, label: "+", round: true },
        { id: "xm", x: 25.4, y: 6.45, w: 2.1, h: 1.1, label: "x + attn", sub: "8 × 16" },
        { id: "n2", x: 21.8, y: 4.4, w: 3.2, h: 1.3, label: "RMSNorm", sub: "gain g₂: 16" },
        { id: "w1", x: 16.8, y: 4.4, w: 3.6, h: 1.3, label: "× W₁", sub: "16×64 → 8×64" },
        { id: "ge", x: 12.2, y: 4.4, w: 3.2, h: 1.3, label: "GELU", sub: "8×64, elementwise" },
        { id: "w2", x: 7.2, y: 4.4, w: 3.6, h: 1.3, label: "× W₂", sub: "64×16 → 8×16" },
        { id: "a2", x: 0.6, y: 6.45, w: 1.1, h: 1.1, label: "+", round: true },
      ],
      edges: [
        { from: "x", to: "a1" }, { from: "x", to: "n1" }, { from: "n1", to: "qkv" }, { from: "qkv", to: "sc" }, { from: "sc", to: "smx" }, { from: "smx", to: "av" }, { from: "av", to: "wo" }, { from: "wo", to: "a1" },
        { from: "a1", to: "xm" },
        { from: "xm", to: "a2" }, { from: "xm", to: "n2" }, { from: "n2", to: "w1" }, { from: "w1", to: "ge" }, { from: "ge", to: "w2" }, { from: "w2", to: "a2" },
      ],
      notes: [
        { x: 3.0, y: 3.45, text: "the stream, 8 × 16" }, { x: 2.2, y: 8.45, text: "the stream after attention; after this + it enters block 2" },
      ],
    },
  });

  Deck.slide({
    id: "tf-stack",
    title: "Two stacked blocks: what each sublayer contributes",
    example: "tokens ids",
    body: String.raw`<p>Stacking repeats the block with new weights; layer 2 reads a stream that layer 1 has already written into. One way to see what each sublayer does is to remove it from the trained model and measure the loss. The plate does that with the exact loss over all 270 queries.</p>
<p>Removing layer 2 entirely, which reads the prediction off the stream after layer 1, raises the loss from 0.6942 to 3.3274. Layer 1 alone already predicts <code>FROM</code>, <code>WHERE</code> and <code>=</code> on our query (p = 1.000, 1.000 and 0.990) but gives <code>users</code> after <code>FROM</code> only 0.051: choosing the table needs layer 2. Removing only layer 2's attention costs much less (0.7384), almost all of it at <code>WHERE</code>.</p>
<p>These numbers show how much the trained model depends on each sublayer. They do not isolate what a sublayer computes: later sublayers were trained on the outputs of earlier ones, so removing one also changes the inputs they expect.</p>`,
    fig: { type: "tf-ablate", caption: "Switch sublayers off in the trained model; the loss is exact over the grammar" },
  });

  Deck.slide({
    id: "tf-logits",
    title: "The final norm and the unembedding give 113 scores",
    example: "tokens ids focus:4 target:5",
    body: String.raw`<p>After the last block comes one more RMSNorm, then the <b>unembedding</b> matrix \(\mathbf{W}_U\) (16 × 113), which turns each position's vector into one score per vocabulary token: 113 logits. Softmax turns them into the next-token distribution (Part 2).</p>
<p>At <code>WHERE</code> the top logits are 7.621 (<code>id</code>), 7.617 (<code>email</code>), 7.502 (<code>age</code>) and 1.994 (<code>name</code>). The three allowed columns lie within 0.12 of each other, so softmax gives them 0.346, 0.345 and 0.307; <code>name</code>, 5.63 below the top, gets 0.001. The grammar's answer is 1/3 each.</p>
<p>The plate's temperature, top-k and top-p controls reshape these probabilities before sampling; Part 4 covers them.</p>`,
    fig: { type: "tinylm", widget: "logits", caption: "The logits at <code>WHERE</code> and the probabilities after decoding settings" },
    terms: [{ term: "Unembedding", def: "The final d × V matrix that maps each position's vector to one logit per vocabulary token." }],
  });

  Deck.slide({
    id: "tf-params",
    title: "9,968 parameters, counted term by term",
    example: "tokens ids",
    body: String.raw`<p>Every parameter of the model sits in one of the tables in the plate. Per layer: 4d² = 1,024 in attention, 8d² = 2,048 in the MLP (d × 4d plus 4d × d) and 2d = 32 in the two RMSNorm gains, 3,104 in all; two layers make 6,208. Around them: V·d = 1,808 in the token embedding, T·d = 128 in the position embedding, d = 16 in the final gain and d·V = 1,808 in the unembedding.</p>
<p>With L = 2, d = 16, V = 113 and T = 8 the formula below gives 9,968. The two vocabulary tables hold 36% of our parameters. In Llama 3 8B they hold 13% and the 32 blocks the other 87%.</p>`,
    math: String.raw`\begin{aligned} N &= 12Ld^2 + 2Vd \\ &\quad + Td + (2L + 1)\,d \\ &= 6{,}144 + 3{,}616 + 128 + 80 \\ &= 9{,}968 \end{aligned}`,
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024, Table 3"],
    fig: {
      type: "table", caption: "Every weight matrix and gain vector of our model", dense: true,
      cols: ["Part", "Shape", "Parameters"], align: ["l", "l", "r"],
      rows: [
        [String.raw`Token embedding \(\mathbf{W}_E\)`, "113 × 16", "1,808"],
        [String.raw`Position embedding \(\mathbf{W}_P\)`, "8 × 16", "128"],
        ["Layer 1: RMSNorm gains", "2 × 16", "32"],
        [String.raw`Layer 1: \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O\)`, "4 × (16 × 16)", "1,024"],
        [String.raw`Layer 1: MLP \(\mathbf{W}_1, \mathbf{W}_2\)`, "16 × 64 + 64 × 16", "2,048"],
        ["Layer 2: the same", "", "3,104"],
        ["Final RMSNorm gain", "16", "16"],
        [String.raw`Unembedding \(\mathbf{W}_U\)`, "16 × 113", "1,808"],
        ["", "Total", "<b>9,968</b>"],
      ],
      highlight: [8],
    },
  });

  Deck.slide({
    id: "tf-flops",
    title: "A forward pass costs about 2 FLOPs per parameter per token",
    example: "tokens ids",
    body: String.raw`<p>Each weight in a matrix product is used once per token, for one multiply and one add: 2 FLOPs. Embedding rows are looked up, not multiplied. So per token our model spends 2 × (2 × 3,072 + 1,808) = 15,904 FLOPs on weights: two layers of attention and MLP matrices, and the unembedding.</p>
<p>Attention adds work that involves no weights: position t computes t + 1 scores and mixes t + 1 values, 4 × 16 × (t + 1) FLOPs per layer. Over our 8 positions that is 2,304 per layer, 4,608 for both. The whole forward pass of our query costs 8 × 15,904 + 4,608 = 131,840 FLOPs. Elementwise work (norms, GELU, softmax, additions) is not counted.</p>
<p>Kaplan et al. (2020) use the same rule for large models: about 2N FLOPs per token plus a context term. Their N leaves out both vocabulary matrices, so their count also drops the unembedding's 2·d·V FLOPs. Part 7 multiplies it by 3 for training.</p>`,
    fig: {
      type: "bars", caption: "FLOPs of one forward pass over our 8 positions, both layers, by component", fmt: "int",
      items: [
        { label: "Q, K, V, O", value: 32768, text: "32,768" },
        { label: "MLP", value: 65536, text: "65,536", hi: true },
        { label: "Unembedding", value: 28928, text: "28,928" },
        { label: "Scores, mixing", value: 4608, text: "4,608" },
      ],
      note: "Total 131,840. The MLP's 65,536 FLOPs are twice the 32,768 of the Q, K, V, O projections, as it has twice the weights per layer (2,048 against 1,024).",
    },
    check: {
      q: "Llama 3 8B has 8,030,261,248 parameters, 525,336,576 of them in the token embedding. About how many FLOPs per token does its forward pass spend on weights?",
      options: [String.raw`\(8.0 \times 10^{9}\)`, String.raw`\(1.5 \times 10^{10}\)`, String.raw`\(1.6 \times 10^{10}\)`, String.raw`\(4.8 \times 10^{10}\)`],
      answer: 1,
      why: String.raw`Every matrix weight costs 2 FLOPs per token and the embedding rows are lookups: 2 × (8,030,261,248 − 525,336,576) ≈ \(1.50 \times 10^{10}\). Counting the embedding gives \(1.6 \times 10^{10}\); \(8.0 \times 10^{9}\) forgets the add; \(4.8 \times 10^{10}\) is the 6N of a training step. Kaplan's 2N, which also leaves out the unembedding, gives \(1.40 \times 10^{10}\).`,
    },
    src: ["Kaplan et al., Scaling Laws for Neural Language Models, arXiv:2001.08361, 2020, section 2.1 and Table 1", "Dubey et al., The Llama 3 Herd of Models, 2024, Table 3, and Meta, llama-models sku_list.py (the shapes the 8B count is computed from)"],
  });

  // ================================================================== 5.5
  Deck.chapter({ title: "Positions" });

  Deck.slide({
    id: "tf-perm",
    title: "Without positions, attention ignores order",
    example: "tokens ids",
    body: String.raw`<p>Attention's output at position i is \(\sum_j a_{ij}\mathbf{v}_j\), a weighted sum over a set of positions. If the vectors carried no position information, shuffling the earlier tokens would shuffle their scores, weights and values together, and the sum would come out the same.</p>
<p>The plate tests this on layer 1 of our trained model at the last position, which reads all 8 tokens. With the position embeddings left out, every shuffle of positions 0 to 6 changes the output by less than 10⁻¹⁵, which is floating-point rounding. With them added, the plate's first shuffle changes an output entry by 1.94.</p>
<p>The causal mask leaks some order, since position i sees exactly i + 1 tokens. Haviv et al. (2022) found that causal language models trained without any position encoding still learn absolute positions, and conjecture that they count the tokens they can see. Every model in this set adds position information explicitly.</p>`,
    fig: { type: "tf-perm", caption: "Shuffle the first 7 tokens and compare layer 1's attention output at position 7", seed: 3 },
    src: ["Haviv et al., Transformer Language Models without Positional Encodings Still Learn Positional Information, Findings of EMNLP 2022"],
  });

  Deck.slide({
    id: "tf-learned-pos",
    title: "Learned absolute positions: one trained vector per position",
    example: "tokens ids",
    body: String.raw`<p>Our model learns its position information the way it learns token information: a table \(\mathbf{W}_P\) of 8 rows of 16 numbers, trained by gradient descent, whose row t is added to the token at position t. GPT-1 used learned positions for 512-token sequences, and GPT-2 extended the context to 1,024.</p>
<p>The table fixes the context length. There is no row 8, so our model cannot accept a ninth token at all; going longer means adding rows and training them on longer sequences.</p>
<p>The plate shows the cosine similarity between every pair of rows. Nothing forces neighboring positions to be similar, and they are not: rows 4 and 5 have cosine −0.34 while rows 0 and 4 have 0.35. Each row only has to be useful to the layers that read it.</p>`,
    fig: {
      type: "heat", caption: "Cosine similarity between the 8 learned position vectors of our model", dp: 2, max: 1.6,
      rows: TOKS.slice(0, 8).map((t, i) => `row ${i}`), cols: TOKS.slice(0, 8).map((t, i) => String(i)),
      matrix: (() => { const P = Array.from({ length: 8 }, (_, t) => M.row(WTS.wpe, t)); return P.map((a) => P.map((b) => a.reduce((z, v, c) => z + v * b[c], 0) / (norm2(a) * norm2(b)))); })(),
      note: "The diagonal is 1. Red: similar directions; blue: opposite.",
    },
    src: ["Radford et al., Improving Language Understanding by Generative Pre-Training (GPT-1), 2018, section 4.1", "Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2), 2019, section 2.3"],
  });

  Deck.slide({
    id: "tf-sinusoid",
    title: "Sinusoidal positions: fixed waves of many wavelengths",
    example: "tokens ids",
    body: String.raw`<p>The original transformer used no position table. It added a fixed vector of sines and cosines: entries 2i and 2i + 1 at position t are \(\sin(t\,\omega_i)\) and \(\cos(t\,\omega_i)\), with \(\omega_i = 10000^{-2i/d}\). For d = 16 the wavelengths \(2\pi/\omega_i\) run from 6.28 positions for pair 0 to 19,869 positions for pair 7.</p>
<p>Fast pairs separate neighbors; slow pairs separate distant positions. Moving k positions turns every pair by the angle \(k\,\omega_i\), whatever t is, so an offset is the same linear map at every position. Vaswani et al. found learned and sinusoidal positions gave nearly identical results; the sinusoids need no table and are defined for every position.</p>`,
    math: String.raw`\begin{aligned} PE_{t,\,2i} &= \sin(t\,\omega_i) \\ PE_{t,\,2i+1} &= \cos(t\,\omega_i) \\ \omega_i &= 10000^{-2i/d} \end{aligned}`,
    fig: {
      type: "plot", caption: "One sine and cosine pair over 64 positions (dots: our 8 positions)", x: [0, 64], y: [-1.25, 1.75], yticks: [-1, -0.5, 0, 0.5, 1], xlabel: "position t", ylabel: "value",
      params: [{ name: "i", label: "pair i", min: 0, max: 7, step: 1, value: 1, fmt: "f0" }],
      fns: [
        { f: (t, p) => Math.sin(t * Math.pow(10000, (-2 * p.i) / 16)), cls: "s1", samples: 480 },
        { f: (t, p) => Math.cos(t * Math.pow(10000, (-2 * p.i) / 16)), cls: "s2", samples: 480 },
      ],
      extra: (p) => ({
        dots: Array.from({ length: 8 }, (_, t) => ({ x: t, y: Math.sin(t * Math.pow(10000, (-2 * p.i) / 16)), r: 3.5, cls: "s1", tip: `t = ${t}: sin = ${Math.sin(t * Math.pow(10000, (-2 * p.i) / 16)).toFixed(3)}` })),
        after(svg, X, Y) {
          [["s1", "sin(t·ωᵢ), entry 2i", 4], ["s2", "cos(t·ωᵢ), entry 2i + 1", 34]].forEach(([cls, label, x0]) => {
            svg.appendChild(s("line", { x1: X(x0), x2: X(x0 + 3), y1: Y(1.45), y2: Y(1.45), class: `series ${cls}` }));
            svg.appendChild(s("text", { x: X(x0 + 3.8), y: Y(1.45) + 4, class: "tf-leg-t" }, label));
          });
        },
      }),
      readouts: [
        { label: String.raw`\(\omega_i\) (radians per position)`, f: (p) => Math.pow(10000, (-2 * p.i) / 16), fmt: (v) => (v >= 0.01 ? v.toFixed(4) : fmt.sci(v, 2)) },
        { label: "wavelength (positions)", f: (p) => (2 * Math.PI) / Math.pow(10000, (-2 * p.i) / 16), fmt: (v) => (v < 100 ? v.toFixed(2) : Math.round(v).toLocaleString("en-US")) },
      ],
      aria: "Sine and cosine of position times omega for the chosen pair",
    },
    src: ["Vaswani et al., Attention Is All You Need, 2017, section 3.5 and Table 3 row (E)"],
  });

  Deck.slide({
    id: "tf-rope",
    title: "Rotary embeddings rotate queries and keys by position",
    example: "tokens ids",
    body: String.raw`<p><b>Rotary position embedding</b> (RoPE) adds nothing to the stream. Inside every attention layer it rotates each 2-D pair of a query or a key by an angle proportional to the position: pair i of the vector at position m turns by \(m\theta_i\), with \(\theta_i = b^{-2i/d_h}\).</p>
<p>A rotation preserves length, and rotating q by mθ and k by nθ changes the angle between them by (m − n)θ. So the score depends on the positions only through m − n. The same two vectors 2 positions apart score the same at positions (3, 1) as at (5, 3): attention sees relative position directly.</p>
<p>Llama 3 uses b = 500,000 and \(d_h\) = 128. Pair 0 turns 1 radian per position; pair 63 turns 2.46 × 10⁻⁶ radians, one full turn every 2.56 million positions. For their 128K context the released Llama 3.1 models also slow pairs 35 to 63 by a factor of 8 and pairs 29 to 34 by less (<code>apply_scaling</code> in Meta's code), so pair 63 turns 3.07 × 10⁻⁷ radians per position, one turn every 20.5 million.</p>`,
    math: String.raw`(R_{m\theta}\,\mathbf{q})\cdot(R_{n\theta}\,\mathbf{k}) = \mathbf{q}\cdot(R_{(n-m)\theta}\,\mathbf{k})`,
    fig: { type: "tf-rope", caption: "One 2-D pair of a query and a key, rotated by their positions", q: [0.6, 0.8], k: [1, 0], theta: 0.5, m: 3, n: 1 },
    terms: [{ term: "Rotary position embedding (RoPE)", def: "Position information applied inside attention: each 2-D pair of a query or key is rotated by its position times a fixed frequency, so scores depend on relative position." }],
    src: ["Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding, arXiv:2104.09864, section 3.2.2, eqs. 15 and 16", "Dubey et al., The Llama 3 Herd of Models, 2024, section 3.2 and Table 3", "Meta, llama3 reference code, model.py, precompute_freqs_cis", "Meta, llama-models, models/llama3/model.py, apply_scaling (Llama 3.1)"],
  });

  Deck.slide({
    id: "tf-rope-worked",
    title: "RoPE worked in two dimensions",
    example: "tokens ids",
    body: String.raw`<p>Take one pair with θ = 0.5, a query q = (0.6, 0.8) and a key k = (1, 0), both of length 1. Unrotated, q · k = 0.6. Put the query at position 3 and the key at position 1; the plate rotates both and takes the dot product.</p>
<p>The result is cos(0.927 + 1.5 − 0.5) = cos 1.927 = −0.349, where 0.927 rad is the angle of q. At positions 5 and 3 the rotations are 2.5 and 1.5, the difference is again 1 rad, and the score is again −0.349.</p>
<p>A real head applies this to all \(d_h/2\) pairs at once, each with its own θ, and the score is the sum over pairs.</p>`,
    fig: {
      type: "steps", caption: "Rotating a query at position 3 and a key at position 1, θ = 0.5",
      steps: [
        { html: String.raw`\(R(\alpha) = \begin{pmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{pmatrix}\); the query turns by 3 × 0.5 = 1.5, the key by 1 × 0.5 = 0.5`, note: "rotation matrix" },
        { html: String.raw`\(R(1.5)\,\mathbf{q}\) has \(x = 0.0707 \cdot 0.6 - 0.9975 \cdot 0.8\) \(= -0.7556\) and \(y = 0.9975 \cdot 0.6 + 0.0707 \cdot 0.8\) \(= 0.6551\)`, note: "cos 1.5 = 0.0707, sin 1.5 = 0.9975" },
        { html: String.raw`\(R(0.5)\,\mathbf{k} = (\cos 0.5,\ \sin 0.5)\) \(= (0.8776,\ 0.4794)\)` },
        { html: String.raw`\(-0.7556 \cdot 0.8776 + 0.6551 \cdot 0.4794\) \(= -0.349\)`, note: "the score" },
        { html: String.raw`At positions 5 and 3: \(R(2.5)\,\mathbf{q}\) \(= (-0.9595,\ -0.2818)\) and \(R(1.5)\,\mathbf{k}\) \(= (0.0707,\ 0.9975)\); their dot product is again \(-0.349\)`, note: "same m − n, same score" },
      ],
    },
    check: {
      q: "With RoPE, a query at position 10 and a key at position 4 have score s. For the same two vectors, which positions also give s?",
      options: ["Query at 16, key at 10", "Query at 10, key at 16", "Query at 20, key at 8", "Query at 5, key at 2"],
      answer: 0,
      why: "The score depends only on m − n. (16, 10) has m − n = 6, like (10, 4). (20, 8) and (5, 2) have 12 and 3. (10, 16) has −6, a rotation the other way, and in a causal model a key after the query is masked anyway.",
    },
  });

  // ================================================================== 5.6
  Deck.chapter({ title: "Training the transformer" });

  Deck.slide({
    id: "tf-train",
    title: "Training our model: the loss over 6,000 steps",
    example: "tokens ids",
    body: String.raw`<p>The plate replays the real training run. The smooth curve is the exact expected loss over all 270 queries, computed from the weights at every 10th step; the noisy one is the loss of the training batches, averaged over 10 steps. The slider loads one of 13 saved snapshots and recomputes the predictions on our query.</p>
<p>At step 0 the weights are random and the loss is 4.7217, close to ln 113 = 4.7274, the loss of a uniform guess over 113 tokens. It falls to 2.3086 by step 50 and 0.8193 by step 100, then creeps down: 0.7179 at step 200 and 0.6942 at step 6,000, against a floor of 0.6937 (Part 2 derives it).</p>
<p>The recipe: Adam with β = (0.9, 0.95), learning rate 0.01 after 100 warmup steps, cosine decay to 0.001, and 128 freshly sampled queries per step.</p>`,
    fig: { type: "tinylm", widget: "train", caption: "The real training run: loss curves and the model's predictions on our query at each saved step" },
  });

  Deck.slide({
    id: "tf-learn-order",
    title: "Fixed tokens are learned first, the filter column last",
    example: "tokens ids",
    body: String.raw`<p>Split the loss by position and subtract each position's floor: what remains is what the model has not yet learned there. The plate plots this excess loss for the 8 positions.</p>
<p>The four fixed tokens (<code>FROM</code>, <code>WHERE</code>, <code>=</code>, <code>;</code>), the first column and the digit come within 0.05 nats of their floors by steps 90 to 100 and stay there; none of them depends on an earlier token. The table after <code>FROM</code>, which depends on the column, follows at step 140. The filter column after <code>WHERE</code> first gets within 0.05 at step 340 and stays there only from step 560.</p>
<p>The filter column is the only prediction in the grammar that depends on two earlier tokens, the table and the selected column. It is also the position where the trained model still has the largest gap to its floor.</p>`,
    fig: {
      type: "chart", caption: "Excess loss (loss minus floor) at each position during training, log scales",
      x: { domain: [10, 6000], type: "log", ticks: [10, 100, 1000, 6000], label: "training step", fmt: (v) => v.toLocaleString("en-US") },
      y: { domain: [0.001, 5], type: "log", ticks: [0.001, 0.01, 0.1, 1], label: "excess loss (nats)", fmt: (v) => String(v) },
      series: [1, 3, 5, 7, 0, 6, 2, 4].map((i) => ({
        points: M.data.exactCurve.filter(([st]) => st >= 10).map(([st, , per]) => [st, Math.max(per[i] - M.data.floor[i], 1e-5)]),
        cls: { 0: "s3", 2: "s1", 4: "s2", 6: "s4" }[i] || "ink thin",
      })),
      hlines: [{ y: 0.05, label: "0.05 nats" }],
      legend: [["s2", "filter column"], ["s1", "table"], ["s3", "first column"], ["s4", "digit"], ["ink", "FROM, WHERE, =, ;"]],
    },
  });

  Deck.slide({
    id: "tf-backprop",
    title: "The backward pass through the output layer",
    example: "tokens ids focus:4 target:5",
    body: String.raw`<p>Each training step needs the gradient of the loss with respect to all 9,968 parameters. Backpropagation (Part 3) gets them in one backward pass that visits the forward computation in reverse. The plate runs its last steps at <code>WHERE</code>, where the true next token is <code>id</code>.</p>
<p>Forward: p(<code>id</code>) = 0.346, so the loss is −ln 0.346 = 1.061. Backward: the gradient with respect to the logit of <code>id</code> is p − 1 = −0.654, and for every other token j it is \(p_j\) (Part 2). The logit is \(\mathbf{h} \cdot \mathbf{w}\), with \(\mathbf{w}\) column 58 of \(\mathbf{W}_U\), so each entry of \(\mathbf{w}\) gets gradient \((p - 1)\,h_c\).</p>
<p>The gradient with respect to \(\mathbf{h}\) is \(\sum_j (p_j - y_j)\,\mathbf{w}_j\), and it keeps flowing down through the final norm, both blocks and the embeddings.</p>`,
    fig: { type: "tinylm", widget: "backprop", caption: "Forward values and backward gradients for the prediction at <code>WHERE</code>" },
  });

  Deck.slide({
    id: "tf-floor",
    title: "The final loss is 0.0005 nats above the floor",
    example: "tokens ids",
    body: String.raw`<p>After 6,000 steps the exact expected loss is 0.6942 nats per token. No model can do better than the grammar's entropy, 0.6937 (Part 2). The remaining gap, averaged over the 8 positions, is 0.0005 nats per token: the model is 0.07% above the floor.</p>
<p>The plate splits it by position. The four fixed tokens cost less than 10⁻⁷. The gap sits at the four positions with real choices, and most of it at <code>WHERE</code>: 0.0026 of the 0.0039 summed over positions, the position that was learned last.</p>
<p>On real text there is no formula for the floor. Here the question "how good is the model" has an exact answer.</p>`,
    fig: {
      type: "table", caption: "Exact expected loss per position at step 6,000, the floor, and the gap", dense: true,
      cols: ["Position: token", "Predicts", "Model", "Floor", "Gap"], align: ["l", "l", "r", "r", "r"],
      get rows() {
        const per = M.exactLoss().perPos, fl = M.data.floor, names = ["first column", "FROM", "table", "WHERE", "filter column", "=", "digit", ";"];
        const rows = per.map((v, i) => [`${i}: <code>${short(TOKS[i])}</code>`, names[i], v.toFixed(5), fl[i].toFixed(5), (v - fl[i]).toFixed(5)]);
        const sum = (a) => a.reduce((x, y) => x + y, 0);
        rows.push(["", "mean", (sum(per) / 8).toFixed(5), (sum(fl) / 8).toFixed(5), `<b>${((sum(per) - sum(fl)) / 8).toFixed(5)}</b>`]);
        return rows;
      },
      highlight: [4],
    },
  });

  Deck.slide({
    id: "tf-heldout",
    title: "Rules or memorization: a model that never saw our query",
    example: "tokens ids",
    body: String.raw`<p>Our model saw every query thousands of times: in expectation 2,370 times for each <code>users</code> or <code>orders</code> query, ours included, and 4,741 times for each <code>products</code> query. Its loss cannot separate learning the grammar from memorizing the queries. A second copy was trained with the same recipe on 243 queries; 27, ours included, were held out.</p>
<p>On our query it still gets the structure: <code>FROM</code>, <code>users</code> and <code>WHERE</code> at p = 1.000, and the filter column <code>id</code> at 0.348. The digit is where it fails: in its training data our prefix appeared with every digit except 7, and it gives <code>7</code> 0.003.</p>
<p>After <code>=</code>, its loss on the queries it trained on is 2.10, below ln 9 = 2.197, the least any model can average when the digit is uniform. On the held-out queries it is 6.76. Going below the floor on training data is memorization, and only held-out data shows its cost.</p>`,
    fig: { type: "tinylm", widget: "heldout", caption: "Our query under the full model and under the copy that never saw it" },
  });

  Deck.slide({
    id: "tf-real",
    title: "Real models are the same parts, much larger",
    example: "tokens ids",
    body: String.raw`<p>The plate lines our configuration up with the three Llama 3.1 models. Every row is a hyperparameter this part has computed with: layers, the stream width d, query and key/value heads, the MLP width, the vocabulary, the context length.</p>
<p>From ours to the 405B model, d grows 1,024-fold (16 to 16,384) and the layer count 63-fold (2 to 126). The parameter count grows about 40 million-fold. For the 405B model, 12·L·d² = 12 × 126 × 16,384² = 4.06 × 10¹¹ comes within 0.01% of the true count, partly by coincidence: its blocks hold 11.875·L·d², and the vocabulary tables nearly make up the difference. For ours the same term is only 6,144 of 9,968. Training tokens grow about 2.5 million-fold, from 6.1 million to 15.6 trillion.</p>`,
    fig: { type: "tinylm", widget: "real", caption: "Our model next to Llama 3.1 8B, 70B and 405B" },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024, Table 3 and section 3", "Meta, Llama 3.1 model card (about 15T pre-training tokens)", "Meta, llama-models, sku_list.py (vocabulary of 128,256)"],
  });

  Deck.slide({
    id: "tf-train-check",
    title: "Check: what the training run measured",
    body: String.raw`<p>The training chapter in four numbers: a loss of 4.7217 at random initialization, 0.6942 after 6,000 steps, a floor of 0.6937, and a held-out copy whose loss after <code>=</code> is 2.10 on queries it trained on and 6.76 on queries it did not.</p>
<p>The question asks what the last pair of numbers shows.</p>`,
    check: {
      q: "The held-out copy reaches loss 2.10 after <code>=</code> on queries it trained on, and 6.76 on held-out queries. The floor after <code>=</code> is ln 9 = 2.197. What does this show?",
      options: ["It beats the floor, so it learned the grammar better than the full model", "It learned which digits followed each prefix in its training set; held-out loss exposes it", "The floor does not apply to a subset of queries", "Held-out queries are longer, so their loss is higher"],
      answer: 1,
      why: "No model can average below ln 9 on a uniformly random digit. Going below it on seen queries means the model fit its particular sample: for our prefix it saw 8 digits, not 9, and gives 7 only 0.003. The cost shows on held-out queries, 6.76. All queries are 9 tokens long.",
    },
  });

  // ================================================================== 5.7
  Deck.chapter({ title: "Modern variants" });

  Deck.slide({
    id: "tf-swiglu",
    title: "SwiGLU: a gated feed-forward layer",
    example: "tokens ids",
    body: String.raw`<p>Llama models use <b>SwiGLU</b> (Shazeer, 2020) where our model has a plain GELU MLP: two input projections instead of one, multiplied entry by entry, so \(\mathrm{SiLU}(z) = z\,\sigma(z)\) of one projection gates the other.</p>
<p>Three d × \(d_{ff}\) matrices cost \(3d\,d_{ff}\) parameters against \(2d\,d_{ff}\) for the plain MLP, so Shazeer multiplied \(d_{ff}\) by 2/3 to keep the count equal: 2,048 instead of 3,072 at d = 768. Llama 3 8B uses \(d_{ff}\) = 14,336 = 3.5d: 3 × 4,096 × 14,336 = 176,160,768 parameters per layer, 81% of its 218,103,808 weights per layer.</p>
<p>The plate plots one hidden unit's output SiLU(a)·b against a, for the b you choose. The second projection b scales the unit's output, sign included, independently of a; in a plain MLP each unit's output is a fixed function of a alone.</p>`,
    math: [String.raw`\mathbf{h} = \mathrm{SiLU}(\mathbf{x}\mathbf{W}_1) \odot \mathbf{x}\mathbf{W}_3`, String.raw`\mathrm{FFN}_{\mathrm{SwiGLU}}(\mathbf{x}) = \mathbf{h}\,\mathbf{W}_2`],
    fig: {
      type: "plot", caption: "One SwiGLU unit: the gate SiLU(a) times the second projection b (thick), with GELU(a) (thin) for comparison", x: [-4, 4], y: [-8, 8], xlabel: "a = x · w₁", ylabel: "unit output",
      params: [{ name: "b", label: "second projection b = x · w₃", min: -2, max: 2, step: 0.1, value: 0.5, fmt: "f1" }],
      fns: [
        { f: (a, p) => (a / (1 + Math.exp(-a))) * p.b, cls: "s1" },
        { f: (a) => M.gelu(a), cls: "ink thin" },
      ],
      readouts: [
        { label: "SiLU(2)", f: () => 2 / (1 + Math.exp(-2)), fmt: "f4" },
        { label: "output at a = 2", f: (p) => (2 / (1 + Math.exp(-2))) * p.b, fmt: "f4" },
      ],
      aria: "SiLU of a times the gate value b, and GELU of a",
    },
    terms: [{ term: "SwiGLU", def: "A feed-forward layer (SiLU(xW₁) ⊙ xW₃)W₂: one projection, passed through SiLU, multiplies a second projection entry by entry." }],
    src: ["Shazeer, GLU Variants Improve Transformer, arXiv:2002.05202, 2020, sections 2 and 3.1", "Dubey et al., The Llama 3 Herd of Models, 2024, Table 3", "Meta, llama3 reference code, model.py, FeedForward"],
  });

  Deck.slide({
    id: "tf-gqa",
    title: "Grouped-query attention shares keys and values across heads",
    example: "tokens ids",
    body: String.raw`<p>In multi-head attention every query head has its own key head and value head. <b>Multi-query attention</b> (MQA, Shazeer 2019) gives all query heads one shared key head and one shared value head. <b>Grouped-query attention</b> (GQA, Ainslie et al. 2023) sits between: the query heads form G groups, and each group shares one key head and one value head. G = 1 is MQA and G = H is multi-head attention.</p>
<p>Llama 3 70B has 64 query heads and 8 key/value heads, so each key/value head serves 8 query heads. \(\mathbf{W}_K\) and \(\mathbf{W}_V\) shrink from 8,192 × 8,192 to 8,192 × 1,024. So does what generation stores per past token: under the causal mask a token's keys and values depend only on the tokens up to it, so they never change and are kept instead of recomputed. In BF16 that is 2 × 80 layers × 8 heads × 128 × 2 bytes = 327,680 bytes, against 2,621,440 with 64 key/value heads; Part 8 builds the KV cache on this number.</p>`,
    fig: { type: "tf-gqa", caption: "Llama 3 70B's 64 query heads and the key/value heads they read", kv: 8, ctx: 131072 },
    terms: [
      { term: "Multi-query attention (MQA)", def: "Attention in which all query heads share a single key head and a single value head." },
      { term: "Grouped-query attention (GQA)", def: "Attention in which query heads are split into G groups and each group shares one key head and one value head; G = 1 is MQA, G = H is multi-head attention." },
    ],
    src: ["Shazeer, Fast Transformer Decoding: One Write-Head is All You Need, arXiv:1911.02150, 2019", "Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, EMNLP 2023, section 2.2", "Dubey et al., The Llama 3 Herd of Models, 2024, section 3.2 and Table 3"],
  });

  Deck.slide({
    id: "tf-moe",
    title: "Mixture of experts: a router picks k experts per token",
    example: "tokens ids",
    body: String.raw`<p>A <b>mixture-of-experts</b> (MoE) layer replaces the one MLP with E expert MLPs and a <b>router</b>, a d × E matrix. For each token the router computes E scores and keeps the top k; the layer outputs a weighted sum of those k experts' outputs (Mixtral weights them by a softmax over the k kept scores). The other E − k experts do no work for that token.</p>
<p>Mixtral 8x7B uses E = 8 and k = 2 in each of its 32 layers. When experts run with fixed-size buffers, each gets a <b>capacity</b>: the Switch Transformer (k = 1) allows capacity factor × T/E of a batch's T tokens per expert, which with k experts per token becomes factor × T·k/E. Tokens over the limit skip the expert and pass on through the residual connection. The Mixtral paper describes no capacity limit; it points to MegaBlocks kernels, which take a variable number of tokens per expert and never drop any.</p>
<p>The plate is a toy router on random scores (our model has no experts) that applies this capacity rule. Part 7 covers placing experts on GPUs.</p>`,
    math: [String.raw`\mathbf{y} = \sum_{i=1}^{E} G(\mathbf{x})_i\, \mathrm{FFN}_i(\mathbf{x})`, String.raw`G(\mathbf{x}) = \mathrm{softmax}\bigl(\mathrm{TopK}(\mathbf{x}\mathbf{W}_g)\bigr)`],
    fig: { type: "tinylm", widget: "moe", caption: "A toy router: 4 queries (32 tokens), 8 experts on 4 GPUs" },
    terms: [
      { term: "Mixture of experts (MoE)", def: "A layer with E expert MLPs of which a router selects k per token; the parameters grow with E, the work per token with k." },
      { term: "Router", def: "In an MoE layer, the d × E matrix whose scores choose the top-k experts for each token and weight their outputs." },
      { term: "Expert capacity", def: "The most tokens an expert accepts from a batch of T tokens: capacity factor × T·k/E. Tokens over it skip the expert through the residual connection." },
    ],
    src: ["Jiang et al., Mixtral of Experts, arXiv:2401.04088, 2024, Table 1 and section 2.1", "Fedus, Zoph and Shazeer, Switch Transformers, JMLR 2022, section 2.2 and eq. 3 (expert capacity)", "Gale et al., MegaBlocks: Efficient Sparse Training with Mixture-of-Experts, arXiv:2211.15841, 2022"],
  });

  Deck.slide({
    id: "tf-moe-params",
    title: "Total and active parameters of an MoE model",
    example: "tokens ids",
    body: String.raw`<p>An MoE model stores every expert but runs only k per token. Counted from Mixtral's published shapes: per layer, attention 41,943,040, experts 8 × 176,160,768, router 32,768 and norms 8,192; times 32 layers, plus 262,144,000 in the two vocabulary tables and 4,096 in the final norm. The total is 46,702,792,704. With 2 of the 8 experts per token, the active count is 12,879,925,248.</p>
<p>So Mixtral stores 46.7B parameters and computes with 12.9B per token (the paper rounds to 47B and 13B). DeepSeek-V3 pushes the ratio further: 671B total and 37B active, with 256 routed experts plus 1 shared expert per MoE layer and 8 routed experts per token.</p>`,
    fig: {
      type: "calc", caption: "Mixtral 8x7B's shapes with other expert counts (d = 4,096, 32 layers, expert width 14,336)",
      inputs: [
        { id: "E", label: "Experts E", options: [2, 4, 8, 16, 32, 64].map((x) => ({ label: String(x), value: x })), value: 8 },
        { id: "k", label: "Active per token k", options: [1, 2, 4].map((x) => ({ label: String(x), value: x })), value: 2 },
      ],
      outputs: [
        { label: "total parameters", f: (v) => 32 * (41943040 + v.E * 176160768 + 4096 * v.E + 8192) + 262144000 + 4096, fmt: "int" },
        { label: "active per token", f: (v) => 32 * (41943040 + Math.min(v.k, v.E) * 176160768 + 4096 * v.E + 8192) + 262144000 + 4096, fmt: "int" },
        { label: "active share", f: (v) => (32 * (41943040 + Math.min(v.k, v.E) * 176160768 + 4096 * v.E + 8192) + 262144000 + 4096) / (32 * (41943040 + v.E * 176160768 + 4096 * v.E + 8192) + 262144000 + 4096), fmt: "pct1" },
      ],
      note: (v) => `Per layer: attention 41,943,040 + ${v.E} experts × 176,160,768 + router 4,096 × ${v.E} + norms 8,192. Experts per token: ${Math.min(v.k, v.E)}${v.k > v.E ? " (k cannot exceed E)" : ""}.`,
    },
    src: ["Jiang et al., Mixtral of Experts, 2024, Table 1 and abstract (47B total, 13B active)", "Mistral AI, Mixtral-8x7B-v0.1 config.json on Hugging Face (untied embeddings, same shapes)", "DeepSeek-AI, DeepSeek-V3 Technical Report, arXiv:2412.19437, abstract and section 4.2"],
  });

  Deck.slide({
    id: "tf-long",
    title: "Sliding-window attention caps the cost of long context",
    example: "tokens ids",
    body: String.raw`<p>Full causal attention scores T(T + 1)/2 pairs per head and keeps the keys and values of every past token. <b>Sliding-window attention</b> lets each position read only the last W positions, itself included: about T·W pairs, and at most W keys and values per layer to keep.</p>
<p>Information still travels further than W. Each layer moves it back by up to W − 1 positions, so after L layers a position can depend on tokens about L·W back. Mistral 7B uses W = 4,096 in all 32 layers, a theoretical span of about 131K tokens. Llama 3 keeps full attention and reaches 128K tokens with a long-context stage of pre-training and slower RoPE rotations (chapter 5.5). At RoFormer's base of 10,000, all 64 pairs of a 128-number head turn a full circle within 128K positions; at 500,000 only 49 do, and with Llama 3.1's rescaling 39, so more pairs keep angles that do not repeat across the whole context.</p>`,
    fig: { type: "tf-swa", caption: "Which pairs are scored, and how far back the last position can reach", T: 16, W: 4, L: 2 },
    terms: [{ term: "Sliding-window attention", def: "Attention in which each position reads only the last W positions, so cost and cache grow with W instead of the full context." }],
    src: ["Jiang et al., Mistral 7B, arXiv:2310.06825, 2023, section 2 and Table 1", "Dubey et al., The Llama 3 Herd of Models, 2024, sections 3.2 and 3.4.2", "Su et al., RoFormer, arXiv:2104.09864 (base 10,000)", "Meta, llama-models, models/llama3/model.py, apply_scaling"],
  });

  Deck.slide({
    id: "tf-tying",
    title: "Weight tying: one table for embedding and unembedding",
    example: "tokens ids",
    body: String.raw`<p>The embedding \(\mathbf{W}_E\) (V × d) maps token ids to vectors; the unembedding \(\mathbf{W}_U\) (d × V) maps vectors to token scores. <b>Weight tying</b> uses one matrix for both, \(\mathbf{W}_U = \mathbf{W}_E^\top\) (Press and Wolf, 2017); the original transformer tied them.</p>
<p>The saving is V·d parameters. Our model does not tie; tying would remove 1,808 of its 9,968 parameters, 18.1%. Llama 3 does not tie either: each of its 128,256 × d tables is 6.5% of the 8B model, 1.5% of the 70B and 0.5% of the 405B. The larger the model, the less a second table costs.</p>`,
    fig: {
      type: "bars", caption: "Share of all parameters in one V × d table", fmt: "pct1",
      items: [
        { label: "ours", value: 1808 / 9968, text: "18.1%", hi: true },
        { label: "Llama 3 8B", value: 525336576 / 8030261248, text: "6.5%" },
        { label: "Llama 3 70B", value: 1050673152 / 70553706496, text: "1.5%" },
        { label: "Llama 3 405B", value: 2101346304 / 405853388800, text: "0.5%" },
      ],
      note: "Totals computed from the published shapes: 8,030,261,248, 70,553,706,496 and 405,853,388,800 parameters.",
    },
    terms: [{ term: "Weight tying", def: "Using the transpose of the token-embedding matrix as the unembedding matrix, which saves V·d parameters." }],
    src: ["Press and Wolf, Using the Output Embedding to Improve Language Models, EACL 2017", "Vaswani et al., 2017, section 3.4", "Meta, llama3 reference code, model.py (separate output layer)"],
  });

  Deck.slide({
    id: "tf-llama3",
    title: "The Llama 3 architecture, hyperparameter by hyperparameter",
    layout: "wide",
    example: "tokens ids",
    body: String.raw`<p>Meta's paper lists the Llama 3 models in one table, reproduced in the plate with three added rows: head size and parameter count computed from the table, and the context length from the text. Every line is a quantity this part has computed with on our model: L blocks, the stream width d, the MLP width \(d_{ff}\) of a SwiGLU layer, query heads H with head size d/H = 128, 8 key/value heads for grouped-query attention, a vocabulary of 128,000 tokens (128,256 rows with the special tokens), and RoPE with base 500,000 (the Llama 3.1 models also rescale the slow pairs, chapter 5.5).</p>
<p>Pre-training used 8K-token sequences; a final long-context stage extended the context to 128K tokens. The peak learning rate falls as the models grow.</p>`,
    fig: {
      type: "table", caption: "Llama 3 hyperparameters (paper, Table 3), with head size and parameter count computed and the context length from section 3.4", dense: true,
      cols: ["", "8B", "70B", "405B", "What it sets"], align: ["l", "r", "r", "r", "l"], rowHeads: true,
      rows: [
        ["Layers", "32", "80", "126", "number of blocks L"],
        ["Model dimension", "4,096", "8,192", "16,384", "width d of the residual stream"],
        ["FFN dimension", "14,336", "28,672", "53,248", String.raw`SwiGLU hidden width \(d_{ff}\)`],
        ["Attention heads", "32", "64", "128", "query heads H"],
        ["Key/value heads", "8", "8", "8", "GQA groups"],
        ["Head size", "128", "128", "128", "d / H (derived)"],
        ["Peak learning rate", String.raw`\(3 \times 10^{-4}\)`, String.raw`\(1.5 \times 10^{-4}\)`, String.raw`\(8 \times 10^{-5}\)`, "Adam step size after warmup"],
        ["Activation", "SwiGLU", "SwiGLU", "SwiGLU", "MLP form"],
        ["Vocabulary", "128,000", "128,000", "128,000", "V (+256 special tokens in the tokenizer)"],
        ["Positions", "RoPE", "RoPE", "RoPE", "base θ = 500,000; 3.1 rescales the slow pairs"],
        ["Context", "128K", "128K", "128K", "after the long-context stage (8K before)"],
        ["Parameters", "8,030,261,248", "70,553,706,496", "405,853,388,800", "computed from the shapes (derived)"],
      ],
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, arXiv:2407.21783, 2024, Table 3 and sections 3.2, 3.4", "Meta, llama-models, sku_list.py (vocabulary 128,256, use_scaled_rope) and models/llama3/model.py (apply_scaling)"],
  });

  Deck.slide({
    id: "tf-vs-llama",
    title: "Our model and Llama 3 8B, line by line",
    example: "tokens ids",
    body: String.raw`<p>The same computation at two sizes. Both are stacks of blocks with causal self-attention and a feed-forward sublayer, pre-norm RMSNorm (ε = 10⁻⁵), a residual stream, a final norm and an untied unembedding. Beyond the sizes, Llama 3 8B changes four parts: SwiGLU instead of GELU, RoPE instead of a position table, 8 key/value heads for 32 query heads, and a byte-level BPE tokenizer built on tiktoken, with 128,256 tokens.</p>
<p>The parameter counts follow from the shapes in both columns: 9,968 for ours and 8,030,261,248 for Llama 3 8B, of which 6,979,584,000 sit in the 32 blocks (218,112,000 each) and 1,050,673,152 in the two vocabulary tables.</p>`,
    fig: {
      type: "table", caption: "Our model next to Llama 3 8B", dense: true,
      cols: ["", "Ours", "Llama 3 8B"], align: ["l", "l", "l"], rowHeads: true,
      rows: [
        ["Vocabulary rows", "113", "128,256"],
        ["Width d", "16", "4,096"],
        ["Blocks", "2", "32"],
        ["Query / key-value heads", "2 / 2", "32 / 8"],
        ["Head size", "8", "128"],
        ["Feed-forward", "16 → 64 → 16, GELU", "4,096 → 14,336 → 4,096, SwiGLU"],
        ["Positions", "learned table, 8 × 16", "RoPE, θ = 500,000"],
        ["Context", "8", "8K (128K for 3.1)"],
        ["Parameters per block", "3,104", "218,112,000"],
        ["Parameters", "9,968", "8,030,261,248"],
      ],
      highlight: [9],
    },
    check: {
      q: "Which single change would most reduce Llama 3 8B's KV cache per token, everything else equal?",
      options: ["Tie the embedding and unembedding", "Use 1 key/value head instead of 8 (MQA)", "Replace SwiGLU with GELU", "Use a learned position table"],
      answer: 1,
      why: "The cache per token is 2 × layers × key/value heads × head size × bytes = 2 × 32 × 8 × 128 × 2 = 131,072 bytes in BF16. One key/value head divides it by 8, to 16,384 bytes. The other changes do not touch keys or values.",
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024, Table 3", "Meta, llama3 reference code, model.py (norm_eps 1e-5, separate output layer)"],
  });
})();
