/* Widgets for Pass 3: training at scale. */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip, slider, segmented, button, player, heatStrip, chart, seq, decades } = window.UI;
  const M = window.TinyLM;
  const W = (window.Widgets = window.Widgets || {});
  const trace = () => W._trace();
  const H100 = { flops: 989e12, bw: 3.35e12, mem: 80e9 };
  const stat = (v, l) => h("div", { class: "stat" }, h("div", { class: "stat-v" }, v), h("div", { class: "stat-l" }, l));
  const logSlider = (o) => slider(Object.assign({}, o, {
    min: Math.log10(o.lo), max: Math.log10(o.hi), step: "any", value: Math.log10(o.value),
    format: (v) => o.format(Math.pow(10, v)), onInput: (v) => o.onInput(Math.pow(10, v)),
  }));
  const human = (n) => (n >= 1e12 ? `${+(n / 1e12).toFixed(n >= 1e13 ? 0 : 1)}T` : n >= 1e9 ? `${+(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B` : n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${+(n / 1e3).toFixed(1)}K` : String(Math.round(n)));

  // ------------------------------------------------------------ flops
  W.flops = {
    init(el) {
      const PRESETS = [
        // N counts matrix weights: the 1,936 embedding and position entries are lookups and cost no FLOPs
        { label: "Ours", N: 7952, D: 6.144e6, gpus: 1, note: "For our model N counts its 7,952 matrix weights; its 1,936 embedding and position entries are table lookups and cost no FLOPs." },
        { label: "Llama 3.1 8B", N: 8e9, D: 15e12, reportedH: 1.46e6 },
        { label: "Llama 3.1 70B", N: 70e9, D: 15e12, reportedH: 7.0e6 },
        { label: "Llama 3.1 405B", N: 405e9, D: 15.6e12, gpus: 16384, reportedH: 30.84e6 },
      ];
      let st = Object.assign({ mfu: 0.4 }, PRESETS[3]);
      const out = el.querySelector(".fl-out"), note = el.querySelector(".fl-note");
      const sN = logSlider({ id: "fl-n", label: "Parameters N", lo: 1e3, hi: 2e12, value: st.N, format: human, onInput: (v) => { st.N = v; st.reportedH = null; st.note = null; draw(); } });
      const sD = logSlider({ id: "fl-d", label: "Training tokens D", lo: 1e5, hi: 1e14, value: st.D, format: human, onInput: (v) => { st.D = v; st.reportedH = null; draw(); } });
      const sG = slider({ id: "fl-g", label: "H100 GPUs", min: 0, max: 15, step: 1, value: Math.log2(st.gpus), format: (v) => Math.pow(2, v).toLocaleString("en-US"), onInput: (v) => { st.gpus = Math.pow(2, v); draw(); } });
      const sU = slider({ id: "fl-u", label: "Utilization (MFU)", min: 0.1, max: 0.6, step: 0.01, value: st.mfu, format: (v) => fmt.pct(v), onInput: (v) => { st.mfu = v; draw(); } });
      el.querySelector(".fl-presets").appendChild(segmented({ label: "Preset", value: 3, options: PRESETS.map((p, i) => ({ value: i, label: p.label })), onChange: (i) => {
        const pr = PRESETS[i];
        st.N = pr.N; st.D = pr.D; st.reportedH = pr.reportedH || null; st.note = pr.note || null;
        if (pr.gpus) st.gpus = pr.gpus;
        sN.set(Math.log10(st.N)); sD.set(Math.log10(st.D)); sG.set(Math.log2(st.gpus)); draw();
      } }));
      el.querySelector(".fl-ctl").append(sN, sD, sG, sU);
      function draw() {
        const C = 6 * st.N * st.D;
        const gpuSec = C / (st.mfu * H100.flops);
        const wall = gpuSec / st.gpus;
        clear(out);
        out.append(stat(fmt.sci(C, 1), "training FLOPs"), stat(gpuSec < 3600 ? fmt.time(gpuSec) : `${fmt.sci(gpuSec / 3600, 2)}`, gpuSec < 3600 ? "on one H100" : "H100 GPU-hours"), stat(fmt.time(wall), `wall-clock on ${st.gpus.toLocaleString("en-US")} GPU${st.gpus > 1 ? "s" : ""}`));
        const base = `${(st.D / st.N).toFixed(0)} tokens per parameter. The estimate assumes ${fmt.pct(st.mfu)} of the H100's 989 TFLOP/s dense BF16 peak.`;
        note.textContent = (st.reportedH
          ? `${base} Meta's model card reports ${fmt.sci(st.reportedH, 2)} H100 GPU-hours; with 6ND, that implies an average utilization of about ${fmt.pct(C / (st.reportedH * 3600 * H100.flops))}.`
          : base) + (st.note ? " " + st.note : "");
      }
      draw();
    },
  };

  // ------------------------------------------------------------ scaling
  W.scaling = {
    init(el) {
      const MODELS = [
        { name: "ours", N: 7952, D: 6.144e6, dx: 8, dy: -8 }, // matrix weights, as in the flops widget
        { name: "GPT-3", N: 175e9, D: 300e9, dx: 8, dy: 14 },
        { name: "Chinchilla", N: 70e9, D: 1.4e12, dx: -10, dy: -10, anchor: "end" },
        { name: "Llama 3.1 8B", N: 8e9, D: 15e12, dx: -10, dy: 4, anchor: "end" },
        { name: "70B", N: 70e9, D: 15e12, dx: 0, dy: -12, anchor: "middle" },
        { name: "405B", N: 405e9, D: 15.6e12, dx: 9, dy: 4 },
      ];
      let C = 3.8e25;
      const chartEl = el.querySelector(".sc-chart"), note = el.querySelector(".sc-note");
      const xd = [1e3, 3e12], yd = [1e5, 3e14];
      const isoLine = (c) => { const pts = []; for (let e = Math.log10(xd[0]); e <= Math.log10(xd[1]) + 1e-9; e += 0.05) { const n = Math.pow(10, e), d = c / (6 * n); if (d >= yd[0] && d <= yd[1]) pts.push([n, d]); } return pts; };
      const ch = chart(chartEl, {
        width: 560, height: 370, margin: { l: 56, r: 18, t: 12, b: 40 }, aria: "Parameters versus training tokens for several models",
        x: { type: "log", domain: xd, ticks: [1e3, 1e5, 1e7, 1e9, 1e11], fmt: (v) => human(v), label: "parameters N" },
        y: { type: "log", domain: yd, ticks: [1e5, 1e7, 1e9, 1e11, 1e13], fmt: (v) => human(v), label: "training tokens D" },
      });
      function draw() {
        const nOpt = Math.sqrt(C / 120), dOpt = 20 * nOpt;
        ch.update({
          series: [
            { points: [[xd[0], 20 * xd[0]], [yd[1] / 20, yd[1]]], cls: "faint", label: "" },
            ...[1e15, 1e20, 1e25].map((c) => ({ points: isoLine(c), cls: "faint" })),
            { points: isoLine(C), cls: "accent" },
          ],
          dots: [
            ...MODELS.map((m) => ({ x: m.N, y: m.D, r: 5, cls: m.name === "ours" ? "" : "ink", label: m.name, dx: m.dx, dy: m.dy, anchor: m.anchor, tip: `${m.name}: ${human(m.N)} parameters, ${human(m.D)} tokens (${Math.round(m.D / m.N).toLocaleString("en-US")} per parameter)` })),
            { x: nOpt, y: dOpt, r: 7, cls: "", tip: `compute-optimal for ${fmt.sci(C, 1)} FLOPs: ${human(nOpt)} parameters, ${human(dOpt)} tokens` },
          ],
          after(svg, X, Y) {
            const ang = (-Math.atan2(Y(1) - Y(10), X(10) - X(1)) * 180) / Math.PI;
            svg.appendChild(s("text", { x: X(1e7), y: Y(2e8) - 6, class: "hl-label", transform: `rotate(${ang.toFixed(1)} ${X(1e7)} ${Y(2e8)})` }, "20 tokens per parameter"));
            [1e15, 1e20, 1e25].forEach((c) => {
              const n = Math.max(xd[0] * 1.6, c / (6 * yd[1] / 4)), d = c / (6 * n);
              if (d > yd[0] && d < yd[1] && n < xd[1]) svg.appendChild(s("text", { x: X(n) + 6, y: Y(d) + 3, class: "hl-label" }, `${fmt.sci(c, 0)} FLOPs`));
            });
          },
        });
        note.innerHTML = `Budget <b>${fmt.sci(C, 1)} FLOPs</b> (red line: every N, D with 6ND = C). The 20-tokens rule puts the best model at <b>${human(nOpt)}</b> parameters trained on <b>${human(dOpt)}</b> tokens.` + (Math.abs(Math.log10(C / 3.8e25)) < 0.05 ? " Meta's own fit for this budget: 402B on 16.55T, about 41 tokens per parameter." : "");
      }
      el.querySelector(".sc-ctl").append(logSlider({ id: "sc-c", label: "Compute budget", lo: 1e18, hi: 1e26, step: 0.05, value: C, format: (v) => `${fmt.sci(v, 1)} FLOPs`, onInput: (v) => { C = v; draw(); }, wide: true }));
      draw();
    },
  };

  // ------------------------------------------------------------ data (MinHash)
  W.data = {
    init(el) {
      const STAGES = ["crawl", "extract text", "language ID", "quality filter", "deduplicate", "safety & PII filter", "mix sources", "tokenize", "shard"];
      const pipe = el.querySelector(".pipe");
      STAGES.forEach((st, i) => {
        pipe.appendChild(h("span", { class: "pipe-stage" + (st === "deduplicate" ? " is-on" : "") }, st));
        if (i < STAGES.length - 1) pipe.appendChild(h("span", { class: "pipe-arrow", "aria-hidden": "true" }, "→"));
      });
      const A0 = "SELECT name FROM users WHERE id = 7;\nSELECT total FROM orders WHERE status = 3;\nSELECT title FROM products WHERE id = 4;\nSELECT email FROM users WHERE age = 5;";
      const B0 = A0.replace("id = 7", "id = 8");
      const docs = el.querySelector(".mh-docs");
      const ta = (id, label, val) => { const t = h("textarea", { class: "txt mh-ta", id, rows: "4", spellcheck: "false" }); t.value = val; docs.appendChild(h("div", { class: "ctl" }, h("label", { class: "ctl-label", for: id }, label), t)); return t; };
      const tA = ta("mh-a", "Document A", A0), tB = ta("mh-b", "Document B", B0);
      let k = 128;
      const sig = el.querySelector(".mh-sig"), out = el.querySelector(".mh-out");
      const fnv = (str) => { let x = 0x811c9dc5; for (let i = 0; i < str.length; i++) { x ^= str.charCodeAt(i); x = Math.imul(x, 0x01000193); } return x >>> 0; };
      const mix = (x, seed) => { let z = (x ^ Math.imul(seed + 0x9e3779b9, 0x85ebca6b)) >>> 0; z = Math.imul(z ^ (z >>> 16), 0x85ebca6b); z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35); return (z ^ (z >>> 16)) >>> 0; };
      const shingles = (text) => { const w = text.split(/\s+|(?=;)/).filter(Boolean); const set = new Set(); for (let i = 0; i + 2 < w.length; i++) set.add(w.slice(i, i + 3).join(" ")); return set; };
      const signature = (set) => { const hs = [...set].map(fnv); return Array.from({ length: k }, (_, i) => Math.min(...hs.map((x) => mix(x, i)))); };
      function draw() {
        const a = shingles(tA.value), b = shingles(tB.value);
        const inter = [...a].filter((x) => b.has(x)).length, uni = new Set([...a, ...b]).size;
        const J = uni ? inter / uni : 1;
        const sa = signature(a), sb = signature(b);
        const match = sa.map((x, i) => x === sb[i]);
        const est = match.filter(Boolean).length / k;
        clear(sig);
        sig.appendChild(h("div", { class: "mh-cells", role: "img", "aria-label": `${match.filter(Boolean).length} of ${k} signature slots match` },
          match.map((m, i) => h("span", { class: "mh-c" + (m ? " on" : ""), "data-tip": `hash ${i + 1}: ${m ? "same minimum" : "different minimum"}` }))));
        clear(out);
        out.append(stat(`${a.size} / ${b.size}`, "3-word shingles in A / B"), stat(J.toFixed(3), "exact Jaccard similarity"), stat(est.toFixed(3), `MinHash estimate, ${k} hashes`),
          stat(est >= 0.8 ? "near-duplicate" : "keep both", "at a 0.8 threshold"));
      }
      [tA, tB].forEach((t) => t.addEventListener("input", draw));
      el.querySelector(".mh-ctl").append(slider({ id: "mh-k", label: "Hash functions", min: 16, max: 256, step: 16, value: k, format: (v) => String(v), onInput: (v) => { k = v; draw(); } }),
        h("p", { class: "fig-cap" }, "Each cell is one hash function: filled when both documents have the same minimum hash. The share of filled cells estimates the Jaccard similarity."));
      draw();
    },
  };

  // ------------------------------------------------------------ memory
  W.memory = {
    init(el) {
      const PARTS = [["BF16 weights", 2, "s1"], ["BF16 gradients", 2, "s2"], ["FP32 master weights", 4, "s3"], ["Adam first moment", 4, "s4"], ["Adam second moment", 4, "s5"]];
      const PRESETS = [{ label: "Ours", N: 9968 }, { label: "8B", N: 8e9 }, { label: "70B", N: 70e9 }, { label: "405B", N: 405e9 }];
      let N = 70e9;
      const chartEl = el.querySelector(".mem-chart"), note = el.querySelector(".mem-note");
      el.querySelector(".mem-ctl").append(segmented({ label: "Model size", value: 2, options: PRESETS.map((p, i) => ({ value: i, label: p.label })), onChange: (i) => { N = PRESETS[i].N; draw(); } }));
      function draw() {
        clear(chartEl);
        const total = 16 * N, maxV = Math.max(total, H100.mem) * 1.08;
        const Wd = 560, Hh = 120, x0 = 10, x1 = Wd - 10, X = (v) => x0 + (v / maxV) * (x1 - x0);
        const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hh}`, class: "chart", role: "img", "aria-label": `Training memory for ${human(N)} parameters: ${fmt.bytes(total)}` });
        let acc = 0;
        PARTS.forEach(([lab, b, cls]) => {
          const w = Math.max(0.5, X(acc + b * N) - X(acc) - 2);
          svg.appendChild(s("rect", { x: X(acc), y: 30, width: w, height: 34, rx: 3, class: `mem-seg ${cls}`, "data-tip": `${lab}: ${b} bytes × ${human(N)} = ${fmt.bytes(b * N)}` }));
          acc += b * N;
        });
        svg.appendChild(s("line", { x1: X(H100.mem), x2: X(H100.mem), y1: 16, y2: 80, class: "vline cursor" }));
        svg.appendChild(s("text", { x: X(H100.mem) + (X(H100.mem) > Wd - 140 ? -6 : 6), y: 22, class: "hl-label", "text-anchor": X(H100.mem) > Wd - 140 ? "end" : "start" }, "one H100: 80 GB"));
        svg.appendChild(s("text", { x: x0, y: 98, class: "tick" }, "0"));
        svg.appendChild(s("text", { x: x1, y: 98, class: "tick", "text-anchor": "end" }, fmt.bytes(maxV)));
        chartEl.appendChild(svg);
        chartEl.appendChild(h("div", { class: "legend" }, PARTS.map(([lab, b, cls]) => h("span", { class: "key" }, h("span", { class: "sw", style: { background: `var(--${cls})` } }), `${lab} (${b} B)`))));
        const gpus = Math.ceil(total / H100.mem);
        note.innerHTML = N < 1e6
          ? `${human(N)} parameters × 16 bytes = <b>${fmt.bytes(total)}</b>: ${(total / H100.mem * 100).toExponential(1)}% of one H100.`
          : `${human(N)} parameters × 16 bytes = <b>${fmt.bytes(total)}</b> of model state. That needs at least <b>${gpus}</b> H100${gpus > 1 ? "s" : ""} before counting activations.`;
      }
      draw();
    },
  };

  // ------------------------------------------------------------ gpu (roofline)
  W.gpu = {
    init(el) {
      let B = 1;
      const d = 8192, ridge = H100.flops / H100.bw;
      const attain = (ai) => Math.min(H100.flops, ai * H100.bw) / 1e12;
      const chartEl = el.querySelector(".rf-chart"), note = el.querySelector(".rf-note");
      const ops = [
        { name: "elementwise add", ai: 1 / 6, tip: "1 FLOP per 6 bytes (two BF16 reads, one write)", dx: 8, dy: 18, anchor: "start" },
        { name: "softmax", ai: 1.25, tip: "≈5 FLOPs per element, 4 bytes moved", dx: -9, dy: -6, anchor: "end" },
        { name: "GELU", ai: 2.5, tip: "≈10 FLOPs per element, 4 bytes moved", dx: 8, dy: 18, anchor: "start" },
      ];
      const ch = chart(chartEl, {
        width: 560, height: 300, margin: { l: 56, r: 18, t: 14, b: 40 }, aria: "Roofline chart of an H100",
        x: { type: "log", domain: [0.1, 1e4], ticks: [0.1, 1, 10, 100, 1000, 1e4], fmt: (v) => (v < 1 ? String(v) : v.toLocaleString("en-US")), label: "arithmetic intensity (FLOPs per byte)" },
        y: { type: "log", domain: [0.1, 2000], ticks: [0.1, 1, 10, 100, 1000], fmt: (v) => String(v), label: "TFLOP/s" },
      });
      function draw() {
        const ai = (B * d) / (d + 2 * B);
        ch.update({
          series: [{ points: [[0.1, attain(0.1)], [ridge, attain(ridge)], [1e4, attain(1e4)]], cls: "thick-ink" }],
          vlines: [{ x: ridge, label: "ridge ≈ 295" }],
          dots: [
            ...ops.map((o) => ({ x: o.ai, y: attain(o.ai), r: 5, cls: "ink", label: o.name, dx: o.dx, dy: o.dy, anchor: o.anchor, tip: `${o.name}: ${o.tip} → at most ${attain(o.ai).toFixed(2)} TFLOP/s` })),
            { x: ai, y: attain(ai), r: 7, cls: "", label: `matmul, batch ${B}`, dx: ai > 300 ? -10 : 10, dy: ai > 300 ? 22 : 20, anchor: ai > 300 ? "end" : "start", tip: `[${B} × ${d}] × [${d} × ${d}] in BF16: ${ai.toFixed(1)} FLOPs/byte → ${attain(ai).toFixed(1)} TFLOP/s` },
          ],
        });
        const bound = ai < ridge ? "memory-bound" : "compute-bound";
        note.innerHTML = `Multiplying <b>${B}</b> token vector${B > 1 ? "s" : ""} by an 8,192 × 8,192 matrix: <b>${ai.toFixed(1)}</b> FLOPs per byte, so at most <b>${attain(ai).toFixed(1)}</b> of 989 TFLOP/s (${fmt.pct(attain(ai) / 989.4, 1)} of peak). It is <b>${bound}</b>.`;
      }
      el.querySelector(".rf-ctl").append(slider({ id: "rf-b", label: "Token vectors per matrix multiply (batch)", min: 0, max: 12, step: 1, value: 0, format: (v) => Math.pow(2, v).toLocaleString("en-US"), onInput: (v) => { B = Math.pow(2, v); draw(); }, wide: true }));
      draw();
    },
  };

  // ------------------------------------------------------------ flash
  W.flash = {
    init(el, deck) {
      const HEADS = W._HEADS;
      let head = 1, pos = 4, tile = 2, step = 0;
      const ctl = el.querySelector(".fa-ctl");
      W._headPicker(ctl, { value: head, onChange: (v) => { head = v; reset(); } });
      W._posPicker(ctl, { value: pos, label: "Asking from", onChange: (i) => { pos = i; reset(); } });
      const nextBtn = button("Next tile", () => { step++; draw(); }, { kind: "primary" });
      ctl.append(segmented({ label: "Tile size", value: tile, options: [1, 2, 3].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { tile = v; reset(); } }), nextBtn, button("Reset", () => reset()));
      const scoresEl = el.querySelector(".fa-scores"), stateEl = el.querySelector(".fa-state"), note = el.querySelector(".fa-note");
      function reset() { step = 0; draw(); }
      function draw() {
        const [l, hh] = HEADS[head];
        const L = trace().layers[l];
        const sc = L.heads[hh].scores[pos].slice(0, pos + 1);
        const vals = L.v.slice(0, pos + 1).map((v) => W._slice8(v, hh));
        const tiles = [];
        for (let i = 0; i < sc.length; i += tile) tiles.push([i, Math.min(sc.length, i + tile)]);
        step = Math.min(step, tiles.length);
        nextBtn.disabled = step >= tiles.length;
        // scores row, tiles marked
        clear(scoresEl);
        scoresEl.appendChild(h("div", { class: "fa-row" }, sc.map((v, j) => {
          const ti = Math.floor(j / tile);
          return h("div", { class: "fa-cell" + (ti < step ? " done" : ti === step ? " next" : ""), style: { "--t": String(ti % 2) } },
            chip(M.running[j]), h("span", { class: "num" }, fmt.minus(v.toFixed(2))), h("span", { class: "fa-t" }, `tile ${ti + 1}`));
        })));
        // run the online softmax over the processed tiles
        let m = -Infinity, ell = 0, o = new Array(8).fill(0);
        const rows = [];
        for (let t = 0; t < step; t++) {
          const [a, b] = tiles[t];
          const mNew = Math.max(m, ...sc.slice(a, b));
          const scale = m === -Infinity ? 0 : Math.exp(m - mNew);
          ell = ell * scale; o = o.map((x) => x * scale);
          for (let j = a; j < b; j++) { const e = Math.exp(sc[j] - mNew); ell += e; o = o.map((x, c) => x + e * vals[j][c]); }
          m = mNew;
          rows.push({ t, m, scale, ell, o: o.slice() });
        }
        clear(stateEl);
        const mAbs = Math.max(...vals.flat().map(Math.abs), 1e-9);
        stateEl.appendChild(h("div", { class: "fa-table" },
          h("div", { class: "fa-tr fa-th" }, h("span", null, "After"), h("span", { class: "r" }, "running max m"), h("span", { class: "r" }, "rescale e^(m_old − m)"), h("span", { class: "r" }, "running sum ℓ"), h("span", null, "o / ℓ (8 numbers)")),
          rows.map((r) => h("div", { class: "fa-tr" }, h("span", null, `tile ${r.t + 1}`), h("span", { class: "num r" }, fmt.minus(r.m.toFixed(2))),
            h("span", { class: "num r" }, r.t === 0 ? "—" : r.scale < 1e-3 ? r.scale.toExponential(1) : r.scale.toFixed(4)), h("span", { class: "num r" }, r.ell.toPrecision(4)),
            heatStrip(r.o.map((x) => x / r.ell), { maxAbs: mAbs, cell: "n8" })))));
        if (step >= tiles.length) {
          const a = M.softmax(sc);
          const exact = new Array(8).fill(0).map((_, c) => a.reduce((acc, w, j) => acc + w * vals[j][c], 0));
          const tiled = o.map((x) => x / ell);
          const diff = Math.max(...exact.map((x, c) => Math.abs(x - tiled[c])));
          stateEl.appendChild(h("div", { class: "vec-row fa-final" }, h("div", { class: "vec-label" }, "standard attention (full softmax)"), heatStrip(exact, { maxAbs: mAbs, cell: "n8" })));
          // with one tile there is no streaming to speak of, and a zero difference is "exactly"
          const match = diff === 0 ? "matches standard attention <b>exactly</b>" : `matches standard attention to <b>${diff.toExponential(1)}</b>, which is floating-point rounding`;
          note.innerHTML = tiles.length > 1
            ? `Done: the tiled result ${match}. No step ever held all ${sc.length} scores at once.`
            : `Done: one tile held all ${sc.length} score${sc.length > 1 ? "s" : ""}, so this is ordinary softmax; the result ${match}. Pick a later position or a smaller tile to see the running rescale.`;
        } else {
          note.innerHTML = step === 0 ? "Each tile of keys is processed once. Press <b>Next tile</b>." : `Processed ${step} of ${tiles.length} tiles. When a tile raises the maximum, everything so far is multiplied by e^(m_old − m_new) before adding the new terms.`;
        }
        deck.strip.set({ focus: [pos] });
      }
      this.draw = draw;
      reset();
    },
    enter() { if (this.draw) this.draw(); },
  };

  // ------------------------------------------------------------ precision
  const FORMATS = [
    { name: "FP32", E: 8, Mb: 23, ieee: true },
    { name: "BF16", E: 8, Mb: 7, ieee: true },
    { name: "FP16", E: 5, Mb: 10, ieee: true },
    { name: "FP8 E4M3", E: 4, Mb: 3, ieee: false, max: 448 },
    { name: "FP8 E5M2", E: 5, Mb: 2, ieee: true },
  ];
  // Round x to the nearest representable value (ties to even), with subnormals.
  function quantize(x, f) {
    const bias = Math.pow(2, f.E - 1) - 1;
    const maxFinite = f.max || (2 - Math.pow(2, -f.Mb)) * Math.pow(2, Math.pow(2, f.E) - 2 - bias);
    if (x === 0 || !isFinite(x)) return { v: x, flag: x === 0 ? "exact" : "special", bits: null };
    const sign = x < 0 ? 1 : 0, a = Math.abs(x);
    const eMin = 1 - bias;
    let e = Math.max(Math.floor(Math.log2(a)), eMin);
    let q = Math.pow(2, e - f.Mb);
    let n = a / q;
    let r = Math.round(n);
    if (Math.abs(n - Math.floor(n) - 0.5) < 1e-12) r = Math.floor(n) % 2 === 0 ? Math.floor(n) : Math.floor(n) + 1;
    let v = r * q;
    if (v > maxFinite) {
      return { v: f.ieee ? (sign ? -Infinity : Infinity) : NaN, flag: "overflow", bits: null, sign };
    }
    // field encoding
    let expField, mant;
    if (v === 0) { expField = 0; mant = 0; }
    else {
      let ev = Math.floor(Math.log2(v));
      if (ev < eMin) { expField = 0; mant = Math.round(v / Math.pow(2, eMin - f.Mb)); }
      else { expField = ev + bias; mant = Math.round(v / Math.pow(2, ev - f.Mb)) - Math.pow(2, f.Mb); }
    }
    const flag = v === 0 ? "underflow" : expField === 0 ? "subnormal" : v === a ? "exact" : "rounded";
    return { v: sign ? -v : v, flag, bits: { sign, expField, mant } };
  }
  W._quantize = quantize;

  W.precision = {
    init(el) {
      const wSample = M.weights().wu.data[17];
      const PRESETS = [
        { label: "a weight", v: wSample },
        { label: "1/3", v: 1 / 3 },
        { label: "gradient 2e-8", v: 2e-8 },
        { label: "activation 300", v: 300 },
        { label: "70,000", v: 70000 },
      ];
      let x = PRESETS[0].v;
      const ctl = el.querySelector(".fp-ctl"), rows = el.querySelector(".fp-rows");
      const input = h("input", { class: "txt", id: "fp-x", type: "text", inputmode: "decimal", value: String(x), "aria-label": "Number to encode" });
      input.addEventListener("input", () => { const v = parseFloat(input.value); if (isFinite(v)) { x = v; draw(); } });
      ctl.append(h("div", { class: "ctl fp-in" }, h("label", { class: "ctl-label", for: "fp-x" }, "Number"), input),
        segmented({ label: "Try", value: 0, options: PRESETS.map((p, i) => ({ value: i, label: p.label })), onChange: (i) => { x = PRESETS[i].v; input.value = String(x); draw(); } }));
      const FLAG = { exact: "exact", rounded: "rounded", subnormal: "subnormal (reduced precision)", underflow: "underflow: became 0", overflow: "overflow", special: "" };
      function draw() {
        clear(rows);
        FORMATS.forEach((f) => {
          const r = quantize(x, f);
          const bitsEl = h("div", { class: "fp-bits", role: "img", "aria-label": `${f.name}: 1 sign bit, ${f.E} exponent bits, ${f.Mb} mantissa bits` });
          const b = r.bits;
          const signBit = b ? b.sign : (r.sign || 0);
          bitsEl.appendChild(h("span", { class: "fb s" }, String(signBit)));
          const ex = b ? b.expField.toString(2).padStart(f.E, "0") : (f.ieee ? "1".repeat(f.E) : "1".repeat(f.E));
          const ma = b ? b.mant.toString(2).padStart(f.Mb, "0") : (f.ieee && r.flag === "overflow" ? "0".repeat(f.Mb) : "1".repeat(f.Mb));
          ex.split("").forEach((c) => bitsEl.appendChild(h("span", { class: "fb e" }, c)));
          const shown = f.Mb > 10 ? ma.slice(0, 10) : ma;
          shown.split("").forEach((c) => bitsEl.appendChild(h("span", { class: "fb m" }, c)));
          if (f.Mb > 10) bitsEl.appendChild(h("span", { class: "fb more" }, `+${f.Mb - 10}`));
          const rel = isFinite(r.v) && x !== 0 ? Math.abs(r.v - x) / Math.abs(x) : null;
          rows.appendChild(h("div", { class: "fp-row" + (r.flag === "overflow" || r.flag === "underflow" ? " is-bad" : "") },
            h("div", { class: "fp-name" }, f.name, h("span", { class: "fp-sub" }, `1 + ${f.E} + ${f.Mb} bits`)),
            bitsEl,
            h("div", { class: "fp-val" }, h("span", { class: "num" }, isNaN(r.v) ? "NaN" : !isFinite(r.v) ? (r.v > 0 ? "+∞" : "−∞") : fmt.minus(+r.v.toPrecision(8) + "")),
              h("span", { class: "fp-flag" }, `${FLAG[r.flag]}${rel !== null && r.flag !== "exact" && r.flag !== "underflow" ? ` · error ${(rel * 100).toPrecision(2)}%` : ""}`))));
        });
      }
      draw();
    },
  };

  // ------------------------------------------------------------ allreduce
  W.allreduce = {
    init(el) {
      let N = 4, phase, t, chunks, sent;
      const ringEl = el.querySelector(".ar-ring"), note = el.querySelector(".ar-note");
      const init = (g, c) => ((g * 3 + c * 2) % 7) + 1;
      function reset() {
        phase = "rs"; t = 0; sent = 0;
        chunks = Array.from({ length: N }, (_, g) => Array.from({ length: N }, (_, c) => ({ v: init(g, c), mask: 1 << g })));
        draw(null);
      }
      function stepOnce() {
        if (phase === "done") return null;
        const moves = [];
        const next = chunks.map((row) => row.map((x) => ({ v: x.v, mask: x.mask })));
        for (let g = 0; g < N; g++) {
          const to = (g + 1) % N;
          const c = phase === "rs" ? (((g - t) % N) + N) % N : (((g + 1 - t) % N) + N) % N;
          moves.push({ from: g, to, c });
          if (phase === "rs") { next[to][c].v += chunks[g][c].v; next[to][c].mask |= chunks[g][c].mask; }
          else { next[to][c] = { v: chunks[g][c].v, mask: chunks[g][c].mask }; }
        }
        chunks = next; sent++; t++;
        if (t === N - 1) { phase = phase === "rs" ? "ag" : "done"; t = 0; }
        return moves;
      }
      const full = (1 << N) - 1;
      function draw(moves) {
        clear(ringEl);
        const Wd = 460, Hd = 330, cx = Wd / 2, cy = Hd / 2 + 4, R = 112;
        const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "chart ar-svg", role: "img", "aria-label": `Ring all-reduce among ${N} GPUs` });
        svg.appendChild(s("defs", null, s("marker", { id: "ar-ah", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "ah-acc" }))));
        const pos = (g) => { const a = -Math.PI / 2 + (2 * Math.PI * g) / N; return [cx + R * Math.cos(a), cy + R * Math.sin(a)]; };
        svg.appendChild(s("circle", { cx, cy, r: R, class: "ar-ring-line" }));
        if (moves) moves.forEach((mv) => {
          const [x1, y1] = pos(mv.from), [x2, y2] = pos(mv.to);
          const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, k = 0.18;
          const qx = mx + (mx - cx) * k, qy = my + (my - cy) * k;
          svg.appendChild(s("path", { d: `M${x1},${y1} Q${qx},${qy} ${x2},${y2}`, class: "ar-move", "marker-end": "url(#ar-ah)", fill: "none" }));
          svg.appendChild(s("text", { x: qx + (qx - cx) * 0.12, y: qy + (qy - cy) * 0.12 + 4, class: "ar-mv-l", "text-anchor": "middle" }, `chunk ${mv.c + 1}`));
        });
        const cw = Math.min(26, 150 / N), chH = 24;
        for (let g = 0; g < N; g++) {
          const [x, y] = pos(g);
          const bw = cw * N + 10;
          svg.appendChild(s("rect", { x: x - bw / 2, y: y - 26, width: bw, height: 52, rx: 9, class: "ar-gpu" }));
          svg.appendChild(s("text", { x: x - bw / 2 + 6, y: y - 13, class: "ar-g-l" }, `GPU ${g + 1}`));
          chunks[g].forEach((ch, c) => {
            const contrib = ch.mask.toString(2).split("").filter((b) => b === "1").length;
            const xx = x - bw / 2 + 5 + c * cw;
            svg.appendChild(s("rect", { x: xx, y: y - 6, width: cw - 2, height: chH, rx: 3, style: `fill:${seq(contrib / N * 0.85)}`, class: "ar-chunk" + (ch.mask === full ? " full" : ""),
              "data-tip": `GPU ${g + 1}, chunk ${c + 1}: ${ch.v} (sum of ${contrib} GPU${contrib > 1 ? "s" : ""})` }));
            svg.appendChild(s("text", { x: xx + (cw - 2) / 2, y: y + 10, class: "ar-v", "text-anchor": "middle" }, String(ch.v)));
          });
        }
        ringEl.appendChild(svg);
        const total = Array.from({ length: N }, (_, c) => Array.from({ length: N }, (_, g) => init(g, c)).reduce((a, b) => a + b, 0));
        const phaseTxt = phase === "rs" ? `Reduce-scatter, step ${t + 1} of ${N - 1}` : phase === "ag" ? `All-gather, step ${t + 1} of ${N - 1}` : "Done";
        note.innerHTML = `${phase === "done" ? "<b>Done.</b> Every GPU holds the full sums" : `Next: <b>${phaseTxt}</b>`} (${total.join(", ")}). Each GPU has sent <b>${sent}</b> chunk${sent === 1 ? "" : "s"}, ${sent} × 1/${N} = ${(sent / N).toFixed(2)} of its gradient; the full run sends 2(N − 1)/N = ${(2 * (N - 1) / N).toFixed(2)}. Darker cells sum more GPUs.`;
      }
      const ctl = el.querySelector(".ar-ctl");
      const play = player(() => { const mv = stepOnce(); draw(mv); return phase !== "done"; }, { interval: 1100, onState: (on) => { playBtn.textContent = on ? "Pause" : "Play"; } });
      const playBtn = button("Play", () => { if (phase === "done") reset(); play.toggle(); });
      ctl.append(segmented({ label: "GPUs", value: N, options: [3, 4, 5, 6, 8].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { N = v; play.pause(); reset(); } }),
        button("Step", () => { const mv = stepOnce(); draw(mv); }, { kind: "primary" }), playBtn, button("Reset", () => { play.pause(); reset(); }));
      // α–β calculator
      let cN = 8, S = 16e9, beta = 450e9, alpha = 5e-6;
      const calcOut = el.querySelector(".ar-out");
      function calc() {
        // NVLink only connects the 8 GPUs inside one server; larger rings cross the network
        const crossesServers = cN > 8 && beta > 50e9;
        const b = crossesServers ? 50e9 : beta;
        const lat = 2 * (cN - 1) * alpha, bwT = (2 * (cN - 1) / cN) * S / b;
        clear(calcOut);
        calcOut.append(stat(fmt.time(lat), "latency part"), stat(fmt.time(bwT), "bandwidth part"), stat(fmt.time(lat + bwT), "total"), stat(fmt.bytes(2 * (cN - 1) / cN * S), "sent per GPU"));
        if (crossesServers) calcOut.appendChild(h("p", { class: "fig-cap ar-warn" }, "NVLink only links the 8 GPUs in one server, so a ring of more GPUs runs at InfiniBand speed (50 GB/s); the numbers above use that."));
      }
      el.querySelector(".ar-calc").append(
        segmented({ label: "Gradient", value: 1, options: [{ value: 0, label: "ours 39.9 KB" }, { value: 1, label: "8B · 16 GB" }, { value: 2, label: "70B · 140 GB" }], onChange: (v) => { S = [39872, 16e9, 140e9][v]; calc(); } }),
        segmented({ label: "Link", value: 0, options: [{ value: 0, label: "NVLink 450 GB/s" }, { value: 1, label: "InfiniBand 50 GB/s" }], onChange: (v) => { beta = v ? 50e9 : 450e9; calc(); } }),
        slider({ id: "ar-n", label: "GPUs in the ring", min: 1, max: 10, step: 1, value: 3, format: (v) => String(Math.pow(2, v)), onInput: (v) => { cN = Math.pow(2, v); calc(); } }),
        slider({ id: "ar-a", label: "Latency per step α", min: 1, max: 20, step: 1, value: 5, format: (v) => `${v} µs`, onInput: (v) => { alpha = v * 1e-6; calc(); } }));
      calc();
      this.play = play;
      reset();
    },
    leave() { if (this.play) this.play.pause(); },
  };

  // ------------------------------------------------------------ zero
  W.zero = {
    init(el) {
      const PRESETS = [{ label: "8B", N: 8e9 }, { label: "70B", N: 70e9 }, { label: "405B", N: 405e9 }];
      let P = 70e9, Nd = 64, stage = 3;
      const STAGES = [
        { k: 0, name: "Plain data parallel", f: (p, n) => [2 * p, 2 * p, 12 * p] },
        { k: 1, name: "ZeRO-1: shard optimizer", f: (p, n) => [2 * p, 2 * p, 12 * p / n] },
        { k: 2, name: "ZeRO-2: + shard gradients", f: (p, n) => [2 * p, 2 * p / n, 12 * p / n] },
        { k: 3, name: "ZeRO-3 / FSDP: + shard weights", f: (p, n) => [2 * p / n, 2 * p / n, 12 * p / n] },
      ];
      const chartEl = el.querySelector(".z-chart"), formula = el.querySelector(".z-formula");
      el.querySelector(".z-ctl").append(
        segmented({ label: "Model", value: 1, options: PRESETS.map((p, i) => ({ value: i, label: p.label })), onChange: (i) => { P = PRESETS[i].N; draw(); } }),
        slider({ id: "z-n", label: "Data-parallel GPUs N", min: 0, max: 10, step: 1, value: 6, format: (v) => String(Math.pow(2, v)), onInput: (v) => { Nd = Math.pow(2, v); draw(); } }));
      function draw() {
        clear(chartEl);
        const vals = STAGES.map((st) => st.f(P, Nd));
        const maxV = Math.max(...vals.map((v) => v[0] + v[1] + v[2]), H100.mem) * 1.05;
        const Wd = 560, rowH = 44, top = 18, lab = 0, x0 = 10, x1 = Wd - 12, Hh = top + STAGES.length * rowH + 10;
        const X = (v) => x0 + (v / maxV) * (x1 - x0);
        const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hh}`, class: "chart", role: "img", "aria-label": "Per-GPU memory for each ZeRO stage" });
        STAGES.forEach((st, i) => {
          const y = top + i * rowH;
          const [w, g, o] = vals[i];
          const tot = w + g + o;
          svg.appendChild(s("text", { x: x0, y: y + 10, class: "z-lab" + (st.k === stage ? " on" : "") }, `${st.name}: ${fmt.bytes(tot)} per GPU`));
          let acc = 0;
          [[w, "s1", "weights"], [g, "s2", "gradients"], [o, "s3", "optimizer state"]].forEach(([v, cls, nm]) => {
            if (v <= 0) return;
            svg.appendChild(s("rect", { x: X(acc), y: y + 16, width: Math.max(0.6, X(acc + v) - X(acc) - 1.5), height: 16, rx: 2, class: `mem-seg ${cls}`, "data-tip": `${nm}: ${fmt.bytes(v)}` }));
            acc += v;
          });
          if (tot > H100.mem) svg.appendChild(s("text", { x: Math.min(X(tot) + 6, x1 - 2), y: y + 29, class: "z-over", "text-anchor": X(tot) + 60 > x1 ? "end" : "start" }, "doesn't fit"));
        });
        svg.appendChild(s("line", { x1: X(H100.mem), x2: X(H100.mem), y1: top - 6, y2: Hh - 6, class: "vline cursor" }));
        svg.appendChild(s("text", { x: X(H100.mem) + 4, y: Hh - 2, class: "hl-label" }, "80 GB"));
        chartEl.appendChild(svg);
        chartEl.appendChild(h("div", { class: "legend" },
          h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--s1)" } }), "BF16 weights"),
          h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--s2)" } }), "BF16 gradients"),
          h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--s3)" } }), "FP32 optimizer state (master weights, 2 Adam moments)")));
        formula.innerHTML = `Per GPU, with Ψ = ${human(P)} parameters and N = ${Nd}: plain 16Ψ = ${fmt.bytes(16 * P)} · stage 1 4Ψ + 12Ψ/N = ${fmt.bytes(4 * P + 12 * P / Nd)} · stage 2 2Ψ + 14Ψ/N = ${fmt.bytes(2 * P + 14 * P / Nd)} · stage 3 16Ψ/N = <b>${fmt.bytes(16 * P / Nd)}</b>`;
      }
      draw();
    },
  };

  // ------------------------------------------------------------ tp
  W.tp = {
    init(el, deck) {
      const pos = 4;
      let deg = 2;
      const Wt = M.weights();
      const w1 = Wt["l0.w1"], w2 = Wt["l0.w2"];
      const ctl = el.querySelector(".tp-ctl"), diag = el.querySelector(".tp-diagram"), rows = el.querySelector(".tp-rows");
      ctl.append(segmented({ label: "Tensor-parallel GPUs", value: deg, options: [{ value: 2, label: "2" }, { value: 4, label: "4" }], onChange: (v) => { deg = v; draw(); } }));
      const GC = ["s1", "s2", "s3", "s4"];
      function draw() {
        const tr = trace();
        const x = tr.layers[0].h2[pos], full = tr.layers[0].mlpOut[pos];
        const per = 64 / deg;
        const partials = [];
        for (let g = 0; g < deg; g++) {
          const cols = Array.from({ length: per }, (_, i) => g * per + i);
          const pre = cols.map((c) => x.reduce((acc, xi, r) => acc + xi * w1.data[r * 64 + c], 0));
          const act = pre.map(M.gelu);
          const part = new Array(16).fill(0);
          cols.forEach((c, i) => { for (let o = 0; o < 16; o++) part[o] += act[i] * w2.data[c * 16 + o]; });
          partials.push(part);
        }
        const sum = new Array(16).fill(0).map((_, o) => partials.reduce((acc, p) => acc + p[o], 0));
        const diff = Math.max(...sum.map((v, o) => Math.abs(v - full[o])));
        // diagram: W1 split by columns, W2 split by rows
        clear(diag);
        const Wd = 520, Hd = 150;
        const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "chart", role: "img", "aria-label": `MLP matrices split across ${deg} GPUs` });
        svg.appendChild(s("text", { x: 20, y: 16, class: "hl-label" }, "W₁: 16 × 64, split by columns"));
        for (let g = 0; g < deg; g++) svg.appendChild(s("rect", { x: 20 + g * (200 / deg), y: 24, width: 200 / deg - 2, height: 50, rx: 3, class: `mem-seg ${GC[g]}` }));
        svg.appendChild(s("text", { x: 20, y: 92, class: "tick" }, "each GPU: its hidden units"));
        svg.appendChild(s("text", { x: 300, y: 16, class: "hl-label" }, "W₂: 64 × 16, split by rows"));
        for (let g = 0; g < deg; g++) svg.appendChild(s("rect", { x: 300, y: 24 + g * (110 / deg), width: 60, height: 110 / deg - 2, rx: 3, class: `mem-seg ${GC[g]}` }));
        svg.appendChild(s("text", { x: 372, y: 64, class: "tick" }, "→ partial outputs"));
        svg.appendChild(s("text", { x: 372, y: 80, class: "tick" }, "→ all-reduce (sum)"));
        diag.appendChild(svg);
        clear(rows);
        const m = Math.max(...full.map(Math.abs), ...partials.flat().map(Math.abs));
        partials.forEach((p, g) => rows.appendChild(h("div", { class: "vec-row" }, h("div", { class: "vec-label" }, h("span", { class: "sw", style: { background: `var(--${GC[g]})` } }), `GPU ${g + 1}: hidden units ${g * per + 1}–${(g + 1) * per}`), heatStrip(p, { maxAbs: m }))));
        rows.appendChild(h("div", { class: "tp-op" }, "↓ all-reduce: add the partial outputs"));
        rows.appendChild(h("div", { class: "vec-row" }, h("div", { class: "vec-label" }, h("b", null, "sum of partials")), heatStrip(sum, { maxAbs: m })));
        rows.appendChild(h("div", { class: "vec-row" }, h("div", { class: "vec-label" }, "single-GPU MLP output"), heatStrip(full, { maxAbs: m })));
        rows.appendChild(h("p", { class: "note-inline" }, `Largest difference between the two: ${diff.toExponential(1)}, which is floating-point rounding. Layer 1 MLP at `, chip(" WHERE"), "."));
        deck.strip.set({ focus: [pos] });
      }
      this.draw = draw;
      draw();
    },
    enter() { if (this.draw) this.draw(); },
  };

  // ------------------------------------------------------------ pp
  function pipeline(p, m, kind) {
    const doneF = Array.from({ length: p }, () => new Array(m).fill(-1));
    const doneB = Array.from({ length: p }, () => new Array(m).fill(-1));
    const nf = new Array(p).fill(0), nb = new Array(p).fill(0);
    const grid = Array.from({ length: p }, () => []);
    let peak = 0;
    for (let t = 0; t < 4 * (m + p) + 4 && nb.some((x) => x < m); t++) {
      const acts = [];
      for (let st = 0; st < p; st++) {
        const canF = nf[st] < m && (st === 0 || (doneF[st - 1][nf[st]] >= 0 && doneF[st - 1][nf[st]] < t));
        const mbB = nb[st];
        const canB = mbB < nf[st] && doneF[st][mbB] >= 0 && doneF[st][mbB] < t && (st === p - 1 || (doneB[st + 1][mbB] >= 0 && doneB[st + 1][mbB] < t));
        let act = null;
        if (kind === "gpipe") {
          if (canF) act = "F"; else if (canB && nf[st] === m) act = "B";
        } else {
          const inflight = nf[st] - nb[st], warm = p - st;
          if (canB && (inflight >= warm || nf[st] === m)) act = "B";
          else if (canF && inflight < warm) act = "F";
          else if (canB) act = "B";
        }
        acts.push(act);
      }
      acts.forEach((act, st) => {
        if (act === "F") { grid[st][t] = { k: "F", mb: nf[st] }; doneF[st][nf[st]] = t; nf[st]++; }
        else if (act === "B") { grid[st][t] = { k: "B", mb: nb[st] }; doneB[st][nb[st]] = t; nb[st]++; }
        else grid[st][t] = null;
        peak = Math.max(peak, nf[st] - nb[st]);
      });
    }
    const T = Math.max(...grid.map((r) => r.length));
    return { grid, T, peak };
  }
  W._pipeline = pipeline;

  W.pp = {
    init(el) {
      let p = 4, m = 8, kind = "1f1b";
      const gridEl = el.querySelector(".pp-grid"), out = el.querySelector(".pp-out");
      el.querySelector(".pp-ctl").append(
        segmented({ label: "Schedule", value: kind, options: [{ value: "gpipe", label: "GPipe" }, { value: "1f1b", label: "1F1B" }], onChange: (v) => { kind = v; draw(); } }),
        slider({ id: "pp-p", label: "Stages p", min: 2, max: 8, step: 1, value: p, format: String, onInput: (v) => { p = v; draw(); } }),
        slider({ id: "pp-m", label: "Micro-batches m", min: 1, max: 16, step: 1, value: m, format: String, onInput: (v) => { m = v; draw(); } }));
      function draw() {
        const { grid, T, peak } = pipeline(p, m, kind);
        clear(gridEl);
        const cw = Math.max(14, Math.min(28, 520 / T)), rh = 26, lab = 62;
        const vw = lab + T * cw + 4, vh = p * rh + 24;
        const svg = s("svg", { viewBox: `0 0 ${vw} ${vh}`, width: vw, height: vh, class: "pp-svg", role: "img", "aria-label": `${kind} schedule with ${p} stages and ${m} micro-batches` });
        for (let st = 0; st < p; st++) {
          svg.appendChild(s("text", { x: lab - 8, y: st * rh + rh * 0.65, class: "tick", "text-anchor": "end" }, `GPU ${st + 1}`));
          for (let t = 0; t < T; t++) {
            const c = grid[st][t];
            const x = lab + t * cw, y = st * rh;
            if (!c) { svg.appendChild(s("rect", { x: x + 1, y: y + 2, width: cw - 2, height: rh - 4, rx: 3, class: "pp-idle" })); continue; }
            svg.appendChild(s("rect", { x: x + 1, y: y + 2, width: cw - 2, height: rh - 4, rx: 3, class: c.k === "F" ? "pp-f" : "pp-b", "data-tip": `GPU ${st + 1}, step ${t + 1}: ${c.k === "F" ? "forward" : "backward"} of micro-batch ${c.mb + 1}` }));
            if (cw >= 18) svg.appendChild(s("text", { x: x + cw / 2, y: y + rh * 0.66, class: c.k === "F" ? "pp-t" : "pp-t inv", "text-anchor": "middle" }, String(c.mb + 1)));
          }
        }
        svg.appendChild(s("text", { x: lab, y: p * rh + 16, class: "tick" }, "time →"));
        gridEl.appendChild(svg);
        gridEl.appendChild(h("div", { class: "legend" },
          h("span", { class: "key" }, h("span", { class: "sw pp-f-sw" }), "forward (micro-batch number)"),
          h("span", { class: "key" }, h("span", { class: "sw pp-b-sw" }), "backward"),
          h("span", { class: "key" }, h("span", { class: "sw pp-idle-sw" }), "idle")));
        const bubble = (T - 2 * m) / T;
        clear(out);
        out.append(stat(fmt.pct(bubble, 1), "idle share of time"), stat(fmt.pct((p - 1) / (m + p - 1), 1), "(p − 1)/(m + p − 1)"), stat(String(peak), "most activations held by one GPU"));
      }
      draw();
    },
  };

  // ------------------------------------------------------------ mesh
  W.mesh = {
    init(el) {
      const view = el.querySelector(".mesh-view"), read = el.querySelector(".mesh-read");
      const PP = 16, TP = 8, DP = 128;
      const cell = 22, gap = 3, lab = 54, top = 30;
      const Wd = lab + PP * (cell + gap) + 30, Hd = top + TP * (cell + gap) + 64;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "chart mesh-svg", role: "img", "aria-label": "One model replica: 16 pipeline stages by 8 tensor-parallel GPUs; 128 replicas in total" });
      for (let k = 3; k >= 1; k--) svg.appendChild(s("rect", { x: lab - 4 + k * 5, y: top - 4 - k * 5, width: PP * (cell + gap) + 6, height: TP * (cell + gap) + 6, rx: 8, class: "mesh-shadow" }));
      svg.appendChild(s("rect", { x: lab - 4, y: top - 4, width: PP * (cell + gap) + 6, height: TP * (cell + gap) + 6, rx: 8, class: "mesh-frame" }));
      svg.appendChild(s("text", { x: lab, y: 14, class: "hl-label" }, "pipeline stages 1–16 (one server each) →"));
      svg.appendChild(s("text", { x: 12, y: top + (TP * (cell + gap)) / 2, class: "hl-label", transform: `rotate(-90 12 ${top + (TP * (cell + gap)) / 2})`, "text-anchor": "middle" }, "8 GPUs per server"));
      const cells = [];
      for (let ss = 0; ss < PP; ss++) for (let t = 0; t < TP; t++) {
        const r = s("rect", { x: lab + ss * (cell + gap), y: top + t * (cell + gap), width: cell, height: cell, rx: 4, class: "mesh-gpu", tabindex: "0" });
        r.addEventListener("mouseenter", () => pick(ss, t));
        r.addEventListener("focus", () => pick(ss, t));
        r.addEventListener("click", () => pick(ss, t));
        cells.push({ r, ss, t });
        svg.appendChild(r);
      }
      svg.appendChild(s("text", { x: lab, y: top + TP * (cell + gap) + 22, class: "hl-label" }, "one model replica: 16 × 8 = 128 GPUs"));
      svg.appendChild(s("text", { x: lab, y: top + TP * (cell + gap) + 40, class: "hl-label strong" }, "× 128 data-parallel replicas = 16,384 GPUs"));
      view.appendChild(svg);
      function pick(ss, t) {
        cells.forEach((c) => {
          c.r.classList.toggle("is-sel", c.ss === ss && c.t === t);
          c.r.classList.toggle("is-tp", c.ss === ss && c.t !== t);
          c.r.classList.toggle("is-pp", c.t === t && c.ss !== ss);
        });
        read.innerHTML = `<b>Server ${ss + 1}, GPU ${t + 1}</b>. Pipeline stage ${ss + 1} of 16: about 8 of the 126 layers. Within the server it holds 1/8 of every matrix in those layers and all-reduces with the other 7 GPUs (lighter) over NVLink in every layer. ${ss < 15 ? `It sends activations forward to stage ${ss + 2}` : "It is the last stage"}${ss > 0 ? ` and gradients back to stage ${ss}` : ""} (same row). The GPUs in the same position in the other 127 replicas form its data-parallel group: they shard its weights, gradients and optimizer state and reduce gradients together.`;
      }
      pick(5, 2);
      const table = el.querySelector(".mesh-table");
      const rowsData = [["8,192", 8, 1, 16, 64, "8,192", 430, "43%"], ["16,384", 8, 1, 16, 128, "8,192", 400, "41%"], ["16,384", 8, 16, 16, 8, "131,072", 380, "38%"]];
      table.append(h("thead", null, h("tr", null, ["GPUs", "TP", "CP", "PP", "DP", "Sequence", "TFLOP/s per GPU", "MFU"].map((x) => h("th", null, x)))),
        h("tbody", null, rowsData.map((r, i) => h("tr", { class: i === 1 ? "is-ours" : "" }, r.map((x) => h("td", { class: "num" }, String(x)))))));
      table.parentNode.after(h("p", { class: "fig-cap" }, "Llama 3 405B pretraining configurations. Source: Meta, The Llama 3 Herd of Models (2024), Table 4."));
    },
  };

  // ------------------------------------------------------------ failures
  W.failures = {
    init(el) {
      let Mh = 54 * 24 / 419, C = 0.5, R = 3;
      const chartEl = el.querySelector(".fx-chart"), out = el.querySelector(".fx-out");
      const ch = chart(chartEl, {
        width: 560, height: 240, margin: { l: 50, r: 16, t: 12, b: 40 }, aria: "Wasted time versus checkpoint interval",
        x: { type: "log", domain: [1, 600], ticks: [1, 3, 10, 30, 100, 300], fmt: (v) => `${v} min`, label: "checkpoint every T minutes" },
        y: { domain: [0, 0.6], ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fmt: (v) => fmt.pct(v) },
      });
      function draw() {
        const Mm = Mh * 60;
        const waste = (T) => C / T + (T / 2 + R) / Mm;
        const pts = [];
        for (let e = 0; e <= Math.log10(600) + 1e-9; e += 0.02) { const T = Math.pow(10, e); pts.push([T, Math.min(waste(T), 5)]); }
        const Topt = Math.sqrt(2 * C * Mm);
        ch.update({ series: [{ points: pts, cls: "accent" }], dots: [{ x: Topt, y: Math.min(waste(Topt), 0.6), r: 6, cls: "ink", label: `√(2CM) = ${Topt.toFixed(1)} min`, dx: 10, dy: -8 }] });
        clear(out);
        out.append(stat(`${Topt.toFixed(1)} min`, "best interval √(2CM)"), stat(fmt.pct(waste(Topt), 1), "time wasted at best"), stat(fmt.pct(1 - waste(Topt), 1), "effective training time"));
      }
      el.querySelector(".fx-ctl").append(
        slider({ id: "fx-m", label: "Mean time between failures M", min: 0.5, max: 24, step: 0.1, value: +Mh.toFixed(1), format: (v) => `${v.toFixed(1)} h`, onInput: (v) => { Mh = v; draw(); } }),
        slider({ id: "fx-c", label: "Checkpoint write time C", min: 0.1, max: 10, step: 0.1, value: C, format: (v) => `${v.toFixed(1)} min`, onInput: (v) => { C = v; draw(); } }),
        slider({ id: "fx-r", label: "Restart time R", min: 0, max: 60, step: 1, value: R, format: (v) => `${v} min`, onInput: (v) => { R = v; draw(); } }));
      draw();
    },
  };
})();

/* Mixture of experts: a toy router, expert placement, all-to-all and capacity. */
(function () {
  "use strict";
  const { h, clear, chip, slider, segmented, fmt } = window.UI;
  const M = window.TinyLM;
  const W = window.Widgets;
  const stat = (v, l) => h("div", { class: "stat" }, h("div", { class: "stat-v" }, v), h("div", { class: "stat-l" }, l));
  W.moe = {
    init(el) {
      const QUERIES = [M.data.running, "SELECT total FROM orders WHERE status = 3;", "SELECT title FROM products WHERE id = 4;", "SELECT email FROM users WHERE age = 5;"];
      const tokens = QUERIES.flatMap((q, g) => M.encode(q).slice(0, 8).map((t) => ({ t, src: g })));
      const E = 8, GPUS = 4;
      const rand = M.mulberry32(17);
      const noise = tokens.map(() => Array.from({ length: E }, () => rand()));
      let k = 2, skew = 0, cap = 0;
      const view = el.querySelector(".moe-view"), out = el.querySelector(".moe-out");
      el.querySelector(".moe-ctl").append(
        segmented({ label: "Experts per token (k)", value: k, options: [{ value: 1, label: "1" }, { value: 2, label: "2" }], onChange: (v) => { k = v; draw(); } }),
        slider({ id: "moe-skew", label: "Router bias toward low-numbered experts (linear ramp)", min: 0, max: 3, step: 0.1, value: skew, format: (v) => v.toFixed(1), onInput: (v) => { skew = v; draw(); } }),
        segmented({ label: "Expert capacity", value: cap, options: [{ value: 0, label: "no limit" }, { value: 1.25, label: "1.25 × average" }, { value: 1, label: "1.0 × average" }], onChange: (v) => { cap = v; draw(); } }));
      function draw() {
        const bias = Array.from({ length: E }, (_, e) => (E - 1 - e) / (E - 1));
        const limit = cap ? Math.ceil((cap * tokens.length * k) / E) : Infinity;
        const bins = Array.from({ length: E }, () => []);
        const dropped = [];
        let crossing = 0;
        tokens.forEach((tok, j) => {
          const scores = noise[j].map((n, e) => n + skew * bias[e]);
          const order = scores.map((v, e) => [v, e]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, e]) => e);
          order.forEach((e) => {
            if (bins[e].length >= limit) { dropped.push({ ...tok, e }); return; }
            bins[e].push(tok);
            if (Math.floor(e / 2) !== tok.src) crossing++;
          });
        });
        clear(view);
        const grid = h("div", { class: "moe-grid" });
        for (let g = 0; g < GPUS; g++) {
          const col = h("div", { class: "moe-gpu" }, h("div", { class: "moe-gpu-h" }, `GPU ${g + 1}`, h("span", { class: "moe-load" }, `${bins[2 * g].length + bins[2 * g + 1].length} tokens`)));
          for (let e = 2 * g; e < 2 * g + 2; e++) {
            col.appendChild(h("div", { class: "moe-exp" }, h("div", { class: "moe-exp-h" }, `expert ${e + 1}`, h("span", { class: "num" }, String(bins[e].length))),
              h("div", { class: "moe-toks" }, bins[e].map((tok) => chip(tok.t, { cls: `src${tok.src}` })))));
          }
          grid.appendChild(col);
        }
        view.appendChild(grid);
        if (dropped.length) view.appendChild(h("div", { class: "moe-drop" }, h("span", { class: "fig-sub" }, `dropped (over capacity of ${limit}): `), dropped.map((tok) => chip(tok.t, { cls: `src${tok.src} is-dropped` }))));
        view.appendChild(h("div", { class: "legend" }, QUERIES.map((q, g) => h("span", { class: "key" }, h("span", { class: `sw src-sw${g}` }), `from GPU ${g + 1}: ${q.split(" ")[1]} …`))));
        const loads = Array.from({ length: GPUS }, (_, g) => bins[2 * g].length + bins[2 * g + 1].length);
        const mean = loads.reduce((a, b) => a + b, 0) / GPUS;
        clear(out);
        out.append(
          stat(`${(Math.max(...loads) / mean).toFixed(2)}×`, "busiest GPU vs. average"),
          stat(`${crossing} of ${tokens.length * k - dropped.length}`, "token copies sent to another GPU (each way)"),
          stat(String(dropped.length), "token copies dropped"),
          stat(`${k} of ${E}`, "experts' work per token"));
      }
      draw();
    },
  };
})();
