/* Part 7 — Training at scale: compute budget, memory of a step, number formats,
 * collective communication, data and model parallelism, running a job.
 * Every computed number is rechecked in tools/verify/part07.mjs. */
(function () {
  "use strict";
  const { h, s, clear, fmt, slider, segmented, button, chart } = window.UI;
  const M = window.TinyLM;
  const GiB = 2 ** 30;
  const H100 = { mem: 80 * GiB, bf16: 989e12 };
  const sup = (n) => String(n).replace(/-/g, "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
  const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, l));
  const sci = (v, d = 2) => fmt.sci(v, d);
  const bytes = (v) => fmt.bytes(v);
  const human = (n) => (n >= 1e12 ? `${+(n / 1e12).toFixed(n >= 1e13 ? 0 : 1)}T` : n >= 1e9 ? `${+(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B` : n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${+(n / 1e3).toFixed(1)}K` : String(Math.round(n)));

  // Rounding helpers shared by the number-format figures.
  const f32b = new Float32Array(1), u32b = new Uint32Array(f32b.buffer);
  // round a double to the nearest BF16 value (ties to even); inputs here are finite and in range
  const bf16 = (x) => { f32b[0] = x; const u = u32b[0]; u32b[0] = ((u + 0x7fff + ((u >>> 16) & 1)) >>> 0) & 0xffff0000; return f32b[0]; };
  // round |y| <= 448 to the nearest FP8 E4M3 value (ties to even, subnormals down to 2^-9)
  function e4m3(y) {
    if (y === 0) return 0;
    const a = Math.abs(y), e = Math.max(Math.floor(Math.log2(a)), -6), q = Math.pow(2, e - 3);
    const n = a / q, fl = Math.floor(n), fr = n - fl;
    const r = fr > 0.5 ? fl + 1 : fr < 0.5 ? fl : (fl % 2 === 0 ? fl : fl + 1);
    return Math.sign(y) * Math.min(448, r * q);
  }

  // ---------------------------------------------------------------- Fig: running sums in FP32 and BF16
  Deck.fig("scale-sum", (el) => {
    let v = 0.01, n = 1000;
    const host = h("div", { class: "scale-sum-chart" });
    const tiles = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const ctl = h("div", { class: "f-ctl" },
      segmented({ label: "Value added", value: v, options: [{ label: "0.1", value: 0.1 }, { label: "0.01", value: 0.01 }, { label: "0.001", value: 0.001 }], onChange: (x) => { v = x; draw(); } }),
      segmented({ label: "Terms n", value: n, options: [100, 1000, 10000, 100000].map((k) => ({ label: k.toLocaleString("en-US"), value: k })), onChange: (x) => { n = x; draw(); } }));
    el.append(ctl, host, tiles, expl);
    function draw() {
      const xb = bf16(v);
      let a32 = 0, a16 = 0, stall = null;
      const p32 = [[0, 0]], p16 = [[0, 0]];
      const every = Math.max(1, Math.round(n / 240));
      for (let i = 1; i <= n; i++) {
        a32 = Math.fround(a32 + xb);
        const nx = bf16(a16 + xb);
        if (nx === a16 && stall === null) stall = i;
        a16 = nx;
        if (i % every === 0 || i === n) { p32.push([i, a32]); p16.push([i, a16]); }
      }
      const exact = n * v, top = Math.max(exact, a32, a16) * 1.08;
      clear(host);
      chart(host, {
        width: 560, height: 250, margin: { l: 56, r: 16, t: 12, b: 38 }, aria: "Running sum against the number of terms in three accumulators",
        x: { domain: [0, n], ticks: window.UI.ticks([0, n], 4), label: "terms added", fmt: (t) => t.toLocaleString("en-US") },
        y: { domain: [0, top], ticks: window.UI.ticks([0, top], 4), label: "running sum" },
        series: [{ points: [[0, 0], [n, exact]], cls: "ink dashed" }, { points: p32, cls: "s1" }, { points: p16, cls: "s2" }],
      });
      host.appendChild(h("div", { class: "legend" },
        h("span", { class: "key" }, h("span", { class: "sw sw-ink" }), "exact n × value"),
        h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "FP32 accumulator"),
        h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "BF16 accumulator")));
      const p = (x) => (+x.toPrecision(7)).toLocaleString("en-US", { maximumFractionDigits: 6 });
      clear(tiles);
      tiles.append(tile(p(exact), "exact sum"), tile(p(a32), "FP32 accumulator"), tile(p(a16), "BF16 accumulator"));
      const eIn = (xb - v) / v, eAcc = (a32 - n * xb) / (n * xb), sg = (e) => `${e < 0 ? "−" : "+"}${(Math.abs(e) * 100).toFixed(3)}%`;
      expl.innerHTML = `Each term is ${v} rounded to BF16, ${xb.toPrecision(6)}. The FP32 sum is off by ${sg((a32 - exact) / exact)}: ${sg(eIn)} from rounding the input and ${sg(eAcc)} from FP32 accumulation. ` +
        (stall ? `The BF16 sum stops changing at term ${stall.toLocaleString("en-US")}, at ${p(a16)}: there half the spacing between BF16 neighbors exceeds the term.` : `The BF16 sum has not stalled yet; its error is ${fmt.pct(Math.abs(a16 - exact) / exact, 2)}.`);
    }
    draw();
  });

  // ---------------------------------------------------------------- Fig: FP8 E4M3 with one scale per tensor or per 128-block
  Deck.fig("scale-fp8", (el) => {
    const rnd = M.mulberry32(11);
    const base = Array.from({ length: 256 }, () => { const u = 1 - rnd(), w = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w); });
    let out = 10000, mode = "tensor";
    const host = h("div", { class: "scroll-x" });
    const tiles = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.append(h("div", { class: "f-ctl sliders" },
      segmented({ label: "Scale factors", value: mode, options: [{ label: "one per tensor", value: "tensor" }, { label: "one per 128 values", value: "block" }], onChange: (x) => { mode = x; draw(); } }),
      slider({ label: "Outlier at index 5", min: 0, max: 5, step: 0.5, value: 4, format: (x) => Math.round(Math.pow(10, x)).toLocaleString("en-US"), onInput: (x) => { out = Math.pow(10, x); draw(); } })), host, tiles, expl);
    function draw() {
      const x = base.slice(); x[5] = out;
      const blocks = [x.slice(0, 128), x.slice(128)];
      const amax = (a) => Math.max(...a.map(Math.abs));
      const scales = mode === "tensor" ? [448 / amax(x), 448 / amax(x)] : blocks.map((b) => 448 / amax(b));
      const xq = x.map((v, i) => e4m3(v * scales[i >> 7]) / scales[i >> 7]);
      const err = (lo, hi) => { let se = 0, k = 0; for (let i = lo; i < hi; i++) if (i !== 5) { se += Math.abs(xq[i] - x[i]) / Math.abs(x[i]); k++; } return se / k; };
      const zeros = (lo, hi) => { let z = 0; for (let i = lo; i < hi; i++) if (x[i] !== 0 && xq[i] === 0) z++; return z; };
      const W = 600, Hh = 180, l = 46, r = 8, t = 22, b = 30, X = (i) => l + ((i + 0.5) / 256) * (W - l - r);
      const Y = (e) => t + (1 - (Math.log10(Math.max(e, 1e-4)) + 4) / 4) * (Hh - t - b);
      const svg = s("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "scale-fp8-svg", role: "img", "aria-label": "Relative rounding error of each of 256 values after FP8 quantization" });
      [1, 0.1, 0.01, 0.001, 0.0001].forEach((g) => {
        svg.appendChild(s("line", { x1: l, x2: W - r, y1: Y(g), y2: Y(g), class: "scale-grid" }));
        svg.appendChild(s("text", { x: l - 6, y: Y(g) + 4, class: "scale-tick", "text-anchor": "end" }, g >= 0.01 ? String(g) : `10${sup(Math.round(Math.log10(g)))}`));
      });
      svg.appendChild(s("line", { x1: X(127.5), x2: X(127.5), y1: t, y2: Hh - b, class: "scale-div" }));
      svg.appendChild(s("text", { x: X(64), y: Hh - 10, class: "scale-tick", "text-anchor": "middle" }, "block 1: values 0–127 (outlier)"));
      svg.appendChild(s("text", { x: X(192), y: Hh - 10, class: "scale-tick", "text-anchor": "middle" }, "block 2: values 128–255"));
      x.forEach((v, i) => {
        if (i === 5) return;
        const e = Math.abs(xq[i] - v) / Math.abs(v);
        svg.appendChild(s("circle", { cx: X(i), cy: Y(e), r: 2.2, class: xq[i] === 0 ? "scale-pt bad" : "scale-pt", "data-tip": `value ${i}: ${v.toFixed(4)} → ${xq[i].toPrecision(4)} (error ${fmt.pct(Math.min(e, 1), 1)})` }));
      });
      svg.appendChild(s("text", { x: l, y: 12, class: "scale-tick" }, "relative error of each value (log scale)"));
      clear(host); host.appendChild(svg);
      clear(tiles);
      tiles.append(tile(scales[0].toPrecision(3) + (mode === "block" ? ` · ${scales[1].toPrecision(3)}` : ""), mode === "block" ? "scales, block 1 · 2" : "scale for all 256"),
        tile(fmt.pct(err(0, 128), 1), "mean error, block 1"), tile(fmt.pct(err(128, 256), 1), "mean error, block 2"), tile(String(zeros(0, 256)), "values flushed to 0"));
      expl.innerHTML = `Each value is multiplied by its scale, rounded to E4M3 (largest 448, smallest step 2⁻⁹ ≈ 0.002), and divided back. The plot shows each value's relative error on a log scale; red points became 0. The tiles average the relative errors, leaving out the outlier itself.`;
    }
    draw();
  });

  // ---------------------------------------------------------------- Fig: the six collectives on 4 ranks, before and after
  Deck.fig("scale-coll", (el) => {
    const N = 4, init = (r, c) => 10 * (r + 1) + (c + 1);
    const sum = (c) => Array.from({ length: N }, (_, r) => init(r, c)).reduce((a, b) => a + b, 0);
    const OPS = [
      { k: "bc", label: "broadcast", before: (r, c) => (r === 0 ? init(0, c) : null), after: (r, c) => init(0, c),
        note: "Rank 0, the root, copies its buffer to every rank. Each other rank must receive all S bytes." },
      { k: "re", label: "reduce", before: init, after: (r, c) => (r === 0 ? sum(c) : init(r, c)), stale: (r) => r !== 0,
        note: "The elementwise sum of the four buffers lands on rank 0; the other buffers are unchanged (faded)." },
      { k: "ar", label: "all-reduce", before: init, after: (r, c) => sum(c),
        note: "Every rank ends with the elementwise sum. A bandwidth-optimal algorithm sends 2(N − 1)/N · S bytes per rank: here 1.5 buffers." },
      { k: "rs", label: "reduce-scatter", before: init, after: (r, c) => (c === r ? sum(c) : null),
        note: "Rank i ends with chunk i of the sum, S/N bytes. Each rank sends (N − 1)/N · S bytes." },
      { k: "ag", label: "all-gather", before: (r, c) => (c === r ? init(r, c) : null), after: (r, c) => init(c, c),
        note: "Rank i contributes chunk i and every rank ends with all four chunks. Each rank receives (N − 1)/N · S bytes." },
      { k: "a2a", label: "all-to-all", before: init, after: (r, c) => init(c, r),
        note: "Rank i sends its chunk j to rank j: the chunks are transposed across ranks. Each rank sends (N − 1)/N · S bytes." },
    ];
    let op = OPS[2];
    const host = h("div", { class: "scroll-x" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.append(h("div", { class: "f-ctl" }, segmented({ label: "Collective", value: op.k, options: OPS.map((o) => ({ label: o.label, value: o.k })), onChange: (k) => { op = OPS.find((o) => o.k === k); draw(); } })), host, expl);
    function draw() {
      const W = 600, cw = 48, ch = 28, gap = 4, top = 40, lab = 52, x0 = [lab, 340];
      const Hh = top + N * (ch + gap) + 8;
      const svg = s("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "scale-coll-svg", role: "img", "aria-label": `${op.label} on 4 ranks: buffers before and after` });
      svg.appendChild(s("defs", null, s("marker", { id: "scale-coll-ah", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "scale-ah" }))));
      ["before", "after"].forEach((t, p) => {
        svg.appendChild(s("text", { x: x0[p], y: 14, class: "scale-lab" }, t));
        for (let c = 0; c < N; c++) svg.appendChild(s("text", { x: x0[p] + c * (cw + gap) + cw / 2, y: top - 4, class: "scale-tick", "text-anchor": "middle" }, `chunk ${c + 1}`));
      });
      for (let r = 0; r < N; r++) {
        const y = top + r * (ch + gap);
        svg.appendChild(s("text", { x: lab - 8, y: y + ch * 0.66, class: "scale-tick", "text-anchor": "end" }, `rank ${r}`));
        [op.before, op.after].forEach((f, p) => {
          for (let c = 0; c < N; c++) {
            const v = f(r, c), isSum = p === 1 && v !== null && v > 50;
            const cls = "scale-cell" + (v === null ? " empty" : ` c${c + 1}`) + (isSum ? " sum" : "") + (p === 1 && op.stale && op.stale(r) ? " stale" : "");
            const g = s("g", { class: cls, "data-tip": v === null ? `rank ${r}, chunk ${c + 1}: no data` : `rank ${r}, chunk ${c + 1}: ${v}` });
            const x = x0[p] + c * (cw + gap);
            g.appendChild(s("rect", { x, y, width: cw, height: ch, rx: 2 }));
            if (v !== null) g.appendChild(s("text", { x: x + cw / 2, y: y + ch * 0.68, "text-anchor": "middle" }, String(v)));
            svg.appendChild(g);
          }
        });
      }
      const ay = top + (N * (ch + gap)) / 2;
      svg.appendChild(s("path", { d: `M${x0[0] + N * (cw + gap) + 10},${ay} L${x0[1] - 14},${ay}`, class: "scale-arrow", "marker-end": "url(#scale-coll-ah)" }));
      clear(host); host.appendChild(svg);
      expl.textContent = op.note;
    }
    draw();
  });

  // ---------------------------------------------------------------- Fig: ZeRO stages, memory per GPU
  Deck.fig("scale-zero", (el) => {
    const MODELS = [{ label: "7.5B", P: 7.5e9 }, { label: "8B", P: 8e9 }, { label: "70B", P: 70e9 }, { label: "405B", P: 405e9 }];
    let mi = 0, lg = 6;
    const host = h("div", { class: "scroll-x" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.append(h("div", { class: "f-ctl sliders" },
      segmented({ label: "Model Ψ", value: mi, options: MODELS.map((m, i) => ({ label: m.label, value: i })), onChange: (i) => { mi = i; draw(); } }),
      slider({ label: "Data-parallel ranks", min: 0, max: 10, step: 1, value: lg, format: (x) => Math.pow(2, x).toLocaleString("en-US"), onInput: (x) => { lg = x; draw(); } })), host,
      h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "BF16 weights"), h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "BF16 gradients"), h("span", { class: "key" }, h("span", { class: "sw sw-s3" }), "FP32 master + Adam moments")),
      expl);
    function draw() {
      const P = MODELS[mi].P, n = Math.pow(2, lg);
      const rows = [
        { name: "plain data parallel", v: [2 * P, 2 * P, 12 * P], comm: "2Ψ" },
        { name: "ZeRO-1: optimizer state sharded", v: [2 * P, 2 * P, 12 * P / n], comm: "2Ψ" },
        { name: "ZeRO-2: + gradients sharded", v: [2 * P, 2 * P / n, 12 * P / n], comm: "2Ψ" },
        { name: "ZeRO-3 / FSDP: + weights sharded", v: [2 * P / n, 2 * P / n, 12 * P / n], comm: "3Ψ" },
      ];
      const W = 600, l = 8, r = 96, rh = 46, top = 8, Hh = top + rows.length * rh + 22, axis = 2 * H100.mem;
      const X = (b) => l + (Math.min(b, axis) / axis) * (W - l - r);
      const svg = s("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "scale-zero-svg", role: "img", "aria-label": "Model state per GPU for each ZeRO stage" });
      rows.forEach((row, i) => {
        const y = top + i * rh, tot = row.v[0] + row.v[1] + row.v[2];
        svg.appendChild(s("text", { x: l, y: y + 12, class: "scale-lab" }, `${row.name} · moves ${row.comm} values per step`));
        let acc = 0;
        row.v.forEach((b, j) => {
          const x1 = X(acc), x2 = X(acc + b);
          if (x2 - x1 > 0.3) svg.appendChild(s("rect", { x: x1, y: y + 18, width: Math.max(0.6, x2 - x1 - 1), height: 18, class: `scale-seg s${j + 1}`, "data-tip": `${["weights", "gradients", "FP32 optimizer state"][j]}: ${bytes(b)}` }));
          acc += b;
        });
        if (tot > axis) svg.appendChild(s("path", { d: `M${X(axis) - 6},${y + 14} l6,13 l-6,13`, class: "scale-over" }));
        svg.appendChild(s("text", { x: Math.min(X(tot), W - r) + 8, y: y + 32, class: "scale-val" + (tot > H100.mem ? " bad" : "") }, bytes(tot)));
      });
      rows.forEach((row, i) => svg.appendChild(s("line", { x1: X(H100.mem), x2: X(H100.mem), y1: top + i * rh + 15, y2: top + i * rh + 39, class: "scale-cap" })));
      svg.appendChild(s("text", { x: X(H100.mem) + 4, y: Hh - 4, class: "scale-capl" }, "80 GiB = 85.9 GB"));
      clear(host); host.appendChild(svg);
      const t = rows.map((rw) => bytes(rw.v[0] + rw.v[1] + rw.v[2]));
      expl.innerHTML = `Ψ = ${MODELS[mi].label} parameters, N<sub>d</sub> = ${n.toLocaleString("en-US")}: 16Ψ = ${t[0]} · 4Ψ + 12Ψ/N<sub>d</sub> = ${t[1]} · 2Ψ + 14Ψ/N<sub>d</sub> = ${t[2]} · 16Ψ/N<sub>d</sub> = <b>${t[3]}</b> per GPU. Bars past twice the H100's capacity are cut off. Activations and buffers come on top.`;
    }
    draw();
  });

  // ---------------------------------------------------------------- Fig: a parallel layout for Llama 3.1 405B
  Deck.fig("scale-layout", (el) => {
    const MD = { N: 405e9, L: 126, h: 16384, P: 3187671040 };
    const B = 16777216, CAP = 79.32 * GiB, F = 400e12;
    const st = { G: 16384, tp: 8, pp: 16, cp: 1, seq: 8192, v: 1 };
    const seg = (label, key, opts, f) => segmented({ label, value: st[key], options: opts.map((o) => ({ label: f ? f(o) : String(o), value: o })), onChange: (x) => { st[key] = x; draw(); } });
    const host = h("div", { class: "scroll-x" }), tiles = h("div", { class: "f-tiles" }), msg = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.append(h("div", { class: "f-ctl" },
      seg("GPUs", "G", [2048, 4096, 8192, 16384], (o) => o.toLocaleString("en-US")), seg("TP", "tp", [1, 2, 4, 8, 16]), seg("PP", "pp", [1, 2, 4, 8, 16, 32]),
      seg("CP", "cp", [1, 2, 4, 8, 16]), seg("Sequence", "seq", [8192, 131072], (o) => o.toLocaleString("en-US")), seg("Interleave v", "v", [1, 2, 4, 8])), host, tiles, msg);
    function draw() {
      const { G, tp, pp, cp, seq } = st;
      const mp = tp * pp * cp, dp = G / mp, seqs = B / seq;
      clear(tiles); clear(host);
      if (dp < 1) {
        tiles.append(tile("—", "data-parallel groups"));
        msg.innerHTML = `<b>Not a layout.</b> TP × CP × PP = ${mp.toLocaleString("en-US")} GPUs per model replica, more than the ${G.toLocaleString("en-US")} in the cluster.`;
        return;
      }
      if (seqs / dp < 1) {
        tiles.append(tile(dp.toLocaleString("en-US"), "data-parallel groups"));
        msg.innerHTML = `<b>Batch too small.</b> A 16M-token step holds ${seqs.toLocaleString("en-US")} sequences of ${seq.toLocaleString("en-US")} tokens, fewer than the ${dp.toLocaleString("en-US")} data-parallel groups. Raise TP, CP or PP, or use fewer GPUs.`;
        return;
      }
      const m = seqs / dp, lps = Math.ceil(MD.L / pp), v = pp === 1 ? 1 : Math.min(st.v, lps);
      const x = pp === 1 ? 0 : (pp - 1) / (v * m), bubble = x / (1 + x);
      const state = 16 * MD.N / G;
      const act = 34 * (seq / cp) * MD.h / tp * lps * Math.min(pp, m);
      const gath = lps * 2 * MD.P / tp;
      const total = state + act + gath;
      const link = tp <= 8 ? 450e9 : 50e9;
      const tpComm = tp === 1 ? 0 : (4 * 2 * (tp - 1) / tp * 2 * (seq / cp) * MD.h) / link;
      const tpComp = (6 * MD.P * (seq / cp)) / tp / F;
      tiles.append(tile(dp.toLocaleString("en-US"), "data-parallel groups"), tile(m.toLocaleString("en-US"), "micro-batches per pipeline per step"),
        tile(fmt.pct(bubble, 1), "pipeline bubble"), tile(`${(total / GiB).toFixed(1)} GiB`, "memory per GPU, estimate"), tile(tp === 1 ? "0%" : fmt.pct(tpComm / tpComp, 1), "TP traffic ÷ compute"));
      // memory bar
      const W = 360, l = 4, r = 6, Hh = 58, axis = Math.max(total, CAP) * 1.08, X = (b) => l + (b / axis) * (W - l - r);
      const svg = s("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "scale-mem-svg", role: "img", "aria-label": `Estimated memory per GPU: ${(total / GiB).toFixed(1)} GiB` });
      let acc = 0;
      [[state, "s3", "model state, sharded"], [act, "s2", "activations"], [gath, "s1", "gathered weights"]].forEach(([b, c, n]) => {
        svg.appendChild(s("rect", { x: X(acc), y: 14, width: Math.max(0.6, X(acc + b) - X(acc) - 1), height: 20, class: `scale-seg ${c}`, "data-tip": `${n}: ${(b / GiB).toFixed(2)} GiB` }));
        acc += b;
      });
      svg.appendChild(s("line", { x1: X(CAP), x2: X(CAP), y1: 6, y2: 42, class: "scale-cap" }));
      svg.appendChild(s("text", { x: X(CAP) - 4, y: 54, class: "scale-capl", "text-anchor": "end" }, "79.32 GiB usable"));
      host.appendChild(svg);
      host.appendChild(h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s3" }), "model state (16 B × Ψ ÷ GPUs)"), h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "activations"), h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "gathered BF16 weights of the stage")));
      const notes = [];
      notes.push(total <= CAP ? `<b class="good">Fits</b> (${(total / GiB).toFixed(1)} of 79.32 GiB).` : `<b class="bad">Does not fit</b> (${(total / GiB).toFixed(1)} GiB): raise PP, CP or TP, or recompute activations.`);
      if (tp > 8) notes.push("TP 16 spans two servers, so its all-reduces run at 50 GB/s instead of 450 GB/s.");
      if (pp > 1 && m < pp) notes.push(`Only ${m} micro-batch${m > 1 ? "es" : ""} for ${pp} stages: most of each step is bubble.`);
      if (st.v > 1 && pp > 1 && st.v > lps) notes.push(`v is capped at the ${lps} layers per stage.`);
      if (st.v > 1 && pp === 1) notes.push("Interleaving needs pipeline stages; v is ignored.");
      msg.innerHTML = notes.join(" ");
    }
    draw();
  });

  // ================================================================== Part 7
  Deck.part({ n: 7, title: "Training at scale", short: "Scale" });

  Deck.slide({
    id: "scale-part",
    layout: "part",
    title: "Training at scale",
    body: String.raw`<p>Llama 3.1 405B was trained with 3.8 × 10²⁵ floating-point operations on up to 16,384 H100 GPUs. Meta's model card lists 30.84 million H100-hours for it. This part works out what such a run needs and how it is organized: the compute budget, the memory of one training step, the number formats, the collective operations that connect the GPUs, every way of splitting a model across them, and what it takes to keep the job alive for months.</p>
<p>The collective-communication chapter uses the α–β cost model and NCCL's own bandwidth definitions. The parallelism chapters end at Meta's real layout: 8-way tensor, 16-way pipeline and 128-way data parallelism.</p>`,
    take: "estimate the FLOPs, memory and communication of a training run, and choose and defend a parallel layout for it on a given cluster.",
  });

  // ================================================================== 7.1
  Deck.chapter({ title: "The compute budget" });

  Deck.slide({
    id: "scale-6nd",
    title: "Training costs about 6 FLOPs per weight per token",
    body: String.raw`<p>Part 3 counted the work of one weight matrix. In the forward pass each weight is used once per token: one multiply and one add, <b>2 FLOPs</b>. The backward pass uses the same matrix twice more: once to send the gradient to the layer's input (\(\delta W^{\top}\)) and once to form the weight's own gradient (\(x^{\top}\delta\)), 2 FLOPs per weight each. That makes 2 + 2 + 2 = 6 FLOPs per weight per token.</p>
<p>Over N weights and D training tokens the <b>training compute</b> is therefore about 6ND. The estimate leaves out three smaller terms: attention's scores and weighted sums, which grow with the context length instead of N (chapter 5.2); norms, activation functions and the optimizer update, a few FLOPs per element; and the embedding lookup, which reads a row and does no arithmetic.</p>`,
    math: String.raw`C \approx 6\,N\,D`,
    fig: {
      type: "steps", caption: "Six FLOPs per weight per token, counted on our model's first MLP matrix", start: 0,
      steps: [
        { html: String.raw`Forward: \(y = xW\). Each of the \(16 \times 64 = 1{,}024\) weights does one multiply-add per token: 2 FLOPs.`, note: "inference stops here" },
        { html: String.raw`Backward to the input: \(\delta_x = \delta_y W^{\top}\), the same 1,024 multiply-adds per token: 2 FLOPs per weight.`, note: "feeds the layer below" },
        { html: String.raw`Backward to the weights: \(\nabla W = x^{\top}\delta_y\), one multiply-add per weight per token: 2 FLOPs.`, note: "what the optimizer uses" },
        { html: String.raw`One training step of our model: 128 queries × 8 positions = 1,024 tokens. This matrix costs \(2 \times 1{,}024 \times 1{,}024 = 2{,}097{,}152\) FLOPs forward and \(4{,}194{,}304\) backward, \(6{,}291{,}456\) in total.` },
        { html: String.raw`Sum over every matrix and every token of training: \(C \approx 6ND\).`, note: "N: matrix weights, D: tokens" },
      ],
    },
    terms: [{ term: "Training compute (6ND)", def: "The floating-point operations a training run needs: about 6 per matrix weight per token, 2 in the forward pass and 4 in the backward pass." }],
    src: ["Kaplan et al., Scaling Laws for Neural Language Models, 2020 (Section 2.1)"],
  });

  Deck.slide({
    id: "scale-flops",
    title: "Our model and Llama 3.1 405B, counted the same way",
    body: String.raw`<p>For our model, N is the 7,952 weights that sit in matrices, the 1,808 of the unembedding included. The other parameters are the 1,936 entries of the token and position tables, which are lookups, and 80 norm gains. Training saw D = 6,000 steps × 128 queries × 8 positions = 6,144,000 tokens, so 6ND = 6 × 7,952 × 6,144,000 = 2.9 × 10¹¹ FLOPs. Attention's own term, \(4T^2d\) per layer per sequence in the forward pass and 3 times that in training, adds 1.9 × 10¹⁰ for 3.1 × 10¹¹ in total: 0.32 ms of an H100 at its dense BF16 peak.</p>
<p>Llama 3.1 405B: 6 × 405 × 10⁹ × 15.6 × 10¹² = 3.8 × 10²⁵ FLOPs, the figure in Meta's paper and 1.2 × 10¹⁴ times ours. At 8,192 tokens per sequence its attention term is 8.4% of 6N if the full T × T score matrix is computed, 4.2% if kernels skip the masked half. It grows linearly with T, which is one reason Meta extended the context from 8K to 128K tokens only in the last stage of pre-training, over about 800B of the 15.6T tokens.</p>`,
    fig: {
      type: "table", caption: "Training compute of the two models", align: ["l", "r", "r"],
      cols: ["", "Our model", "Llama 3.1 405B"],
      rows: [
        ["Matrix weights N", "7,952", "405 × 10⁹"],
        ["Training tokens D", "6,144,000", "15.6 × 10¹²"],
        ["Tokens per weight", "773", "39"],
        ["6ND (FLOPs)", "2.93 × 10¹¹", "3.79 × 10²⁵"],
        ["Attention term", "1.9 × 10¹⁰ (T = 8)", "+8.4% at T = 8,192"],
      ],
      note: "Llama's attention share assumes every sequence is 8,192 tokens and the full score matrix is computed; with causal skipping it halves.",
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Sections 1, 3.2.1 and 3.4.2)", "Kaplan et al., Scaling Laws for Neural Language Models, 2020"],
  });

  Deck.slide({
    id: "scale-mfu",
    title: "Model FLOPs utilization turns FLOPs into GPU-days",
    example: "tokens ids",
    body: String.raw`<p><b>Model FLOPs utilization</b> (MFU) is the FLOPs the model needs per second of training, counted from the model as on the last sheet, divided by the hardware's peak. Work the system adds, such as recomputing activations (chapter 7.2), does not count, so MFU measures useful work only. The definition is from Google's PaLM paper, and Meta uses it.</p>
<p>Meta reports 400 TFLOP/s per GPU for Llama 3.1 405B on 16,384 H100s, 40% of the 989 TFLOP/s dense BF16 peak (400/989 = 40.4%; Table 4 lists 41%, so its 400 TFLOP/s is itself rounded). At that rate the 3.79 × 10²⁵ FLOPs take 3.79 × 10²⁵ ÷ (16,384 × 400 × 10¹²) = 5.8 × 10⁶ s, which is 67 days or 26.3 million GPU-hours.</p>
<p>The model card lists 30.84 million H100-hours for the 405B model. Spread over those hours, 6ND corresponds to 341 TFLOP/s, 34.5% of peak, an average over everything the hours include: long-context stages, restarts and lost work.</p>`,
    fig: { type: "tinylm", widget: "flops", caption: "Training compute, GPU-hours and wall-clock time from 6ND, the GPU count and MFU" },
    terms: [{ term: "Model FLOPs utilization (MFU)", def: "The model's FLOPs per second during training (6N per token plus attention, without recomputation) divided by the hardware's peak FLOP/s." }],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Table 4)", "Meta, Llama 3.1 model card (training GPU-hours)", "Chowdhery et al., PaLM: Scaling Language Modeling with Pathways, 2022 (MFU definition)", "NVIDIA H100 Tensor Core GPU Architecture whitepaper (989 TFLOP/s dense BF16, SXM)"],
  });

  const CH = { E: 1.69, A: 406.4, B: 410.7, a: 0.34, b: 0.28 };
  const chL = (N, D) => CH.E + CH.A / Math.pow(N, CH.a) + CH.B / Math.pow(D, CH.b);
  Deck.slide({
    id: "scale-laws",
    title: "Loss falls as a power law in parameters and tokens",
    body: String.raw`<p>A <b>scaling law</b> is a fitted formula for the loss as a function of model size N and training tokens D; Kaplan et al. (2020) found power laws in each. Hoffmann et al. (2022) trained over 400 models, from 70 million to 16 billion parameters on 5 to 500 billion tokens, and fit</p>
\[L(N, D) = 1.69 + \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.28}}\]
<p>in nats per token. The constant is the loss no model gets below, the entropy of the text (chapter 2.5 computed it for our grammar). Doubling N divides the second term by \(2^{0.34} = 1.27\), doubling D the third by \(2^{0.28} = 1.21\).</p>
<p>For Chinchilla, 70B parameters on 1.4T tokens, the fit gives 1.69 + 0.083 + 0.163 = 1.937. Their other two methods found that optimal N and D grow equally with compute, about 20 tokens per parameter: 1B parameters on 20.2B tokens, 67B on 1.5T. The rounded fit above puts the optimum at Gopher's budget (5.76 × 10²³ FLOPs) at 32B parameters, 93 tokens each; the unrounded exponents (α = 0.3392, β = 0.2849, E = 1.6934) give 40B, the paper's figure. A refit of the same data by Besiroglu et al. (2024) gives 72B, 18 tokens per parameter.</p>`,
    fig: {
      type: "plot", caption: "The fitted loss against training tokens for one model size; the dot marks 20 tokens per parameter",
      x: [9, 13.5], y: [1.6, 3.2], xlabel: "training tokens D", ylabel: "loss (nats per token)", xticks: [9, 10, 11, 12, 13], yticks: [1.6, 2, 2.4, 2.8, 3.2],
      xaxis: { fmt: (v) => `10${sup(v)}` },
      params: [{ name: "lgN", label: "parameters N", min: 8, max: 12, step: 0.005, value: Math.log10(70e9), fmt: (v) => human(Math.pow(10, v)) }],
      fns: [{ f: (x, p) => chL(Math.pow(10, p.lgN), Math.pow(10, x)), cls: "s1" }],
      extra: (p) => {
        const N = Math.pow(10, p.lgN), fl = CH.E + CH.A / Math.pow(N, CH.a), d20 = Math.log10(20 * N);
        return {
          hlines: [{ y: fl, cls: "" }, { y: CH.E, label: "E = 1.69", cls: "red" }],
          dots: d20 <= 13.5 && chL(N, 20 * N) <= 3.2 ? [{ x: d20, y: chL(N, 20 * N), r: 5, cls: "", tip: `D = 20N = ${human(20 * N)} tokens: loss ${chL(N, 20 * N).toFixed(3)}` }] : [],
        };
      },
      readouts: [
        { label: "loss at D = 20N", f: (p) => chL(Math.pow(10, p.lgN), 20 * Math.pow(10, p.lgN)), fmt: "f3" },
        { label: String.raw`limit as \(D \to \infty\)`, f: (p) => CH.E + CH.A / Math.pow(Math.pow(10, p.lgN), CH.a), fmt: "f3" },
      ],
    },
    terms: [{ term: "Scaling law", def: "A fitted formula for the loss as a function of parameters N and training tokens D, such as L = E + A/N^α + B/D^β; used to choose N and D before an expensive run." }],
    src: ["Hoffmann et al., Training Compute-Optimal Large Language Models, 2022 (equation 10, Tables 2 and 3, Figure 4)", "Besiroglu et al., Chinchilla Scaling: A Replication Attempt, 2024 (Table 1)", "Kaplan et al., Scaling Laws for Neural Language Models, 2020"],
  });

  Deck.slide({
    id: "scale-overtrain",
    title: "Small models are trained far past compute-optimal",
    example: "tokens ids",
    body: String.raw`<p>A <b>compute-optimal</b> model is the N that gives the lowest loss for a fixed training budget C = 6ND. With 20 tokens per parameter, C = 6 · N · 20N = 120N², so N = √(C/120).</p>
<p>That optimum ignores what the model costs to use. Each generated token costs about 2N FLOPs (Part 8), on every request for the model's whole deployment. A smaller model trained on more tokens can reach the same loss with less inference work, even though its training used compute less efficiently.</p>
<p>Meta's own scaling-law fit for Llama 3 put the best model for 3.8 × 10²⁵ FLOPs at 402B parameters on 16.55T tokens, 41 tokens per parameter, and it trained 405B on 15.6T. The 8B model saw about 15T tokens: 1,875 per parameter, 94 times the Chinchilla ratio. Meta's paper says the smaller models were trained much longer than compute-optimal and beat compute-optimal models at the same inference budget.</p>`,
    fig: { type: "tinylm", widget: "scaling", caption: "Models placed by parameters and training tokens; the line marks 20 tokens per parameter" },
    check: {
      q: "Chinchilla's budget was 6 × 70 × 10⁹ × 1.4 × 10¹² = 5.88 × 10²³ FLOPs. With 20 tokens per parameter, which model size is compute-optimal for it?",
      options: ["7B", "70B", "280B", "700B"], answer: 1,
      why: "C = 120N², so N = √(5.88 × 10²³ / 120) = √(4.9 × 10²¹) = 7.0 × 10¹⁰ parameters, trained on 20N = 1.4T tokens. Gopher used a similar budget for 280B parameters on 300B tokens, about 1 token per parameter, and lost to Chinchilla.",
    },
    terms: [{ term: "Compute-optimal", def: "The split of a fixed training budget C = 6ND between parameters N and tokens D that gives the lowest loss; about 20 tokens per parameter in the Chinchilla fits." }],
    src: ["Hoffmann et al., Training Compute-Optimal Large Language Models, 2022", "Dubey et al., The Llama 3 Herd of Models, 2024 (Sections 1 and 3.2.1)", "Meta, Llama 3.1 model card (~15T training tokens)"],
  });

  // ================================================================== 7.2
  Deck.chapter({ title: "Memory of a training step" });

  Deck.slide({
    id: "scale-state",
    title: "Mixed-precision Adam keeps 16 bytes per parameter",
    body: String.raw`<p>A training step needs more than the weights. With Adam in mixed precision (chapter 7.3 explains the formats) every parameter carries five numbers: a BF16 weight that the matrix multiplies read (2 bytes), a BF16 gradient (2), an FP32 <b>master weight</b> that receives the updates (4), and Adam's two moving averages in FP32 (4 + 4). That is 16 bytes, the accounting used by the ZeRO paper.</p>
<p>A 70B model needs 16 × 70 × 10⁹ = 1.12 × 10¹² bytes of this <b>model state</b>. One H100 has 80 GiB of HBM, 85.9 × 10⁹ bytes (CUDA reports 79.32 GiB of it as usable). The state alone fills 13.04 of them, so at least 14 GPUs before a single activation is stored. Llama 3.1 405B needs 6.48 × 10¹² bytes, 76 GPUs.</p>
<p>Recipes vary. Llama 3 accumulates gradients over micro-batches and reduces them in FP32, which costs 2 more bytes per parameter wherever an FP32 gradient is held. Our model trains in FP32 with no separate master copy: 4 bytes each for weight, gradient and both averages, also 16, which is 159,488 bytes.</p>`,
    fig: {
      type: "calc", caption: "Model state of a training step. GPU counts use 80 GiB (85.9 GB) per H100.",
      inputs: [
        { id: "N", label: "parameters", min: 1e4, max: 2e12, value: 70e9, log: true, step: 0.001, fmt: (v) => human(v) },
        { id: "g", label: "gradients held in", options: [{ label: "BF16 (2 B)", value: 2 }, { label: "FP32 (4 B)", value: 4 }], value: 2 },
      ],
      outputs: [
        { label: "BF16 weights", f: (v) => 2 * v.N, fmt: "bytes" },
        { label: "gradients", f: (v) => v.g * v.N, fmt: "bytes" },
        { label: "FP32 master + 2 moments", f: (v) => 12 * v.N, fmt: "bytes" },
        { label: "total model state", f: (v) => (14 + v.g) * v.N, fmt: "bytes" },
        { label: "H100s to hold it", f: (v) => Math.ceil((14 + v.g) * v.N / H100.mem), fmt: "int" },
      ],
      note: (v) => `${14 + v.g} bytes × ${human(v.N)} parameters = ${bytes((14 + v.g) * v.N)}, ${((14 + v.g) * v.N / H100.mem).toFixed(2)} H100s of 80 GiB.`,
    },
    terms: [
      { term: "Master weights", def: "An FP32 copy of the weights kept for the optimizer update when the forward and backward passes use BF16 or FP16 copies." },
      { term: "Model state", def: "Everything stored per parameter across steps: weights, gradients and optimizer state. 16 bytes per parameter for mixed-precision Adam." },
    ],
    src: ["Rajbhandari et al., ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, 2020 (Section 3.1)", "Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.2, numerical stability)", "NVIDIA H100 datasheet (80 GB HBM3)"],
  });

  Deck.slide({
    id: "scale-act-ours",
    title: "Activations: what the backward pass needs from the forward pass",
    example: "tokens ids",
    body: String.raw`<p>The backward pass multiplies each incoming gradient by values from the forward pass (Part 3). A weight gradient \(x^{\top}\delta\) needs the layer's input x, GELU's derivative needs its input, softmax's needs its output. The forward pass must keep these tensors, the <b>activations</b>, until the backward pass has used them. Unlike the 16 bytes per parameter, they grow with batch size × sequence length.</p>
<p>One block of our model keeps 272 numbers per token (the table). In FP32 that is 1,088 bytes per token per layer. Our query's 8 positions through 2 layers keep 17,408 bytes, and a training batch of 128 queries keeps 2,228,224 bytes, 56 times the 39,872 bytes of the model's FP32 weights.</p>
<p>At Llama scale the same bookkeeping, per token and per layer, is multiplied by a hidden size a thousand times larger, sequences of thousands of tokens and over a hundred layers.</p>`,
    fig: {
      type: "table", caption: "Tensors one block of our model keeps per token for the backward pass", align: ["l", "r", "l"],
      cols: ["Tensor", "Numbers", "Needed for"],
      rows: [
        ["input of the first RMSNorm", "16", "the norm's derivative"],
        ["normed input", "16", "gradients of W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>"],
        ["queries and keys", "32", "gradient of the scores"],
        ["attention weights, 2 heads × 8", "16", "softmax derivative and mixing"],
        ["values", "16", "gradient of the attention weights"],
        ["concatenated heads", "16", "gradient of W<sub>O</sub>"],
        ["input of the second RMSNorm", "16", "the norm's derivative"],
        ["normed input", "16", "gradient of W<sub>1</sub>"],
        ["GELU input", "64", "GELU's derivative"],
        ["GELU output", "64", "gradient of W<sub>2</sub>"],
        ["<b>total</b>", "<b>272</b>", ""],
      ],
      note: "Attention weights are kept for all 8 key positions, masked ones included, as a dense kernel computes them.",
    },
    terms: [{ term: "Activation memory", def: "The forward-pass tensors kept for the backward pass. It grows with micro-batch size × sequence length × layers, unlike the per-parameter model state." }],
  });

  const KM = [
    { label: "22B", h: 6144, a: 64, L: 48 }, { label: "GPT-3 175B", h: 12288, a: 96, L: 96 },
    { label: "MT-NLG 530B", h: 20480, a: 128, L: 105 }, { label: "1T", h: 25600, a: 160, L: 128 },
  ];
  Deck.slide({
    id: "scale-act-formula",
    title: "Activation memory per layer is sbh(34 + 5as/h) bytes",
    body: String.raw`<p>Korthikanti et al. counted the same tensors for a GPT-style layer with sequence length s, micro-batch b, hidden size h and a attention heads, storing activations in 16 bits and dropout masks in 1 byte per element:</p>
\[\text{bytes per layer} = s\,b\,h\left(34 + \frac{5as}{h}\right)\]
<p>The 34sbh part is the inputs of the linear layers, norms and GELU and the dropout masks: 11sbh in attention, 19sbh in the MLP, 4sbh in the two norms. The \(5as^2b\) part is the softmax output (\(2as^2b\)), its dropout mask (\(as^2b\)) and the dropout output (\(2as^2b\)); it grows with s².</p>
<p>GPT-3 175B (h = 12,288, a = 96, 96 layers) at s = 2,048: 5as/h = 80, so the score terms are 70% of the total. One sequence keeps 2.87 GB per layer, 275 GB for all 96 layers, more than three H100s for a single sequence. An attention kernel that recomputes the scores in the backward pass instead of storing them, as FlashAttention does (Part 6), removes the 5as/h term: 0.86 GB per layer, 82 GB per sequence.</p>`,
    fig: {
      type: "calc", caption: "Activation memory of one micro-batch, from Korthikanti et al.'s formula (no tensor or sequence parallelism)",
      inputs: [
        { id: "m", label: "model", options: KM.map((k, i) => ({ label: k.label, value: i })), value: 1 },
        { id: "ls", label: "sequence length s", min: 9, max: 15, step: 1, value: 11, fmt: (x) => Math.pow(2, x).toLocaleString("en-US") },
        { id: "b", label: "micro-batch b", min: 1, max: 8, step: 1, value: 1, fmt: "int" },
        { id: "fa", label: "attention scores", options: [{ label: "stored", value: 1 }, { label: "recomputed", value: 0 }], value: 1 },
      ],
      outputs: [
        { label: "per layer", f: (v) => { const k = KM[v.m], s0 = Math.pow(2, v.ls); return s0 * v.b * k.h * (34 + v.fa * 5 * k.a * s0 / k.h); }, fmt: "bytes" },
        { label: "all layers", f: (v) => { const k = KM[v.m], s0 = Math.pow(2, v.ls); return k.L * s0 * v.b * k.h * (34 + v.fa * 5 * k.a * s0 / k.h); }, fmt: "bytes" },
        { label: "share of the score terms", f: (v) => { const k = KM[v.m], s0 = Math.pow(2, v.ls), q = v.fa * 5 * k.a * s0 / k.h; return q / (34 + q); }, fmt: "pct" },
      ],
      note: (v) => { const k = KM[v.m]; return `${k.label}: h = ${k.h.toLocaleString("en-US")}, a = ${k.a}, ${k.L} layers. Model configurations from Korthikanti et al., Table 3.`; },
    },
    src: ["Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models, 2022 (Section 4.1, equation 1, Table 3)"],
  });

  Deck.slide({
    id: "scale-recompute",
    title: "Activation checkpointing trades a second forward pass for memory",
    body: String.raw`<p><b>Activation checkpointing</b>, also called recomputation, keeps only each layer's input during the forward pass. In the backward pass it reruns that layer's forward computation from the saved input to rebuild the activations it needs, then discards them again. Memory per layer falls to the input alone, 2sbh bytes in 16 bits: 50.3 MB per layer for one GPT-3 sequence of 2,048 tokens, 4.8 GB for all 96 layers, down from 275 GB.</p>
<p>The price is a second forward pass: 2 + 2 + 4 = 8 FLOPs per weight per token instead of 6, a third more compute. Korthikanti et al. measured 36% extra time on their 530B and 1T models. Recomputing only the attention-score part, cheap in FLOPs and large in memory (<b>selective recomputation</b>), together with sequence parallelism (chapter 7.6) cost 2% on the same models.</p>
<p>Meta trained Llama 3 405B at 8K-token sequences without activation checkpointing, after removing tensors that would never be used again.</p>`,
    fig: {
      type: "table", caption: "Activation memory of one GPT-3 175B layer for one 2,048-token sequence, and the extra compute", align: ["l", "r", "r"],
      cols: ["Strategy", "Kept per layer", "Extra compute"],
      rows: [
        ["store everything", "sbh(34 + 5as/h) = 2.87 GB", "0"],
        ["selective recomputation (scores)", "34sbh = 0.86 GB", "2.7% of FLOPs"],
        ["full recomputation", "2sbh = 0.050 GB", "33% of FLOPs"],
      ],
      note: "FLOP overheads for GPT-3 175B from Korthikanti et al.; full recomputation adds one forward pass, 2 of every 6 FLOPs.",
    },
    terms: [{ term: "Activation checkpointing", def: "Keeping only some activations (for example each layer's input) and recomputing the rest during the backward pass; one extra forward pass in exchange for far less memory." }],
    src: ["Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models, 2022 (Sections 4.2 and 5)", "Chen et al., Training Deep Nets with Sublinear Memory Cost, 2016", "Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.2)"],
  });

  Deck.slide({
    id: "scale-70b",
    title: "A 70B model's training state alone needs 14 GPUs",
    body: String.raw`<p>Put the pieces together for a 70B model. The BF16 weights are 140 GB and the gradients another 140 GB. The FP32 master weights and Adam's two moments are 280 GB each. In total 1.12 TB, while one H100 holds 85.9 GB: 13.04 GPUs' worth, so 14 at least, and more once activations, temporary buffers and memory fragmentation are added.</p>
<p>The state must be split across GPUs. There are three ways to split it, and real runs combine them:</p>
<ul><li>shard the per-parameter state across the data-parallel replicas (ZeRO and FSDP, chapter 7.5);</li>
<li>split every weight matrix across a few GPUs (tensor parallelism, chapter 7.6);</li>
<li>give each GPU a range of layers (pipeline parallelism, chapter 7.6).</li></ul>`,
    fig: {
      type: "bars", caption: "Model state of a 70B model next to one H100 (80 GiB)", fmt: "f0",
      items: [
        { label: "BF16 weights", value: 140, text: "140 GB" },
        { label: "BF16 gradients", value: 140, text: "140 GB" },
        { label: "FP32 master", value: 280, text: "280 GB" },
        { label: "Adam moment 1", value: 280, text: "280 GB" },
        { label: "Adam moment 2", value: 280, text: "280 GB" },
        { label: "total", value: 1120, text: "1,120 GB" },
        { label: "one H100", value: 85.9, text: "85.9 GB", hi: true },
      ],
    },
    src: ["Rajbhandari et al., ZeRO, 2020", "NVIDIA H100 datasheet (80 GB HBM3, 3.35 TB/s, SXM)"],
  });

  Deck.slide({
    id: "scale-accum",
    title: "Global batch = micro-batch × accumulation steps × replicas",
    body: String.raw`<p>A step's batch does not have to fit in memory at once. Part 3 showed <b>gradient accumulation</b>: run several micro-batches forward and backward one after another, add their gradients, then take one optimizer step. Activation memory is set by the micro-batch; the optimizer sees the whole batch.</p>
<p>With data parallelism (chapter 7.5) every replica does this on different data at the same time, so the <b>global batch</b> is the product of micro-batch size, accumulation steps and the number of replicas.</p>
<p>Llama 3.1 405B's main stage used 16M-token steps: 2,048 sequences of 8,192 tokens, 16,777,216 tokens. On 16,384 GPUs with 128 data-parallel groups, each group processed 16 sequences per step; on 8,192 GPUs, 64 groups processed 32 each. At 16M tokens per step, 15.6T tokens take about 930,000 steps.</p>`,
    fig: {
      type: "calc", caption: "Global batch from its three factors",
      inputs: [
        { id: "mb", label: "micro-batch (sequences)", min: 1, max: 8, step: 1, value: 1, fmt: "int" },
        { id: "acc", label: "accumulation steps", min: 1, max: 64, step: 1, value: 16, fmt: "int" },
        { id: "dp", label: "data-parallel replicas", min: 0, max: 10, step: 1, value: 7, fmt: (x) => Math.pow(2, x).toLocaleString("en-US") },
        { id: "s", label: "sequence length", options: [{ label: "4,096", value: 4096 }, { label: "8,192", value: 8192 }, { label: "131,072", value: 131072 }], value: 8192 },
      ],
      outputs: [
        { label: "sequences per step", f: (v) => v.mb * v.acc * Math.pow(2, v.dp), fmt: "int" },
        { label: "tokens per step", f: (v) => v.mb * v.acc * Math.pow(2, v.dp) * v.s, fmt: "int" },
        { label: "steps for 15.6T tokens", f: (v) => 15.6e12 / (v.mb * v.acc * Math.pow(2, v.dp) * v.s), fmt: "int" },
      ],
    },
    check: {
      q: "Micro-batches of 2 sequences, 8 accumulation steps, 64 data-parallel replicas, 4,096 tokens per sequence. How many tokens does one optimizer step see?",
      options: ["65,536", "524,288", "4,194,304", "33,554,432"], answer: 2,
      why: "2 × 8 × 64 = 1,024 sequences per step, × 4,096 tokens = 4,194,304 tokens, the 4M-token batch Llama 3 started with. 524,288 leaves out the accumulation steps (2 × 64 × 4,096); 65,536 is one replica's step (2 × 8 × 4,096).",
    },
    terms: [
      { term: "Gradient accumulation", def: "Adding the gradients of several micro-batches before one optimizer step, so the batch the optimizer sees can be larger than what fits in memory at once." },
      { term: "Global batch size", def: "Tokens (or sequences) per optimizer step across all replicas: micro-batch × accumulation steps × data-parallel replicas." },
    ],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Table 4 and Section 3.4.1)"],
  });

  // ================================================================== 7.3
  Deck.chapter({ title: "Number formats" });

  Deck.slide({
    id: "scale-formats",
    title: "A float spends its bits on range or on precision",
    body: String.raw`<p>A floating-point number is \((-1)^{s} \times 1.m \times 2^{e - \text{bias}}\): a sign bit, an exponent field e that sets the scale, and mantissa bits m that set how finely values are spaced. With k mantissa bits, neighboring values differ by a factor of about \(1 + 2^{-k}\), so rounding changes a value by at most \(2^{-k-1}\) of itself. The exponent width sets the range: the largest finite value and the smallest normal one.</p>
<p>BF16 keeps FP32's 8 exponent bits and cuts the mantissa to 7: the same range as FP32 with a spacing of \(2^{-7} = 0.0078\). FP16 has 10 mantissa bits but only 5 exponent bits: 8 times finer spacing, and a largest value of 65,504. TF32 is a tensor-core input format with FP32's range and FP16's 10-bit mantissa, stored in 32 bits. FP8 comes in two layouts: E4M3 with more precision, E5M2 with more range.</p>`,
    fig: {
      type: "table", caption: "Bit layouts, range and precision", align: ["l", "r", "r", "r", "r"], minWidth: 560,
      cols: ["Format", "Sign · exp · mantissa", "Largest", "Smallest normal", "Spacing at 1"],
      rows: [
        ["FP32", "1 · 8 · 23", "3.40 × 10³⁸", "1.18 × 10⁻³⁸", "1.2 × 10⁻⁷"],
        ["TF32", "1 · 8 · 10", "3.40 × 10³⁸", "1.18 × 10⁻³⁸", "9.8 × 10⁻⁴"],
        ["BF16", "1 · 8 · 7", "3.39 × 10³⁸", "1.18 × 10⁻³⁸", "7.8 × 10⁻³"],
        ["FP16", "1 · 5 · 10", "65,504", "6.10 × 10⁻⁵", "9.8 × 10⁻⁴"],
        ["FP8 E4M3", "1 · 4 · 3", "448", "0.0156", "0.125"],
        ["FP8 E5M2", "1 · 5 · 2", "57,344", "6.10 × 10⁻⁵", "0.25"],
      ],
      highlight: [2],
      note: "E4M3 gives up infinities and keeps one NaN pattern to reach 448; E5M2 follows the IEEE conventions.",
    },
    terms: [
      { term: "BF16", def: "16-bit floating point with 1 sign, 8 exponent and 7 mantissa bits: FP32's range with about 2–3 significant decimal digits." },
      { term: "FP8 (E4M3, E5M2)", def: "8-bit floating-point formats: E4M3 (4 exponent, 3 mantissa bits, max 448) and E5M2 (5 and 2, max 57,344)." },
    ],
    src: ["IEEE 754-2019 (binary32, binary16)", "Micikevicius et al., FP8 Formats for Deep Learning, 2022 (Table 1)", "NVIDIA A100 Tensor Core GPU Architecture whitepaper (TF32)"],
  });

  Deck.slide({
    id: "scale-sums",
    title: "A long sum stalls when the accumulator is narrow",
    body: String.raw`<p>To add a small number to a large one, the hardware aligns their exponents, and the small number's bits that fall below the large one's spacing are rounded away. In BF16, neighboring values between 4 and 8 are 2⁻⁵ = 0.03125 apart. Adding 0.01 to 4 gives 4.01, less than half a spacing above 4, so the result rounds back to 4. A running BF16 sum of 0.01s stops growing at 4, however many terms follow.</p>
<p>The plate adds the same value n times in two accumulators. With 1,000 terms of 0.01 the exact sum is 10; the FP32 accumulator gives 10.0098, off by the 0.098% that BF16 rounding of the input 0.01 introduced, while the BF16 accumulator stops at 4 after 350 terms.</p>
<p>A dot product of length 4,096 inside a matrix multiply, or a gradient summed over millions of tokens, is this kind of sum. Tensor cores multiply BF16 inputs but accumulate in FP32, and Llama 3 accumulates and reduces gradients in FP32.</p>`,
    fig: { type: "scale-sum", caption: "Adding the same BF16 value n times with an FP32 and a BF16 accumulator" },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.2)", "NVIDIA H100 Tensor Core GPU Architecture whitepaper (FP32 accumulation of BF16 products)"],
  });

  Deck.slide({
    id: "scale-loss-scaling",
    title: "FP16 needs loss scaling; BF16 does not",
    example: "tokens ids",
    body: String.raw`<p>FP16's smallest normal value is 6.1 × 10⁻⁵ and its smallest subnormal \(2^{-24} = 6.0 \times 10^{-8}\). Gradients are often smaller. Micikevicius et al. found that in one speech model about 5% of weight-gradient values were below \(2^{-24}\), so FP16 would turn them into zeros.</p>
<p><b>Loss scaling</b> multiplies the loss by a factor S before the backward pass. By the chain rule every gradient is then multiplied by S, which lifts small values into FP16's range; the weight gradients are divided by S again before the update. They used S from 8 to 32K. If S times the largest gradient passes 65,504, gradients overflow to infinity; dynamic loss scaling (PyTorch's GradScaler) then skips that step and halves S, and doubles S again after 2,000 clean steps.</p>
<p>BF16 has FP32's exponent range, down to 1.2 × 10⁻³⁸, so a gradient that FP16 flushes to zero survives in BF16 with reduced precision, and no scaling is needed. Try the 2 × 10⁻⁸ gradient and the 70,000 activation in the plate.</p>`,
    fig: { type: "tinylm", widget: "precision", caption: "One number in five formats: the bits each stores and the value it rounds to" },
    terms: [{ term: "Loss scaling", def: "Multiplying the loss by a factor S before the backward pass so that small FP16 gradients do not round to zero, then dividing the gradients by S before the update." }],
    src: ["Micikevicius et al., Mixed Precision Training, 2018 (Sections 3.1 and 3.2)", "PyTorch documentation, torch.amp.GradScaler (growth_factor 2.0, backoff_factor 0.5, growth_interval 2000)"],
  });

  Deck.slide({
    id: "scale-mixed",
    title: "The mixed-precision recipe: BF16 arithmetic, FP32 state",
    body: String.raw`<p>Mixed-precision training puts each quantity in the narrowest format that does not damage it:</p>
<ul><li>matrix multiplies read BF16 weights and activations and accumulate in FP32 inside the tensor cores: 989 TFLOP/s dense on an H100 SXM, against 67 TFLOP/s for FP32 on the ordinary cores;</li>
<li>activations are stored in BF16, which halves activation memory;</li>
<li>reductions that are sensitive to rounding (softmax sums, norm statistics, the loss, gradient accumulation) run in FP32;</li>
<li>the optimizer updates FP32 master weights with FP32 moments, then writes a BF16 copy for the next step.</li></ul>
<p>The master copy exists because updates are small. Llama 3 405B's learning rate ends at 8 × 10⁻⁷. An Adam update of that size on a weight of 0.01 is 1/12,500 of it, and BF16 can only change a value by steps of about 1/128 to 1/256 of itself, so the update would round away. FP32's step is 2⁻²³ of the value, about 1/8,400,000, and keeps it.</p>`,
    fig: {
      type: "diagram", caption: "Which format each tensor lives in during one step", w: 15, h: 6.4, unit: 40,
      nodes: [
        { id: "m", x: 0.3, y: 0.4, w: 3.6, h: 1.3, label: "master weights", sub: "FP32, 4 B", kind: "blue" },
        { id: "w", x: 5.1, y: 0.4, w: 3.6, h: 1.3, label: "weights copy", sub: "BF16, 2 B" },
        { id: "f", x: 9.9, y: 0.4, w: 4.8, h: 1.3, label: "matrix multiplies", sub: "BF16 inputs, FP32 accumulate", kind: "hi" },
        { id: "a", x: 9.9, y: 2.6, w: 4.8, h: 1.2, label: "activations", sub: "BF16, kept for backward" },
        { id: "g", x: 5.1, y: 4.7, w: 3.6, h: 1.3, label: "gradients", sub: "BF16, FP32 sums" },
        { id: "o", x: 0.3, y: 4.7, w: 3.6, h: 1.3, label: "Adam moments", sub: "FP32, 4 + 4 B", kind: "blue" },
      ],
      edges: [
        { from: "m", to: "w", label: "round" }, { from: "w", to: "f" }, { from: "f", to: "a", label: "forward", dx: 34, dy: 10 },
        { from: "a", to: "g", label: "backward", dx: 30, dy: 16 }, { from: "g", to: "o", label: "update" }, { from: "o", to: "m", label: "step", dx: 20 },
      ],
    },
    terms: [{ term: "Mixed precision", def: "Training with BF16 or FP16 matrix multiplies and stored activations while keeping master weights, optimizer state and sensitive sums in FP32." }],
    src: ["Micikevicius et al., Mixed Precision Training, 2018", "NVIDIA H100 Tensor Core GPU Architecture whitepaper (SXM: 989 TFLOP/s dense BF16, 67 TFLOP/s FP32)", "Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.4.1, learning rate)"],
  });

  Deck.slide({
    id: "scale-fp8",
    title: "FP8 training needs a scale factor for every block of values",
    body: String.raw`<p>H100 tensor cores multiply FP8 matrices at twice the BF16 rate, 1,979 TFLOP/s dense. E4M3 stores values only up to 448 with a relative spacing of 1/8 to 1/16, so each tensor is multiplied by a scale factor before rounding, typically s = 448 / max|x|, and the product is divided by it afterwards.</p>
<p>With one scale per tensor, a single large value sets s for all the others. After scaling, values below \(2^{-10} \approx 0.001\), half of E4M3's smallest step \(2^{-9}\), become 0, and values below \(2^{-6}\) keep fewer than 3 mantissa bits. Raise the outlier in the plate and watch block 2's error with one scale per tensor.</p>
<p>DeepSeek-V3 trained in FP8 with E4M3 for all tensors and fine-grained scales: one per 1 × 128 tile of activations (one token, 128 channels) and one per 128 × 128 block of weights. It also copies the tensor cores' partial sums to FP32 every 128 products, because FP8 accumulation on its H800s keeps only about 14 bits, which in their test gave errors near 2% for sums of 4,096 products. Its FP8 runs stayed within 0.25% (relative) of the BF16 loss.</p>`,
    fig: { type: "scale-fp8", caption: "256 activations rounded to FP8 E4M3 with one scale factor per tensor or one per 128 values" },
    src: ["DeepSeek-AI, DeepSeek-V3 Technical Report, 2024 (Section 3.3)", "Micikevicius et al., FP8 Formats for Deep Learning, 2022", "NVIDIA H100 Tensor Core GPU Architecture whitepaper (1,979 TFLOP/s dense FP8)"],
  });

  // ================================================================== 7.4
  Deck.chapter({ title: "Collective communication" });

  Deck.slide({
    id: "scale-p2p",
    title: "Point-to-point messages and collective operations",
    body: String.raw`<p>Each GPU in a training job is driven by one process, its <b>rank</b>. A <b>point-to-point</b> operation moves a buffer from one rank to another: a send matched by a receive. A <b>collective operation</b> is called by every rank of a group together, and its result is defined on all of their buffers: a sum across ranks, a copy to every rank, a transpose.</p>
<p>MPI defined these operations for CPU clusters. NCCL, NVIDIA's collective communication library, implements the same ones for GPUs as CUDA kernels on CUDA streams and moves the data over NVLink, PCIe, InfiniBand or RoCE. PyTorch's <code>torch.distributed</code> calls NCCL.</p>
<p>Every form of parallelism in this part comes down to one or two of these operations (the table). Its communication cost is the cost of those operations on the links they cross, which is what the rest of this chapter computes.</p>`,
    fig: {
      type: "table", caption: "The communication behind each form of parallelism", minWidth: 520,
      cols: ["Parallelism", "Operation", "When"],
      rows: [
        ["data parallel", "all-reduce of gradients", "once per step"],
        ["ZeRO-3 / FSDP", "all-gather of weights, reduce-scatter of gradients", "per layer, forward and backward"],
        ["tensor parallel", "all-reduce of activations (or reduce-scatter + all-gather)", "2 per layer forward, 2 backward"],
        ["pipeline parallel", "send / receive of activations and their gradients", "per micro-batch, between neighbor stages"],
        ["expert parallel", "all-to-all of tokens", "2 per MoE layer forward, 2 backward"],
        ["context parallel", "all-gather of keys and values, or ring send / receive", "per attention layer"],
      ],
    },
    terms: [
      { term: "Rank", def: "One process in a distributed job, usually driving one GPU, identified by an integer from 0 to N − 1." },
      { term: "Collective operation", def: "A communication operation that every rank of a group calls together, with a result defined on all ranks' buffers: broadcast, reduce, all-reduce, reduce-scatter, all-gather, all-to-all." },
    ],
    src: ["NVIDIA NCCL User Guide (collective operations)", "Dubey et al., The Llama 3 Herd of Models, 2024 (Sections 3.3.2 and 3.3.3)"],
  });

  Deck.slide({
    id: "scale-collectives",
    title: "Six collectives, defined by what each rank holds afterwards",
    body: String.raw`<p>With N ranks, each holding a buffer of S bytes cut into N chunks:</p>
<ul><li><b>broadcast</b>: the root's buffer is copied to every rank;</li>
<li><b>reduce</b>: the elementwise sum of all buffers lands on the root;</li>
<li><b>all-reduce</b>: every rank gets the elementwise sum;</li>
<li><b>reduce-scatter</b>: rank i gets chunk i of the sum, S/N bytes;</li>
<li><b>all-gather</b>: rank i contributes chunk i, and every rank ends with all N chunks;</li>
<li><b>all-to-all</b>: rank i sends its chunk j to rank j, a transpose of chunks across ranks.</li></ul>
<p>An all-reduce gives the same result as a reduce-scatter followed by an all-gather of the reduced chunks. That identity is the ring algorithm on the next sheet, and FSDP and sequence parallelism split their all-reduces at it to do useful work between the halves.</p>
<p>In the plate rank r's chunk c holds 10(r + 1) + (c + 1), so every sum can be checked by eye: chunk 1 sums to 11 + 21 + 31 + 41 = 104.</p>`,
    fig: { type: "scale-coll", caption: "Buffers of 4 ranks before and after each collective" },
    terms: [
      { term: "All-reduce", def: "A collective that leaves the elementwise sum (or other reduction) of all ranks' buffers on every rank." },
      { term: "Reduce-scatter", def: "A collective that sums all ranks' buffers and leaves chunk i of the result, 1/N of it, on rank i." },
      { term: "All-gather", def: "A collective in which each rank contributes one chunk and every rank ends with all N chunks." },
      { term: "All-to-all", def: "A collective in which rank i sends its j-th chunk to rank j, transposing the chunks across ranks; used to route tokens to experts." },
    ],
    src: ["NVIDIA NCCL User Guide (collective operations)", "Thakur, Rabenseifner and Gropp, Optimization of Collective Communication Operations in MPICH, 2005"],
  });

  Deck.slide({
    id: "scale-ring",
    title: "Ring all-reduce: a reduce-scatter, then an all-gather",
    example: "tokens ids",
    body: String.raw`<p>Arrange N ranks in a ring and cut the buffer into N chunks of S/N bytes. In each of N − 1 <b>reduce-scatter</b> steps every rank sends one chunk to the next rank and adds the chunk it receives into its own copy. Afterwards rank i holds the complete sum of one chunk. In N − 1 <b>all-gather</b> steps the finished chunks travel once around the ring.</p>
<p>Each rank sends 2(N − 1) chunks, \(2\frac{N-1}{N}S\) bytes, and receives as many. Every link is busy in every step, and when the ranks do the additions themselves on incompressible data, no all-reduce moves fewer bytes per rank (Patarasuk and Yuan): the ring is bandwidth-optimal. Switches that add (in-network reduction, two sheets on) are outside that bound. Its weakness is the 2(N − 1) steps in sequence, each paying the latency of one message.</p>
<p>The BF16 gradients of an 8B model are 16 GB. On 8 GPUs each sends 2 × 7/8 × 16 = 28 GB, 62 ms at NVLink's 450 GB/s per direction. On 1,024 GPUs each sends 31.97 GB, almost the 2S limit; across servers at 50 GB/s that takes 0.64 s.</p>`,
    math: String.raw`T_{\text{ring}} = 2(N-1)\,\alpha + 2\,\frac{N-1}{N}\cdot\frac{S}{B}`,
    fig: { type: "tinylm", widget: "allreduce", caption: "Ring all-reduce step by step, and its time in the α–β model" },
    terms: [{ term: "Ring all-reduce", def: "An all-reduce in N − 1 reduce-scatter steps and N − 1 all-gather steps around a ring; each rank sends 2(N − 1)/N of the buffer, the minimum possible." }],
    src: ["Patarasuk and Yuan, Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations, 2009", "Thakur, Rabenseifner and Gropp, 2005", "NVIDIA H100 Tensor Core GPU Architecture whitepaper (NVLink 4: 18 links, 900 GB/s total)"],
  });

  const AB = (v) => {
    const p = Math.pow(2, v.lp), n = v.n, a = v.a * 1e-6, b = 1 / v.bw, lg = v.lp;
    return { ring: 2 * (p - 1) * a + 2 * ((p - 1) / p) * n * b, rd: lg * (a + n * b), rab: 2 * lg * a + 2 * ((p - 1) / p) * n * b, tree: 2 * lg * (a + n * b) };
  };
  Deck.slide({
    id: "scale-alphabeta",
    title: "The α–β model prices latency and bandwidth separately",
    body: String.raw`<p>The <b>α–β model</b> charges α + mβ to send an m-byte message: α is the fixed start-up latency, β the time per byte, one over the bandwidth. Thakur, Rabenseifner and Gropp give each all-reduce algorithm's cost for an n-byte vector on p ranks (the additions left out):</p>
<ul><li>ring: \(2(p-1)\alpha + 2\frac{p-1}{p}n\beta\);</li>
<li>recursive doubling: \(\log_2 p\,(\alpha + n\beta)\), the whole vector in every step;</li>
<li>recursive halving, then doubling (Rabenseifner): \(2\log_2 p\,\alpha + 2\frac{p-1}{p}n\beta\);</li>
<li>binomial-tree reduce, then broadcast: \(2\log_2 p\,(\alpha + n\beta)\).</li></ul>
<p>Small messages are latency-bound and favor few steps; large ones are bandwidth-bound and favor moving only 2(p − 1)/p · n bytes. With p = 1,024, α = 10 µs and 50 GB/s, a 1 MB all-reduce takes 20.5 ms by ring, almost all latency, and 0.24 ms by Rabenseifner's algorithm. A 16 GB one takes 0.66 s by ring and 3.2 s by recursive doubling. NCCL implements rings, pipelined trees (double binary trees since version 2.4) and, where the hardware allows, reductions inside the switches (two sheets on), and picks one per call from the size and the topology.</p>`,
    fig: {
      type: "calc", caption: "All-reduce time of four algorithms in the α–β model",
      inputs: [
        { id: "lp", label: "ranks p", min: 1, max: 14, step: 1, value: 10, fmt: (x) => Math.pow(2, x).toLocaleString("en-US") },
        { id: "n", label: "message n", min: 1e3, max: 1.6e10, value: 1e6, log: true, fmt: (x) => bytes(x) },
        { id: "a", label: "latency α (µs)", min: 1, max: 30, step: 1, value: 10, fmt: "int" },
        { id: "bw", label: "bandwidth", options: [{ label: "50 GB/s", value: 50e9 }, { label: "450 GB/s", value: 450e9 }], value: 50e9 },
      ],
      outputs: [
        { label: "ring", f: (v) => AB(v).ring, fmt: "time" },
        { label: "recursive doubling", f: (v) => AB(v).rd, fmt: "time" },
        { label: "Rabenseifner", f: (v) => AB(v).rab, fmt: "time" },
        { label: "binomial tree", f: (v) => AB(v).tree, fmt: "time" },
      ],
      note: (v) => { const t = AB(v), best = Object.entries({ ring: t.ring, "recursive doubling": t.rd, Rabenseifner: t.rab, "binomial tree": t.tree }).sort((x, y) => x[1] - y[1])[0][0]; return `Fastest here: <b>${best}</b>. Ring latency term ${fmt.time(2 * (Math.pow(2, v.lp) - 1) * v.a * 1e-6)}, bandwidth term ${fmt.time(2 * (1 - 1 / Math.pow(2, v.lp)) * v.n / v.bw)}.`; },
    },
    terms: [{ term: "Alpha-beta (α–β) cost model", def: "A cost model for messages: sending m bytes takes α + mβ, a fixed latency α plus m times the inverse bandwidth β." }],
    src: ["Thakur, Rabenseifner and Gropp, Optimization of Collective Communication Operations in MPICH, IJHPCA 2005", "Jeaugey, Massively Scale Your Deep Learning Training with NCCL 2.4, NVIDIA Technical Blog, 2019"],
  });

  const BUS = { ar: (n) => 2 * (n - 1) / n, rs: (n) => (n - 1) / n, ag: (n) => (n - 1) / n, bc: () => 1, a2a: (n) => (n - 1) / n };
  Deck.slide({
    id: "scale-busbw",
    title: "NCCL reports algorithm bandwidth and bus bandwidth",
    body: String.raw`<p>nccl-tests, NVIDIA's benchmark suite for NCCL, prints two bandwidths per collective. <b>Algorithm bandwidth</b> is the buffer size over the time, algbw = S/t. It is what a training step sees, but for an all-reduce it falls as ranks are added, because each rank must move 2(N − 1)/N · S bytes, more than S once N > 2.</p>
<p><b>Bus bandwidth</b> rescales it to the bytes each rank puts on its link in the bandwidth-optimal algorithm: busbw = algbw × 2(N − 1)/N for all-reduce; × (N − 1)/N for reduce-scatter, all-gather and all-to-all; × 1 for broadcast and reduce. A good ring implementation keeps busbw constant as N changes, and it can be compared directly with the link's hardware speed. The factor is applied whatever algorithm ran, so with in-network reduction busbw can exceed the link speed.</p>
<p>Example: an all-reduce of 4 GB on 8 GPUs that takes 20 ms has an algbw of 200 GB/s and a busbw of 200 × 2 × 7/8 = 350 GB/s, 78% of NVLink's 450 GB/s per direction.</p>`,
    fig: {
      type: "calc", caption: "From a measured time to algbw and busbw",
      inputs: [
        { id: "op", label: "collective", options: [{ label: "all-reduce", value: "ar" }, { label: "reduce-scatter", value: "rs" }, { label: "all-gather", value: "ag" }, { label: "broadcast", value: "bc" }, { label: "all-to-all", value: "a2a" }], value: "ar" },
        { id: "ln", label: "ranks N", min: 1, max: 6, step: 1, value: 3, fmt: (x) => String(Math.pow(2, x)) },
        { id: "S", label: "buffer S", options: [{ label: "256 MB", value: 256e6 }, { label: "1 GB", value: 1e9 }, { label: "4 GB", value: 4e9 }, { label: "16 GB", value: 16e9 }], value: 4e9 },
        { id: "t", label: "measured time (ms)", min: 1, max: 500, step: 1, value: 20, fmt: "int" },
      ],
      outputs: [
        { label: "algbw = S / t", f: (v) => v.S / (v.t / 1e3) / 1e9, fmt: (x) => `${x.toFixed(1)} GB/s` },
        { label: "busbw", f: (v) => (v.S / (v.t / 1e3) / 1e9) * BUS[v.op](Math.pow(2, v.ln)), fmt: (x) => `${x.toFixed(1)} GB/s` },
        { label: "busbw ÷ 450 GB/s", f: (v) => (v.S / (v.t / 1e3)) * BUS[v.op](Math.pow(2, v.ln)) / 450e9, fmt: "pct" },
      ],
      note: (v) => `Factor for this collective at N = ${Math.pow(2, v.ln)}: ${BUS[v.op](Math.pow(2, v.ln)).toFixed(3)}. Compare with 450 GB/s inside an H100 server and 50 GB/s per GPU between servers.`,
    },
    terms: [{ term: "Bus bandwidth (busbw)", def: "NCCL's normalized bandwidth: algorithm bandwidth S/t times a factor per collective (2(N − 1)/N for all-reduce), comparable with the hardware link speed at any N." }],
    src: ["NVIDIA nccl-tests documentation, PERFORMANCE.md (algbw and busbw)"],
  });

  Deck.slide({
    id: "scale-links",
    title: "NVLink inside a server, InfiniBand or RoCE between servers",
    body: String.raw`<p>Inside an HGX H100 server, eight GPUs connect through four NVSwitch chips. Each GPU has 18 NVLink 4 links, 450 GB/s per direction in total (Part 6), and reaches every other GPU in the server at that rate. Between servers each GPU has its own 400 Gb/s network adapter: 50 GB/s per direction, 9 times less.</p>
<p>NVIDIA's DGX SuperPOD reference architecture wires these adapters <b>rail-optimized</b>: GPU i of every server connects to the same leaf switch, rail i, so GPUs with the same local index are one switch apart, and collectives across servers can use all 8 rails at once.</p>
<p>Llama 3 405B's network is organized by racks instead: RoCE (Ethernet with remote direct memory access) at 400 Gb/s per GPU, 16 GPUs (two servers) under one top-of-rack switch; 192 racks form a pod of 3,072 GPUs with full bisection bandwidth; 8 pods form a 24K-GPU cluster through aggregation switches that are oversubscribed 1:7, so traffic between pods gets at most a seventh of the bandwidth. Meta's scheduler and parallel layout keep the heaviest traffic inside a pod.</p>`,
    fig: {
      type: "diagram", caption: "Two 8-GPU servers on a rail-optimized network (two of eight rails drawn); each GPU reaches its NIC through a PCIe switch", w: 15, h: 7.4, unit: 40,
      groups: [{ x: 0.2, y: 0.4, w: 6.8, h: 4.3, label: "server 1" }, { x: 8, y: 0.4, w: 6.8, h: 4.3, label: "server 2" }],
      nodes: [
        { id: "g1", x: 0.6, y: 0.9, w: 6, h: 0.9, label: "GPU 0 … GPU 7", sub: "H100 SXM" },
        { id: "s1", x: 0.6, y: 2.6, w: 2.8, h: 1.6, label: "4 × NVSwitch", sub: "450 GB/s per GPU", kind: "blue" },
        { id: "n1", x: 3.8, y: 2.6, w: 2.8, h: 1.6, label: "8 NICs", sub: "400 Gb/s each", kind: "ghost" },
        { id: "g2", x: 8.4, y: 0.9, w: 6, h: 0.9, label: "GPU 0 … GPU 7", sub: "H100 SXM" },
        { id: "s2", x: 8.4, y: 2.6, w: 2.8, h: 1.6, label: "4 × NVSwitch", sub: "450 GB/s per GPU", kind: "blue" },
        { id: "n2", x: 11.6, y: 2.6, w: 2.8, h: 1.6, label: "8 NICs", sub: "400 Gb/s each", kind: "ghost" },
        { id: "r0", x: 2.6, y: 5.9, w: 3.6, h: 1.0, label: "rail 0 switch", kind: "dark" },
        { id: "r7", x: 8.8, y: 5.9, w: 3.6, h: 1.0, label: "rail 7 switch", kind: "dark" },
      ],
      edges: [
        { from: "g1", to: "s1", both: true, label: "NVLink", dx: -26 }, { from: "g1", to: "n1", label: "PCIe", dx: 22 },
        { from: "g2", to: "s2", both: true, label: "NVLink", dx: -26 }, { from: "g2", to: "n2", label: "PCIe", dx: 22 },
        { from: "n1", to: "r0" }, { from: "n2", to: "r0" }, { from: "n1", to: "r7" }, { from: "n2", to: "r7" },
      ],
      notes: [{ x: 4.2, y: 5.55, text: "50 GB/s per GPU", anchor: "end" }],
    },
    terms: [
      { term: "Rail-optimized network", def: "A cluster network in which NIC i of every server connects to the same leaf switch (rail i), so same-index GPUs across servers are one switch hop apart." },
      { term: "Oversubscription", def: "The ratio of bandwidth entering a switch layer from below to the bandwidth leaving it upward; 1:7 means traffic crossing that layer can get only a seventh of full bandwidth." },
    ],
    src: ["NVIDIA DGX H100 User Guide (eight ConnectX-7 400 Gb/s compute ports)", "NVIDIA DGX SuperPOD Reference Architecture, DGX H100 (rail-optimized compute fabric)", "Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.1)"],
  });

  Deck.slide({
    id: "scale-innet",
    title: "In-network reduction: the switch does the additions",
    body: String.raw`<p>The ring's 2(N − 1)/N · S bytes per rank is a lower bound only when the ranks do the additions. <b>In-network reduction</b> moves them into the switches. With NVLink SHARP (NVLS), on Hopper GPUs with third-generation NVSwitch, each GPU sends its buffer into the switch once; the switch adds the N contributions and multicasts the sum back. Each GPU then moves about S per direction instead of 2(N − 1)/N · S, which is 1.75 S for 8 GPUs. InfiniBand switches with SHARP do the same between servers.</p>
<p>NCCL's algorithm list (<code>NCCL_ALGO</code>) now reads: Ring and Tree; CollNetDirect and CollNetChain (2.14), which offload to network switches through a plugin; NVLS (2.17) inside a server and NVLSTree (2.18), NVLS inside servers with trees between them; and PAT (2.23). NVLS is on by default where the hardware supports it (<code>NCCL_NVLS_ENABLE</code>).</p>
<p>nccl-tests still multiplies algbw by 2(N − 1)/N, so an 8-GPU all-reduce that reaches an algbw of 300 GB/s through NVLS reports a busbw of 525 GB/s on 450 GB/s links.</p>`,
    fig: {
      type: "calc", caption: "Bytes each GPU sends for one all-reduce, with the GPUs adding (ring) or the switch adding (NVLS)",
      inputs: [
        { id: "n", label: "GPUs", options: [2, 4, 8].map((x) => ({ label: String(x), value: x })), value: 8 },
        { id: "S", label: "buffer S", options: [{ label: "1 GB", value: 1e9 }, { label: "16 GB", value: 16e9 }], value: 16e9 },
      ],
      outputs: [
        { label: "ring, per GPU", f: (v) => 2 * (v.n - 1) / v.n * v.S, fmt: "bytes" },
        { label: "NVLS, per GPU (about)", f: (v) => v.S, fmt: "bytes" },
        { label: "ring at 450 GB/s", f: (v) => 2 * (v.n - 1) / v.n * v.S / 450e9, fmt: "time" },
        { label: "NVLS at 450 GB/s", f: (v) => v.S / 450e9, fmt: "time" },
        { label: "busbw reported for NVLS", f: (v) => 450 * 2 * (v.n - 1) / v.n, fmt: (x) => `${x.toFixed(0)} GB/s` },
      ],
      note: "Ideal times on one direction of NVLink; the last tile is what nccl-tests would print if NVLS ran at full link speed.",
    },
    terms: [{ term: "In-network reduction", def: "Performing the additions of a reduction inside the switches (NVLink SHARP, InfiniBand SHARP) so each GPU sends its data once, about S bytes, instead of 2(N − 1)/N · S." }],
    src: ["NVIDIA NCCL User Guide, environment variables (NCCL_ALGO versions, NCCL_NVLS_ENABLE)", "Patarasuk and Yuan, Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations, 2009 (Lemma 4 and its assumptions)", "Graham et al., Scalable Hierarchical Aggregation Protocol (SHArP), 2016", "NVIDIA nccl-tests documentation, PERFORMANCE.md"],
  });

  Deck.slide({
    id: "scale-overlap",
    title: "Communication hides behind computation if it can start early",
    body: String.raw`<p>NCCL runs each collective as a kernel on its own CUDA stream, so a GPU can compute and communicate at the same time. In the backward pass the last layer's gradients are ready first. Their all-reduce can start while earlier layers are still computing, and only the communication still running after the last backward kernel lengthens the step.</p>
<p>In the plate, 4 layers take 6 ms each to run backward and each layer's all-reduce takes 4 ms. One after the other, the backward pass and its communication take 24 + 16 = 40 ms. Overlapped, the work ends at 28 ms: only the first layer's all-reduce is exposed.</p>
<p>Overlap is not free. Communication kernels occupy SMs and HBM bandwidth while they run: DeepSeek-V3 dedicated 20 of the H800's 132 SMs to its own all-to-all kernels. And a tensor-parallel all-reduce is hard to hide at all, because the very next matrix multiply needs its result.</p>`,
    fig: {
      type: "timeline", caption: "Backward pass of 4 layers with each layer's gradient all-reduce overlapped", rows: ["backward", "all-reduce"], total: 30, unit: "ms",
      items: [
        { row: 0, start: 0, len: 6, label: "layer 4", kind: "compute" }, { row: 0, start: 6, len: 6, label: "layer 3", kind: "compute" },
        { row: 0, start: 12, len: 6, label: "layer 2", kind: "compute" }, { row: 0, start: 18, len: 6, label: "layer 1", kind: "compute" },
        { row: 1, start: 6, len: 4, label: "AR 4", kind: "comm", tip: "6–10 ms, hidden" }, { row: 1, start: 12, len: 4, label: "AR 3", kind: "comm", tip: "12–16 ms, hidden" },
        { row: 1, start: 18, len: 4, label: "AR 2", kind: "comm", tip: "18–22 ms, hidden" }, { row: 1, start: 24, len: 4, label: "AR 1", kind: "comm", tip: "24–28 ms, exposed" },
      ],
    },
    check: {
      q: "A ring all-reduce of a 1 GB buffer runs on 8 GPUs. How many bytes does each GPU send?",
      options: ["0.875 GB", "1.75 GB", "2 GB", "7 GB"], answer: 1,
      why: "2(N − 1)/N × S = 2 × 7/8 × 1 GB = 1.75 GB: 0.875 GB in the reduce-scatter and 0.875 GB in the all-gather. 2 GB is the limit for large N; 7 GB would be sending the whole buffer to each of the 7 other GPUs.",
    },
    src: ["DeepSeek-AI, DeepSeek-V3 Technical Report, 2024 (Section 3.2.2)", "NVIDIA NCCL User Guide (CUDA stream semantics)"],
  });

  // ================================================================== 7.5
  Deck.chapter({ title: "Data parallelism" });

  const DPN = [{ label: "8B", P: 8e9 }, { label: "70B", P: 70e9 }, { label: "405B", P: 405e9 }];
  Deck.slide({
    id: "scale-dp",
    title: "Data parallelism: replicas compute on different data, then average gradients",
    body: String.raw`<p>In <b>data parallelism</b> each of N GPUs holds a full copy of the model and processes 1/N of the global batch. The gradient of the batch's mean loss is the mean of the replicas' gradients, so one all-reduce (sum, then divide by N) gives every replica the same gradient. All replicas apply the same update and stay identical.</p>
<p>Its cost has a simple shape. For Ψ parameters each GPU sends about 2S bytes per step, with S = 2Ψ bytes of BF16 gradients, and computes 6ΨT FLOPs for its T tokens. At F FLOP/s and a link bandwidth of B bytes/s, communication over computation is (4Ψ/B) ÷ (6ΨT/F) = 2F/(3BT). The model size cancels.</p>
<p>At 400 TFLOP/s and 50 GB/s between servers that is 16% for T = 32,768 tokens per GPU per step (four 8K sequences) and 2.0% for T = 262,144. Over NVLink the first case is 1.8%. Data parallelism stays cheap while each GPU has enough tokens per step, and overlap (last sheet) hides much of the rest.</p>`,
    fig: {
      type: "calc", caption: "Gradient all-reduce against compute for one data-parallel step (ring, large N; 400 TFLOP/s per GPU)",
      inputs: [
        { id: "m", label: "model", options: DPN.map((d, i) => ({ label: d.label, value: i })), value: 0 },
        { id: "lt", label: "tokens per GPU per step", min: 12, max: 20, step: 1, value: 15, fmt: (x) => Math.pow(2, x).toLocaleString("en-US") },
        { id: "bw", label: "link", options: [{ label: "InfiniBand 50 GB/s", value: 50e9 }, { label: "NVLink 450 GB/s", value: 450e9 }], value: 50e9 },
      ],
      outputs: [
        { label: "compute 6ΨT / F", f: (v) => 6 * DPN[v.m].P * Math.pow(2, v.lt) / 400e12, fmt: "time" },
        { label: "all-reduce 4Ψ / B", f: (v) => 4 * DPN[v.m].P / v.bw, fmt: "time" },
        { label: "communication ÷ compute", f: (v) => 2 * 400e12 / (3 * v.bw * Math.pow(2, v.lt)), fmt: "pct1" },
      ],
    },
    terms: [{ term: "Data parallelism", def: "Training with a full model replica per GPU (or group), each on different data, with gradients averaged by an all-reduce before every update." }],
    src: ["Li et al., PyTorch Distributed: Experiences on Accelerating Data Parallel Training, 2020"],
  });

  const BK = (v) => {
    const p = Math.pow(2, v.lp), S = 16e9, nb = Math.ceil(S / v.b), lat = 2 * (p - 1) * v.a * 1e-6, bw = 2 * ((p - 1) / p) * v.b / (p > 8 ? 50e9 : v.bw);
    return { nb, lat, bw, per: lat + bw, tot: nb * (lat + bw) };
  };
  Deck.slide({
    id: "scale-bucket",
    title: "Gradient buckets start the all-reduce during the backward pass",
    body: String.raw`<p>One all-reduce per parameter tensor would pay the latency α hundreds of times per step; one all-reduce for everything could not start until the backward pass ends. PyTorch's DistributedDataParallel groups gradients into <b>buckets</b>, 25 MiB by default (<code>bucket_cap_mb</code>), in roughly the reverse order of the forward pass, and launches an asynchronous all-reduce as soon as every gradient in a bucket is ready.</p>
<p>The bucket size trades latency against overlap. The 16 GB of an 8B model's BF16 gradients make 611 buckets of 25 MiB. On 8 GPUs over NVLink, a ring all-reduce of one bucket spends 70 µs on latency (α = 5 µs) and 0.10 ms moving data. On 1,024 GPUs at 50 GB/s, the latency term is 10.2 ms per bucket against 1.05 ms of data: small buckets on many ranks are latency-bound. That is one reason NCCL switches to tree algorithms for such calls and large jobs use bigger buckets or a hierarchical reduction.</p>`,
    fig: {
      type: "calc", caption: "Bucketed ring all-reduce of 16 GB of gradients (an 8B model in BF16)",
      inputs: [
        { id: "b", label: "bucket size", min: 1e6, max: 4e9, value: 25 * 2 ** 20, log: true, step: 0.0001, fmt: (x) => bytes(x) },
        { id: "lp", label: "ranks p", min: 1, max: 10, step: 1, value: 3, fmt: (x) => Math.pow(2, x).toLocaleString("en-US") },
        { id: "bw", label: "link", options: [{ label: "NVLink 450 GB/s", value: 450e9 }, { label: "InfiniBand 50 GB/s", value: 50e9 }], value: 450e9 },
        { id: "a", label: "latency α (µs)", min: 1, max: 20, step: 1, value: 5, fmt: "int" },
      ],
      outputs: [
        { label: "buckets", f: (v) => BK(v).nb, fmt: "int" },
        { label: "per bucket", f: (v) => BK(v).per, fmt: "time" },
        { label: "latency share", f: (v) => BK(v).lat / BK(v).per, fmt: "pct" },
        { label: "all buckets", f: (v) => BK(v).tot, fmt: "time" },
      ],
      note: "More than 8 ranks span servers, so the ring runs at 50 GB/s whatever the link setting.",
    },
    terms: [{ term: "Gradient bucket", def: "A group of gradient tensors reduced by one collective call as soon as all of them are computed, so communication overlaps the rest of the backward pass." }],
    src: ["PyTorch documentation, torch.nn.parallel.DistributedDataParallel (bucket_cap_mb = 25)", "Li et al., PyTorch Distributed: Experiences on Accelerating Data Parallel Training, 2020"],
  });

  Deck.slide({
    id: "scale-zero",
    title: "ZeRO shards the training state across data-parallel ranks",
    body: String.raw`<p>Plain data parallelism stores all 16 bytes per parameter on every rank: N<sub>d</sub> identical copies. <b>ZeRO</b> (Rajbhandari et al.) removes the copies in three stages. For Ψ parameters on N<sub>d</sub> ranks, the bytes per GPU are:</p>
<ul><li>stage 1, optimizer state sharded: \(4\Psi + 12\Psi/N_d\);</li>
<li>stage 2, gradients sharded too: \(2\Psi + 14\Psi/N_d\);</li>
<li>stage 3, weights sharded too: \(16\Psi/N_d\).</li></ul>
<p>The paper's example, 7.5B parameters on 64 GPUs: 120 GB per GPU without ZeRO, then 31.4 GB, 16.6 GB and 1.9 GB.</p>
<p>Stages 1 and 2 move the same bytes as plain data parallelism. The gradient all-reduce becomes a reduce-scatter, since each rank only needs the summed gradient of its own shard to update it, followed by an all-gather of the updated weights: 2Ψ values per step, 4Ψ bytes in BF16. Stage 3 must also gather the weights for the backward pass: 3Ψ values (6Ψ bytes), 1.5 times as much, in exchange for memory that falls as 1/N<sub>d</sub>.</p>`,
    fig: { type: "scale-zero", caption: "Model state per GPU for each ZeRO stage; the red line is one H100" },
    terms: [{ term: "ZeRO", def: "Zero Redundancy Optimizer: sharding optimizer state (stage 1), gradients (stage 2) and weights (stage 3) across data-parallel ranks instead of replicating them." }],
    src: ["Rajbhandari et al., ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, 2020 (Sections 5 and 7, Figure 1)"],
  });

  Deck.slide({
    id: "scale-fsdp",
    title: "FSDP gathers each layer's weights just before using them",
    body: String.raw`<p>PyTorch's <b>FSDP</b> (fully sharded data parallel) implements ZeRO stage 3. It flattens the parameters of each unit, typically one transformer block, and gives each rank a 1/N<sub>d</sub> shard. Before a unit's forward pass an all-gather assembles its full weights, and afterwards they are freed. The backward pass gathers them again, computes the gradients, and reduce-scatters those so that each rank keeps the summed gradient of its own shard, which its optimizer then updates.</p>
<p>Prefetching the next unit's all-gather while the current unit computes hides most of the traffic (the plate). Memory per GPU is the shards plus one or two full units. Every parameter crosses the network three times per step. When a group spans slow links, hybrid sharding shards within a group of servers and replicates across groups.</p>
<p>Llama 3's data-parallel dimension was FSDP, placed outermost because prefetched all-gathers and deferred reduce-scatters tolerate latency. Meta did not reshard after the forward pass, to avoid the second all-gather: each GPU kept its pipeline stage's weights gathered until the backward pass, and parameters crossed the network twice per step. Gradients were reduce-scattered in FP32.</p>`,
    fig: {
      type: "timeline", caption: "FSDP over 3 units: all-gathers (AG) prefetched ahead of compute, reduce-scatters (RS) after each backward", rows: ["compute", "network"], total: 30, unit: "ms",
      items: [
        { row: 0, start: 1.5, len: 3, label: "fwd 1", kind: "compute" }, { row: 0, start: 4.5, len: 3, label: "fwd 2", kind: "compute" }, { row: 0, start: 7.5, len: 3, label: "fwd 3", kind: "compute" },
        { row: 0, start: 10.5, len: 6, label: "bwd 3", kind: "compute" }, { row: 0, start: 16.5, len: 6, label: "bwd 2", kind: "compute" }, { row: 0, start: 22.5, len: 6, label: "bwd 1", kind: "compute" },
        { row: 1, start: 0, len: 1.5, label: "AG", kind: "comm", tip: "gather unit 1 (exposed)" }, { row: 1, start: 1.5, len: 1.5, label: "AG", kind: "comm", tip: "gather unit 2" },
        { row: 1, start: 4.5, len: 1.5, label: "AG", kind: "comm", tip: "gather unit 3" }, { row: 1, start: 7.5, len: 1.5, label: "AG", kind: "comm", tip: "gather unit 3 again for backward" },
        { row: 1, start: 10.5, len: 1.5, label: "AG", kind: "comm", tip: "gather unit 2 for backward" }, { row: 1, start: 16.5, len: 1.5, label: "RS", kind: "s4", tip: "reduce-scatter unit 3 gradients" },
        { row: 1, start: 18, len: 1.5, label: "AG", kind: "comm", tip: "gather unit 1 for backward" }, { row: 1, start: 22.5, len: 1.5, label: "RS", kind: "s4", tip: "reduce-scatter unit 2 gradients" },
        { row: 1, start: 28.5, len: 1.5, label: "RS", kind: "s4", tip: "reduce-scatter unit 1 gradients (exposed)" },
      ],
    },
    terms: [{ term: "FSDP", def: "Fully sharded data parallel (PyTorch): ZeRO stage 3, with each unit's weights all-gathered before use and its gradients reduce-scattered after the backward pass." }],
    src: ["Zhao et al., PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel, 2023", "Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.2)"],
  });

  Deck.slide({
    id: "scale-batch",
    title: "The useful global batch has a ceiling, so runs ramp it up",
    body: String.raw`<p>Data parallelism adds GPUs by adding replicas. At a fixed per-GPU batch that grows the global batch; at a fixed global batch each GPU gets fewer tokens and works less efficiently. Llama 3 kept 16M tokens per step when it went from 8,192 to 16,384 GPUs; each data-parallel group's share halved from 32 to 16 sequences, and MFU fell from 43% to 41%.</p>
<p>A larger batch averages more examples, so its gradient is less noisy, but beyond a point it no longer cuts the number of steps needed in proportion. McCandlish et al. tie that point to the gradient noise scale, which grows during training as the loss falls. Runs therefore start with smaller batches. GPT-3 raised its batch linearly from 32K tokens over the first 4 to 12 billion tokens. Llama 3 405B started at 4M tokens in 4,096-token sequences, doubled to 8M (sequences of 8,192) after 252M tokens and to 16M after 2.87T.</p>`,
    fig: {
      type: "chart", caption: "Llama 3 405B's batch size against tokens trained (log scale)",
      x: { domain: [1e8, 2e13], type: "log", ticks: [1e8, 1e9, 1e10, 1e11, 1e12, 1e13], label: "tokens trained", fmt: (v) => human(v) },
      y: { domain: [0, 18], ticks: [0, 4, 8, 12, 16], label: "batch (M tokens)" },
      series: [{ points: [[1e8, 4], [2.52e8, 4], [2.52e8, 8], [2.87e12, 8], [2.87e12, 16], [1.56e13, 16]], cls: "s1" }],
      dots: [{ x: 2.52e8, y: 8, r: 4, cls: "ink", tip: "252M tokens: 4M → 8M", label: "252M", dx: 6, dy: -6 }, { x: 2.87e12, y: 16, r: 4, cls: "ink", tip: "2.87T tokens: 8M → 16M", label: "2.87T", dx: -8, dy: -8, anchor: "end" }],
    },
    check: {
      q: "Llama 3 405B's first stage covered 252M tokens at about 4M tokens per step. About how many optimizer steps was that?",
      options: ["6", "60", "600", "60,000"], answer: 1,
      why: "252 × 10⁶ ÷ (1,024 sequences × 4,096 tokens = 4,194,304) = 60.1 steps. The small batch covers only the first few dozen updates; most of the run used 16M-token steps.",
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.4.1, Table 4)", "McCandlish et al., An Empirical Model of Large-Batch Training, 2018", "Brown et al., Language Models are Few-Shot Learners, 2020 (Appendix B)"],
  });

  // ================================================================== 7.6
  Deck.chapter({ title: "Model parallelism" });

  Deck.slide({
    id: "scale-tp",
    title: "Tensor parallelism splits the MLP by columns, then by rows",
    example: "tokens ids focus:4",
    body: String.raw`<p><b>Tensor parallelism</b> splits each weight matrix across t GPUs that all work on the same tokens. Megatron-LM splits the MLP's first matrix A (d × 4d) by columns: GPU i computes \(XA_i\), its own 1/t of the hidden units, and applies GELU locally, since GELU acts on each unit separately. The second matrix B is split by rows: GPU i multiplies its hidden units by \(B_i\) and holds a partial output. The partials sum to the output, so one all-reduce produces Y on every GPU.</p>
<p>Splitting A by rows instead would put a sum before the GELU, and GELU of a sum is not the sum of GELUs: a second all-reduce would be needed. The column-then-row order needs one all-reduce in the forward pass and one in the backward pass, for the input's gradient.</p>
<p>The plate runs our model's real layer-1 MLP at <code>WHERE</code> on 2 or 4 simulated GPUs; the partial outputs add up to the single-GPU output to within rounding.</p>`,
    math: { tex: String.raw`Y = \operatorname{GELU}(XA)\,B = \sum_{i=1}^{t} \operatorname{GELU}(XA_i)\,B_i`, note: String.raw`\(A = [A_1, \dots, A_t]\) split by columns, \(B\) by rows into \(B_1, \dots, B_t\)` },
    fig: { type: "tinylm", widget: "tp", caption: "Our layer-1 MLP split across simulated GPUs" },
    terms: [{ term: "Tensor parallelism", def: "Splitting each weight matrix of a layer across t GPUs that process the same tokens, with all-reduces (or reduce-scatter and all-gather) to combine partial results." }],
    src: ["Shoeybi et al., Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism, 2019 (Section 3)"],
  });

  const LL = { h: 16384, P: 3187671040 };
  Deck.slide({
    id: "scale-tp-attn",
    title: "Attention splits by heads; a layer needs four all-reduces",
    body: String.raw`<p>Attention heads are independent until the output projection, so tensor parallelism gives each GPU 1/t of the heads: its columns of W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> and its rows of W<sub>O</sub>. One all-reduce sums the partial outputs. With the MLP's, a layer needs 2 all-reduces of the full b × s × d activation in the forward pass and 2 in the backward pass. Llama 3 405B's 8 key-value heads split exactly over t = 8: each GPU gets 16 query heads and 1 key-value head.</p>
<p>For one 8,192-token sequence at d = 16,384 in BF16, each all-reduce is 268 MB, and the ring sends 2 × 7/8 × 268 = 470 MB per GPU: 1.04 ms over NVLink. The layer has 3.19 × 10⁹ weights, so its forward matrix multiplies on one of the 8 GPUs take 2 × 3.19 × 10⁹ × 8,192 / 8 = 6.5 × 10¹² FLOPs, 16 ms at 400 TFLOP/s. The two all-reduces add 2.1 ms, 13% if not overlapped. Over 50 GB/s InfiniBand they would take 19 ms, longer than the compute, which is why tensor parallelism stays inside one NVLink domain.</p>`,
    fig: {
      type: "calc", caption: "Tensor-parallel traffic of one Llama 3 405B layer in the forward pass (ring all-reduce, 400 TFLOP/s)",
      inputs: [
        { id: "ls", label: "tokens per micro-batch", min: 10, max: 17, step: 1, value: 13, fmt: (x) => Math.pow(2, x).toLocaleString("en-US") },
        { id: "t", label: "TP degree t", options: [2, 4, 8, 16].map((x) => ({ label: String(x), value: x })), value: 8 },
        { id: "bw", label: "link", options: [{ label: "NVLink 450 GB/s", value: 450e9 }, { label: "InfiniBand 50 GB/s", value: 50e9 }], value: 450e9 },
      ],
      outputs: [
        { label: "one all-reduce", f: (v) => 2 * Math.pow(2, v.ls) * LL.h, fmt: "bytes" },
        { label: "2 all-reduces", f: (v) => 2 * (2 * (v.t - 1) / v.t) * 2 * Math.pow(2, v.ls) * LL.h / (v.t > 8 ? 50e9 : v.bw), fmt: "time" },
        { label: "forward compute", f: (v) => 2 * LL.P * Math.pow(2, v.ls) / v.t / 400e12, fmt: "time" },
        { label: "traffic ÷ compute", f: (v) => (2 * (2 * (v.t - 1) / v.t) * 2 * Math.pow(2, v.ls) * LL.h / (v.t > 8 ? 50e9 : v.bw)) / (2 * LL.P * Math.pow(2, v.ls) / v.t / 400e12), fmt: "pct" },
      ],
      note: "t = 16 spans two servers, so it runs at 50 GB/s whatever the link setting. Layer weights: 2d² for W<sub>Q</sub> and W<sub>O</sub>, 2 × d × 1,024 for W<sub>K</sub> and W<sub>V</sub> (8 heads of 128), 3 × d × 53,248 for the SwiGLU MLP: 3,187,671,040 at d = 16,384.",
    },
    src: ["Shoeybi et al., Megatron-LM, 2019", "Dubey et al., The Llama 3 Herd of Models, 2024 (Table 3)"],
  });

  Deck.slide({
    id: "scale-sp",
    title: "Sequence parallelism splits the rest of the layer along the tokens",
    body: String.raw`<p>Tensor parallelism splits the matrix multiplies, but the norms, dropout and residual additions between them still run on the full b × s × h tensor on every GPU. Korthikanti et al. count what that leaves: \(sbh\,(10 + 24/t + 5as/(ht))\) bytes of activations per layer, where the 10sbh does not shrink with t.</p>
<p><b>Sequence parallelism</b> splits those parts along the sequence instead: each of the t GPUs normalizes s/t of the tokens. Each forward all-reduce splits into its halves: a reduce-scatter leaves each GPU the summed activations of its own s/t tokens for the norm and dropout, and an all-gather rebuilds the full tensor for the next matrix multiply; the backward pass mirrors them. No extra bytes move, and every activation is now split t ways: \(sbh\,(34 + 5as/h)/t\) per layer.</p>
<p>For GPT-3 175B at t = 8 and s = 2,048, that is 0.36 GB per layer per sequence instead of 0.58 GB. Recomputing the attention scores as well leaves 0.11 GB, 27 times less than the 2.87 GB with no parallelism.</p>`,
    fig: {
      type: "table", caption: "Activation memory of one GPT-3 175B layer, one 2,048-token sequence, t = 8", align: ["l", "l", "r"],
      cols: ["Scheme", "Bytes per layer", "GPT-3 175B"],
      rows: [
        ["no parallelism", "sbh(34 + 5as/h)", "2.87 GB"],
        ["tensor parallel", "sbh(10 + 24/t + 5as/(ht))", "0.58 GB"],
        ["tensor + sequence parallel", "sbh(34 + 5as/h)/t", "0.36 GB"],
        ["… + selective recomputation", "34sbh/t", "0.11 GB"],
      ],
      highlight: [2],
    },
    terms: [{ term: "Sequence parallelism", def: "Splitting the non-matrix parts of a tensor-parallel layer (norms, dropout) along the sequence, turning each all-reduce into a reduce-scatter and an all-gather." }],
    src: ["Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models, 2022 (Sections 4.2.1–4.2.2, Table 2)"],
  });

  Deck.slide({
    id: "scale-pp",
    title: "Pipeline parallelism splits the layers, and the pipeline has a bubble",
    example: "tokens ids",
    body: String.raw`<p><b>Pipeline parallelism</b> gives each of p GPUs a contiguous range of layers, a stage. Activations go forward from stage to stage and their gradients come back, as point-to-point messages of b × s × h values per micro-batch. To keep the stages busy, each step's batch is cut into m micro-batches that follow one another through the pipeline.</p>
<p>GPipe runs all m forward passes, then all m backward passes. With times \(t_f\) and \(t_b\) per micro-batch and stage, a step takes \((m + p - 1)(t_f + t_b)\). Stage i idles i forward slots before its first micro-batch arrives, \((p - 1 - i)(t_f + t_b)\) between its last forward and first backward, and i backward slots at the end: \((p - 1)(t_f + t_b)\) on every stage. That idle share, the <b>bubble</b>, is (p − 1)/(m + p − 1); Megatron-LM writes it relative to the ideal time, (p − 1)/m. With p = 16, m = 16 leaves 48% of the time idle and m = 64 leaves 19%.</p>
<p>GPipe holds every micro-batch's activations until the backward passes begin. The <b>1F1B</b> schedule (PipeDream-Flush) runs p − i − 1 forward passes on stage i, then alternates one forward with one backward: the same bubble, with at most p − i micro-batches held instead of m.</p>`,
    math: String.raw`\text{bubble} = \frac{p-1}{m+p-1}`,
    fig: { type: "tinylm", widget: "pp", caption: "GPipe and 1F1B schedules: who computes what, and when" },
    terms: [
      { term: "Pipeline parallelism", def: "Splitting a model's layers into consecutive stages on different GPUs and streaming micro-batches through them." },
      { term: "Pipeline bubble", def: "The idle time of pipeline stages while the pipeline fills and drains: (p − 1)/(m + p − 1) of a step for p stages and m micro-batches." },
      { term: "1F1B schedule", def: "A pipeline schedule that alternates one forward and one backward pass per stage after a warm-up, so a stage holds at most p micro-batches of activations." },
    ],
    src: ["Huang et al., GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism, 2019", "Narayanan et al., Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM, 2021 (Section 2.2)", "Narayanan et al., Memory-Efficient Pipeline-Parallel DNN Training (PipeDream-Flush), 2021"],
  });

  Deck.slide({
    id: "scale-interleave",
    title: "Interleaving shrinks the pipeline bubble",
    body: String.raw`<p>The <b>interleaved</b> schedule (Megatron-LM) gives each GPU v separate chunks of layers: with p = 4 and v = 2, GPU 0 holds stages 0 and 4 of 8. A micro-batch passes every GPU v times in smaller pieces, so filling and draining the pipeline takes 1/v as long, and the overhead falls to (p − 1)/(v·m). The price is v times as many point-to-point transfers, and Megatron-LM requires m to be a multiple of p.</p>
<p>With p = 16 and m = 16, the overhead 15/16 becomes 15/64 at v = 4: a bubble of 19% of the step instead of 48%.</p>
<p>Llama 3 used an interleaved schedule and gives its bubble ratio as (PP − 1)/(V·M), the same formula. It also made the number of consecutive micro-batches per stage tunable, so a step could have fewer micro-batches than stages, removed one layer from the first and last stages to balance the embedding and the output layer, and sent activations asynchronously.</p>`,
    fig: {
      type: "calc", caption: "Bubble and activation memory of three pipeline schedules",
      inputs: [
        { id: "p", label: "stages p", min: 2, max: 32, step: 1, value: 16, fmt: "int" },
        { id: "m", label: "micro-batches m", min: 1, max: 128, step: 1, value: 16, fmt: "int" },
        { id: "v", label: "chunks per GPU v", min: 1, max: 8, step: 1, value: 4, fmt: "int" },
      ],
      outputs: [
        { label: "GPipe and 1F1B bubble", f: (v) => (v.p - 1) / (v.m + v.p - 1), fmt: "pct1" },
        { label: "interleaved bubble", f: (v) => { const x = (v.p - 1) / (v.v * v.m); return x / (1 + x); }, fmt: "pct1" },
        { label: "micro-batches held, GPipe", f: (v) => v.m, fmt: "int" },
        { label: "micro-batches held, 1F1B", f: (v) => Math.min(v.p, v.m), fmt: "int" },
      ],
      note: "Bubble as a share of the step: overhead x relative to the ideal time gives x/(1 + x). The interleaved formula assumes m is a multiple of p, as in Megatron-LM.",
    },
    terms: [
      { term: "Interleaved schedule", def: "A pipeline schedule in which each GPU holds v non-adjacent chunks of layers, cutting the bubble to (p − 1)/(v·m) for v times more point-to-point communication." },
    ],
    src: ["Narayanan et al., Megatron-LM, 2021 (Section 2.2.2)", "Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.2)"],
  });

  Deck.slide({
    id: "scale-ep",
    title: "Expert parallelism moves tokens to their experts with all-to-all",
    body: String.raw`<p>A mixture-of-experts layer (Part 5) sends each token to k of E expert MLPs. <b>Expert parallelism</b> places different experts on different GPUs. Each GPU sends every token's hidden vector to the GPUs that hold its k experts (an all-to-all called dispatch), the experts run, and a second all-to-all returns their outputs (combine). The backward pass repeats both: four all-to-alls per MoE layer per micro-batch.</p>
<p>DeepSeek-V3 has 256 routed experts and 1 shared expert per MoE layer, k = 8 and hidden size 7,168, and spreads each layer's routed experts over 64 GPUs on 8 nodes. Dispatch carries each token to its 8 experts in FP8, 8 × 7,168 = 57,344 bytes if every copy travels separately; combine stays in BF16, 114,688 bytes. Each token may reach at most 4 nodes, so at most 4 copies cross InfiniBand (50 GB/s), the scarce link; NVLink inside a node (160 GB/s on DeepSeek's H800s, 3.2 times more) forwards them to the experts.</p>
<p>Load balance sets the speed, since the GPU with the most popular experts finishes last. GShard and Switch cap the tokens per expert with a capacity factor and drop the overflow; DeepSeek-V3 adjusts a per-expert bias in the router instead.</p>`,
    fig: {
      type: "calc", caption: "Dispatch traffic per GPU for one MoE layer, uniform routing",
      inputs: [
        { id: "lt", label: "tokens per GPU", min: 10, max: 16, step: 1, value: 12, fmt: (x) => Math.pow(2, x).toLocaleString("en-US") },
        { id: "k", label: "experts per token k", min: 1, max: 8, step: 1, value: 8, fmt: "int" },
        { id: "g", label: "expert-parallel GPUs", options: [8, 16, 32, 64].map((x) => ({ label: String(x), value: x })), value: 64 },
        { id: "by", label: "dispatch format", options: [{ label: "FP8", value: 1 }, { label: "BF16", value: 2 }], value: 1 },
      ],
      outputs: [
        { label: "bytes per token", f: (v) => v.k * 7168 * v.by, fmt: "int" },
        { label: "sent per GPU", f: (v) => Math.pow(2, v.lt) * v.k * 7168 * v.by * (v.g - 1) / v.g, fmt: "bytes" },
        { label: "if all over 50 GB/s", f: (v) => Math.pow(2, v.lt) * v.k * 7168 * v.by * (v.g - 1) / v.g / 50e9, fmt: "time" },
      ],
      note: "Hidden size 7,168 as in DeepSeek-V3. With uniform routing a fraction (G − 1)/G of the copies leaves the GPU; node-limited routing caps how many cross InfiniBand.",
    },
    terms: [{ term: "Expert parallelism", def: "Placing different experts of a mixture-of-experts layer on different GPUs and routing tokens to them with all-to-all communication." }],
    src: ["DeepSeek-AI, DeepSeek-V3 Technical Report, 2024 (Sections 3.2, 3.3.4 and 4.2)", "Lepikhin et al., GShard, 2020", "Fedus et al., Switch Transformers, 2021"],
  });

  Deck.slide({
    id: "scale-cp",
    title: "Context parallelism splits one long sequence across GPUs",
    body: String.raw`<p>At 131,072 tokens one sequence keeps 34sbh/t = 9.1 GB of activations per layer on each of 8 tensor-parallel GPUs, 73 GB for a stage of 8 layers. One micro-batch in flight fills the GPU, and a 16M-token step has only 128 such sequences. <b>Context parallelism</b> splits each sequence across c GPUs, so each holds 1/c of every activation and a pipeline can keep c times more micro-batches in flight. Only attention needs other tokens' keys and values. CP ranks hold the same weights, so their gradients are also reduced across the CP group.</p>
<p><b>Ring attention</b> (Liu et al.) passes key-value blocks around a ring of GPUs, computing on each block while the next is in flight. Llama 3 instead all-gathers K and V, exposing the all-gather; with grouped-query attention K and V have 8 heads against Q's 128 (together 1/8 of Q), and the all-gather moves O(s) data against O(s²) attention work.</p>
<p>Late tokens attend to more keys than early ones, so Llama 3 cuts the sequence into 2c chunks and gives rank i chunks i and 2c − 1 − i, which balances the work. Its 131,072-token stage used TP 8 × CP 16 × PP 16 × DP 8 = 16,384 GPUs.</p>`,
    fig: {
      type: "table", caption: "Causal attention work per rank for c = 4 ranks and 8 chunks (chunk j costs j + ½ chunk-pairs)", align: ["l", "l", "r", "l", "r"], minWidth: 520,
      cols: ["Rank", "Contiguous chunks", "Work", "Llama 3 chunks", "Work"],
      rows: [
        ["0", "0, 1", "2", "0, 7", "8"],
        ["1", "2, 3", "6", "1, 6", "8"],
        ["2", "4, 5", "10", "2, 5", "8"],
        ["3", "6, 7", "14", "3, 4", "8"],
      ],
      note: "With contiguous chunks the last rank does 14 units while the first does 2, and every rank waits for the last: 1.75 times the balanced time.",
    },
    terms: [
      { term: "Context parallelism", def: "Splitting one sequence across GPUs along the token dimension, with keys and values exchanged (all-gathered or passed around a ring) for attention." },
      { term: "Ring attention", def: "Context-parallel attention that passes key-value blocks around a ring of GPUs, overlapping each transfer with blockwise attention on the previous block." },
    ],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.2, Table 4)", "Liu et al., Ring Attention with Blockwise Transformers for Near-Infinite Context, 2023"],
  });

  Deck.slide({
    id: "scale-mesh",
    title: "Llama 3.1 405B: TP 8 × PP 16 × DP 128 on 16,384 GPUs",
    example: "tokens ids",
    body: String.raw`<p>Meta orders the dimensions [TP, CP, PP, DP] from innermost to outermost. The innermost needs the most bandwidth and the lowest latency and stays inside a server; the outermost may cross several switch hops.</p>
<ul><li>TP = 8 fills one server's NVLink domain: four all-reduces per layer per micro-batch.</li>
<li>PP = 16 spans 16 servers with about 8 of the 126 layers each, and sends one activation tensor per micro-batch across each stage boundary.</li>
<li>DP = 128 is FSDP. TP and PP cut the model into 8 × 16 = 128 pieces; FSDP shards each piece's weights, gradients and optimizer state across that piece's 128 data-parallel copies. Meta did not reshard after the forward pass, so each GPU keeps its piece's BF16 weights gathered through the step: 2 × 405 × 10⁹ / 128 bytes = 5.9 GiB.</li></ul>
<p>The 16 bytes × 405 × 10⁹ of model state therefore come to 6.48 × 10¹² / 16,384 = 396 MB per GPU; most of the 80 GiB holds activations and gathered weights. Meta reports 400 TFLOP/s per GPU (41% MFU) in this configuration, 430 (43%) on 8,192 GPUs with DP 64, and 380 (38%) in the 128K-context stage.</p>`,
    fig: { type: "tinylm", widget: "mesh", caption: "One model replica as a 16 × 8 grid of GPUs, times 128 replicas; hover a GPU to see its groups" },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.2, Table 4, Figure 5)"],
  });

  Deck.slide({
    id: "scale-layout",
    title: "Choosing a layout: TP, then CP, then PP, then DP",
    body: String.raw`<p>A layout factors the GPU count, G = TP × CP × PP × DP. The usual order of decisions:</p>
<ol><li>TP as large as needed, but within one server's NVLink domain (8): its traffic grows relative to compute as each GPU's slice of the matrices gets thinner.</li>
<li>CP only for long sequences, when one sequence's activations do not fit.</li>
<li>PP until weights and activations fit, with enough micro-batches (m several times p) to keep the bubble small.</li>
<li>DP with every GPU left, sharded with FSDP. The global batch caps it: 16M tokens at 8,192 per sequence is 2,048 sequences, so DP × m = 2,048 with one-sequence micro-batches.</li></ol>
<p>The plate applies these rules to Llama 3.1 405B, with memory estimated from this part's formulas: 16 bytes per parameter spread over all GPUs, Korthikanti's 34sbh/t per layer for 1F1B's in-flight micro-batches, and the stage's BF16 weights, which Llama 3 keeps gathered through the step. Try TP 16, or the 131,072-token sequence without CP. Meta built a memory estimator and a performance-projection tool to search this space.</p>`,
    fig: { type: "scale-layout", caption: "A parallel layout for Llama 3.1 405B at 16M tokens per step: memory, bubble and tensor-parallel traffic (estimates)" },
    check: {
      q: "Llama 3's 131,072-token stage kept 16M tokens per step with DP = 8. How many sequences did each data-parallel group process per step?",
      options: ["1", "8", "16", "128"], answer: 2,
      why: "16,777,216 / 131,072 = 128 sequences per step, divided over 8 data-parallel groups: 16 each, the 'batch size/DP' of Meta's Table 4. Each sequence is itself split over 16 CP ranks.",
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.2, Table 4)", "Korthikanti et al., 2022", "Narayanan et al., Megatron-LM, 2021"],
  });

  // ================================================================== 7.7
  Deck.chapter({ title: "Running a training job" });

  Deck.slide({
    id: "scale-data",
    title: "The data pipeline: crawl, filter, deduplicate, mix, tokenize",
    example: "tokens ids",
    body: String.raw`<p>Llama 3's 15T tokens started as web crawls. Meta removed domains likely to hold unsafe content or personal data, extracted text from HTML with its own parser, and deduplicated at three levels: URLs; whole documents, with MinHash; and lines, dropping any line that appeared more than 6 times in a bucket of 30M documents. Heuristic filters and model-based quality classifiers followed, then separate pipelines for code, math and reasoning, and multilingual text. The final mix was about 50% general knowledge, 25% math and reasoning, 17% code and 8% multilingual tokens, tokenized into shards that the data loaders read in a fixed, resumable order.</p>
<p><b>MinHash</b> estimates the Jaccard similarity of two documents' sets of word n-grams from k minima: under a random permutation of all possible n-grams (a min-wise independent family), two documents share the minimum with probability equal to their Jaccard similarity; practical hash functions approximate such permutations. Locality-sensitive hashing then buckets the signatures so that only likely duplicates are compared. The plate runs it on two short query logs.</p>`,
    fig: { type: "tinylm", widget: "data", caption: "The pipeline's stages, and MinHash on two query logs" },
    terms: [{ term: "MinHash", def: "A signature of k minimum hash values of a document's n-gram set; the fraction of matching minima between two documents estimates their Jaccard similarity." }],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.1)", "Broder, On the Resemblance and Containment of Documents, 1997"],
  });

  Deck.slide({
    id: "scale-ckpt",
    title: "Checkpoints: what to save, how big, how fast",
    body: String.raw`<p>A checkpoint must let the job continue exactly where it stopped: the FP32 master weights and both Adam moments (12 bytes per parameter; the BF16 weights are a rounding of the master copy), the step count and position in the learning-rate schedule, the data loaders' positions and the random-number generators' states. For Llama 3.1 405B that is 12 × 405 × 10⁹ = 4.86 TB.</p>
<p>Every GPU writes its own shard in parallel: 4.86 TB over 16,384 GPUs is 297 MB each, and Meta reports per-GPU checkpoint state from 1 MB to 4 GB. Llama 3's storage (Meta's Tectonic file system) sustained 2 TB/s with 7 TB/s peaks, so 4.86 TB needs 2.4 s of the sustained rate if the writes are spread evenly. They are not: all GPUs write at once, and Meta names these bursts, which saturate the storage fabric, as a major challenge.</p>
<p>Asynchronous checkpointing first copies each shard to host memory, which pauses training only for the copy, then writes it to storage in the background while the next steps run.</p>`,
    fig: {
      type: "calc", caption: "Size of a sharded checkpoint and its write time at full storage bandwidth",
      inputs: [
        { id: "N", label: "parameters", min: 1e9, max: 2e12, value: 405e9, log: true, step: 0.001, fmt: (x) => human(x) },
        { id: "b", label: "bytes per parameter saved", options: [{ label: "12 (master + Adam)", value: 12 }, { label: "14 (+ BF16 weights)", value: 14 }], value: 12 },
        { id: "lg", label: "GPUs writing", min: 6, max: 15, step: 1, value: 14, fmt: (x) => Math.pow(2, x).toLocaleString("en-US") },
        { id: "bw", label: "storage bandwidth (TB/s)", min: 0.5, max: 10, step: 0.5, value: 2, fmt: "f1" },
      ],
      outputs: [
        { label: "checkpoint size", f: (v) => v.N * v.b, fmt: "bytes" },
        { label: "per GPU", f: (v) => v.N * v.b / Math.pow(2, v.lg), fmt: "bytes" },
        { label: "write time", f: (v) => v.N * v.b / (v.bw * 1e12), fmt: "time" },
      ],
    },
    terms: [{ term: "Checkpoint", def: "A saved copy of everything needed to resume training exactly: master weights, optimizer state, step and schedule position, data-loader positions and random-number states." }],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.1)", "PyTorch documentation, torch.distributed.checkpoint (async_save)"],
  });

  Deck.slide({
    id: "scale-failures",
    title: "At 16,384 GPUs something fails every few hours",
    example: "tokens ids",
    body: String.raw`<p>During a 54-day period of Llama 3 405B pre-training, Meta counted 466 job interruptions: 47 planned (firmware upgrades, configuration or dataset changes) and 419 unexpected, one every 3.1 hours. Training is synchronous, so one failed GPU can force a restart of the whole 16,384-GPU job from its last checkpoint.</p>
<p>About 78% of the unexpected interruptions were attributed to confirmed or suspected hardware problems; faulty GPUs and their HBM3 memory lead the table. Six were silent data corruption: a GPU computing wrong values without reporting an error. Automation handled almost all of them; only three needed significant manual intervention, and the run kept more than 90% effective training time.</p>
<p>Scale shows up in other ways too. Throughput varied by 1 to 2% with the time of day, because mid-day heat changed the GPUs' dynamic voltage and frequency scaling. And when tens of thousands of GPUs pause together, waiting for a checkpoint or a collective, the data center's power draw swings by tens of megawatts.</p>`,
    fig: {
      type: "bars", caption: "Root causes of the 419 unexpected interruptions in 54 days (Meta's Table 5)", fmt: "int",
      items: [
        { label: "faulty GPU", value: 148, text: "148", hi: true }, { label: "GPU HBM3 memory", value: 72, text: "72" }, { label: "software bug", value: 54, text: "54" },
        { label: "network switch or cable", value: 35, text: "35" }, { label: "host maintenance", value: 32, text: "32" }, { label: "GPU SRAM memory", value: 19, text: "19" },
        { label: "GPU system processor", value: 17, text: "17" }, { label: "NIC", value: 7, text: "7" }, { label: "NCCL watchdog timeout", value: 7, text: "7" },
        { label: "silent data corruption", value: 6, text: "6" }, { label: "GPU thermal / sensor", value: 6, text: "6" }, { label: "seven other causes", value: 16, text: "16" },
      ],
      note: "The other 16: SSD 3, power supply 3, and 2 each for server chassis, IO expansion board, dependency, CPU and system memory. The paper's percentage column lists 30.1% for faulty GPUs; its count, 148 of 419, is 35.3%.",
    },
    terms: [{ term: "Silent data corruption", def: "A hardware fault that produces wrong results without raising an error, detectable only by checking results (for example, loss or gradient anomalies)." }],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.4, Table 5)"],
  });

  const waste = (T, p) => p.C / T + (T / 2 + p.R) / (p.M * 60);
  Deck.slide({
    id: "scale-interval",
    title: "The checkpoint interval balances write cost against lost work",
    body: String.raw`<p>If writing a checkpoint stalls training for C, failures arrive every M on average, and a restart takes R, then checkpointing every T wastes a fraction</p>
\[w(T) \approx \frac{C}{T} + \frac{T/2 + R}{M}\]
<p>of the time: the writes, plus for each failure half an interval of lost work on average and the restart. Setting the derivative to zero gives Young's interval \(T^{*} = \sqrt{2CM}\) (1974), at which the two T terms are equal.</p>
<p>At Llama 3's rate of one unexpected interruption every 3.1 hours, a 1-minute blocking write gives T* = 19 minutes and loses 10% of the time before counting restarts. Asynchronous checkpointing shrinks the blocking part to the copy into host memory: at C = 6 s, T* = 6.1 minutes and writes plus lost work cost 3.3%. The restart time R is then the larger term, which is why Meta also cut job start-up time. The plate's C and R are illustrative assumptions; Meta did not publish them.</p>`,
    fig: {
      type: "plot", caption: "Wasted fraction of time against the checkpoint interval; the dot is Young's optimum",
      x: [0, 2.78], y: [0, 0.5], xlabel: "checkpoint interval T (minutes)", ylabel: "time wasted", xticks: [0, 1, 2, Math.log10(300)], yticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5],
      xaxis: { fmt: (v) => String(Math.round(Math.pow(10, v))) }, yaxis: { fmt: (v) => `${Math.round(v * 100)}%` },
      params: [
        { name: "M", label: "time between failures M (h)", min: 0.5, max: 24, step: 0.1, value: 3.1, fmt: "f1" },
        { name: "C", label: "blocking write C (min)", min: 0.05, max: 10, step: 0.05, value: 1, fmt: "f2" },
        { name: "R", label: "restart R (min)", min: 0, max: 60, step: 1, value: 10, fmt: "f0" },
      ],
      fns: [{ f: (x, p) => Math.min(waste(Math.pow(10, x), p), 0.55), cls: "s1" }],
      extra: (p) => { const T = Math.sqrt(2 * p.C * p.M * 60), w = waste(T, p); return { dots: w <= 0.5 && T <= 600 ? [{ x: Math.log10(T), y: w, r: 5, cls: "", tip: `T* = ${T.toFixed(1)} min, waste ${(w * 100).toFixed(1)}%` }] : [] }; },
      readouts: [
        { label: "best interval T*", f: (p) => Math.sqrt(2 * p.C * p.M * 60), fmt: (v) => `${v.toFixed(1)} min` },
        { label: "waste at T*", f: (p) => waste(Math.sqrt(2 * p.C * p.M * 60), p), fmt: (v) => (v >= 1 ? "≥ 100%" : `${(v * 100).toFixed(1)}%`) },
        { label: "effective training time", f: (p) => 1 - waste(Math.sqrt(2 * p.C * p.M * 60), p), fmt: (v) => (v <= 0 ? "0% (model breaks down)" : `${(v * 100).toFixed(1)}%`) },
      ],
    },
    check: {
      q: "Failures arrive every 3.1 hours (186 minutes). A run checkpoints every 30 minutes with a 1-minute blocking write and takes 10 minutes to restart. What fraction of time is wasted?",
      options: ["about 3%", "about 10%", "about 17%", "about 35%"], answer: 2,
      why: "C/T = 1/30 = 3.3% for writes, plus (30/2 + 10)/186 = 13.4% for lost work and restarts: 16.8%. Young's interval √(2 × 1 × 186) = 19.3 minutes would lower it to 15.7%; most of what remains is the restart time.",
    },
    src: ["Young, A First Order Approximation to the Optimum Checkpoint Interval, Communications of the ACM, 1974", "Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.3.4)"],
  });

  Deck.slide({
    id: "scale-spikes",
    title: "Loss spikes: prevention, then rollback",
    body: String.raw`<p>A <b>loss spike</b> is a sudden jump in the loss, sometimes followed by divergence. Warmup keeps early updates small: Llama 3 405B raised its learning rate linearly over 8,000 steps and started with a quarter of its final batch. Gradient clipping by global norm (Part 3) rescales any gradient whose norm passes a threshold, so one bad batch cannot flood Adam's moving averages; Adam itself limits a single step, since one huge gradient moves each weight by about lr · (1 − β₁)/√(1 − β₂) = 0.45 lr at β₁ = 0.9, β₂ = 0.95. PaLM added a <b>z-loss</b>, \(10^{-4}\log^2 Z\) with Z the softmax normalizer, to keep log Z near 0 and the output logits from drifting large.</p>
<p>When a spike happens anyway, the remedy is a rollback. PaLM's largest model spiked about 20 times despite gradient clipping. Each time the team restarted from a checkpoint about 100 steps before the spike and skipped about 200 to 500 data batches. Replaying the same batches from a different checkpoint did not spike, so the cause was the combination of the batches with that model state; the data alone did not cause it.</p>
<p>Llama 3 405B, with its smaller early batch, saw few loss spikes and needed no intervention for divergence.</p>`,
    fig: {
      type: "table", caption: "Defenses against loss spikes in two published runs",
      cols: ["Defense", "What it does", "Published setting"],
      rows: [
        ["learning-rate warmup", "small early updates", "Llama 3 405B: linear over 8,000 steps to 8 × 10⁻⁵"],
        ["batch-size ramp", "smaller, more careful early steps", "Llama 3 405B: 4M → 8M → 16M tokens"],
        ["gradient clipping", "caps the global gradient norm before Adam sees it", "used by PaLM"],
        ["z-loss", "keeps the softmax normalizer near 1", "PaLM: 10⁻⁴ · log² Z"],
        ["rollback and skip", "undo the spike, avoid the batches", "PaLM: about 100 steps back, skip 200–500 batches"],
      ],
    },
    terms: [{ term: "z-loss", def: "An auxiliary loss proportional to the squared log of the softmax normalizer, (log Z)², that keeps output logits from growing large; PaLM used a weight of 10⁻⁴." }],
    src: ["Chowdhery et al., PaLM: Scaling Language Modeling with Pathways, 2022 (Sections 5 and 5.1)", "Dubey et al., The Llama 3 Herd of Models, 2024 (Section 3.4.1)"],
  });

  Deck.slide({
    id: "scale-monitor",
    title: "What to watch while the job runs",
    body: String.raw`<p>A run is watched as time series, step by step:</p>
<ul><li><b>loss</b> on the training batches and on fixed held-out sets per data source;</li>
<li><b>gradient norm</b>, whose jumps often come before loss spikes;</li>
<li><b>throughput</b>, tokens per second and MFU; a slow step points at a <b>straggler</b>, one slow GPU or link that every synchronous collective waits for;</li>
<li><b>memory</b> per GPU, to catch out-of-memory failures before sequence lengths or batch sizes change;</li>
<li><b>communication health</b>: NCCL watchdog timeouts and hangs. Meta used PyTorch's NCCL flight recorder, which keeps each collective's metadata and stack traces in a ring buffer, to find hangs and stragglers.</li></ul>
<p>The expected values come from arithmetic. Llama 3.1 405B at 400 TFLOP/s per GPU on 16,384 GPUs with 16M-token steps: 6 × 405 × 10⁹ × 16,777,216 / (16,384 × 400 × 10¹²) = 6.2 s per step, 2.7M tokens per second, 233 billion tokens a day.</p>`,
    fig: {
      type: "calc", caption: "Expected step time and throughput from 6N per token",
      inputs: [
        { id: "N", label: "parameters", min: 1e9, max: 2e12, value: 405e9, log: true, step: 0.001, fmt: (x) => human(x) },
        { id: "lg", label: "GPUs", min: 8, max: 15, step: 1, value: 14, fmt: (x) => Math.pow(2, x).toLocaleString("en-US") },
        { id: "tf", label: "TFLOP/s per GPU", min: 100, max: 700, step: 10, value: 400, fmt: "int" },
        { id: "b", label: "tokens per step", options: [{ label: "4M", value: 4194304 }, { label: "8M", value: 8388608 }, { label: "16M", value: 16777216 }], value: 16777216 },
      ],
      outputs: [
        { label: "step time", f: (v) => 6 * v.N * v.b / (Math.pow(2, v.lg) * v.tf * 1e12), fmt: "time" },
        { label: "tokens per second", f: (v) => v.b / (6 * v.N * v.b / (Math.pow(2, v.lg) * v.tf * 1e12)), fmt: "sci" },
        { label: "tokens per day", f: (v) => 86400 * v.b / (6 * v.N * v.b / (Math.pow(2, v.lg) * v.tf * 1e12)), fmt: "sci" },
        { label: "MFU", f: (v) => v.tf * 1e12 / 989e12, fmt: "pct" },
      ],
    },
    terms: [{ term: "Straggler", def: "A GPU, host or link that runs slower than the rest; in synchronous training every collective, and so every step, waits for it." }],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Sections 3.3.4 and 3.4.1, Table 4)"],
  });

  Deck.slide({
    id: "scale-eval",
    title: "Evaluation during training, and before it",
    body: String.raw`<p>Held-out loss shows that training is working; it does not show what the model can do. Benchmarks can run on saved checkpoints while training continues.</p>
<p>Meta also evaluated the 405B model before training it. It trained scaling-law models from 40M to 16B parameters with budgets from 6 × 10¹⁸ to 10²² FLOPs, fit how their negative log-likelihood of the correct answers on a benchmark fell with compute, then mapped that likelihood to accuracy with a sigmoid fitted on those models and the Llama 2 family. Extrapolated over four orders of magnitude, the prediction for ARC Challenge only slightly underestimated the final model.</p>
<p>The same approach judged data. Annealing a half-trained Llama 3 8B over 40B tokens, with 30% of the mix taken from a candidate dataset, measured what that dataset was worth before it went into the 405B run.</p>`,
    check: {
      q: "Before training the 405B model, Meta's scaling-law runs related which quantity to training FLOPs, before mapping it to benchmark accuracy?",
      options: ["training loss on the pre-training data", "the negative log-likelihood of the correct answers on the benchmark", "benchmark accuracy directly", "the number of training tokens"], answer: 1,
      why: "Meta fit the normalized negative log-likelihood of the correct answer against FLOPs on runs from 6 × 10¹⁸ to 10²² FLOPs, then mapped that likelihood to accuracy with a sigmoid fitted on those models and the Llama 2 family. Extrapolated four orders of magnitude, it slightly underestimated the 405B model on ARC Challenge.",
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Sections 3.1.3 and 3.2.1)"],
  });
})();
