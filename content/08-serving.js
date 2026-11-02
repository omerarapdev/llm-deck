/* Part 8 — Inference and serving: prefill and decode, the KV cache, batching and
 * scheduling, faster decoding, parallelism for inference, serving systems. */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip, slider, segmented, button } = window.UI;
  const M = window.TinyLM;

  // ------------------------------------------------------------ reference hardware and models
  // NVIDIA H100 SXM datasheet: "80 GB" HBM3 (80 GiB), 3.35 TB/s; whitepaper: 989 TFLOP/s dense BF16, 1,979 dense FP8.
  // CUDA reports 79.32 GiB of the 80 GiB as the device total (PyTorch on an H100 80GB), and vLLM's budget is
  // gpu_memory_utilization (0.92 by default) times that reported total. Every "does it fit" below uses that budget.
  const GIB = 2 ** 30;
  const H100 = { mem: 80 * GIB, memCuda: 79.32 * GIB, bw: 3.35e12, bf16: 989e12, fp8: 1979e12, pcie: 64e9 };
  const UTIL = 0.92;
  const usable = (gpus) => gpus * H100.memCuda * UTIL;
  // Llama 3.1 config.json (hidden, FFN, layers, query heads, KV heads, head dim, vocabulary); untied output layer.
  const LLAMA = {
    "8B": { name: "Llama 3.1 8B", L: 32, d: 4096, ffn: 14336, H: 32, KV: 8, hd: 128, V: 128256 },
    "70B": { name: "Llama 3.1 70B", L: 80, d: 8192, ffn: 28672, H: 64, KV: 8, hd: 128, V: 128256 },
    "405B": { name: "Llama 3.1 405B", L: 126, d: 16384, ffn: 53248, H: 128, KV: 8, hd: 128, V: 128256 },
  };
  const layerParams = (m) => 2 * m.d * m.H * m.hd + 2 * m.d * m.KV * m.hd + 3 * m.d * m.ffn + 2 * m.d;
  const totalParams = (m) => 2 * m.V * m.d + m.L * layerParams(m) + m.d;
  // a decode step reads every weight except the input embedding table (a lookup of one row per token)
  const readParams = (m) => totalParams(m) - m.V * m.d;
  const kvPerToken = (m, bytes = 2) => 2 * m.L * m.KV * m.hd * bytes;
  // attention FLOPs for one query token over ctx cached positions: q·k scores and the weighted sum of values
  const attnFlops = (m, ctx) => 4 * m.L * m.H * m.hd * ctx;

  /* One forward pass on `gpus` H100s (tensor parallel), roofline model:
   * time = max(bytes / bandwidth, FLOPs / peak). `tokens` new tokens share one read of the weights;
   * kvTokens cached positions are read; attn is the attention FLOPs of the pass. */
  function passTime(m, { tokens, kvTokens = 0, attn = 0, gpus = 8, wBytes = 2, kvBytes = 2, peak = H100.bf16, eff = 1, wRead }) {
    const bw = gpus * H100.bw * eff, fl = gpus * peak * eff;
    const bytes = (wRead ?? readParams(m) * wBytes) + kvTokens * kvPerToken(m, kvBytes);
    const flops = 2 * readParams(m) * tokens + attn;
    const tm = bytes / bw, tc = flops / fl;
    return { t: Math.max(tm, tc), tm, tc, bytes, flops };
  }
  // B requests decoding one token each, every request holding ctx cached tokens
  function decodeStep(m, B, ctx, o = {}) {
    const r = passTime(m, Object.assign({ tokens: B, kvTokens: B * ctx, attn: B * attnFlops(m, ctx) }, o));
    const gpus = o.gpus || 8, wBytes = o.wBytes || 2, kvBytes = o.kvBytes || 2;
    r.mem = (o.wTotal ?? totalParams(m) * wBytes) + B * ctx * kvPerToken(m, kvBytes);
    r.cap = usable(gpus);
    r.fits = r.mem <= r.cap;
    r.kvRead = B * ctx * kvPerToken(m, kvBytes);
    r.wRead = o.wRead ?? readParams(m) * wBytes;
    return r;
  }
  const S = { H100, LLAMA, totalParams, readParams, kvPerToken, attnFlops, passTime, decodeStep };

  // memory plan of one replica: v.m model, v.u utilization (8B on 1 GPU, 70B on 8)
  function budget(v) {
    const m = LLAMA[v.m], gpus = v.m === "70B" ? 8 : 1, total = gpus * H100.memCuda;
    const use = v.u * total, w = totalParams(m) * 2, kv = Math.max(0, use - w);
    return { total, use, w, kv, rest: total - use };
  }
  // bring back a cache of 2^lc tokens: copy over PCIe (64 GB/s per GPU) or recompute with a prefill at peak
  function reload(v) {
    const m = LLAMA[v.m], gpus = v.m === "70B" ? 8 : 1, T = pow2(v.lc);
    const copy = (kvPerToken(m) * T) / gpus / H100.pcie;
    const prefill = passTime(m, { tokens: T, attn: (attnFlops(m, 1) * T * (T + 1)) / 2, gpus }).t;
    return { copy, prefill };
  }


  const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, l));
  const pfmt = (p) => fmt.p(p);
  const commas = (x) => Math.round(x).toLocaleString("en-US");
  const ms = (sec, d = 2) => `${(sec * 1e3).toFixed(d)} ms`;
  const pow2 = (v) => Math.pow(2, v);

  // ------------------------------------------------------------ fig: prefill then decode, pass by pass, on our model
  Deck.fig("serve-passes", (el, sp, api) => {
    const P = 3, ids = M.runningIds, toks = M.running, N = 6;
    const tr = M.forward(ids.slice(0, M.config.T));
    let k = 0;
    const row = h("div", { class: "serve-pass-row" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const tiles = h("div", { class: "f-tiles" });
    const next = button("Next pass ▸", () => { if (k < N - 1) { k++; draw(); } }, { kind: "primary" });
    const back = button("Back to the prefill", () => { k = 0; draw(); });
    el.append(row, expl, tiles, h("div", { class: "f-ctl" }, back, next));
    const scoresOf = (i) => (i === 0 ? (P * (P + 1)) / 2 : P + i);
    function draw() {
      clear(row);
      let scores = 0;
      for (let i = 0; i < N; i++) {
        const inp = i === 0 ? toks.slice(0, P) : [toks[P + i - 1]];
        const p = tr.probs[P + i - 1][ids[P + i]];
        if (i <= k) scores += scoresOf(i);
        row.appendChild(h("div", {
          class: "serve-pass" + (i === 0 ? " pre" : "") + (i === k ? " cur" : "") + (i > k ? " later" : ""),
          "data-tip": `pass ${i + 1}: ${inp.length} token${inp.length > 1 ? "s" : ""} in; attends over ${P + i} positions; ${scoresOf(i)} attention scores per head per layer; p(${toks[P + i].trim()}) = ${p.toFixed(4)}`,
        },
        h("div", { class: "serve-pass-k" }, i === 0 ? "prefill" : `decode ${i}`),
        h("div", { class: "serve-pass-in" }, inp.map((t) => chip(t))),
        h("div", { class: "serve-pass-ar", "aria-hidden": "true" }, "↓"),
        h("div", { class: "serve-pass-out" }, chip(toks[P + i], { cls: "is-focus" }), h("span", { class: "serve-pass-p" }, `p ${pfmt(p)}`))));
      }
      const out = toks[P + k].trim(), p = tr.probs[P + k - 1][ids[P + k]];
      if (k === 0) expl.innerHTML = `<b>Prefill.</b> The 3 prompt tokens go through both layers together. Every position computes and stores its keys and values; only the last position's prediction is used: <code>${out}</code>, p = ${pfmt(p)}.`;
      else if (k < N - 1) expl.innerHTML = `<b>Decode ${k}.</b> Only <code>${toks[P + k - 1].trim()}</code> goes in. Its query attends to ${P + k} stored keys, and the pass reads every weight again to produce one token: <code>${out}</code>, p = ${pfmt(p)}.`;
      else expl.innerHTML = `<b>Done.</b> 6 passes produced 6 tokens. Attention scores per head and layer: 6 in the prefill, then 4 + 5 + 6 + 7 + 8, total 36, the same 36 as one pass over all 8 input tokens.`;
      clear(tiles);
      tiles.append(tile(String(k + 1), "forward passes"), tile(String(P + k), "tokens fed in"), tile(String(k + 1), "tokens out"), tile(String(scores), "attention scores per head per layer"));
      next.disabled = k >= N - 1;
      api.strip.set({ prefix: P + k + 1 < 9 ? P + k + 1 : null });
    }
    draw();
    return { enter: draw };
  });

  // ------------------------------------------------------------ fig: batching decode on Llama 3.1 70B, 8 x H100
  Deck.fig("serve-batch", (el) => {
    const m = LLAMA["70B"];
    let lb = 6, lc = 11;
    const ctl = h("div", { class: "f-ctl sliders" });
    const charts = h("div", { class: "serve-two" });
    const c1 = h("div"), c2 = h("div");
    charts.append(h("div", null, h("p", { class: "serve-sub" }, "tokens per second, each request"), c1), h("div", null, h("p", { class: "serve-sub" }, "tokens per second, whole server"), c2));
    const tiles = h("div", { class: "f-tiles" });
    const note = h("p", { class: "f-note", "aria-live": "polite" });
    el.append(ctl, charts, tiles, note);
    const xs = [];
    for (let e = 0; e <= 10.0001; e += 0.125) xs.push(pow2(e));
    const xax = { type: "log", domain: [1, 1024], ticks: [1, 4, 16, 64, 256, 1024], fmt: String, label: "requests in the batch" };
    const ch1 = window.UI.chart(c1, { width: 300, height: 200, margin: { l: 44, r: 10, t: 10, b: 38 }, x: xax, y: { domain: [0, 200], ticks: [0, 50, 100, 150, 200] }, aria: "Tokens per second for each request against batch size" });
    const ch2 = window.UI.chart(c2, { width: 300, height: 200, margin: { l: 54, r: 10, t: 10, b: 38 }, x: xax, y: { type: "log", domain: [100, 100000], ticks: [100, 1000, 10000, 100000], fmt: (v) => v.toLocaleString("en-US") }, aria: "Total tokens per second against batch size" });
    function draw() {
      const B = pow2(lb), ctx = pow2(lc);
      const fitPts = (f) => xs.filter((b) => decodeStep(m, b, ctx).fits).map((b) => [b, f(b)]);
      const oomPts = (f) => xs.filter((b) => !decodeStep(m, b, ctx).fits).map((b) => [b, f(b)]);
      const per = (b) => 1 / decodeStep(m, b, ctx).t, tot = (b) => b / decodeStep(m, b, ctx).t;
      const r = decodeStep(m, B, ctx);
      const bMax = Math.floor((r.cap - totalParams(m) * 2) / (ctx * kvPerToken(m)));
      const vl = bMax >= 1 && bMax <= 1024 ? [{ x: bMax, label: `memory full: ${commas(bMax)}`, cls: "red" }] : [];
      const dot = (y) => [{ x: B, y, r: 6, cls: r.fits ? "" : "hollow", tip: `batch ${B}: ${y.toFixed(0)} tokens/s` }];
      ch1.update({ series: [{ points: fitPts(per), cls: "s1" }, { points: oomPts(per), cls: "faint dashed" }], dots: dot(per(B)), vlines: vl });
      ch2.update({ series: [{ points: fitPts(tot), cls: "s1" }, { points: oomPts(tot), cls: "faint dashed" }], dots: dot(tot(B)), vlines: vl });
      const why = !r.fits ? "does not fit" : r.tc >= r.tm ? "compute" : r.kvRead > r.wRead ? "KV-cache reads" : "weight reads";
      clear(tiles);
      tiles.append(tile(ms(r.t), "per decode step"), tile(commas(1 / r.t), "tokens/s, each request"), tile(commas(B / r.t), "tokens/s, total"),
        tile(why, "limited by"), tile(`${(r.mem / 1e9).toFixed(1)} GB`, `of ${(r.cap / 1e9).toFixed(1)} GB usable`));
      note.innerHTML = `Reads per step: ${(r.wRead / 1e9).toFixed(1)} GB of weights + ${B} × ${commas(ctx)} × 327,680 B = ${(r.kvRead / 1e9).toFixed(1)} GB of caches, at 26.8 TB/s: ${ms(r.tm)}. Math: ${B} × (139.0 GFLOP + attention) at 7,912 TFLOP/s: ${ms(r.tc)}.` +
        (r.fits ? "" : ` <b>With the weights that is ${(r.mem / 1e9).toFixed(0)} GB: more than the ${(r.cap / 1e9).toFixed(1)} GB vLLM would use.</b>`);
    }
    ctl.append(
      slider({ label: "Requests in the batch", min: 0, max: 10, step: 1, value: lb, format: (v) => String(pow2(v)), onInput: (v) => { lb = v; draw(); } }),
      slider({ label: "Context per request (tokens)", min: 7, max: 17, step: 1, value: lc, format: (v) => commas(pow2(v)), onInput: (v) => { lc = v; draw(); } }));
    draw();
  });


  // ------------------------------------------------------------ fig: prefix caching on a stream of our queries
  Deck.fig("serve-prefix", (el) => {
    const V = M.vocab, NMAX = 16, rand = M.mulberry32(1);
    const reqs = Array.from({ length: NMAX }, () => M.encode(M.sampleQuery(rand)).map(M.idOf));
    const seen = new Set(), matched = [];
    reqs.forEach((t) => {
      let m = 0;
      while (m < t.length && seen.has(t.slice(0, m + 1).join(","))) m++;
      matched.push(m);
      for (let i = 1; i <= t.length; i++) seen.add(t.slice(0, i).join(","));
    });
    let n = 12;
    const list = h("div", { class: "serve-pfx-list" });
    const tiles = h("div", { class: "f-tiles" });
    const legend = h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw serve-sw-reused" }), "found in the cache"), h("span", { class: "key" }, h("span", { class: "sw serve-sw-new" }), "computed by the prefill"));
    const ctl = h("div", { class: "f-ctl sliders" }, slider({ label: "Requests arrived", min: 1, max: NMAX, step: 1, value: n, format: String, onInput: (v) => { n = v; draw(); } }));
    el.append(ctl, legend, h("div", { class: "scroll-x" }, list), tiles);
    function draw() {
      clear(list);
      let comp = 0;
      const stored = new Set();
      for (let i = 0; i < n; i++) {
        const t = reqs[i], reused = Math.min(matched[i], t.length - 1);
        comp += t.length - reused;
        for (let j = 1; j <= t.length; j++) stored.add(t.slice(0, j).join(","));
        list.appendChild(h("div", { class: "serve-pfx-row", "data-tip": `request ${i + 1}: ${reused} of 9 tokens found in the cache, ${t.length - reused} computed` },
          h("span", { class: "serve-pfx-n" }, String(i + 1)),
          h("span", { class: "serve-pfx-chips" }, t.map((id, j) => chip(V[id], { cls: j < reused ? "serve-reused" : "" })))));
      }
      clear(tiles);
      tiles.append(tile(String(n * 9), "prompt tokens"), tile(String(comp), "computed with the cache"), tile(`${Math.round((100 * (n * 9 - comp)) / (n * 9))}%`, "prefill saved"), tile(String(stored.size), "token entries stored"));
    }
    draw();
  });

  // ------------------------------------------------------------ fig: where an H100's memory goes
  function memoryBar(el, parts, total) {
    clear(el);
    const bar = h("div", { class: "serve-membar", role: "img", "aria-label": parts.map((p) => `${p.label} ${p.v.toFixed(1)} GB`).join(", ") });
    parts.forEach((p) => bar.appendChild(h("span", { class: `serve-membar-seg ${p.cls}`, style: { flexGrow: String(Math.max(p.v, 0)) }, "data-tip": `${p.label}: ${p.v.toFixed(1)} GB` }, p.v / total > 0.12 ? p.label : "")));
    el.append(bar, h("div", { class: "legend" }, parts.map((p) => h("span", { class: "key" }, h("span", { class: `sw ${p.cls}` }), `${p.label} ${p.v.toFixed(1)} GB`))));
  }


  // ------------------------------------------------------------ fig: chunked prefill against running decodes
  const CH = { B: 64, CTX: 2048 };
  // steps that process a P-token prompt with `budget` tokens per step, the B running decodes scheduled first
  function chunkPlan(P, budget) {
    const m = LLAMA["70B"], c = Math.max(1, budget - CH.B), out = [];
    for (let o = 0; o < P;) {
      const ci = Math.min(c, P - o);
      const r = passTime(m, { tokens: CH.B + ci, kvTokens: CH.B * CH.CTX + o, attn: CH.B * attnFlops(m, CH.CTX) + attnFlops(m, 1) * (ci * o + (ci * (ci + 1)) / 2) });
      out.push({ o, c: ci, t: r.t, tm: r.tm, tc: r.tc, flops: r.flops });
      o += ci;
    }
    return out;
  }
  Deck.fig("serve-chunk", (el) => {
    let lp = 13, lb = 9;
    const host = h("div", { class: "scroll-x" });
    const tiles = h("div", { class: "f-tiles" });
    const note = h("p", { class: "f-note", "aria-live": "polite" });
    const ctl = h("div", { class: "f-ctl sliders" },
      slider({ label: "New prompt (tokens)", min: 10, max: 15, step: 1, value: lp, format: (v) => commas(pow2(v)), onInput: (v) => { lp = v; draw(); } }),
      slider({ label: "Token budget per step", min: 7, max: 16, step: 1, value: lb, format: (v) => commas(pow2(v)), onInput: (v) => { lb = v; draw(); } }));
    el.append(ctl, host, tiles, note);
    const base = decodeStep(LLAMA["70B"], CH.B, CH.CTX).t;
    function draw() {
      const P = pow2(lp), budget = pow2(lb), plan = chunkPlan(P, budget);
      const total = plan.reduce((a, x) => a + x.t, 0), worst = Math.max(...plan.map((x) => x.t));
      const Wd = 600, Hd = 112, lab = 92, X = (t) => lab + (t / total) * (Wd - lab - 12);
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "serve-chunk-svg", role: "img", "aria-label": `${plan.length} steps; first token after ${ms(total, 1)}` });
      svg.appendChild(s("text", { x: lab - 8, y: 30, class: "serve-tl-r", "text-anchor": "end" }, "new prompt"));
      svg.appendChild(s("text", { x: lab - 8, y: 64, class: "serve-tl-r", "text-anchor": "end" }, `${CH.B} decodes`));
      let t = 0;
      plan.forEach((x, i) => {
        const x0 = X(t), x1 = X(t + x.t);
        svg.appendChild(s("rect", { x: x0 + 0.5, y: 14, width: Math.max(1, x1 - x0 - 1), height: 24, class: "serve-chunk-box", "data-tip": `step ${i + 1}: prompt tokens ${commas(x.o + 1)}–${commas(x.o + x.c)} with ${CH.B} decodes, ${ms(x.t)}` }));
        if (x1 - x0 > 34) svg.appendChild(s("text", { x: (x0 + x1) / 2, y: 30, class: "serve-chunk-l", "text-anchor": "middle" }, commas(x.c)));
        svg.appendChild(s("line", { x1: x1, x2: x1, y1: 52, y2: 74, class: "serve-chunk-tick", "data-tip": `${CH.B} tokens out at ${ms(t + x.t, 1)}` }));
        t += x.t;
      });
      svg.appendChild(s("line", { x1: lab, x2: Wd - 12, y1: 86, y2: 86, class: "serve-tl-axis" }));
      svg.appendChild(s("text", { x: lab, y: 104, class: "serve-tl-r" }, "0 ms"));
      svg.appendChild(s("text", { x: Wd - 12, y: 104, class: "serve-tl-r", "text-anchor": "end" }, `${ms(total, 1)}: first token of the new prompt`));
      clear(host); host.appendChild(svg);
      clear(tiles);
      tiles.append(tile(String(plan.length), plan.length > 1 ? "steps (chunks)" : "step, unchunked"), tile(ms(worst, 1), "longest step: decodes wait"), tile(ms(base), "decode step alone"), tile(ms(total, 1), "new prompt's TTFT"));
      note.innerHTML = `Each step: ${CH.B} running requests at ${commas(CH.CTX)} tokens of context decode one token, plus up to ${commas(Math.max(1, budget - CH.B))} prompt tokens. Llama 3.1 70B, 8 H100s, time = max(bytes ÷ 26.8 TB/s, FLOPs ÷ 7,912 TFLOP/s); each chunk also reads the cache of the chunks before it.`;
    }
    draw();
  });

  // ------------------------------------------------------------ fig: preemption by recomputation, block by block
  const PRE = { cap: 11, bs: 4, reqs: [{ name: "A", prompt: 6, out: 9 }, { name: "B", prompt: 5, out: 14 }, { name: "C", prompt: 7, out: 6 }, { name: "D", prompt: 4, out: 12 }] };
  // first-come-first-served; running requests grow first, preempting the latest-arrived running request when no block is free;
  // then waiting requests are admitted in queue order. A (re)admission is one prefill step that also yields a token.
  function preemptSim(cap) {
    const st = PRE.reqs.map((r, i) => Object.assign({}, r, { i, gen: 0, state: "waiting", blocks: 0, log: [] }));
    const waiting = st.slice(), running = [], used = [];
    let free = cap, t = 0;
    const blocksFor = (n) => Math.ceil(n / PRE.bs);
    for (; t < 80 && st.some((r) => r.state !== "done"); t++) {
      for (const r of running.slice().sort((a, b) => a.i - b.i)) {
        if (r.state !== "running") continue;
        const add = blocksFor(r.prompt + r.gen + 1) - r.blocks;
        while (add > free && r.state === "running") {
          const v = running.slice().sort((a, b) => b.i - a.i)[0];
          v.state = "waiting"; free += v.blocks; v.blocks = 0;
          running.splice(running.indexOf(v), 1); waiting.unshift(v); v.log.push({ t, kind: "out" });
        }
        if (r.state !== "running") continue;
        free -= add; r.blocks += add; r.gen++; r.log.push({ t, kind: "decode" });
        if (r.gen === r.out) { r.state = "done"; free += r.blocks; r.blocks = 0; running.splice(running.indexOf(r), 1); }
      }
      while (waiting.length) {
        const r = waiting[0], b = blocksFor(r.prompt + r.gen + 1);
        if (b > free) break;
        waiting.shift(); free -= b; r.blocks = b; r.state = "running"; running.push(r);
        r.log.push({ t, kind: r.gen ? "recompute" : "prefill", n: r.prompt + r.gen });
        r.gen++;
        if (r.gen === r.out) { r.state = "done"; free += r.blocks; r.blocks = 0; running.splice(running.indexOf(r), 1); }
      }
      used.push(cap - free);
    }
    return { st, steps: t, used };
  }
  Deck.fig("serve-preempt", (el) => {
    let cap = PRE.cap;
    const host = h("div", { class: "scroll-x" }), tiles = h("div", { class: "f-tiles" });
    const legend = h("div", { class: "legend" }, ["prefill", "decode", "recompute", "preempted"].map((k) => h("span", { class: "key" }, h("span", { class: `sw serve-pe-${k}` }), k)));
    el.append(h("div", { class: "f-ctl sliders" }, slider({ label: "KV blocks on the GPU (4 tokens each)", min: 9, max: 16, step: 1, value: cap, format: String, onInput: (v) => { cap = v; draw(); } })), host, legend, tiles);
    function draw() {
      const r = preemptSim(cap), N = Math.max(r.steps, 1);
      const Wd = 600, lab = 70, rh = 26, Hd = PRE.reqs.length * rh + 34, cw = (Wd - lab - 8) / N;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "serve-pe-svg", role: "img", "aria-label": `schedule of 4 requests over ${N} steps` });
      r.st.forEach((q, row) => {
        svg.appendChild(s("text", { x: lab - 8, y: row * rh + 17, class: "serve-tl-r", "text-anchor": "end" }, `${q.name} (${q.prompt}+${q.out})`));
        let outAt = null;
        q.log.forEach((e) => {
          if (e.kind === "out") { outAt = e.t; return; }
          if (outAt !== null) {
            svg.appendChild(s("rect", { x: lab + outAt * cw + 1, y: row * rh + 4, width: Math.max(1, (e.t - outAt) * cw - 2), height: rh - 8, class: "serve-pe-preempted", "data-tip": `${q.name} preempted at step ${outAt + 1}: blocks freed, waits` }));
            outAt = null;
          }
          svg.appendChild(s("rect", { x: lab + e.t * cw + 1, y: row * rh + 4, width: Math.max(1, cw - 2), height: rh - 8, class: `serve-pe-${e.kind}`, "data-tip": `step ${e.t + 1}: ${q.name} ${e.kind === "decode" ? "decodes one token" : `${e.kind}s ${e.n} tokens`}` }));
        });
      });
      svg.appendChild(s("text", { x: lab - 8, y: Hd - 6, class: "serve-tl-r", "text-anchor": "end" }, "step"));
      for (let t = 0; t < N; t += 5) svg.appendChild(s("text", { x: lab + t * cw + cw / 2, y: Hd - 6, class: "serve-tl-r", "text-anchor": "middle" }, String(t + 1)));
      clear(host); host.appendChild(svg);
      const pre = r.st.reduce((a, q) => a + q.log.filter((e) => e.kind === "out").length, 0);
      const redo = r.st.reduce((a, q) => a + q.log.filter((e) => e.kind === "recompute").reduce((x, e) => x + e.n, 0), 0);
      clear(tiles);
      tiles.append(tile(String(N), "steps to finish all"), tile(String(pre), "preemptions"), tile(String(redo), "tokens computed again"), tile(`${cap * PRE.bs}`, "token slots"));
    }
    draw();
  });


  // ------------------------------------------------------------ GPTQ vs round-to-nearest on our model
  // Per-output-channel symmetric b-bit grid. GPTQ (Frantar et al., Algorithm 1 with block size 1): for each input
  // dimension i in order, round row i of W (all outputs), then subtract err * U[i][j] from rows j > i, where U is the
  // upper Cholesky factor of H^-1, H = 2 X^T X (+1% damping), X the matrix's inputs over the grammar (weighted by p).
  const GQ = { mats: ["l0.wq", "l0.wk", "l0.wv", "l0.wo", "l0.w1", "l0.w2", "l1.wq", "l1.wk", "l1.wv", "l1.wo", "l1.w1", "l1.w2"] };
  function gqClone(Wt) { const o = {}; for (const [k, t] of Object.entries(Wt)) o[k] = { shape: t.shape, data: Float64Array.from(t.data) }; return o; }
  function gqScales(t, qmax) { const [r, c] = t.shape, sc = []; for (let j = 0; j < c; j++) { let m = 0; for (let i = 0; i < r; i++) m = Math.max(m, Math.abs(t.data[i * c + j])); sc.push(m / qmax || 1); } return sc; }
  const gqRound = (x, sc, qmax) => Math.max(-qmax, Math.min(qmax, Math.round(x / sc))) * sc;
  function gqRtn(t, bits) { const qmax = 2 ** (bits - 1) - 1, [r, c] = t.shape, sc = gqScales(t, qmax), d = new Float64Array(t.data.length); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) d[i * c + j] = gqRound(t.data[i * c + j], sc[j], qmax); return { shape: t.shape, data: d }; }
  function gqInputs(Wt, name, Q) {
    const X = [], wts = [], m = name.match(/^l(\d)\.(\w+)$/);
    for (const q of Q) {
      const L = M.forward(q.ids, M.finalStep, { weights: Wt }).layers[+m[1]];
      for (const row of { wq: L.h1, wk: L.h1, wv: L.h1, wo: L.concat, w1: L.h2, w2: L.act }[m[2]]) { X.push(row); wts.push(q.p); }
    }
    return { X, wts };
  }
  function gqInvCholUpper(H) {
    const n = H.length, L = H.map(() => new Array(n).fill(0)), Li = H.map(() => new Array(n).fill(0)), U = H.map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let a = H[i][j]; for (let k = 0; k < j; k++) a -= L[i][k] * L[j][k]; L[i][j] = i === j ? Math.sqrt(a) : a / L[j][j]; }
    for (let i = 0; i < n; i++) { Li[i][i] = 1 / L[i][i]; for (let j = 0; j < i; j++) { let a = 0; for (let k = j; k < i; k++) a -= L[i][k] * Li[k][j]; Li[i][j] = a / L[i][i]; } }
    const Hi = H.map((_, i) => H.map((__, j) => { let a = 0; for (let k = 0; k < n; k++) a += Li[k][i] * Li[k][j]; return a; }));
    for (let i = 0; i < n; i++) {
      let a = Hi[i][i]; for (let k = 0; k < i; k++) a -= U[k][i] * U[k][i]; U[i][i] = Math.sqrt(a);
      for (let j = i + 1; j < n; j++) { let b = Hi[i][j]; for (let k = 0; k < i; k++) b -= U[k][i] * U[k][j]; U[i][j] = b / U[i][i]; }
    }
    return U;
  }
  function gqGptq(t, X, wts, bits) {
    const qmax = 2 ** (bits - 1) - 1, [nin, nout] = t.shape, sc = gqScales(t, qmax);
    const H = Array.from({ length: nin }, () => new Array(nin).fill(0));
    X.forEach((x, k) => { const a = 2 * wts[k]; for (let i = 0; i < nin; i++) { if (!x[i]) continue; for (let j = 0; j < nin; j++) H[i][j] += a * x[i] * x[j]; } });
    let md = 0; for (let i = 0; i < nin; i++) md += H[i][i]; md /= nin;
    for (let i = 0; i < nin; i++) { if (H[i][i] === 0) H[i][i] = 1; H[i][i] += 0.01 * md; }
    const U = gqInvCholUpper(H), W = Array.from({ length: nin }, (_, i) => Array.from(t.data.subarray(i * nout, (i + 1) * nout)));
    for (let i = 0; i < nin; i++) {
      const err = new Array(nout);
      for (let o = 0; o < nout; o++) { const qv = gqRound(W[i][o], sc[o], qmax); err[o] = (W[i][o] - qv) / U[i][i]; W[i][o] = qv; }
      for (let j = i + 1; j < nin; j++) { const u = U[i][j]; for (let o = 0; o < nout; o++) W[j][o] -= err[o] * u; }
    }
    return { shape: t.shape, data: Float64Array.from(W.flat()) };
  }
  const gqCache = new Map();
  function gqRun(bits) {
    if (gqCache.has(bits)) return gqCache.get(bits);
    // calibration: 32 queries sampled with mulberry32(5), equal weights; evaluation: all 270 queries
    const cal = M.mulberry32(5), W0 = M.weights();
    const Q = Array.from({ length: 32 }, () => ({ ids: M.encode(M.sampleQuery(cal)).map(M.idOf).slice(0, M.config.T), p: 1 / 32 }));
    const R = gqClone(W0), G = gqClone(W0);
    for (const n of GQ.mats) R[n] = gqRtn(W0[n], bits);
    for (const n of GQ.mats) { const { X, wts } = gqInputs(G, n, Q); G[n] = gqGptq(G[n], X, wts, bits); }
    const ev = (Wt) => {
      const tr = M.forward(M.runningIds.slice(0, M.config.T), M.finalStep, Wt ? { weights: Wt } : {});
      const pr = tr.probs[2], top = pr.indexOf(Math.max(...pr));
      return { loss: M.exactLoss(M.finalStep, Wt ? { weights: Wt } : {}).mean, pu: pr[M.idOf(" users")], top: M.vocab[top], ptop: pr[top] };
    };
    const out = { fp: ev(null), rtn: ev(R), gptq: ev(G) };
    gqCache.set(bits, out);
    return out;
  }
  S.gqRun = gqRun;
  Deck.fig("serve-gptq", (el) => {
    let bits = 4;
    const host = h("div");
    el.append(h("div", { class: "f-ctl" }, segmented({ label: "Bits per weight", value: bits, options: [{ value: 4, label: "4" }, { value: 3, label: "3" }], onChange: (v) => { bits = v; draw(); } })), host);
    function draw() {
      const r = gqRun(bits);
      const row = (name, x, hi) => h("tr", { class: hi ? "hi" : "" }, h("td", null, name), h("td", { class: "r" }, x.loss.toFixed(4)), h("td", { class: "r" }, pfmt(x.pu)), h("td", null, chip(x.top), ` ${pfmt(x.ptop)}`));
      clear(host);
      host.appendChild(h("div", { class: "scroll-x" }, h("table", { class: "f-table" },
        h("thead", null, h("tr", null, h("th", null, "weights"), h("th", { class: "r" }, "loss, 270 queries"), h("th", { class: "r" }, "p(users) after FROM"), h("th", null, "top token after FROM"))),
        h("tbody", null, row("FP32", r.fp), row(`${bits}-bit, round to nearest`, r.rtn), row(`${bits}-bit, GPTQ`, r.gptq, true)))));
      host.appendChild(h("p", { class: "f-note" }, `Loss floor of the grammar: 0.6937 nats per token. Both ${bits}-bit versions use the same grid, one scale per output channel; only the choice of rounding differs.`));
    }
    draw();
  });

  // ------------------------------------------------------------ fig: split-KV attention with a log-sum-exp merge (FlashDecoding)
  Deck.fig("serve-flash", (el) => {
    const tr = M.forward(M.runningIds.slice(0, M.config.T)), pos = 7, hd = M.config.d / M.config.H;
    let head = "0-1", parts = 2;
    const host = h("div");
    el.append(h("div", { class: "f-ctl" },
      segmented({ label: "Head", value: head, options: [{ value: "0-0", label: "L1 H1" }, { value: "0-1", label: "L1 H2" }, { value: "1-0", label: "L2 H1" }, { value: "1-1", label: "L2 H2" }], onChange: (v) => { head = v; draw(); } }),
      segmented({ label: "Split the 8 keys into", value: parts, options: [1, 2, 4, 8].map((v) => ({ value: v, label: `${v} chunk${v > 1 ? "s" : ""}` })), onChange: (v) => { parts = v; draw(); } })), host);
    const vec = (a) => h("span", { class: "serve-vec" }, h("span", null, a.slice(0, 4).map((x) => fmt.minus(x.toFixed(3))).join("  ")), h("span", null, a.slice(4).map((x) => fmt.minus(x.toFixed(3))).join("  ")));
    function draw() {
      const [l, hh] = head.split("-").map(Number), sc = tr.layers[l].heads[hh].scores[pos].slice(0, pos + 1);
      const V = tr.layers[l].v.map((row) => row.slice(hh * hd, (hh + 1) * hd));
      const size = (pos + 1) / parts, chunks = [];
      for (let c = 0; c < parts; c++) {
        const js = Array.from({ length: size }, (_, k) => c * size + k), m = Math.max(...js.map((j) => sc[j]));
        const e = js.map((j) => Math.exp(sc[j] - m)), ell = e.reduce((a, b) => a + b, 0);
        const o = Array.from({ length: hd }, (_, d) => js.reduce((a, j, k) => a + e[k] * V[j][d], 0) / ell);
        chunks.push({ js, m, ell, o });
      }
      const mAll = Math.max(...chunks.map((c) => c.m)), w = chunks.map((c) => Math.exp(c.m - mAll) * c.ell), Lsum = w.reduce((a, b) => a + b, 0);
      const merged = Array.from({ length: hd }, (_, d) => chunks.reduce((a, c, k) => a + w[k] * c.o[d], 0) / Lsum);
      const att = tr.layers[l].heads[hh].att[pos], full = Array.from({ length: hd }, (_, d) => att.reduce((a, x, j) => a + x * (V[j] ? V[j][d] : 0), 0));
      const diff = Math.max(...merged.map((x, d) => Math.abs(x - full[d])));
      clear(host);
      const rows = chunks.map((c) => h("tr", null, h("td", null, `${c.js[0]}–${c.js[c.js.length - 1]}`), h("td", { class: "r" }, fmt.minus(c.m.toFixed(3))), h("td", { class: "r" }, c.ell.toFixed(3)), h("td", { class: "r" }, Math.exp(c.m - mAll).toFixed(3)), h("td", null, vec(c.o))));
      rows.push(h("tr", { class: "hi" }, h("td", null, "merged"), h("td", { class: "r" }, fmt.minus(mAll.toFixed(3))), h("td", { class: "r" }, Lsum.toFixed(3)), h("td", { class: "r" }, ""), h("td", null, vec(merged))));
      rows.push(h("tr", null, h("td", null, "all 8 at once"), h("td", { class: "r" }, ""), h("td", { class: "r" }, ""), h("td", { class: "r" }, ""), h("td", null, vec(full))));
      host.appendChild(h("div", { class: "scroll-x" }, h("table", { class: "f-table dense" },
        h("thead", null, h("tr", null, h("th", null, "positions"), h("th", { class: "r" }, "chunk max"), h("th", { class: "r" }, "sum"), h("th", { class: "r" }, "weight"), h("th", null, "output (8 numbers)"))),
        h("tbody", null, rows))));
      host.appendChild(h("p", { class: "f-note" }, "Chunk max m: the largest score in the chunk. Sum ℓ: Σ e^(score − m) over the chunk. Weight: e^(m − M), with M the largest chunk max. Merged = Σ weight × ℓ × chunk output ÷ Σ weight × ℓ."));
      host.appendChild(h("p", { class: "f-mm-expl" }, `Largest difference between the merged output and one pass over all keys: ${diff === 0 ? "0" : fmt.sci(diff, 1)}.`));
    }
    draw();
  });


  // ------------------------------------------------------------ tensor-parallel decode step for Llama 3.1 70B
  const NVLINK_DIR = 450e9; // H100 SXM NVLink: 900 GB/s total, 450 GB/s each way
  function tpStep(v) {
    const m = LLAMA["70B"], N = v.n, B = pow2(v.lb), ctx = 2048;
    const w = (totalParams(m) * 2) / N, kv = (B * ctx * kvPerToken(m)) / N;
    const r = decodeStep(m, B, ctx, { gpus: N });
    const S = B * m.d * 2, ar = N > 1 ? 2 * m.L * (v.a * 1e-6 + ((2 * (N - 1)) / N) * (S / NVLINK_DIR)) : 0;
    return { w, kv, fits: w + kv <= usable(1), t: r.t, ar, total: r.t + ar };
  }

  // ------------------------------------------------------------ a replica: weights, cache room, decode bound
  // Llama 3.1 405B FP8 as released: attention, the first and last layers' MLPs, embeddings and output layer in BF16
  function fp8Mix405() {
    const m = LLAMA["405B"], attn = 2 * m.d * m.H * m.hd + 2 * m.d * m.KV * m.hd, mlp = 3 * m.d * m.ffn, table = m.V * m.d;
    const total = m.L * attn * 2 + (m.L - 2) * mlp + 2 * mlp * 2 + 2 * table * 2 + (2 * m.L + 1) * m.d * 2;
    return { total, read: total - table * 2 };
  }
  function weightBytes(key, wb) {
    const m = LLAMA[key];
    if (wb === 1 && key === "405B") return fp8Mix405();
    return { total: totalParams(m) * wb, read: readParams(m) * wb };
  }
  function replica(v) {
    const m = LLAMA[v.m], N = v.n, B = pow2(v.lb), ctx = pow2(v.lc), wt = weightBytes(v.m, v.wb);
    const w = wt.total / N, room = usable(1) - w;
    const perTok = kvPerToken(m) / N, maxB = room > 0 ? Math.floor(room / (perTok * ctx)) : 0;
    const r = decodeStep(m, B, ctx, { gpus: N, wRead: wt.read, wTotal: wt.total });
    return { w, room, maxB, fits: room > 0 && B <= maxB, t: r.t };
  }

  // ------------------------------------------------------------ fig: load balancing by count or by work
  Deck.fig("serve-lb", (el) => {
    const R = 4, N = 32, rand = M.mulberry32(28);
    const lens = Array.from({ length: N }, () => 32 * pow2(Math.floor(rand() * 6)));
    function assign(policy) {
      const load = new Array(R).fill(0), who = [];
      lens.forEach((L, i) => { const r = policy === "rr" ? i % R : load.indexOf(Math.min(...load)); load[r] += L; who.push(r); });
      return { load, who };
    }
    let pol = "rr";
    const host = h("div"), tiles = h("div", { class: "f-tiles" });
    el.append(h("div", { class: "f-ctl" }, segmented({ label: "Router", value: pol, options: [{ value: "rr", label: "round robin" }, { value: "ll", label: "fewest tokens outstanding" }], onChange: (v) => { pol = v; draw(); } })), host, tiles);
    function draw() {
      const a = assign(pol), total = lens.reduce((x, y) => x + y, 0), mean = total / R, mx = Math.max(...a.load), cap = 6000;
      const Wd = 600, lab = 84, rh = 30, Hd = R * rh + 30, X = (t) => lab + (t / cap) * (Wd - lab - 10);
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "serve-lb-svg", role: "img", "aria-label": `tokens per replica: ${a.load.join(", ")}` });
      for (let r = 0; r < R; r++) {
        svg.appendChild(s("text", { x: lab - 8, y: r * rh + 19, class: "serve-tl-r", "text-anchor": "end" }, `replica ${r + 1}`));
        let x = 0;
        lens.forEach((L, i) => {
          if (a.who[i] !== r) return;
          svg.appendChild(s("rect", { x: X(x) + 0.5, y: r * rh + 5, width: Math.max(1, X(x + L) - X(x) - 1), height: rh - 10, class: `serve-lb-req s${(i % 5) + 1}`, "data-tip": `request ${i + 1}: ${L} tokens` }));
          x += L;
        });
        svg.appendChild(s("text", { x: X(a.load[r]) + 5, y: r * rh + 19, class: "serve-lb-v" }, commas(a.load[r])));
      }
      svg.appendChild(s("line", { x1: X(mean), x2: X(mean), y1: 0, y2: R * rh, class: "serve-lb-mean" }));
      svg.appendChild(s("text", { x: X(mean), y: R * rh + 16, class: "serve-tl-r", "text-anchor": "middle" }, `even share ${commas(mean)}`));
      clear(host); host.appendChild(h("div", { class: "scroll-x" }, svg));
      clear(tiles);
      tiles.append(tile(commas(total), `tokens in ${N} requests`), tile(commas(mx), "tokens on the busiest replica"), tile(`${((mx / mean - 1) * 100).toFixed(0)}%`, "above the even share"));
    }
    draw();
  });


  // ------------------------------------------------------------ grammar-constrained decoding with the real model
  // The grammar, written over the model's tokens: the canonical tokenizations of the 270 queries (all 9 tokens long).
  // A token is allowed after a prefix if some valid query continues the prefix with it.
  const GR = (function () {
    const V = M.vocab, Q = M.allQueries().map((q) => M.encode(q.text).map(M.idOf));
    const SEL = M.idOf("SELECT"), SEMI = M.idOf(";"), valid = new Set(Q.map((t) => t.join(",")));
    const amemo = new Map(), lmemo = new Map();
    function allowed(pre) {
      const k = pre.join(",");
      if (!amemo.has(k)) {
        const set = new Set();
        for (const t of Q) { let ok = pre.length < t.length; for (let i = 0; ok && i < pre.length; i++) ok = t[i] === pre[i]; if (ok) set.add(t[pre.length]); }
        amemo.set(k, set);
      }
      return amemo.get(k);
    }
    function logits(pre) { const k = pre.join(","); if (!lmemo.has(k)) lmemo.set(k, M.forward(pre).logits[pre.length - 1]); return lmemo.get(k); }
    // next-token distribution at temperature tau; with the mask, forbidden tokens get 0 and the rest is renormalized
    function dist(pre, tau, mask) {
      const raw = M.softmax(logits(pre), tau);
      if (!mask) return { p: raw, raw, kept: 1 };
      const A = allowed(pre);
      let z = 0;
      for (let i = 0; i < raw.length; i++) if (A.has(i)) z += raw[i];
      return { p: raw.map((v, i) => (A.has(i) ? v / z : 0)), raw, kept: z };
    }
    function sample(rand, tau, mask) {
      const out = [SEL];
      while (out.length < 9) { const x = M.sampleFrom(dist(out, tau, mask).p, rand); out.push(x); if (x === SEMI) break; }
      return out;
    }
    const isValid = (t) => valid.has(t.join(","));
    const runs = new Map();
    function run(tau, mask, n = 1000, seed = 1) {
      const key = `${tau}|${mask}|${n}|${seed}`;
      if (!runs.has(key)) {
        const rand = M.mulberry32(seed), bad = [];
        let v = 0;
        for (let i = 0; i < n; i++) { const t = sample(rand, tau, mask); if (isValid(t)) v++; else if (bad.length < 3) bad.push(t.map((x) => V[x]).join("")); }
        runs.set(key, { valid: v, n, bad });
      }
      return runs.get(key);
    }
    let full = null;
    // exact probability that an unmasked sample is a valid query: the sum over the 270 queries of their probability
    function exact(tau) {
      if (!full) full = Q.map((t) => M.forward(t.slice(0, M.config.T)).logits);
      let acc = 0;
      Q.forEach((t, qi) => { let pr = 1; for (let i = 1; i < t.length; i++) pr *= M.softmax(full[qi][i - 1], tau)[t[i]]; acc += pr; });
      return acc;
    }
    return { V, Q, SEL, SEMI, allowed, dist, sample, isValid, run, exact };
  })();

  Deck.fig("serve-grammar", (el, sp, api) => {
    let tau = 2, mask = true, seed = 1, rand, seq;
    const reset = () => { rand = M.mulberry32(seed); seq = [GR.SEL]; draw(); };
    const done = () => seq.length >= 9 || seq[seq.length - 1] === GR.SEMI;
    const step = () => { if (done()) return; seq.push(M.sampleFrom(GR.dist(seq, tau, mask).p, rand)); draw(); };
    const chips = h("div", { class: "serve-gr-seq" }), table = h("div"), expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const next = button("Sample next token ▸", step, { kind: "primary" });
    el.append(
      h("div", { class: "f-ctl sliders" },
        slider({ label: "Temperature", min: 0.25, max: 4, step: 0.25, value: tau, format: (v) => v.toFixed(2), onInput: (v) => { tau = v; reset(); } }),
        segmented({ label: "Grammar mask", value: "on", options: [{ value: "on", label: "on" }, { value: "off", label: "off" }], onChange: (v) => { mask = v === "on"; reset(); } })),
      chips, table, expl,
      h("div", { class: "f-ctl" }, next, button("Finish the query", () => { while (!done()) seq.push(M.sampleFrom(GR.dist(seq, tau, mask).p, rand)); draw(); }), button("New sample", () => { seed++; reset(); })));
    function draw() {
      clear(chips);
      seq.forEach((id) => chips.appendChild(chip(GR.V[id])));
      if (!done()) chips.appendChild(h("span", { class: "chip chip-q is-target" }, h("span", { class: "chip-t" }, "?")));
      clear(table);
      if (done()) {
        const ok = GR.isValid(seq);
        expl.innerHTML = `${ok ? "<b>Valid.</b> One of the 270 queries of the grammar." : "<b>Invalid.</b> Not a query of the grammar."} Sample ${seed} at temperature ${tau.toFixed(2)}, mask ${mask ? "on" : "off"}.`;
        expl.className = `f-mm-expl serve-gr-${ok ? "ok" : "bad"}`;
      } else {
        const d = GR.dist(seq, tau, true), A = GR.allowed(seq);
        const order = d.raw.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 7);
        A.forEach((i) => { if (!order.some((o) => o[1] === i)) order.push([d.raw[i], i]); });
        const rows = order.slice(0, 10).map(([v, i]) => h("tr", { class: A.has(i) ? "" : "serve-gr-no" },
          h("td", null, chip(GR.V[i])), h("td", { class: "r" }, v.toFixed(3)), h("td", null, A.has(i) ? "allowed" : "masked"), h("td", { class: "r" }, mask ? (A.has(i) ? d.p[i].toFixed(3) : "0") : v.toFixed(3))));
        table.appendChild(h("div", { class: "scroll-x" }, h("table", { class: "f-table dense" },
          h("thead", null, h("tr", null, h("th", null, "next token"), h("th", { class: "r" }, "model"), h("th", null, "grammar"), h("th", { class: "r" }, "sampled from"))), h("tbody", null, rows))));
        expl.className = "f-mm-expl";
        expl.innerHTML = `${A.size} of 113 tokens are allowed here. At temperature ${tau.toFixed(2)} the model gives the forbidden ones ${((1 - d.kept) * 100).toFixed(1)}% of its probability; ${mask ? "the mask sets them to 0 and divides the rest by " + d.kept.toFixed(3) + "." : "with the mask off they can be drawn."}`;
      }
      next.disabled = done();
      api.strip.set({ tokens: seq.map((x) => GR.V[x]).concat(done() ? [] : ["?"]), prefix: null });
    }
    reset();
    return { enter: draw };
  });

  Deck.fig("serve-grammar-rates", (el) => {
    let tau = 2;
    const host = h("div"), tiles = h("div", { class: "f-tiles" }), bad = h("div", { class: "serve-gr-bad" });
    el.append(h("div", { class: "f-ctl sliders" }, slider({ label: "Temperature", min: 0.25, max: 4, step: 0.25, value: tau, format: (v) => v.toFixed(2), onInput: (v) => { tau = v; draw(); } })), host, tiles, bad);
    const taus = [];
    for (let t = 0.25; t <= 4.0001; t += 0.125) taus.push(+t.toFixed(3));
    let curve = null;
    const ch = window.UI.chart(host, { width: 560, height: 230, margin: { l: 50, r: 14, t: 12, b: 38 }, aria: "Share of valid queries against temperature", x: { domain: [0.25, 4], ticks: [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], label: "temperature" }, y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], label: "valid" } });
    function draw() {
      if (!curve) curve = taus.map((t) => [t, GR.exact(t)]);
      const off = GR.run(tau, false), on = GR.run(tau, true), ex = GR.exact(tau);
      ch.update({
        series: [{ points: curve, cls: "s2", label: "" }, { points: [[0.25, 1], [4, 1]], cls: "s1" }],
        dots: [{ x: tau, y: ex, r: 6, cls: "s2", tip: `exact P(valid) without the mask: ${ex.toFixed(4)}` }, { x: tau, y: off.valid / off.n, r: 4, cls: "hollow", tip: `${off.valid} of 1,000 samples valid without the mask` }, { x: tau, y: 1, r: 6, cls: "s1", tip: "with the mask: 1,000 of 1,000" }],
        vlines: [{ x: tau, cls: "cursor" }],
      });
      clear(tiles);
      tiles.append(tile(ex.toFixed(4), "exact share valid, no mask"), tile(`${commas(off.valid)} / 1,000`, "sampled valid, no mask"), tile(`${commas(on.valid)} / 1,000`, "sampled valid, with mask"));
      clear(bad);
      if (off.bad.length) bad.append(h("p", { class: "f-note" }, "First invalid samples without the mask:"), ...off.bad.map((t) => h("code", { class: "serve-gr-badq" }, t)));
    }
    draw();
    el.appendChild(h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "no mask: exact share of valid samples"), h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "with the mask")));
  });


  // ------------------------------------------------------------ cost per million tokens and SLO capacity (Llama 3.1 70B, 8 H100s)
  function costCalc(v) {
    const m = LLAMA["70B"], B = pow2(v.lb), ctx = pow2(v.lc), r = decodeStep(m, B, ctx);
    const tps = (B / r.t) * v.eff, perHour = v.price * 8;
    const pfTps = (8 * H100.bf16 * v.eff) / (2 * readParams(m));
    return { fits: r.fits, tps, out: (perHour / 3600 / tps) * 1e6, inp: (perHour / 3600 / pfTps) * 1e6, pfTps };
  }
  // steady state with chunked prefill: each step decodes B requests and processes B·P/O prompt tokens,
  // so prompts enter as fast as answers finish. Largest B whose step meets the TPOT target and whose caches fit.
  function sloPlan(v) {
    const m = LLAMA["70B"], P = pow2(v.lp), O = pow2(v.lo), ctx = P + O / 2, kv = kvPerToken(m);
    const room = usable(8) - totalParams(m) * 2;
    const stepT = (B) => { const c = (B * P) / O; return passTime(m, { tokens: B + c, kvTokens: B * ctx, attn: B * attnFlops(m, ctx) + c * attnFlops(m, P / 2) }).t / v.eff; };
    let best = 0;
    for (let B = 1; B <= 8192; B++) { if (B * (P + O) * kv > room || stepT(B) * 1e3 > v.slo) break; best = B; }
    if (!best) return { B: 0 };
    const t = stepT(best), mu = best / (O * t), reps = Math.ceil(v.rate / mu);
    return { B: best, t, mu, reps, gpus: 8 * reps, ttft: Math.max(1, O / best) * t };
  }
  // the consolidating request: 2,048-token prompt, 256-token answer, joining 64 requests at 2,048 tokens, budget 512
  function requestTimeline() {
    const m = LLAMA["70B"], plan = chunkPlan(2048, 512);
    const ttft = plan.reduce((a, x) => a + x.t, 0), lastChunk = plan[plan.length - 1];
    const allCompute = plan.slice(0, -1).every((x) => x.tc > x.tm);
    const limits = allCompute ? (lastChunk.tc > lastChunk.tm ? "compute" : "compute; last chunk HBM reads") : "HBM reads";
    let dec = 0, first = 0, last = 0;
    for (let i = 1; i <= 255; i++) {
      const r = passTime(m, { tokens: 65, kvTokens: 64 * 2048 + 2048 + i, attn: 64 * attnFlops(m, 2048) + attnFlops(m, 2048 + i) });
      if (i === 1) first = r.t;
      last = r.t; dec += r.t;
    }
    return { chunks: plan.length, ttft, dec, first, last, total: ttft + dec, limits };
  }
  S.costCalc = costCalc; S.sloPlan = sloPlan; S.requestTimeline = requestTimeline;

  // ================================================================== PART 8
  Deck.part({ n: 8, title: "Inference and serving", short: "Serving" });

  // ------------------------------------------------------------------ 8.1
  Deck.chapter({ title: "Two phases" });

  Deck.slide({
    id: "serve-part",
    layout: "part",
    title: "Inference and serving",
    body: String.raw`<p>A trained model is used by sending it requests. This part follows one request through a production server: the prompt is processed in one parallel pass, then the answer is produced one token per forward pass, while hundreds of other requests share the same GPUs.</p>
<p>Serving is a different systems problem from training. Producing a token reads every weight, so memory bandwidth and the memory taken by each request's cached keys and values set the limits, and most of the engineering is scheduling: what to batch, what to keep in memory, what to compute again.</p>`,
    take: "Predict the latency and throughput of a model on given GPUs from bytes and FLOPs, size its KV cache, explain how an engine like vLLM batches and pages requests, and choose quantization, speculative decoding and parallelism for a deployment.",
  });

  Deck.slide({
    id: "serve-phases",
    title: "A request runs one prefill pass, then one decode pass per token",
    example: "tokens ids prefix:3",
    body: String.raw`<p>A user sends the prompt <code>SELECT name FROM</code>. The server first runs the model once over all 3 prompt tokens together. This pass is the <b>prefill</b>: it computes the keys and values of every prompt position, and the next-token distribution at the last position gives the first output token, <code>users</code> (p = 1.000).</p>
<p>Then the server <b>decodes</b>: each forward pass takes only the newest token, attends to the stored keys and values of every earlier token, and produces one more token. Our 6 output tokens take 6 passes: 1 prefill and 5 decodes.</p>
<p>The two phases do the same arithmetic per token. They differ in shape. The prefill sends 3 tokens through every weight matrix in one pass; a decode pass sends 1. A real prompt has hundreds or thousands of tokens, so its prefill is one large matrix multiplication, while decode stays at one token per request per pass however long the answer grows.</p>`,
    fig: { type: "serve-passes", caption: "Our model serving the prompt SELECT name FROM, pass by pass. Each card shows the tokens fed in and the token produced, with the probability the model gave it." },
    take: "Prefill is parallel over the prompt; decode is sequential over the output. An answer of n tokens costs 1 prefill pass and n − 1 decode passes.",
    terms: [
      { term: "Prefill", def: "The first forward pass of a request: all prompt tokens are processed together, their keys and values are stored, and the last position's prediction gives the first output token." },
      { term: "Decode", def: "Every later forward pass of a request: one new token goes in, attends to the stored keys and values, and one token comes out." },
    ],
  });

  Deck.slide({
    id: "serve-bound",
    title: "Prefill is compute-bound; decode is memory-bound",
    body: String.raw`<p>A forward pass has to bring every weight from HBM to the tensor cores at least once. What changes is how many tokens share that trip. A pass over T tokens does 2 FLOPs per weight per token and, in BF16, reads 2 bytes per weight: T FLOPs per byte of weights. That is its arithmetic intensity (Part 6).</p>
<p>An H100 SXM does 989 TFLOP/s of dense BF16 math and reads 3.35 TB/s from HBM, so a kernel needs 295 FLOPs per byte to keep the tensor cores busy: the ridge point of Part 6's roofline. A 2,048-token prefill has intensity 2,048 and is compute-bound. A decode step at batch 1 has intensity 1: it waits on memory and can use at most 1/295 = 0.34% of the peak.</p>
<p>For Llama 3.1 70B on 8 H100s, a pass reads 139.0 GB of weights, which takes at least 5.19 ms. A 2,048-token prefill needs 2,048 × 139.0 GFLOP = 284.7 TFLOP of matrix math: 36.0 ms at the peak of 8 GPUs, seven times the read. A decode token needs 139.0 GFLOP: 17.6 µs of math against 5.19 ms of reading.</p>`,
    fig: {
      type: "plot", caption: "Attainable math rate of one H100 for a pass that sends T tokens through the weights (weight bytes only). Drag T.",
      x: [0, 14], y: [0, 1100], xlabel: "tokens per pass (log scale)", ylabel: "TFLOP/s per GPU",
      xaxis: { ticks: [0, 2, 4, 6, 8, 10, 12, 14], fmt: (v) => commas(pow2(v)) }, yaxis: { ticks: [0, 200, 400, 600, 800, 1000] },
      params: [{ name: "lt", label: "tokens per pass T", min: 0, max: 14, step: 1, value: 0, fmt: (v) => commas(pow2(v)) }],
      fns: [{ f: (x) => Math.min(989, 3.35 * pow2(x)), cls: "s1", samples: 280 }],
      extra: (p) => ({
        dots: [
          { x: 0, y: 3.35, r: 4.5, cls: "ink", label: "decode, batch 1", dx: 8, dy: -8, tip: "1 token per pass: 3.35 TFLOP/s" },
          { x: 11, y: 989, r: 4.5, cls: "ink", label: "prefill, 2,048 tokens", anchor: "end", dx: -8, dy: 18, tip: "2,048 tokens per pass: 989 TFLOP/s" },
          { x: p.lt, y: Math.min(989, 3.35 * pow2(p.lt)), r: 6.5, cls: "", tip: `T = ${commas(pow2(p.lt))}` },
        ],
        vlines: [{ x: Math.log2(989 / 3.35), label: "ridge: 295" }],
      }),
      readouts: [
        { label: "FLOPs per byte of weights", f: (p) => pow2(p.lt), fmt: "int" },
        { label: "share of peak math used", f: (p) => Math.min(1, pow2(p.lt) / (989 / 3.35)), fmt: (v) => (v >= 0.995 ? "100%" : `${(v * 100).toFixed(v < 0.1 ? 2 : 1)}%`) },
        { label: "70B pass on 8 GPUs, at least", f: (p) => passTime(LLAMA["70B"], { tokens: pow2(p.lt) }).t, fmt: (v) => ms(v) },
        { label: "limited by", f: (p) => (pow2(p.lt) >= 989 / 3.35 ? "compute" : "memory"), fmt: (v) => v },
      ],
    },
    take: "Count the tokens that share one read of the weights. Below about 300 per pass, an H100 waits on HBM; above it, on the tensor cores.",
    src: ["NVIDIA H100 Tensor Core GPU datasheet and architecture whitepaper (H100 SXM)", "Llama 3.1 70B config.json (Meta, 2024)"],
  });

  Deck.slide({
    id: "serve-metrics",
    title: "Latency is two numbers: time to first token and time per token",
    body: String.raw`<p>A user sees the first token after the <b>time to first token</b> (TTFT): queueing, tokenization and the prefill. Each further token arrives one decode step later. That interval is the <b>time per output token</b> (TPOT), also reported as inter-token latency. An answer of n tokens is complete after \(\text{TTFT} + (n-1) \cdot \text{TPOT}\).</p>
<p>Take Llama 3.1 70B alone on 8 H100s, a 2,048-token prompt and a 500-token answer. The prefill needs 290.2 TFLOP including attention, so TTFT ≥ 36.7 ms. Each decode step reads 139.0 GB of weights and up to 0.83 GB of cache, so TPOT ≥ 5.21 ms, and the answer takes at least 36.7 ms + 499 steps of about 5.21 ms = 2.64 s.</p>
<p>A server is judged on throughput (output tokens per second over all requests) under a <b>service-level objective</b> (SLO) on both latencies, stated as percentiles because users notice the slow tail: for example, 99% of requests with TTFT under 500 ms and TPOT under 50 ms. <b>Goodput</b> is the highest request rate at which a server still meets its SLO; DistServe defines it per GPU.</p>`,
    fig: {
      type: "timeline", caption: "The first 100 ms of that request: one prefill, then decode steps; each step streams one token.",
      rows: ["GPU", "tokens out"], total: 100, unit: "ms",
      items: (function () {
        const m = LLAMA["70B"], P = 2048, it = [];
        const pf = passTime(m, { tokens: P, attn: (attnFlops(m, 1) * P * (P + 1)) / 2 }).t * 1e3;
        it.push({ row: 0, start: 0, len: pf, label: `prefill ${pf.toFixed(1)} ms`, kind: "compute", tip: `prefill of 2,048 tokens: ${pf.toFixed(2)} ms (compute-bound)` });
        it.push({ row: 1, start: pf - 0.6, len: 1.2, kind: "comm", tip: `token 1 at ${pf.toFixed(1)} ms: TTFT` });
        let t = pf;
        for (let i = 1; t < 99; i++) {
          const d = decodeStep(m, 1, P + i).t * 1e3;
          if (t + d > 100) break;
          it.push({ row: 0, start: t, len: d, label: String(i), kind: "mem", tip: `decode step ${i}: ${d.toFixed(2)} ms (memory-bound)` });
          t += d;
          it.push({ row: 1, start: t - 0.6, len: 1.2, kind: "comm", tip: `token ${i + 1} at ${t.toFixed(1)} ms` });
        }
        return it;
      })(),
      note: "Lower bounds for one request alone: prefill at the peak math rate of 8 GPUs, decode at their full HBM bandwidth. Hover a box for its time.",
    },
    take: "TTFT is set by the prefill and the queue in front of it; TPOT is set by one decode step, which every request in the batch shares.",
    terms: [
      { term: "Time to first token (TTFT)", def: "Time from a request's arrival to its first output token: queueing, tokenization and the prefill." },
      { term: "Time per output token (TPOT)", def: "Interval between consecutive output tokens of one request: one decode step. Also called inter-token latency." },
      { term: "Service-level objective (SLO)", def: "A latency target a server must meet, usually a percentile: for example, 99% of requests with TTFT under 500 ms." },
      { term: "Goodput", def: "The highest request rate at which a server meets its SLO (DistServe defines it per GPU)." },
    ],
    src: ["Zhong et al., DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving, OSDI 2024"],
  });

  Deck.slide({
    id: "serve-min-step",
    title: "In BF16, Llama 3.1 70B on 8 H100s needs at least 5.19 ms per step",
    body: String.raw`<p>A decode step must read every weight it multiplies. Count them from the published configuration: 80 layers with model width 8,192, 64 query heads and 8 key/value heads of 128 numbers each, an MLP of width 28,672, and an output layer over the 128,256-token vocabulary. The input embedding table does not count: a step looks up one row of it per token.</p>
<p>With tensor parallelism (8.5) each of the 8 GPUs holds one eighth of every weight matrix and reads its share at the same time as the others. The step can finish no sooner than 17.38 GB ÷ 3.35 TB/s.</p>
<p>That bound, 5.19 ms per step or 193 tokens per second for one request, holds however fast the tensor cores are: the math of a token takes 17.6 µs at their peak. Everything else (the KV cache, the communication between the 8 GPUs at every layer, HBM running below its rated bandwidth) only adds to it.</p>`,
    fig: {
      type: "steps", caption: "The bandwidth bound for one decode step, from the model's configuration",
      steps: [
        { html: String.raw`Per layer: attention \(2 \cdot 8192 \cdot 8192 + 2 \cdot 8192 \cdot 1024 = 150{,}994{,}944\); MLP \(3 \cdot 8192 \cdot 28672 = 704{,}643{,}072\); two norms \(16{,}384\). Total <b>855,654,400</b>.`, note: "q and o are 8,192 wide; k and v are 8 × 128 = 1,024" },
        { html: String.raw`Read per step: \(80 \times 855{,}654{,}400 + 128{,}256 \times 8192 + 8192 = 69{,}503{,}033{,}344\) weights.`, note: "layers + output layer + final norm" },
        { html: String.raw`In BF16 (2 bytes): <b>139.0 GB</b>; per GPU with 8-way tensor parallelism: <b>17.38 GB</b>.` },
        { html: String.raw`\(17.38\ \text{GB} \div 3.35\ \text{TB/s} = 5.19\ \text{ms}\), so at most <b>193 tokens/s</b> for one request.` },
        { html: String.raw`Math per token: \(2 \times 69.5\text{B} = 139.0\) GFLOP, \(17.6\ \mu\text{s}\) at \(8 \times 989\) TFLOP/s: 0.34% of the step.`, note: "the tensor cores wait" },
      ],
    },
    take: "Minimum decode step = bytes of weights per GPU ÷ HBM bandwidth. For 70B in BF16 on 8 H100s: 5.19 ms.",
    src: ["Llama 3.1 70B config.json (Meta, 2024)", "NVIDIA H100 Tensor Core GPU datasheet (H100 SXM)"],
  });

  Deck.slide({
    id: "serve-batch",
    title: "Batching shares one read of the weights among many requests",
    body: String.raw`<p>At batch 1 a decode step spends 5.19 ms reading weights to do 17.6 µs of math. Put B requests in the same step and the weights are read once for all of them, so the step does B times the work in almost the same time. Three limits end this.</p>
<ul><li><b>Cache reads.</b> Every request reads its own keys and values each step: 327,680 bytes per token of context, 0.67 GB for a 2,048-token request. From 208 such requests on, the caches outweigh the weights.</li>
<li><b>Compute.</b> Each token costs 139.0 GFLOP plus attention. With 128-token contexts the step turns compute-bound at 324 requests; at 1,477 tokens of context or more it never does, because the cache reads grow faster than the math.</li>
<li><b>Capacity.</b> With 92% of the memory CUDA reports usable (8.2), 626.8 GB minus 141.1 GB of weights leaves 485.7 GB for caches: 723 requests of 2,048 tokens.</li></ul>
<p>Per-request speed falls as B grows while total throughput rises: at 2,048 tokens of context, 64 requests get 147 tokens/s each and 9,426 tokens/s together.</p>`,
    fig: { type: "serve-batch", caption: "Llama 3.1 70B in BF16 on 8 H100s, every request at the same context length. Time = max(bytes ÷ 26.8 TB/s, FLOPs ÷ 7,912 TFLOP/s); activations ignored. Dashed: the caches do not fit in 92% of the 79.32 GiB that CUDA reports per GPU." },
    check: {
      q: "Llama 3.1 70B decodes alone on 8 H100s with 2,048 tokens of context. A second request with the same context joins the batch. What happens?",
      options: ["The step time doubles and total throughput stays the same", "The step time rises by about 0.5% and total throughput nearly doubles", "The step time halves", "Nothing changes: the second request waits for the first"],
      answer: 1,
      why: "The step is spent reading 139.0 GB of weights, which both requests share. The second request adds its own cache (0.67 GB) and 139 GFLOP of math, which hides under the reads: 5.21 ms becomes 5.24 ms, and throughput goes from 192 to 382 tokens/s.",
    },
    src: ["Llama 3.1 70B config.json (Meta, 2024)", "NVIDIA H100 Tensor Core GPU datasheet (H100 SXM)"],
  });

  // ------------------------------------------------------------------ 8.2
  Deck.chapter({ title: "The KV cache" });

  Deck.slide({
    id: "serve-kvcache",
    title: "Past keys and values never change, so the server keeps them",
    example: "tokens ids prefix:3",
    body: String.raw`<p>Attention at a new position needs the keys and values of every earlier position in every layer. Those never change once computed. A token's key and value in layer l come from its own residual stream, which, because of the causal mask, depends only on that token and the ones before it. Nothing generated later can alter them.</p>
<p>So the server computes each token's keys and values once and stores them: the <b>KV cache</b>. A decode step computes them for the one new token and reads the rest from memory. Without the cache, step n would recompute them for all n tokens, and an answer of n tokens would cost about n²/2 token computations instead of n.</p>
<p>In our model a token's entry is a 16-number key and a 16-number value in each of the 2 layers: 64 numbers. Decoding the 8 input positions of our query from the 3-token prompt computes 8 entries with the cache and 33 without (3 + 4 + 5 + 6 + 7 + 8). The plate checks live that the logits are the same either way.</p>`,
    layout: "wide",
    fig: { type: "tinylm", widget: "kvcache", caption: "The cache of our query as it is decoded. Each row is one token's keys and values in both layers and heads." },
    take: "The KV cache trades memory for compute: no step recomputes the past, but every step still reads all of it.",
    terms: [{ term: "KV cache", def: "The stored keys and values of every earlier token of a request, in every layer, so that each decode step computes them only for the new token." }],
  });

  Deck.slide({
    id: "serve-kvsize",
    title: "A token's cache entry is 2 × layers × KV heads × head size numbers",
    math: { tex: String.raw`\text{bytes per token} = 2 \cdot L \cdot h_{kv} \cdot d_{head} \cdot b`, note: "2: a key and a value · L: layers · h<sub>kv</sub>: key/value heads · d<sub>head</sub>: numbers per head · b: bytes per number" },
    body: String.raw`<p>Every layer stores one key vector and one value vector per key/value head for every token. The cache therefore grows linearly with the context and with the number of requests, and every decode step reads all of it.</p>
<p>Our model in FP32: 2 × 2 layers × 2 heads × 8 × 4 bytes = 256 bytes per token, and 2,048 bytes for the 8 positions of our query. Llama 3.1 70B in BF16: 2 × 80 × 8 × 128 × 2 = 327,680 bytes (320 KiB) per token. At its full context of 131,072 tokens, one request needs 42.9 GB (40 GiB): half of an H100's 80 GiB, for a single conversation. Llama 3.1 8B needs 131,072 bytes per token and 405B needs 516,096.</p>
<p>Storing the cache in FP8 halves every figure (8.4).</p>`,
    fig: {
      type: "calc", caption: "KV cache size. Choose a model, the precision of the cache, the context length and the number of requests.",
      inputs: [
        { id: "m", label: "Model", options: [{ label: "8B", value: "8B" }, { label: "70B", value: "70B" }, { label: "405B", value: "405B" }], value: "70B" },
        { id: "b", label: "Cache precision", options: [{ label: "BF16", value: 2 }, { label: "FP8", value: 1 }], value: 2 },
        { id: "lc", label: "Context per request (tokens)", min: 10, max: 17, step: 1, value: 17, fmt: (v) => commas(pow2(v)) },
        { id: "n", label: "Requests", min: 1, max: 256, step: 1, value: 1, fmt: "int" },
      ],
      outputs: [
        { label: "bytes per token", f: (v) => kvPerToken(LLAMA[v.m], v.b), fmt: (x) => `${commas(x)} B` },
        { label: "per request", f: (v) => kvPerToken(LLAMA[v.m], v.b) * pow2(v.lc), fmt: (x) => `${(x / 1e9).toFixed(2)} GB` },
        { label: "all requests", f: (v) => kvPerToken(LLAMA[v.m], v.b) * pow2(v.lc) * v.n, fmt: (x) => `${(x / 1e9).toFixed(1)} GB` },
        { label: "H100s (80 GiB each) to hold it", f: (v) => (kvPerToken(LLAMA[v.m], v.b) * pow2(v.lc) * v.n) / H100.mem, fmt: (x) => x.toFixed(x < 10 ? 2 : 0) },
      ],
      note: (v) => { const m = LLAMA[v.m]; return `2 × ${m.L} layers × ${m.KV} KV heads × ${m.hd} × ${v.b} byte${v.b > 1 ? "s" : ""} = ${commas(kvPerToken(m, v.b))} bytes per token, × ${commas(pow2(v.lc))} tokens × ${v.n} request${v.n > 1 ? "s" : ""}.`; },
    },
    take: "KV bytes = per-token size × context × requests. For Llama 3.1 70B in BF16: 0.33 MB per token, 42.9 GB at 128K.",
    src: ["Llama 3.1 8B, 70B and 405B config.json (Meta, 2024)"],
  });

  Deck.slide({
    id: "serve-gqa",
    title: "Sharing keys and values among query heads shrinks the cache",
    body: String.raw`<p>The cache stores keys and values per key/value head, so the number of those heads sets its size. Part 5 introduced the options; the counts below are numbers stored per token in each layer.</p>
<ul><li><b>Multi-head attention</b> (MHA): every query head keeps its own keys and values, 2 × 64 × 128 = 16,384 numbers in a 70B-shaped model. Over 80 layers in BF16 that is 2,621,440 bytes per token, and 343.6 GB (320 GiB) for one 131,072-token request: all the HBM of four H100s.</li>
<li><b>Grouped-query attention</b> (GQA), as in Llama 3.1 70B: 64 query heads share 8 key/value heads, so 2,048 numbers, 8 times fewer.</li>
<li><b>Multi-query attention</b> (MQA): all query heads share one key/value head: 256 numbers.</li>
<li><b>Multi-head latent attention</b> (MLA), in DeepSeek-V2 and V3: a 512-number compressed vector and a 64-number rotary key, 576 numbers, 3.6 times fewer than GQA. Every head's keys and values are linear maps of the compressed vector, and those maps can be folded into the query and output projections, so decode attends over the cached vectors directly. Over DeepSeek-V3's 61 layers in BF16: 70,272 bytes per token.</li></ul>
<p>Pope et al. report that multi-query attention, with the cache split over chips by batch, allowed up to 32 times longer contexts for PaLM 540B.</p>`,
    fig: {
      type: "bars", caption: "Numbers cached per token in each layer: head size 128, and 64 query heads for MHA, GQA and MQA",
      items: [
        { label: "MHA (64)", value: 16384, text: "16,384" },
        { label: "GQA (8)", value: 2048, text: "2,048", hi: true },
        { label: "MLA", value: 576, text: "576" },
        { label: "MQA (1)", value: 256, text: "256" },
      ],
      note: "Per token over 80 layers in BF16: MHA 2,621 KB, GQA 328 KB, MLA 92 KB, MQA 41 KB. At 131,072 tokens: 343.6 GB, 42.9 GB, 12.1 GB and 5.4 GB per request.",
    },
    take: "The KV cache scales with the number of key/value heads; query heads add nothing to it. GQA's 8 groups cut Llama 3.1 70B's cache 8-fold.",
    src: ["Llama 3.1 70B config.json (Meta, 2024)", "DeepSeek-AI, DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, 2024 (section 2.1)", "DeepSeek-AI, DeepSeek-V3 Technical Report, 2024 (sections 2.1.1 and 4.2)", "Pope et al., Efficiently Scaling Transformer Inference, MLSys 2023"],
  });

  Deck.slide({
    id: "serve-kv-budget",
    title: "Memory left after the weights decides how many requests fit",
    body: String.raw`<p>An engine fills a GPU in a fixed order. vLLM may use a fraction of the GPU's memory, set by <code>gpu_memory_utilization</code> (0.92 by default in current releases, 0.9 before). It loads the weights, measures the memory a forward pass needs for activations, and divides what is left into KV-cache blocks. The number of blocks caps how many tokens all running requests can hold together.</p>
<p>The fraction applies to the total CUDA reports. An "80 GB" H100 has 80 GiB of HBM3 (85.9 × 10⁹ bytes), of which CUDA reports 79.32 GiB (85.17 × 10⁹). Llama 3.1 8B on one H100: 0.92 × 85.17 GB = 78.36 GB; minus 16.06 GB of BF16 weights leaves 62.30 GB, before activations. At 131,072 bytes per token that holds 475,274 tokens: 58 requests of 8,192 tokens, or 3 of the full 131,072.</p>
<p>Llama 3.1 70B on 8 H100s with tensor parallelism: 626.8 GB minus 141.1 GB of weights leaves 485.7 GB, which is 1,482,354 tokens: 180 requests of 8,192 tokens.</p>
<p>Weights are a fixed cost; the cache is the variable one. Halving the cache (FP8, fewer KV heads) doubles the batch, and through 8.1, the throughput.</p>`,
    fig: {
      type: "calc", caption: "Memory plan of one replica, from the 79.32 GiB per GPU that CUDA reports. Activation memory is left out; it comes off the cache.",
      inputs: [
        { id: "m", label: "Replica", options: [{ label: "8B on 1 H100", value: "8B" }, { label: "70B on 8 H100s", value: "70B" }], value: "8B" },
        { id: "u", label: "gpu_memory_utilization", min: 0.8, max: 0.95, step: 0.01, value: 0.92, fmt: "f2" },
        { id: "lc", label: "Context per request (tokens)", min: 10, max: 17, step: 1, value: 13, fmt: (v) => commas(pow2(v)) },
      ],
      outputs: [
        { label: "left for the cache", f: (v) => budget(v).kv, fmt: (x) => `${(x / 1e9).toFixed(2)} GB` },
        { label: "tokens that fit", f: (v) => Math.floor(budget(v).kv / kvPerToken(LLAMA[v.m])), fmt: "int" },
        { label: "requests at this context", f: (v) => Math.floor(budget(v).kv / kvPerToken(LLAMA[v.m]) / pow2(v.lc)), fmt: "int" },
      ],
      after: (el, v) => { const b = budget(v); memoryBar(el, [{ label: "weights", v: b.w / 1e9, cls: "serve-c-w" }, { label: "KV cache", v: b.kv / 1e9, cls: "serve-c-kv" }, { label: "not used", v: b.rest / 1e9, cls: "serve-c-free" }], b.total / 1e9); },
      note: (v) => { const b = budget(v); return `${v.u.toFixed(2)} × ${(b.total / 1e9).toFixed(2)} GB = ${(b.use / 1e9).toFixed(2)} GB usable; weights ${(b.w / 1e9).toFixed(2)} GB; cache ${(b.kv / 1e9).toFixed(2)} GB ÷ ${commas(kvPerToken(LLAMA[v.m]))} B per token.`; },
    },
    take: "Requests that fit = (usable memory − weights − activations) ÷ (cache per token × context).",
    src: ["vLLM engine arguments documentation (gpu_memory_utilization), v0.32", "PyTorch out-of-memory report on an H100 80GB (total capacity 79.32 GiB), Hugging Face discussion nvidia/Llama-3.1-Nemotron-70B-Instruct #10", "NVIDIA H100 Tensor Core GPU architecture whitepaper (80 GB HBM3)", "Llama 3.1 8B and 70B config.json (Meta, 2024)"],
  });

  Deck.slide({
    id: "serve-prefix",
    title: "Requests that start with the same tokens can share their cache",
    example: "tokens ids",
    body: String.raw`<p>A token's key and value depend only on that token and the tokens before it. Two requests that begin with the same tokens therefore have identical keys and values over that shared beginning, and the server can compute and store them once. This is <b>prefix caching</b>. It pays most for a system prompt or few-shot examples that precede every request, and for the earlier turns of a conversation.</p>
<p>SGLang keeps cached sequences in a radix tree over tokens (RadixAttention). A new request walks the tree to its longest cached prefix and prefills only the rest; least-recently-used leaves are evicted when memory runs short, and waiting requests with longer cached prefixes run first. SGLang, which combines this cache with other optimizations, reports up to 6.4 times the throughput of earlier systems.</p>
<p>The plate feeds 12 of our grammar's queries, each used whole as a prompt: 73 of their 108 tokens need computing. A query seen before still recomputes its last token, whose hidden state (not stored in the cache) gives the next-token distribution.</p>`,
    fig: { type: "serve-prefix", caption: "Queries arriving one by one, each used as a prompt. Hatched tokens are found in the cache that earlier requests left." },
    take: "Equal prefixes give equal keys and values. Cache them once and every later request skips that part of its prefill.",
    terms: [{ term: "Prefix caching", def: "Reusing the stored keys and values of a token sequence for every request that starts with the same tokens, so their prefill skips that part." }],
    src: ["Zheng et al., SGLang: Efficient Execution of Structured Language Model Programs, NeurIPS 2024"],
  });

  Deck.slide({
    id: "serve-kv-offload",
    title: "A cache that does not fit can wait in CPU memory",
    body: String.raw`<p>When HBM runs out, a server can copy caches it does not need right now to the host's DRAM and copy them back later, instead of discarding them and running the prefill again. The vLLM paper calls this swapping. The same move lets a prefix cache hold more than fits in HBM.</p>
<p>The trade is PCIe bytes against prefill FLOPs. An H100 SXM's PCIe Gen5 link is rated at 128 GB/s, 64 GB/s in each direction. A 32,768-token Llama 3.1 70B cache is 10.74 GB, or 1.34 GB per GPU when each of 8 GPUs holds one KV head: 21.0 ms to copy back if every link runs at its rated speed. Recomputing it is a 32,768-token prefill of 5,962 TFLOP: 753.6 ms at the peak of 8 GPUs, 36 times longer.</p>
<p>The gap widens with context, since recomputing attention grows with the square of the length while the bytes grow linearly. Quantizing the cache (8.4) halves both the bytes to move and the bytes to keep.</p>`,
    fig: {
      type: "calc", caption: "Bringing a cache back: copy it over PCIe, or compute it again with a prefill at peak FLOP/s.",
      inputs: [
        { id: "m", label: "Replica", options: [{ label: "8B on 1 H100", value: "8B" }, { label: "70B on 8 H100s", value: "70B" }], value: "70B" },
        { id: "lc", label: "Tokens in the cache", min: 10, max: 17, step: 1, value: 15, fmt: (v) => commas(pow2(v)) },
      ],
      outputs: [
        { label: "cache size", f: (v) => kvPerToken(LLAMA[v.m]) * pow2(v.lc), fmt: (x) => `${(x / 1e9).toFixed(2)} GB` },
        { label: "copy back over PCIe", f: (v) => reload(v).copy, fmt: (x) => ms(x, 1) },
        { label: "recompute (prefill)", f: (v) => reload(v).prefill, fmt: (x) => ms(x, 1) },
        { label: "recompute ÷ copy", f: (v) => reload(v).prefill / reload(v).copy, fmt: "x" },
      ],
      note: (v) => `Copy: ${commas(kvPerToken(LLAMA[v.m]))} B × ${commas(pow2(v.lc))} tokens, split over ${v.m === "70B" ? 8 : 1} GPU${v.m === "70B" ? "s" : ""}, each at 64 GB/s. Recompute: 2 × weights × tokens plus attention, at ${v.m === "70B" ? "8 × " : ""}989 TFLOP/s.`,
    },
    check: {
      q: "How much memory does the BF16 KV cache of one 32,768-token request take in Llama 3.1 70B (80 layers, 64 query heads, 8 KV heads of 128)?",
      options: ["5.4 GB", "10.7 GB", "85.9 GB", "141 GB"],
      answer: 1,
      why: "2 (key and value) × 80 × 8 × 128 × 2 bytes = 327,680 bytes per token, × 32,768 tokens = 10.7 GB. 5.4 GB forgets the values, 85.9 GB uses the 64 query heads instead of the 8 KV heads, and 141 GB is the weights.",
    },
    src: ["Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023", "NVIDIA H100 Tensor Core GPU architecture whitepaper (PCIe Gen5, H100 SXM)", "Llama 3.1 70B config.json (Meta, 2024)"],
  });

  // ------------------------------------------------------------------ 8.3
  Deck.chapter({ title: "Batching and scheduling" });

  Deck.slide({
    id: "serve-static",
    title: "Static batching leaves slots idle until the longest request ends",
    body: String.raw`<p>The simplest server collects a batch of requests, runs prefill and decode for all of them together, and returns the batch when every request has finished. Requests finish at different times: one answer needs 2 tokens, another 8. A finished request's slot stays in the batch, doing nothing, until the longest one ends.</p>
<p>The plate shows 4 requests that need 5, 2, 8 and 3 output tokens. The batch runs 8 decode steps; of its 32 slot-steps, 18 produce tokens, 56%. A request that arrives during step 1 waits for all 8 before it can start, and B's user, whose answer was ready after 2 steps, receives it after 8.</p>
<p>The waste grows with the spread of output lengths, which in real traffic runs from a few tokens to thousands. Prompts of different lengths are also padded to the longest in the batch for the prefill.</p>`,
    fig: {
      type: "timeline", caption: "A static batch of 4 requests needing 5, 2, 8 and 3 tokens. Dashed: a slot holding a finished request.",
      rows: ["slot 1: A", "slot 2: B", "slot 3: C", "slot 4: D", "queue: E"], total: 8, unit: "decode steps",
      items: (function () {
        const lens = [5, 2, 8, 3], kinds = ["s1", "s2", "s3", "s4"], it = [];
        lens.forEach((n, r) => {
          for (let k = 0; k < n; k++) it.push({ row: r, start: k, len: 1, label: "ABCD"[r], kind: kinds[r], tip: `step ${k + 1}: ${"ABCD"[r]} token ${k + 1}` });
          if (n < 8) it.push({ row: r, start: n, len: 8 - n, label: "idle", kind: "idle", tip: `${"ABCD"[r]} finished after ${n} steps; slot idle for ${8 - n}` });
        });
        it.push({ row: 4, start: 0.5, len: 7.5, label: "E waits for the whole batch", kind: "idle", tip: "E arrives during step 1 and starts after step 8" });
        return it;
      })(),
    },
    take: "A static batch runs as long as its longest request. Every shorter request wastes its slot for the difference.",
  });

  Deck.slide({
    id: "serve-continuous",
    title: "Continuous batching refills the batch at every step",
    example: "tokens ids",
    body: String.raw`<p>Orca (Yu et al., 2022) schedules at the granularity of one iteration: after every forward pass the scheduler retires finished requests and admits waiting ones. This is <b>iteration-level scheduling</b>, now called continuous batching. A slot freed at step 2 is reused at step 3.</p>
<p>Requests in one step are at different positions: some are in prefill with many tokens, others decode one token at different context lengths. Orca's <b>selective batching</b> flattens all their tokens into one matrix for every operation except attention (the linear layers, layer norm, GELU) and runs attention separately per request, since attention has no weights to share and each request has its own cache.</p>
<p>On the 10 requests of the plate, continuous batching finishes in 13 steps instead of 20, keeps 87% of the slots busy instead of 56%, and returns the average request at step 7.6 instead of 10.7. Orca reported 36.9 times the throughput of FasterTransformer at the same latency on GPT-3 175B.</p>`,
    fig: { type: "tinylm", widget: "batching", caption: "Ten requests, 4 slots: static batches against per-step scheduling" },
    take: "Schedule every step: a slot freed by a finished request is refilled at the next one.",
    terms: [
      { term: "Continuous batching", def: "Iteration-level scheduling: finished requests leave and waiting ones join after every forward pass, instead of once per batch (Orca)." },
      { term: "Selective batching", def: "Running every operation except attention on the flattened tokens of all requests in a step, and attention separately per request (Orca)." },
    ],
    src: ["Yu et al., Orca: A Distributed Serving System for Transformer-Based Generative Models, OSDI 2022"],
  });

  Deck.slide({
    id: "serve-paged",
    title: "PagedAttention stores each request's cache in fixed-size blocks",
    example: "tokens ids",
    body: String.raw`<p>If each request's cache must be contiguous, the server has to reserve room for the longest answer the request might produce, because it cannot move the cache later. Most of that reservation is never used, and the gaps between reservations are too small for new requests. In the vLLM paper's measurements, re-implementations of Orca that reserve for the maximum length, for the next power of two, or for the true output length kept token states in 20.4%, 26.8% and 38.2% of their KV memory.</p>
<p>vLLM's <b>PagedAttention</b> splits the cache into blocks of a fixed number of tokens (16 by default) placed anywhere in HBM. Each request has a <b>block table</b> mapping its logical blocks (tokens 0–15, 16–31, and so on) to physical blocks, and the attention kernel follows the table. A new block is allocated only when the last one fills, so a request wastes at most one partly filled block; vLLM measured 96.3% of KV memory holding token states. Blocks also carry reference counts, so requests with a common prefix can point at the same physical blocks.</p>
<p>The paper reports 2 to 4 times the throughput of FasterTransformer and Orca at the same latency.</p>`,
    fig: { type: "tinylm", widget: "paged", caption: "Four requests with contiguous reservations (left) and in blocks of 4 tokens (right). Blue: prompt tokens; green: generated; dashed: reserved but empty." },
    take: "Paging the cache removes the reservation and fragmentation waste, so more requests fit in the same memory.",
    terms: [
      { term: "PagedAttention", def: "vLLM's attention over a KV cache stored in fixed-size blocks anywhere in memory, located through a per-request block table." },
      { term: "Block table", def: "A request's map from logical KV blocks (consecutive token ranges) to physical blocks in GPU memory." },
    ],
    src: ["Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023"],
  });

  Deck.slide({
    id: "serve-chunked",
    title: "Chunked prefill keeps decode steps short while a long prompt arrives",
    body: String.raw`<p>A new prompt's prefill competes with running decodes for the same steps. An 8,192-token prompt on Llama 3.1 70B, run in one step with 64 requests decoding, needs 1,236 TFLOP: 156.2 ms at the peak of 8 GPUs, during which no running request gets a token. Their time per output token jumps from 6.79 ms to 156.2 ms.</p>
<p><b>Chunked prefill</b> (Sarathi, Agrawal et al.) runs the prompt in pieces, each in a step beside the decodes. Sarathi-Serve sets a token budget per step, schedules every running decode first and fills the rest with prompt chunks, so decodes never stall. With a 512-token budget, the prompt takes 19 steps of at most 10.20 ms, and its first token arrives after 180.4 ms instead of 156.2 ms.</p>
<p>Chunks use math that memory-bound decode steps leave idle. Very small chunks cost more because every step reads the weights and all live caches, at least 6.79 ms here. With a 128-token budget the prompt takes 128 steps and its first token arrives after 875.4 ms, of which re-reading earlier chunks' caches is only 6.4 ms. Sarathi-Serve reports 2.6 times vLLM's serving capacity for Mistral-7B on one A100 and up to 5.6 times for Falcon-180B; vLLM V1 enables chunked prefill by default.</p>`,
    fig: { type: "serve-chunk", caption: "An arriving prompt processed in chunks next to 64 running decodes. Each tick below is one token for every running request." },
    take: "A token budget per step bounds the decode step time; the long prompt pays with a slightly later first token.",
    terms: [{ term: "Chunked prefill", def: "Splitting a prompt's prefill into pieces that run in successive steps alongside other requests' decodes, so decode steps stay short." }],
    src: ["Agrawal et al., SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills, 2023", "Agrawal et al., Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve, OSDI 2024", "vLLM optimization documentation, v0.32"],
  });

  Deck.slide({
    id: "serve-preempt",
    title: "When blocks run out, the newest request gives its blocks back",
    body: String.raw`<p>Requests grow by one token per step, and the server does not know how long each answer will be. Admitting requests optimistically keeps the GPU full, so sometimes a running request needs a new block when none is free. The server must <b>preempt</b> a request.</p>
<p>vLLM serves first come, first served and preempts the latest-arrived request, all of its blocks at once. Two ways bring it back. <b>Swapping</b> copies the blocks to CPU memory and back (8.2). <b>Recomputation</b> discards them and later runs one prefill over the prompt plus the tokens generated so far. The vLLM paper found swapping slow for small blocks, since each block is a separate small PCIe transfer, while recomputation's cost does not depend on block size. vLLM V1 recomputes by default.</p>
<p>In the plate, 4 requests share 11 blocks of 4 tokens. At step 5, D, the last to arrive, is preempted after generating 4 tokens. When C finishes at step 6, D is readmitted and recomputes its 8 tokens in one prefill; it finishes at step 13 instead of 12. With 10 blocks, D is preempted twice; with 13 or more, never.</p>`,
    fig: { type: "serve-preempt", caption: "Four requests (prompt + output tokens) sharing a small cache. Drag the number of blocks." },
    take: "Preemption trades repeated work for a full GPU. Recomputing costs a prefill; swapping costs PCIe traffic.",
    terms: [{ term: "Preemption", def: "Taking a running request's KV blocks away when memory runs out; it resumes later by swapping its cache back in or recomputing it." }],
    src: ["Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023 (sections 4.5, 7.3)", "vLLM optimization documentation, v0.32"],
  });

  Deck.slide({
    id: "serve-disagg",
    title: "Prefill and decode can run on separate pools of GPUs",
    body: String.raw`<p>Even chunked, prefill and decode share every step: a long prefill slows the decodes in its step, and the two phases want different batch sizes and parallelism. <b>Disaggregation</b> runs them on separate GPUs. A prefill instance processes the prompt and sends the request's KV cache to a decode instance, which generates the rest. Each pool is sized and parallelized for its own phase.</p>
<p>The price is moving the cache. A 2,048-token Llama 3.1 70B prompt leaves 0.67 GB of keys and values. With 8-way tensor parallelism each GPU holds 83.9 MB of it, and in a DGX H100 each GPU has its own 400 Gb/s network adapter, so the 8 transfers take 1.68 ms in parallel, against a prefill of at least 36.7 ms. Splitwise sends each layer's cache as soon as that layer is done, overlapping transfer with the rest of the prefill, and measured about 5 ms left unhidden on H100 machines with 400 Gb/s InfiniBand.</p>
<p>DistServe reports serving 7.4 times more requests than DeepSpeed-MII, or a 12.6 times tighter SLO than vLLM, with over 90% of requests within their latency targets. Splitwise reports 1.4 times the throughput at 20% lower cost.</p>`,
    fig: {
      type: "calc", caption: "Shipping a prompt's cache from the prefill pool to the decode pool: Llama 3.1 70B, BF16 cache, 8 GPUs per side, one link per GPU",
      inputs: [
        { id: "lp", label: "Prompt (tokens)", min: 9, max: 15, step: 1, value: 11, fmt: (v) => commas(pow2(v)) },
        { id: "g", label: "Link", options: [{ label: "100 Gb/s", value: 100 }, { label: "200 Gb/s", value: 200 }, { label: "400 Gb/s", value: 400 }, { label: "800 Gb/s", value: 800 }], value: 400 },
      ],
      outputs: [
        { label: "cache per GPU", f: (v) => (kvPerToken(LLAMA["70B"]) * pow2(v.lp)) / 8, fmt: (x) => `${(x / 1e6).toFixed(1)} MB` },
        { label: "transfer, 8 links in parallel", f: (v) => (kvPerToken(LLAMA["70B"]) * pow2(v.lp)) / 8 / ((v.g * 1e9) / 8), fmt: (x) => ms(x, 2) },
        { label: "prefill at peak, 8 GPUs", f: (v) => passTime(LLAMA["70B"], { tokens: pow2(v.lp), attn: (attnFlops(LLAMA["70B"], 1) * pow2(v.lp) * (pow2(v.lp) + 1)) / 2 }).t, fmt: (x) => ms(x, 1) },
        { label: "left after the prefill, sent layer by layer", f: (v) => { const X = (kvPerToken(LLAMA["70B"]) * pow2(v.lp)) / 8 / ((v.g * 1e9) / 8), Tp = passTime(LLAMA["70B"], { tokens: pow2(v.lp), attn: (attnFlops(LLAMA["70B"], 1) * pow2(v.lp) * (pow2(v.lp) + 1)) / 2 }).t; return Math.max(X / 80, X - (79 / 80) * Tp); }, fmt: (x) => ms(x, 2) },
      ],
      note: "Each GPU sends its eighth of the cache over its own link. Layer by layer, layer i's cache is ready after i/80 of the prefill and each link sends layers in order, so the time left after the prefill is max(one layer's transfer, whole transfer − 79/80 of the prefill).",
    },
    check: {
      q: "A server decodes 64 requests and receives an 8,192-token prompt without chunked prefill. What do the 64 running requests see?",
      options: ["Nothing: decode and prefill use different hardware units", "One token-to-token gap of about 156 ms instead of 6.8 ms", "They are preempted and must recompute their caches", "Their time per token halves while the prompt is processed"],
      answer: 1,
      why: "The prefill and the decodes share one step. Its 1,236 TFLOP keep 8 H100s busy for 156.2 ms, so every running request waits that long for its next token, instead of the 6.79 ms of a decode step alone. Chunking or disaggregation removes that stall.",
    },
    terms: [{ term: "Disaggregation", def: "Running prefill and decode on separate GPU pools and moving each request's KV cache from the first to the second." }],
    src: ["Zhong et al., DistServe, OSDI 2024", "Patel et al., Splitwise: Efficient Generative LLM Inference Using Phase Splitting, ISCA 2024", "NVIDIA DGX H100 User Guide (8 single-port ConnectX-7 adapters, up to 400 Gb/s, one per GPU)"],
  });

  // ------------------------------------------------------------------ 8.4
  Deck.chapter({ title: "Making decode faster" });

  Deck.slide({
    id: "serve-speculative",
    title: "Speculative decoding drafts tokens and verifies them in one pass",
    example: "tokens ids prefix:3",
    body: String.raw`<p>A decode step at small batch reads every weight to produce one token, so scoring k + 1 positions in one pass costs about as much as scoring 1. <b>Speculative decoding</b> uses that: a small draft model proposes k tokens; the large model scores them all in one pass, keeps the longest run it accepts and adds one token of its own.</p>
<p>A drafted token x is accepted with probability min(1, p(x)/q(x)), so a draft is accepted with probability Σ min(p, q), the overlap of the two distributions. With acceptance rate α and k drafts, a pass yields the first expression below in tokens. The speedup also pays for k draft steps of relative cost c: the second expression (Leviathan et al.). At α = 0.8, k = 4 and c = 0.1: 3.36 tokens per pass, 2.40 times faster.</p>
<p>The plate's draft is our model at training step 150. After <code>WHERE</code> it puts 0.458 on <code>id</code> and still gives <code>name</code>, the column already selected, 0.162; the final model gives <code>name</code> 0.001: overlap 0.668. Before the digit both spread over 1 to 9: overlap 0.926. Leviathan et al. report 2 to 3 times faster T5-XXL decoding, Chen et al. 2 to 2.5 times for Chinchilla 70B.</p>`,
    math: [
      { tex: String.raw`E[\text{tokens per pass}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}`, note: "p, q: the large and the draft model's probabilities · α: probability each draft is accepted · k: drafted tokens per pass" },
      { tex: String.raw`\text{speedup} = \frac{1 - \alpha^{k+1}}{(1 - \alpha)(kc + 1)}`, note: "c: time of one draft step ÷ time of one large-model step" },
    ],
    fig: { type: "tinylm", widget: "speculative", caption: "Speculative decoding of our query: step-150 draft, final model as the verifier" },
    take: "The acceptance rate is the overlap Σ min(p, q) of the draft's and the target's distributions, whatever their confidence.",
    terms: [{ term: "Speculative decoding", def: "Decoding where a cheap draft proposes several tokens and the large model verifies them in one pass, accepting each with probability min(1, p/q)." }],
    src: ["Leviathan, Kalman and Matias, Fast Inference from Transformers via Speculative Decoding, ICML 2023", "Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling, 2023"],
  });

  Deck.slide({
    id: "serve-spec-exact",
    title: "The accept-or-resample rule reproduces the large model's distribution",
    example: "tokens ids prefix:5",
    body: String.raw`<p>Speculative decoding changes the cost of sampling and leaves the sampled distribution as it was. When a draft x is rejected, the large model samples a replacement from the leftover distribution max(0, p − q), normalized. The total probability of emitting x is then exactly p(x).</p>
<p>The proof has two lines. The draft emits x and it is kept with probability q(x) · min(1, p(x)/q(x)) = min(p(x), q(x)). A rejection happens with probability 1 − Σ min(p, q), which equals Σ max(0, p − q), so the replacement path adds exactly max(0, p(x) − q(x)). The two add up to p(x).</p>
<p>The plate checks it on our two models after <code>SELECT name FROM users WHERE</code>. The draft offers <code>id</code> too often and <code>email</code> too rarely; acceptance trims the first and resampling tops up the second. With greedy decoding the rule keeps exactly the large model's tokens.</p>`,
    math: { tex: String.raw`P(x) = \min(p, q) + (1 - \beta)\,\frac{\max(0,\, p - q)}{1 - \beta} = p(x)`, note: "p, q at x · β = Σ<sub>y</sub> min(p(y), q(y)), and Σ<sub>y</sub> max(0, p(y) − q(y)) = 1 − β" },
    fig: {
      type: "steps", caption: "Accept or resample at the position after WHERE: p from the final model, q from the step-150 draft",
      steps: [
        { html: String.raw`Final model p: <code>id</code> 0.3460, <code>email</code> 0.3448, <code>age</code> 0.3072, <code>name</code> 0.0012. Draft q: 0.4582, 0.1833, 0.1363, 0.1623.` },
        { html: String.raw`Overlap \(\beta = \sum \min(p, q) = 0.6676\): a draft is accepted 66.76% of the time; a rejection happens with probability 0.3324.` },
        { html: String.raw`<code>id</code>: drafted with 0.4582 and kept with probability \(0.3460 / 0.4582\), so it is emitted with \(0.4582 \times 0.3460 / 0.4582 = 0.3460\) = p(<code>id</code>). Never resampled, since p \(\lt\) q.`, note: "acceptance trims what the draft overrates" },
        { html: String.raw`<code>email</code>: drafted with 0.1833 and always kept. After a rejection (0.3324) it is resampled with probability \((0.3448 - 0.1833) / 0.3324\), which adds \(0.3448 - 0.1833 = 0.1615\). Total \(0.1833 + 0.1615 = 0.3448\) = p(<code>email</code>).`, note: "resampling adds what it underrates" },
        { html: String.raw`Over all 113 tokens the emitted distribution matches p to within \(3 \times 10^{-16}\), floating-point rounding.` },
      ],
    },
    check: {
      q: "Each drafted token is accepted with probability α = 0.8 and the draft proposes k = 4 tokens per pass. How many tokens does one pass of the large model yield on average?",
      options: ["2.36", "3.36", "4", "5"],
      answer: 1,
      why: "(1 − 0.8⁵)/(1 − 0.8) = 0.672/0.2 = 3.36. The 2.36 counts only accepted drafts (0.8 + 0.64 + 0.512 + 0.41) and forgets the token the large model adds itself; 4 and 5 assume every draft is accepted.",
    },
    src: ["Leviathan, Kalman and Matias, Fast Inference from Transformers via Speculative Decoding, ICML 2023 (section 2.3, appendix A.1)"],
  });

  Deck.slide({
    id: "serve-medusa",
    title: "Extra heads or a one-layer draft can replace the separate draft model",
    body: String.raw`<p>A separate draft model has to share the target's tokenizer, works best when trained or distilled to match it, and takes memory and time of its own. Two designs draft from the target model's own last hidden state instead.</p>
<p><b>Medusa</b> adds several decoding heads on the last hidden state: head k predicts the token k + 1 positions ahead, in the same pass as the normal output head. The top candidates of the heads form a tree of continuations, which the target verifies in one pass with a tree-shaped attention mask. Its typical acceptance scheme keeps a candidate the target finds probable enough, which does not reproduce the target's distribution exactly; standard rejection sampling is the exact alternative.</p>
<p><b>EAGLE</b> drafts at the feature level: one decoder layer, using the target's embedding and output head, predicts the next hidden state from the current hidden states and the tokens shifted one step ahead. It keeps the exact accept-or-resample rule.</p>`,
    fig: {
      type: "compare", caption: "Two ways to draft without a second model, as reported by their authors",
      left: { title: "Medusa (Cai et al., 2024)", items: ["Extra heads on the last hidden state, one per future position", "Candidates arranged as a tree, verified with tree attention", "Typical acceptance: outputs close to the target's distribution, without an exact match", "Over 2.2× faster with the base model frozen (Medusa-1); 2.3 to 2.8× trained jointly (Medusa-2)"] },
      right: { title: "EAGLE (Li et al., 2024)", items: ["One decoder layer plus a projection, sharing the target's embedding and output head", "Predicts the next hidden state from hidden states and tokens one step ahead", "Exact acceptance: the target's distribution is preserved", "2.7 to 3.5× lower latency on LLaMA2-Chat 70B, throughput doubled"] },
    },
    take: "Drafting from the target's hidden state removes the second model; the acceptance rule decides whether outputs stay exactly the target's.",
    src: ["Cai et al., Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads, ICML 2024", "Li et al., EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty, ICML 2024"],
  });

  Deck.slide({
    id: "serve-quant",
    title: "Quantization stores weights in a few bits plus a shared scale",
    example: "tokens ids",
    math: String.raw`s = \frac{\max |w|}{2^{b-1} - 1}, \qquad q = \operatorname{round}(w / s), \qquad w \approx s \cdot q`,
    body: String.raw`<p>At small batch a decode step's time is about its bytes of weights ÷ bandwidth, so fewer bits per weight speed up every step. Symmetric integer quantization gives each group of weights a scale s mapping its largest magnitude to the largest b-bit integer, and stores every weight as the nearest multiple of s. Weight-only kernels multiply the integers back by s on the fly and compute in BF16.</p>
<p>Llama 3.1 70B: 141.1 GB in BF16, 70.6 GB in 8 bits, 35.3 GB in 4 bits, plus the scales. On 8 H100s at batch 1 the decode bound, counting the integers alone, drops from 5.19 ms to 2.59 and 1.30 ms. With 64 requests of 8,192 tokens the caches are most of the bytes, and 4-bit weights move the bound only from 11.60 to 7.71 ms.</p>
<p>The plate quantizes our whole model and scores all 270 queries exactly. With one scale per output channel, 8, 6 and 5 bits give losses 0.6942, 0.6946 and 0.6953 (unquantized 0.6942). At 4 bits the model breaks: after <code>SELECT name FROM</code> it predicts <code>FROM</code> again with p = 1.000. One scale per matrix fails already at 5 bits (loss 1.2244). Large models use small groups, such as AWQ's 128 weights per scale.</p>`,
    fig: { type: "tinylm", widget: "quant", caption: "Our model quantized live and evaluated on all 270 queries. The size tile counts all 9,968 parameters at the chosen width and leaves out the scales." },
    take: "At small batch, fewer bits per weight cut decode time in proportion. Measure the loss: a model can be fine at 5 bits and broken at 4.",
    terms: [{ term: "Quantization (weights)", def: "Storing weights as low-bit integers with a scale per group, w ≈ s · round(w / s), to cut memory and the bytes each decode step reads." }],
    src: ["Llama 3.1 70B config.json (Meta, 2024)", "Lin et al., AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration, MLSys 2024"],
  });

  Deck.slide({
    id: "serve-gptq",
    title: "GPTQ and AWQ choose the rounding by what it does to a layer's output",
    body: String.raw`<p>Rounding each weight to its nearest level ignores how the errors add up in the layer's output. <b>GPTQ</b> (Frantar et al.) quantizes each layer to minimize ‖WX − ŴX‖² over calibration inputs X. It rounds one input dimension at a time and, after each, updates the weights not yet rounded so they absorb the error, using the inverse of H = 2XXᵀ. This runs once, offline: 175B-parameter models took about four GPU hours, at 3 or 4 bits per weight.</p>
<p><b>AWQ</b> (Lin et al.) keeps plain rounding and changes what gets rounded. It finds that about 1% of the weights matter most, those in input channels with large activations, and multiplies each input channel by s = s<sub>X</sub><sup>α</sup> (s<sub>X</sub>: the channel's mean activation magnitude) before quantizing, dividing the activations by the same s. α in [0, 1] comes from a grid search; there is no backpropagation, and groups hold 128 weights.</p>
<p>On our model, the 12 attention and MLP matrices at 4 bits per output channel: round to nearest gives loss 1.2545 and p(users) = 0.000. GPTQ, calibrated on 32 queries sampled with <code>mulberry32(5)</code> and evaluated on all 270, gives 0.6948 and 1.000.</p>`,
    fig: { type: "serve-gptq", run: gqRun, caption: "The 12 attention and MLP matrices of our model quantized per output channel; embeddings and output layer stay FP32. GPTQ is calibrated on 32 sampled queries; losses are over all 270." },
    take: "GPTQ fixes rounding errors with the remaining weights; AWQ rescales the channels that matter before rounding. Both need only a little calibration data.",
    terms: [
      { term: "GPTQ", def: "Post-training quantization that rounds a layer's weights one input dimension at a time and updates the rest to minimize the layer's output error, using the inverse of H = 2XXᵀ." },
      { term: "AWQ", def: "Activation-aware weight quantization: scales input channels with large activations up before rounding (s = s_X^α), protecting the weights that matter most." },
    ],
    src: ["Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, ICLR 2023", "Lin et al., AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration, MLSys 2024"],
  });

  Deck.slide({
    id: "serve-fp8",
    title: "FP8 on Hopper halves the bytes and doubles the math rate",
    body: String.raw`<p>Hopper's tensor cores multiply FP8 matrices at 1,979 TFLOP/s dense, twice their BF16 rate (Part 7 covers the E4M3 and E5M2 formats). Serving with FP8 weights and FP8 activations (W8A8) therefore halves the bytes a decode step reads and doubles the peak math of a prefill: decode becomes faster because it is memory-bound, prefill because it is compute-bound.</p>
<p>Meta's recipe for Llama 3 405B (Llama 3 paper, section 6.2) quantizes most matrix multiplications of the feed-forward layers, which take about half of inference time, with one scale per row of the weight and activation matrices; the activation scales are computed on the fly and capped at 1,200. The self-attention weights and the first and last layers stay in BF16. It reports up to 50% higher prefill throughput than BF16 inference on two machines. Counting from the configuration, those weights take 487.2 GB, against 811.7 GB in BF16: they fit on one 8-GPU H100 machine with room for the cache.</p>`,
    fig: {
      type: "table", caption: "BF16 against FP8 (W8A8) on H100s, bounds from bytes and peak FLOP/s", align: ["l", "r", "r"],
      cols: ["", "BF16", "FP8"],
      rows: [
        ["Dense peak per H100, TFLOP/s", "989", "1,979"],
        ["Llama 3.1 70B weights, GB", "141.1", "70.6"],
        ["70B decode step bound on 8 GPUs, ms", "5.19", "2.59"],
        ["70B prefill of 2,048 tokens, linear layers, ms", "36.0", "18.0"],
        ["Llama 3.1 405B weights, GB", "811.7", "487.2"],
      ],
      note: "The 405B FP8 figure is Meta's mix: attention, the first and last layers' MLPs, embeddings and output layer stay in BF16, as in the released configuration.",
    },
    take: "FP8 helps both phases: half the bytes for decode, twice the FLOP/s for prefill. Keep the sensitive layers in BF16.",
    src: ["NVIDIA H100 Tensor Core GPU architecture whitepaper (FP8 and BF16 dense, H100 SXM)", "Dubey et al., The Llama 3 Herd of Models, 2024 (section 6.2)", "Llama 3.1 405B and 405B-FP8 config.json (Meta, 2024)"],
  });

  Deck.slide({
    id: "serve-kv-quant",
    title: "An FP8 cache doubles the requests that fit and halves their reads",
    body: String.raw`<p>The cache can be quantized independently of the weights. vLLM stores it in FP8 with <code>kv_cache_dtype</code> set to <code>fp8</code> (E4M3) or <code>fp8_e5m2</code>, one byte per number instead of two. Llama 3.1 70B then needs 163,840 bytes per token.</p>
<p>The gains are the ones of 8.2 and 8.1, halved. On 8 H100s with 485.7 GB for the cache, 8,192-token requests that fit go from 180 to 361. A decode step for 64 such requests reads 171.8 GB of cache in BF16 and 85.9 GB in FP8, so its bound drops from 11.60 ms to 8.39 ms: at long contexts the cache is most of what a step reads.</p>
<p>E4M3 keeps 3 mantissa bits, so a number in its normal range is stored with a relative error of at most 2⁻⁴ = 6.25%. Whether a model's attention tolerates that is measured, as for weights.</p>`,
    fig: {
      type: "calc", caption: "Llama 3.1 70B on 8 H100s, weights in BF16: the cache in BF16 or FP8",
      inputs: [
        { id: "b", label: "Cache precision", options: [{ label: "BF16", value: 2 }, { label: "FP8", value: 1 }], value: 1 },
        { id: "lc", label: "Context per request (tokens)", min: 10, max: 17, step: 1, value: 13, fmt: (v) => commas(pow2(v)) },
        { id: "lb", label: "Requests decoding", min: 0, max: 9, step: 1, value: 6, fmt: (v) => String(pow2(v)) },
      ],
      outputs: [
        { label: "bytes per token", f: (v) => kvPerToken(LLAMA["70B"], v.b), fmt: (x) => `${commas(x)} B` },
        { label: "requests that fit", f: (v) => Math.floor((usable(8) - totalParams(LLAMA["70B"]) * 2) / (kvPerToken(LLAMA["70B"], v.b) * pow2(v.lc))), fmt: "int" },
        { label: "cache read per step", f: (v) => kvPerToken(LLAMA["70B"], v.b) * pow2(v.lc) * pow2(v.lb), fmt: (x) => `${(x / 1e9).toFixed(1)} GB` },
        { label: "decode step bound", f: (v) => decodeStep(LLAMA["70B"], pow2(v.lb), pow2(v.lc), { kvBytes: v.b }).t, fmt: (x) => ms(x) },
      ],
      note: (v) => { const r = decodeStep(LLAMA["70B"], pow2(v.lb), pow2(v.lc), { kvBytes: v.b }); return r.fits ? `Memory in use: ${(r.mem / 1e9).toFixed(1)} GB of the ${(r.cap / 1e9).toFixed(1)} GB vLLM uses (0.92 × 8 × 79.32 GiB); ${((r.cap - totalParams(LLAMA["70B"]) * 2) / 1e9).toFixed(1)} GB of it for the cache.` : `<b>${pow2(v.lb)} requests of ${commas(pow2(v.lc))} tokens need ${(r.mem / 1e9).toFixed(0)} GB with the weights: more than the ${(r.cap / 1e9).toFixed(1)} GB vLLM uses.</b>`; },
    },
    take: "At long contexts the cache dominates both memory and the bytes read per step; FP8 halves both.",
    src: ["vLLM engine arguments documentation (kv_cache_dtype), v0.32", "Llama 3.1 70B config.json (Meta, 2024)"],
  });

  Deck.slide({
    id: "serve-flashdecoding",
    title: "FlashDecoding splits a long cache across thread blocks",
    example: "tokens ids focus:7",
    body: String.raw`<p>FlashAttention (Part 6) gives each thread block one head of one request and a block of query positions. In decode a request has one query position, so the grid is batch × heads: Llama 3.1 70B on 8 H100s at batch 1 has 8 heads per GPU, 8 thread blocks for 132 SMs.</p>
<p><b>FlashDecoding</b> also splits the keys and values into chunks processed in parallel. Each chunk returns a partial output and two scalars: its largest score m and its sum of exponentials ℓ. A final reduction rescales each partial output by e<sup>m<sub>i</sub> − M</sup>, with M the overall maximum, and adds them. The result is exact: the online softmax of Part 6, applied across chunks.</p>
<p>The plate does it for our query's last position. In layer 1, head 2, the first half of the keys has a largest score of 4.249 and the second half 1.955, so the second half's partial output is rescaled by e<sup>1.955 − 4.249</sup> = 0.101; with their sums of exponentials (1.027 and 1.761) the halves contribute 85% and 15% of the result. Its authors report up to 8 times faster generation for very long sequences (CodeLlama-34B, up to 64K tokens) and an attention kernel up to 50 times faster.</p>`,
    fig: { type: "serve-flash", caption: "Attention of our query's last position computed in chunks and merged, against one pass over all keys" },
    take: "Split the cache as well as the batch: per-chunk maxima and sums let partial attention results merge exactly.",
    terms: [{ term: "FlashDecoding", def: "Decode-time attention that splits the KV cache into chunks processed in parallel and merges the partial results with their log-sum-exp scalars." }],
    src: ["Dao, Haziza, Massa and Sizov, Flash-Decoding for long-context inference, Stanford CRFM / PyTorch blog, 2023"],
  });

  // vLLM v0.32 default CUDA-graph capture sizes for decode batches
  const CAPTURE = [1, 2, 4].concat(Array.from({ length: 31 }, (_, i) => 8 * (i + 1)), Array.from({ length: 17 }, (_, i) => 256 + 16 * i));
  const graphFor = (b) => CAPTURE.find((c) => c >= b);

  Deck.slide({
    id: "serve-cuda-graphs",
    title: "Decode steps replay CUDA graphs captured for fixed batch sizes",
    body: String.raw`<p>Part 6 showed what a CUDA graph saves: one CPU call replays a recorded sequence of kernels, removing most of the per-kernel launch overhead. In serving, that matters most for decode, because tensor parallelism shrinks the GPU work per layer. Llama 3.1 70B on 8 H100s reads 213.9 MB of weights per GPU per layer, 63.9 µs of HBM time; Llama 3.1 8B on 8 GPUs, 54.5 MB and 16.3 µs. A layer runs several kernels, and when each lasts only microseconds the CPU's launch rate sets the pace while the GPU waits.</p>
<p>A graph replays the shapes and memory addresses it was recorded with, so an engine captures one graph per batch size, and a step must use a graph captured for at least its batch, with the extra slots left empty. vLLM (v0.32) by default captures full graphs for decode batches of 1, 2 and 4, then every multiple of 8 up to 256 and of 16 up to 512, and piecewise graphs for prefill and mixed batches. Larger batches run without a graph, and <code>enforce_eager</code> turns graphs off.</p>
<p>With these sizes a batch of 13 runs in the graph for 16, leaving 3 slots empty, and a batch of 260 in the graph for 272.</p>`,
    fig: {
      type: "calc", caption: "Which captured graph runs a decode batch, with vLLM's default capture sizes (v0.32)",
      inputs: [{ id: "b", label: "Requests in the decode batch", min: 1, max: 600, step: 1, value: 13, fmt: "int" }],
      outputs: [
        { label: "graph used", f: (v) => graphFor(v.b), fmt: (x) => (x ? `batch ${commas(x)}` : "none: runs eagerly") },
        { label: "empty slots", f: (v) => (graphFor(v.b) ? graphFor(v.b) - v.b : 0), fmt: "int" },
        { label: "share of slots empty", f: (v) => (graphFor(v.b) ? (graphFor(v.b) - v.b) / graphFor(v.b) : 0), fmt: "pct" },
      ],
      note: "Capture sizes: 1, 2, 4, then 8 to 248 in steps of 8, then 256 to 512 in steps of 16.",
    },
    check: {
      q: "Llama 3.1 70B decodes on 8 H100s at batch 1 with 2,048 tokens of context. Which change lowers the step's bandwidth bound the most?",
      options: ["An FP8 KV cache", "4-bit weights", "FlashDecoding", "Capturing the step in a CUDA graph"],
      answer: 1,
      why: "The bound is 139.0 GB of weight reads plus 0.67 GB of cache, at 26.8 TB/s. 4-bit weights cut the 139.0 GB to about a quarter. An FP8 cache saves 0.34 GB of 139.7, and FlashDecoding speeds up attention over a long cache; at 2,048 tokens both are small. A CUDA graph moves no bytes: it can bring a step closer to the bound and cannot lower it.",
    },
    src: ["vLLM compilation configuration (cudagraph_mode, capture sizes) and engine arguments (enforce_eager), v0.32", "Llama 3.1 8B and 70B config.json (Meta, 2024)"],
  });

  // ------------------------------------------------------------------ 8.5
  Deck.chapter({ title: "Parallelism for inference" });

  Deck.slide({
    id: "serve-tp",
    title: "Tensor parallelism splits each step, at two all-reduces per layer",
    body: String.raw`<p>Tensor parallelism (Part 7) splits every weight matrix across N GPUs: attention by heads, the MLP's first matrices by columns and its last by rows. Each GPU reads 1/N of the weights per step, so the bandwidth bound of a decode step falls by N. For Llama 3.1 70B: 141.1 GB of weights do not fit one 80 GB H100; on 2 GPUs a step takes at least 20.7 ms, on 4 at least 10.4 ms, on 8 at least 5.19 ms.</p>
<p>The price is communication on the critical path. After the attention output projection and after the MLP, each GPU holds a partial sum, and an all-reduce adds them before the next layer can start: 2 per layer, 160 per decode step. Each moves the batch's hidden vectors, 16,384 bytes per request, and a ring all-reduce sends 2(N − 1)/N of that per GPU. At batch 1 over NVLink (450 GB/s each way) that takes 64 ns; the fixed latency α of each call, which depends on the library and is measured with nccl-tests, dominates. If α were 10 µs, the 160 calls would add 1.6 ms to the 5.19 ms step.</p>`,
    fig: {
      type: "calc", caption: "Llama 3.1 70B in BF16, 2,048 tokens of context per request. α is an input: measure it on your system.",
      inputs: [
        { id: "n", label: "GPUs (tensor parallel)", options: [{ label: "1", value: 1 }, { label: "2", value: 2 }, { label: "4", value: 4 }, { label: "8", value: 8 }], value: 8 },
        { id: "lb", label: "Requests in the batch", min: 0, max: 8, step: 1, value: 0, fmt: (v) => String(pow2(v)) },
        { id: "a", label: "All-reduce latency, microseconds", min: 1, max: 30, step: 1, value: 10, fmt: "int" },
      ],
      outputs: [
        { label: "weights per GPU", f: (v) => tpStep(v).w, fmt: (x) => `${(x / 1e9).toFixed(1)} GB` },
        { label: "step bound from HBM", f: (v) => (tpStep(v).fits ? tpStep(v).t : NaN), fmt: (x) => (Number.isNaN(x) ? "does not fit" : ms(x)) },
        { label: "160 all-reduces", f: (v) => (tpStep(v).fits ? tpStep(v).ar : NaN), fmt: (x) => (Number.isNaN(x) ? "–" : ms(x)) },
        { label: "tokens/s per request", f: (v) => (tpStep(v).fits ? 1 / tpStep(v).total : NaN), fmt: (x) => (Number.isNaN(x) ? "–" : commas(x)) },
      ],
      note: (v) => { const r = tpStep(v); return r.fits ? `Each all-reduce: α + 2 × ${v.n - 1}/${v.n} × ${commas(pow2(v.lb) * 16384)} B ÷ 450 GB/s. Step = HBM bound + all-reduces (they sit on the critical path).` : `<b>${(r.w / 1e9).toFixed(1)} GB of weights and ${(r.kv / 1e9).toFixed(1)} GB of cache per GPU exceed the ${(usable(1) / 1e9).toFixed(1)} GB vLLM uses per GPU.</b>`; },
    },
    take: "Tensor parallelism trades HBM time for collective latency: worth it while 2L all-reduces cost less than the bandwidth they save.",
    src: ["Llama 3.1 70B config.json (Meta, 2024)", "NVIDIA H100 Tensor Core GPU architecture whitepaper (NVLink, H100 SXM)"],
  });

  Deck.slide({
    id: "serve-pp",
    title: "Pipeline parallelism adds memory and throughput across machines",
    body: String.raw`<p>Pipeline parallelism puts consecutive layers on different GPUs. A token passes through the stages one after another, so its time per token is the sum of the stages. In exchange the machines exchange only one set of hidden vectors per stage boundary instead of running two all-reduces per layer, which suits the slower links between them.</p>
<p>Llama 3.1 405B has 811.7 GB of BF16 weights: more than one 8-GPU H100 machine holds. The Llama 3 paper serves it on 16 GPUs in two machines, with tensor parallelism inside each machine and pipeline parallelism between them. Each GPU reads about 50.5 GB per token: 14.99 ms for the first stage and 15.14 ms for the second, which also holds the output layer. That is 30.13 ms through both, at most 33 tokens per second for one request.</p>
<p>With one batch in flight, each machine waits while the other computes. Inference has no backward pass, so nothing forces a pipeline flush: the paper runs two micro-batches at once, which keeps both machines busy and raised throughput at the same local batch size, at the cost of extra synchronization latency.</p>`,
    fig: {
      type: "timeline", caption: "Llama 3.1 405B in BF16 on two 8-GPU machines (M1, M2): one micro-batch in flight against two (A and B)",
      rows: ["M1, 1 batch", "M2, 1 batch", "M1, 2 batches", "M2, 2 batches"], unit: "ms",
      total: 91,
      items: (function () {
        const m = LLAMA["405B"], layer = (totalParams(m) - 2 * m.V * m.d - m.d) / m.L;
        const s1 = (63 * layer * 2) / 8 / H100.bw * 1e3, s2 = ((63 * layer + m.V * m.d + m.d) * 2) / 8 / H100.bw * 1e3, it = [], end = 91;
        const add = (row, start, len, label, kind, tip) => { if (start + len <= end + 1e-9) it.push({ row, start, len, label, kind, tip }); };
        for (let k = 0, t = 0; k < 4; k++, t += s1 + s2) {
          add(0, t, s1, `token ${k + 1}`, "s1", `layers 1–63: ${s1.toFixed(2)} ms`);
          add(1, t + s1, s2, `token ${k + 1}`, "s1", `layers 64–126 and output layer: ${s2.toFixed(2)} ms`);
        }
        let f1 = 0, f2 = 0;
        const done = { A: 0, B: 0 };
        for (let k = 0; k < 8; k++) {
          const mb = k % 2 ? "B" : "A", tok = Math.floor(k / 2) + 1;
          const a = Math.max(f1, done[mb]); f1 = a + s1;
          const b = Math.max(f2, f1); f2 = b + s2; done[mb] = f2;
          add(2, a, s1, `${mb}${tok}`, k % 2 ? "s2" : "s1", `machine 1, micro-batch ${mb}, token ${tok}`);
          add(3, b, s2, `${mb}${tok}`, k % 2 ? "s2" : "s1", `machine 2, micro-batch ${mb}, token ${tok}`);
        }
        return it;
      })(),
      note: "Each box is one machine's share of a decode step, its weights read from HBM at 3.35 TB/s per GPU: 14.99 ms for layers 1–63, 15.14 ms for layers 64–126 and the output layer. Transfers between machines are left out.",
    },
    take: "Pipeline stages add memory and throughput; per-token latency stays the sum of the stages.",
    check: {
      q: "Why did Meta serve Llama 3.1 405B in BF16 with tensor parallelism inside each 8-GPU machine and pipeline parallelism between the two machines, instead of tensor parallelism across all 16 GPUs?",
      options: ["Pipeline parallelism shortens the time per token", "Tensor parallelism needs two all-reduces per layer, 252 per decode step, which the slower links between machines would delay", "Tensor parallelism cannot split attention heads across machines", "Pipeline parallelism needs no memory for the KV cache"],
      answer: 1,
      why: "With 126 layers, tensor parallelism puts 252 all-reduces on the critical path of every decode step. NVLink inside a machine makes them cheap; between machines bandwidth is lower and latency higher. Pipeline parallelism crosses the machine boundary once per step, with one hidden vector per request.",
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (section 6.1)", "Llama 3.1 405B config.json (Meta, 2024)"],
  });

  Deck.slide({
    id: "serve-ep",
    title: "DeepSeek-V3 serves its experts one per GPU, across 320 GPUs",
    body: String.raw`<p>A mixture-of-experts layer (Part 5) sends each token to a few experts, and expert parallelism (Part 7) puts different experts on different GPUs. DeepSeek-V3 has 671B parameters, 37B active per token: each MoE layer has 256 routed experts, a token uses 8 of them plus 1 shared expert. On average an expert sees 8/256 = 1/32 of the tokens in a step, so an expert's weights are read for few tokens unless the batch across the deployment is very large.</p>
<p>Its report deploys prefill and decode separately (8.3). Prefill runs on units of 4 nodes, 32 GPUs: attention with 4-way tensor parallelism plus sequence parallelism and 8-way data parallelism, MoE layers with 32-way expert parallelism. Decode runs on units of 40 nodes, 320 GPUs: attention 4-way tensor and 80-way data parallel, MoE layers 320-way expert parallel, one expert per GPU, with 64 GPUs holding redundant copies of busy experts and the shared expert. Tokens reach their experts through all-to-all over InfiniBand.</p>`,
    fig: {
      type: "table", caption: "DeepSeek-V3's minimum deployment units, as published (H800 GPUs, 8 per node)",
      cols: ["", "Prefill", "Decode"],
      rows: [
        ["GPUs per unit", "32 (4 nodes)", "320 (40 nodes)"],
        ["Attention", "TP4 + SP, DP8", "TP4 + SP, DP80"],
        ["MoE layers", "EP32", "EP320"],
        ["Experts per GPU", "8 routed + 1 redundant", "1"],
        ["Load balancing", "32 redundant experts, from load statistics", "64 GPUs for redundant and shared experts"],
      ],
      note: "TP: tensor parallel, SP: sequence parallel, DP: data parallel, EP: expert parallel. Redundant experts are chosen from serving statistics and updated periodically.",
    },
    take: "Expert parallelism gives each expert a GPU and gathers its tokens from the whole deployment, so each expert's weights serve a large batch.",
    src: ["DeepSeek-AI, DeepSeek-V3 Technical Report, 2024 (sections 3.4 and 4.2)"],
  });

  Deck.slide({
    id: "serve-replica",
    title: "Size a replica from memory first, then from latency",
    body: String.raw`<p>A <b>replica</b> is the smallest group of GPUs that holds a copy of the model and serves requests alone. Its size starts from two checks.</p>
<p>Memory: weights ÷ GPUs must leave room for the cache of the batch you want. Llama 3.1 70B in BF16 needs 2 H100s or more; on 2, with 92% of the reported 79.32 GiB usable, 7.80 GB per GPU remain for the cache: 23 requests of 2,048 tokens. On 4 GPUs, 43.08 GB per GPU: 256 such requests, minus activations.</p>
<p>Latency: the decode step bound is weight bytes per GPU ÷ HBM bandwidth (plus cache reads and all-reduces), so the target TPOT sets a minimum GPU count. Llama 3.1 8B fits on one GPU at 4.48 ms per step. 70B in FP8 fits even on one GPU, with room for 11 such requests, at 20.75 ms per step; on 2 GPUs the bound is 10.37 ms, on 4, 5.19 ms.</p>
<p>A wider replica also has more cache room per GPU, since each holds a smaller share of the weights. For 70B at 2,048 tokens, 4 GPUs fit 256 requests and 8 fit 723; at those batches the bound gives 2,759 and 3,880 tokens/s per GPU. The all-reduces (8.5) grow with the GPU count.</p>`,
    fig: {
      type: "calc", caption: "One replica on H100s: does it fit, and how fast can it decode? Cache in BF16; 92% of the 79.32 GiB CUDA reports per GPU usable. FP8: every weight for 8B and 70B; Meta's released mix for 405B (487.2 GB).",
      inputs: [
        { id: "m", label: "Model", options: [{ label: "8B", value: "8B" }, { label: "70B", value: "70B" }, { label: "405B", value: "405B" }], value: "70B" },
        { id: "wb", label: "Weights", options: [{ label: "BF16", value: 2 }, { label: "FP8", value: 1 }], value: 2 },
        { id: "n", label: "GPUs", options: [1, 2, 4, 8].map((x) => ({ label: String(x), value: x })), value: 4 },
        { id: "lc", label: "Context (tokens)", min: 10, max: 17, step: 1, value: 11, fmt: (v) => commas(pow2(v)) },
        { id: "lb", label: "Requests decoding", min: 0, max: 9, step: 1, value: 5, fmt: (v) => String(pow2(v)) },
      ],
      outputs: [
        { label: "weights per GPU", f: (v) => replica(v).w, fmt: (x) => `${(x / 1e9).toFixed(1)} GB` },
        { label: "requests that fit", f: (v) => replica(v).maxB, fmt: "int" },
        { label: "decode step bound", f: (v) => (replica(v).fits ? replica(v).t : NaN), fmt: (x) => (Number.isNaN(x) ? "does not fit" : ms(x)) },
        { label: "tokens/s per GPU", f: (v) => (replica(v).fits ? pow2(v.lb) / replica(v).t / v.n : NaN), fmt: (x) => (Number.isNaN(x) ? "–" : commas(x)) },
      ],
      note: "Step bound from bytes and peak FLOP/s only; all-reduces (8.5) and activations come on top.",
    },
    take: "Start from the fewest GPUs that hold weights plus cache and meet the TPOT target, compare tokens per GPU-second at wider sizes, then add replicas for traffic.",
    terms: [{ term: "Replica", def: "One complete copy of a model on a group of GPUs that serves requests independently of the other copies." }],
    src: ["Llama 3.1 8B, 70B and 405B config.json (Meta, 2024)", "Llama 3.1 405B-FP8 config.json (Meta, 2024)", "NVIDIA H100 Tensor Core GPU datasheet (H100 SXM)"],
  });

  Deck.slide({
    id: "serve-lb",
    title: "Balance replicas by the tokens they have outstanding",
    body: String.raw`<p>Replicas share no state, so more traffic means more replicas behind a router, whose job is to keep them equally loaded. Requests differ in cost by orders of magnitude: answers of 32 and 1,024 tokens occupy a replica for 32 and 1,024 decode steps.</p>
<p>Round robin gives every replica the same number of requests. In the plate, 32 requests of 32 to 1,024 output tokens arrive together at 4 replicas. Round robin leaves the busiest replica 4,352 tokens to generate, 40% above an even share of 3,104 and 3.6 times the least busy one; sending each request to the replica with the fewest tokens outstanding cuts the busiest to 3,584, 15% above. Over 40 random draws, the medians are 40% and 14%.</p>
<p>With continuous batching the imbalance shows up as a larger batch and cache on the busy replica: at 2,048 tokens of context a 70B step for 128 requests takes 8.39 ms against 6.79 ms for 64, so every one of its requests streams 24% slower, and it runs out of blocks first. A router sees each replica's queue, batch and cache use, and can also prefer a replica that holds a request's prefix (8.6).</p>`,
    fig: { type: "serve-lb", caption: "32 requests (seeded) of 32 to 1,024 output tokens, assigned in arrival order to 4 replicas. Each bar is the tokens a replica must generate." },
    check: {
      q: "Round robin leaves one 70B replica (8 H100s) with 128 running requests at 2,048 tokens of context, while another has 64. With continuous batching, what do the busy replica's users see?",
      options: ["Nothing: continuous batching evens out the load between replicas", "Each decode step takes 8.39 ms instead of 6.79 ms, so their tokens arrive 24% slower, and the replica runs out of cache blocks first", "Their requests wait in a queue until the other replica is free", "The same speed, because decode is memory-bound"],
      answer: 1,
      why: "Continuous batching works inside one replica, so the busy one runs all 128 requests together. Each step reads 139.0 GB of weights plus 128 caches of 0.67 GB, 224.9 GB in all, 8.39 ms at 26.8 TB/s; with 64 requests it reads 182.0 GB in 6.79 ms. Memory-bound means the step grows with those bytes.",
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (section 6.1)"],
  });

  // ------------------------------------------------------------------ 8.6
  Deck.chapter({ title: "Serving systems" });

  Deck.slide({
    id: "serve-path",
    title: "A request passes through eight stages from text to streamed tokens",
    example: "tokens ids prefix:3",
    body: String.raw`<p>Follow our prompt through a server. The text is tokenized into ids. The scheduler admits the request when it has room for its tokens in the KV cache. A prefill pass processes the prompt and the sampler draws the first token from the last position's distribution, after applying temperature, top-p or a grammar mask. The new id is turned back into text and sent to the client at once; then decode steps repeat, batched with other requests, until a stop condition, and the request's blocks are freed.</p>
<p>Detokenization is incremental: a server converts each new token in the context of the previous ones, because one character can span several tokens (a multi-byte UTF-8 character split across byte-level tokens). Streaming is what makes the time to first token visible to the user: OpenAI-compatible servers send each piece as a server-sent event as soon as it exists.</p>
<p>Tokenizing, scheduling, detokenizing and streaming run on the CPU. The forward passes and the sampling run on the GPU; for a grammar, vLLM's scheduler builds the token bitmask on the CPU and the worker applies it to the logits on the GPU. An engine keeps the CPU stages off the GPU's critical path, or a decode step of 5 ms waits on Python.</p>`,
    fig: {
      type: "table", caption: "Our prompt SELECT name FROM through a server running our model", dense: true,
      cols: ["Stage", "What happens", "Our request"],
      rows: [
        ["1 Tokenize", "text to token ids", "<code>SELECT name FROM</code> → 43, 93, 47"],
        ["2 Schedule", "wait for KV blocks", "3 tokens: 1 block of 16"],
        ["3 Prefill", "one pass over the prompt", "keys and values of 3 tokens"],
        ["4 Sample", "draw from the last distribution", "<code>users</code> (id 63), p = 1.000"],
        ["5 Detokenize", "ids to text, incrementally", "“ users”"],
        ["6 Stream", "send the piece now", "one server-sent event"],
        ["7 Decode", "one pass per token, in a batch", "<code>WHERE</code>, <code>id</code>, <code>=</code>, <code>7</code>"],
        ["8 Finish", "stop string or length limit; free blocks", "<code>;</code> matches the stop string"],
      ],
    },
    take: "Tokenize, schedule, prefill, sample, detokenize, stream, decode, finish: forward passes and sampling on the GPU, the rest on the CPU.",
    src: ["OpenAI API documentation, Streaming responses (data-only server-sent events)", "Kwon et al., PagedAttention, SOSP 2023 (block size 16)", "vLLM source, v1 structured output (apply_grammar_bitmask)"],
  });

  Deck.slide({
    id: "serve-vllm",
    title: "vLLM: a scheduler with a KV cache manager, driving GPU workers",
    body: String.raw`<p>vLLM was built around PagedAttention. Its paper describes a centralized scheduler that owns the KV cache manager, which keeps every request's block table, and one worker per GPU holding a shard of the model and that GPU's KV blocks. Each step the scheduler sends the workers the token ids and block tables to run.</p>
<p>The current architecture (V1, introduced in 2025) splits this across processes. An API server process handles HTTP, tokenization, detokenization and streaming, so this CPU work overlaps the GPU's. An EngineCore process runs a busy loop. Its scheduler decides how many tokens of each request to process in the next step, treating prompt and output tokens alike, as a map from request to token count: that one rule covers prefill, chunked prefill and decode. The KV cache manager allocates blocks and finds cached prefixes; an executor runs the step on the GPU workers, and the sampled tokens flow back.</p>
<p>Prefix caching and chunked prefill are on by default, and decode steps replay CUDA graphs.</p>`,
    fig: {
      type: "diagram", caption: "vLLM V1, as described in its paper and its V1 announcement", w: 15, h: 10.2, unit: 40,
      groups: [{ x: 0.2, y: 2.5, w: 14.6, h: 4.6, label: "EngineCore process: one loop per step" }],
      nodes: [
        { id: "cl", x: 0.6, y: 0.5, w: 3.2, h: 1.4, label: "Client", sub: "HTTP" },
        { id: "api", x: 5.4, y: 0.5, w: 5.2, h: 1.4, label: "API server process", sub: "tokenize · detokenize · stream", kind: "blue" },
        { id: "kvm", x: 0.8, y: 3.2, w: 3.4, h: 1.6, label: "KV cache\nmanager", sub: "block tables" },
        { id: "sch", x: 5.4, y: 3.2, w: 5.2, h: 1.6, label: "Scheduler", sub: "tokens per request this step", kind: "hi" },
        { id: "ex", x: 5.4, y: 5.5, w: 5.2, h: 1.3, label: "Executor" },
        { id: "gpus", x: 1.2, y: 8.2, w: 13.6, h: 1.6, label: "8 GPU workers", sub: "each: a tensor-parallel shard of the model and its KV blocks", kind: "dark" },
      ],
      edges: [
        { from: "cl", to: "api", both: true, label: "text" },
        { from: "api", to: "sch", both: true, label: "token ids · new tokens", dx: 76 },
        { from: "sch", to: "kvm", both: true, label: "blocks" },
        { from: "sch", to: "ex", label: "step plan", dx: 40 },
        { from: "ex", to: "gpus", both: true, label: "inputs, block tables · sampled tokens", dx: 124 },
      ],
      note: "Each step the executor sends the batch's inputs and block tables to all 8 workers and gets the sampled tokens back; the API server detokenizes them and streams them out.",
    },
    take: "One scheduler decides every step; workers only execute. Separating CPU-side request handling from the step loop keeps the GPUs fed.",
    src: ["Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023 (section 4)", "vLLM team, vLLM V1: A Major Upgrade to vLLM's Core Architecture, January 2025", "vLLM optimization documentation, v0.32"],
  });

  Deck.slide({
    id: "serve-grammar",
    title: "Constrained decoding masks every token the grammar forbids",
    example: "tokens",
    body: String.raw`<p>A server can guarantee the shape of an output by changing one step of sampling. Before each token is drawn, it sets the probability of every token that cannot continue a valid output to zero and renormalizes the rest. This is <b>constrained decoding</b>, and it needs no extra forward pass.</p>
<p>For our grammar the allowed set follows the query: after <code>SELECT</code>, the 9 column tokens; then <code>FROM</code>; then only tables that have that column (after <code>SELECT name</code>, only <code>users</code>); <code>WHERE</code>; another column of the same table; <code>=</code>; a digit; <code>;</code>. At every step at most 9 of the 113 tokens remain.</p>
<p>At temperature 1 the model already puts almost all its probability on allowed tokens. Higher temperatures flatten its distribution, and probability flows to forbidden tokens: after <code>SELECT name FROM</code> the model gives <code>users</code> 0.851 at temperature 3 and 0.561 at temperature 4. The mask returns all of it to the allowed tokens, so here <code>users</code> gets probability 1 at any temperature.</p>`,
    note: "Engines check grammars over characters, so a word split into pieces is allowed too. Our 9-token window leaves no room for a split, so both checks allow the same tokens.",
    fig: { type: "serve-grammar", caption: "One sample drawn token by token at the chosen temperature. Struck-through rows are tokens the grammar forbids at this position." },
    take: "Mask, renormalize, sample: the output follows the grammar by construction, whatever the model's probabilities.",
    terms: [{ term: "Constrained decoding", def: "Sampling in which every token that cannot continue a valid output (by a grammar, regular expression or schema) gets probability zero before the draw." }],
  });

  Deck.slide({
    id: "serve-grammar-rates",
    title: "With the mask, every sample is a valid query at any temperature",
    example: "tokens",
    body: String.raw`<p>Without the mask, validity falls with temperature. Adding up the model's probability of each of the 270 valid queries gives the exact chance that an unmasked sample is valid: 0.9984 at temperature 1, 0.9045 at 2, 0.5408 at 3 and 0.1662 at 4.</p>
<p>Sampling agrees. With the seeded generator <code>mulberry32(1)</code>, 1,000 samples without the mask give 998 valid queries at temperature 1, 906 at 2 and 521 at 3. With the mask, 1,000 of 1,000 are valid at every temperature: no forbidden token can be drawn, and every allowed prefix can be completed within the model's window.</p>
<p>The failures show what flattening does: a filter column equal to the selected one (<code>SELECT name FROM users WHERE name = 4;</code>), a column of another table, a table name where a digit belongs. The mask guarantees validity and leaves the proportions to the model: at temperature 3 the masked samples are spread over the 270 queries at a total-variation distance (½ Σ |p − q| over the queries) of 0.123 from the grammar's distribution, against 0.028 at temperature 1.</p>`,
    fig: { type: "serve-grammar-rates", run: (tau, mask) => GR.run(tau, mask), exact: (tau) => GR.exact(tau), caption: "Share of valid queries against temperature: exact without the mask (curve), 1,000 seeded samples (circle), and with the mask (top line)." },
    take: "A mask turns validity from a probability into a guarantee; temperature still decides which valid outputs you get.",
  });

  Deck.slide({
    id: "serve-json",
    title: "Structured output compiles a schema into a token mask",
    body: String.raw`<p>JSON that matches a schema, or the arguments of a function call, is produced the same way. The engine compiles the schema into an automaton over characters and, at each step, allows the tokens whose characters keep the automaton in a live state.</p>
<p>Testing all 128,256 tokens of Llama 3 against the automaton at every step costs O(V) per token. Outlines (Willard and Louf) compiles a regular expression into a finite-state machine and precomputes, for each state, the set of allowed tokens, so the mask becomes a lookup, O(1) on average. Nested JSON needs a stack: XGrammar runs a pushdown automaton for context-free grammars and precomputes the answer for context-independent tokens, whose validity does not depend on the stack. For Llama 3.1 with a JSON grammar, 1,134 of the 128K tokens, under 1%, are context-dependent and must be checked at run time. XGrammar reports up to 100 times lower per-token cost on context-free grammars than earlier engines.</p>
<p>A schema constrains syntax only: the plate's schema accepts a column that the table does not have.</p>`,
    fig: {
      type: "code", lang: "text", caption: "A JSON schema for our queries, as sent with a request for structured output",
      code: `{
  "type": "object",
  "properties": {
    "column": { "enum": ["id", "name", "email", "age", "user_id",
                         "total", "status", "title", "price"] },
    "table":  { "enum": ["users", "orders", "products"] },
    "where":  { "enum": ["id", "name", "email", "age", "user_id",
                         "total", "status", "title", "price"] },
    "value":  { "type": "integer", "minimum": 1, "maximum": 9 }
  },
  "required": ["column", "table", "where", "value"],
  "additionalProperties": false
}`,
      notes: {
        2: "The output must be one JSON object. Before the first token, only tokens that can start one are allowed.",
        4: "An <code>enum</code> becomes an alternation of fixed strings: after <code>\"column\": \"</code>, only tokens that continue one of the 9 names.",
        6: "Nothing ties the table to the column: <code>{\"column\": \"title\", \"table\": \"users\", …}</code> passes, though our grammar rejects it. Expressing that needs one branch per table.",
        7: "Same nine names: the schema cannot require <code>where</code> ≠ <code>column</code> without listing the allowed pairs.",
        9: "An integer from 1 to 9 is the single character class <code>[1-9]</code>.",
        11: "<code>required</code> and <code>additionalProperties: false</code> fix which keys appear: four, no others.",
      },
    },
    take: "Grammar engines precompute which tokens each automaton state allows, so masking costs a lookup per step instead of a scan of the vocabulary.",
    terms: [{ term: "Structured output", def: "Generation constrained to a format such as a JSON schema, by masking at each step the tokens that would leave the format." }],
    src: ["Willard and Louf, Efficient Guided Generation for Large Language Models, 2023", "Dong et al., XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models, 2024"],
  });

  Deck.slide({
    id: "serve-cost",
    title: "Cost per million tokens is GPU price divided by throughput",
    body: String.raw`<p>A replica costs the same per hour whatever it serves, so the price of a token is the price of the GPUs divided by the tokens they produce. At an assumed $2.00 per H100-hour, Llama 3.1 70B on 8 GPUs costs $16.00 per hour.</p>
<p>Output tokens come from decode steps. At 64 requests of 2,048 tokens the roofline gives 9,426 tokens/s, 33.9 million per hour: $0.47 per million output tokens. At batch 1, 192 tokens/s: $23.16 per million. Batching is the biggest lever on the price.</p>
<p>Prompt tokens come from compute-bound prefill, 139.0 GFLOP each. At the peak of 8 GPUs that is 56,918 tokens/s, $0.078 per million: a prompt token costs about a sixth of an output token here. Real engines reach a fraction of both bounds; the plate scales the throughput by that fraction.</p>`,
    fig: {
      type: "calc", caption: "Cost of Llama 3.1 70B on 8 H100s. The GPU price is an assumed input.",
      inputs: [
        { id: "price", label: "Price per GPU-hour ($)", min: 1, max: 8, step: 0.25, value: 2, fmt: "f2" },
        { id: "lb", label: "Requests decoding", min: 0, max: 9, step: 1, value: 6, fmt: (v) => String(pow2(v)) },
        { id: "lc", label: "Context (tokens)", min: 10, max: 15, step: 1, value: 11, fmt: (v) => commas(pow2(v)) },
        { id: "eff", label: "Fraction of the roofline achieved", min: 0.2, max: 1, step: 0.05, value: 1, fmt: "pct" },
      ],
      outputs: [
        { label: "output tokens/s", f: (v) => (costCalc(v).fits ? costCalc(v).tps : NaN), fmt: (x) => (Number.isNaN(x) ? "does not fit" : commas(x)) },
        { label: "$ per million output tokens", f: (v) => (costCalc(v).fits ? costCalc(v).out : NaN), fmt: (x) => (Number.isNaN(x) ? "–" : `$${x.toFixed(2)}`) },
        { label: "$ per million prompt tokens", f: (v) => costCalc(v).inp, fmt: (x) => `$${x.toFixed(3)}` },
      ],
      note: (v) => `$${(v.price * 8).toFixed(2)} per hour ÷ (tokens/s × 3,600) × 10⁶. Prompt tokens at ${commas(costCalc(v).pfTps)} tokens/s (compute-bound prefill).`,
    },
    take: "$ per million tokens = $ per hour × 10⁶ ÷ (tokens/s × 3,600). Batch size moves it more than anything else.",
    src: ["Llama 3.1 70B config.json (Meta, 2024)", "NVIDIA H100 Tensor Core GPU datasheet (H100 SXM)"],
  });

  Deck.slide({
    id: "serve-slo",
    title: "The largest batch that meets the SLO sets the replica count",
    body: String.raw`<p>Plan backwards from the latency target. The TPOT objective caps the step time; the step time grows with the batch; so the SLO caps the batch, and the batch sets a replica's throughput.</p>
<p>With chunked prefill in steady state, each step decodes B requests and processes B × P/O prompt tokens, so prompts enter as fast as answers finish (P: prompt length, O: answer length). For Llama 3.1 70B on 8 H100s with 2,048-token prompts, 256-token answers and a 50 ms TPOT target, that is 8 prompt tokens per decode: the step is compute-bound, and 309 requests per step take 49.92 ms. Each replica completes 309 ÷ (256 × 49.92 ms) = 24.2 requests/s, so 50 requests/s need 3 replicas, 24 GPUs.</p>
<p>These are roofline bounds: scale by the fraction of peak you measure, and leave headroom for bursts, since queueing delay grows sharply as utilization nears 100%.</p>`,
    fig: {
      type: "calc", caption: "Replicas of Llama 3.1 70B on 8 H100s for a request rate and a TPOT target (queueing not modeled)",
      inputs: [
        { id: "rate", label: "Requests per second", min: 1, max: 200, step: 1, value: 50, fmt: "int" },
        { id: "lp", label: "Prompt tokens", min: 8, max: 14, step: 1, value: 11, fmt: (v) => commas(pow2(v)) },
        { id: "lo", label: "Answer tokens", min: 5, max: 11, step: 1, value: 8, fmt: (v) => commas(pow2(v)) },
        { id: "slo", label: "TPOT target (ms)", min: 10, max: 100, step: 1, value: 50, fmt: "int" },
        { id: "eff", label: "Fraction of the roofline achieved", min: 0.2, max: 1, step: 0.05, value: 1, fmt: "pct" },
      ],
      outputs: [
        { label: "requests per step", f: (v) => sloPlan(v).B, fmt: (x) => (x ? commas(x) : "target too tight") },
        { label: "step time", f: (v) => (sloPlan(v).B ? sloPlan(v).t : NaN), fmt: (x) => (Number.isNaN(x) ? "–" : ms(x)) },
        { label: "requests/s per replica", f: (v) => (sloPlan(v).B ? sloPlan(v).mu : NaN), fmt: (x) => (Number.isNaN(x) ? "–" : x.toFixed(1)) },
        { label: "replicas (GPUs)", f: (v) => (sloPlan(v).B ? `${sloPlan(v).reps} (${sloPlan(v).gpus})` : "–"), fmt: (x) => x },
      ],
      note: (v) => { const r = sloPlan(v); return r.B ? `Prefill time per prompt with chunking, without queueing: about ${ms(r.ttft, 1)}. Cache room bounds the batch at ${commas(Math.floor((usable(8) - totalParams(LLAMA["70B"]) * 2) / ((pow2(v.lp) + pow2(v.lo)) * kvPerToken(LLAMA["70B"]))))} requests.` : "Even one request per step exceeds the TPOT target at this efficiency."; },
    },
    take: "SLO → largest batch → requests per second per replica → replica count. Then add headroom for bursts.",
    src: ["Agrawal et al., Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve, OSDI 2024", "Llama 3.1 70B config.json (Meta, 2024)"],
  });

  Deck.slide({
    id: "serve-fleet",
    title: "A fleet routes each request to where its cache and its phase fit best",
    example: "tokens ids prefix:3",
    body: String.raw`<p>At fleet scale the decisions of this part are made per request by a router in front of many replicas. Three of them follow from earlier sheets.</p>
<ul><li><b>Prefix-aware routing.</b> Prefix caching (8.2) only helps if requests sharing a prefix reach the same replica. A router that tracks or predicts which replica holds which prefixes sends a conversation's next turn back to the replica that has its earlier turns cached.</li>
<li><b>Load.</b> Among replicas that could serve a request, prefer the one with the least outstanding work and free cache (8.5).</li>
<li><b>Phase.</b> With disaggregation (8.3), the prompt goes to a prefill pool and the KV cache moves to a decode pool. DeepSeek-V3 runs prefill on units of 32 GPUs and decode on units of 320.</li></ul>
<p>Goodput, the request rate served within the SLO (8.1), is the number this layer is judged by.</p>`,
    fig: { type: "tinylm", widget: "fleet", caption: "Our request through a disaggregated fleet" },
    take: "Routing decides cache hits, balance and phase placement; the engine inside each replica does the rest.",
    src: ["Zheng et al., SGLang, NeurIPS 2024", "DeepSeek-AI, DeepSeek-V3 Technical Report, 2024 (section 3.4)", "Zhong et al., DistServe, OSDI 2024"],
  });

  Deck.slide({
    id: "serve-timeline",
    title: "One request, end to end, on Llama 3.1 70B and 8 H100s",
    body: String.raw`<p>Put the part together for one request: a 2,048-token prompt and a 256-token answer, arriving at a replica of Llama 3.1 70B on 8 H100s that is already decoding 64 requests of 2,048 tokens. The engine uses chunked prefill with a 512-token budget per step.</p>
<p>The prompt enters in 5 chunks of up to 448 tokens beside the 64 decodes. The four 448-token steps are compute-bound (9.07 to 9.27 ms); the last, with 256 prompt tokens, is bound by its 6.81 ms of reads. The first token arrives after 43.5 ms. Then the request joins the batch: 255 decode steps of 6.81 ms each, memory-bound, reading 139.0 GB of weights and 65 caches. The answer is complete after 1.78 s.</p>
<p>Every number came from bytes and FLOPs: the KV cache size, the chunk plan, the batch's cache reads. A real engine adds CPU and communication time and reaches a fraction of peak, so measure, then compare with these bounds to see how far each stage is from its limit.</p>`,
    fig: {
      type: "table", caption: "The request's phases with their bounds (roofline: max of bytes ÷ 26.8 TB/s and FLOPs ÷ 7,912 TFLOP/s)", align: ["l", "r", "r", "l"],
      cols: ["Phase", "Steps", "Time", "Limited by"],
      rows: (function () {
        const r = requestTimeline();
        return [
          ["Prefill in chunks, beside 64 decodes", String(r.chunks), ms(r.ttft, 1), r.limits],
          ["First token (TTFT)", "", ms(r.ttft, 1), ""],
          ["Decode, batch of 65", "255", `${(r.dec).toFixed(2)} s`, "HBM reads"],
          ["Time per output token", "", `${ms(r.first)} → ${ms(r.last)}`, ""],
          ["Answer complete", String(r.chunks + 255), `${r.total.toFixed(2)} s`, ""],
        ];
      })(),
      highlight: [1, 4],
    },
    check: {
      q: "Which change lowers this request's time to first token, at the cost of the other 64 requests' time per output token?",
      options: ["Raising the token budget per step so the whole prompt goes in one step", "Storing the KV cache in FP8", "Adding a speculative draft model", "Lowering the batch to 32 requests"],
      answer: 0,
      why: "With a 2,112-token budget the prompt runs in one step of about 38 ms instead of 5 chunked steps totalling 43.5 ms, but every running request then waits about 38 ms for that step's token instead of about 9 ms. FP8 caches and drafts speed up decode steps; they do not change how the prompt is chunked.",
    },
    src: ["Llama 3.1 70B config.json (Meta, 2024)", "NVIDIA H100 Tensor Core GPU datasheet (H100 SXM)", "Agrawal et al., Sarathi-Serve, OSDI 2024"],
  });
})();
