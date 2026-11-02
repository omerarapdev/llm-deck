/* Widgets for Pass 4: serving. */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip, slider, segmented, button, player, barList, topRows, heatStrip, chart, seq } = window.UI;
  const M = window.TinyLM;
  const W = (window.Widgets = window.Widgets || {});
  const V = M.vocab, ids = M.runningIds, toks = M.running;
  const stat = (v, l) => h("div", { class: "stat" }, h("div", { class: "stat-v" }, v), h("div", { class: "stat-l" }, l));
  const SERVER = { gpus: 8, flops: 8 * 989e12, bw: 8 * 3.35e12, mem: 8 * 80e9 };
  const L70 = { N: 70e9, wBytes: 140e9, kvTok: 2 * 80 * 8 * 128 * 2 };

  // ------------------------------------------------------------ prefill
  W.prefill = {
    init(el, deck) {
      const P = 3;
      let k = 0;
      const tr = W._trace();
      const passes = el.querySelector(".pf-passes");
      const nextBtn = button("Next pass", () => { if (k < 6) { k++; draw(); } else { k = 0; draw(); } }, { kind: "primary" });
      el.querySelector(".pf-ctl").append(nextBtn, button("Reset", () => { k = 0; draw(); }));
      function draw() {
        clear(passes);
        const row = h("div", { class: "pf-row" });
        for (let i = 0; i <= Math.min(k, 5); i++) {
          const inputs = i === 0 ? toks.slice(0, P) : [toks[P + i - 1]];
          const outTok = toks[P + i], p = tr.probs[P + i - 1][ids[P + i]];
          row.appendChild(h("div", { class: "pf-pass" + (i === 0 ? " is-prefill" : "") + (i === k ? " is-new" : "") },
            h("div", { class: "pf-kind" }, i === 0 ? "prefill" : `decode ${i}`),
            h("div", { class: "pf-in" }, inputs.map((t) => chip(t))),
            h("div", { class: "pf-arrow" }, "↓"),
            h("div", { class: "pf-out" }, chip(outTok, { cls: "is-focus" }), h("span", { class: "num" }, `p ${fmt.p(p)}`))));
        }
        passes.appendChild(row);
        passes.appendChild(h("p", { class: "fig-cap" }, k >= 5
          ? "6 forward passes produced 6 tokens. The prefill read 3 tokens at once; each decode pass read only 1, reusing the keys and values computed before (the KV cache)."
          : `Pass ${k + 1} of 6. ${k === 0 ? "The prefill reads all 3 prompt tokens in one pass and outputs the first new token." : "Each decode pass feeds in only the newest token."}`));
        nextBtn.textContent = k >= 5 ? "Start over" : "Next pass";
        deck.strip.set({ prefix: Math.min(P + k + 1, 9) < 9 ? P + k + 1 : null });
      }
      // 70B calculator
      let prompt = 2000, out = 500;
      const bar = el.querySelector(".pf-bar"), res = el.querySelector(".pf-out");
      function calc() {
        const ttft = (2 * L70.N * prompt) / (SERVER.flops * 0.4);
        const tpot = (L70.wBytes + (prompt + out / 2) * L70.kvTok) / SERVER.bw;
        const total = ttft + (out - 1) * tpot;
        clear(bar);
        bar.appendChild(h("div", { class: "pf-track", role: "img", "aria-label": `prefill ${fmt.time(ttft)}, decode ${fmt.time(total - ttft)}` },
          h("span", { class: "pf-seg pre", style: { flexGrow: String(ttft) }, "data-tip": `prefill: ${fmt.time(ttft)}` }, "prefill"),
          h("span", { class: "pf-seg dec", style: { flexGrow: String(total - ttft) }, "data-tip": `decode: ${out - 1} steps × ${fmt.time(tpot)}` }, `decode × ${out - 1}`)));
        clear(res);
        res.append(stat(fmt.time(ttft), "time to first token"), stat(fmt.time(tpot), "per output token"), stat(fmt.time(total), "whole response"));
      }
      el.querySelector(".pf-calc").append(
        slider({ id: "pf-p", label: "Prompt tokens", min: 4, max: 15, step: 1, value: Math.log2(2048), format: (v) => Math.pow(2, v).toLocaleString("en-US"), onInput: (v) => { prompt = Math.pow(2, v); calc(); } }),
        slider({ id: "pf-o", label: "Output tokens", min: 1, max: 2000, step: 1, value: out, format: (v) => v.toLocaleString("en-US"), onInput: (v) => { out = v; calc(); } }));
      prompt = 2048;
      calc();
      el.querySelector(".pf-calc").after(h("p", { class: "fig-cap" }, "Prefill at 40% of peak dense BF16 compute; decode limited by reading 140 GB of weights plus the KV cache from 8 × 3.35 TB/s of HBM."));
      this.draw = draw;
      draw();
    },
    enter() { if (this.draw) this.draw(); },
  };

  // ------------------------------------------------------------ kvcache
  W.kvcache = {
    init(el, deck) {
      let t = 3;
      const table = el.querySelector(".kv-table"), out = el.querySelector(".kv-out"), check = el.querySelector(".kv-check");
      const ctl = el.querySelector(".kv-ctl");
      const nextBtn = button("Decode next token", () => { if (t < 8) { t++; draw(); } }, { kind: "primary" });
      ctl.append(nextBtn, button("Back to the prompt", () => { t = 3; draw(); }));
      function draw() {
        const inc = M.forwardIncremental(ids.slice(0, t));
        const full = M.forward(ids.slice(0, t));
        let diff = 0;
        full.logits.forEach((r, i) => r.forEach((v, j) => { diff = Math.max(diff, Math.abs(v - inc.logits[i][j])); }));
        clear(table);
        const all = inc.cache.flatMap((c) => c.k.concat(c.v)).flat();
        const m = Math.max(...all.map(Math.abs));
        table.appendChild(h("div", { class: "kvc-row kvc-head" }, h("span", null, "Token"),
          ...["L1 H1", "L1 H2", "L2 H1", "L2 H2"].map((x) => h("span", null, `${x}  k | v`))));
        for (let i = 0; i < t; i++) {
          const groups = [];
          for (let l = 0; l < 2; l++) for (let hh = 0; hh < 2; hh++) {
            const kk = inc.cache[l].k[i].slice(hh * 8, hh * 8 + 8), vv = inc.cache[l].v[i].slice(hh * 8, hh * 8 + 8);
            groups.push(h("span", { class: "kvc-g" }, heatStrip(kk, { maxAbs: m, cell: "n8 kvc" }), heatStrip(vv, { maxAbs: m, cell: "n8 kvc" })));
          }
          table.appendChild(h("div", { class: "kvc-row" + (i === t - 1 && t > 3 ? " is-new" : "") }, h("span", { class: "kvc-tok" }, chip(toks[i])), ...groups));
        }
        // the prefill computes the 3 prompt tokens once either way; without a cache every decode step recomputes all tokens so far
        let without = 3;
        for (let u = 4; u <= t; u++) without += u;
        const withC = t;
        clear(out);
        out.append(stat(String(t), "tokens in the cache"), stat(String(withC), "token entries computed, with cache"), stat(String(without), "without cache (recompute every step)"));
        check.innerHTML = `Live check: next-token logits from cached decoding vs. full recomputation differ by at most <b>${diff === 0 ? "0" : diff.toExponential(1)}</b>. ${t >= 8 ? "The query is complete; the last decode step predicts <code>;</code>." : ""}`;
        nextBtn.disabled = t >= 8;
        deck.strip.set({ prefix: t < 9 ? t : null });
      }
      this.draw = draw;
      draw();
    },
    enter() { if (this.draw) this.draw(); },
  };

  // ------------------------------------------------------------ kvsize
  W.kvsize = {
    init(el) {
      const MODELS = [
        { label: "Ours", L: 2, kv: 2, qh: 2, hd: 8, w: 9968 * 4, ours: true },
        { label: "8B", L: 32, kv: 8, qh: 32, hd: 128, w: 16e9 },
        { label: "70B", L: 80, kv: 8, qh: 64, hd: 128, w: 140e9 },
        { label: "405B", L: 126, kv: 8, qh: 128, hd: 128, w: 810e9 },
      ];
      let mi = 2, ctx = 8192, users = 64, bytes = 2, gqa = true;
      const bar = el.querySelector(".ks-bar"), out = el.querySelector(".ks-out"), note = el.querySelector(".ks-note");
      const ctl = el.querySelector(".ks-ctl");
      ctl.append(
        segmented({ label: "Model", value: mi, options: MODELS.map((m, i) => ({ value: i, label: m.label })), onChange: (v) => { mi = v; draw(); } }),
        segmented({ label: "Cache precision", value: bytes, options: [{ value: 4, label: "FP32" }, { value: 2, label: "BF16" }, { value: 1, label: "FP8" }], onChange: (v) => { bytes = v; draw(); } }),
        segmented({ label: "Key/value heads", value: "gqa", options: [{ value: "gqa", label: "Grouped (8)" }, { value: "mha", label: "One per query head" }], onChange: (v) => { gqa = v === "gqa"; draw(); } }),
        slider({ id: "ks-ctx", label: "Context length", min: 7, max: 17, step: 1, value: 13, format: (v) => Math.pow(2, v).toLocaleString("en-US"), onInput: (v) => { ctx = Math.pow(2, v); draw(); } }),
        slider({ id: "ks-u", label: "Concurrent requests", min: 0, max: 8, step: 1, value: 6, format: (v) => String(Math.pow(2, v)), onInput: (v) => { users = Math.pow(2, v); draw(); } }));
      function draw() {
        const m = MODELS[mi];
        const kvh = m.ours ? m.kv : gqa ? m.kv : m.qh;
        const perTok = 2 * m.L * kvh * m.hd * bytes;
        const perSeq = perTok * ctx, total = perSeq * users;
        const free = SERVER.mem - m.w;
        clear(out);
        out.append(stat(fmt.ibytes(perTok), "per token"), stat(fmt.ibytes(perSeq), `per request (${ctx.toLocaleString("en-US")} tokens)`), stat(fmt.ibytes(total), `for ${users} requests`));
        clear(bar);
        if (!m.ours) {
          const cap = SERVER.mem, maxV = Math.max(cap, m.w + total) * 1.03;
          const pct = (v) => `${(v / maxV) * 100}%`;
          bar.appendChild(h("div", { class: "ks-track", role: "img", "aria-label": `weights ${fmt.bytes(m.w)} and KV cache ${fmt.bytes(total)} against 640 GB` },
            h("span", { class: "ks-w", style: { width: pct(m.w) }, "data-tip": `weights (BF16): ${fmt.bytes(m.w)}` }),
            h("span", { class: "ks-kv", style: { width: pct(total) }, "data-tip": `KV cache: ${fmt.bytes(total)}` }),
            h("span", { class: "ks-cap", style: { left: pct(cap) } }, h("span", null, "8 × H100 = 640 GB"))));
          bar.appendChild(h("div", { class: "legend" }, h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--ink-3)" } }), "weights in BF16"), h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--accent)" } }), "KV cache")));
        }
        if (m.ours) note.innerHTML = `Our model: 2 layers × ${kvh} heads × 8 numbers × 2 (key and value) × ${bytes} bytes = <b>${perTok} bytes</b> per token. The cache for our query peaks at 8 tokens × ${perTok} = ${(perTok * 8).toLocaleString("en-US")} bytes: the final <code>;</code> is predicted but never fed back in.`;
        else if (free <= 0) note.innerHTML = `The BF16 weights alone (${fmt.bytes(m.w)}) exceed one 8-GPU server's 640 GB, so this model needs FP8 weights or more than one server.`;
        else {
          const fits = Math.floor(free / perSeq);
          note.innerHTML = `After the weights, an 8-GPU H100 server has ${fmt.bytes(free)} left: room for <b>${fits.toLocaleString("en-US")}</b> request${fits === 1 ? "" : "s"} of ${ctx.toLocaleString("en-US")} tokens${total > free ? `, so ${users} at once don't fit` : ""}.`;
        }
      }
      draw();
    },
  };

  // ------------------------------------------------------------ membound
  W.membound = {
    init(el) {
      let ctx = 2048, Bsel = 64;
      const charts = el.querySelector(".mb-charts"), out = el.querySelector(".mb-out");
      const step = (B) => {
        const tm = (L70.wBytes + B * ctx * L70.kvTok) / SERVER.bw, tc = (2 * L70.N * B) / SERVER.flops;
        return { t: Math.max(tm, tc), tm, tc };
      };
      const c1 = h("div", { class: "mb-c" }), c2 = h("div", { class: "mb-c" });
      charts.append(h("div", null, h("div", { class: "fig-sub" }, "tokens per second, each user"), c1), h("div", null, h("div", { class: "fig-sub" }, "tokens per second, whole server"), c2));
      const xs = [];
      for (let e = 0; e <= 10; e += 0.1) xs.push(Math.pow(2, e));
      const xax = { type: "log", domain: [1, 1024], ticks: [1, 4, 16, 64, 256, 1024], fmt: String, label: "requests in the batch" };
      const ch1 = chart(c1, { width: 300, height: 190, margin: { l: 44, r: 10, t: 10, b: 36 }, x: xax, y: { domain: [0, 200], ticks: [0, 50, 100, 150, 200] }, aria: "Per-user decode speed vs batch size" });
      const ch2 = chart(c2, { width: 300, height: 190, margin: { l: 50, r: 10, t: 10, b: 36 }, x: xax, y: { type: "log", domain: [100, 100000], ticks: [100, 1000, 10000, 100000], fmt: (v) => v.toLocaleString("en-US") }, aria: "Total decode throughput vs batch size" });
      function draw() {
        const per = xs.map((B) => [B, 1 / step(B).t]), tot = xs.map((B) => [B, B / step(B).t]);
        const sel = step(Bsel);
        ch1.update({ series: [{ points: per, cls: "accent" }], dots: [{ x: Bsel, y: 1 / sel.t, r: 6, cls: "" }] });
        ch2.update({ series: [{ points: tot, cls: "accent" }], dots: [{ x: Bsel, y: Bsel / sel.t, r: 6, cls: "" }] });
        const kvRead = Bsel * ctx * L70.kvTok;
        const why = sel.tc > sel.tm ? "compute" : kvRead > L70.wBytes ? "reading KV caches" : "reading weights";
        clear(out);
        out.append(stat(fmt.time(sel.t), "per decode step"), stat(Math.round(1 / sel.t).toLocaleString("en-US"), "tokens/s per user"), stat(Math.round(Bsel / sel.t).toLocaleString("en-US"), "tokens/s total"), stat(why, "bottleneck"));
      }
      el.querySelector(".mb-ctl").append(
        slider({ id: "mb-b", label: "Requests in the batch", min: 0, max: 10, step: 1, value: 6, format: (v) => String(Math.pow(2, v)), onInput: (v) => { Bsel = Math.pow(2, v); draw(); } }),
        slider({ id: "mb-ctx", label: "Context per request", min: 7, max: 15, step: 1, value: 11, format: (v) => Math.pow(2, v).toLocaleString("en-US"), onInput: (v) => { ctx = Math.pow(2, v); draw(); } }));
      draw();
    },
  };

  // ------------------------------------------------------------ batching
  function simulateBatching(lens, slots, mode) {
    const grid = Array.from({ length: slots }, () => []);
    const finish = new Array(lens.length).fill(0);
    let t = 0, next = 0;
    const active = new Array(slots).fill(null);
    if (mode === "static") {
      while (next < lens.length) {
        const batch = [];
        for (let sl = 0; sl < slots && next < lens.length; sl++) batch.push({ sl, r: next++, left: 0 });
        const span = Math.max(...batch.map((b) => lens[b.r]));
        for (let k = 0; k < span; k++, t++) {
          for (let sl = 0; sl < slots; sl++) {
            const b = batch.find((x) => x.sl === sl);
            grid[sl][t] = b && k < lens[b.r] ? { r: b.r, last: k === lens[b.r] - 1 } : null;
            if (b && k === lens[b.r] - 1) finish[b.r] = t + 1;
          }
        }
      }
    } else {
      while (next < lens.length || active.some((a) => a)) {
        for (let sl = 0; sl < slots; sl++) if (!active[sl] && next < lens.length) active[sl] = { r: next++, done: 0 };
        for (let sl = 0; sl < slots; sl++) {
          const a = active[sl];
          if (!a) { grid[sl][t] = null; continue; }
          a.done++;
          grid[sl][t] = { r: a.r, last: a.done === lens[a.r] };
          if (a.done === lens[a.r]) { finish[a.r] = t + 1; active[sl] = null; }
        }
        t++;
      }
    }
    const busy = lens.reduce((x, y) => x + y, 0);
    return { grid, T: t, util: busy / (t * slots), meanFinish: finish.reduce((x, y) => x + y, 0) / finish.length };
  }
  W.batching = {
    init(el) {
      const LENS = [5, 2, 8, 3, 6, 2, 4, 7, 3, 5];
      const NAMES = "ABCDEFGHIJ".split("");
      const gridEl = el.querySelector(".cb-grid"), out = el.querySelector(".cb-out");
      el.querySelector(".cb-ctl").appendChild(h("p", { class: "fig-cap" }, "Ten queued requests A–J need ", LENS.map((l, i) => `${NAMES[i]} ${l}`).join(", "), " output tokens. The GPU runs 4 requests per decode step."));
      function render(mode) {
        const r = simulateBatching(LENS, 4, mode);
        const cw = 26, rh = 26, lab = 58;
        const vw = lab + Math.max(r.T, 20) * cw + 6, vh = 4 * rh + 6;
        const svg = s("svg", { viewBox: `0 0 ${vw} ${vh}`, width: vw, height: vh, class: "cb-svg", role: "img", "aria-label": `${mode} batching timeline, ${r.T} steps` });
        for (let sl = 0; sl < 4; sl++) {
          svg.appendChild(s("text", { x: lab - 8, y: sl * rh + rh * 0.66, class: "tick", "text-anchor": "end" }, `slot ${sl + 1}`));
          for (let t = 0; t < r.T; t++) {
            const c = r.grid[sl][t];
            const x = lab + t * cw, y = sl * rh;
            if (!c) { svg.appendChild(s("rect", { x: x + 1, y: y + 2, width: cw - 2, height: rh - 4, rx: 3, class: "pp-idle", "data-tip": `slot ${sl + 1}, step ${t + 1}: idle` })); continue; }
            svg.appendChild(s("rect", { x: x + 1, y: y + 2, width: cw - 2, height: rh - 4, rx: 3, class: c.r % 2 ? "cb-a" : "cb-b", "data-tip": `slot ${sl + 1}, step ${t + 1}: request ${NAMES[c.r]}${c.last ? " (last token)" : ""}` }));
            svg.appendChild(s("text", { x: x + cw / 2, y: y + rh * 0.66, class: "pp-t" + (c.r % 2 ? "" : " inv"), "text-anchor": "middle" }, NAMES[c.r]));
          }
        }
        return { svg, r };
      }
      const a = render("static"), b = render("continuous");
      gridEl.append(h("div", { class: "fig-sub" }, `Static batching: ${a.r.T} steps`), a.svg, h("div", { class: "fig-sub" }, `Continuous batching: ${b.r.T} steps`), b.svg);
      out.append(stat(`${a.r.T} → ${b.r.T}`, "decode steps to finish all"), stat(`${fmt.pct(a.r.util)} → ${fmt.pct(b.r.util)}`, "slot utilization"), stat(`${a.r.meanFinish.toFixed(1)} → ${b.r.meanFinish.toFixed(1)}`, "average step a request finishes"));
    },
  };

  // ------------------------------------------------------------ paged
  W.paged = {
    init(el) {
      const REQ = [
        { name: "A", prompt: 5, out: 4 },
        { name: "B", prompt: 5, out: 9 },
        { name: "C", prompt: 3, out: 6 },
        { name: "D", prompt: 6, out: 2 },
      ];
      const MAXOUT = 12, BLOCK = 4;
      let t = 3, share = true;
      const maps = el.querySelector(".pg-maps"), out = el.querySelector(".pg-out");
      const ctl = el.querySelector(".pg-ctl");
      ctl.append(
        slider({ id: "pg-t", label: "Decode steps so far", min: 0, max: 9, step: 1, value: t, format: String, onInput: (v) => { t = v; draw(); } }),
        segmented({ label: "A and B share their prompt", value: "on", options: [{ value: "on", label: "Prefix caching on" }, { value: "off", label: "Off" }], onChange: (v) => { share = v === "on"; draw(); } }));
      el.querySelector(".pg-ctl").after(h("p", { class: "fig-cap" }, "A and B both start with the prompt SELECT name FROM users WHERE (5 tokens). Contiguous slots reserve room for the prompt plus 12 output tokens."));
      function draw() {
        const live = REQ.map((r) => ({ ...r, len: r.prompt + Math.min(t, r.out), done: t >= r.out }));
        // contiguous: each live request reserves prompt + MAXOUT
        clear(maps);
        const contig = h("div", { class: "pg-col" }, h("div", { class: "fig-title" }, "Contiguous reservation"));
        let resv = 0, stored = 0;
        live.forEach((r) => {
          const row = h("div", { class: "pg-row" }, h("span", { class: "pg-name" }, r.name));
          if (r.done) { row.appendChild(h("span", { class: "pg-freed" }, "finished, freed")); contig.appendChild(row); return; }
          resv += r.prompt + MAXOUT; stored += r.len;
          for (let i = 0; i < r.prompt + MAXOUT; i++) row.appendChild(h("span", { class: "pg-slot" + (i < r.prompt ? " prompt" : i < r.len ? " used" : " empty") }));
          contig.appendChild(row);
        });
        // paged: blocks of 4, A and B can share the first full prompt block
        const paged = h("div", { class: "pg-col" }, h("div", { class: "fig-title" }, `Paged, ${BLOCK} tokens per block`));
        const blocks = [];
        const sharedBlock = { owners: [], fill: BLOCK, prompt: true, shared: true };
        live.forEach((r) => {
          if (r.done) return;
          let start = 0;
          if (share && (r.name === "A" || r.name === "B")) {
            if (!sharedBlock.owners.length) blocks.push(sharedBlock);
            sharedBlock.owners.push(r.name);
            start = BLOCK;
          }
          for (let pos = start; pos < r.len; pos += BLOCK) {
            const fill = Math.min(BLOCK, r.len - pos);
            // colour each slot by what it holds: prompt tokens up to r.prompt, generated after
            blocks.push({ owners: [r.name], fill, pos, promptEnd: r.prompt, partial: fill < BLOCK });
          }
        });
        const blockTokens = blocks.reduce((acc, b) => acc + b.fill, 0);
        const grid = h("div", { class: "pg-blocks" });
        blocks.forEach((b, i) => grid.appendChild(h("div", { class: "pg-block" + (b.shared ? " shared" : ""), "data-tip": `physical block ${i}: ${b.owners.join(" + ")}, ${b.fill} of ${BLOCK} slots used${b.shared ? " (shared prefix)" : ""}` },
          h("span", { class: "pg-owner" }, b.owners.join("+")),
          h("span", { class: "pg-fill" }, Array.from({ length: BLOCK }, (_, k) => h("span", { class: "pg-slot" + (k < b.fill ? (b.shared || b.pos + k < b.promptEnd ? " prompt" : " used") : " empty") }))))));
        paged.appendChild(grid);
        maps.append(contig, paged);
        const logical = live.filter((r) => !r.done).reduce((a, r) => a + r.len, 0);
        clear(out);
        out.append(stat(String(logical), "tokens the live requests hold"), stat(`${resv} slots · ${resv ? fmt.pct(stored / resv) : "–"}`, "contiguous: reserved · used"), stat(`${blocks.length * BLOCK} slots · ${blocks.length ? fmt.pct(blockTokens / (blocks.length * BLOCK)) : "–"}`, "paged: allocated · used"));
      }
      draw();
    },
  };

  // ------------------------------------------------------------ speculative
  W.speculative = {
    init(el, deck) {
      const DRAFT = 150, P = 3;
      let k = 4, seed = 1, rand, seq, rounds, passes;
      const seqEl = el.querySelector(".sp-seq"), roundsEl = el.querySelector(".sp-rounds"), out = el.querySelector(".sp-out");
      function reset() { rand = M.mulberry32(seed); seq = ids.slice(0, P); rounds = []; passes = 0; draw(); }
      function round() {
        if (seq.length >= 9 || V[seq[seq.length - 1]] === ";") return;
        const kEff = Math.min(k, 8 - seq.length);
        const drafts = [], qs = [];
        const cur = seq.slice();
        for (let i = 0; i < kEff; i++) {
          const q = M.forward(cur, DRAFT).probs[cur.length - 1];
          const x = M.sampleFrom(q, rand);
          drafts.push(x); qs.push(q); cur.push(x);
          if (V[x] === ";") break;
        }
        const target = M.forward(seq.concat(drafts).slice(0, 8)).probs;
        passes++;
        const log = { proposals: [], extra: null };
        let accepted = 0;
        for (let i = 0; i < drafts.length; i++) {
          const p = target[seq.length - 1 + i], q = qs[i], x = drafts[i];
          const ratio = Math.min(1, p[x] / q[x]);
          if (rand() < ratio) { log.proposals.push({ x, ok: true, ratio }); accepted++; continue; }
          log.proposals.push({ x, ok: false, ratio });
          const resid = p.map((pv, j) => Math.max(0, pv - q[j]));
          const z = resid.reduce((a, b) => a + b, 0);
          log.extra = { x: M.sampleFrom(resid.map((r) => r / z), rand), kind: "replacement" };
          break;
        }
        const newSeq = seq.concat(drafts.slice(0, accepted));
        if (!log.extra && newSeq.length < 9 && V[newSeq[newSeq.length - 1]] !== ";") {
          log.extra = { x: M.sampleFrom(target[newSeq.length - 1], rand), kind: "bonus" };
        }
        seq = newSeq.concat(log.extra ? [log.extra.x] : []);
        log.gained = seq.length - (newSeq.length - accepted);
        rounds.push(log);
      }
      function draw() {
        clear(seqEl);
        seq.forEach((id, i) => seqEl.appendChild(chip(V[id], { cls: i < P ? "is-prompt" : "" })));
        clear(roundsEl);
        rounds.forEach((r, i) => roundsEl.appendChild(h("div", { class: "sp-round" },
          h("span", { class: "gen-k" }, `pass ${i + 1}`),
          h("span", { class: "sp-props" }, r.proposals.map((p) => h("span", { class: "sp-p " + (p.ok ? "ok" : "no"), "data-tip": `accept probability min(1, p/q) = ${p.ratio.toFixed(2)}` }, chip(V[p.x]), h("span", { class: "mark" }, p.ok ? "✓" : "✗"))),
            r.extra ? h("span", { class: "sp-p extra" }, chip(V[r.extra.x], { cls: "is-focus" }), h("span", { class: "mark" }, r.extra.kind === "bonus" ? "bonus" : "fixed")) : null),
          h("span", { class: "num" }, `+${r.gained}`))));
        if (!rounds.length) roundsEl.appendChild(h("p", { class: "fig-cap" }, "Prompt: SELECT name FROM. Each round, the draft proposes up to k tokens and the final model checks them all in a single pass."));
        const gen = seq.length - P;
        clear(out);
        out.append(stat(String(passes), "large-model passes"), stat(String(gen), "tokens generated"), stat(passes ? (gen / passes).toFixed(2) : "–", "tokens per pass (plain decoding: 1)"));
        deck.strip.set({ tokens: seq.map((x) => V[x]).concat(toks.slice(seq.length)), prefix: seq.length < 9 ? seq.length : null });
      }
      el.querySelector(".sp-ctl").append(
        segmented({ label: "Draft length k", value: k, options: [2, 3, 4, 5].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { k = v; reset(); } }),
        button("Run one pass", () => { round(); draw(); }, { kind: "primary" }),
        button("Finish the query", () => { for (let i = 0; i < 9; i++) round(); draw(); }),
        button("New run", () => { seed++; reset(); }));
      // formula explorer
      let a = 0.8, kk = 4, c = 0.1;
      const calcOut = el.querySelector(".sp-calc-out");
      function calc() {
        const e = (1 - Math.pow(a, kk + 1)) / (1 - a);
        clear(calcOut);
        calcOut.append(stat(e.toFixed(2), "tokens per large-model pass"), stat(`${(e / (kk * c + 1)).toFixed(2)}×`, "speedup if a draft step costs c of a large step"));
      }
      el.querySelector(".sp-calc").append(
        slider({ id: "sp-a", label: "Acceptance rate α", min: 0.3, max: 0.98, step: 0.01, value: a, format: (v) => v.toFixed(2), onInput: (v) => { a = v; calc(); } }),
        slider({ id: "sp-k", label: "Draft tokens k", min: 1, max: 10, step: 1, value: kk, format: String, onInput: (v) => { kk = v; calc(); } }),
        slider({ id: "sp-c", label: "Draft cost c", min: 0.01, max: 0.5, step: 0.01, value: c, format: (v) => v.toFixed(2), onInput: (v) => { c = v; calc(); } }));
      calc();
      this.draw = draw;
      reset();
    },
    enter() { if (this.draw) this.draw(); },
  };

  // ------------------------------------------------------------ quant
  function quantizeModel(Wt, bits, perChannel) {
    if (bits >= 32) return Wt;
    const out = {}, qmax = Math.pow(2, bits - 1) - 1;
    for (const [name, t] of Object.entries(Wt)) {
      if (t.shape.length < 2) { out[name] = t; continue; }
      const [r, c] = t.shape, data = new Float64Array(t.data.length);
      const scales = [];
      if (perChannel) for (let j = 0; j < c; j++) { let m = 0; for (let i = 0; i < r; i++) m = Math.max(m, Math.abs(t.data[i * c + j])); scales.push(m / qmax || 1); }
      else { let m = 0; for (const v of t.data) m = Math.max(m, Math.abs(v)); scales.push(m / qmax || 1); }
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
        const sc = perChannel ? scales[j] : scales[0];
        data[i * c + j] = Math.max(-qmax, Math.min(qmax, Math.round(t.data[i * c + j] / sc))) * sc;
      }
      out[name] = { shape: t.shape, data };
    }
    return out;
  }
  W._quantizeModel = quantizeModel;

  W.quant = {
    init(el) {
      let bits = 8, perCh = true;
      const Wt = M.weights();
      const rowEl = el.querySelector(".qt-row"), out = el.querySelector(".qt-out");
      const bars = barList(el.querySelector(".qt-bars"), { labelWidth: "6.4rem" });
      const baseLoss = M.exactLoss().mean;
      const cache = new Map();
      el.querySelector(".qt-ctl").append(
        segmented({ label: "Bits per weight", value: bits, options: [32, 8, 6, 5, 4, 3, 2].map((b) => ({ value: b, label: b === 32 ? "FP32" : String(b) })), onChange: (v) => { bits = v; draw(); } }),
        segmented({ label: "Scale", value: "ch", options: [{ value: "ch", label: "one per output channel" }, { value: "t", label: "one per matrix" }], onChange: (v) => { perCh = v === "ch"; draw(); } }));
      function draw() {
        const key = `${bits}-${perCh}`;
        if (!cache.has(key)) {
          const Wq = quantizeModel(Wt, bits, perCh);
          const ex = M.exactLoss(M.finalStep, { weights: Wq });
          const tr = M.forward(ids.slice(0, 8), M.finalStep, { weights: Wq });
          cache.set(key, { Wq, ex, tr });
        }
        const { Wq, ex, tr } = cache.get(key);
        // one row of weights: the first 16 entries of W_U's column for " id"
        const col = M.idOf(" id"), wu = Wt.wu, wq = Wq.wu;
        const orig = Array.from({ length: 16 }, (_, r) => wu.data[r * wu.shape[1] + col]);
        const quan = Array.from({ length: 16 }, (_, r) => wq.data[r * wq.shape[1] + col]);
        const m = Math.max(...orig.map(Math.abs));
        clear(rowEl);
        rowEl.append(
          h("div", { class: "vec-row" }, h("div", { class: "vec-label" }, "16 weights (output column for ", chip(" id"), ")"), heatStrip(orig, { maxAbs: m })),
          h("div", { class: "vec-row" }, h("div", { class: "vec-label" }, bits >= 32 ? "same, unquantized" : `after ${bits}-bit rounding`), heatStrip(quan, { maxAbs: m, tipFmt: (v, i) => `dim ${i}: ${fmt.minus(v.toFixed(4))} (was ${fmt.minus(orig[i].toFixed(4))})` })));
        const params = M.config.params;
        const sizeB = bits >= 32 ? params * 4 : (params * bits) / 8;
        clear(out);
        out.append(stat(ex.mean.toFixed(4), `loss on all queries (FP32: ${baseLoss.toFixed(4)})`),
          stat(fmt.p(tr.probs[2][M.idOf(" users")]), "p(users) after FROM"),
          stat(fmt.bytes(sizeB), "weights in memory"));
        bars.update([" id", " email", " age", " name"].map((t) => ({ key: t, label: chip(t), value: tr.probs[4][M.idOf(t)], cls: t === " id" ? "is-target" : "" })));
      }
      el.querySelector(".qt-bars").before(h("div", { class: "fig-sub" }, "Next token after …users WHERE"));
      draw();
    },
  };

  // ------------------------------------------------------------ fleet
  W.fleet = {
    init(el, deck) {
      const dia = el.querySelector(".fl-diagram"), note = el.querySelector(".fl2-note");
      const Wd = 560, Hd = 250;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "chart fleet-svg", role: "img", "aria-label": "Request path: router, prefill pool, KV transfer, decode pool" });
      svg.appendChild(s("defs", null, s("marker", { id: "fl-ah", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "ah" }))));
      const box = (x, y, w, hgt, title, sub, cls) => {
        const g = s("g", { class: "fl-box " + (cls || "") });
        g.appendChild(s("rect", { x, y, width: w, height: hgt, rx: 10 }));
        g.appendChild(s("text", { x: x + 10, y: y + 20, class: "ln-label" }, title));
        if (sub) g.appendChild(s("text", { x: x + 10, y: y + 36, class: "ln-sub" }, sub));
        svg.appendChild(g);
        return g;
      };
      const arrow = (x1, y1, x2, y2, label, ly) => {
        svg.appendChild(s("line", { x1, y1, x2, y2, class: "loop-arrow", "marker-end": "url(#fl-ah)" }));
        if (label) svg.appendChild(s("text", { x: (x1 + x2) / 2, y: ly ?? (y1 + y2) / 2 - 6, class: "ln-sub", "text-anchor": "middle" }, label));
      };
      const stages = [
        box(10, 95, 92, 56, "Users", "requests in"),
        box(132, 95, 112, 56, "Router", "prefix-aware"),
        box(276, 20, 128, 70, "Prefill pool", "compute-bound"),
        box(276, 160, 128, 70, "Decode pool", "bandwidth-bound"),
      ];
      // GPU squares inside pools
      for (let i = 0; i < 8; i++) {
        svg.appendChild(s("rect", { x: 286 + i * 14, y: 64, width: 10, height: 10, rx: 2, class: "fl-gpu" }));
        svg.appendChild(s("rect", { x: 286 + i * 14, y: 204, width: 10, height: 10, rx: 2, class: "fl-gpu" }));
      }
      arrow(102, 123, 130, 123);
      arrow(244, 112, 274, 62, "prompt");
      arrow(340, 92, 340, 158, "", 0);
      svg.appendChild(s("text", { x: 348, y: 130, class: "ln-sub" }, "KV cache over RDMA"));
      arrow(404, 195, 470, 195, "tokens");
      box(472, 165, 80, 56, "Stream", "to the user");
      dia.appendChild(svg);
      const STEPS = [
        [0, "A user sends SELECT name FROM."],
        [1, "The router picks a replica, preferring one that already caches this prompt's prefix, so the prefill can skip those tokens."],
        [2, "A prefill server runs the 3-token prompt in one pass and produces the first token, users."],
        [3, "The KV cache moves to a decode server, which generates WHERE, id, =, 7, ; one step at a time, batched with other requests."],
      ];
      let k = 0;
      function show() {
        stages.forEach((g, i) => g.classList.toggle("is-on", i === STEPS[k][0]));
        note.textContent = `${k + 1}/${STEPS.length}: ${STEPS[k][1]}`;
      }
      el.querySelector(".fl2-ctl").append(button("Next stage", () => { k = (k + 1) % STEPS.length; show(); }, { kind: "primary" }));
      show();
    },
  };
})();
