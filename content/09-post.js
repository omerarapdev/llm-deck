/* Part 9 — Post-training: from a model that continues text to an assistant and a reasoner. */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip, slider, segmented, button } = window.UI;
  const M = window.TinyLM;
  const ids = (text) => M.encode(text).map(M.idOf);
  const num = (v, d = 3) => fmt.minus((Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : v).toFixed(d));
  const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, l));
  const int = (x) => Math.round(x).toLocaleString("en-US");
  const gb = (b) => (b >= 1e11 ? `${int(b / 1e9)} GB` : `${(b / 1e9).toFixed(1)} GB`);

  /* ------------------------------------------------------------------ shared model numbers */
  // Probability of each token of `answer` given `prompt` and the answer tokens before it.
  function answerProbs(prompt, answer) {
    const pre = ids(prompt), ans = ids(answer), all = pre.concat(ans);
    const tr = M.forward(all.slice(0, Math.min(all.length - 1, M.config.T)));
    return ans.map((id, i) => tr.probs[pre.length - 1 + i][id]);
  }


  const sig = (x) => 1 / (1 + Math.exp(-x));
  // The task used from chapter 9.3 on: "the email of user 7".
  const TASK = { prompt: "SELECT email FROM users WHERE", answer: " id = 7;" };
  let pTaskCache = null;
  const pTask = () => (pTaskCache === null ? (pTaskCache = answerProbs(TASK.prompt, TASK.answer).reduce((a, b) => a * b, 1)) : pTaskCache);
  // KL-regularized optimum with a reward gap g for the asked-for answer: pi* = pi_ref e^{r/beta} / Z
  const qStar = (beta, gap, p0) => 1 / (1 + ((1 - p0) / p0) * Math.exp(-gap / beta));
  const xlog = (x, y) => (x > 0 ? x * Math.log(x / y) : 0);
  const klTwo = (q, p0) => xlog(q, p0) + xlog(1 - q, 1 - p0);
  const odds = (v) => (v >= 1 ? `${v < 1000 ? v.toFixed(2) : fmt.sci(v, 2)} : 1` : `1 : ${(1 / v < 1000 ? (1 / v).toFixed(2) : fmt.sci(1 / v, 2))}`);
  const clip = (x, a, b) => Math.max(a, Math.min(b, x));
  const ppoObj = (rho, A, eps) => Math.min(rho * A, clip(rho, 1 - eps, 1 + eps) * A);
  const ppoGrad = (rho, A, eps) => (rho * A <= clip(rho, 1 - eps, 1 + eps) * A ? A : 0);

  /* ------------------------------------------------------------------ fig: a base model completes a prompt */
  Deck.fig("post-continue", (el, sp) => {
    const pre = ids(sp.prompt), goal = ids(sp.answer);
    const exact = answerProbs(sp.prompt, sp.answer);
    const pGoal = exact.reduce((a, b) => a * b, 1);
    let rand, drawn;
    const dots = h("div", { class: "post-dots", role: "img", "aria-label": "outcome of every sample so far" });
    const list = h("ol", { class: "post-samples" });
    const tiles = h("div", { class: "f-tiles" });
    const promptRow = h("div", { class: "post-prompt" }, h("span", { class: "post-tag" }, "prompt"), pre.map((id) => chip(M.vocab[id])), h("span", { class: "post-gap" }, "…"));
    function sampleOne() {
      const seq = pre.slice();
      for (let t = 0; t < goal.length; t++) seq.push(M.sampleFrom(M.nextDist(seq), rand));
      const out = seq.slice(pre.length);
      return { out, ok: out.every((id, i) => id === goal[i]) };
    }
    function more(n) { for (let i = 0; i < n; i++) drawn.push(sampleOne()); draw(); }
    function reset() { rand = M.mulberry32(sp.seed); drawn = []; more(sp.start || 10); }
    function draw() {
      clear(dots); clear(list); clear(tiles);
      drawn.forEach((d, i) => dots.appendChild(h("span", { class: "post-dot" + (d.ok ? " ok" : ""), "data-tip": `sample ${i + 1}:${d.out.map((id) => M.vocab[id]).join("")}` })));
      drawn.slice(-6).forEach((d, k) => list.appendChild(h("li", { class: d.ok ? "ok" : "" },
        h("span", { class: "post-n" }, String(drawn.length - Math.min(6, drawn.length) + k + 1)),
        h("span", { class: "post-chips" }, d.out.map((id) => chip(M.vocab[id]))),
        h("span", { class: "post-mark" }, d.ok ? "✓ the asked-for query" : "✗ another query"))));
      const ok = drawn.filter((d) => d.ok).length;
      tiles.append(tile(`${ok} of ${drawn.length}`, "samples that answer the request"), tile(fmt.pct(ok / drawn.length, 1), "share so far"), tile(fmt.pct(pGoal, 2), "exact probability"));
    }
    el.append(promptRow, list, dots, tiles,
      h("p", { class: "f-mm-expl" }, `Exact: p(${sp.answer.trim().split(" ")[0]}) × p(=) × p(7) × p(;) = ${exact.map((p) => num(p, 3)).join(" × ")} = ${num(pGoal, 4)}, one sample in ${Math.round(1 / pGoal)}.`),
      h("div", { class: "f-ctl" }, button("Sample 10 more", () => more(10), { kind: "primary" }), button("Sample 100 more", () => more(100)), button("Reset", reset)));
    reset();
  });

  /* ------------------------------------------------------------------ fig: a chat template as one sequence */
  Deck.fig("post-chat", (el, sp) => {
    const TIP = {
      "<|begin_of_text|>": "Start of the sequence (Meta: equivalent to the BOS token).",
      "<|start_header_id|>": "Opens the header that names who writes the next message.",
      "<|end_header_id|>": "Closes the header. Two newline characters follow it.",
      "<|eot_id|>": "End of the message in a turn. The model emits it to stop.",
      "\\n\\n": "Two newline characters (0x0A 0x0A), part of the format.",
    };
    const seg = (x, gen) => {
      const special = /^<\|/.test(x);
      const role = /^(system|user|assistant)$/.test(x);
      return h("span", { class: "post-seg" + (special ? " sp" : x === "\\n\\n" ? " nl" : role ? " role" : " txt") + (gen ? " gen" : ""), "data-tip": TIP[x] || (role ? `Role of the message: ${x}.` : gen ? "Written by the model; the loss is computed on these tokens." : "Message text.") }, x);
    };
    const msg = (role, text) => ["<|start_header_id|>", role, "<|end_header_id|>", "\\n\\n"].concat(text === null ? [] : [text, "<|eot_id|>"]);
    const prompt = ["<|begin_of_text|>"].concat(msg("system", sp.system), msg("user", sp.user), msg("assistant", null));
    el.append(
      h("div", { class: "post-chat" }, h("div", { class: "post-chat-l" }, "prompt, given"), h("div", { class: "post-chat-row" }, prompt.map((x) => seg(x, false)))),
      h("div", { class: "post-chat" }, h("div", { class: "post-chat-l" }, "written by the model"), h("div", { class: "post-chat-row" }, [sp.answer, "<|eot_id|>"].map((x) => seg(x, true)))),
      h("p", { class: "f-note" }, "Filled boxes are special tokens, one vocabulary entry each; \\n\\n is two newline characters. Hover any box for its role."));
  });


  /* ------------------------------------------------------------------ fig: packing and attention masks */
  Deck.fig("post-pack", (el, sp) => {
    const L = sp.lengths.slice();
    let mode = sp.mode || "doc";
    const uid = window.UI.nextId("pk");
    const host = h("div", { class: "post-svg-host" });
    const tiles = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const EX = ["s1", "s2", "s3"];
    function draw() {
      const T = L.reduce((a, b) => a + b, 0), mx = Math.max(...L);
      const own = [];
      L.forEach((n, e) => { for (let i = 0; i < n; i++) own.push(e); });
      const Wd = 380, x0 = 50, top = 22, n = mode === "pad" ? mx : T;
      const tok = Math.min(20, (Wd - x0 - 4) / n);
      const rows = mode === "pad" ? 3 : 1;
      const maskTop = top + rows * (tok + 4) + 30;
      const span = mode === "pad" ? (Wd - 2 * 14) / 3 : 220, cell = span / n;
      const Hd = maskTop + span + 8;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "post-pack-svg", role: "img", "aria-label": "token slots and the attention mask" });
      svg.appendChild(s("defs", null, s("pattern", { id: `${uid}-h`, width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, s("line", { x1: 0, y1: 0, x2: 0, y2: 5, class: "post-hatch" }))));
      svg.appendChild(s("text", { x: 0, y: 14, class: "post-ax" }, mode === "pad" ? "one row per example, padded to the longest" : "one packed row"));
      const slot = (x, y, e, label, tip) => {
        svg.appendChild(s("rect", { x, y, width: tok - 2, height: tok, class: e < 0 ? "post-pad" : `post-tok ${EX[e]}`, style: e < 0 ? `fill:url(#${uid}-h)` : null, "data-tip": tip }));
        if (label && tok >= 14) svg.appendChild(s("text", { x: x + (tok - 2) / 2, y: y + tok * 0.68, class: "post-tok-l", "text-anchor": "middle" }, label));
      };
      if (mode === "pad") L.forEach((len, e) => { for (let i = 0; i < mx; i++) slot(x0 + i * tok, top + e * (tok + 4), i < len ? e : -1, i < len ? String(i) : "", i < len ? `example ${e + 1}, token ${i + 1}` : "padding"); });
      else own.forEach((e, i) => slot(x0 + i * tok, top, e, String(i), `example ${e + 1}, position ${i}`));
      for (let e = 0; e < rows; e++) svg.appendChild(s("text", { x: x0 - 6, y: top + e * (tok + 4) + tok * 0.7, class: "post-ax", "text-anchor": "end" }, mode === "pad" ? `ex ${e + 1}` : "packed"));
      svg.appendChild(s("text", { x: 0, y: maskTop - 10, class: "post-ax" }, "attention mask: row = query, column = key"));
      let cross = 0, kept = 0;
      const grid = (gx, size, allowed, colorOf) => {
        for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) {
          const a = allowed(i, j), c = colorOf(i, j);
          svg.appendChild(s("rect", { x: gx + j * cell, y: maskTop + i * cell, width: cell, height: cell, class: a ? (c === "x" ? "post-m cross" : `post-m on ${c}`) : c === "pad" ? "post-m pad" : "post-m off", style: c === "pad" ? `fill:url(#${uid}-h)` : null }));
        }
        svg.appendChild(s("rect", { x: gx, y: maskTop, width: size * cell, height: size * cell, class: "post-m-frame" }));
      };
      if (mode === "pad") {
        L.forEach((len, e) => grid(e * (span + 14), mx, (i, j) => i < len && j < len && j <= i, (i, j) => (i < len && j < len ? EX[e] : "pad")));
        kept = L.reduce((a, l) => a + (l * (l + 1)) / 2, 0);
      } else {
        grid(0, T, (i, j) => j <= i && (mode === "causal" || own[i] === own[j]), (i, j) => (own[i] === own[j] ? EX[own[i]] : "x"));
        [["s1", "example 1"], ["s2", "example 2"], ["s3", "example 3"], ["cross", "across examples"], ["off", "masked out"]].forEach(([c, t], k) => {
          svg.appendChild(s("rect", { x: span + 16, y: maskTop + 4 + k * 22, width: 14, height: 14, class: `post-sw ${c}` }));
          svg.appendChild(s("text", { x: span + 36, y: maskTop + 15 + k * 22, class: "post-ax" }, t));
        });
        for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) { if (own[i] === own[j]) kept++; else if (mode === "causal") cross++; }
      }
      clear(host); host.appendChild(svg);
      const slots = mode === "pad" ? 3 * mx : T;
      clear(tiles);
      tiles.append(tile(String(slots), "token slots"), tile(`${slots - T} (${fmt.pct((slots - T) / slots)})`, "padding slots"),
        tile(String(kept + cross), "query–key pairs allowed"), tile(String(cross), "pairs across examples"));
      expl.innerHTML = mode === "pad"
        ? `Each example gets its own row of ${mx} slots. ${slots - T} of ${slots} slots are padding: computed, then thrown away.`
        : mode === "causal"
          ? `No padding, but the causal mask lets each token see every earlier token of the row: ${cross} of the ${kept + cross} allowed pairs cross from one example into another.`
          : `Block-diagonal mask: ${L.map((l) => (l * (l + 1)) / 2).join(" + ")} = ${kept} pairs, exactly what ${L.length} separate sequences would compute, with no padding.`;
    }
    const ctl = h("div", { class: "f-ctl sliders" });
    ctl.appendChild(segmented({ label: "Layout", value: mode, options: [{ value: "pad", label: "Padded" }, { value: "causal", label: "Packed, causal mask" }, { value: "doc", label: "Packed, per-example mask" }], onChange: (v) => { mode = v; draw(); } }));
    L.forEach((v, e) => ctl.appendChild(slider({ label: `example ${e + 1} length`, min: 1, max: 8, step: 1, value: v, format: (x) => String(x), onInput: (x) => { L[e] = x; draw(); } })));
    el.append(host, tiles, expl, ctl);
    draw();
  });

  /* ------------------------------------------------------------------ fig: LoRA shapes and counts */
  Deck.fig("post-lora", (el, sp) => {
    const P = sp.presets;
    let pi = sp.preset || 0, r = sp.r || 16;
    const host = h("div", { class: "post-svg-host" });
    const tiles = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const rBox = h("div");
    const llamaLayer = 2 * (4096 + 4096) + 2 * (4096 + 1024) + 3 * (4096 + 14336); // r(d+k) sum over the 7 linear layers of one Llama 3 8B block, per unit r
    const llamaTotal = 2 * 128256 * 4096 + 32 * (2 * 4096 * 4096 + 2 * 4096 * 1024 + 3 * 4096 * 14336 + 2 * 4096) + 4096;
    function rOptions() {
      const { d, k } = P[pi];
      return [1, 2, 4, 8, 16, 64, 256].filter((x) => x <= Math.min(d, k));
    }
    function buildR() {
      const opts = rOptions();
      if (!opts.includes(r)) r = opts[opts.length - 1];
      clear(rBox);
      rBox.appendChild(segmented({ label: "Rank r", value: r, options: opts.map((x) => ({ value: x, label: String(x) })), onChange: (v) => { r = v; draw(); } }));
    }
    const dec = (x) => (Number.isInteger(x) ? int(x) : int(Math.floor(x)) + (x % 1).toFixed(1).slice(1));
    function draw() {
      const { d, k } = P[pi];
      const Wd = 400, Hd = 230, x0 = 46, y0 = 36;
      // scale so that W0 fits a 140 px box and W0, B and A fit side by side
      const sc = Math.min(140 / Math.max(d, k), (Wd - x0 - 10 - 52 - 30) / (2 * k + r));
      const wW = k * sc, wH = d * sc, thin = (v) => Math.max(1.5, v * sc);
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "post-lora-svg", role: "img", "aria-label": `LoRA: a ${d} by ${k} matrix and its rank ${r} factors` });
      const lab = (x, y, t, cls = "post-ax", anchor = "middle") => svg.appendChild(s("text", { x, y, class: cls, "text-anchor": anchor }, t));
      svg.appendChild(s("rect", { x: x0, y: y0, width: wW, height: wH, class: "post-w0", "data-tip": `W₀: ${int(d)} × ${int(k)} = ${int(d * k)} frozen weights` }));
      lab(x0 + wW / 2, y0 - 10, "W₀ (frozen)", "post-ax b");
      lab(x0 + wW / 2, y0 + wH + 18, `${int(d)} × ${int(k)}`);
      const xp = x0 + wW + 26;
      lab(xp, y0 + wH / 2 + 9, "+", "post-op");
      const bx = xp + 26, bw = thin(r), ax = bx + bw + 30, aw = wW, ah = thin(r);
      svg.appendChild(s("rect", { x: bx, y: y0, width: bw, height: wH, class: "post-b", "data-tip": `B: ${int(d)} × ${r} = ${int(d * r)} trainable, starts at zero` }));
      lab(bx + bw / 2, y0 - 10, "B", "post-ax b");
      lab(bx + bw / 2, y0 + wH + 18, `${int(d)} × ${r}`);
      lab(bx + bw + 15, y0 + Math.min(wH, ah) / 2 + 6, "×", "post-op sm");
      svg.appendChild(s("rect", { x: ax, y: y0, width: aw, height: ah, class: "post-a", "data-tip": `A: ${r} × ${int(k)} = ${int(r * k)} trainable, random Gaussian start` }));
      lab(ax + aw / 2, y0 - 10, "A", "post-ax b");
      lab(ax + aw / 2, y0 + ah + 18, `${r} × ${int(k)}`);
      lab(Wd - 4, Hd - 6, "to scale; thin factors drawn at 1.5 px", "post-ax note", "end");
      clear(host); host.appendChild(svg);
      const full = d * k, lora = r * (d + k), lim = full / (d + k);
      clear(tiles);
      tiles.append(tile(int(full), "weights in W₀ (frozen)"), tile(int(lora), "trainable: r(d + k)"), tile(fmt.pct(lora / full, 2), "trainable / frozen"));
      const whole = `On all 224 linear layers of Llama 3 8B, rank ${r} trains ${(32 * r * llamaLayer / 1e6).toFixed(1)}M parameters, ${fmt.pct(32 * r * llamaLayer / llamaTotal, 2)} of 8.03B.`;
      expl.innerHTML = (lora < full
        ? `r(d + k) = ${r} × (${int(d)} + ${int(k)}) = ${int(lora)} trainable numbers instead of ${int(full)}. LoRA saves parameters while r is below dk/(d + k) = ${dec(lim)}. `
        : `At r = ${r} the factors hold ${int(lora)} numbers, ${lora === full ? "as many as" : "more than"} the ${int(full)} of W₀: LoRA saves nothing once r reaches dk/(d + k) = ${dec(lim)}. `) + whole;
    }
    el.append(host, tiles, expl, h("div", { class: "f-ctl" },
      segmented({ label: "Matrix", value: pi, options: P.map((p, i) => ({ value: i, label: p.label })), onChange: (v) => { pi = v; buildR(); draw(); } }), rBox));
    buildR();
    draw();
  });


  /* ------------------------------------------------------------------ the 3-action and tabular policies used in 9.4 */
  const COLS = [" id", " name", " age"];
  let tabCache = null;
  // Our model's distributions after "SELECT email FROM users WHERE" (filter column) and after "... id =" (digit),
  // as logits of the small tabular policy the GRPO widget also uses.
  function tabular() {
    if (tabCache) return tabCache;
    const pre = ids(TASK.prompt);
    const pc = M.forward(pre).probs[pre.length - 1];
    const pd = M.forward(pre.concat(ids(" id ="))).probs[pre.length + 1];
    tabCache = { zc: COLS.map((c) => Math.log(pc[M.idOf(c)])), zd: [1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => Math.log(pd[M.idOf(" " + d)])) };
    return tabCache;
  }
  // GRPO on the tabular policy, exactly as the rl widget runs it: G samples (column then digit), group-normalized
  // advantages with the population standard deviation, one gradient-ascent step of size lr on the logits.
  function grpoTab(reward, seed, { G = 8, N = 40, lr = 1 } = {}) {
    let { zc, zd } = tabular();
    zc = zc.slice(); zd = zd.slice();
    const rand = M.mulberry32(seed), hist = [];
    let zero = 0;
    for (let u = 0; u <= N; u++) {
      const pc = M.softmax(zc), pd = M.softmax(zd);
      hist.push({ id: pc[0], ok: pc[0] * pd[6] });
      if (u === N) break;
      const g = Array.from({ length: G }, () => { const c = M.sampleFrom(pc, rand), d = M.sampleFrom(pd, rand); return { c, d, r: reward(c, d) }; });
      const mean = g.reduce((a, x) => a + x.r, 0) / G, sd = Math.sqrt(g.reduce((a, x) => a + (x.r - mean) ** 2, 0) / G);
      if (sd === 0) { zero++; continue; }
      const gc = [0, 0, 0], gd = new Array(9).fill(0);
      g.forEach((x) => {
        const A = (x.r - mean) / sd;
        for (let i = 0; i < 3; i++) gc[i] += (A * ((i === x.c ? 1 : 0) - pc[i])) / G;
        for (let i = 0; i < 9; i++) gd[i] += (A * ((i === x.d ? 1 : 0) - pd[i])) / G;
      });
      zc = zc.map((v, i) => v + lr * gc[i]);
      zd = zd.map((v, i) => v + lr * gd[i]);
    }
    return { hist, zero };
  }
  const REWARD = { proxy: (c) => (c === 0 ? 1 : 0), truth: (c, d) => (c === 0 && d === 6 ? 1 : 0) };

  /* ------------------------------------------------------------------ fig: policy-gradient estimates, exactly */
  // Three actions (the filter column, digit fixed at 7), rewards (1, 0, 0) plus a shift c, optional baseline.
  function pgStats(c, useBaseline) {
    const pi = M.softmax(tabular().zc), r = [1, 0, 0].map((x) => x + c);
    const J = pi.reduce((a, p, i) => a + p * r[i], 0), b = useBaseline ? J : 0;
    const g = pi.map((_, a) => pi.map((p, k) => (r[a] - b) * ((k === a ? 1 : 0) - p)));
    const Eg = pi.map((_, k) => pi.reduce((acc, p, a) => acc + p * g[a][k], 0));
    const exact = pi.map((p, k) => p * ([1, 0, 0][k] + c - J));
    const var_ = pi.reduce((acc, p, a) => acc + p * g[a].reduce((q, x) => q + x * x, 0), 0) - Eg.reduce((q, x) => q + x * x, 0);
    return { pi, r, J, b, g, Eg, exact, var: var_ };
  }
  Deck.fig("post-pg", (el, sp) => {
    let c = sp.shift || 0, base = !!sp.baseline;
    const tableHost = h("div", { class: "scroll-x" });
    const tiles = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const vec = (v) => `(${v.map((x) => num(x, 3)).join(", ")})`;
    function draw() {
      const st = pgStats(c, base);
      const t = h("table", { class: "f-table dense post-pg" },
        h("thead", null, h("tr", null, ["action", "π(a)", "reward r", "weight r − b", "estimate g if sampled"].map((x, j) => h("th", { class: j && j < 4 ? "r" : "" }, x)))),
        h("tbody", null, COLS.map((col, a) => h("tr", { class: a === 0 ? "hi" : "" },
          h("td", { class: "nowrap" }, h("code", null, `${col.trim()} = 7`)), h("td", { class: "r" }, num(st.pi[a], 3)), h("td", { class: "r" }, num(st.r[a], 1)),
          h("td", { class: "r" }, num(st.r[a] - st.b, 3)), h("td", { class: "mono" }, vec(st.g[a])))),
        h("tr", { class: "sum" }, h("td", null, "mean"), h("td", null, ""), h("td", { class: "r" }, `J = ${num(st.J, 3)}`), h("td", { class: "r" }, `b = ${num(st.b, 3)}`), h("td", { class: "mono" }, vec(st.Eg)))));
      clear(tableHost); tableHost.appendChild(t);
      clear(tiles);
      tiles.append(tile(vec(st.Eg), window.UI.rich(String.raw`mean of the estimates, \(\mathbb{E}[g]\)`)), tile(vec(st.exact), window.UI.rich(String.raw`exact gradient \(\nabla_z J\)`)), tile(num(st.var, 3), window.UI.rich(String.raw`variance of \(g\), summed over components`)));
      expl.innerHTML = `Rewards (${st.r.map((x) => num(x, 1)).join(", ")}), baseline b = ${num(st.b, 3)}. The mean of the estimates equals the exact gradient for every shift and baseline; only the spread changes.`;
    }
    el.append(tableHost, tiles, expl, h("div", { class: "f-ctl sliders" },
      slider({ label: "add to every reward", min: -2, max: 10, step: 0.5, value: c, format: (x) => fmt.minus(x.toFixed(1)), onInput: (x) => { c = x; draw(); } }),
      segmented({ label: "Baseline", value: base ? "mean" : "none", options: [{ value: "none", label: "None" }, { value: "mean", label: "Mean reward" }], onChange: (v) => { base = v === "mean"; draw(); } })));
    draw();
  });

  /* ------------------------------------------------------------------ fig: optimizing a proxy reward */
  Deck.fig("post-hack", (el, sp) => {
    let seed = sp.seed || 5;
    const host = h("div");
    const tiles = h("div", { class: "f-tiles" });
    const N = sp.updates || 40;
    const ch = window.UI.chart(host, {
      width: 560, height: 260, margin: { l: 46, r: 16, t: 12, b: 38 }, aria: "proxy reward and true success over GRPO updates",
      x: { domain: [0, N], ticks: [0, 10, 20, 30, 40], label: "updates" }, y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], label: "probability", fmt: (v) => fmt.pct(v) },
    });
    function draw() {
      const a = grpoTab(REWARD.proxy, seed, { N }), b = grpoTab(REWARD.truth, seed, { N });
      const pts = (r, k) => r.hist.map((x, i) => [i, x[k]]);
      ch.update({ series: [{ points: pts(a, "id"), cls: "s2" }, { points: pts(a, "ok"), cls: "s1" }, { points: pts(b, "ok"), cls: "s3 dashed" }],
        dots: [{ x: N, y: a.hist[N].id, r: 4, cls: "s2", tip: `proxy reward ${fmt.pct(a.hist[N].id, 1)}` }, { x: N, y: a.hist[N].ok, r: 4, cls: "s1", tip: `truly correct ${fmt.pct(a.hist[N].ok, 1)}` }, { x: N, y: b.hist[N].ok, r: 4, cls: "s3", tip: `truly correct ${fmt.pct(b.hist[N].ok, 1)}` }] });
      clear(tiles);
      tiles.append(tile(fmt.pct(a.hist[N].id, 1), "proxy reward, trained on the proxy"), tile(fmt.pct(a.hist[N].ok, 1), "truly correct, trained on the proxy"), tile(fmt.pct(b.hist[N].ok, 1), "truly correct, trained on the real check"));
    }
    el.append(host, h("div", { class: "legend" },
      h("span", { class: "key" }, h("span", { class: "sw sw-s2" }), "proxy reward (exactly one row), trained on the proxy"),
      h("span", { class: "key" }, h("span", { class: "sw sw-s1" }), "truly correct, trained on the proxy"),
      h("span", { class: "key" }, h("span", { class: "sw sw-s3" }), "truly correct, trained on the real check (dashed)")),
      tiles, h("div", { class: "f-ctl" }, segmented({ label: "Random seed", value: seed, options: [5, 6, 7, 8].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { seed = v; draw(); } })));
    draw();
  });

  Deck.part({ n: 9, title: "Post-training", short: "Post-training" });

  Deck.chapter({ title: "From base model to assistant" });

  Deck.slide({
    id: "post-part",
    layout: "part",
    title: "Post-training",
    body: String.raw`<p>Pretraining produces a model that continues text. Post-training turns it into an assistant and a reasoner, in three kinds of stage: supervised fine-tuning on example answers, preference training from comparisons between answers, and reinforcement learning against programs that check answers.</p>
<p>Each stage has its own loss, which you will compute by hand on small examples and on our model. The last chapters cover the systems that run reinforcement learning on language models and the ways post-trained models are evaluated.</p>`,
    take: "Write and evaluate the losses of SFT, reward modeling, PPO, DPO and GRPO, size the data and memory each stage needs, and explain how RL training and evaluation are run at scale.",
  });

  Deck.slide({
    id: "post-base",
    title: "A base model continues text; it does not follow requests",
    example: "tokens prefix:5",
    body: String.raw`<p>A <b>base model</b> is the model straight out of pretraining. It has learned the distribution of its training text, so given a prompt it writes what usually comes next. A request is just more text to continue.</p>
<p>Our model cannot read English, so give it the most helpful prompt there is for the request <i>the name of user 7</i>: the first five tokens of the answer, <code>SELECT name FROM users WHERE</code>. It completes the query the way its training data did. It picks <code>id</code> with probability 0.346 and then the digit 7 with probability 0.108, so it writes the asked-for query with probability 0.346 × 0.108 = 0.037, about one sample in 27. Almost every other completion is a valid query that answers a different question.</p>
<p>Large base models behave the same way with requests in prose. The InstructGPT paper calls next-token prediction a different objective from "follow the user's instructions helpfully and safely".</p>`,
    fig: { type: "post-continue", caption: "Samples of 4 tokens from the trained model after the prompt; each square is one sample (green: the asked-for query)", prompt: "SELECT name FROM users WHERE", answer: " id = 7;", seed: 9, start: 30 },
    take: "Pretraining teaches what text looks like. Nothing in the loss connects a request to its answer, so post-training has to add that.",
    terms: [{ term: "Base model", def: "A model after pretraining only. It continues text with what usually follows in its training data; it has not been trained to answer requests." }],
    src: ["Ouyang et al., Training language models to follow instructions with human feedback (InstructGPT), 2022"],
  });

  Deck.slide({
    id: "post-template",
    title: "A chat template turns a conversation into one token sequence",
    example: "text",
    body: String.raw`<p>A chat model still reads and writes a single sequence of tokens. A <b>chat template</b> fixes how a conversation is laid out in that sequence: where each message starts, who wrote it, and where it ends.</p>
<p>Llama 3 marks this with <b>special tokens</b>, vocabulary entries reserved for structure. Its tokenizer appends 256 of them to the ordinary vocabulary, and by default it encodes their text as ordinary characters, so a user cannot type one. Each message is <code>&lt;|start_header_id|&gt;</code>, the role, <code>&lt;|end_header_id|&gt;</code>, two newlines, the text, then <code>&lt;|eot_id|&gt;</code>.</p>
<p>The prompt ends with an open assistant header. The model writes the answer and then <code>&lt;|eot_id|&gt;</code>, and generation stops at that token. Supervised fine-tuning is where the model learns this layout, including where to stop.</p>`,
    fig: { type: "post-chat", caption: "Our request in the Llama 3 instruct format", system: "Answer with one SQL query.", user: "What is the name of user 7?", answer: "SELECT name FROM users WHERE id = 7;" },
    note: "Generation also stops at Llama 3's end-of-sequence token.",
    terms: [
      { term: "Chat template", def: "The fixed layout that turns a conversation (system, user and assistant messages) into one token sequence, using special tokens to mark roles and message ends." },
      { term: "Special token", def: "A vocabulary entry reserved for structure, such as Llama 3's <code>&lt;|eot_id|&gt;</code>. Ordinary text is not encoded to it." },
    ],
    src: ["Meta, Llama 3 model card and prompt formats (llama.com)", "Meta, llama3 repository, llama/tokenizer.py", "Dubey et al., The Llama 3 Herd of Models, 2024"],
  });

  Deck.slide({
    id: "post-pipeline",
    title: "The pipeline: fine-tune on answers, train on preferences, then reinforce",
    layout: "wide",
    body: String.raw`<p>Each recipe below starts from a base model and combines three kinds of stage. Supervised fine-tuning trains on target answers. Preference training uses judgments, from people or from models, of which of two answers is better. Reinforcement learning samples answers from the model and updates it on the reward they earn.</p>
<p>InstructGPT and Llama 3 fine-tune first, so the answers being compared already have the right form. DeepSeek-R1-Zero ran RL directly on the base model; its answers were hard to read and mixed languages, so DeepSeek-R1 added SFT back: thousands of examples before its first RL stage, and about 800,000 before its second.</p>
<p>The data is small next to pretraining: tens of thousands of prompts for InstructGPT and about 800,000 examples for DeepSeek-R1, where Llama 3 pretrained on 15.6 trillion tokens.</p>`,
    fig: {
      type: "table", caption: "Three published post-training recipes, stage by stage", dense: true, minWidth: 640,
      cols: ["Stage", "InstructGPT (2022)", "Llama 3 (2024)", "DeepSeek-R1 (2025)"],
      rows: [
        ["Supervised fine-tuning", "12,725 prompts with answers written by about 40 labelers; 16 epochs", "Model answers picked by a reward model (best of 10 to 30 samples) plus synthetic data; learning rate 10⁻⁵, 8.5K–9K steps", "Thousands of long reasoning examples first; later about 800K examples (600K reasoning, 200K other)"],
        ["Preferences", "Reward model of 6B parameters on 33,207 prompts, each with 4 to 9 ranked answers", "Reward model; DPO with β = 0.1 on the latest preference data", "Reward models for helpfulness and harmlessness in the last RL stage"],
        ["Reinforcement learning", "PPO on 31,144 prompts, KL coefficient 0.02", "Not used: DPO needed less compute than PPO and did better on instruction following", "GRPO with rule-based rewards: answers checked by rules, code run against tests"],
        ["Order", "SFT → reward model → PPO, once", "Six rounds of reward model → rejection sampling → SFT → DPO", "SFT → RL → SFT → RL"],
      ],
    },
    src: ["Ouyang et al., InstructGPT, 2022 (Table 6, §3.5, App. C)", "Dubey et al., The Llama 3 Herd of Models, 2024 (§4)", "DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (§2–3, App. B)"],
  });

  Deck.slide({
    id: "post-stages",
    title: "Each stage changes the model in a different way",
    layout: "wide",
    body: String.raw`<p>The stages differ in what they train on and in what the loss rewards, so they change different things. Pretraining sets what the model knows and how text is distributed. The later stages change which of its possible continuations it writes. The LIMA authors fine-tuned a 65B model on 1,000 curated examples and concluded that almost all knowledge is learned in pretraining.</p>
<p>Post-training is also cheap by comparison. DeepSeek-V3 spent 5K of its 2,788K H800 GPU hours on post-training, 0.18%. Reasoning models spend more: DeepSeek-R1 took 147K, 101K of them for R1-Zero, a pure RL run.</p>`,
    fig: {
      type: "table", caption: "What each stage trains on, its loss, and what it changes", dense: true, minWidth: 640,
      cols: ["Stage", "Trains on", "Loss", "Changes"],
      rows: [
        ["Pretraining (Parts 4–7)", "Raw text: 15.6T tokens for Llama 3", "Next-token cross-entropy on every token", "Knowledge, grammar, the distribution of text"],
        ["Supervised fine-tuning", "Prompts with one target answer: 10³ to 10⁷ examples", "The same cross-entropy, on answer tokens only", "Format, turns, where to stop"],
        ["Reward model", "Prompts with two or more ranked answers", String.raw`\(-\ln \sigma(r_w - r_l)\)`, "Nothing in the policy: it trains a separate scorer"],
        ["PPO or DPO", "Prompts (PPO samples answers) or chosen–rejected pairs (DPO)", "Reward with a KL penalty, or the DPO loss", "Which of the plausible answers the model prefers"],
        ["RL with checkers", "Prompts whose answers a program can check", "Policy gradient with group-relative advantages (GRPO)", "How the model reaches an answer, including how long it reasons"],
      ],
    },
    check: {
      q: "You have 50,000 prompts, each with two stored answers labelled better and worse. Which method trains on them without generating any new answers during training?",
      options: ["Supervised fine-tuning on both answers", "DPO", "PPO with a reward model", "GRPO"],
      answer: 1,
      why: "DPO's loss needs only the log-probabilities of the two stored answers under the model being trained and under a frozen reference. PPO and GRPO sample new answers from the current model at every step. SFT on both answers would teach the worse one too.",
    },
    src: ["Zhou et al., LIMA: Less Is More for Alignment, 2023", "DeepSeek-AI, DeepSeek-V3 Technical Report, 2024 (Table 1)", "DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (Table 7)", "Dubey et al., The Llama 3 Herd of Models, 2024"],
  });

  Deck.chapter({ title: "Supervised fine-tuning" });

  Deck.slide({
    id: "post-sft-data",
    title: "SFT data: prompts paired with the answer to imitate",
    body: String.raw`<p>Each <b>supervised fine-tuning</b> (SFT) example is a prompt and one target answer, laid out with the chat template. The model learns to write answers like the targets, so the targets are chosen for quality more than for number.</p>
<p>InstructGPT's labelers wrote the answers to 12,725 prompts. LIMA fine-tuned a 65B model on 1,000 examples, about 750,000 tokens: 400 from Stack Exchange, 200 from wikiHow, 150 from Reddit's WritingPrompts, 50 from Super-Natural Instructions and 200 written by the authors. At the other end, Llama 3 generated over 2.7 million synthetic examples for code alone. Most of Llama 3's targets are model-generated: for each prompt the current model writes 10 to 30 answers and a reward model keeps the best one, which is called <b>rejection sampling</b>. DeepSeek-R1 kept only sampled reasoning traces whose final answer was correct.</p>
<p>For InstructGPT, validation loss was a poor guide: its SFT model overfit validation loss after 1 epoch, yet 16 epochs gave better reward-model scores and human ratings.</p>`,
    fig: {
      type: "table", caption: "Four SFT datasets", dense: true, minWidth: 560,
      cols: ["Model", "Examples", "Who wrote the answers", "Training"],
      rows: [
        ["InstructGPT (2022)", "12,725 prompts", "Labelers", "16 epochs, cosine learning-rate decay"],
        ["LIMA (2023)", "1,000 (≈750K tokens)", "Q&A sites (750), Super-Natural Instructions (50), the authors (200)", "15 epochs"],
        ["Llama 3 (2024)", "Not stated; 846 tokens each on average", "The model, best of 10 to 30 by a reward model; synthetic data", "Learning rate 10⁻⁵, 8.5K–9K steps"],
        ["DeepSeek-R1 (2025)", "≈800K", "The model, kept if correct; 200K from DeepSeek-V3's SFT data", "2 to 3 epochs"],
      ],
    },
    terms: [
      { term: "Supervised fine-tuning (SFT)", def: "Further training of a pretrained model with the next-token loss on prompt–answer pairs, counting only the answer tokens." },
      { term: "Rejection sampling", def: "Sampling several answers per prompt from the current model and keeping the best one according to a reward model or a checker, to use as a training target." },
    ],
    src: ["Ouyang et al., InstructGPT, 2022 (§3.5, Table 6)", "Zhou et al., LIMA, 2023 (Table 1, §2)", "Dubey et al., The Llama 3 Herd of Models, 2024 (§4.1.3, §4.2.2, §4.3.1, Table 7)", "DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (App. B.3–B.4)"],
  });

  Deck.slide({
    id: "post-sft-mask",
    title: "The SFT loss counts only the answer tokens",
    example: "tokens ids",
    body: String.raw`<p>SFT keeps the pretraining loss, the cross-entropy of each next token, and changes only which positions count. The prompt is context. Each target token gets a mask value \(m_t\): 1 if it belongs to the answer, 0 if it belongs to the prompt.</p>
<p>In the plate, the conversation is 18 tokens, so 17 next-token predictions. Ten are trained: the 9 tokens of our query and the end marker, which teaches the model to stop. Llama 3 trains its SFT stage this way, masking the loss on prompt tokens.</p>
<p>The average needs one more decision. Averaging over all answer tokens in a batch gives a 100-token answer 91% of the weight next to a 10-token one; averaging within each example first gives each 50%.</p>`,
    math: String.raw`L_{\text{SFT}} = -\frac{1}{\sum_t m_t} \sum_{t} m_t \, \ln p_\theta(x_{t+1} \mid x_{\le t})`,
    fig: { type: "tinylm", widget: "sft", caption: "One SFT example; switch the loss mask to see which predictions are trained (the marker and English tokens are illustrative)" },
    note: "The plate's marker tokens are made up. The chat-template sheet in 9.1 shows Llama 3's real ones.",
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (§4.1.3)"],
  });

  Deck.slide({
    id: "post-packing",
    title: "Packing fills the context; a per-example mask keeps examples apart",
    body: String.raw`<p>SFT examples vary in length. Padding each one to the longest wastes the padded slots: 8 of 24 for the default lengths 5, 3 and 8. <b>Packing</b> concatenates the examples into one row instead.</p>
<p>The plain causal mask then lets a token of the third example attend to all of the first two. A <b>block-diagonal mask</b> allows attention only inside an example. Here it keeps 15 + 6 + 36 = 57 query–key pairs of the 136 the causal mask allows, exactly what three separate sequences would compute. The loss mask must also drop the one target that crosses a boundary: the next example's first token is not a continuation of this one.</p>
<p>Llama 3 used such a mask in pretraining. It had little effect at standard lengths and mattered in continued pretraining on very long sequences.</p>`,
    fig: { type: "post-pack", caption: "Three examples padded, packed with the causal mask, and packed with a per-example mask", lengths: [5, 3, 8], mode: "doc" },
    terms: [
      { term: "Packing", def: "Concatenating several training examples into one sequence of the context length instead of padding each one, so no slots are wasted." },
      { term: "Block-diagonal mask", def: "An attention mask that allows a position to attend only to earlier positions of the same packed example." },
    ],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (§3.2)"],
  });

  Deck.slide({
    id: "post-lora",
    title: "LoRA freezes the weights and trains a low-rank correction",
    body: String.raw`<p><b>LoRA</b> (low-rank adaptation) freezes a pretrained matrix \(W_0\) of shape d × k and learns \(\Delta W = BA\), with B of shape d × r and A of shape r × k for a small rank r. The layer computes \(W_0 x + \tfrac{\alpha}{r} BAx\).</p>
<p>The trainable count drops from dk to r(d + k). For d = k = 4,096 and r = 16: 16,777,216 becomes 16 × 8,192 = 131,072, which is 0.78%. On all 224 linear layers of Llama 3 8B, r = 16 trains 41.9M parameters, 0.52% of 8.03B. Only these need gradients and Adam state: for GPT-3 175B, LoRA cut training memory from 1.2 TB to 350 GB.</p>
<p>B starts at zero, so training starts from the unchanged model. Afterwards \(W_0 + \tfrac{\alpha}{r}BA\) is one matrix of the original shape, so inference costs nothing extra.</p>`,
    math: String.raw`\underbrace{dk}_{W_0} \;\to\; \underbrace{r(d + k)}_{B,\ A}`,
    fig: {
      type: "post-lora", caption: "A frozen matrix and its two trainable factors, to scale", r: 16, preset: 1,
      presets: [{ label: "Our W_q 16 × 16", d: 16, k: 16 }, { label: "Llama 3 8B W_q", d: 4096, k: 4096 }, { label: "W_k 1,024 × 4,096", d: 1024, k: 4096 }, { label: "MLP up 14,336 × 4,096", d: 14336, k: 4096 }],
    },
    terms: [{ term: "LoRA", def: "Low-rank adaptation: training a correction BA of rank r to a frozen weight matrix, with r(d + k) trainable numbers instead of dk." }],
    src: ["Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021 (§4.1, §4.2)", "Dubey et al., The Llama 3 Herd of Models, 2024 (Table 3)"],
  });

  Deck.slide({
    id: "post-qlora",
    title: "QLoRA stores the frozen weights in 4 bits",
    body: String.raw`<p>With LoRA, the frozen weights take most of the memory. <b>QLoRA</b> stores them in NF4, a 4-bit format whose 16 levels are placed for normally distributed weights, and converts each block to BF16 when it is used. The adapters stay in BF16.</p>
<p>Every block of 64 weights needs a scale. In FP32 that adds 32/64 = 0.5 bits per weight. QLoRA quantizes the scales too, to 8 bits in blocks of 256: 8/64 + 32/(64 × 256) = 0.127 bits, a saving of 0.373 bits per weight, about 3 GB on a 65B model.</p>
<p>For 65B parameters, full fine-tuning at 16 bytes each needs 1,040 GB, 13 GPUs of 80 GB. (QLoRA's paper says more than 780 GB, which is 12 bytes per parameter.) The 4-bit weights take 65 × 10⁹ × 4.127 / 8 bytes = 33.5 GB. QLoRA's memory breakdown puts the whole 65B fine-tuning at 45.0 GB for batch size 1, 512 tokens and gradient checkpointing, with activation gradients estimated without attention: under one 48 GB GPU.</p>`,
    fig: {
      type: "calc", caption: "Memory for weights and trainable state (activations excluded)",
      inputs: [
        { id: "N", label: "parameters", min: 1e9, max: 4.05e11, value: 6.5e10, log: true, step: "any", fmt: (v) => `${(v / 1e9).toFixed(v < 1e10 ? 1 : 0)}B` },
        { id: "s", label: "trainable share (LoRA)", min: 0.001, max: 0.05, step: 0.001, value: 0.01, fmt: (v) => `${(100 * v).toFixed(1)}%` },
      ],
      outputs: [
        { label: "full fine-tuning, 16 B per parameter", f: (v) => 16 * v.N, fmt: (b) => gb(b) },
        { label: "LoRA, BF16 base + adapters", f: (v) => 2 * v.N + 16 * v.s * v.N, fmt: (b) => gb(b) },
        { label: "QLoRA, NF4 base + adapters", f: (v) => (4.127 / 8) * v.N + 16 * v.s * v.N, fmt: (b) => gb(b) },
      ],
      note: (v) => `Base weights: ${gb(2 * v.N)} in BF16, ${gb((4.127 / 8) * v.N)} in NF4 with double-quantized scales. Adapters: ${int(v.s * v.N / 1e6)}M trainable parameters at 16 bytes each (weight, gradient, FP32 copy, two Adam moments) = ${gb(16 * v.s * v.N)}.`,
    },
    terms: [{ term: "QLoRA", def: "LoRA on a base model whose frozen weights are stored in 4-bit NF4 with quantized scales, dequantized to BF16 block by block when used." }],
    src: ["Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs, 2023 (§1, §3, Fig. 6, App. G)"],
  });

  Deck.slide({
    id: "post-forget",
    title: "Fine-tuning on narrow data erases knowledge; mixing in old data keeps it",
    body: String.raw`<p>Fine-tuning moves weights that also encode things the new data never mentions, so the model can lose them: <b>forgetting</b>. We measured it on our model in PyTorch, with 300 steps of Adam (learning rate 10⁻³, 128 queries per step) on <code>orders</code> queries only.</p>
<p>Loss on orders queries fell from 0.725 to 0.586 nats per token. Loss on users queries rose from 0.721 to 2.916, and on products queries from 0.637 to 2.841. After <code>SELECT name FROM</code> the model now gives <code>users</code> probability 9.3 × 10⁻⁷, though only users has a <code>name</code> column.</p>
<p>Drawing a quarter of each batch from the original grammar instead keeps users at 0.888 and products at 0.801, keeps p(users) at 1.000, and still brings orders down to 0.610. InstructGPT did the same during PPO: 8 times as many pretraining examples as RL episodes, gradient weight γ = 27.8, which reduced its regressions on public benchmarks.</p>`,
    fig: {
      type: "table", caption: "Our model after 300 steps of fine-tuning (loss in nats per token, exact over the grammar)", align: ["l", "r", "r", "r", "r"],
      cols: ["Model", "Orders", "Users", "Products", "p(users | SELECT name FROM)"],
      rows: [
        ["Before fine-tuning", "0.725", "0.721", "0.637", "1.000"],
        ["Orders only", "<b>0.586</b>", "<b>2.916</b>", "<b>2.841</b>", "<b>9.3 × 10⁻⁷</b>"],
        ["Orders + 25% original", "0.610", "0.888", "0.801", "1.000"],
      ],
      highlight: [1],
      note: "Run: <code>tools/verify/part09_experiments.py</code>, seeded and single-threaded.",
    },
    check: {
      q: "After fine-tuning on orders queries only, the model gives <code>users</code> probability 9.3 × 10⁻⁷ after <code>SELECT name FROM</code>, though no training query contained <code>name</code>. Why?",
      options: ["Every training query had orders after FROM, and the weights that predict the table are shared by all contexts", "name is also a column of orders", "Weight decay shrank the weights that encoded users", "The learning rate was too small to keep users"],
      answer: 0,
      why: "Gradient steps change weights that every context uses. 300 steps that always rewarded <code>orders</code> after <code>FROM</code> raised it in every context, including <code>SELECT name FROM</code>, which the new data never corrected. The run used Adam without weight decay, and <code>name</code> is a column of <code>users</code> only. Replaying 25% original queries keeps those contexts in the loss.",
    },
    terms: [{ term: "Forgetting", def: "Loss of abilities a model had before fine-tuning, caused by updates on data that does not exercise them." }],
    src: ["Ouyang et al., InstructGPT, 2022 (§4.2, App. C.4)"],
  });

  Deck.chapter({ title: "Preferences" });

  Deck.slide({
    id: "post-prefs-data",
    title: "Preference data: one prompt, answers ranked by a judge",
    body: String.raw`<p>Many qualities of an answer are easier to judge than to write down. Preference data records judgments: for one prompt, two or more answers and which one a person (or a model) prefers. In each <b>preference pair</b> the better answer is <b>chosen</b> \(y_w\), the other <b>rejected</b> \(y_l\).</p>
<p>From here on the request is <i>the email of user 7</i>, whose right answer is <code>SELECT email FROM users WHERE id = 7;</code>. Ranking four answers gives \(\binom{4}{2} = 6\) pairs (plate). InstructGPT's labelers ranked 4 to 9 answers per prompt, up to 36 pairs, and all pairs of one prompt went into one batch element: as separate examples, the reward model overfit within one epoch.</p>
<p>Llama 3 compares two answers from two different models. Annotators grade the preference as significantly better, better, slightly better or marginally better, and only the first two grades are used for training.</p>`,
    fig: {
      type: "table", caption: "A ranking of four answers to “the email of user 7” and the six pairs it yields", dense: true,
      cols: ["Pair", "Chosen \\(y_w\\)", "Rejected \\(y_l\\)"],
      rows: [
        ["1", "A <code>… WHERE id = 7;</code> (right)", "B <code>SELECT name … id = 7;</code> (wrong column)"],
        ["2", "A", "C <code>… WHERE id = 3;</code> (wrong user)"],
        ["3", "A", "D <code>… WHERE age = 7;</code> (no rows)"],
        ["4", "B", "C"], ["5", "B", "D"], ["6", "C", "D"],
      ],
      note: "Ranking A ≻ B ≻ C ≻ D, as a labeler might judge them. <code>…</code> stands for <code>SELECT email FROM users WHERE</code>.",
    },
    terms: [{ term: "Preference pair", def: "A prompt with two answers, one labelled chosen (preferred, \\(y_w\\)) and one rejected (\\(y_l\\)), the unit of data for reward models and DPO." }],
    src: ["Ouyang et al., InstructGPT, 2022 (§3.5)", "Dubey et al., The Llama 3 Herd of Models, 2024 (§4.2.1)"],
  });

  Deck.slide({
    id: "post-bt",
    title: "Bradley–Terry: the probability of a preference is σ of a reward difference",
    body: String.raw`<p>To learn from pairs, model the probability that A is preferred to B. Bradley and Terry (1952) gave each item a positive strength \(s\) and set \(p(A \succ B) = s_A / (s_A + s_B)\). With \(s = e^{r}\) this becomes the sigmoid of a reward difference.</p>
<p>Worked: \(r_A = 1.2\) and \(r_B = -0.3\) give \(\Delta = 1.5\), so \(p(A \succ B) = \sigma(1.5) = 0.818\): odds of \(e^{1.5} = 4.48\) to 1.</p>
<p>Only differences matter. \(r_A = 11.2\) and \(r_B = 9.7\) give the same 0.818, so a reward has no natural zero; before RL, InstructGPT shifted its reward model so that labeler-written answers scored 0 on average. A difference of 0 means a coin flip, and each further unit of difference multiplies the odds by e = 2.718.</p>`,
    math: String.raw`p(A \succ B) = \frac{e^{r_A}}{e^{r_A} + e^{r_B}} = \frac{1}{1 + e^{-(r_A - r_B)}} = \sigma(r_A - r_B)`,
    fig: {
      type: "plot", caption: "Probability that A is preferred, as a function of the reward difference", x: [-6, 6], y: [0, 1], xlabel: "reward difference, A minus B", ylabel: "probability A wins",
      params: [{ name: "d", label: "reward difference", min: -6, max: 6, step: 0.1, value: 1.5, fmt: "f1" }],
      fns: [{ f: (x) => sig(x), cls: "s1" }],
      hlines: [{ y: 0.5, label: "coin flip" }],
      extra: (p) => ({ dots: [{ x: p.d, y: sig(p.d), r: 6, tip: `σ(${fmt.minus(p.d.toFixed(1))}) = ${sig(p.d).toFixed(4)}` }] }),
      readouts: [
        { label: String.raw`\(p(A \succ B) = \sigma(\Delta)\)`, f: (p) => sig(p.d), fmt: "f3" },
        { label: "odds, A to B", f: (p) => Math.exp(p.d), fmt: odds },
        { label: String.raw`loss \(-\ln \sigma(\Delta)\)`, f: (p) => -Math.log(sig(p.d)), fmt: "f3" },
      ],
    },
    terms: [{ term: "Bradley–Terry model", def: "A model of paired comparisons: \\(p(A \\succ B) = s_A/(s_A + s_B)\\); with strengths \\(s = e^r\\) it is \\(\\sigma(r_A - r_B)\\)." }],
    src: ["Bradley and Terry, Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons, Biometrika 39, 1952", "Ouyang et al., InstructGPT, 2022 (§3.5)"],
  });

  Deck.slide({
    id: "post-rm",
    title: "A reward model learns to score the chosen answer higher",
    body: String.raw`<p>A <b>reward model</b> is a language model with its output layer replaced by one linear unit: it reads the prompt and an answer and returns one number. InstructGPT built its 6B reward model from a language model with the unembedding removed.</p>
<p>It is trained to make the observed preferences likely under Bradley–Terry, so its loss on a pair is \(-\ln \sigma(r_w - r_l)\). The derivative with respect to \(\Delta = r_w - r_l\) is \(-\sigma(-\Delta) = -(1 - p)\): a pair already ranked with confidence gets almost no update, a pair ranked the wrong way gets up to 1.</p>
<p>The plate works four pairs. The mean loss is 0.652; the pair the model gets wrong (\(\Delta = -1\)) has the largest loss, 1.313, and the largest gradient, −0.731. Reward models are imperfect judges: Stiennon et al. found that doubling the comparisons raised accuracy by about 1.1%, doubling the model by about 1.8%.</p>`,
    math: [String.raw`L_{\text{RM}} = -\ln \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)`, String.raw`\frac{\partial L_{\text{RM}}}{\partial \Delta} = -\sigma(-\Delta) = -(1 - p)`],
    fig: {
      type: "table", caption: "Reward-model loss on four pairs", align: ["l", "r", "r", "r", "r"],
      cols: ["Pair", String.raw`\(\Delta = r_w - r_l\)`, String.raw`\(p = \sigma(\Delta)\)`, String.raw`loss \(-\ln p\)`, String.raw`\(\partial L / \partial \Delta\)`],
      rows: [["1", "2.0", "0.881", "0.127", "−0.119"], ["2", "0.5", "0.622", "0.474", "−0.378"], ["3", "0.0", "0.500", "0.693", "−0.500"], ["4", "−1.0", "0.269", "1.313", "−0.731"], ["mean", "", "", "<b>0.652</b>", ""]],
      highlight: [3],
      note: "Pair 4 is ranked the wrong way round: the reward model scores the rejected answer higher.",
    },
    terms: [{ term: "Reward model", def: "A network that maps a prompt and an answer to one number, trained on preference pairs with the loss \\(-\\ln\\sigma(r_w - r_l)\\)." }],
    src: ["Ouyang et al., InstructGPT, 2022 (§3.5, Eq. 1)", "Stiennon et al., Learning to summarize from human feedback, 2020 (§3.4, §4.3)"],
  });

  Deck.slide({
    id: "post-rlhf",
    title: "RLHF with PPO keeps four models in memory",
    body: String.raw`<p><b>RLHF</b> (reinforcement learning from human feedback) trains the <b>policy</b>, the model that writes answers, to earn reward. Each step samples answers and scores them with the frozen reward model. From the score it subtracts β times the log-ratio of each generated token against a frozen <b>reference</b> copy of the SFT model; averaged over the policy's own samples, that is β times their KL divergence. A trained <b>value model</b> predicts the reward still to come. In its simplest form, token t's advantage is the reward collected from token t to the end of the answer minus the value model's prediction \(V(x, y_{\lt t})\); PPO (next sheet) weights each token's update by it.</p>
<p>If the policy doubles the probability of each of 20 answer tokens, the log-ratios sum to \(20\, \ln 2 = 13.86\) nats, which costs 0.277 reward at InstructGPT's β = 0.02. With all four models at 7B parameters, the two trained ones need 7 × 10⁹ × 16 bytes = 112 GB each and the two frozen ones 14 GB each in BF16: 252 GB before activations, at least 4 GPUs of 80 GB. InstructGPT paired a 175B policy with a 6B reward model and a 6B value model.</p>`,
    math: String.raw`R(x, y) = r_\phi(x, y) - \beta \sum_t \ln \frac{\pi_\theta(y_t \mid x, y_{\lt t})}{\pi_{\text{ref}}(y_t \mid x, y_{\lt t})}`,
    fig: {
      type: "diagram", caption: "One RLHF step: what each model contributes", w: 14, h: 7.2, unit: 40,
      nodes: [
        { id: "x", x: 0.1, y: 3.7, w: 1.7, h: 1.4, label: "Prompt x" },
        { id: "pol", x: 2.3, y: 3.7, w: 2.6, h: 1.4, label: "Policy", sub: "trained, from SFT", kind: "blue" },
        { id: "rm", x: 7.6, y: 0.2, w: 2.6, h: 1.3, label: "Reward model", sub: "frozen" },
        { id: "ref", x: 7.6, y: 2.0, w: 2.6, h: 1.3, label: "Reference", sub: "frozen SFT copy" },
        { id: "val", x: 7.6, y: 5.7, w: 2.6, h: 1.3, label: "Value model", sub: "trained, from RM", kind: "blue" },
        { id: "ppo", x: 12.0, y: 3.7, w: 1.9, h: 1.4, label: "PPO step", kind: "dark" },
      ],
      edges: [
        { from: "x", to: "pol" },
        { from: "pol", to: "rm", label: "answer y", dx: -34, dy: -2 },
        { from: "pol", to: "ref", label: "answer y", dx: 6, dy: -4 },
        { from: "pol", to: "val", label: "answer y", dx: -30, dy: 18 },
        { from: "pol", to: "ppo", label: "log-probs" },
        { from: "rm", to: "ppo", label: "reward", dx: 30, dy: -2 },
        { from: "ref", to: "ppo", label: "KL", dx: 4, dy: -2 },
        { from: "val", to: "ppo", label: "values", dx: 28, dy: 18 },
      ],
      notes: [{ x: 11.0, y: 6.35, text: "updates policy" }, { x: 11.0, y: 6.85, text: "and value model" }],
    },
    terms: [
      { term: "RLHF", def: "Reinforcement learning from human feedback: optimizing a policy against a reward model trained on human preferences, with a KL penalty to a reference model." },
      { term: "Policy", def: "In reinforcement learning, the model that chooses actions; here, the language model that writes answers." },
      { term: "Value model", def: "A network that predicts the expected future reward from a partial answer; PPO uses it as a baseline to compute advantages." },
    ],
    src: ["Ouyang et al., InstructGPT, 2022 (§3.5, Eq. 2, App. C.4)", "Stiennon et al., Learning to summarize from human feedback, 2020 (§3.4)"],
  });

  Deck.slide({
    id: "post-ppo",
    title: "PPO clips the probability ratio so one batch can be reused safely",
    body: String.raw`<p>PPO (proximal policy optimization) reuses each batch of samples for several gradient steps. The objective weights each token's <b>advantage</b> A (how much better than expected its outcome was) by the <b>probability ratio</b> \(\rho = \pi_\theta / \pi_{\text{old}}\) between the current policy and the one that sampled it.</p>
<p>Unclipped, \(\rho A\) would reward pushing ρ without limit. PPO takes the smaller of \(\rho A\) and the same term with ρ clipped to \([1 - \varepsilon, 1 + \varepsilon]\). Worked with ε = 0.2: a token with \(\ln \pi_\theta = -0.9\), \(\ln \pi_{\text{old}} = -1.3\) has \(\rho = e^{0.4} = 1.492\). With A = 2, the objective is min(2.984, 1.2 × 2) = 2.4 and its gradient is 0: no further push. With A = −1, it is min(−1.492, −1.2) = −1.492, unclipped: the penalty for having moved toward a bad token stays in full. At ρ = 1, \(\nabla_\theta(\rho A) = A\, \nabla_\theta \ln \pi_\theta\): the policy-gradient step that 9.4 derives, weighted by the advantage.</p>
<p>On PPO's continuous-control benchmark (3 seeds on each of 7 environments, 21 runs per setting), ε = 0.2 had the best average normalized score: 0.82, against 0.76 for 0.1, 0.70 for 0.3 and −0.39 without clipping.</p>`,
    math: String.raw`L^{\text{CLIP}} = \mathbb{E}_t\big[\min\big(\rho_t A_t,\ \operatorname{clip}(\rho_t, 1 - \varepsilon, 1 + \varepsilon) A_t\big)\big]`,
    fig: {
      type: "plot", caption: "PPO's objective for one token as a function of the ratio (dashed: without the clip)", x: [0, 2.5], y: [-5, 5], xlabel: "probability ratio, new over old", ylabel: "objective",
      params: [
        { name: "A", label: "advantage", min: -2, max: 2, step: 0.1, value: 2, fmt: "f1" },
        { name: "eps", label: "clip range", min: 0.05, max: 0.5, step: 0.01, value: 0.2, fmt: "f2" },
        { name: "rho", label: "ratio", min: 0, max: 2.5, step: 0.01, value: 1.49, fmt: "f2" },
      ],
      fns: [{ f: (x, p) => x * p.A, cls: "faint dashed" }, { f: (x, p) => ppoObj(x, p.A, p.eps), cls: "s2" }],
      extra: (p) => ({ hlines: [{ y: 0 }], vlines: [{ x: 1 - p.eps, label: "1 − ε" }, { x: 1 + p.eps, label: "1 + ε" }], dots: [{ x: p.rho, y: ppoObj(p.rho, p.A, p.eps), r: 6, tip: `objective ${ppoObj(p.rho, p.A, p.eps).toFixed(3)}` }] }),
      readouts: [
        { label: "objective at this ratio", f: (p) => ppoObj(p.rho, p.A, p.eps), fmt: "f3" },
        { label: "gradient with respect to the ratio", f: (p) => ppoGrad(p.rho, p.A, p.eps), fmt: "f1" },
        { label: "clip active", f: (p) => (ppoGrad(p.rho, p.A, p.eps) === 0 && p.A !== 0 ? 1 : 0), fmt: (v) => (v ? "yes, no gradient" : "no") },
      ],
    },
    terms: [
      { term: "Advantage", def: "How much better an action turned out than the policy expected: its return minus a baseline. Positive advantages make the action more likely." },
      { term: "Probability ratio", def: "In PPO, \\(\\rho = \\pi_\\theta(a)/\\pi_{\\text{old}}(a)\\): how much more likely the current policy makes an action than the policy that sampled it." },
    ],
    src: ["Schulman et al., Proximal Policy Optimization Algorithms, 2017 (Eq. 7, Table 1, Algorithm 1)"],
  });

  Deck.slide({
    id: "post-dpo",
    title: "DPO turns the reward-model loss into a loss on the policy",
    body: String.raw`<p><b>DPO</b> (direct preference optimization) trains the policy on preference pairs directly: no reward model, and no sampling during training. The derivation (plate) starts from the objective RLHF optimizes. Its optimal policy fixes the reward up to a constant, so the reward can be written as β times a log-ratio between the policy and the reference. Put into Bradley–Terry, the constant cancels.</p>
<p>On our model, with \(y_w\) = <code>id = 7;</code> and \(y_l\) = <code>age = 7;</code> after <code>SELECT email FROM users WHERE</code>: \(\ln \pi_{\text{ref}}(y_w) = -3.273\) and \(\ln \pi_{\text{ref}}(y_l) = -3.323\). Training starts at \(\pi_\theta = \pi_{\text{ref}}\), so both log-ratios are 0 and the loss is \(\ln 2 = 0.693\) for every pair. Its gradient raises \(\ln \pi_\theta(y_w)\) and lowers \(\ln \pi_\theta(y_l)\) with weight \(\beta \sigma(0) = \beta / 2\), 0.05 at β = 0.1.</p>`,
    math: String.raw`L_{\text{DPO}} = -\ln \sigma\Big(\beta \ln \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \ln \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)`,
    fig: {
      type: "steps", caption: "From the RLHF objective to the DPO loss", start: 0,
      steps: [
        { html: String.raw`\[\max_\pi\ \mathbb{E}_{y \sim \pi}\big[r(x, y)\big] - \beta\, \mathrm{KL}\big(\pi \,\|\, \pi_{\text{ref}}\big)\]`, note: "the objective RLHF optimizes" },
        { html: String.raw`\[\begin{gathered} = -\beta\, \mathbb{E}_{y \sim \pi}\Big[\ln \frac{\pi(y)}{\pi_{\text{ref}}(y)\, e^{r(y)/\beta} / Z}\Big] \\ +\, \beta \ln Z = -\beta\, \mathrm{KL}(\pi \,\|\, \pi^*) + \beta \ln Z \end{gathered}\]`, note: String.raw`\(Z\) sums \(\pi_{\text{ref}}\, e^{r/\beta}\) over all answers` },
        { html: String.raw`\[\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, e^{r(x, y)/\beta}\]`, note: "a KL is 0 only at π = π*" },
        { html: String.raw`\[r(x, y) = \beta \ln \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \ln Z(x)\]`, note: "solve for the reward" },
        { html: String.raw`\[\begin{gathered}\sigma\big(r(x, y_w) - r(x, y_l)\big) \\ = \sigma\Big(\beta \ln \frac{\pi^*(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \ln \frac{\pi^*(y_l)}{\pi_{\text{ref}}(y_l)}\Big)\end{gathered}\]`, note: "Bradley–Terry: β ln Z cancels" },
        { html: String.raw`\[\begin{gathered} L_{\text{DPO}} = -\ln \sigma(\beta\, m), \\ m = \ln \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \ln \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\end{gathered}\]`, note: String.raw`fit \(\pi_\theta\) in place of \(\pi^*\)` },
        { html: String.raw`\[\begin{gathered}\pi_\theta = \pi_{\text{ref}} \Rightarrow m = 0 \\ L = -\ln \sigma(0) = \ln 2 = 0.693\end{gathered}\]`, note: "every pair, at the start" },
      ],
    },
    terms: [{ term: "DPO", def: "Direct preference optimization: training a policy on preference pairs with \\(-\\ln\\sigma\\) of β times the difference of the two answers' log-ratios to a frozen reference." }],
    src: ["Rafailov et al., Direct Preference Optimization: Your Language Model is Secretly a Reward Model, 2023 (Eq. 3–7)"],
  });

  Deck.slide({
    id: "post-beta",
    title: "β sets how far the trained policy may move from the reference",
    body: String.raw`<p>β is the weight of the KL penalty in the objective DPO optimizes, so a larger β keeps the policy closer to the reference. The optimum makes this exact: \(\pi^* \propto \pi_{\text{ref}}\, e^{r/\beta}\).</p>
<p>Our model writes the right query with \(\pi_{\text{ref}} = 0.038\). Give it a reward 1 higher than every other answer (people would prefer it \(\sigma(1)\) = 73% of the time). The optimum multiplies its odds by \(e^{1/\beta}\): at β = 1, 2.72× and \(\pi^*\) = 0.097; at β = 0.5, 7.39× and 0.225; at β = 0.1, 22,026× and 0.999, at a KL of 3.26 nats from the reference.</p>
<p>If people always prefer the right answer, the reward gap is unbounded and the optimum gives every other answer probability 0, whatever β is. The DPO loss never reaches 0, so on such pairs the margin keeps growing at any β (our run on the next sheet drove rejected answers to about \(e^{-37}\)); β only rescales it. The loss is 0.1 at \(\beta m = 2.25\): a margin of 22.5 nats at β = 0.1, 4.5 at β = 0.5. Llama 3 used β = 0.1; the DPO paper used 0.5 for summarization.</p>`,
    fig: {
      type: "plot", caption: "The KL-regularized optimum for the right query on our model (dashed: the reference probability, and β = 0.1 and 0.5)", x: [0, 2], y: [0, 1], xlabel: "beta", ylabel: "optimal p(right query)",
      params: [
        { name: "b", label: "beta", min: 0.02, max: 2, step: 0.01, value: 0.1, fmt: "f2" },
        { name: "g", label: "reward gap", min: 0.25, max: 3, step: 0.25, value: 1, fmt: "f2" },
      ],
      fns: [{ f: (x, p) => (x > 0 ? qStar(x, p.g, pTask()) : null), cls: "s1" }],
      extra: (p) => ({ hlines: [{ y: pTask() }], vlines: [{ x: 0.1 }, { x: 0.5 }], dots: [{ x: p.b, y: qStar(p.b, p.g, pTask()), r: 6, tip: `optimum ${qStar(p.b, p.g, pTask()).toFixed(4)}` }] }),
      readouts: [
        { label: String.raw`optimal \(\pi^*\) of the right query`, f: (p) => qStar(p.b, p.g, pTask()), fmt: "f3" },
        { label: String.raw`odds multiplier \(e^{\text{gap}/\beta}\)`, f: (p) => Math.exp(p.g / p.b), fmt: (v) => (v < 100 ? `${v.toFixed(2)}×` : v < 1e6 ? `${int(v)}×` : `${fmt.sci(v, 2)}×`) },
        { label: String.raw`\(\mathrm{KL}(\pi^* \,\|\, \pi_{\text{ref}})\), nats`, f: (p) => klTwo(qStar(p.b, p.g, pTask()), pTask()), fmt: "f3" },
        { label: "DPO margin for loss 0.1, nats", f: (p) => (-0.1 - Math.log(1 - Math.exp(-0.1))) / p.b, fmt: "f1" },
      ],
    },
    take: "Larger β, stronger pull toward the reference: the policy moves less. Small β lets a modest reward difference move the policy almost all the way.",
    src: ["Rafailov et al., DPO, 2023 (Eq. 3–4, §6, App. B)", "Dubey et al., The Llama 3 Herd of Models, 2024 (§4.1.4)"],
  });

  Deck.slide({
    id: "post-dpo-leak",
    title: "DPO on our model pushes probability out of the grammar",
    body: String.raw`<p>We ran DPO on our model in PyTorch: chosen <code>id = 7;</code>, rejected one of the 26 other grammatical answers to <code>SELECT email FROM users WHERE</code>, β = 0.1, 100 Adam steps at learning rate 10⁻³, 64 pairs per step.</p>
<p>The chosen answer rose from probability 0.038 to 0.167, and every rejected answer fell to about \(e^{-37}\). But the 27 grammatical answers together now hold only 0.167: the other 0.833 moved to strings outside the grammar, led by <code>id = 7 status</code> at 0.098. The loss depends only on the margin, and nothing holds the rest of the distribution in place. Every answer shares the tokens <code>=</code> and <code>;</code>, and pushing them down in rejected answers lowered them after <code>id = 7</code> too.</p>
<p>Llama 3 describes the same conflict for tokens common to both answers, and made two changes: it masks header and termination tokens out of the DPO loss, and adds an NLL term on the chosen answer with weight 0.2. Both help here: with <code>=</code> and <code>;</code> masked, 0.962 of the probability stays inside the grammar, and the NLL term raises that to 0.993 (plate).</p>`,
    fig: {
      type: "table", caption: "100 DPO steps on our model (β = 0.1)", dense: true, minWidth: 560, align: ["l", "r", "r", "l"],
      cols: ["Variant", "p(chosen)", "p(any grammatical answer)", "Most likely answer"],
      rows: [
        ["Before training", "0.038", "0.998", "<code>id = 3;</code> 0.040"],
        ["DPO", "0.167", "<b>0.167</b>", "<code>id = 7;</code>, then <code>id = 7 status</code> 0.098"],
        ["DPO + NLL 0.2", "0.461", "0.461", "<code>id = 7;</code>"],
        ["DPO, <code>=</code> and <code>;</code> masked", "0.529", "0.962", "<code>id = 7;</code>"],
        ["Masked + NLL 0.2", "0.577", "0.993", "<code>id = 7;</code>"],
      ],
      highlight: [1],
      note: "Run: <code>tools/verify/part09_experiments.py</code>. NLL is the mean token loss of the chosen answer.",
    },
    note: "At learning rate 10⁻⁴ the masked and unmasked runs agree to 4 decimals: while p(=) and p(;) are near 1, their gradients are near 0.",
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (§4.1.4)"],
  });

  Deck.slide({
    id: "post-ppo-dpo",
    title: "PPO or DPO: what each costs, and what each risks",
    layout: "wide",
    body: String.raw`<p>Both aim at the optimum of the same KL-regularized objective. PPO gets there with fresh samples scored by a reward model; DPO fits fixed pairs, and pins the policy down only on the answers in those pairs. The DPO paper reports a win rate against reference summaries of about 61% for DPO and 57% for PPO (judged by GPT-4). Llama 3 tried PPO and chose DPO because it needed less compute for large models and did better on instruction following.</p>
<p>Each has a typical failure. PPO can over-optimize an imperfect reward model: Stiennon et al. found human preference falling while the reward kept rising. DPO can lower the chosen answer's probability or move probability to answers in neither set, as on the previous sheet.</p>`,
    fig: {
      type: "table", caption: "PPO-based RLHF and DPO side by side", dense: true, minWidth: 560,
      cols: ["", "RLHF with PPO", "DPO"],
      rows: [
        ["Models in memory", "4: policy and value model trained; reward model and reference frozen", "2: policy trained; reference frozen"],
        ["Sampling during training", "Every step: generate answers, score them", "None: log-probabilities of stored pairs"],
        ["Data", "Prompts; preferences enter through the reward model", "Fixed chosen–rejected pairs; Llama 3 collects new pairs from its latest model every round"],
        ["Distance from the reference", String.raw`Explicit KL penalty β per token`, String.raw`Implicit: β in the loss`],
        ["Typical failure", "Reward over-optimization: the reward model stops agreeing with people", "Probability drifts to answers in neither set; fixed with NLL and masking"],
      ],
    },
    check: {
      q: String.raw`β = 0.1. After some training the policy gives the chosen answer twice its reference probability and the rejected answer half of its reference probability. What is the DPO loss on this pair?`,
      options: ["0.693", "0.626", "0.223", "0.765"],
      answer: 1,
      why: String.raw`The margin is \(\ln 2 - \ln 0.5 = \ln 4 = 1.386\), so the loss is \(-\ln\sigma(0.1 \times 1.386) = -\ln 0.535 = 0.626\). 0.693 is the loss before any training (margin 0); 0.223 forgets β; 0.765 flips the sign of the margin.`,
    },
    src: ["Rafailov et al., DPO, 2023 (§6.2)", "Dubey et al., The Llama 3 Herd of Models, 2024 (§4.1.4)", "Stiennon et al., Learning to summarize from human feedback, 2020 (§4.3)"],
  });

  Deck.chapter({ title: "RL with verifiable rewards" });

  Deck.slide({
    id: "post-verifier",
    title: "A program that checks answers can replace the reward model",
    body: String.raw`<p>For math, code and SQL, correctness can be checked by a program, so a learned reward model is not required. DeepSeek-R1-Zero used two rule-based rewards: an accuracy reward (the final math answer in a fixed format, compared with the known result; code compiled and run against test cases) and a format reward for putting the reasoning between <code>&lt;think&gt;</code> tags. Its authors avoided neural reward models for reasoning because they are susceptible to reward hacking in large-scale RL.</p>
<p>Our checker for <i>the email of user 7</i> runs the query on a 9-row <code>users</code> table and gives reward 1 only if the result is exactly <code>gus@example.com</code>. The plate lists every outcome with its probability under our model: the reward is 1 with probability 0.038, and 0.998 of the samples are grammatical queries. A <b>verifiable reward</b> like this is exact and cheap to compute, but exists only where answers can be checked.</p>`,
    fig: {
      type: "table", caption: "What the checker returns for each answer to SELECT email FROM users WHERE …", align: ["l", "l", "r", "r"],
      cols: ["Answer", "Checker sees", "Reward", "Probability"],
      rows: [
        ["<code>id = 7;</code>", "<code>gus@example.com</code>", "1", "0.038"],
        ["<code>id = d;</code>, d ≠ 7", "another user's email", "0", "0.314"],
        ["<code>name = d;</code>", "no rows: no name is a digit", "0", "0.312"],
        ["<code>age = d;</code>", "no rows: ages are 19 to 46", "0", "0.335"],
        ["anything else", "not one of the 27 grammatical answers", "0", "0.002"],
      ],
      highlight: [0],
      note: "Probabilities are exact sums over our model's next-token probabilities.",
    },
    terms: [{ term: "Verifiable reward", def: "A reward computed by a program that checks the answer (unit tests, a math answer match, running a query) instead of by a learned reward model." }],
    src: ["DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (§2.2)"],
  });

  Deck.slide({
    id: "post-reinforce",
    title: "REINFORCE: raise the log-probability of each sample in proportion to its reward",
    body: String.raw`<p>RL needs the gradient of the expected reward \(J = \mathbb{E}_{a \sim \pi}[r(a)]\), but the checker has no derivative. The <b>policy gradient</b> needs none: \(\nabla J = \mathbb{E}[r(a)\, \nabla \ln \pi(a)]\), so sampling an action and scaling \(\nabla \ln \pi(a)\) by its reward gives an unbiased estimate. Williams (1992) named this family of updates <b>REINFORCE</b>.</p>
<p>Take one decision from our task with the digit fixed at 7: the filter column, with three actions. Our model's probabilities are 0.352, 0.313 and 0.335, and only <code>id</code> earns reward 1, so J = 0.352. The plate works the gradient. A single sample of <code>id</code> gives the estimate (0.648, −0.313, −0.335); a sample of the other two gives 0. Weighted by their probabilities, the estimates average to the exact gradient (0.228, −0.110, −0.118). One step of size 1 after sampling <code>id</code> raises π(id) from 0.352 to 0.590.</p>
<p>For a whole answer, \(\ln \pi(y \mid x) = \sum_t \ln \pi(y_t \mid x, y_{\lt t})\), so one reward (in GRPO, one advantage \(\hat A_i\)) multiplies the gradient of every token of the answer.</p>`,
    math: String.raw`\nabla_\theta J = \mathbb{E}_{a \sim \pi_\theta}\big[r(a)\, \nabla_\theta \ln \pi_\theta(a)\big]`,
    fig: {
      type: "steps", caption: "REINFORCE on three actions, with our model's probabilities", start: 0,
      steps: [
        { html: String.raw`\[\begin{gathered}\pi = (0.352, 0.313, 0.335) \\ r = (1, 0, 0)\end{gathered}\]`, note: String.raw`\(\pi = \operatorname{softmax}(z)\) over id, name, age after WHERE; digit 7` },
        { html: String.raw`\[J = \textstyle\sum_a \pi(a)\, r(a) = 0.352\]`, note: "expected reward" },
        { html: String.raw`\[\nabla J = \textstyle\sum_a r(a)\, \nabla \pi(a) = \sum_a \pi(a)\, r(a)\, \nabla \ln \pi(a)\]`, note: String.raw`because \(\nabla \pi = \pi\, \nabla \ln \pi\)` },
        { html: String.raw`\[\nabla_z \ln \pi(a) = e_a - \pi\]`, note: "softmax: one-hot minus probabilities (Part 2)" },
        { html: String.raw`\[\begin{gathered}\nabla_z J = \textstyle\sum_a \pi(a)\, r(a)\, (e_a - \pi) \\ = 0.352\,(0.648, -0.313, -0.335) \\ = (0.228, -0.110, -0.118)\end{gathered}\]`, note: "the exact gradient" },
        { html: String.raw`\[\begin{gathered}g = r(a)\, (e_a - \pi) \\ a = \text{id}: (0.648, -0.313, -0.335) \\ a = \text{name or age}: (0, 0, 0)\end{gathered}\]`, note: "one sample, one estimate" },
        { html: String.raw`\[\begin{gathered}\mathbb{E}[g] = 0.352\,(0.648, -0.313, -0.335) \\ +\, 0.648\,(0, 0, 0) = \nabla_z J\end{gathered}\]`, note: "unbiased" },
        { html: String.raw`\[z \leftarrow z + g:\quad \pi(\text{id}) = 0.352 \to 0.590\]`, note: "one update after sampling id" },
      ],
    },
    terms: [
      { term: "Policy gradient", def: "The gradient of expected reward with respect to the policy's parameters, \\(\\mathbb{E}[r(a)\\,\\nabla \\ln \\pi(a)]\\), estimated from sampled actions and their rewards." },
      { term: "REINFORCE", def: "Williams' (1992) policy-gradient update: \\(\\Delta w = \\alpha (r - b)\\, \\partial \\ln \\pi / \\partial w\\) for a sampled action, with a baseline b." },
    ],
    src: ["Williams, Simple statistical gradient-following algorithms for connectionist reinforcement learning, Machine Learning 8, 1992 (§4)", "Shao et al., DeepSeekMath, 2024 (§4.1.2)"],
  });

  Deck.slide({
    id: "post-baseline",
    title: "Subtracting a baseline keeps the estimate unbiased and cuts its variance",
    body: String.raw`<p>Adding a constant to every reward should not change what the policy learns, and in expectation it does not: \(\mathbb{E}[\nabla \ln \pi(a)] = \nabla \sum_a \pi(a) = \nabla 1 = 0\). It does change each single estimate. With rewards (11, 10, 10) a sample of <code>name</code>, a wrong answer, pushes its own probability up with weight 10.</p>
<p>So REINFORCE subtracts a <b>baseline</b> b from the reward. Any b that does not depend on the sampled action leaves the mean unchanged. Measured on the plate's three actions: with rewards (1, 0, 0) the total variance of the estimate is 0.144 without a baseline and 0.070 with b = J. With rewards (11, 10, 10) it is 71.2 without and again 0.070 with.</p>
<p>Williams' REINFORCE already includes this term. PPO's value model is a learned baseline per token; GRPO, next, uses the mean reward of a group of samples.</p>`,
    math: String.raw`\nabla_\theta J = \mathbb{E}_{a \sim \pi_\theta}\big[(r(a) - b)\, \nabla_\theta \ln \pi_\theta(a)\big]`,
    fig: { type: "post-pg", caption: "Every possible single-sample estimate, its mean and its variance (computed exactly)", shift: 0, baseline: false },
    terms: [{ term: "Baseline", def: "A value b subtracted from the reward in a policy-gradient estimate. It leaves the expected gradient unchanged and, chosen well, reduces its variance." }],
    src: ["Williams, Simple statistical gradient-following algorithms for connectionist reinforcement learning, 1992 (§4)"],
  });

  Deck.slide({
    id: "post-grpo",
    title: "GRPO: the other answers in a group are the baseline",
    example: "tokens ids",
    body: String.raw`<p><b>GRPO</b> (group relative policy optimization) samples a group of G answers to one prompt and scores each against the others: advantage = (reward − group mean) / group standard deviation. It needs no value model, which in PPO is usually about as large as the policy. Like PPO it clips the ratio, and it adds β times a KL estimate to the loss rather than to the reward: β = 0.04 in DeepSeekMath, 0.001 for R1-Zero.</p>
<p>In a group of 8 with one correct answer, the mean is 0.125 and the standard deviation 0.331, so the correct answer gets A = +2.65 and each wrong one −0.38. A group with no correct answer has all A = 0 and teaches nothing. Our model starts at p(correct) = 0.038, so (1 − 0.038)⁸ = 73% of first groups of 8 are wasted, and 8% of groups of 64, the size DeepSeekMath used.</p>
<p>To run in the page, the plate's policy covers only the two choices that matter, the column after WHERE and the digit after =, started from our model's probabilities. It takes one step per group, so the clip never binds, and drops the KL term. From Reset, Run 10 rounds raises p(correct) from 3.8% to 55.2%.</p>`,
    math: [String.raw`\hat A_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G)}`, String.raw`\begin{gathered}\mathcal{J} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_t \ell_{i,t} \;-\; \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) \\ \ell_{i,t} = \min\big(\rho_{i,t} \hat A_i,\ \operatorname{clip}(\rho_{i,t}, 1 - \varepsilon, 1 + \varepsilon)\, \hat A_i\big)\end{gathered}`],
    fig: { type: "tinylm", widget: "rl", caption: "GRPO on “the email of user 7”: sample a group, check each query, update" },
    note: "Training all 9,968 weights by backpropagation instead (Adam, 10⁻³), 60 groups took p(correct) from 3.8% to 9.9%; 38 groups had no correct answer.",
    terms: [{ term: "GRPO", def: "Group relative policy optimization: PPO's clipped objective with advantages computed from a group of answers to the same prompt, (r − mean)/std, and no value model." }],
    src: ["Shao et al., DeepSeekMath, 2024 (§4.1.1, §4.1.2, §4.2)", "DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (§2.1)"],
  });

  Deck.slide({
    id: "post-reasoning",
    title: "Reasoning models trade more tokens per answer for accuracy",
    body: String.raw`<p>RL with checkable rewards produced <b>reasoning models</b>: models that write a long chain of thought before the answer. Spending more tokens per answer, or more answers per problem, is called <b>test-time compute</b>. OpenAI reported that o1's performance improves with both more RL training and more time spent thinking. On AIME 2024 it solved 74% of problems with one sample, 83% by majority vote over 64 samples and 93% by re-ranking 1,000 samples with a learned scoring function, against 12% for GPT-4o in the post's text. Its appendix lists GPT-4o at 9.3% with one sample and 13.4% with a vote of 64, the figures in the plate.</p>
<p>DeepSeek-R1-Zero, trained with GRPO on DeepSeek-V3-Base using only rule-based rewards, rose from 15.6% to 77.9% on AIME 2024 during RL, writing hundreds to thousands of tokens per answer. The plate compares the two kinds of test-time compute on one benchmark: majority voting over 64 samples lifts GPT-4o only from 9.3% to 13.4%, while one reasoning sample of DeepSeek-R1 scores 79.8%.</p>`,
    fig: {
      type: "bars", caption: "AIME 2024 accuracy: GPT-4o from the o1 post's appendix (also in DeepSeek-R1's Table 15), DeepSeek-R1 from its report", fmt: "pct1", max: 1,
      items: [
        { label: "GPT-4o, 1 sample", value: 0.093, text: "9.3%" },
        { label: "GPT-4o, vote of 64", value: 0.134, text: "13.4%" },
        { label: "R1, 1 sample", value: 0.798, text: "79.8%", hi: true },
        { label: "R1, vote of 64", value: 0.867, text: "86.7%" },
        { label: "R1, any of 64", value: 0.9, text: "90.0%" },
      ],
      note: "“Any of 64” (pass@64) counts a problem as solved if one of 64 samples is right; it needs a checker to pick that sample.",
    },
    terms: [
      { term: "Reasoning model", def: "A model trained, usually with RL on checkable problems, to write a long chain of thought before its final answer." },
      { term: "Test-time compute", def: "Computation spent at inference to improve an answer: longer reasoning per sample, or more samples combined by voting or re-ranking." },
    ],
    src: ["OpenAI, Learning to reason with LLMs, 12 September 2024 (text and Appendix A)", "DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (§2.3, Table 15, App. E.4)"],
  });

  Deck.slide({
    id: "post-hacking",
    title: "Reward hacking: the policy optimizes the check, not the intent",
    body: String.raw`<p>A policy maximizes the reward it is given, flaws included. Exploiting a flaw is <b>reward hacking</b>. Suppose the checker for <i>the email of user 7</i> only required the query to return exactly one row. All nine <code>id = d</code> queries pass, so this proxy rewards nine answers, one of them right.</p>
<p>The plate runs the GRPO of the previous sheet against the proxy, seed 5, 40 groups of 8. The proxy reward climbs from 35% to 99.5%, but the share of truly correct queries ends at 5.7%: every digit earns the same reward, so the digit gets no signal and drifts. With the real check, the same run reaches 96.8%.</p>
<p>Learned rewards get hacked the same way. Stiennon et al. found that optimizing harder against their reward model first improved summaries and then made them worse while the predicted preference kept rising. DeepSeek-R1 saw a helpfulness reward model's score rise while CodeForces accuracy fell.</p>`,
    fig: { type: "post-hack", caption: "GRPO against a proxy check (one row returned) and against the real check (gus@example.com returned)", seed: 5, updates: 40 },
    check: {
      q: "Which reward is hardest to hack for “the email of user 7”?",
      options: ["The query parses", "The query returns exactly one row", "The query returns exactly gus@example.com", "The query mentions the column id"],
      answer: 2,
      why: "Our model already writes a grammatical query 99.8% of the time, so parsing rewards almost everything; one row is returned by all nine <code>id = d</code> queries; mentioning id is satisfied by the same nine. Only the exact result singles out the right answer.",
    },
    terms: [{ term: "Reward hacking", def: "A policy raising its reward by exploiting flaws in the reward function rather than by doing the intended task." }],
    src: ["Stiennon et al., Learning to summarize from human feedback, 2020 (§4.3, Fig. 5)", "DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (App. B.5, Fig. 6)"],
  });

  Deck.chapter({ title: "RL infrastructure" });

  Deck.slide({
    id: "post-rlinfra",
    title: "Generation dominates: an inference engine runs inside the training loop",
    example: "tokens ids",
    body: String.raw`<p>An RL step has four stages (plate): rollout workers generate answers with an inference engine, verifiers score them, the trainer updates the weights, and the new weights go back to the rollout workers. Generation is the expensive stage. Training processes all of an answer's tokens in one forward and backward pass; generation produces them one decode step at a time, and decode is memory-bound (Part 8).</p>
<p>OpenRLHF's authors write that generation often takes over 90% of RLHF and RLVR runtime when models write thousands of tokens per answer. HybridFlow measured up to 81.2% in NeMo-Aligner, whose generator had no KV cache. And a synchronous step waits for its longest answer, which leaves GPUs idle.</p>
<p>The scale for DeepSeek-R1-Zero: each rollout generated 8,192 answers of up to 32,768 tokens, up to 268 million tokens, and later 65,536 tokens each, up to 537 million.</p>`,
    fig: { type: "tinylm", widget: "rlinfra", caption: "The RL loop as a distributed system; step through one iteration" },
    terms: [{ term: "Rollout", def: "Generating answers from the current policy for a batch of prompts; in RL for language models, done by an inference engine inside the training loop." }],
    src: ["Hu et al., OpenRLHF, arXiv 2405.11143v6, 2025 (§1)", "Sheng et al., HybridFlow (veRL), 2024 (§8)", "Fu et al., AReaL, 2025 (abstract)", "DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (§2.1)"],
  });

  Deck.slide({
    id: "post-sync",
    title: "Every step, the new weights must reach the generators",
    body: String.raw`<p>After each update the generators need the new weights: 140 GB per step for a 70B model in BF16. HybridFlow cites this case: with training and generation on different GPUs, the transfer took up to 36.4% of an RLHF iteration.</p>
<p>The two stages also want different layouts. Training is compute-bound and splits the model widely; HybridFlow's example puts a 7B model in 8 pieces on 8 GPUs. Generation is memory-bound and prefers fewer pieces and more copies: the same 7B in 2 pieces, replicated 4 times. So moving the weights means resharding them. HybridFlow's 3D-HybridEngine reshards in place on the same GPUs, without a second copy of the weights, and cut this transition for 70B models by up to 71.2% against DeepSpeed-Chat and 89.1% against OpenRLHF.</p>
<p>The plate gives the floor, bytes ÷ bandwidth. One copy over one 400 Gb/s network port (50 GB/s each way) takes 2.8 s. Over one H100's NVLink it takes at least 0.31 s: the datasheet's 900 GB/s counts both directions, 18 links × 25 GB/s = 450 GB/s each way. 140 GB does not fit on one 80 GB GPU in any case; with the weights on n GPUs sending in parallel, divide by n.</p>`,
    fig: {
      type: "calc", caption: "Time to move the weights once, one way, at full link bandwidth",
      inputs: [
        { id: "N", label: "parameters", min: 1e9, max: 1e12, value: 7e10, log: true, step: "any", fmt: (v) => `${(v / 1e9).toFixed(v < 1e10 ? 1 : 0)}B` },
        { id: "B", label: "Weight format", options: [{ label: "BF16, 2 bytes", value: 2 }, { label: "FP8, 1 byte", value: 1 }], value: 2 },
        { id: "bw", label: "Link per sending GPU", options: [{ label: "400 Gb/s port, 50 GB/s", value: 50e9 }, { label: "H100 NVLink, 450 GB/s", value: 450e9 }], value: 50e9 },
        { id: "n", label: "GPUs sending in parallel", options: [1, 2, 4, 8].map((x) => ({ label: String(x), value: x })), value: 1 },
      ],
      outputs: [
        { label: "bytes to move", f: (v) => v.N * v.B, fmt: (b) => gb(b) },
        { label: "bytes per sending GPU", f: (v) => (v.N * v.B) / v.n, fmt: (b) => gb(b) },
        { label: "time at full bandwidth", f: (v) => (v.N * v.B) / (v.bw * v.n), fmt: "time" },
      ],
      note: (v) => `${gb(v.N * v.B)} ÷ (${v.n} × ${int(v.bw / 1e9)} GB/s). A floor: real transfers add protocol overhead, and resharding can move more than one copy.`,
    },
    src: ["Sheng et al., HybridFlow: A Flexible and Efficient RLHF Framework, 2024 (§2.3, §8)", "NVIDIA, H100 Tensor Core GPU datasheet (NVLink 900 GB/s)", "NVIDIA, H100 whitepaper (18 NVLink links)", "NVIDIA, ConnectX-7 datasheet (400 Gb/s)"],
  });

  Deck.slide({
    id: "post-async",
    title: "Asynchronous RL overlaps generation and training on slightly stale samples",
    layout: "wide",
    body: String.raw`<p>In a synchronous loop the generators and the trainer wait for each other: in the plate (generate 6 units, train 3, sync 1) a step takes 10 units and the trainer is busy 30% of the time. <b>Asynchronous RL</b> lets generation run ahead with the weights it has. A step then takes 6 units, 1.67 times the throughput, but each batch comes from a policy one update old. Such training is <b>off-policy</b>; the number of updates in between is the batch's <b>staleness</b>.</p>
<p>With stale samples, PPO's \(\pi_{\text{old}}\) is the old sampling policy, so the clip holds each update near it. AReaL decouples the two: the importance weight uses the sampling policy, the clip centers on a recent one. In its ablation (1.5B model, AIME 2024, staleness up to 4) standard PPO scored 23.3, the decoupled objective 42.2, synchronous training 42.0. Capping staleness at 4 to 8 updates, AReaL reports up to 2.77 times the speed of synchronous systems.</p>
<p>Noukhovitch et al. trained a LLaMA 3.1 8B chatbot about 40% faster than synchronously. <b>Online DPO</b>, which applies the DPO loss to two fresh samples per prompt ranked by a reward model instead of to 9.3's fixed pairs, was the most robust to staleness.</p>`,
    fig: {
      type: "timeline", caption: "Synchronous and asynchronous RL over 30 time units (blue: generate, red: train, green: weight sync)", total: 30, unit: "units",
      rows: ["sync: generate", "sync: train", "async: generate", "async: train"],
      items: [
        { row: 0, start: 0, len: 6, label: "batch 1", kind: "s1" }, { row: 0, start: 10, len: 6, label: "batch 2", kind: "s1" }, { row: 0, start: 20, len: 6, label: "batch 3", kind: "s1" },
        { row: 1, start: 6, len: 3, label: "train", kind: "s2" }, { row: 1, start: 9, len: 1, label: "", kind: "s3", tip: "weight sync" },
        { row: 1, start: 16, len: 3, label: "train", kind: "s2" }, { row: 1, start: 19, len: 1, label: "", kind: "s3", tip: "weight sync" },
        { row: 1, start: 26, len: 3, label: "train", kind: "s2" }, { row: 1, start: 29, len: 1, label: "", kind: "s3", tip: "weight sync" },
        { row: 2, start: 0, len: 6, label: "batch 1", kind: "s1" }, { row: 2, start: 6, len: 6, label: "batch 2, stale", kind: "s1", tip: "sampled with the weights before update 1" }, { row: 2, start: 12, len: 6, label: "batch 3, stale", kind: "s1", tip: "sampled one update behind" }, { row: 2, start: 18, len: 6, label: "batch 4, stale", kind: "s1", tip: "sampled one update behind" }, { row: 2, start: 24, len: 6, label: "batch 5, stale", kind: "s1", tip: "sampled one update behind" },
        { row: 3, start: 6, len: 3, label: "train", kind: "s2" }, { row: 3, start: 9, len: 1, label: "", kind: "s3", tip: "weight sync" },
        { row: 3, start: 12, len: 3, label: "train", kind: "s2" }, { row: 3, start: 15, len: 1, label: "", kind: "s3", tip: "weight sync" },
        { row: 3, start: 18, len: 3, label: "train", kind: "s2" }, { row: 3, start: 21, len: 1, label: "", kind: "s3", tip: "weight sync" },
        { row: 3, start: 24, len: 3, label: "train", kind: "s2" }, { row: 3, start: 27, len: 1, label: "", kind: "s3", tip: "weight sync" },
      ],
      note: "In 30 units the synchronous loop trains 3 batches and the asynchronous one 4; from then on 1 batch every 10 units against 1 every 6.",
    },
    note: "DeepSeek-R1-Zero trains each rollout of 8,192 answers as 16 mini-batches, so the last one is used 15 updates after it was sampled.",
    terms: [
      { term: "Asynchronous RL", def: "Running generation and training at the same time on different GPUs, so that each batch is sampled by a policy one or more updates older than the one it trains." },
      { term: "Staleness", def: "How many policy updates separate the weights that sampled a batch from the weights being trained on it." },
      { term: "Off-policy", def: "Training a policy on samples generated by a different policy, for example an older version of itself." },
      { term: "Online DPO", def: "DPO on pairs sampled from the current policy during training and ranked by a reward model, instead of on a fixed set of pairs." },
    ],
    src: ["Noukhovitch et al., Asynchronous RLHF, ICLR 2025 (abstract, §2)", "Fu et al., AReaL, 2025 (§5.2, §7.4, Table 2)", "DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (§2.1)"],
  });

  Deck.slide({
    id: "post-rlscale",
    title: "Scaling RL: thousands of steps on hundreds of GPUs",
    body: String.raw`<p>RL is small next to pretraining but large next to the rest of post-training. DeepSeek-R1-Zero ran 10,400 GRPO steps, each on 32 questions with 16 answers apiece, on 64 × 8 H800 GPUs for about 198 hours: 101K GPU hours, 3.8% of the 2,664K that pretrained DeepSeek-V3 and 20 times its 5K of post-training. OpenAI reported that o1 kept improving with more RL compute, and that the constraints on scaling it differ substantially from pretraining's.</p>
<p>Three things change with scale. Answers get longer (R1-Zero's limit went from 32,768 to 65,536 tokens at step 8,200), so generation takes a larger share. The reference model for the KL term is refreshed: R1-Zero replaced it with the current policy every 400 steps. And rewards are chosen with care: DeepSeek kept learned reward models out of R1-Zero's RL, and R1's last RL stage used them only for general prompts, in the final 400 of 1,700 steps. Others train learned checkers: DeepSeekMath's GRPO used a learned reward model, and Kimi k1.5's chain-of-thought reward model judged math answers correctly 98.5% of the time in spot checks, against 84.4% for a classic one.</p>`,
    fig: {
      type: "bars", caption: "DeepSeek-V3 and DeepSeek-R1 training cost, in thousands of H800 GPU hours", fmt: "f0",
      items: [
        { label: "V3 pretraining", value: 2664, text: "2,664K" },
        { label: "V3 long context", value: 119, text: "119K" },
        { label: "V3 post-training", value: 5, text: "5K" },
        { label: "R1-Zero RL", value: 101, text: "101K", hi: true },
        { label: "R1 SFT data", value: 5, text: "5K" },
        { label: "R1 SFT and RL", value: 41, text: "41K" },
      ],
    },
    check: {
      q: "DeepSeek-R1-Zero trained on 64 × 8 H800 GPUs for about 198 hours. About how many GPU hours is that?",
      options: ["12,672", "101,376", "1,584", "811,008"],
      answer: 1,
      why: "64 × 8 = 512 GPUs, and 512 × 198 h = 101,376 GPU hours, the 101K in DeepSeek-R1's cost table. 12,672 counts 64 GPUs; 1,584 counts 8; 811,008 multiplies by 8 twice.",
    },
    src: ["DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (§2.1, §3.2, App. B.4.4, Table 7)", "DeepSeek-AI, DeepSeek-V3 Technical Report, 2024 (Table 1)", "OpenAI, Learning to reason with LLMs, 2024", "Shao et al., DeepSeekMath, 2024 (§4.2)", "Kimi Team, Kimi k1.5, 2025 (§2.3.5)"],
  });

  Deck.chapter({ title: "Evaluation" });

  Deck.slide({
    id: "post-bench",
    title: "A benchmark measures generalization only if its items were held out",
    body: String.raw`<p>Post-training is steered by evaluations, so a score has to mean what it claims. A benchmark measures how well a model handles new problems only if its items were not in the training data. If they were, the benchmark is <b>contaminated</b> and partly measures recall.</p>
<p>Our two models show the size of the effect. Part 5 trained a second model without 27 of the 270 queries. On those 27, the model that trained on them has a loss of 0.687 nats per token and the one that never saw them 1.272: a benchmark made of them would favor the first by 0.586, although on the other 243 queries the two are within 0.013.</p>
<p>Llama 3 flags an evaluation example as contaminated when enough of its tokens lie in 8-grams that also occur in the pretraining corpus. 85% of HellaSwag was flagged, with an estimated gain of 14.8 points for the 8B model; 52% of Natural Questions was flagged with almost no gain. Its post-training data was decontaminated by exact match against benchmark prompts.</p>`,
    fig: {
      type: "table", caption: "Loss per token of our two models on the held-out and the shared queries", align: ["l", "r", "r"],
      cols: ["Model", "27 held-out queries", "243 other queries"],
      rows: [["Trained on all 270 (contaminated)", "0.687", "0.695"], ["Trained without the 27 (clean)", "1.272", "0.682"]],
      highlight: [0],
      note: "Means over each query's 8 predictions, weighted by the query's probability in the grammar, recomputed from both sets of weights.",
    },
    terms: [{ term: "Contamination", def: "Overlap between a benchmark's test items and a model's training data, which makes the benchmark measure recall instead of generalization." }],
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (§5.1.4, Table 15, §5.2)"],
  });

  Deck.slide({
    id: "post-judge",
    title: "An LLM judge agrees with people about as often as people agree with each other",
    body: String.raw`<p>Open-ended answers have no checker, and human ratings are slow. An <b>LLM judge</b> is a strong model prompted to compare two answers or grade one. On MT-Bench, Zheng et al. found GPT-4's pairwise verdicts matched human experts on 85% of non-tied votes, against 81% agreement between two experts.</p>
<p>Judges have measurable biases (plate). Shown the same two answers in both orders, GPT-4 gave the same verdict 65.0% of the time and favored the first position in 30.0%. Padding an answer with a reworded copy of its own list fooled GPT-4 in 8.7% of 23 cases and Claude-v1 in 91.3%. GPT-4 gave its own answers a 10% higher win rate than humans did, though the authors could not confirm a self-preference from so little data.</p>
<p>The usual fix for position bias is to judge twice with the order swapped and count a win only when both verdicts agree. DeepSeek-R1's helpfulness preferences came from DeepSeek-V3 judging each pair four times with the answers' positions randomized.</p>`,
    fig: {
      type: "table", caption: "Biases of three LLM judges", dense: true, minWidth: 520, align: ["l", "r", "r", "r"],
      cols: ["Judge", "Same verdict when the order is swapped", "Favors the first answer", "Fooled by a padded list"],
      rows: [["GPT-4", "65.0%", "30.0%", "8.7%"], ["GPT-3.5", "46.2%", "50.0%", "91.3%"], ["Claude-v1", "23.8%", "75.0%", "91.3%"]],
      note: "Position tests use two similar answers per MT-Bench question with the default prompt; the padded-list test uses 23 answers.",
    },
    check: {
      q: "A judge prefers answer A when A is shown first, and answer B when B is shown first. What does the swap rule record?",
      options: ["A wins", "B wins", "A tie", "Run it a third time and take the majority"],
      answer: 2,
      why: "The rule counts a win only when both orders agree. These verdicts disagree, so the judge preferred whichever answer came first: a tie.",
    },
    terms: [{ term: "LLM judge", def: "A language model prompted to compare or grade answers, used where no program can check them." }],
    src: ["Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, 2023 (Tables 2, 3, 5; §3.3–3.4)", "DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2 (§3.1)"],
  });

  Deck.slide({
    id: "post-life",
    title: "The life of a model, from random weights to a deployed assistant",
    layout: "wide",
    body: String.raw`<p>Every stage of this set, in order, with what it did to our model and what it does at frontier scale. The frontier column is about 40 million times larger in parameters and about 2.5 million times larger in training tokens: 15.6 trillion against our 6.1 million.</p>
<p>The computation is the same at both scales. Pretraining and SFT minimize a cross-entropy; reward models and DPO minimize \(-\ln \sigma\) of a difference; PPO and GRPO follow a clipped policy gradient. SFT, DPO and GRPO ran in this part on our 9,968 weights, with numbers you can check; at frontier scale the same losses run on hundreds of billions.</p>`,
    take: "Post-training costs little compute next to pretraining and decides how the model answers. Every stage is a loss you can now write down and evaluate by hand.",
    fig: {
      type: "table", caption: "From initialization to serving", dense: true, minWidth: 640,
      cols: ["Stage", "Our model", "At frontier scale", "Where"],
      rows: [
        ["Initialization", "9,968 random weights, seed 7", "Llama 3.1 405B: 405 billion weights", "Part 3"],
        ["Pretraining", "6,000 Adam steps; loss 0.6942 against a floor of 0.6937", "15.6T tokens, 3.8 × 10²⁵ FLOPs, up to 16K H100 GPUs", "Parts 4–7"],
        ["Supervised fine-tuning", "Narrow fine-tuning forgot that <code>name</code> belongs to <code>users</code>; 25% replay kept it", "Rejection-sampled answers; six rounds for Llama 3", "9.2"],
        ["Preference training", "DPO: the right query from 0.038 to 0.577 with masking and NLL", "Reward model + PPO (InstructGPT); DPO with β = 0.1 (Llama 3)", "9.3"],
        ["RL with checkers", "GRPO on a SQL checker: 3.8% to 55.2% in 10 groups with a 12-logit policy; 3.8% to 9.9% in 60 groups on all 9,968 weights", "DeepSeek-R1-Zero: AIME 2024 from 15.6% to 77.9%", "9.4–9.5"],
        ["Evaluation", "Held-out loss 1.272, against 0.687 when contaminated", "Contamination analysis; LLM judges with swapped order", "9.6"],
        ["Serving", "KV cache of 8 positions", "Batched, paged, quantized inference", "Part 8"],
      ],
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024", "Ouyang et al., InstructGPT, 2022", "DeepSeek-AI, DeepSeek-R1, arXiv 2501.12948v2"],
  });
})();
