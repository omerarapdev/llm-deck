/* Widgets for the Start slides and Pass 1 (the whole loop). */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip, slider, segmented, button, player, barList, topRows } = window.UI;
  const M = window.TinyLM;
  const W = (window.Widgets = window.Widgets || {});
  const V = M.vocab;
  const ids = M.runningIds, toks = M.running;
  const FINAL = M.finalStep;
  const finalTrace = () => (W._ft = W._ft || M.forward(ids.slice(0, 8)));

  // Render a query string with role colors, for lists of example queries.
  function queryLine(text, cls) {
    const line = h("code", { class: "qline" + (cls ? " " + cls : "") });
    for (const w of M.pretokenize(text)) {
      const r = window.UI.roleOf(w);
      line.appendChild(h("span", { class: `qt q-${r}` }, w));
    }
    return line;
  }
  W._queryLine = queryLine;

  // ------------------------------------------------------------ cover
  W.cover = {
    init(el) {
      const row = el.querySelector('[data-row="tokens"]');
      const pred = el.querySelector('[data-row="pred"]');
      const typed = el.querySelector(".typed");
      const full = M.data.running;
      function renderChips(animated) {
        clear(row);
        toks.forEach((t, i) => {
          const c = chip(t, { id: ids[i] });
          if (animated) { c.classList.add("pop-in"); c.style.setProperty("--i", i); }
          row.appendChild(c);
        });
      }
      function renderPred() {
        clear(pred);
        const p = finalTrace().probs[4];
        const order = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
        pred.append(
          h("span", { class: "cp-lead" }, "next token after ", h("code", null, "…WHERE"), ":"),
          ...order.map(([v, i]) => h("span", { class: "cp-item" }, chip(V[i]), h("span", { class: "num" }, fmt.p(v)))));
      }
      renderChips(false);
      renderPred();
      let busy = false;
      el.querySelector('[data-act="replay"]').addEventListener("click", () => {
        if (busy) return;
        busy = true;
        clear(row);
        let k = 0;
        typed.textContent = "";
        const t = setInterval(() => {
          k++;
          typed.textContent = full.slice(0, k);
          if (k >= full.length) {
            clearInterval(t);
            setTimeout(() => { renderChips(true); busy = false; }, 250);
          }
        }, window.UI.reduceMotion() ? 0 : 38);
      });
    },
  };

  // ------------------------------------------------------------ passes
  W.passes = {
    init(el, deck) {
      const list = el.querySelector(".pass-list");
      const Q = {
        1: "What does a language model compute, and what does training change?",
        2: "How does it turn tokens into a prediction, and how does it learn?",
        3: "What changes when the model has hundreds of billions of parameters?",
        4: "How do you serve many users quickly and cheaply?",
        5: "How does post-training turn a predictor into an assistant?",
      };
      for (const p of window.PASSES.filter((x) => x.n >= 1 && x.n <= 5)) {
        const idx = deck.slides.map((sl, i) => [sl, i]).filter(([sl]) => Number(sl.dataset.pass) === p.n).map(([, i]) => i);
        const range = idx.length ? `slides ${idx[0] + 1}–${idx[idx.length - 1] + 1}` : "coming up";
        list.appendChild(h("li", null, h("button", {
          type: "button", class: "pass-item", disabled: !idx.length,
          onclick: () => idx.length && deck.go(idx[0]),
        }, h("span", { class: "pass-n" }, String(p.n)),
        h("span", { class: "pass-body" }, h("span", { class: "pass-name" }, p.name), h("span", { class: "pass-q" }, Q[p.n])),
        h("span", { class: "pass-range" }, range))));
      }
    },
  };

  // ------------------------------------------------------------ world
  W.world = {
    init(el) {
      const schema = el.querySelector(".schema");
      for (const [t, cols] of Object.entries(M.data.schema)) {
        schema.appendChild(h("div", { class: "table-card" },
          h("div", { class: "table-name" }, chip(" " + t)),
          h("div", { class: "table-cols" }, cols.map((c) => chip(" " + c)))));
      }
      schema.appendChild(h("div", { class: "legend" },
        h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--s1)" } }), "table"),
        h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--s2)" } }), "column")));
      const list = el.querySelector(".sample-list");
      let rand = M.mulberry32(7);
      function fill() {
        clear(list);
        list.appendChild(h("li", { class: "is-running" }, queryLine(M.data.running), h("span", { class: "tag" }, "our query")));
        const seen = new Set([M.data.running]);
        while (list.children.length < 7) {
          const q = M.sampleQuery(rand);
          if (seen.has(q)) continue;
          seen.add(q);
          list.appendChild(h("li", null, queryLine(q)));
        }
      }
      fill();
      el.querySelector('[data-act="sample"]').addEventListener("click", () => { rand = M.mulberry32((Math.random() * 1e9) | 0); fill(); });
    },
  };

  // ------------------------------------------------------------ predict
  const NOTES = [
    "Nine different column names can come first. <code>id</code> leads because all three tables have one; <code>name</code> gets about 1/12, exactly how often the data starts that way.",
    "<code>FROM</code> always follows the first column, so it gets 1.000.",
    "Only the <code>users</code> table has a <code>name</code> column, so <code>users</code> gets 1.000.",
    "<code>WHERE</code> always follows the table: 1.000.",
    "<code>users</code> has four columns, but <code>name</code> was already selected and the rule forbids reusing it. Three remain, each near 1/3.",
    "<code>=</code> always follows the filter column: 1.000.",
    "All nine digits are equally common in the data, so each gets about 1/9 = 0.111.",
    "<code>;</code> ends every query: 1.000.",
  ];
  W.predict = {
    init(el, deck) {
      const tr = finalTrace();
      const prefixRow = el.querySelector(".prefix-row");
      const note = el.querySelector(".pred-note");
      const bars = barList(el.querySelector(".pred-bars"), { labelWidth: "7.4rem" });
      let pos = 2;
      const play = player(() => { set((pos + 1) % 8); }, { interval: 2200, onState: (on) => { playBtn.textContent = on ? "Pause" : "Play"; } });
      const sl = slider({ label: "Tokens given to the model", min: 1, max: 8, value: 3, format: (v) => `${v} of 8`, onInput: (v) => { play.pause(); set(v - 1); } });
      const playBtn = button("Play", () => play.toggle(), { kind: "ghost" });
      el.querySelector(".pred-ctl").append(sl, playBtn);
      function set(p) {
        pos = p;
        sl.set(p + 1);
        clear(prefixRow);
        toks.slice(0, p + 1).forEach((t) => prefixRow.appendChild(chip(t)));
        prefixRow.appendChild(h("span", { class: "chip chip-q is-target" }, h("span", { class: "chip-t" }, "?")));
        bars.update(topRows(tr.probs[p], 6, { target: ids[p + 1] }));
        note.innerHTML = NOTES[p];
        deck.strip.set({ prefix: p + 1 });
      }
      this.set = set;
      this.play = play;
      set(2);
    },
    enter() { this.set && this.set(2); },
    leave() { this.play && this.play.pause(); },
  };

  // ------------------------------------------------------------ generate
  W.generate = {
    init(el) {
      const PROMPTS = [
        { label: "SELECT", n: 1 },
        { label: "SELECT name FROM", n: 3 },
        { label: "SELECT name FROM users WHERE", n: 5 },
      ];
      let prompt = PROMPTS[0], T = 1, busy = false;
      const valid = new Set(M.allQueries().map((q) => q.text));
      const seqEl = el.querySelector(".gen-seq");
      const stepsEl = el.querySelector(".gen-steps");
      const logEl = el.querySelector(".gen-log");
      const rateEl = el.querySelector(".gen-rate");
      const promptSel = h("select", { class: "sel", id: "gen-prompt", "aria-label": "Prompt" },
        PROMPTS.map((p, i) => h("option", { value: String(i) }, p.label)));
      promptSel.addEventListener("change", () => { prompt = PROMPTS[+promptSel.value]; showPrompt(); });
      const temp = slider({ id: "gen-temp", label: "Temperature", min: 0.1, max: 3, step: 0.1, value: 1, format: (v) => v.toFixed(1), onInput: (v) => { T = v; rateEl.textContent = ""; } });
      const genBtn = button("Generate", () => run(), { kind: "primary" });
      const rateBtn = button("Test 300 from this prompt", () => rate(), { kind: "ghost" });
      el.querySelector(".gen-ctl").append(h("div", { class: "ctl" }, h("label", { class: "ctl-label", for: "gen-prompt" }, "Prompt"), promptSel), temp, genBtn, rateBtn);

      function showPrompt() {
        clear(seqEl); clear(stepsEl);
        ids.slice(0, prompt.n).forEach((id) => seqEl.appendChild(chip(V[id], { cls: "is-prompt" })));
        stepsEl.appendChild(h("p", { class: "fig-cap" }, "Press Generate. Each row is one forward pass: the model's top choices, and the token that was drawn."));
      }
      function sample(cur) {
        const p = M.nextDist(cur, { temperature: T });
        return { p, t: M.sampleFrom(p) };
      }
      function run() {
        if (busy) return;
        busy = true; genBtn.disabled = true;
        showPrompt();
        clear(stepsEl);
        const cur = ids.slice(0, prompt.n);
        const tick = () => {
          const { p, t } = sample(cur);
          cur.push(t);
          const order = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
          const shown = order.some(([, i]) => i === t) ? order : order.concat([[p[t], t]]);
          stepsEl.appendChild(h("div", { class: "gen-step" },
            h("span", { class: "gen-k" }, `pass ${cur.length - prompt.n}`),
            h("span", { class: "gen-top" }, shown.map(([v, i]) => h("span", { class: "gen-opt" + (i === t ? " is-picked" : "") }, chip(V[i]), h("span", { class: "num" }, fmt.p(v)))))));
          seqEl.appendChild(chip(V[t], { cls: "is-new" }));
          if (V[t] === ";" || cur.length >= 9) {
            const text = cur.map((i) => V[i]).join("");
            const ok = valid.has(text);
            logEl.prepend(h("li", { class: ok ? "ok" : "bad" },
              h("span", { class: "mark", "aria-label": ok ? "follows the grammar" : "breaks the grammar" }, ok ? "✓" : "✗"),
              W._queryLine(text), h("span", { class: "tag" }, `T ${T.toFixed(1)}`)));
            while (logEl.children.length > 5) logEl.lastChild.remove();
            busy = false; genBtn.disabled = false;
            return;
          }
          setTimeout(tick, window.UI.reduceMotion() ? 0 : 420);
        };
        setTimeout(tick, window.UI.reduceMotion() ? 0 : 200);
      }
      function rate() {
        let ok = 0;
        const n = 300;
        for (let k = 0; k < n; k++) {
          const cur = ids.slice(0, prompt.n);
          while (cur.length < 9) { const { t } = sample(cur); cur.push(t); if (V[t] === ";") break; }
          if (valid.has(cur.map((i) => V[i]).join(""))) ok++;
        }
        rateEl.textContent = `${ok} of ${n} valid at temperature ${T.toFixed(1)}`;
      }
      showPrompt();
    },
  };

  // ------------------------------------------------------------ learn
  W.learn = {
    init(el) {
      const table = el.querySelector(".learn-table");
      const stats = el.querySelector(".learn-stats");
      const tr0 = M.forward(ids.slice(0, 8), 0), trF = finalTrace();
      const rows = [];
      table.appendChild(h("div", { class: "lt-row lt-head", role: "row" },
        h("span", { role: "columnheader" }, "Prefix ends with"), h("span", { role: "columnheader" }, "True next"),
        h("span", { role: "columnheader" }, "p(true next)"), h("span", { role: "columnheader", class: "r" }, "Loss")));
      for (let i = 0; i < 8; i++) {
        const fill = h("span", { class: "lt-fill" });
        const pv = h("span", { class: "num" }), lv = h("span", { class: "num r" });
        table.appendChild(h("div", { class: "lt-row", role: "row" },
          h("span", { class: "lt-prefix", role: "cell" }, i > 0 ? h("span", { class: "ell" }, "…") : null, chip(toks[i])),
          h("span", { role: "cell" }, chip(toks[i + 1])),
          h("span", { class: "lt-p", role: "cell" }, h("span", { class: "lt-track" }, fill), pv),
          h("span", { role: "cell", class: "r" }, lv)));
        rows.push({ fill, pv, lv });
      }
      const sMean = h("div", { class: "stat-v" }), sAll = h("div", { class: "stat-v" });
      const floorMean = M.data.floor.reduce((a, b) => a + b, 0) / M.data.floor.length;
      stats.append(
        h("div", { class: "stat" }, sMean, h("div", { class: "stat-l" }, "loss, our query")),
        h("div", { class: "stat" }, sAll, h("div", { class: "stat-l" }, "loss, averaged over the data")),
        h("div", { class: "stat" }, h("div", { class: "stat-v" }, floorMean.toFixed(4)), h("div", { class: "stat-l" }, "lowest possible average (the floor)")));
      const exact = { 0: M.exactLoss(0).mean, [FINAL]: M.exactLoss(FINAL).mean };
      function show(step) {
        const t = step === 0 ? tr0 : trF;
        let sum = 0;
        rows.forEach((r, i) => {
          const p = t.probs[i][ids[i + 1]], l = -Math.log(p);
          sum += l;
          r.fill.style.width = `${p * 100}%`;
          r.pv.textContent = fmt.p(p);
          r.lv.textContent = l.toFixed(2);
        });
        sMean.textContent = (sum / 8).toFixed(4);
        sAll.textContent = exact[step].toFixed(4);
      }
      const seg = segmented({ label: "Model", options: [{ value: 0, label: "Untrained" }, { value: FINAL, label: "Trained, 6,000 steps" }], value: FINAL, onChange: show });
      el.querySelector(".learn-toggle").appendChild(seg);
      show(0);
      requestAnimationFrame(() => setTimeout(() => show(FINAL), 60));
      this.seg = seg; this.show = show;
    },
    enter() {
      if (!this.show) return;
      this.show(0); this.seg.set(0);
      setTimeout(() => { this.show(FINAL); this.seg.set(FINAL); }, window.UI.reduceMotion() ? 0 : 500);
    },
  };

  // ------------------------------------------------------------ loops
  W.loops = {
    init(el, deck) {
      const grid = el.querySelector(".loops-grid");
      const LOOPS = [
        { title: "Training loop", nodes: [
          { label: "Batch of text", sub: "data pipeline", go: "data", plan: 3 },
          { label: "Tokenize", sub: "text → token ids", go: "bpe", plan: 2 },
          { label: "Forward pass", sub: "probabilities", go: "qkv", plan: 2 },
          { label: "Loss", sub: "−ln p(true next)", go: "loss", plan: 1 },
          { label: "Backward pass", sub: "gradients", go: "backprop", plan: 1 },
          { label: "Optimizer update", sub: "AdamW step", go: "optim", plan: 1 },
        ], side: { label: "spread over thousands of GPUs", go: "allreduce", plan: 5 } },
        { title: "Inference loop", nodes: [
          { label: "Prompt", sub: "a user request", go: "prefill", plan: 6 },
          { label: "Tokenize", sub: "text → token ids", go: "bpe", plan: 2 },
          { label: "Forward pass", sub: "prefill, then decode", go: "prefill", plan: 6 },
          { label: "Sample", sub: "temperature, top-p", go: "logits", plan: 2 },
          { label: "Append", sub: "reuse the KV cache", go: "kvcache", plan: 6 },
        ], back: 2, side: { label: "weights shaped by post-training", go: "sft", plan: 7 } },
      ];
      for (const L of LOOPS) {
        const n = L.nodes.length, gap = 58, top = 46, nodeH = 42, Wd = 300;
        const Hh = top + n * gap + 34;
        const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hh}`, class: "loop-svg", role: "group", "aria-label": L.title });
        svg.appendChild(s("defs", null, s("marker", { id: `ah-${L.title.split(" ")[0]}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
          s("path", { d: "M0,0 L10,5 L0,10 z", class: "ah" }))));
        const mk = `url(#ah-${L.title.split(" ")[0]})`;
        svg.appendChild(s("text", { x: 20, y: 24, class: "loop-title" }, L.title));
        L.nodes.forEach((nd, i) => {
          const y = top + i * gap, x = 20, w = 214;
          const idx = deck.byId(nd.go);
          const g = s("g", { class: "loop-node" + (idx < 0 ? " is-later" : ""), tabindex: idx < 0 ? -1 : 0, role: "link",
            "aria-label": `${nd.label}: ${nd.sub}. ${idx < 0 ? "Covered later." : "Go to slide " + (idx + 1)}` });
          g.appendChild(s("rect", { x, y, width: w, height: nodeH, rx: 9 }));
          g.appendChild(s("text", { x: x + 12, y: y + 18, class: "ln-label" }, nd.label));
          g.appendChild(s("text", { x: x + 12, y: y + 33, class: "ln-sub" }, nd.sub));
          g.appendChild(s("text", { x: x + w - 10, y: y + 18, class: "ln-go", "text-anchor": "end" }, idx < 0 ? "" : String(idx + 1)));
          g.appendChild(s("text", { x: x + w - 10, y: y + 33, class: "ln-plan", "text-anchor": "end" }, `plan ${nd.plan}`));
          const goTo = () => { if (idx >= 0) deck.go(idx); };
          g.addEventListener("click", goTo);
          g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); goTo(); } });
          svg.appendChild(g);
          if (i < n - 1) svg.appendChild(s("line", { x1: x + w / 2, y1: y + nodeH + 2, x2: x + w / 2, y2: y + gap - 3, class: "loop-arrow", "marker-end": mk }));
        });
        // loop back: training restarts with the next batch, decoding returns to the forward pass
        const yLast = top + (n - 1) * gap + nodeH / 2, yFirst = top + (L.back || 0) * gap + nodeH / 2, xr = 234;
        svg.appendChild(s("path", { d: `M${xr},${yLast} H${xr + 34} V${yFirst} H${xr + 4}`, class: "loop-arrow back", fill: "none", "marker-end": mk }));
        svg.appendChild(s("text", { x: xr + 40, y: (yFirst + yLast) / 2, class: "ln-sub", transform: `rotate(90 ${xr + 40} ${(yFirst + yLast) / 2})`, "text-anchor": "middle" }, "repeat"));
        const sideIdx = deck.byId(L.side.go);
        const side = h("button", { type: "button", class: "loop-side", disabled: sideIdx < 0, onclick: () => sideIdx >= 0 && deck.go(sideIdx) },
          h("span", null, L.side.label), h("span", { class: "ln-plan-h" }, `${sideIdx >= 0 ? "slide " + (sideIdx + 1) + " · " : ""}plan ${L.side.plan}`));
        grid.appendChild(h("div", { class: "loop-col" }, svg, side));
      }
    },
  };
})();
