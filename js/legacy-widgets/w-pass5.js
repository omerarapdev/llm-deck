/* Widgets for Pass 5 (becoming an assistant) and the wrap-up slides. */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip, slider, segmented, button, barList, chart } = window.UI;
  const M = window.TinyLM;
  const W = (window.Widgets = window.Widgets || {});
  const V = M.vocab, toks = M.running;
  const stat = (v, l) => h("div", { class: "stat" }, h("div", { class: "stat-v" }, v), h("div", { class: "stat-l" }, l));
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  // ------------------------------------------------------------ sft
  W.sft = {
    init(el) {
      const user = ["<|user|>", " Name", " of", " user", " 7", "?", "<|end|>"];
      const asst = ["<|assistant|>", ...toks, "<|end|>"];
      let masked = true;
      const seqEl = el.querySelector(".sft-seq"), note = el.querySelector(".sft-note");
      el.querySelector(".sft-ctl").append(segmented({ label: "Loss mask", value: "on", options: [{ value: "on", label: "Answer only" }, { value: "off", label: "Every token" }], onChange: (v) => { masked = v === "on"; draw(); } }));
      function draw() {
        clear(seqEl);
        const all = [...user.map((t) => ({ t, who: "user" })), ...asst.map((t, i) => ({ t, who: i === 0 ? "marker" : "assistant" }))];
        let trained = 0;
        const rowFor = (label, items) => h("div", { class: "sft-turn" }, h("div", { class: "sft-who" }, label), h("div", { class: "sft-toks" }, items));
        const chipFor = ({ t, who }, idx) => {
          // the first token is never a prediction target
          const isTarget = idx > 0 && (masked ? who === "assistant" : true);
          if (isTarget) trained++;
          const c = chip(t, { cls: (t.startsWith("<|") ? "chip-special " : "") + (isTarget ? "is-trained" : "is-context") });
          c.appendChild(h("span", { class: "chip-under" }, isTarget ? "loss" : "—"));
          return c;
        };
        const items = all.map(chipFor);
        seqEl.append(rowFor("user turn", items.slice(0, user.length)), rowFor("assistant turn", items.slice(user.length)));
        note.innerHTML = masked
          ? `<b>${trained} of ${all.length - 1}</b> predictions are trained: the 9 tokens of our query and the end marker, so the model also learns to stop. The user's turn is context only.`
          : `With no mask, all <b>${trained}</b> predictions are trained, including the user's words. The model would also learn to write users' questions, which wastes capacity on the wrong behavior.`;
      }
      draw();
    },
  };

  // ------------------------------------------------------------ prefs
  W.prefs = {
    init(el) {
      const pair = el.querySelector(".pref-pair");
      pair.append(
        h("div", { class: "pref-prompt" }, h("span", { class: "tag" }, "prompt"), " Name of user 7?"),
        h("div", { class: "pref-ans good" }, h("span", { class: "mark" }, "✓ preferred"), h("code", null, M.data.running)),
        h("div", { class: "pref-ans bad" }, h("span", { class: "mark" }, "✗ rejected"), h("code", null, "DROP TABLE users;")));
      let margin = 2, beta = 0.1, dpoM = 0;
      const out = el.querySelector(".pref-out"), chartEl = el.querySelector(".pref-chart");
      const ch = chart(chartEl, {
        width: 520, height: 180, margin: { l: 44, r: 14, t: 10, b: 36 }, aria: "DPO loss versus the preference margin",
        x: { domain: [-30, 30], ticks: [-30, -15, 0, 15, 30], label: "log-ratio of preferred − log-ratio of rejected" },
        y: { domain: [0, 3.5], ticks: [0, 1, 2, 3] },
      });
      function draw() {
        const pts = [];
        for (let x = -30; x <= 30; x += 0.5) pts.push([x, -Math.log(sigmoid(beta * x))]);
        const loss = -Math.log(sigmoid(beta * dpoM));
        ch.update({ series: [{ points: pts, cls: "accent" }], dots: [{ x: dpoM, y: loss, r: 6, cls: "ink", label: `loss ${loss.toFixed(3)}`, dx: 8, dy: -8 }] });
        clear(out);
        out.append(
          stat(fmt.pct(sigmoid(margin), 1), `reward model: P(✓ beats ✗) when r✓ − r✗ = ${margin.toFixed(1)}`),
          stat(loss.toFixed(3), `DPO loss at margin ${dpoM.toFixed(0)}, β = ${beta.toFixed(2)}`));
      }
      el.querySelector(".pref-ctl").append(
        slider({ id: "pref-m", label: "Reward difference r✓ − r✗", min: -4, max: 4, step: 0.1, value: margin, format: (v) => v.toFixed(1), onInput: (v) => { margin = v; draw(); } }),
        slider({ id: "pref-d", label: "DPO margin (how much more the model favors ✓ than the reference does)", min: -30, max: 30, step: 1, value: dpoM, format: (v) => v.toFixed(0), onInput: (v) => { dpoM = v; draw(); } }),
        slider({ id: "pref-b", label: "β", min: 0.02, max: 0.5, step: 0.01, value: beta, format: (v) => v.toFixed(2), onInput: (v) => { beta = v; draw(); } }));
      draw();
    },
  };

  // ------------------------------------------------------------ rl (GRPO)
  const USERS = [
    [1, "ada", 31], [2, "bo", 25], [3, "cy", 42], [4, "di", 19], [5, "ed", 37], [6, "flo", 28], [7, "gus", 33], [8, "hal", 46], [9, "ivy", 22],
  ].map(([id, name, age]) => ({ id, name, email: `${name}@example.com`, age }));
  function runSQL(q) {
    const m = q.match(/^SELECT (\w+) FROM (\w+) WHERE (\w+) = (\d+);$/);
    if (!m || m[2] !== "users") return { error: "unsupported query" };
    const [, sel, , col, val] = m;
    if (!(sel in USERS[0]) || !(col in USERS[0])) return { error: "unknown column" };
    return { rows: USERS.filter((r) => String(r[col]) === val).map((r) => r[sel]) };
  }
  W._runSQL = runSQL;

  W.rl = {
    init(el) {
      const COLS = [" id", " name", " age"], DIG = ["1", "2", "3", "4", "5", "6", "7", "8", "9"];
      const pre = ["SELECT", " email", " FROM", " users", " WHERE"].map(M.idOf);
      const base = M.forward(pre).probs[4];
      const z0c = COLS.map((c) => Math.log(base[M.idOf(c)]));
      const pd = M.forward(pre.concat([M.idOf(" id"), M.idOf(" =")])).probs[6];
      const z0d = DIG.map((d) => Math.log(pd[M.idOf(" " + d)]));
      const TARGET = "gus@example.com";
      let zc, zd, G = 8, lr = 1, group, history, updates, rand;
      const dbEl = el.querySelector(".rl-db"), groupEl = el.querySelector(".rl-group"), distEl = el.querySelector(".rl-dist"), chartEl = el.querySelector(".rl-chart");
      dbEl.appendChild(h("div", { class: "fig-sub" }, "Table users (id, name, email, age)"));
      dbEl.appendChild(h("div", { class: "rl-rows" }, USERS.map((r) => h("div", { class: "rl-row" + (r.id === 7 ? " is-target" : "") },
        h("span", { class: "num" }, String(r.id)), h("span", null, r.email), h("span", { class: "num dim" }, String(r.age))))));
      dbEl.appendChild(h("p", { class: "fig-cap" }, "Task: “email of user 7”. Correct output: ", h("code", null, TARGET), "."));
      const colBars = barList(h("div"), { labelWidth: "5.4rem" }), digBars = barList(h("div"), { labelWidth: "5.4rem" });
      distEl.append(h("div", { class: "rl-d" }, h("div", { class: "fig-sub" }, "policy: column after WHERE"), colBars.root), h("div", { class: "rl-d" }, h("div", { class: "fig-sub" }, "policy: value after ="), digBars.root));
      const ch = chart(chartEl, {
        width: 520, height: 160, margin: { l: 44, r: 14, t: 10, b: 34 }, aria: "Probability of a correct query over updates",
        x: { domain: [0, 40], ticks: [0, 10, 20, 30, 40], label: "updates" },
        y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => fmt.pct(v) },
      });
      const sm = (z) => M.softmax(z);
      function pCorrect() { return sm(zc)[0] * sm(zd)[6]; }
      function reset() {
        zc = z0c.slice(); zd = z0d.slice(); group = null; updates = 0; rand = M.mulberry32(5);
        history = [[0, pCorrect()]];
        draw();
      }
      function sample() {
        const pc = sm(zc), pdd = sm(zd);
        group = Array.from({ length: G }, () => {
          const c = M.sampleFrom(pc, rand), d = M.sampleFrom(pdd, rand);
          const q = `SELECT email FROM users WHERE${COLS[c]} = ${DIG[d]};`;
          const res = runSQL(q);
          const r = res.rows && res.rows.length === 1 && res.rows[0] === TARGET ? 1 : 0;
          return { c, d, q, res, r };
        });
        const mean = group.reduce((a, g) => a + g.r, 0) / G;
        const sd = Math.sqrt(group.reduce((a, g) => a + (g.r - mean) ** 2, 0) / G);
        group.forEach((g) => { g.A = sd > 0 ? (g.r - mean) / sd : 0; });
        group.sd = sd;
      }
      function update() {
        // one update per group: a used group is stale, so draw a fresh one (and stop at the chart's 40 updates)
        if (updates >= 40) return;
        if (!group || group.used) sample();
        if (group.sd > 0) {
          const pc = sm(zc), pdd = sm(zd);
          const gc = new Array(COLS.length).fill(0), gd = new Array(DIG.length).fill(0);
          group.forEach((g) => {
            for (let i = 0; i < COLS.length; i++) gc[i] += (g.A * ((i === g.c ? 1 : 0) - pc[i])) / G;
            for (let i = 0; i < DIG.length; i++) gd[i] += (g.A * ((i === g.d ? 1 : 0) - pdd[i])) / G;
          });
          zc = zc.map((v, i) => v + lr * gc[i]);
          zd = zd.map((v, i) => v + lr * gd[i]);
        }
        updates++;
        history.push([updates, pCorrect()]);
        group.used = true;
      }
      function draw() {
        clear(groupEl);
        if (!group) groupEl.appendChild(h("p", { class: "fig-cap" }, "Press Sample a group: the policy writes G queries, and each one is run against the table above."));
        else {
          const ok = group.filter((g) => g.r).length;
          groupEl.appendChild(h("div", { class: "fig-sub" }, `${ok} of ${G} correct${group.sd === 0 ? " · every reward is the same, so all advantages are 0 and this group teaches nothing" : ""}${group.used ? " · used for an update" : ""}`));
          const list = h("ol", { class: "rl-list" });
          group.forEach((g) => list.appendChild(h("li", { class: g.r ? "ok" : "" },
            h("span", { class: "mark" }, g.r ? "✓" : "✗"),
            h("code", { class: "qline" }, g.q),
            h("span", { class: "rl-res" }, g.res.rows ? (g.res.rows.length ? g.res.rows.join(", ") : "no rows") : g.res.error),
            h("span", { class: "num rl-adv" }, `A ${fmt.signed(g.A)}`))));
          groupEl.appendChild(list);
        }
        const pc = sm(zc), pdd = sm(zd);
        colBars.update(COLS.map((c, i) => ({ key: c, label: chip(c), value: pc[i], cls: i === 0 ? "is-target" : "" })));
        digBars.update(DIG.map((d, i) => ({ key: d, label: chip(" " + d), value: pdd[i], cls: i === 6 ? "is-target" : "" })));
        ch.update({ series: [{ points: history, cls: "accent" }], hlines: [{ y: history[0][1], label: `start ${fmt.pct(history[0][1], 1)}` }], dots: [{ x: updates, y: pCorrect(), r: 5, cls: "", label: fmt.pct(pCorrect(), 1), dx: 8, dy: -6 }] });
      }
      el.querySelector(".rl-ctl").append(
        segmented({ label: "Group size G", value: G, options: [4, 8, 16].map((v) => ({ value: v, label: String(v) })), onChange: (v) => { G = v; group = null; draw(); } }),
        button("Sample a group", () => { sample(); draw(); }, { kind: "primary" }),
        button("Update on it", () => { update(); draw(); }),
        button("Run 10 rounds", () => { for (let i = 0; i < 10 && updates < 40; i++) { sample(); update(); } draw(); }),
        button("Reset", reset));
      reset();
    },
  };

  // ------------------------------------------------------------ rlinfra
  W.rlinfra = {
    init(el) {
      const dia = el.querySelector(".ri-diagram"), note = el.querySelector(".ri-note");
      const Wd = 520, Hd = 300;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "chart ri-svg", role: "img", "aria-label": "RL loop: rollout workers, verifiers, trainer, weight sync" });
      svg.appendChild(s("defs", null, s("marker", { id: "ri-ah", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "ah" }))));
      const B = [
        { x: 30, y: 30, t: "Rollout workers", sub: "inference engine (vLLM, SGLang)" },
        { x: 290, y: 30, t: "Verifiers", sub: "sandboxed SQL, tests, checkers" },
        { x: 290, y: 200, t: "Trainer", sub: "advantages + update (FSDP)" },
        { x: 30, y: 200, t: "Weight sync", sub: "broadcast new weights" },
      ];
      const boxes = B.map((b) => {
        const g = s("g", { class: "fl-box" });
        g.appendChild(s("rect", { x: b.x, y: b.y, width: 200, height: 70, rx: 12 }));
        g.appendChild(s("text", { x: b.x + 14, y: b.y + 28, class: "ln-label" }, b.t));
        g.appendChild(s("text", { x: b.x + 14, y: b.y + 48, class: "ln-sub" }, b.sub));
        svg.appendChild(g);
        return g;
      });
      const arr = (d, label, lx, ly) => {
        svg.appendChild(s("path", { d, class: "loop-arrow", fill: "none", "marker-end": "url(#ri-ah)" }));
        svg.appendChild(s("text", { x: lx, y: ly, class: "ln-sub", "text-anchor": "middle" }, label));
      };
      arr("M230,65 H288", "answers", 259, 56);
      arr("M390,100 V198", "rewards", 420, 152);
      arr("M290,235 H232", "new weights", 261, 226);
      arr("M130,200 V102", "", 0, 0);
      svg.appendChild(s("text", { x: 100, y: 155, class: "ln-sub", "text-anchor": "middle" }, "next group"));
      dia.appendChild(svg);
      const STEPS = [
        "Rollout workers sample a group of 8 answers to “email of user 7”, e.g. SELECT email FROM users WHERE age = 3;",
        "Verifiers run each query on a sandboxed copy of the table. One returns gus@example.com: reward 1. The rest get 0.",
        "The trainer turns rewards into group-relative advantages and takes a gradient step on the sharded model.",
        "The updated weights are broadcast to the rollout workers, which start the next group with the new policy.",
      ];
      let k = 0;
      const show = () => { boxes.forEach((g, i) => g.classList.toggle("is-on", i === k)); note.textContent = `${k + 1}/4: ${STEPS[k]}`; };
      el.querySelector(".ri-ctl").append(button("Next stage", () => { k = (k + 1) % 4; show(); }, { kind: "primary" }));
      show();
    },
  };

  // ------------------------------------------------------------ recap
  W.recap = {
    init(el, deck) {
      const ex = M.exactLoss().mean;
      const tr = W._trace();
      const STAGES = [
        { go: "world", t: "Plain text", d: `${M.data.running} One of the 270 queries the grammar allows.` },
        { go: "bpe", t: "Tokens", d: "74 learned BPE merges turn 36 characters into 9 tokens." },
        { go: "embed", t: "Vectors", d: "8 input tokens become 8 vectors of 16 learned numbers." },
        { go: "heads", t: "Attention", d: `WHERE looks at users (${tr.layers[0].heads[1].att[4][3].toFixed(2)}) and name (${tr.layers[0].heads[0].att[4][1].toFixed(2)}); FROM looks at name (${tr.layers[1].heads[1].att[2][1].toFixed(2)}).` },
        { go: "predict", t: "Predictions", d: `users ${fmt.p(tr.probs[2][M.idOf(" users")])} after FROM; id ${fmt.p(tr.probs[4][M.idOf(" id")])} after WHERE; 7 ${fmt.p(tr.probs[6][M.idOf(" 7")])} after =.` },
        { go: "train", t: "Training", d: `6,000 Adam steps bring the average loss to ${ex.toFixed(4)}, against a floor of 0.6937.` },
        { go: "flops", t: "Scale", d: "The same recipe at 405B parameters: 3.8 × 10²⁵ FLOPs on 16,384 GPUs." },
        { go: "kvcache", t: "Serving", d: "Prefill, then decode with a KV cache of 256 bytes per token." },
        { go: "rl", t: "Post-training", d: "Fine-tuning, preferences, and RL against a SQL verifier." },
      ];
      const list = el.querySelector(".journey");
      STAGES.forEach((st, i) => {
        const idx = deck.byId(st.go);
        list.appendChild(h("li", null, h("button", { type: "button", class: "journey-item", onclick: () => idx >= 0 && deck.go(idx) },
          h("span", { class: "j-n" }, String(i + 1)),
          h("span", { class: "j-body" }, h("span", { class: "j-t" }, st.t), h("span", { class: "j-d" }, st.d)),
          h("span", { class: "j-go" }, idx >= 0 ? `slide ${idx + 1}` : ""))));
      });
    },
  };

  // ------------------------------------------------------------ plan
  W.plan = {
    init(el, deck) {
      const STEPS = [
        { n: 0, t: "Foundations", s: ["learn", "loss", "gradient", "precision"], next: "Linear algebra, probability and log-softmax by hand; fluent Python, NumPy and PyTorch." },
        { n: 1, t: "Deep learning fundamentals", s: ["gradient", "backprop", "optim", "train"], next: "Build a tiny autograd engine; train an MLP; read the Adam and AdamW papers." },
        { n: 2, t: "Transformers and LLM architecture", s: ["bpe", "embed", "need", "qkv", "mix", "heads", "block", "logits", "real"], next: "Implement a GPT from scratch; add RoPE, grouped-query attention, SwiGLU, mixture of experts." },
        { n: 3, t: "Pretraining", s: ["data", "scaling", "train", "loss"], next: "Deduplication and filtering at scale, fitting scaling laws, μP, evaluation harnesses." },
        { n: 4, t: "Accelerators, kernels, compilers", s: ["gpu", "flash", "precision"], next: "CUDA and Triton kernels, FlashAttention source, profiling with Nsight, torch.compile." },
        { n: 5, t: "Distributed training", s: ["memory", "allreduce", "zero", "tp", "pp", "mesh", "failures"], next: "NCCL internals, FSDP2, Megatron-LM, context and expert parallelism, async checkpointing." },
        { n: 6, t: "Inference and serving", s: ["prefill", "kvcache", "kvsize", "membound", "batching", "paged", "speculative", "quant", "fleet"], next: "Read vLLM or SGLang end to end; write a mini engine with continuous batching and a paged KV cache." },
        { n: 7, t: "Post-training and RL", s: ["sft", "prefs", "rl", "rlinfra"], next: "PPO-based RLHF, DPO variants, GRPO; read veRL or OpenRLHF." },
        { n: 8, t: "Getting hired", s: ["recap"], next: "Public projects with write-ups, merged open-source PRs, back-of-envelope interview math." },
      ];
      const list = el.querySelector(".plan-list");
      STEPS.forEach((st) => list.appendChild(h("li", { class: "plan-item" },
        h("span", { class: "j-n" }, String(st.n)),
        h("div", { class: "j-body" },
          h("span", { class: "j-t" }, st.t),
          h("span", { class: "plan-slides" }, st.s.map((id) => {
            const i = deck.byId(id);
            return i < 0 ? null : h("button", { type: "button", class: "slide-link", title: deck.slides[i].dataset.title, onclick: () => deck.go(i) }, String(i + 1));
          })),
          h("span", { class: "j-d" }, h("b", null, "Next: "), st.next)))));
    },
  };

  // ------------------------------------------------------------ sources
  W.sources = {
    init(el) {
      const ax = (id) => `https://arxiv.org/abs/${id}`;
      const GROUPS = [
        ["Foundations", [
          ["Vaswani et al. 2017", "Attention Is All You Need", ax("1706.03762")],
          ["Sennrich, Haddow & Birch 2016", "Neural Machine Translation of Rare Words with Subword Units (BPE)", ax("1508.07909")],
          ["Radford et al. 2019", "Language Models are Unsupervised Multitask Learners (GPT-2)", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"],
          ["Kingma & Ba 2015", "Adam: A Method for Stochastic Optimization", ax("1412.6980")],
          ["Loshchilov & Hutter 2019", "Decoupled Weight Decay Regularization (AdamW)", ax("1711.05101")],
          ["Zhang & Sennrich 2019", "Root Mean Square Layer Normalization", ax("1910.07467")],
          ["Hendrycks & Gimpel 2016", "Gaussian Error Linear Units (GELUs)", ax("1606.08415")],
        ]],
        ["Architecture", [
          ["Su et al. 2021", "RoFormer: Enhanced Transformer with Rotary Position Embedding", ax("2104.09864")],
          ["Shazeer 2020", "GLU Variants Improve Transformer (SwiGLU)", ax("2002.05202")],
          ["Ainslie et al. 2023", "GQA: Training Generalized Multi-Query Transformer Models", ax("2305.13245")],
          ["DeepSeek-AI 2024", "DeepSeek-V3 Technical Report", ax("2412.19437")],
        ]],
        ["Scaling and pretraining", [
          ["Kaplan et al. 2020", "Scaling Laws for Neural Language Models", ax("2001.08361")],
          ["Hoffmann et al. 2022", "Training Compute-Optimal Large Language Models (Chinchilla)", ax("2203.15556")],
          ["Brown et al. 2020", "Language Models are Few-Shot Learners (GPT-3)", ax("2005.14165")],
          ["Llama Team, Meta 2024", "The Llama 3 Herd of Models", ax("2407.21783")],
          ["Meta 2024", "Llama 3.1 model card", "https://huggingface.co/meta-llama/Llama-3.1-405B"],
          ["Broder 1997", "On the Resemblance and Containment of Documents (MinHash)", "https://doi.org/10.1109/SEQUEN.1997.666900"],
        ]],
        ["Hardware, kernels, numerics", [
          ["NVIDIA", "H100 Tensor Core GPU datasheet", "https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306"],
          ["Williams, Waterman & Patterson 2009", "Roofline: An Insightful Visual Performance Model", "https://doi.org/10.1145/1498765.1498785"],
          ["Dao et al. 2022", "FlashAttention", ax("2205.14135")],
          ["Dao 2023", "FlashAttention-2", ax("2307.08691")],
          ["Shah et al. 2024", "FlashAttention-3", ax("2407.08608")],
          ["Micikevicius et al. 2018", "Mixed Precision Training", ax("1710.03740")],
          ["Micikevicius et al. 2022", "FP8 Formats for Deep Learning", ax("2209.05433")],
        ]],
        ["Distributed training", [
          ["Patarasuk & Yuan 2009", "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations", "https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf"],
          ["Thakur, Rabenseifner & Gropp 2005", "Optimization of Collective Communication Operations in MPICH", "https://doi.org/10.1177/1094342005051521"],
          ["Rajbhandari et al. 2020", "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", ax("1910.02054")],
          ["Shoeybi et al. 2019", "Megatron-LM", ax("1909.08053")],
          ["Huang et al. 2019", "GPipe", ax("1811.06965")],
          ["Narayanan et al. 2021", "Efficient Large-Scale Language Model Training on GPU Clusters", ax("2104.04473")],
          ["Young 1974", "A First Order Approximation to the Optimum Checkpoint Interval", "https://doi.org/10.1145/361147.361115"],
        ]],
        ["Serving", [
          ["Yu et al. 2022", "Orca: A Distributed Serving System for Transformer-Based Generative Models", "https://www.usenix.org/conference/osdi22/presentation/yu"],
          ["Kwon et al. 2023", "Efficient Memory Management for LLM Serving with PagedAttention (vLLM)", ax("2309.06180")],
          ["Leviathan, Kalman & Matias 2023", "Fast Inference from Transformers via Speculative Decoding", ax("2211.17192")],
          ["Chen et al. 2023", "Accelerating Large Language Model Decoding with Speculative Sampling", ax("2302.01318")],
          ["Frantar et al. 2023", "GPTQ", ax("2210.17323")],
          ["Lin et al. 2024", "AWQ: Activation-aware Weight Quantization", ax("2306.00978")],
          ["Zhong et al. 2024", "DistServe: Disaggregating Prefill and Decoding", ax("2401.09670")],
        ]],
        ["Post-training", [
          ["Ouyang et al. 2022", "Training Language Models to Follow Instructions with Human Feedback", ax("2203.02155")],
          ["Bai et al. 2022", "Constitutional AI: Harmlessness from AI Feedback", ax("2212.08073")],
          ["Rafailov et al. 2023", "Direct Preference Optimization", ax("2305.18290")],
          ["Shao et al. 2024", "DeepSeekMath (introduces GRPO)", ax("2402.03300")],
          ["Sheng et al. 2024", "HybridFlow: A Flexible and Efficient RLHF Framework (veRL)", ax("2409.19256")],
        ]],
      ];
      const list = el.querySelector(".src-list");
      GROUPS.forEach(([g, items]) => list.appendChild(h("li", { class: "src-group" }, h("span", { class: "menu-pass-name" }, g),
        h("ul", null, items.map(([who, title, url]) => h("li", null, h("a", { href: url, target: "_blank", rel: "noopener" }, title), h("span", { class: "src-who" }, ` · ${who}`)))))));
    },
  };
})();

/* Pass openers: recap the facts so far and list the pass's slides. */
(function () {
  "use strict";
  const { h } = window.UI;
  const W = window.Widgets;
  W.opener = {
    init(el, deck) {
      const sl = el.closest(".slide");
      const me = deck.slides.indexOf(sl);
      const pass = sl.dataset.pass;
      const facts = deck.slides.slice(0, me).map((x) => x.dataset.fact).filter(Boolean).slice(-4);
      const fl = el.querySelector(".op-facts");
      if (!facts.length) fl.appendChild(h("li", { class: "empty" }, "Only the query itself: SELECT name FROM users WHERE id = 7;, one of 270 queries our grammar allows."));
      facts.forEach((f) => { const li = h("li"); li.innerHTML = f; fl.appendChild(li); });
      const list = el.querySelector(".op-slides");
      deck.slides.forEach((x, i) => {
        if (x.dataset.pass !== pass || x === sl) return;
        list.appendChild(h("li", null, h("button", { type: "button", class: "op-item", onclick: () => deck.go(i) },
          h("span", { class: "menu-num" }, String(i + 1)), h("span", null, x.dataset.title))));
      });
    },
  };
})();
