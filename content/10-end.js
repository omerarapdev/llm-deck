/* Part 10 — Wrap-up: the query's whole journey, interview map, reading list,
 * every source cited in the set, and how the set was checked. Links resolve
 * at render time from the other parts' chapter titles, so they follow edits. */
(function () {
  "use strict";
  const { h, s, clear } = window.UI;
  const M = window.TinyLM;
  const pad3 = (n) => String(n).padStart(3, "0");

  // Find the first sheet of the chapter in part `n` whose title matches `re`; fall back to the part opener.
  function find(n, re) {
    const p = Deck.parts.find((x) => x.n === n);
    if (!p || !p.slides.length) return null;
    const c = p.chapters.find((x) => re.test(x.title));
    return (c ? c.slides[0] : p.slides[0]);
  }
  const link = (sl, label) => sl
    ? h("button", { type: "button", class: "end-link", onclick: () => Deck.go(sl.index) }, label || sl.chapter.title, h("span", { class: "end-no" }, pad3(sl.index + 1)))
    : h("span", { class: "end-link missing" }, label || "—");

  // Fig: the query's journey, stage by stage, with the sheet that explains each stage.
  Deck.fig("end-journey", (el) => {
    const ids = M.runningIds, tr = M.forward(ids.slice(0, M.config.T));
    const loss = tr.probs.reduce((a, p, i) => a - Math.log(p[ids[i + 1]]), 0) / M.config.T;
    const kvBytes = 2 * M.config.L * M.config.d * 4 * M.config.T;
    const stages = [
      ["Text", `<code>${M.data.running}</code>`, find(4, /text as numbers|characters|bytes/i)],
      ["Tokens", `9 tokens, ids ${ids.join(", ")}`, find(4, /tokeni[sz]/i)],
      ["Vectors", "each id selects a row of 16 numbers, plus a position row", find(4, /neural language|embedding/i)],
      ["Attention", "each position mixes earlier positions, 2 layers × 2 heads", find(5, /self-attention|attention step/i)],
      ["Probabilities", `113 logits per position, softmax; p(users | … FROM) = ${tr.probs[2][ids[3]].toFixed(3)}`, find(2, /classification/i)],
      ["Loss", `average −ln p over 8 positions = ${loss.toFixed(3)} nats`, find(2, /loss/i)],
      ["Gradients", "backpropagation gives ∂loss/∂w for all 9,968 parameters", find(3, /backprop/i)],
      ["Update", "Adam moves every parameter; 6,000 steps of 128 queries", find(3, /deep networks|optimi|train/i)],
      ["GPU", "every step above is a handful of matrix multiplies and kernels", find(6, /matrix multiplication|matmul/i)],
      ["Cluster", "at Llama scale: 16,384 GPUs, several kinds of parallelism at once", find(7, /model parallel/i)],
      ["Serving", `prefill, then decode with a KV cache (${kvBytes.toLocaleString("en-US")} bytes for our 8 positions in FP32)`, find(8, /two phases|prefill|kv cache/i)],
      ["Assistant", "fine-tuning and RL turn the predictor into an assistant", find(9, /base model|supervised/i)],
    ];
    const list = h("ol", { class: "end-journey" });
    stages.forEach(([name, what, sl], i) => list.appendChild(h("li", null,
      h("span", { class: "end-step" }, String(i + 1).padStart(2, "0")),
      h("span", { class: "end-name" }, name),
      h("span", { class: "end-what", html: what }),
      link(sl))));
    el.appendChild(list);
  });

  // Fig: interview topics mapped to chapters of this set.
  Deck.fig("end-interview", (el, sp) => {
    const t = h("table", { class: "f-table" }, h("thead", null, h("tr", null, h("th", null, "Interview topic"), h("th", null, "Read"))));
    const tb = h("tbody");
    for (const [topic, refs] of sp.rows) {
      const cell = h("td", { class: "end-refs" });
      refs.forEach(([n, re, label]) => cell.appendChild(link(find(n, re), label)));
      tb.appendChild(h("tr", null, h("th", null, topic), cell));
    }
    t.appendChild(tb);
    el.appendChild(h("div", { class: "scroll-x" }, t));
  });

  // Fig: every source cited anywhere in the set, with the sheets that cite it.
  Deck.fig("end-sources", (el) => {
    const bySrc = new Map();
    for (const sl of Deck.slides) for (const src of sl.src || []) {
      const k = src.trim();
      if (!bySrc.has(k)) bySrc.set(k, []);
      bySrc.get(k).push(sl);
    }
    const items = [...bySrc.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    const box = h("div", { class: "end-src" });
    box.appendChild(h("p", { class: "f-note" }, `${items.length} sources cited on ${new Set([...bySrc.values()].flat()).size} sheets.`));
    const ol = h("ol", { class: "end-src-list" });
    for (const [src, sheets] of items) {
      const li = h("li", null, h("span", { class: "end-src-t" }, src), " ");
      sheets.slice(0, 6).forEach((sl) => li.appendChild(h("button", { type: "button", class: "term-go", onclick: () => Deck.go(sl.index) }, pad3(sl.index + 1))));
      if (sheets.length > 6) li.appendChild(h("span", { class: "dim" }, ` +${sheets.length - 6}`));
      ol.appendChild(li);
    }
    box.appendChild(ol);
    el.appendChild(box);
  });

  // Fig: the set's own statistics, counted live.
  Deck.fig("end-stats", (el) => {
    const figs = Deck.slides.filter((x) => x.fig).length;
    const custom = new Set(Deck.slides.filter((x) => x.fig).map((x) => x.fig.type)).size;
    const checks = Deck.slides.filter((x) => x.check).length;
    const tile = (v, l) => h("div", { class: "f-spec" }, h("div", { class: "f-spec-v" }, v), h("div", { class: "f-spec-l" }, l));
    el.appendChild(h("div", { class: "f-specs" },
      tile(String(Deck.slides.length), "sheets"), tile(String(Deck.parts.length), "parts"), tile(String(Deck.chapters.length), "chapters"),
      tile(String(figs), "plates"), tile(String(custom), "figure types"), tile(String(checks), "checks"), tile(String(Deck.terms.length), "terms defined"),
      tile(M.config.params.toLocaleString("en-US"), "model parameters")));
  });

  Deck.part({ n: 10, title: "Wrap-up", short: "Wrap-up" });
  Deck.chapter({ title: "The whole journey" });

  Deck.slide({
    id: "end-part",
    layout: "part",
    title: "Wrap-up",
    body: "<p>The last sheets put the set back together: the whole path of the query in one list, a map from interview topics to chapters, what to read next, every source the set cites, and how the numbers were checked.</p>",
    take: "Trace any step of training or serving back to the sheet that explains it.",
  });

  Deck.slide({
    id: "end-journey",
    title: "Everything that happened to one query",
    example: "tokens ids",
    layout: "wide",
    body: `<p>The query entered as 36 characters. It became 9 tokens, then 9 vectors of 16 numbers, then 8 probability distributions over 113 tokens, then one loss, then 9,968 gradients and an update. At scale the same steps run as GPU kernels across a cluster, and at serving time as prefill and decode with a cache. Each stage below links to the chapter that explains it.</p>`,
    note: "Every number on this sheet is recomputed from the trained weights when you open it.",
    fig: { type: "end-journey", caption: "The running example, from text to served assistant" },
  });

  Deck.chapter({ title: "Where to go from here" });

  Deck.slide({
    id: "end-interview",
    title: "Interview topics, mapped to chapters",
    layout: "wide",
    body: `<p>LLM training and inference interviews at frontier labs mix ML fundamentals with systems design. The table maps the usual topics to the chapters that cover them. Every link opens the first sheet of the chapter.</p>
<p>For each topic, be able to do the arithmetic without notes: parameter and FLOP counts, memory per GPU, KV cache size, decode step time from bandwidth, all-reduce time from the α–β model.</p>`,
    fig: {
      type: "end-interview", caption: "Topic → chapters",
      rows: [
        ["Loss, gradients, backpropagation", [[2, /loss/i], [2, /gradient descent/i], [3, /backprop/i]]],
        ["Transformer architecture, parameter and FLOP counts", [[5, /self-attention|attention step/i], [5, /block/i], [5, /modern|variant/i]]],
        ["GPU execution model and kernels", [[6, /inside|h100/i], [6, /cuda programming|programming model/i], [6, /execut|warp/i], [6, /memory access/i]]],
        ["Matmul and attention kernels", [[6, /matrix multiplication/i], [6, /kernels for transformers|transformer/i]]],
        ["Training memory and mixed precision", [[7, /memory/i], [7, /number formats|precision/i]]],
        ["Collectives and parallelism", [[7, /collective/i], [7, /data parallel/i], [7, /model parallel/i]]],
        ["Inference: KV cache, batching, latency", [[8, /two phases/i], [8, /kv cache/i], [8, /batching/i]]],
        ["Inference optimizations", [[8, /faster|decode/i], [8, /parallelism/i], [8, /serving systems/i]]],
        ["Post-training and RL", [[9, /supervised/i], [9, /preference/i], [9, /verifiable|rl with/i], [9, /infrastructure/i]]],
      ],
    },
  });

  Deck.slide({
    id: "end-reading",
    title: "Papers to read next, in order",
    body: `<p>Each of these is the primary source for a part of this set. Read them in this order; each takes an evening.</p>
<ol>
<li>Vaswani et al., <i>Attention Is All You Need</i>, 2017.</li>
<li>Kaplan et al., <i>Scaling Laws for Neural Language Models</i>, 2020; Hoffmann et al., <i>Training Compute-Optimal Large Language Models</i>, 2022.</li>
<li>Dao et al., <i>FlashAttention</i>, 2022, and <i>FlashAttention-2</i>, 2023.</li>
<li>Rajbhandari et al., <i>ZeRO</i>, 2020; Narayanan et al., <i>Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM</i>, 2021.</li>
<li>Kwon et al., <i>Efficient Memory Management for Large Language Model Serving with PagedAttention</i>, 2023.</li>
<li>Leviathan et al., <i>Fast Inference from Transformers via Speculative Decoding</i>, 2023.</li>
<li>Ouyang et al., <i>Training language models to follow instructions with human feedback</i>, 2022; Rafailov et al., <i>Direct Preference Optimization</i>, 2023.</li>
<li>Dubey et al., <i>The Llama 3 Herd of Models</i>, 2024; DeepSeek-AI, <i>DeepSeek-V3 Technical Report</i>, 2024.</li>
</ol>`,
    note: "Then write the code: a transformer, a CUDA matmul, a ring all-reduce.",
    take: "Read the paper, then reproduce one table or figure from it at small scale. That is the fastest way to find what you did not understand.",
    src: ["Vaswani et al., Attention Is All You Need, 2017", "Dubey et al., The Llama 3 Herd of Models, 2024"],
  });

  Deck.slide({
    id: "end-sources",
    title: "Every source cited in this set",
    layout: "wide",
    body: "<p>Facts about real systems (hardware specifications, model configurations, published results) carry their source on the sheet that states them. This list collects them, with the sheets that cite each one.</p>",
    fig: { type: "end-sources", caption: "Sources and the sheets that cite them" },
  });

  Deck.slide({
    id: "end-colophon",
    title: "How this set was made and checked",
    body: `<p>The running-example model was trained for this set with PyTorch: a 2-layer, 2-head transformer with 9,968 parameters, trained with Adam for 6,000 steps on queries sampled from the grammar, single-threaded with seed 7, so the run is bit-reproducible. Its weights ship with the page. A JavaScript copy of the forward pass matches PyTorch's logits to within 0.0001.</p>
<p>Every computed number in the text (worked examples, model probabilities, FLOP and byte counts) is recomputed by a verification script for its part, which fails if the text and the computation disagree. Every sheet is rendered in a headless browser at desktop and phone widths, in both themes, and checked for errors and overflow.</p>`,
    fig: { type: "end-stats", caption: "This set, counted when you opened it" },
    take: "If you find an error, the sheet number in the margin identifies it exactly.",
  });
})();
