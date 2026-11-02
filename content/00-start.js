/* Part 0 — Orientation: what this set is, the running example, how to read a sheet. */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip } = window.UI;
  const M = window.TinyLM;

  // Fig: the running query with the probability the trained model gives each next token.
  Deck.fig("cover-ribbon", (el) => {
    const ids = M.runningIds, toks = M.running;
    const tr = M.forward(ids.slice(0, M.config.T));
    const W = 640, H = 214, n = toks.length, colW = W / n;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "cv-svg", role: "img", "aria-label": "Probability the model assigns to each next token of the query" });
    const base = 150, top = 22;
    svg.appendChild(s("line", { x1: 0, x2: W, y1: base, y2: base, class: "cv-base" }));
    svg.appendChild(s("line", { x1: 0, x2: W, y1: top, y2: top, class: "cv-one" }));
    svg.appendChild(s("text", { x: 4, y: top - 6, class: "cv-ax" }, "p = 1"));
    for (let i = 1; i < n; i++) {
      const p = tr.probs[i - 1][ids[i]];
      const x = i * colW + colW * 0.22, w = colW * 0.56, hgt = (base - top) * p;
      svg.appendChild(s("rect", { x, y: base - hgt, width: w, height: Math.max(hgt, 1), class: p > 0.99 ? "cv-bar sure" : "cv-bar", "data-tip": `p(${M.running[i].trim()} | ${toks.slice(0, i).join("").trim()}) = ${p.toFixed(4)}` }));
      svg.appendChild(s("text", { x: x + w / 2, y: base - hgt - 6, class: "cv-p", "text-anchor": "middle" }, p > 0.9995 ? "1.000" : p.toFixed(3)));
    }
    toks.forEach((t, i) => svg.appendChild(s("text", { x: i * colW + colW / 2, y: base + 24, class: "cv-tok", "text-anchor": "middle" }, t.trim())));
    svg.appendChild(s("text", { x: W - 4, y: base + 50, class: "cv-note", "text-anchor": "end" }, "blue: the model is certain (p > 0.99)"));
    el.appendChild(h("div", { class: "scroll-x" }, svg));
  });

  Deck.part({ n: 0, title: "Orientation", short: "Orientation" });

  Deck.chapter({ title: "What this set is" });

  Deck.slide({
    id: "cover",
    title: "One query, end to end",
    example: "tokens",
    body: `<p class="lede">This drawing set explains how large language models are trained and served. It starts with derivatives and loss functions and ends with GPU kernels, training clusters and inference servers.</p>
<p>Everything is built around one SQL query, <code>SELECT name FROM users WHERE id = 7;</code>, and one real model trained on queries like it. The model has <span class="num">9,968</span> parameters and runs inside this page. Every number shown about it is computed from its trained weights while you read.</p>
<p>A frontier model differs from this one in size, data and hardware. The computation is the same: the same matrix multiplications, the same loss, the same gradients, the same attention. When the set reaches GPUs and clusters, the sizes switch to Llama 3 and H100 numbers and the query stays.</p>`,
    note: "Use <b>← →</b> or swipe. <b>G</b> opens the list of every sheet.",
    fig: { type: "cover-ribbon", caption: "The trained model reads the query left to right. Each bar is the probability it gave the token that really came next." },
    take: "After <code>FROM</code> the model is certain: <code>users</code> is the only table with a <code>name</code> column. After <code>=</code> it is not: any digit 1 to 9 can follow, so the best it can do is about 1/9.",
  });

  Deck.slide({
    id: "running-example",
    title: "The running example is a tiny SQL language",
    example: "text",
    body: `<p>The model was trained on queries from one grammar: <code>SELECT</code> a column <code>FROM</code> a table <code>WHERE</code> a different column <code>=</code> a digit.</p>
<p>A query is generated in four independent choices: a table (uniform over 3), an output column (uniform over that table's columns), a filter column (uniform over the remaining ones) and a digit from 1 to 9.</p>
<p>That gives <span class="num">108 + 108 + 54 = 270</span> distinct queries. They are not equally likely. A <code>products</code> query has probability \\(\\tfrac13 \\cdot \\tfrac13 \\cdot \\tfrac12 \\cdot \\tfrac19 = \\tfrac1{162}\\), a <code>users</code> query \\(\\tfrac13 \\cdot \\tfrac14 \\cdot \\tfrac13 \\cdot \\tfrac19 = \\tfrac1{324}\\).</p>
<p>The tokenizer was built from <span class="num">2,000</span> queries sampled this way. The model itself trained on fresh samples: 128 queries per step for 6,000 steps, <span class="num">768,000</span> in all. On average it saw each distinct query 2,844 times: about 2,370 times for each <code>users</code> or <code>orders</code> query and 4,741 times for each <code>products</code> query.</p>
<p>Because the grammar is known exactly, we can compute the best loss any model could reach on it. Real text has no such formula, which is part of why the toy is useful.</p>`,
    note: "Our query is one of the 108 <b>users</b> queries.",
    fig: {
      type: "table", caption: "The schema the grammar draws from",
      cols: ["Table", "Columns", "Output × filter × digit", "Queries"], align: ["l", "l", "r", "r"],
      rows: [
        ["<code>users</code>", "<code>id</code> <code>name</code> <code>email</code> <code>age</code>", "4 × 3 × 9", "108"],
        ["<code>orders</code>", "<code>id</code> <code>user_id</code> <code>total</code> <code>status</code>", "4 × 3 × 9", "108"],
        ["<code>products</code>", "<code>id</code> <code>title</code> <code>price</code>", "3 × 2 × 9", "54"],
        ["", "", "Total", "<b>270</b>"],
      ],
      highlight: [0],
      note: "The filter column must differ from the output column, so a table with k columns gives k × (k − 1) column pairs.",
    },
    terms: [{ term: "Running example", def: "The query <code>SELECT name FROM users WHERE id = 7;</code> and the 9,968-parameter model trained on its grammar. Most sheets return to it." }],
  });

  Deck.slide({
    id: "the-model",
    title: "The model you will take apart",
    example: "tokens ids",
    body: `<p>The model is a transformer, the architecture behind every current large language model. Ours is the smallest one that still shows every part working.</p>
<p>It splits text into tokens from a vocabulary of 113, turns each token into a list of 16 numbers, passes those lists through 2 layers that each contain 2 attention heads and a small feed-forward network, and ends with one probability for each of the 113 tokens.</p>
<p>All of that is fixed by <span class="num">9,968</span> numbers, its parameters. Training chose them in 6,000 steps of gradient descent, each step on 128 freshly sampled queries. The chips above show our query as the model sees it: 9 tokens, each with its id in the vocabulary.</p>
<p>You will be able to compute every one of these steps by hand by the end of Part 5.</p>`,
    note: "Llama 3.1 405B has about <b>40 million</b> times as many parameters.",
    fig: {
      type: "specs", caption: "Our model next to Llama 3.1 405B",
      items: [
        { v: "9,968", label: "Parameters", note: "Llama 3.1 405B: 405 billion" },
        { v: "113", label: "Vocabulary", note: "Llama 3: 128,256 tokens" },
        { v: "16", label: "Numbers per token", note: "Llama 3.1 405B: 16,384" },
        { v: "2 × 2", label: "Layers × heads", note: "Llama 3.1 405B: 126 × 128" },
        { v: "8", label: "Context (tokens)", note: "Llama 3.1: 128K" },
        { v: "6.1M", label: "Training tokens", note: "6,000 steps × 128 queries × 8 positions; Llama 3.1 405B: 15.6 trillion" },
      ],
      note: "Llama figures from Meta's paper <i>The Llama 3 Herd of Models</i> (2024).",
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024"],
  });

  Deck.chapter({ title: "How to read a sheet" });

  Deck.slide({
    id: "how-to-read",
    title: "Each sheet makes one point and shows it",
    body: `<p>Every slide in this set is drawn as a sheet in an engineering drawing set. The parts are always in the same places.</p>
<ul>
<li><b>Margin.</b> The sheet number, the chapter code, and pencil notes: short remarks that are worth remembering but not essential.</li>
<li><b>Text.</b> One idea, explained in full sentences, usually with a worked example in real numbers.</li>
<li><b>Plate.</b> The figure, labelled <i>Fig.</i> with the sheet number. Most plates are interactive: sliders, step buttons and hover readouts.</li>
<li><b>Note.</b> The one sentence to keep if you keep nothing else.</li>
<li><b>Check.</b> A question with an explanation behind each answer.</li>
</ul>
<p>Terms in bold are defined where they first appear and collected under <b>Terms</b> in the top bar.</p>`,
    note: "Hover or tap anything with a dotted outline to read its exact value.",
    check: {
      q: "A sheet says the model gives <code>users</code> probability 1.000 after <code>SELECT name FROM</code>. Where would you look for the exact value behind that rounded number?",
      options: ["In the margin note", "By hovering over the bar in the plate", "In the Terms panel", "Nowhere; it is a rounded estimate"],
      answer: 1,
      why: "Plates show exact values on hover or tap. The 1.000 is a rounded display of a probability computed from the trained weights.",
    },
  });

  Deck.slide({
    id: "route",
    title: "The route: ten parts, each built on the last",
    layout: "wide",
    body: `<p>The set climbs from first principles to production systems. Parts 1 to 5 build the model itself; Parts 6 to 9 are the systems work that makes it fast and large enough to matter.</p>`,
    fig: {
      type: "diagram", caption: "Parts of the set and what each one adds", w: 24, h: 7.4, unit: 40,
      groups: [
        { x: 0.2, y: 0.3, w: 15.1, h: 3.1, label: "The model" },
        { x: 0.2, y: 3.9, w: 23.6, h: 3.2, label: "The systems around it" },
      ],
      nodes: [
        { id: "p1", x: 0.6, y: 1.0, w: 2.6, h: 1.6, label: "1 Math", sub: "derivatives, vectors" },
        { id: "p2", x: 3.6, y: 1.0, w: 2.6, h: 1.6, label: "2 Learning", sub: "loss, descent" },
        { id: "p3", x: 6.6, y: 1.0, w: 2.6, h: 1.6, label: "3 Networks", sub: "layers, backprop" },
        { id: "p4", x: 9.6, y: 1.0, w: 2.6, h: 1.6, label: "4 Language", sub: "tokens, n-grams" },
        { id: "p5", x: 12.6, y: 1.0, w: 2.4, h: 1.6, label: "5 Transformer", sub: "attention", kind: "hi" },
        { id: "p6", x: 0.6, y: 4.6, w: 5.2, h: 1.6, label: "6 GPUs & CUDA", sub: "SMs, warps, memory, kernels", kind: "blue" },
        { id: "p7", x: 6.6, y: 4.6, w: 5.2, h: 1.6, label: "7 Training at scale", sub: "precision, parallelism" },
        { id: "p8", x: 12.6, y: 4.6, w: 5.2, h: 1.6, label: "8 Inference", sub: "KV cache, batching" },
        { id: "p9", x: 18.6, y: 4.6, w: 4.8, h: 1.6, label: "9 Post-training", sub: "SFT, RLHF, RL" },
      ],
      edges: [
        { from: "p1", to: "p2" }, { from: "p2", to: "p3" }, { from: "p3", to: "p4" }, { from: "p4", to: "p5" },
        { from: "p6", to: "p7" }, { from: "p7", to: "p8" }, { from: "p8", to: "p9" },
        { from: "p5", to: "p7", label: "train it", dashed: true },
      ],
      notes: [{ x: 16.2, y: 1.6, text: "you can build it", anchor: "start" }, { x: 16.2, y: 2.3, text: "by hand here", anchor: "start" }],
    },
    take: "If you already know some parts, the register (<b>G</b>) lists every sheet; each part opens with a contents sheet.",
  });

  Deck.slide({
    id: "prerequisites",
    title: "What you need to bring",
    body: `<p>School algebra is enough to start. Part 1 builds the rest: functions, derivatives, the chain rule, vectors, matrices, probability, logarithms and the exponential.</p>
<p>The set uses a small, fixed notation. Scalars are lowercase italic (\\(x\\), \\(w\\)). Vectors are bold lowercase (\\(\\mathbf{x}\\)). Matrices are bold uppercase (\\(\\mathbf{W}\\)). An index picks one entry: \\(x_i\\) is the \\(i\\)-th entry of \\(\\mathbf{x}\\).</p>
<p>Code is Python with PyTorch for models, and CUDA C++ for GPU kernels. Every snippet is short enough to read line by line, and the plates walk through them one line at a time.</p>`,
    note: "Systems people: Part 6 is where your instincts start paying off.",
    fig: {
      type: "table", caption: "Notation used throughout",
      cols: ["Symbol", "Read as", "Example"],
      rows: [
        ["\\(x,\\ w,\\ b\\)", "a number (scalar)", "\\(w = 0.5\\)"],
        ["\\(\\mathbf{x}\\)", "a vector, a list of numbers", "\\(\\mathbf{x} = (1, -2, 3)\\)"],
        ["\\(\\mathbf{W}\\)", "a matrix, a grid of numbers", "\\(16 \\times 113\\) in our output layer"],
        ["\\(\\sum_{i=1}^{n}\\)", "sum over i from 1 to n", "\\(\\sum_{i=1}^{3} i = 6\\)"],
        ["\\(\\frac{\\partial L}{\\partial w}\\)", "how fast L changes as w changes, others held fixed", "chapters 1.2 and 1.4"],
        ["\\(\\ln,\\ e^{x}\\)", "natural log and exponential", "\\(\\ln e^{2} = 2\\)"],
        ["\\(p(y \\mid x)\\)", "probability of y given x", "\\(p(\\texttt{users} \\mid \\texttt{FROM}) = 1.0\\)"],
      ],
    },
  });
})();
