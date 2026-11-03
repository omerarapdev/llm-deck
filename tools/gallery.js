/* Component gallery: one sheet per core figure type, with a realistic spec.
 * Rendered by gallery.html (generated from index.html by tools/make_gallery.mjs).
 * Writers: copy a spec from here as a starting point. */
(function () {
  "use strict";
  Deck.part({ n: 0, title: "Component gallery", short: "Gallery" });
  Deck.chapter({ title: "Opener" });
  Deck.slide({ id: "g-part", layout: "part", title: "Component gallery", body: "<p>A part opener: big numeral, a short promise, and the chapter list with sheet ranges, generated automatically.</p>", take: "Pick the right figure for each idea." });

  Deck.chapter({ title: "Text figures" });
  Deck.slide({
    id: "g-table", title: "table: rows and columns", body: "<p>Use for small data, comparisons with numbers, and notation.</p>",
    fig: { type: "table", caption: "Squared error for three points, w = 0.5", cols: ["x", "y", "ŷ = w·x", "(ŷ − y)²"], align: ["r", "r", "r", "r"],
      rows: [["1", "2", "0.5", "2.25"], ["2", "3", "1.0", "4.00"], ["3", "7", "1.5", "30.25"], ["", "", "mean", "<b>12.17</b>"]], highlight: [2], note: "The third point contributes most of the loss." },
  });
  Deck.slide({
    id: "g-steps", title: "steps: a derivation one line at a time", body: "<p>Use for worked derivations. Each step can carry a pencil note.</p>",
    fig: { type: "steps", caption: "Derivative of the squared error", start: 0, steps: [
      { html: String.raw`\[L(w) = (w x - y)^2\]`, note: "one data point" },
      { html: String.raw`\[\frac{dL}{dw} = 2 (w x - y) \cdot \frac{d}{dw}(w x - y)\]`, note: "chain rule" },
      { html: String.raw`\[\frac{dL}{dw} = 2 (w x - y)\, x\]` },
      { html: String.raw`\[x = 3,\ y = 7,\ w = 0.5:\quad 2 (1.5 - 7) \cdot 3 = -33\]`, note: "negative: increase w" },
    ] },
  });
  Deck.slide({
    id: "g-code", title: "code: a listing with line notes", body: "<p>Use for Python, PyTorch and CUDA. Notes are keyed by line number and stepped through.</p>",
    fig: { type: "code", lang: "cuda", caption: "Vector addition kernel", code: `__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread index
  if (i < n) c[i] = a[i] + b[i];
}

// launch: 4 blocks of 256 threads cover n = 1000
add<<<4, 256>>>(a, b, c, 1000);`, notes: { 1: "<code>__global__</code> marks a function that runs on the GPU and is launched from the CPU.", 2: "Each thread computes its own index from its block number and its position in the block.", 3: "Threads past the end of the array do nothing: 4 × 256 = 1,024 threads, 24 of them idle.", 7: "<code>&lt;&lt;&lt;blocks, threads&gt;&gt;&gt;</code> sets the grid shape." } },
  });
  Deck.slide({
    id: "g-specs", title: "specs: a datasheet", body: "<p>Use for hardware and model specifications.</p>",
    fig: { type: "specs", caption: "NVIDIA H100 SXM, selected specifications", items: [
      { v: "132", label: "SMs" }, { v: "80 GB", label: "HBM3", note: "3.35 TB/s" }, { v: "989 TFLOP/s", label: "BF16 dense" }, { v: "50 MB", label: "L2 cache" }] },
  });
  Deck.slide({
    id: "g-compare", title: "compare: two columns", body: "<p>Use for two designs side by side.</p>",
    fig: { type: "compare", caption: "Two ways to batch requests", left: { title: "Static batching", items: ["Wait for a full batch", "Every request finishes together", "Short requests wait for long ones"] }, right: { title: "Continuous batching", items: ["Admit requests every step", "Finished requests leave at once", "GPU stays full"] } },
  });
  Deck.slide({
    id: "g-quiz", title: "check: a question", body: "<p>A check with no figure becomes the plate.</p>",
    check: { q: "What is the derivative of \\(x^2\\) at \\(x = 3\\)?", options: ["3", "6", "9", "2"], answer: 1, why: "\\(\\frac{d}{dx}x^2 = 2x\\), and \\(2 \\cdot 3 = 6\\)." },
  });

  Deck.chapter({ title: "Plots and data" });
  Deck.slide({
    id: "g-plot", title: "plot: functions with live parameters", body: "<p>Functions are JavaScript. Parameters become sliders; readouts are computed live.</p>",
    fig: { type: "plot", caption: "A line through three points; drag w to change the slope", x: [0, 4], y: [-1, 9], xlabel: "x", ylabel: "y",
      params: [{ name: "w", label: "slope w", min: 0, max: 3, step: 0.01, value: 0.5 }],
      fns: [{ f: (x, p) => p.w * x, label: (p) => `y = ${p.w.toFixed(2)}x` }],
      points: [[1, 2], [2, 3], [3, 7]], residuals: true, residualSquares: true,
      readouts: [{ label: "mean squared error", f: (p) => ([[1, 2], [2, 3], [3, 7]].reduce((a, [x, y]) => a + (p.w * x - y) ** 2, 0) / 3), fmt: "f3" }] },
  });
  Deck.slide({
    id: "g-chart", title: "chart: data series", body: "<p>For precomputed data: training curves, measurements. Log axes supported.</p>",
    fig: { type: "chart", caption: "Training loss of the running-example model", x: { domain: [0, 6000], label: "step" }, y: { domain: [0, 5], label: "loss (nats)" },
      series: [{ points: window.TinyLM.data.exactCurve.map((r) => [r[0], r[1]]), cls: "s1", label: "exact loss" }],
      hlines: [{ y: window.TinyLM.data.floor.reduce((a, b) => a + b, 0) / window.TinyLM.data.floor.length, label: "floor 0.694", cls: "red" }], legend: [["s1", "exact expected loss"]] },
  });
  Deck.slide({
    id: "g-bars", title: "bars: magnitudes", body: "<p>For comparing a handful of values.</p>",
    fig: { type: "bars", caption: "Memory traffic per decode step", fmt: "f1", items: [{ label: "weights", value: 16.1, text: "16.1 GB" }, { label: "KV cache", value: 2.7, text: "2.7 GB", hi: true }, { label: "activations", value: 0.1, text: "0.1 GB" }] },
  });
  Deck.slide({
    id: "g-heat", title: "heat: a matrix of numbers", body: "<p>Signed values use the diverging scale (blue negative, red positive).</p>",
    fig: { type: "heat", caption: "A 4 × 4 weight matrix", rows: ["r1", "r2", "r3", "r4"], cols: ["c1", "c2", "c3", "c4"], matrix: [[0.5, -1.2, 0.1, 0.8], [-0.3, 0.9, -1.5, 0.2], [1.1, 0.0, -0.4, -0.9], [0.2, 0.6, 1.3, -0.1]] },
  });
  Deck.slide({
    id: "g-sequence", title: "sequence: values under tokens", body: "<p>For per-token quantities.</p>",
    fig: { type: "sequence", caption: "Loss at each position of the query", tokens: ["SELECT", " name", " FROM", " users"], focus: [1], rows: [{ label: "p(next)", values: [0.085, 1.0, 1.0, 1.0] }, { label: "loss", values: [2.46, 0, 0, 0], dp: 2 }] },
  });
  Deck.slide({
    id: "g-timeline", title: "timeline: who is busy when", body: "<p>For pipelines, overlap of compute and communication, schedules.</p>",
    fig: { type: "timeline", caption: "Overlapping compute with gradient communication", rows: ["GPU compute", "Network"], total: 10, unit: "ms", items: [
      { row: 0, start: 0, len: 3, label: "layer 3 bwd", kind: "compute" }, { row: 0, start: 3, len: 3, label: "layer 2 bwd", kind: "compute" }, { row: 0, start: 6, len: 3, label: "layer 1 bwd", kind: "compute" },
      { row: 1, start: 3, len: 2.5, label: "all-reduce 3", kind: "comm" }, { row: 1, start: 6, len: 2.5, label: "all-reduce 2", kind: "comm" }, { row: 1, start: 9, len: 1, label: "", kind: "idle" }] },
  });

  Deck.chapter({ title: "Mechanisms" });
  Deck.slide({
    id: "g-matmul", title: "matmul: one entry at a time", body: "<p>Steps through a matrix product cell by cell.</p>",
    fig: { type: "matmul", caption: "A 2×3 matrix times a 3×2 matrix", A: [[1, 2, 0], [0, 1, 3]], B: [[2, 1], [0, 1], [1, 4]] },
  });
  Deck.slide({
    id: "g-graph", title: "graph: forward and backward", body: "<p>Nodes are evaluated in order, then gradients flow back along each edge.</p>",
    fig: { type: "graph", caption: "L = (w·x + b − y)²", dp: 2, nodes: [
      { id: "w", op: "param", value: 0.5, x: 0, y: 0 }, { id: "x", op: "input", value: 3, x: 0, y: 1 }, { id: "b", op: "param", value: 0.2, x: 0, y: 2 }, { id: "y", op: "input", value: 7, x: 2, y: 2 },
      { id: "m", label: "m", op: "mul", in: ["w", "x"], x: 1, y: 0.5 }, { id: "z", label: "z", op: "add", in: ["m", "b"], x: 2, y: 1 },
      { id: "e", label: "e", op: "sub", in: ["z", "y"], x: 3, y: 1.5 }, { id: "L", label: "L", op: "square", in: ["e"], x: 4, y: 1.5 }] },
  });
  Deck.slide({
    id: "g-network", title: "network: a small neural network", body: "<p>Weights, biases and activations; inputs become sliders.</p>",
    fig: { type: "network", caption: "2 inputs, 3 hidden units (ReLU), 1 output", layers: [2, 3, 1], acts: ["relu", "linear"], input: [1, 0.5],
      W: [[[1, -1], [0.5, 1], [-1, 2]], [[1, -0.5, 0.8]]], b: [[0, 0.1, -0.2], [0.05]], inputSliders: [{ min: -2, max: 2 }, { min: -2, max: 2 }] },
  });
  Deck.slide({
    id: "g-gd1", title: "gd: gradient descent in one dimension", body: "<p>Step, take 10 steps, change the learning rate.</p>",
    fig: { type: "gd", caption: "L(w) = (w − 3)² + 1, from w = −1", f: (w) => (w - 3) ** 2 + 1, df: (w) => 2 * (w - 3), start: -1, lr: 0.1, x: [-2, 7], y: [0, 18], lrMax: 1.1 },
  });
  Deck.slide({
    id: "g-gd2", title: "gd: gradient descent on a surface", body: "<p>A long narrow valley: the classic case where plain gradient descent zig-zags.</p>",
    fig: { type: "gd", dim: 2, caption: "L = w₁² + 10 w₂²", f: (a, b) => a * a + 10 * b * b, grad: (a, b) => [2 * a, 20 * b], start: [-4, 1.5], lr: 0.09, x: [-5, 5], y: [-2, 2], min: [0, 0], lrMax: 0.12 },
  });
  Deck.slide({
    id: "g-diagram", title: "diagram: boxes, arrows, notes, dimensions", body: "<p>A grid of units (40px). Boxes, groups, arrows with labels, pencil notes, dimension lines.</p>",
    fig: { type: "diagram", caption: "Memory hierarchy of one GPU", w: 14, h: 6, nodes: [
      { id: "reg", x: 0.5, y: 0.5, w: 3, h: 1.2, label: "Registers", sub: "256 KB per SM", kind: "hi" },
      { id: "smem", x: 0.5, y: 2.3, w: 3, h: 1.2, label: "Shared memory", sub: "up to 228 KB per SM", kind: "blue" },
      { id: "l2", x: 5.5, y: 2.3, w: 3, h: 1.2, label: "L2 cache", sub: "50 MB" },
      { id: "hbm", x: 10.5, y: 2.3, w: 3, h: 1.2, label: "HBM", sub: "80 GB", kind: "dark" }],
      edges: [{ from: "reg", to: "smem", both: true }, { from: "smem", to: "l2", both: true, label: "on chip" }, { from: "l2", to: "hbm", both: true, label: "3.35 TB/s" }],
      groups: [{ x: 0.2, y: 0.2, w: 3.6, h: 3.6, label: "one SM" }],
      notes: [{ x: 5.5, y: 1.4, text: "faster, smaller ←" }],
      dims: [{ at: [5.5, 4.6, 13.5, 4.6], label: "off the SM" }] },
  });
  Deck.slide({
    id: "g-calc", title: "calc: a calculator", body: "<p>Inputs become sliders (log sliders for wide ranges) or segmented buttons; outputs are computed live.</p>",
    fig: { type: "calc", caption: "Training compute ≈ 6 × parameters × tokens", inputs: [
      { id: "N", label: "parameters", min: 1e6, max: 1e12, value: 8e9, log: true, fmt: "sci" },
      { id: "D", label: "training tokens", min: 1e9, max: 3e13, value: 1.5e13, log: true, fmt: "sci" },
      { id: "gpu", label: "GPU", options: [{ label: "A100", value: 312e12 }, { label: "H100", value: 989e12 }], value: 989e12 }],
      outputs: [{ label: "FLOPs", f: (v) => 6 * v.N * v.D, fmt: "sci" }, { label: "GPU-days at 40% MFU", f: (v) => (6 * v.N * v.D) / (0.4 * v.gpu) / 86400, fmt: "int" }],
      note: (v) => `6 × ${window.UI.format(v.N, "sci")} × ${window.UI.format(v.D, "sci")}` },
  });
  Deck.slide({
    id: "g-vectors", title: "vectors: drag two arrows", body: "<p>Dot product, lengths and angle update as you drag.</p>",
    fig: { type: "vectors", caption: "Drag the arrow tips", a: [3, 1], b: [1, 2], showSum: true },
  });
  Deck.slide({
    id: "g-tinylm", title: "tinylm: the trained model's widgets", example: "tokens", body: "<p>The interactive widgets from the first version, running on the trained model. See AUTHORING.md for the list.</p>",
    fig: { type: "tinylm", widget: "predict", caption: "Next-token prediction from the trained model" },
  });
})();
