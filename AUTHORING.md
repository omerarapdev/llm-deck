# Authoring guide

This deck is a **drawing set**: about 370 sheets that take a strong software engineer with no machine-learning background to the level of an LLM training and inference engineer at a frontier lab. Every sheet is a slide. Read this whole file before writing.

## 1. The reader

- A Staff software engineer with a PhD in HPC (MPI collectives, FPGA offload) and years on database query optimizers. Fluent in systems, C/C++/Java, complexity, caches, networks. **No** machine-learning background: has never seen a loss function, a gradient step, or a neural network written down.
- They asked for depth: *"you can't teach someone LLMs in 57 slides"*. The previous version was rejected for being too high level, skipping basics, and having no GPU/CUDA architecture. Build every idea from the ground up, then go all the way to production detail.
- They dislike analogies and story framing. They learn from **concrete numbers, worked examples, tables and figures**. Never explain X by saying it is like Y. Explain what X computes, with numbers.
- Their systems background is an asset: when a topic touches it (collectives, memory hierarchies, scheduling, cost models), go deeper and use precise terms, not less.

## 2. Voice

- Second person, present tense, plain declarative sentences. Short paragraphs (2 to 4 sentences).
- **One idea per sheet.** If a sheet needs two figures or more than ~200 words, split it.
- Define every term the first time it appears, in the sentence that uses it, and add it to `terms`.
- Every abstract statement is followed by a number. "Large learning rates diverge" is incomplete; "at lr = 1.1 on L(w) = (w − 3)², each step multiplies the error by −1.2, so after 10 steps it has grown 6.2×" is complete.
- Prefer the running example's real numbers (Section 6) over invented ones.
- Say *why* each step exists: what breaks without it, measured.
- No filler: no "crucial", "delve", "leverage", "simply", "it's worth noting", "let's dive in", no exclamation marks, no rhetorical questions as openers, no "not X but Y" constructions, no em-dash asides (use a period, a colon or parentheses). `tools/validate.mjs` warns on these.
- No analogies: no "imagine", "think of it as", "just like", "picture this". No characters, stories or metaphors.
- Titles state the point as a sentence or a precise noun phrase: "The gradient points uphill" or "Shared memory bank conflicts", not "Gradients!" or "A closer look".
- Numbers: thousands separators (9,968), a true minus sign (−0.5) in prose, units always (ms, GB, TFLOP/s, tokens). Decimal units (10⁹ bytes, 10¹² bytes/s) for bandwidth and for model sizes. HBM capacity is binary: an "80 GB" H100 has 80 GiB (85.9 × 10⁹ bytes), of which CUDA reports a little less as usable; when a calculation depends on what fits in memory, say which figure you use and why.
- Math: TeX inside `\( … \)` (inline) and `\[ … \]` (display) in any string, or the `math` field for the sheet's key equation. Write strings that contain TeX as `String.raw` template literals so backslashes stay single. Inside TeX write `\lt` and `\gt` instead of `<` and `>`.

## 3. A sheet

```js
Deck.part({ n: 2, title: "Learning from data", short: "Learning" });   // once per file
Deck.chapter({ title: "Loss functions" });                               // starts chapter 2.k

Deck.slide({
  id: "learn-mse",                     // REQUIRED, unique, kebab-case, starts with your part's prefix
  title: "Mean squared error averages the squared misses",               // REQUIRED
  body: String.raw`<p>…</p><p>…</p>`,  // REQUIRED: HTML; <p>, <ul>, <ol>, <code>, <b>, <i>, <mark>, <table>, <pre>
  math: String.raw`L(w) = \frac{1}{n} \sum_{i=1}^{n} (w x_i - y_i)^2`,    // optional key equation (or [..] or {tex, note})
  take: "The one sentence to keep.",   // optional, shown boxed under the text
  note: "A margin remark under 30 words.", // optional, handwritten in the margin
  fig: { type: "plot", caption: "…", /* type-specific fields */ },        // optional plate
  check: { q: "…", options: ["…", "…", "…"], answer: 1, why: "…" },       // optional question
  terms: [{ term: "Mean squared error (MSE)", def: "…" }],               // optional glossary entries
  src: ["Author, Title, Year"],        // REQUIRED when the sheet states facts about real systems or papers
  example: "tokens ids focus:4",       // optional running-example strip (Section 6)
  layout: "split",                     // optional: split (default with fig) | text (default without) | wide | even | part
});
```

- `layout: "wide"` puts the figure under the text at full width (use for wide diagrams, timelines, big tables). `"even"` gives text and figure equal columns. `"part"` makes the part opener (Section 5).
- A `check` on a sheet without `fig` becomes the plate. With a `fig`, it appears under the text.
- Use `<mark>` sparingly for the one phrase that matters. Use `<span class="num">` for important numbers in prose if you want them set in the numeric font.
- Use `<b>` the first time a term is defined.

### Ids
Ids appear in URLs and must be unique across the whole deck. Each part has a prefix:

| Part | Prefix | File |
|---|---|---|
| 1 Math toolkit | `math-` | `content/01-math.js` |
| 2 Learning from data | `learn-` | `content/02-learning.js` |
| 3 Neural networks | `nn-` | `content/03-networks.js` |
| 4 Language modeling | `lm-` | `content/04-language.js` |
| 5 The transformer | `tf-` | `content/05-transformer.js` |
| 6 GPUs and CUDA | `gpu-` | `content/06-gpu.js` |
| 7 Training at scale | `scale-` | `content/07-scale.js` |
| 8 Inference and serving | `serve-` | `content/08-serving.js` |
| 9 Post-training | `post-` | `content/09-post.js` |
| 10 Wrap-up | `end-` | `content/10-end.js` |

A duplicate id throws and stops the rest of that file from loading, so never reuse one.

## 4. Figures

A figure is a plate: `fig: { type, caption, ...spec }`. Every plate needs a caption that says what it shows. Interactivity is the point of this deck: most sheets should have a plate where the reader can change something and watch a number change.

Open `gallery.html` (or read `tools/gallery.js` and the screenshots in `tools/shots/gallery/`) to see every core type with a working spec. Functions in specs are real JavaScript functions.

| type | use for | key fields |
|---|---|---|
| `table` | small data, notation, comparisons | `cols`, `rows` (HTML strings), `align` (`"r"` per column), `highlight` (row indexes), `note`, `dense`, `rowHeads` |
| `steps` | derivations and worked calculations, revealed one step at a time | `steps: [{html, note}]`, `start` (index shown first; omit to show all) |
| `code` | Python, PyTorch, CUDA C++ with clickable line notes | `lang` (`python` / `cuda` / `c` / `text`), `code`, `notes: {line: html}`, `highlight: [lines]` |
| `plot` | functions with slider parameters and live readouts | `x`, `y` (domains), `xlabel`, `ylabel`, `params: [{name,label,min,max,step,value,fmt}]`, `fns: [{f(x,p), label, cls}]`, `points: [[x,y,cls]]`, `residuals`, `residualSquares`, `readouts: [{label, f(p), fmt}]`, `hlines`, `vlines`, `extra(p)` returning more chart config |
| `chart` | precomputed data series, log axes | `x: {domain, type:"log", ticks, label, fmt}`, `y`, `series: [{points, cls, label}]`, `dots`, `hlines`, `vlines`, `legend: [[cls, label]]` |
| `bars` | a few magnitudes | `items: [{label, value, text, hi}]`, `fmt`, `max`, `note` |
| `heat` | a small matrix | `matrix`, `rows`, `cols`, `signed` (false for sequential), `dp`, `note` |
| `sequence` | per-token values under token chips | `tokens`, `rows: [{label, values, dp}]`, `focus` |
| `matmul` | matrix product, one entry at a time | `A`, `B`, `dp`, `aName`, `bName`, `cName` |
| `graph` | computational graph, forward then backward with real numbers | `nodes: [{id, label, op, in, value, x, y}]`, ops: `input param const add sub mul div square exp log neg sigmoid tanh relu sum`, `dp`, `gx`, `gy`, `bw` |
| `network` | small MLP forward pass with numbers | `layers`, `W` (per layer, [out][in]), `b`, `acts`, `input`, `inputSliders`, `names`, `explain(x, vals, pre)` |
| `gd` | gradient descent, 1-D curve or 2-D surface | 1-D: `f`, `df`, `start`, `lr`, `x`, `y`; 2-D: `dim: 2`, `f(a,b)`, `grad(a,b)`, `start: [a,b]`, `x`, `y`, `min`; `lrMin`, `lrMax` |
| `diagram` | architecture, data flow, hardware block diagrams | grid units (default 40 px): `w`, `h`, `nodes: [{id,x,y,w,h,label,sub,kind,tip,round}]` (kinds: `hi blue ghost dark hatch s1..s5`), `edges: [{from,to,label,dashed,both,elbow,kind}]`, `groups: [{x,y,w,h,label,kind}]`, `notes: [{x,y,text,to}]` (handwritten), `dims: [{at:[x1,y1,x2,y2], label}]` |
| `calc` | calculators: sliders in, numbers out | `inputs: [{id,label,min,max,step,value,fmt,log}]` or `{id,label,options:[{label,value}]}`, `outputs: [{label, f(v), fmt}]`, `note(v)`, `after(el, v)` |
| `timeline` | schedules, overlap, pipelines | `rows`, `items: [{row,start,len,label,kind,tip}]` (kinds: `compute comm mem idle bubble s1..s5`), `total`, `unit`, `axis` (label, or false), `labelWidth` (px, default 90) |
| `vectors` | 2-D vectors, dot products | `a`, `b`, `range`, `showSum` |
| `specs` | datasheets | `items: [{v, label, note}]` |
| `compare` | two designs side by side | `left: {title, items}`, `right` |
| `html` | anything static | `html` |
| `tinylm` | widgets that run the trained model (Section 6) | `widget` |

Formats (`fmt`): `"int"`, `"f0"`…`"f6"`, `"sci"`, `"bytes"`, `"ibytes"`, `"time"` (seconds in), `"pct"`, `"pct1"`, `"x"`, or a function.

Labels in diagrams: a node's `label` is one or two short lines (`"Shared memory\nper SM"`), `sub` is one short line in a smaller face. Keep labels shorter than the box: about 7 px per character at the default size. Leave 0.5 units above a group's top edge free for its label.

### Custom figures
When no core type shows the mechanism, write your own component in your content file:

```js
Deck.fig("gpu-coalesce", (el, spec, api) => {
  const { h, s, clear, button, slider } = window.UI;
  // build DOM/SVG inside `el`; read spec fields; return { enter(), leave() } if you start timers
});
```

Rules for custom figures:
- Name it with your prefix (`gpu-…`, `math-…`). Put its CSS in `css/parts/NN.css`, with class names that start with the same prefix.
- Colours come only from CSS variables: `--ink --ink-2 --ink-3 --paper --plate --plate-2 --frame --rule --rule-2 --red --red-soft --blue --blue-soft --hi --good --bad --on-ink` and the data series `--s1 … --s5` (validated for colour-vision deficiency in both themes; use them in order). Never a literal colour: the page has a light (graph paper) and a dark (blueprint) theme.
- Text in SVG: `font-family: var(--font-display)` for labels, `var(--font-mono)` for numbers and code, `var(--font-hand)` for pencil notes. Text uses ink colours, never series colours.
- SVG uses a `viewBox` and scales to the plate (`width: 100%; height: auto`). If it cannot be read below ~520 px wide, wrap it in `h("div", { class: "scroll-x" }, svg)` and give the SVG a `min-width`.
- Controls: use `UI.slider`, `UI.segmented`, `UI.button` inside `h("div", { class: "f-ctl" }, …)`. Outputs: `h("div", { class: "f-tiles" }, …)` with tiles, or an explanation line `h("p", { class: "f-mm-expl" }, …)`.
- Every hoverable mark gets `data-tip="exact value"`. Give the SVG `role="img"` and an `aria-label`.
- Animations: respect `UI.reduceMotion()`; anything driven by a timer must stop in `leave()`. Prefer step buttons over autoplay.
- Deterministic: use `TinyLM.mulberry32(seed)` for randomness so the figure shows the same thing every time.
- `UI` helpers: `h(tag, attrs, ...kids)`, `s(svgTag, attrs, ...kids)`, `clear(el)`, `fmt` (`p f pct int sci bytes ibytes signed minus time`), `UI.format(v, fmt)`, `chip(token)`, `slider({label,min,max,step,value,format,onInput})`, `segmented({label,options,value,onChange})`, `button(label, onClick, {kind:"primary"})`, `chart(el, cfg)`, `heatStrip(values)`, `seq(w)` and `div(v, max)` (sequential and diverging fills), `rich(html)` (HTML with TeX), `tex(src, display)`, `ticks([a,b])`, `ACT` (activation functions), `nextId(prefix)`.

## 5. Parts and chapters

- Each content file starts with `Deck.part({ n, title, short })` and then a part opener sheet: `{ id: "<prefix>part", layout: "part", title, body, take }`. The body says in 2 to 3 sentences what the part covers and why it matters. `take` lists what the reader will be able to do afterwards (one sentence). The chapter list with sheet ranges is generated.
- Then `Deck.chapter({ title })` before each chapter's sheets. The chapter code (2.3 etc.) is automatic.
- End each chapter with a sheet that consolidates: a worked problem, a summary table, or a `check`.
- Place `check`s regularly: at least one per chapter, with distractors that are common mistakes and a `why` that explains the right answer with numbers.

## 6. The running example and the trained model

The running example is `SELECT name FROM users WHERE id = 7;` and the grammar `SELECT c1 FROM t WHERE c2 = v;` over three tables (`users: id, name, email, age`; `orders: id, user_id, total, status`; `products: id, title, price`), c2 ≠ c1, v ∈ 1..9. 270 distinct queries. A query picks the table uniformly, then c1 uniformly, then c2 uniformly from the rest, then v uniformly.

The trained model (`window.TinyLM`) runs in the page:
- 9 tokens: `["SELECT"," name"," FROM"," users"," WHERE"," id"," ="," 7",";"]`, ids `[43,93,47,63,52,58,53,107,10]`. Tokens carry their leading space.
- Config: vocab V = 113 (39 characters + 74 BPE merges), context T = 8, d = 16, 2 layers, 2 heads (head size 8), MLP 16 → 64 → 16 with GELU (tanh form), RMSNorm (eps 1e-5) before attention and MLP and at the end, learned position embeddings, separate unembedding `wu` (16 × 113), no biases. 9,968 parameters. Adam (β = 0.9, 0.95, no weight decay), lr 0.01 with 100 warmup steps and cosine decay to 0.001, 6,000 steps, batches of 128 freshly sampled queries, seed 7.
- Final model: exact expected loss 0.6942 nats per token; the floor (entropy of the grammar) is 0.6937. p(next token) along our query: name 0.0854 (ideal 1/12), FROM 1.000, users 1.000, WHERE 1.000, id 0.3460 (ideal 1/3), = 1.000, 7 0.1077 (ideal 1/9), ; 1.000.
- API: `TinyLM.forward(ids, step?)` returns a trace `{ tokEmb, posEmb, x0, layers: [{ h1, q, k, v, heads: [{scores, att}], concat, attnOut, mid, h2, pre, act, mlpOut, post }], final, logits, probs }`; `TinyLM.encode(text)` → tokens; `TinyLM.idOf(token)`; `TinyLM.vocab`; `TinyLM.weights(step)` → named matrices (`wte wpe l0.wq … wu`); `TinyLM.snapSteps` (training snapshots 0, 25, 50, 75, 100, 150, 200, 300, 400, 800, 1600, 3000, 6000); `TinyLM.exactLoss(step)` → `{mean, perPos}`; `TinyLM.nextDist(ids, {temperature, topK, topP})`; `TinyLM.sampleQuery(rand)`; `TinyLM.allQueries()` (with probabilities); `TinyLM.data.exactCurve` (`[step, loss, perPos]` every 10 steps); `TinyLM.data.batchLoss10`; `TinyLM.data.floor` (per-position floor array); `TinyLM.data.holdout` (a second model trained without 27 queries, `weights("holdout")`); `TinyLM.mulberry32(seed)`.
- **Strip.** `example: "tokens"` shows the query as token chips above the text. Add `ids` to show token ids, `focus:4` to highlight position 4 (0-based), `target:5` to mark the token being predicted, `prefix:3` to show only the first 3 tokens and a `?`. Use `example: "text"` for the raw string. Use the strip on every sheet that talks about the query.

Use the model and the query wherever they make an idea concrete: probabilities after `WHERE` as a softmax example, the 8 training pairs of one query, the size of the embedding table (113 × 16), the FLOPs of one forward pass, the KV cache of 8 positions. When the topic needs real scale, switch to Llama 3 / H100 numbers and say so.

### Widgets from the first version
`fig: { type: "tinylm", widget: "<name>" }` mounts a widget that runs the model. Each was verified in the first version (see `legacy/index.html` for its original text). Set the sheet's `example` to the strip value listed, because the widget updates that strip.

| widget | example | shows |
|---|---|---|
| `world` | text | the schema and sampled queries |
| `predict` | tokens prefix:3 | next-token distribution, step through the query |
| `generate` | tokens | sampling with temperature, validity rate |
| `learn` | tokens | the 8 training pairs of one query, p and loss for each, before/after training |
| `bpe` | tokens ids | BPE merges applied step by step to the query |
| `embed` | tokens ids | token and position embeddings, 2-D projection of token vectors |
| `need` | tokens ids focus:1,3 target:4 | why a position must look back (edit the query) |
| `qkv` | tokens ids focus:4 | attention scores q·k for a position |
| `mix` | tokens ids focus:4 | softmax weights and the weighted sum of values |
| `heads` | tokens ids focus:4 | attention patterns of each head |
| `block` | tokens ids focus:4 | the residual stream through the blocks |
| `logits` | tokens ids focus:4 target:5 | final vector to logits to probabilities |
| `loss` | tokens ids | loss per position vs the floor |
| `gradient` | tokens ids focus:4 target:5 | softmax + cross-entropy gradient p − y on four logits |
| `backprop` | tokens ids focus:4 target:5 | backward pass through the last layers |
| `optim` | tokens ids | SGD vs Adam on a surface, the learning-rate schedule |
| `train` | tokens ids | training snapshots: loss curve and predictions over steps |
| `heldout` | tokens ids | memorization vs generalization (held-out model) |
| `real` | tokens ids | our config vs real models |
| `flops` | tokens ids | 6ND training compute |
| `scaling` | tokens ids | compute-optimal parameters vs tokens |
| `data` | tokens ids | MinHash dedup on two query logs |
| `memory` | tokens ids | training memory of a 70B model |
| `gpu` | tokens ids | H100 roofline |
| `flash` | tokens ids focus:4 | FlashAttention tiling and memory traffic |
| `precision` | tokens ids | FP32/BF16/FP16/FP8 bit layouts and rounding |
| `allreduce` | tokens ids | data parallelism and all-reduce time |
| `zero` | tokens ids | ZeRO stages memory |
| `tp` | tokens ids focus:4 | tensor-parallel MLP |
| `pp` | tokens ids | pipeline schedule and bubble |
| `moe` | tokens ids | mixture-of-experts routing |
| `mesh` | tokens ids | 4-D parallel layout of Llama 3.1 405B |
| `failures` | tokens ids | failure statistics at scale |
| `prefill` | tokens ids prefix:3 | prefill vs decode timing |
| `kvcache` | tokens ids prefix:3 | KV cache reuse during decode |
| `kvsize` | tokens ids | KV cache size calculator |
| `membound` | tokens ids | decode is memory-bound; batching |
| `batching` | tokens ids | static vs continuous batching |
| `paged` | tokens ids | PagedAttention blocks |
| `speculative` | tokens ids prefix:3 | speculative decoding acceptance |
| `quant` | tokens ids | weight quantization error |
| `fleet` | tokens ids prefix:3 | serving at fleet scale |
| `sft` | tokens ids | SFT loss masking |
| `prefs` | tokens ids | preference training (reward model, DPO) |
| `rl` | tokens ids | RL with checkable rewards |
| `rlinfra` | tokens ids | the RL loop as a distributed system |

Reuse a widget when it fits your sheet exactly. Do not use `cover`, `passes`, `loops`, `recap`, `plan`, `sources`: they belong to the old deck's navigation. Each widget appears on at most one sheet in the whole deck, and the part that owns the topic owns the widget (listed in CURRICULUM.md).

## 7. Verification (the reader will publish this; errors are not acceptable)

1. **Arithmetic and model numbers.** Every number you compute (worked examples, model probabilities, FLOP counts, byte counts, bandwidth times) must be recomputed in `tools/verify/partNN.mjs`, which asserts that the sheet text contains it. See `tools/verify/part00.mjs` and `tools/verify/lib.mjs` (`claim`, `near`, `equal`, `ok`, `r`, `int`, `TinyLM`, `done`). Compute in the script; never hard-code an answer you did by hand without recomputing it. For numerical experiments that need Python (PyTorch, numpy), write `tools/verify/partNN_*.py` and run it with `.venv/bin/python` (create it with `python3 -m venv .venv && .venv/bin/pip install numpy torch`); have it print the numbers and put them in the `.mjs` checks.
2. **Facts about real systems** (hardware specs, model configs, paper results, dates) come from primary sources: vendor datasheets and documentation, the papers themselves, official engineering blogs. Look them up (web search and fetch are available); do not rely on memory for any specific number. Put the source in `src`. If two sources disagree or you cannot confirm a number, leave it out or state the range and the source. Mark the spec variant (H100 **SXM** vs PCIe, dense vs sparse FLOP/s).
3. **Claims about mechanisms** must be exact. If a simplification is needed, say what it leaves out.
4. **Interactive figures must be correct at every slider position**, not just the default. Check extreme values.

## 8. Commands

```bash
node tools/validate.mjs --part 6                 # structure, TeX, HTML, style warnings
node tools/verify/part06.mjs                     # your numeric checks
node tools/render.mjs --part 6 --shots --out tools/shots/part06            # render in Chrome, report errors and overflow, save PNGs
node tools/render.mjs --part 6 --mobile --shots --out tools/shots/part06   # phone width (390 px)
node tools/render.mjs --part 6 --dark --shots --out tools/shots/part06     # blueprint theme
node tools/make_gallery.mjs && node tools/render.mjs --page gallery.html --shots --out tools/shots/gallery
```

Look at your screenshots (they are PNG files you can open). Check: labels not overlapping, nothing cut off, figures readable at 390 px, both themes legible, no empty plates, the plate not much taller than needed.

The page is `index.html`; it loads every part's content file. To read it in a browser, `python3 tools/serve.py 8741` and open `http://localhost:8741/#<sheet-id>` (add `?static` to switch motion off).

## 9. Files you may touch

Only your own part: `content/NN-*.js`, `css/parts/NN.css`, `tools/verify/partNN*.{mjs,py}`, and `tools/shots/partNN/`. Do not edit anything in `js/`, `css/sheet.css`, `index.html`, `tools/*.mjs` or other parts' files: other writers are working in parallel. If a core component has a bug or lacks something you need, work around it with a custom figure and describe the problem in your final report.

Other writers' files may be half-written while you work. Errors from their files are reported as warnings for you and do not fail your checks.

## 10. Definition of done

- The sheet count in CURRICULUM.md for your part is met (±10%), every listed topic is covered in order, and each sheet makes one point.
- `node tools/validate.mjs --part N` has no errors and no style warnings you can fix.
- `node tools/verify/partNN.mjs` passes and covers every computed number in your part.
- `node tools/render.mjs --part N` passes on desktop, mobile and dark, and you have looked at the screenshots.
- Every sheet that states a fact about a real system has `src`.
- Your final report lists: sheet count per chapter, custom figures you wrote, facts you could not verify (and what you did), and any core bugs you hit.
