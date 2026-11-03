# One Query, End to End

An interactive drawing set that teaches how large language models are trained and served, from derivatives and loss functions to GPU kernels, distributed training, inference servers and post-training. It follows one SQL query, `SELECT name FROM users WHERE id = 7;`, through a real 9,968-parameter transformer that runs in the browser, and switches to Llama 3 and H100 numbers wherever scale matters.

400 sheets in eleven parts: orientation, the math toolkit, learning from data, neural networks, language modeling, the transformer, GPUs and CUDA, training at scale, inference and serving, post-training, and a wrap-up.

Every slide is a sheet in an engineering drawing set: sheet number and pencil notes in the margin, figures mounted as plates, a title block. The light theme is graph paper, the dark theme is blueprint.

## Open it

- Online: https://omerarapdev.github.io/llm-deck/
- `index.html` from any static host, or `python3 tools/serve.py 8741` and open http://localhost:8741.
- `dist/one-query.html` is the same set as one self-contained file (build it with `python3 tools/bundle.py`).

Keys: `←` `→` sheets, `G` register of every sheet, `T` terms, `Home` / `End`. Swipe on touch screens. Add `?static` to the address to switch motion off. Each sheet has its own address (`#learn-mse`).

## Layout

| Path | Role |
|---|---|
| `content/NN-*.js` | The sheets, one file per part, written as data (`Deck.part`, `Deck.chapter`, `Deck.slide`) |
| `js/core/engine.js` | Sheet rendering, navigation, ruler, register, terms, themes |
| `js/core/figs.js` | Figure components: plots, gradient descent, computational graphs, matmul, networks, diagrams, code listings, calculators, timelines |
| `js/core/model.js` | The trained model in JavaScript: tokenizer, forward pass, KV-cache decoding, sampling |
| `js/legacy-widgets/` | Interactive widgets that run the model (first version), mounted through the `tinylm` figure |
| `css/sheet.css` | The drawing-set design; `css/parts/NN.css` holds each part's custom figures |
| `AUTHORING.md`, `CURRICULUM.md` | The writing rules and the outline every part follows |
| `legacy/` | The first, 57-slide version |

## How the numbers are checked

| Command | Checks |
|---|---|
| `.venv/bin/python tools/build_model.py` | Builds the SQL corpus and BPE tokenizer, trains the model and a held-out copy with PyTorch (single-threaded, seed 7, bit-reproducible), writes `data/model-data.js` and fixtures |
| `node tools/verify_js.mjs` | The JavaScript model against PyTorch: logits to 1e-4, attention to 1e-5, loss to 1e-5 |
| `node tools/validate.mjs` | Structure, ids, figure types, TeX that must parse, balanced HTML, writing-style rules |
| `node tools/verify/all.mjs` | Every part's numeric claims, recomputed and matched against the sheet text |
| `node tools/render.mjs [--mobile] [--dark] [--shots]` | Every sheet rendered in headless Chrome: script errors, failed figures, overflow; optional screenshots |

Facts about real systems carry their sources on the sheet that states them; the wrap-up collects them all. Each part was drafted against `AUTHORING.md`, then reviewed independently for errors in math, facts, code and figures, and corrected.
