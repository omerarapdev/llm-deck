# One Query, End to End

An interactive deck that teaches how large language models are trained and served by following one SQL query, `SELECT name FROM users WHERE id = 7;`, through a real 9,968-parameter transformer that runs in the browser.

57 slides in five passes: the whole loop, inside the model, training at scale, serving, and becoming an assistant. A second copy of the model, trained without 27 of the 270 queries, powers the slide on generalization versus memorization.

## Open it

- `index.html` works from any static host (GitHub Pages, `python3 tools/serve.py`).
- `dist/one-query.html` is the same deck as one self-contained file. It opens straight from disk.

Keys: `←` `→` move, `M` slide list, `N` notes, swipe on touch screens. Add `?static` to the URL to turn off motion.

## How the numbers are made

| File | Role |
|---|---|
| `tools/build_model.py` | Builds the toy SQL corpus and the BPE tokenizer, trains the main model and a held-out copy with PyTorch (single-threaded, seed 7, bit-reproducible), and writes `data/model-data.js` plus test fixtures. |
| `js/model.js` | The same tokenizer and forward pass in JavaScript. Every number on the slides is computed by it. |
| `tools/verify_js.mjs` | Checks the JavaScript model against PyTorch: logits to 1e-4, attention to 1e-5, exact loss to 1e-5, all 75 tokenizer stages. |
| `tools/check_text.mjs` | Recomputes every model number quoted in the slide text (122 checks) and fails if any drifted. |
| `tools/bundle.py` | Inlines CSS, JS and data into `dist/`. |

Rebuild everything:

```bash
python3 -m venv .venv && .venv/bin/pip install numpy torch
.venv/bin/python tools/build_model.py
node tools/verify_js.mjs && node tools/check_text.mjs
python3 tools/bundle.py
```

Facts about real systems (Llama 3, H100, ZeRO, vLLM and others) are cited on the Sources slide.
