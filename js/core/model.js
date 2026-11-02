/* TinyLM: the deck's running-example model, executed in the browser.
 *
 * Mirrors tools/build_model.py exactly: the same BPE tokenizer and the same
 * forward pass (learned positions, pre-norm RMSNorm, 2 layers x 2 heads,
 * GELU MLP). Weights come from data/model-data.js, one snapshot per training
 * step we kept. Every probability, attention weight and loss on the slides
 * is computed here, so the numbers always agree with each other.
 */
(function () {
  "use strict";
  const D = window.MODEL_DATA;
  const C = D.config;

  // ------------------------------------------------------------ tokenizer
  const PRETOK = / ?[A-Za-z_]+| ?[0-9]+| ?[^\sA-Za-z_0-9]+/g;
  const rankOf = new Map(D.merges.map((m, i) => [m[0] + "\u0001" + m[1], i]));
  const stoi = new Map(D.vocab.map((s, i) => [s, i]));
  const baseChars = new Set(D.vocab.slice(0, D.baseCount));

  function pretokenize(text) {
    return text.match(PRETOK) || [];
  }

  // Merge pairs in rank order; `upto` limits the merges to the first k learned.
  function encodeWord(word, upto = Infinity) {
    let s = Array.from(word);
    while (s.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < s.length - 1; i++) {
        const r = rankOf.get(s[i] + "\u0001" + s[i + 1]);
        if (r !== undefined && r < upto && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      const a = s[best], b = s[best + 1], out = [];
      for (let i = 0; i < s.length; i++) {
        if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { out.push(a + b); i++; }
        else out.push(s[i]);
      }
      s = out;
    }
    return s;
  }

  function encode(text, upto = Infinity) {
    return pretokenize(text).flatMap((w) => encodeWord(w, upto));
  }

  const idOf = (tok) => (stoi.has(tok) ? stoi.get(tok) : -1);

  // ------------------------------------------------------------ weights
  function halfToFloat(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function decodeSnapshot(snap) {
    const bin = atob(snap.b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const view = new DataView(bytes.buffer);
    const n = snap.dtype === "f32" ? bytes.length / 4 : bytes.length / 2;
    const flat = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      flat[i] = snap.dtype === "f32" ? view.getFloat32(i * 4, true) : halfToFloat(view.getUint16(i * 2, true));
    }
    const W = {};
    let off = 0;
    for (const [name, shape] of C.shapes) {
      const size = shape.reduce((a, b) => a * b, 1);
      W[name] = { shape, data: flat.subarray(off, off + size) };
      off += size;
    }
    return W;
  }

  const cache = new Map();
  // `step` is a saved training step, or "holdout" for the model trained without the held-out queries
  function weights(step = finalStep) {
    if (!cache.has(step)) cache.set(step, decodeSnapshot(step === "holdout" ? D.holdout : D.snaps[String(step)]));
    return cache.get(step);
  }
  const finalStep = D.snapSteps[D.snapSteps.length - 1];

  // ------------------------------------------------------------ math
  const row = (M, i) => Array.from(M.data.subarray(i * M.shape[1], (i + 1) * M.shape[1]));

  function vecMat(x, M) {           // x: [n], M: [n, m] -> [m]
    const [n, m] = M.shape, out = new Array(m).fill(0);
    for (let i = 0; i < n; i++) {
      const xi = x[i], base = i * m;
      if (xi === 0) continue;
      for (let j = 0; j < m; j++) out[j] += xi * M.data[base + j];
    }
    return out;
  }

  function rmsnorm(x, g, eps = C.eps) {
    let ms = 0;
    for (const v of x) ms += v * v;
    const r = 1 / Math.sqrt(ms / x.length + eps);
    return x.map((v, i) => v * r * g.data[i]);
  }

  const GELU_K = Math.sqrt(2 / Math.PI);
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(GELU_K * (x + 0.044715 * x * x * x)));

  function softmax(z, temperature = 1) {
    let mx = -Infinity;
    for (const v of z) if (v / temperature > mx) mx = v / temperature;
    const e = z.map((v) => Math.exp(v / temperature - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  const add = (a, b) => a.map((v, i) => v + b[i]);

  /* Full forward pass with every intermediate kept, for the slides.
   * ids: token ids (length <= T). Returns a trace object. */
  function forward(ids, step = finalStep, opts = {}) {
    const W = opts.weights || weights(step);
    const T = ids.length, d = C.d, H = C.H, hd = d / H;
    const tokEmb = ids.map((id) => row(W.wte, id));
    const posEmb = ids.map((_, t) => row(W.wpe, t));
    let x = tokEmb.map((e, t) => add(e, posEmb[t]));
    const tr = { ids, tokEmb, posEmb, x0: x.map((v) => v.slice()), layers: [] };
    for (let l = 0; l < C.L; l++) {
      const P = (k) => W[`l${l}.${k}`];
      const h1 = x.map((v) => rmsnorm(v, P("n1")));
      const q = h1.map((v) => vecMat(v, P("wq")));
      const k = h1.map((v) => vecMat(v, P("wk")));
      const v = h1.map((u) => vecMat(u, P("wv")));
      const heads = [];
      const concat = Array.from({ length: T }, () => new Array(d).fill(0));
      for (let h = 0; h < H; h++) {
        const sl = (vec) => vec.slice(h * hd, (h + 1) * hd);
        const scores = [], att = [];
        for (let i = 0; i < T; i++) {
          const qi = sl(q[i]);
          const srow = [];
          for (let j = 0; j < T; j++) {
            if (j > i) { srow.push(-Infinity); continue; }
            const kj = sl(k[j]);
            let dot = 0;
            for (let c = 0; c < hd; c++) dot += qi[c] * kj[c];
            srow.push(dot / Math.sqrt(hd));
          }
          const a = softmax(srow.slice(0, i + 1)).concat(new Array(T - i - 1).fill(0));
          scores.push(srow); att.push(a);
          for (let j = 0; j <= i; j++) {
            const vj = sl(v[j]);
            for (let c = 0; c < hd; c++) concat[i][h * hd + c] += a[j] * vj[c];
          }
        }
        heads.push({ scores, att });
      }
      const attnOut = concat.map((u) => vecMat(u, P("wo")));
      const mid = x.map((u, t) => add(u, attnOut[t]));
      const h2 = mid.map((u) => rmsnorm(u, P("n2")));
      const pre = h2.map((u) => vecMat(u, P("w1")));
      const act = pre.map((u) => u.map(gelu));
      const mlpOut = act.map((u) => vecMat(u, P("w2")));
      const post = mid.map((u, t) => add(u, mlpOut[t]));
      tr.layers.push({ h1, q, k, v, heads, concat, attnOut, mid, h2, pre, act, mlpOut, post });
      x = post;
    }
    tr.final = x.map((u) => rmsnorm(u, W.nf));
    tr.logits = tr.final.map((u) => vecMat(u, W.wu));
    tr.probs = tr.logits.map((z) => softmax(z));
    return tr;
  }

  /* Decode one token at a time with a KV cache, the way a server does.
   * Each step computes q, k, v only for the newest token and attends over the
   * cached keys and values. Returns logits per position plus the cache. */
  function forwardIncremental(ids, step = finalStep) {
    const W = weights(step);
    const d = C.d, H = C.H, hd = d / H;
    const cache = Array.from({ length: C.L }, () => ({ k: [], v: [] }));
    const logits = [];
    let kvComputed = 0;
    ids.forEach((id, t) => {
      let x = add(row(W.wte, id), row(W.wpe, t));
      for (let l = 0; l < C.L; l++) {
        const P = (k) => W[`l${l}.${k}`];
        const h1 = rmsnorm(x, P("n1"));
        const q = vecMat(h1, P("wq"));
        cache[l].k.push(vecMat(h1, P("wk")));
        cache[l].v.push(vecMat(h1, P("wv")));
        kvComputed++;
        const o = new Array(d).fill(0);
        for (let hh = 0; hh < H; hh++) {
          const sc = cache[l].k.map((kj) => {
            let dot = 0;
            for (let c = 0; c < hd; c++) dot += q[hh * hd + c] * kj[hh * hd + c];
            return dot / Math.sqrt(hd);
          });
          const a = softmax(sc);
          cache[l].v.forEach((vj, j) => { for (let c = 0; c < hd; c++) o[hh * hd + c] += a[j] * vj[hh * hd + c]; });
        }
        x = add(x, vecMat(o, P("wo")));
        const h2 = rmsnorm(x, P("n2"));
        x = add(x, vecMat(vecMat(h2, P("w1")).map(gelu), P("w2")));
      }
      logits.push(vecMat(rmsnorm(x, W.nf), W.wu));
    });
    return { logits, cache, kvComputed };
  }

  // ------------------------------------------------------------ sampling
  function filterProbs(p, { topK = 0, topP = 1 } = {}) {
    const order = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    let keep = order;
    if (topK > 0) keep = keep.slice(0, topK);
    if (topP < 1) {
      let cum = 0;
      const out = [];
      for (const e of keep) { out.push(e); cum += e[0]; if (cum >= topP) break; }
      keep = out;
    }
    const z = keep.reduce((s, e) => s + e[0], 0);
    const q = new Array(p.length).fill(0);
    for (const [v, i] of keep) q[i] = v / z;
    return q;
  }

  function sampleFrom(p, rand = Math.random) {
    let r = rand(), cum = 0;
    for (let i = 0; i < p.length; i++) { cum += p[i]; if (r < cum) return i; }
    return p.length - 1;
  }

  // Next-token distribution at the last position, with decoding controls.
  function nextDist(ids, { step = finalStep, temperature = 1, topK = 0, topP = 1 } = {}) {
    const tr = forward(ids, step);
    const logits = tr.logits[ids.length - 1];
    return filterProbs(softmax(logits, temperature), { topK, topP });
  }

  // ------------------------------------------------------------ the grammar
  function allQueries() {
    const out = [];
    const tables = Object.keys(D.schema);
    for (const t of tables) {
      const cols = D.schema[t];
      for (const c1 of cols) {
        const rest = cols.filter((c) => c !== c1);
        for (const c2 of rest) {
          for (const v of D.values) {
            const p = 1 / tables.length / cols.length / rest.length / D.values.length;
            out.push({ text: `SELECT ${c1} FROM ${t} WHERE ${c2} = ${v};`, p, t, c1, c2, v });
          }
        }
      }
    }
    return out;
  }

  function sampleQuery(rand = Math.random) {
    const tables = Object.keys(D.schema);
    const pick = (a) => a[Math.floor(rand() * a.length)];
    const t = pick(tables), c1 = pick(D.schema[t]);
    const c2 = pick(D.schema[t].filter((c) => c !== c1)), v = pick(D.values);
    return `SELECT ${c1} FROM ${t} WHERE ${c2} = ${v};`;
  }

  // Exact expected loss over every query the grammar can produce.
  const lossCache = new Map();
  function exactLoss(step = finalStep, opts = {}) {
    if (!opts.weights && lossCache.has(step)) return lossCache.get(step);
    const per = new Array(C.T).fill(0);
    for (const q of allQueries()) {
      const ids = encode(q.text).map(idOf);
      const tr = forward(ids.slice(0, C.T), step, opts);
      for (let i = 0; i < C.T; i++) per[i] += -Math.log(tr.probs[i][ids[i + 1]]) * q.p;
    }
    const res = { mean: per.reduce((a, b) => a + b, 0) / C.T, perPos: per };
    if (!opts.weights) lossCache.set(step, res);
    return res;
  }

  // Small seeded PRNG so "random" demos replay identically.
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const running = encode(D.running);

  window.TinyLM = {
    data: D, config: C, vocab: D.vocab, finalStep, snapSteps: D.snapSteps,
    pretokenize, encodeWord, encode, idOf, baseChars, weights, decodeSnapshot,
    forward, forwardIncremental, softmax, filterProbs, sampleFrom, nextDist, gelu, rmsnorm, vecMat, row,
    allQueries, sampleQuery, exactLoss, mulberry32,
    running, runningIds: running.map(idOf),
  };
})();
