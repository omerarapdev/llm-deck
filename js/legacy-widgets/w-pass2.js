/* Widgets for Pass 2: inside the model. */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip, slider, segmented, button, player, barList, topRows, heatStrip, chart, seq, div } = window.UI;
  const M = window.TinyLM;
  const W = (window.Widgets = window.Widgets || {});
  const V = M.vocab, C = M.config;
  const ids = M.runningIds, toks = M.running;
  const FINAL = M.finalStep;
  const IN = ids.slice(0, 8);

  const traces = new Map();
  const trace = (step = FINAL) => {
    if (!traces.has(step)) traces.set(step, M.forward(IN, step));
    return traces.get(step);
  };
  W._trace = trace;

  // A row of the 8 input tokens that acts as a position picker.
  function posPicker(container, { value, onChange, label = "Position", maxPos = 7 }) {
    const row = h("div", { class: "pos-picker", role: "group", "aria-label": label });
    const btns = toks.slice(0, maxPos + 1).map((t, i) => {
      const b = h("button", { type: "button", class: "pos-btn", "aria-pressed": String(i === value), "aria-label": `${label}: ${window.UI.tokLabel(t)} (position ${i})`, onclick: () => set(i, true) }, chip(t), h("span", { class: "pos-i" }, String(i)));
      return b;
    });
    row.append(...btns);
    container.appendChild(h("div", { class: "ctl" }, h("span", { class: "ctl-label" }, label), row));
    function set(i, fire) {
      value = i;
      btns.forEach((b, k) => b.setAttribute("aria-pressed", String(k === i)));
      if (fire) onChange(i);
    }
    return { set, get: () => value };
  }
  const HEADS = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const headName = (l, hh) => `Layer ${l + 1} · head ${hh + 1}`;
  function headPicker(container, { value, onChange }) {
    return container.appendChild(segmented({
      label: "Head", value,
      options: HEADS.map(([l, hh], i) => ({ value: i, label: `L${l + 1} H${hh + 1}` })),
      onChange,
    }));
  }
  const slice8 = (vec, hh) => vec.slice(hh * 8, hh * 8 + 8);
  Object.assign(W, { _posPicker: posPicker, _headPicker: headPicker, _HEADS: HEADS, _headName: headName, _slice8: slice8 });

  // ------------------------------------------------------------ bpe
  W.bpe = {
    init(el, deck) {
      const log = M.data.bpeLog, K = log.length;
      const seqEl = el.querySelector(".bpe-seq"), kEl = el.querySelector(".bpe-k"), nEl = el.querySelector(".bpe-n");
      const mergeEl = el.querySelector(".bpe-merge");
      let k = 0;
      const playBtn = button("Play merges", () => { if (k >= K) set(0); play.toggle(); }, { kind: "primary" });
      const play = player(() => { if (k >= K) return false; set(k + 1); return k < K; }, { interval: 420, onState: (on) => { playBtn.textContent = on ? "Pause" : "Play merges"; } });
      const sl = slider({ id: "bpe-k", label: "Merges applied", min: 0, max: K, value: 0, format: (v) => `${v} of ${K}`, onInput: (v) => { play.pause(); set(v); } });
      el.querySelector(".bpe-ctl").append(sl, playBtn, button("Show all 74", () => { play.pause(); set(K); }));
      const pairChip = (a, b) => h("span", { class: "pair" }, chip(a), h("span", { class: "plus" }, "+"), chip(b));
      function set(v) {
        k = v;
        sl.set(v);
        const seg = M.encode(M.data.running, v);
        const newTok = v > 0 ? log[v - 1].pair.join("") : null;
        clear(seqEl);
        seg.forEach((t) => seqEl.appendChild(chip(t, { id: M.idOf(t), cls: t === newTok ? "is-focus" : "" })));
        kEl.textContent = v;
        nEl.textContent = seg.length;
        clear(mergeEl);
        if (v === 0) {
          mergeEl.append(h("span", null, "Every character starts as its own token. Press ", h("b", null, "Play merges"), " or drag the slider."));
        } else {
          const e = log[v - 1];
          const nextOnes = v < K ? log[v].top.slice(0, 3) : [];
          mergeEl.append(
            h("div", { class: "merge-main" }, h("span", { class: "merge-k" }, `Merge ${v}`), pairChip(e.pair[0], e.pair[1]), h("span", { class: "arrow" }, "→"), chip(newTok, { id: M.idOf(newTok), cls: "is-focus" }),
              h("span", { class: "merge-n" }, `${e.count.toLocaleString("en-US")} times`)),
            nextOnes.length ? h("div", { class: "merge-next" }, "Most frequent pairs now: ",
              nextOnes.map(([a, b, c]) => h("span", { class: "nx" }, pairChip(a, b), h("span", { class: "num" }, c.toLocaleString("en-US"))))) : h("div", { class: "merge-next" }, "No pairs left: every word in the corpus is now a single token."));
        }
        deck.strip.set({ tokens: seg, ids: true });
      }
      const input = el.querySelector("#bpe-input"), out = el.querySelector(".bpe-out");
      function tokenizeInput() {
        clear(out);
        const tt = M.encode(input.value);
        const row = h("div", { class: "bpe-try-row" });
        tt.forEach((t) => {
          const id = M.idOf(t);
          row.appendChild(chip(t, { id: id >= 0 ? id : "?", cls: id < 0 ? "is-unknown" : "", title: id < 0 ? "not in the vocabulary" : undefined }));
        });
        const unknown = tt.filter((t) => M.idOf(t) < 0).length;
        out.append(row, h("p", { class: "fig-cap" }, `${tt.length} tokens for ${input.value.length} characters`,
          unknown ? `. ${unknown} piece${unknown > 1 ? "s" : ""} (dashed) contain characters this tokenizer never saw, so the model could not read them.` : "."));
      }
      input.addEventListener("input", tokenizeInput);
      tokenizeInput();
      this.set = set;
      this.cur = () => k;
      this.play = play;
      set(0);
    },
    enter() { if (this.set) this.set(this.cur()); },
    leave() { if (this.play) this.play.pause(); },
  };

  // ------------------------------------------------------------ embed
  function pca2(vectors) {
    const n = vectors.length, d = vectors[0].length;
    const mean = new Array(d).fill(0);
    vectors.forEach((r) => r.forEach((v, i) => { mean[i] += v / n; }));
    const X = vectors.map((r) => r.map((v, i) => v - mean[i]));
    const Cv = Array.from({ length: d }, () => new Array(d).fill(0));
    X.forEach((r) => { for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Cv[i][j] += (r[i] * r[j]) / n; });
    const vecs = [];
    for (let c = 0; c < 2; c++) {
      let v = Array.from({ length: d }, (_, i) => Math.cos(i * 1.7 + c));
      for (let it = 0; it < 300; it++) {
        let w = Cv.map((row) => row.reduce((acc, x, j) => acc + x * v[j], 0));
        for (const u of vecs) { const dp = w.reduce((acc, x, i) => acc + x * u[i], 0); w = w.map((x, i) => x - dp * u[i]); }
        const nr = Math.hypot(...w) || 1;
        v = w.map((x) => x / nr);
      }
      // fix the sign so the picture doesn't flip between runs
      const big = v.reduce((bi, x, i) => (Math.abs(x) > Math.abs(v[bi]) ? i : bi), 0);
      if (v[big] < 0) v = v.map((x) => -x);
      vecs.push(v);
    }
    return X.map((r) => vecs.map((u) => r.reduce((acc, x, i) => acc + x * u[i], 0)));
  }

  W.embed = {
    init(el, deck) {
      const used = [...new Set(M.allQueries().flatMap((q) => M.encode(q.text)))];
      const sumEl = el.querySelector(".emb-sum");
      let pos = 3;
      const picker = posPicker(el.querySelector(".emb-tokens"), { value: pos, label: "Token", onChange: (i) => { pos = i; showSum(); } });
      function showSum() {
        const tr = trace();
        const te = tr.tokEmb[pos], pe = tr.posEmb[pos], x0 = tr.x0[pos];
        const m = Math.max(...te.map(Math.abs), ...pe.map(Math.abs), ...x0.map(Math.abs));
        clear(sumEl);
        const rowOf = (label, vec, note) => h("div", { class: "vec-row" },
          h("div", { class: "vec-label" }, label, note ? h("span", { class: "vec-note" }, note) : null), heatStrip(vec, { maxAbs: m }));
        sumEl.append(
          rowOf(h("span", null, "token row ", h("b", { class: "num" }, String(IN[pos]))), te, window.UI.tokLabel(toks[pos])),
          h("div", { class: "vec-op" }, "+"),
          rowOf(h("span", null, "position row ", h("b", { class: "num" }, String(pos))), pe),
          h("div", { class: "vec-op" }, "="),
          rowOf(h("span", null, "input to layer 1"), x0, "16 numbers"));
        deck.strip.set({ focus: [pos] });
      }
      // embedding map
      const mapEl = el.querySelector(".emb-map");
      const roleCls = { kw: "ink", op: "ink", table: "s1", col: "s2", val: "s3", piece: "ink" };
      function drawMap(step) {
        clear(mapEl);
        const Wt = M.weights(step);
        const pts = pca2(used.map((t) => M.row(Wt.wte, M.idOf(t))));
        const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
        const pad = (a, b) => [a - (b - a) * 0.12, b + (b - a) * 0.12];
        const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
        const digitIdx = used.map((t, i) => [t, i]).filter(([t]) => /^ [0-9]$/.test(t)).map(([, i]) => i);
        const placed = [];
        const dots = used.map((t, i) => {
          const role = window.UI.roleOf(t);
          return { x: pts[i][0], y: pts[i][1], r: 5, cls: roleCls[role], tip: `${window.UI.tokLabel(t)} (${role === "kw" || role === "op" ? "keyword" : role === "col" ? "column" : role === "val" ? "digit" : "table"})`, t, i };
        });
        chart(mapEl, {
          width: 520, height: 260, margin: { l: 14, r: 14, t: 10, b: 14 },
          aria: "Token embeddings projected onto their two main directions",
          x: { domain: [x0, x1], ticks: [] }, y: { domain: [y0, y1], ticks: [] },
          dots,
          after(svg, X, Y) {
            // digits get one group label; other tokens get labels where they don't collide
            const dx = digitIdx.map((i) => X(pts[i][0])), dy = digitIdx.map((i) => Y(pts[i][1]));
            const cx = dx.reduce((a, b) => a + b, 0) / dx.length, cy = Math.min(...dy) - 10;
            const spread = Math.max(...dx) - Math.min(...dx) + Math.max(...dy) - Math.min(...dy);
            if (spread < 80) { svg.appendChild(s("text", { x: cx, y: cy, class: "d-label", "text-anchor": "middle" }, "digits 1–9")); placed.push([cx - 30, cy - 10, 60, 14]); }
            dots.forEach((d) => {
              if (digitIdx.includes(d.i) && spread < 80) return;
              const lx = X(d.x) + 7, ly = Y(d.y) + 4, w = d.t.trim().length * 6.6 + 4;
              if (placed.some(([px, py, pw, ph]) => lx < px + pw && lx + w > px && ly - 10 < py + ph && ly > py)) return;
              placed.push([lx, ly - 10, w, 13]);
              svg.appendChild(s("text", { x: lx, y: ly, class: "e-label" }, d.t.trim()));
            });
          },
        });
        mapEl.appendChild(h("div", { class: "legend" },
          h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--ink-2)" } }), "keyword or symbol"),
          h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--s1)" } }), "table"),
          h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--s2)" } }), "column"),
          h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--s3)" } }), "digit")));
      }
      const seg = segmented({ label: "Weights", value: FINAL, options: [{ value: 0, label: "Untrained" }, { value: FINAL, label: "Trained" }], onChange: drawMap });
      el.querySelector(".emb-toggle").appendChild(seg);
      drawMap(FINAL);
      showSum();
      this.show = () => { picker.set(pos); showSum(); };
    },
    enter() { if (this.show) this.show(); },
  };

  // ------------------------------------------------------------ need
  W.need = {
    init(el, deck) {
      const cols = [...new Set(Object.values(M.data.schema).flat())];
      const tables = Object.keys(M.data.schema);
      let c1 = "name", t = "users";
      const sel = (id, opts, val, on) => {
        const e = h("select", { class: "sel", id, "aria-label": id === "need-c1" ? "selected column" : "table" }, opts.map((o) => h("option", { value: o, selected: o === val }, o)));
        e.addEventListener("change", () => on(e.value));
        return e;
      };
      const editor = el.querySelector(".need-editor");
      editor.append(h("code", { class: "need-q" }, h("b", null, "SELECT "), sel("need-c1", cols, c1, (v) => { c1 = v; run(); }),
        h("b", null, " FROM "), sel("need-t", tables, t, (v) => { t = v; run(); }), h("b", null, " WHERE "), h("span", { class: "hl" }, "?")));
      const fromBars = barList(el.querySelector(".need-from"), { labelWidth: "6.6rem" });
      const whereBars = barList(el.querySelector(".need-where"), { labelWidth: "6.6rem" });
      const note = el.querySelector(".need-note");
      function run() {
        const q = [" " + c1, " FROM", " " + t, " WHERE"];
        const tk = ["SELECT", ...q];
        const idsQ = tk.map(M.idOf);
        const tr = M.forward(idsQ);
        const okTables = tables.filter((x) => M.data.schema[x].includes(c1));
        fromBars.update(topRows(tr.probs[2], 3, {}).map((r) => Object.assign(r, { cls: okTables.includes(V[+r.key].trim()) ? "is-target" : "" })));
        const allowed = (M.data.schema[t] || []).filter((c) => c !== c1);
        whereBars.update(topRows(tr.probs[4], 4, {}).map((r) => Object.assign(r, { cls: allowed.includes(V[+r.key].trim()) && M.data.schema[t].includes(c1) ? "is-target" : "" })));
        const valid = M.data.schema[t].includes(c1);
        note.innerHTML = valid
          ? `<code>${c1}</code> exists in ${okTables.map((x) => `<code>${x}</code>`).join(" and ")}, so that is where the FROM prediction goes. After <code>WHERE</code> the allowed columns are ${allowed.map((x) => `<code>${x}</code>`).join(", ")}: the columns of <code>${t}</code> except <code>${c1}</code>. Red bars are the tokens the rules allow.`
          : `<code>${t}</code> has no <code>${c1}</code> column, so this prefix never appears in the training data. The model still answers, but nothing in training told it what is right here.`;
        deck.strip.set({ tokens: [...tk, " id", " =", " 7", ";"], prefix: 5, focus: [1, 3], target: null });
      }
      this.run = run;
      run();
    },
    enter() { if (this.run) this.run(); },
  };

  // ------------------------------------------------------------ qkv
  W.qkv = {
    init(el, deck) {
      let head = 1, pos = 4;
      const ctl = el.querySelector(".qkv-ctl");
      headPicker(ctl, { value: head, onChange: (v) => { head = v; draw(); } });
      const picker = posPicker(ctl, { value: pos, label: "Asking from", onChange: (i) => { pos = i; draw(); } });
      const qEl = el.querySelector(".qkv-q"), rowsEl = el.querySelector(".qkv-rows");
      function draw() {
        const [l, hh] = HEADS[head];
        const L = trace().layers[l];
        const q = slice8(L.q[pos], hh);
        const scores = L.heads[hh].scores[pos];
        const keys = L.k.map((v) => slice8(v, hh));
        const m = Math.max(...q.map(Math.abs), ...keys.flat().map(Math.abs));
        clear(qEl);
        qEl.append(h("div", { class: "vec-row" }, h("div", { class: "vec-label" }, "query from ", chip(toks[pos]), h("span", { class: "vec-note" }, headName(l, hh))), heatStrip(q, { maxAbs: m, cell: "n8" })));
        clear(rowsEl);
        rowsEl.appendChild(h("div", { class: "kv-row kv-head", role: "row" }, h("span", null, "Position"), h("span", null, "Key k (8 numbers)"), h("span", { class: "r" }, "Score")));
        const valid = scores.slice(0, pos + 1);
        const best = valid.indexOf(Math.max(...valid));
        const smax = Math.max(...valid.map(Math.abs), 1e-9);
        toks.slice(0, 8).forEach((t, j) => {
          const masked = j > pos;
          const sc = scores[j];
          rowsEl.appendChild(h("div", { class: "kv-row" + (masked ? " is-masked" : "") + (j === best ? " is-best" : ""), role: "row" },
            h("span", { class: "kv-tok" }, h("span", { class: "pos-i" }, String(j)), chip(t)),
            masked ? h("span", { class: "masked" }, "future: not visible") : heatStrip(keys[j], { maxAbs: m, cell: "n8" }),
            h("span", { class: "kv-score" }, masked ? "—" : h("span", { class: "score-bar" }, h("span", { class: "sb-fill" + (sc < 0 ? " neg" : ""), style: { width: `${(Math.abs(sc) / smax) * 100}%` } })), masked ? null : h("span", { class: "num" }, fmt.minus(sc.toFixed(2))))));
        });
        deck.strip.set({ focus: [pos] });
      }
      this.draw = draw;
      draw();
    },
    enter() { if (this.draw) this.draw(); },
  };

  // ------------------------------------------------------------ mix
  W.mix = {
    init(el, deck) {
      let head = 1, pos = 4;
      const ctl = el.querySelector(".mix-ctl");
      headPicker(ctl, { value: head, onChange: (v) => { head = v; draw(); } });
      posPicker(ctl, { value: pos, label: "Asking from", onChange: (i) => { pos = i; draw(); } });
      const rowsEl = el.querySelector(".mix-rows"), outEl = el.querySelector(".mix-out");
      function draw() {
        const [l, hh] = HEADS[head];
        const L = trace().layers[l];
        const att = L.heads[hh].att[pos], scores = L.heads[hh].scores[pos];
        const vals = L.v.map((v) => slice8(v, hh));
        const out = slice8(L.concat[pos], hh);
        const m = Math.max(...vals.flat().map(Math.abs), ...out.map(Math.abs));
        clear(rowsEl);
        rowsEl.appendChild(h("div", { class: "mx-row mx-head", role: "row" }, h("span", null, "Position"), h("span", { class: "r" }, "Score"), h("span", null, "Weight"), h("span", null, "Value v (8 numbers)")));
        toks.slice(0, 8).forEach((t, j) => {
          const masked = j > pos;
          rowsEl.appendChild(h("div", { class: "mx-row" + (masked ? " is-masked" : ""), role: "row" },
            h("span", { class: "kv-tok" }, h("span", { class: "pos-i" }, String(j)), chip(t)),
            h("span", { class: "num r" }, masked ? "masked" : fmt.minus(scores[j].toFixed(2))),
            h("span", { class: "w-cell" }, h("span", { class: "w-track" }, h("span", { class: "w-fill", style: { width: `${masked ? 0 : att[j] * 100}%` } })), h("span", { class: "num" }, masked ? "0" : att[j].toFixed(2))),
            h("span", { class: "v-cell", style: { opacity: masked ? 0.25 : String(0.3 + 0.7 * att[j]) } }, heatStrip(vals[j], { maxAbs: m, cell: "n8" }))));
        });
        clear(outEl);
        outEl.append(h("div", { class: "vec-row" }, h("div", { class: "vec-label" }, "head output = Σ weight × value", h("span", { class: "vec-note" }, headName(l, hh) + " at " + toks[pos].trim())), heatStrip(out, { maxAbs: m, cell: "n8" })));
        deck.strip.set({ focus: [pos] });
      }
      this.draw = draw;
      draw();
    },
    enter() { if (this.draw) this.draw(); },
  };

  // ------------------------------------------------------------ heads
  W.heads = {
    init(el, deck) {
      const PRESETS = [M.data.running, "SELECT total FROM orders WHERE status = 3;", "SELECT id FROM products WHERE price = 2;", "SELECT email FROM users WHERE age = 5;"];
      let q = PRESETS[0], sel = { m: 1, i: 4, j: 3 };
      const ctl = el.querySelector(".heads-ctl");
      const select = h("select", { class: "sel", id: "heads-q", "aria-label": "Query" }, PRESETS.map((p, i) => h("option", { value: String(i) }, p)));
      select.addEventListener("change", () => { q = PRESETS[+select.value]; draw(); });
      ctl.append(h("div", { class: "ctl" }, h("label", { class: "ctl-label", for: "heads-q" }, "Query"), select));
      const grid = el.querySelector(".heads-grid"), read = el.querySelector(".heads-read");
      function draw() {
        const tk = M.encode(q), qi = tk.map(M.idOf);
        const tr = M.forward(qi.slice(0, 8));
        clear(grid);
        HEADS.forEach(([l, hh], m) => {
          const att = tr.layers[l].heads[hh].att;
          const cell = 22, lab = 62, top = 8, size = 8 * cell;
          const svg = s("svg", { viewBox: `0 0 ${lab + size + 6} ${top + size + 58}`, class: "attn-map", role: "img", "aria-label": `${headName(l, hh)} attention weights` });
          for (let i = 0; i < 8; i++) {
            svg.appendChild(s("text", { x: lab - 6, y: top + i * cell + cell * 0.68, class: "am-lab", "text-anchor": "end" }, tk[i].trim()));
            svg.appendChild(s("text", { x: lab + i * cell + cell / 2, y: top + size + 8, class: "am-lab", "text-anchor": "end", transform: `rotate(-60 ${lab + i * cell + cell / 2} ${top + size + 8})` }, tk[i].trim()));
            for (let j = 0; j < 8; j++) {
              const x = lab + j * cell, y = top + i * cell;
              if (j > i) { svg.appendChild(s("rect", { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, class: "am-masked" })); continue; }
              const w = att[i][j];
              const r = s("rect", { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, rx: 3, style: `fill:${seq(w)}`, class: "am-cell" + (sel.m === m && sel.i === i && sel.j === j ? " is-sel" : ""), tabindex: "-1",
                "data-tip": `${headName(l, hh)}\nfrom ${tk[i].trim()} (pos ${i}) to ${tk[j].trim()} (pos ${j}): ${w.toFixed(2)}` });
              r.addEventListener("click", () => { sel = { m, i, j }; draw(); });
              svg.appendChild(r);
            }
          }
          grid.appendChild(h("figure", { class: "attn-fig" + (sel.m === m ? " is-sel" : "") }, h("figcaption", null, headName(l, hh)), svg));
        });
        const [l, hh] = HEADS[sel.m];
        const w = tr.layers[l].heads[hh].att[sel.i][sel.j];
        const rowW = tr.layers[l].heads[hh].att[sel.i];
        const top = rowW.indexOf(Math.max(...rowW));
        read.innerHTML = `<b>${headName(l, hh)}</b>, asking from <code>${tk[sel.i].trim()}</code> (position ${sel.i}): weight <b class="num">${w.toFixed(2)}</b> on <code>${tk[sel.j].trim()}</code>. Its largest weight in that row is on <code>${tk[top].trim()}</code> (${rowW[top].toFixed(2)}).`;
        deck.strip.set({ tokens: tk, focus: [sel.i], target: sel.j });
      }
      this.draw = draw;
      draw();
    },
    enter() { if (this.draw) this.draw(); },
  };

  // ------------------------------------------------------------ block
  W.block = {
    init(el, deck) {
      let pos = 4;
      posPicker(el.querySelector(".block-ctl"), { value: pos, label: "Position", onChange: (i) => { pos = i; draw(); } });
      const rows = el.querySelector(".block-rows");
      const norm = (v) => Math.sqrt(v.reduce((a, b) => a + b * b, 0));
      function draw() {
        const tr = trace();
        const items = [
          { label: "x₀ = token + position", vec: tr.x0[pos], kind: "stream" },
          { label: "+ layer 1 attention", vec: tr.layers[0].attnOut[pos], kind: "delta" },
          { label: "stream after attention", vec: tr.layers[0].mid[pos], kind: "stream" },
          { label: "+ layer 1 MLP", vec: tr.layers[0].mlpOut[pos], kind: "delta" },
          { label: "stream after layer 1", vec: tr.layers[0].post[pos], kind: "stream" },
          { label: "+ layer 2 attention", vec: tr.layers[1].attnOut[pos], kind: "delta" },
          { label: "+ layer 2 MLP", vec: tr.layers[1].mlpOut[pos], kind: "delta" },
          { label: "stream after layer 2", vec: tr.layers[1].post[pos], kind: "stream" },
        ];
        clear(rows);
        items.forEach((it) => rows.appendChild(h("div", { class: `vec-row blk-${it.kind}` },
          h("div", { class: "vec-label" }, it.label, h("span", { class: "vec-note" }, `size ${norm(it.vec).toFixed(1)}`)),
          heatStrip(it.vec, { maxAbs: Math.max(...it.vec.map(Math.abs)) }))));
        deck.strip.set({ focus: [pos] });
      }
      this.draw = draw;
      draw();
    },
    enter() { if (this.draw) this.draw(); },
  };

  // ------------------------------------------------------------ logits
  W.logits = {
    init(el) {
      const pos = 4;
      const z = trace().logits[pos];
      let T = 1, topK = 0, topP = 1;
      const table = el.querySelector(".lg-table"), note = el.querySelector(".lg-note");
      const order = z.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 8).map(([, i]) => i);
      const zmax = Math.max(...order.map((i) => Math.abs(z[i])));
      const ctl = el.querySelector(".lg-ctl");
      ctl.append(
        slider({ id: "lg-t", label: "Temperature", min: 0.1, max: 3, step: 0.1, value: 1, format: (v) => v.toFixed(1), onInput: (v) => { T = v; draw(); } }),
        slider({ id: "lg-k", label: "Top-k", min: 0, max: 8, step: 1, value: 0, format: (v) => (v === 0 ? "off" : String(v)), onInput: (v) => { topK = v; draw(); } }),
        slider({ id: "lg-p", label: "Top-p", min: 0.05, max: 1, step: 0.05, value: 1, format: (v) => (v >= 1 ? "off" : v.toFixed(2)), onInput: (v) => { topP = v; draw(); } }));
      function draw() {
        const base = M.softmax(z, T);
        const p = M.filterProbs(base, { topK, topP });
        clear(table);
        table.appendChild(h("div", { class: "lg-row lg-head" }, h("span", null, "Token"), h("span", null, "Logit"), h("span", null, "Probability")));
        order.forEach((i) => {
          const kept = p[i] > 0;
          table.appendChild(h("div", { class: "lg-row" + (kept ? "" : " is-off") + (V[i] === " id" ? " is-target" : "") },
            chip(V[i]),
            h("span", { class: "lg-z" }, h("span", { class: "zbar" }, h("span", { class: "zfill" + (z[i] < 0 ? " neg" : ""), style: { width: `${(Math.abs(z[i]) / zmax) * 50}%`, [z[i] < 0 ? "right" : "left"]: "50%" } })), h("span", { class: "num" }, fmt.minus(z[i].toFixed(2)))),
            h("span", { class: "lg-p" }, h("span", { class: "w-track" }, h("span", { class: "w-fill", style: { width: `${p[i] * 100}%` } })), h("span", { class: "num" }, kept ? fmt.p(p[i]) : "0"))));
        });
        const keptN = p.filter((x) => x > 0).length;
        const restMass = order.reduce((acc, i) => acc - base[i], 1);
        note.textContent = `${keptN} of 113 tokens can be drawn. Tokens outside the top 8 hold ${fmt.p(Math.max(0, restMass))} of the probability before filtering.`;
      }
      draw();
    },
  };

  // ------------------------------------------------------------ loss
  W.loss = {
    init(el) {
      const chartEl = el.querySelector(".loss-chart"), stats = el.querySelector(".loss-stats"), legend = el.querySelector(".loss-legend");
      const tr = trace();
      const run = tr.probs.map((p, i) => -Math.log(p[ids[i + 1]]));
      const ex = M.exactLoss();
      // what a perfect model would score on our query: −ln of the data's true probabilities
      const runIdeal = [Math.log(12), 0, 0, 0, Math.log(3), 0, Math.log(9), 0];
      let mode = "all";
      function draw() {
        clear(chartEl);
        const all = mode === "all";
        const vals = all ? ex.perPos : run;
        const ref = all ? M.data.floor : runIdeal;
        const Wd = 560, Hd = 240, ml = 40, mb = 36, mt = 12, bw = (Wd - ml - 10) / 8;
        const ymax = 2.6;
        const Y = (v) => mt + (1 - v / ymax) * (Hd - mt - mb);
        const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "chart", role: "img", "aria-label": all ? "Average loss and floor at each position" : "Our query's loss and the ideal model's loss at each position" });
        [0, 0.5, 1, 1.5, 2, 2.5].forEach((v) => {
          svg.appendChild(s("line", { x1: ml, x2: Wd - 6, y1: Y(v), y2: Y(v), class: "grid" }));
          svg.appendChild(s("text", { x: ml - 6, y: Y(v) + 4, class: "tick", "text-anchor": "end" }, v.toFixed(1)));
        });
        const ROLE = ["column", "FROM", "table", "WHERE", "filter", "=", "value", ";"];
        for (let i = 0; i < 8; i++) {
          const x = ml + i * bw + bw * 0.18, w = bw * 0.64;
          const v = vals[i], f = ref[i];
          svg.appendChild(s("rect", { x, y: Y(v), width: w, height: Math.max(0.5, Y(0) - Y(v)), rx: 3, class: "bar-acc",
            "data-tip": all ? `${ROLE[i]}: average loss ${v.toFixed(3)} · floor ${f.toFixed(3)}` : `predict ${toks[i + 1].trim()} after ${toks[i].trim()}\nloss ${v.toFixed(3)} · ideal model ${f.toFixed(3)} · perplexity ${Math.exp(v).toFixed(2)}` }));
          svg.appendChild(s("line", { x1: x - 4, x2: x + w + 4, y1: Y(f), y2: Y(f), class: "floor-mark" }));
          svg.appendChild(s("text", { x: x + w / 2, y: Y(Math.max(v, f)) - 6, class: "bar-num", "text-anchor": "middle" }, v.toFixed(2)));
          svg.appendChild(s("text", { x: x + w / 2, y: Hd - mb + 16, class: "tick", "text-anchor": "middle" }, all ? ROLE[i] : toks[i + 1].trim()));
        }
        svg.appendChild(s("line", { x1: ml, x2: Wd - 6, y1: Y(0), y2: Y(0), class: "axis" }));
        svg.appendChild(s("text", { x: 12, y: (mt + Hd - mb) / 2, class: "ax-label", "text-anchor": "middle", transform: `rotate(-90 12 ${(mt + Hd - mb) / 2})` }, "loss (nats)"));
        chartEl.appendChild(svg);
        clear(legend);
        legend.append(
          h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--accent)" } }), all ? "trained model, averaged over the data" : "trained model on our query"),
          h("span", { class: "key" }, h("span", { class: "sw sw-floor" }), all ? "floor: entropy of the data, a true lower bound" : "ideal model on our query (not a bound for one query)"));
        const mean = vals.reduce((a, b) => a + b, 0) / 8, fm = ref.reduce((a, b) => a + b, 0) / 8;
        clear(stats);
        stats.append(
          h("div", { class: "stat" }, h("div", { class: "stat-v" }, mean.toFixed(4)), h("div", { class: "stat-l" }, all ? "average loss" : "loss, our query")),
          h("div", { class: "stat" }, h("div", { class: "stat-v" }, fm.toFixed(4)), h("div", { class: "stat-l" }, all ? "floor" : "ideal model, our query")),
          h("div", { class: "stat" }, h("div", { class: "stat-v" }, Math.exp(mean).toFixed(3)), h("div", { class: "stat-l" }, "perplexity")));
      }
      el.querySelector(".loss-toggle").appendChild(segmented({ label: "Data", value: mode, options: [{ value: "all", label: "Averaged over the data" }, { value: "run", label: "Our query" }], onChange: (v) => { mode = v; draw(); } }));
      draw();
    },
  };

  // ------------------------------------------------------------ gradient
  W.gradient = {
    init(el) {
      const names = [" id", " name", " email", " age"];
      const truth = [1 / 3, 0, 1 / 3, 1 / 3];
      let z, step, lr = 0.5, rand, history, last;
      const bars = barList(el.querySelector(".gd-bars"), { labelWidth: "6.4rem" });
      const lastEl = el.querySelector(".gd-last"), chartEl = el.querySelector(".gd-chart");
      const expLoss = (p) => -truth.reduce((acc, q, i) => acc + (q > 0 ? q * Math.log(p[i]) : 0), 0);
      function reset() {
        z = [0, 0, 0, 0]; step = 0; rand = M.mulberry32(11); last = null;
        history = [[0, expLoss(M.softmax(z))]];
        draw();
      }
      function doStep() {
        const p = M.softmax(z);
        const r = rand();
        const tgt = r < 1 / 3 ? 0 : r < 2 / 3 ? 2 : 3;
        const g = p.map((pi, i) => pi - (i === tgt ? 1 : 0));
        z = z.map((zi, i) => zi - lr * g[i]);
        step++;
        last = { tgt, g, p };
        history.push([step, expLoss(M.softmax(z))]);
      }
      function many(n) { for (let i = 0; i < n; i++) doStep(); draw(); }
      function draw() {
        const p = M.softmax(z);
        bars.update(names.map((t, i) => ({ key: t, value: p[i], label: chip(t), cls: last && last.tgt === i ? "is-target" : "", valueText: fmt.p(p[i]) })));
        clear(lastEl);
        if (!last) lastEl.append("All four logits start at 0, so each token gets 0.250. Press ", h("b", null, "Step"), ".");
        else lastEl.append(h("span", null, `Step ${step}: the data said `), chip(names[last.tgt]), h("span", null, ". Gradient p − y = "),
          h("span", { class: "num" }, "[" + last.g.map((x) => fmt.signed(x)).join(", ") + "]"),
          h("span", null, `. Logits now [${z.map((x) => fmt.minus(x.toFixed(2))).join(", ")}].`));
        clear(chartEl);
        const xmax = Math.max(20, history[history.length - 1][0]);
        chart(chartEl, {
          width: 520, height: 170, margin: { l: 44, r: 16, t: 10, b: 34 }, aria: "Expected loss over steps",
          x: { domain: [0, xmax], ticks: [0, Math.round(xmax / 2), xmax], label: "steps" },
          y: (() => { const top = Math.max(1.45, ...history.map((p) => p[1])) + 0.05; const t = []; for (let v = 1; v <= top + 1e-9; v += top > 2 ? 0.5 : 0.1) t.push(+v.toFixed(1)); return { domain: [1, top], ticks: t, fmt: (v) => v.toFixed(1) }; })(),
          hlines: [{ y: Math.log(3), label: "floor ln 3 = 1.099" }],
          series: [{ points: history, cls: "accent" }],
        });
      }
      el.querySelector(".gd-ctl").append(
        button("Step", () => many(1), { kind: "primary" }), button("10 steps", () => many(10)), button("100 steps", () => many(100)), button("Reset", reset),
        slider({ id: "gd-lr", label: "Learning rate", min: 0.05, max: 5, step: 0.05, value: lr, format: (v) => v.toFixed(2), onInput: (v) => { lr = v; } }));
      reset();
    },
  };

  // ------------------------------------------------------------ backprop
  W.backprop = {
    init(el) {
      const pos = 4, tgt = M.idOf(" id");
      const tr = trace();
      const hvec = tr.final[pos];
      const Wu = M.weights().wu;
      const wcol = Array.from({ length: C.d }, (_, r) => Wu.data[r * Wu.shape[1] + tgt]);
      const zi = tr.logits[pos][tgt], p = tr.probs[pos][tgt], L = -Math.log(p);
      const dz = p - 1, dw = hvec.map((x) => dz * x);
      const graph = el.querySelector(".bp-graph"), note = el.querySelector(".bp-note");
      let stage = 0;
      const m = Math.max(...hvec.map(Math.abs), ...wcol.map(Math.abs));
      const md = Math.max(...dw.map(Math.abs));
      const STEPS = [
        { title: h("span", null, "final vector ", h("i", null, "h"), " at ", h("code", null, "WHERE")), fwd: () => heatStrip(hvec, { maxAbs: m }), bwd: "∂loss/∂h = Σⱼ (pⱼ − yⱼ) wⱼ, and it keeps flowing into layer 2" },
        { title: h("span", null, "logit for ", chip(" id"), ": ", h("i", null, "z = h · w")), fwd: () => h("span", { class: "num" }, zi.toFixed(3)), bwd: `∂loss/∂z = p − 1 = ${fmt.minus(dz.toFixed(3))}` },
        { title: "softmax over all 113 logits", fwd: () => h("span", { class: "num" }, `p(id) = ${p.toFixed(3)}`), bwd: `∂loss/∂p = −1/p = ${fmt.minus((-1 / p).toFixed(3))}` },
        { title: "loss = −ln p", fwd: () => h("span", { class: "num" }, L.toFixed(3)), bwd: "∂loss/∂loss = 1 (the starting point)" },
      ];
      function draw() {
        clear(graph);
        const f = stage >= 1, b = stage >= 2;
        const chain = h("ol", { class: "bp-chain" });
        STEPS.forEach((st, i) => {
          chain.appendChild(h("li", { class: "bp-step" + (f ? " fwd" : "") + (b ? " bwd" : "") },
            h("div", { class: "bp-title" }, st.title),
            h("div", { class: "bp-val" }, f ? st.fwd() : h("span", { class: "dim" }, "not computed yet")),
            h("div", { class: "bp-grad" }, b ? st.bwd : h("span", { class: "dim" }, "gradient: after the backward pass"))));
        });
        graph.append(h("div", { class: "bp-legend" }, h("span", { class: "fw" }, "↓ forward: values"), h("span", { class: "bw" }, "↑ backward: gradients")), chain);
        graph.appendChild(h("div", { class: "bp-w" },
          h("div", { class: "vec-row" }, h("div", { class: "vec-label" }, "weights ", h("i", null, "w"), ": column ", h("b", { class: "num" }, String(tgt)), " of the 16 × 113 output matrix"), heatStrip(wcol, { maxAbs: m })),
          h("div", { class: "vec-row" + (b ? "" : " dim") }, h("div", { class: "vec-label" }, "their gradient ∂loss/∂w = (p − 1) · h", h("span", { class: "vec-note" }, b ? "computed" : "after backward")), b ? heatStrip(dw, { maxAbs: md }) : heatStrip(dw.map(() => 0), { maxAbs: 1 }))));
        note.innerHTML = stage === 0
          ? "Press <b>Forward</b> to compute the prediction at <code>WHERE</code>, then <b>Backward</b> to send the gradient back."
          : stage === 1
            ? `Forward pass, top to bottom: the model gives <code>id</code> probability ${p.toFixed(3)}, so the loss is ${L.toFixed(3)}. Each step keeps what it will need on the way back.`
            : `Backward pass, bottom to top: multiply local derivatives. −1/p times the softmax slope p(1 − p) gives p − 1 = ${fmt.minus(dz.toFixed(3))} for the logit. Each weight's gradient is that number times the matching entry of <i>h</i>. The same rule continues through every layer below.`;
      }
      el.querySelector(".bp-ctl").append(
        button("Forward", () => { stage = Math.max(stage, 1); draw(); }, { kind: "primary" }),
        button("Backward", () => { stage = 2; draw(); }),
        button("Reset", () => { stage = 0; draw(); }));
      draw();
    },
  };

  // ------------------------------------------------------------ optim
  W.optim = {
    init(el) {
      const K = 25;
      const f = (a, b) => 0.5 * (K * a * a + b * b);
      const grad = (a, b) => [K * a, b];
      const start = [-1.0, 1.0];
      let lr = 0.07, paths, st, steps;
      const plot = el.querySelector(".opt-plot");
      const lrAt = () => lr;
      function reset() {
        steps = 0;
        paths = { sgd: [start.slice()], adam: [start.slice()] };
        st = { m: [0, 0], v: [0, 0] };
        draw();
      }
      function step() {
        const eta = lrAt(steps);
        steps++;
        let [a, b] = paths.sgd[paths.sgd.length - 1];
        let g = grad(a, b);
        paths.sgd.push([a - eta * g[0], b - eta * g[1]]);
        [a, b] = paths.adam[paths.adam.length - 1];
        g = grad(a, b);
        st.m = st.m.map((x, i) => 0.9 * x + 0.1 * g[i]);
        st.v = st.v.map((x, i) => 0.999 * x + 0.001 * g[i] * g[i]);
        const mh = st.m.map((x) => x / (1 - Math.pow(0.9, steps))), vh = st.v.map((x) => x / (1 - Math.pow(0.999, steps)));
        paths.adam.push([a - eta * mh[0] / (Math.sqrt(vh[0]) + 1e-8), b - eta * mh[1] / (Math.sqrt(vh[1]) + 1e-8)]);
      }
      const R = 1.3;
      const inBox = (p) => Math.abs(p[0]) <= R && Math.abs(p[1]) <= R;
      function draw() {
        clear(plot);
        const Wd = 440, Hd = 300, X = (a) => Wd / 2 + a * (Wd / 2 / R), Y = (b) => Hd / 2 - b * (Hd / 2 / R);
        const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "chart opt-svg", role: "img", "aria-label": "Paths of SGD and Adam on an elongated bowl" });
        const clipId = window.UI.nextId("oclip");
        svg.appendChild(s("defs", null, s("clipPath", { id: clipId }, s("rect", { x: 0, y: 0, width: Wd, height: Hd }))));
        [0.02, 0.1, 0.3, 0.6, 1.0, 1.6, 2.5].forEach((c) => {
          svg.appendChild(s("ellipse", { cx: X(0), cy: Y(0), rx: Math.sqrt((2 * c) / K) * (Wd / 2 / R), ry: Math.sqrt(2 * c) * (Hd / 2 / R), class: "contour" }));
        });
        svg.appendChild(s("line", { x1: 0, x2: Wd, y1: Y(0), y2: Y(0), class: "grid" }));
        svg.appendChild(s("line", { x1: X(0), x2: X(0), y1: 0, y2: Hd, class: "grid" }));
        svg.appendChild(s("text", { x: Wd - 6, y: Y(0) - 6, class: "tick", "text-anchor": "end" }, "a: steep"));
        svg.appendChild(s("text", { x: X(0) + 6, y: 14, class: "tick" }, "b: flat"));
        const cls = { sgd: "s1", adam: "accent" };
        for (const [k, path] of Object.entries(paths)) {
          svg.appendChild(s("polyline", { points: path.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" "), class: `series thin ${cls[k]}`, fill: "none", "clip-path": `url(#${clipId})` }));
          const lp = path[path.length - 1];
          if (inBox(lp)) svg.appendChild(s("circle", { cx: X(lp[0]), cy: Y(lp[1]), r: 5, class: `dot ${k === "sgd" ? "s1" : ""}` }));
        }
        svg.appendChild(s("circle", { cx: X(start[0]), cy: Y(start[1]), r: 4, class: "dot ink" }));
        svg.appendChild(s("text", { x: X(start[0]) + 8, y: Y(start[1]) - 6, class: "d-label" }, "start"));
        svg.appendChild(s("circle", { cx: X(0), cy: Y(0), r: 3, class: "dot ink" }));
        plot.appendChild(svg);
        plot.appendChild(h("div", { class: "legend" },
          h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--s1)" } }), "SGD"),
          h("span", { class: "key" }, h("span", { class: "sw", style: { background: "var(--accent)" } }), "Adam (β₁ = 0.9, β₂ = 0.999)")));
        const where = (k) => {
          const p = paths[k][paths[k].length - 1];
          if (!isFinite(f(p[0], p[1])) || Math.abs(p[0]) > 1e3) return "diverged";
          return `a ${fmt.minus(p[0].toFixed(3))}, b ${fmt.minus(p[1].toFixed(3))}`;
        };
        plot.appendChild(h("p", { class: "fig-cap opt-info", "aria-live": "polite" },
          `After ${steps} step${steps === 1 ? "" : "s"}: SGD at ${where("sgd")}; Adam at ${where("adam")}. The minimum is at a = 0, b = 0.`));
      }
      el.querySelector(".opt-ctl").append(
        button("1 step", () => { step(); draw(); }, { kind: "primary" }),
        button("20 steps", () => { for (let i = 0; i < 20; i++) step(); draw(); }),
        button("Reset", reset),
        slider({ id: "opt-lr", label: "Learning rate (both)", min: 0.01, max: 0.1, step: 0.005, value: lr, format: (v) => v.toFixed(3), onInput: (v) => { lr = v; reset(); } }));
      reset();
      // learning-rate schedule, same function as tools/build_model.py
      const cfg = M.config;
      const sched = (t) => (t < cfg.warmup ? cfg.lr * (t + 1) / cfg.warmup : cfg.lr * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * (t - cfg.warmup) / (cfg.steps - cfg.warmup)))));
      const pts = [];
      for (let t = 0; t <= cfg.steps; t += 20) pts.push([t, sched(t)]);
      chart(el.querySelector(".opt-lr"), {
        width: 520, height: 130, margin: { l: 50, r: 14, t: 8, b: 30 }, aria: "Learning rate schedule",
        x: { domain: [0, cfg.steps], ticks: [0, 1000, 2000, 3000, 4000, 5000, 6000], fmt: (v) => v.toLocaleString("en-US") },
        y: { domain: [0, 0.011], ticks: [0, 0.005, 0.01], fmt: (v) => v.toFixed(3) },
        series: [{ points: pts, cls: "accent" }],
      });
    },
  };

  // ------------------------------------------------------------ train
  W.train = {
    init(el, deck) {
      const steps = M.snapSteps;
      let si = steps.length - 1;
      const chartEl = el.querySelector(".tr-chart"), grid = el.querySelector(".tr-grid");
      const floorMean = M.data.floor.reduce((a, b) => a + b, 0) / 8;
      const batch = M.data.batchLoss10.map((v, i) => [i * 10 + 5, v]);
      const exact = M.data.exactCurve.map(([st, v]) => [Math.max(st, 1), v]);
      const ch = chart(chartEl, {
        width: 560, height: 230, margin: { l: 44, r: 16, t: 10, b: 36 }, aria: "Training loss curves",
        x: { type: "log", domain: [1, 6000], ticks: [1, 10, 100, 1000, 6000], fmt: (v) => v.toLocaleString("en-US"), label: "training step (log scale)" },
        y: { type: "log", domain: [0.6, 5.2], ticks: [0.7, 1, 2, 3, 5], fmt: (v) => String(v) },
        series: [{ points: batch, cls: "faint" }, { points: exact, cls: "accent" }],
        hlines: [{ y: floorMean, label: "floor 0.6937" }],
        vlines: [{ x: Math.max(steps[si], 1), cls: "cursor" }],
      });
      const sl = slider({ id: "tr-step", label: "Saved model at step", min: 0, max: steps.length - 1, value: si, format: (v) => steps[v].toLocaleString("en-US"), onInput: (v) => { si = v; draw(); }, wide: true });
      el.querySelector(".tr-ctl").append(sl);
      const playBtn = button("Play training", () => { if (si === steps.length - 1) { si = 0; draw(); } play.toggle(); }, { kind: "primary" });
      const play = player(() => { if (si >= steps.length - 1) return false; si++; draw(); return si < steps.length - 1; }, { interval: 900, onState: (on) => { playBtn.textContent = on ? "Pause" : "Play training"; } });
      el.querySelector(".tr-ctl").append(playBtn);
      function draw() {
        const st = steps[si];
        sl.set(si);
        ch.update({ vlines: [{ x: Math.max(st, 1), cls: "cursor", label: `step ${st.toLocaleString("en-US")}` }] });
        const tr = trace(st);
        clear(grid);
        const ROLE = ["first column", "FROM", "table", "WHERE", "filter column", "=", "value", "end ;"];
        const avgNow = M.data.exactCurve.find(([x]) => x === st)[2];
        for (let i = 0; i < 8; i++) {
          const p = tr.probs[i][ids[i + 1]];
          const curve = M.data.exactCurve.map(([x, , per]) => [Math.max(x, 1), per[i]]);
          const card = h("div", { class: "tr-card" },
            h("div", { class: "tr-top" }, h("span", { class: "tr-role" }, ROLE[i]), chip(toks[i + 1])),
            h("div", { class: "tr-spark" }));
          grid.appendChild(card);
          chart(card.lastChild, {
            width: 150, height: 54, margin: { l: 2, r: 2, t: 4, b: 4 }, aria: `average loss at position ${i}`,
            x: { type: "log", domain: [1, 6000], ticks: [] }, y: { domain: [0, 5], ticks: [] },
            series: [{ points: curve, cls: "accent thin" }],
            hlines: [{ y: M.data.floor[i] }],
            vlines: [{ x: Math.max(st, 1), cls: "cursor" }],
          });
          card.appendChild(h("div", { class: "tr-foot" }, h("span", null, `avg ${avgNow[i].toFixed(2)}`), h("span", { class: "dim" }, `floor ${M.data.floor[i].toFixed(2)}`)));
          card.appendChild(h("div", { class: "tr-ours" }, "our query: p ", h("b", { class: "num" }, fmt.p(p))));
        }
        deck.strip.set({ under: [null].concat(tr.probs.map((p, i) => fmt.p(p[ids[i + 1]]))) });
      }
      this.draw = draw;
      this.play = play;
      draw();
    },
    enter() { if (this.draw) this.draw(); },
    leave() { if (this.play) this.play.pause(); },
  };

  // ------------------------------------------------------------ real
  W.real = {
    init(el) {
      const MODELS = [
        { name: "Ours", layers: 2, d: 16, heads: "2 / 2", ff: 64, vocab: 113, ctx: "8", params: 9968, tokens: 6.144e6 },
        { name: "Llama 3.1 8B", layers: 32, d: 4096, heads: "32 / 8", ff: 14336, vocab: 128256, ctx: "128K", params: 8e9, tokens: 15e12 },
        { name: "Llama 3.1 70B", layers: 80, d: 8192, heads: "64 / 8", ff: 28672, vocab: 128256, ctx: "128K", params: 70e9, tokens: 15e12 },
        { name: "Llama 3.1 405B", layers: 126, d: 16384, heads: "128 / 8", ff: 53248, vocab: 128256, ctx: "128K", params: 405e9, tokens: 15.6e12 },
      ];
      const ROWS = [
        ["Layers", (m) => m.layers.toLocaleString("en-US")],
        ["Vector width", (m) => m.d.toLocaleString("en-US")],
        ["Query / KV heads", (m) => m.heads],
        ["MLP width", (m) => m.ff.toLocaleString("en-US")],
        ["Vocabulary", (m) => m.vocab.toLocaleString("en-US")],
        ["Context (tokens)", (m) => m.ctx],
        ["Parameters", (m) => (m.params < 1e6 ? m.params.toLocaleString("en-US") : `${m.params / 1e9}B`)],
        ["Training tokens", (m) => (m.tokens < 1e9 ? `${(m.tokens / 1e6).toFixed(1)}M` : `${m.tokens >= 15.5e12 ? "15.6" : "~15"}T`)],
      ];
      const table = el.querySelector(".cmp-table");
      table.append(h("thead", null, h("tr", null, h("th", null, ""), MODELS.map((m, i) => h("th", { class: i === 0 ? "is-ours" : "" }, m.name)))),
        h("tbody", null, ROWS.map(([lab, f]) => h("tr", null, h("th", { scope: "row" }, lab), MODELS.map((m, i) => h("td", { class: "num" + (i === 0 ? " is-ours" : "") }, f(m)))))));
      table.parentNode.after(h("p", { class: "fig-cap" }, "Our “training tokens” counts every next-token prediction: 6,000 steps × 128 queries × 8 positions. Llama figures: Meta, The Llama 3 Herd of Models (2024), Table 3; vocabulary of the released models, 128,000 tokens plus 256 reserved special tokens; ~15T training tokens per the Llama 3.1 model card."));
      const bars = el.querySelector(".real-bars");
      let pick = 3;
      function draw() {
        clear(bars);
        const m = MODELS[pick];
        const items = [["parameters", m.params / MODELS[0].params], ["training tokens", m.tokens / MODELS[0].tokens]];
        items.forEach(([lab, ratio]) => bars.appendChild(h("div", { class: "ratio" },
          h("div", { class: "ratio-v" }, `${ratio >= 1e6 ? (ratio / 1e6).toFixed(1) + " million" : Math.round(ratio).toLocaleString("en-US")}×`),
          h("div", { class: "ratio-l" }, `more ${lab} than ours`))));
      }
      el.querySelector(".real-ctl").append(segmented({ label: "Compare ours with", value: pick, options: MODELS.slice(1).map((m, i) => ({ value: i + 1, label: m.name.replace("Llama 3.1 ", "") })), onChange: (v) => { pick = v; draw(); } }));
      draw();
    },
  };
})();

/* Held-out evaluation: the model trained on all queries vs. one that never saw 27 of them. */
(function () {
  "use strict";
  const { h, s, clear, fmt, chip } = window.UI;
  const M = window.TinyLM;
  const W = window.Widgets;
  const stat = (v, l) => h("div", { class: "stat" }, h("div", { class: "stat-v" }, v), h("div", { class: "stat-l" }, l));
  // per-position loss of a model, averaged separately over held-out and seen queries
  function splitLoss(step) {
    const held = new Set(M.data.holdout.queries);
    const acc = { held: new Array(8).fill(0), seen: new Array(8).fill(0) }, w = { held: 0, seen: 0 };
    for (const q of M.allQueries()) {
      const ids = M.encode(q.text).map(M.idOf), key = held.has(q.text) ? "held" : "seen";
      const tr = M.forward(ids.slice(0, 8), step);
      w[key] += q.p;
      for (let i = 0; i < 8; i++) acc[key][i] += -Math.log(tr.probs[i][ids[i + 1]]) * q.p;
    }
    const norm = (k) => acc[k].map((x) => x / w[k]);
    return { held: norm("held"), seen: norm("seen") };
  }
  W._splitLoss = splitLoss;
  W.heldout = {
    init(el) {
      const all = M.runningIds, ids = all.slice(0, 8), toks = M.running;
      const A = M.forward(ids), B = M.forward(ids, "holdout");
      const table = el.querySelector(".ho-table");
      table.appendChild(h("div", { class: "ho-row ho-head" }, h("span", null, "Next token"), h("span", null, "saw all 270"), h("span", null, "never saw ours")));
      for (let i = 0; i < 8; i++) {
        const pa = A.probs[i][all[i + 1]], pb = B.probs[i][all[i + 1]];
        const bar = (p, cls) => h("span", { class: "ho-cell" }, h("span", { class: "w-track" }, h("span", { class: `w-fill ${cls}`, style: { width: `${p * 100}%` } })), h("span", { class: "num" }, fmt.p(p)));
        table.appendChild(h("div", { class: "ho-row" + (Math.abs(pa - pb) > 0.05 ? " is-gap" : "") }, chip(toks[i + 1]), bar(pa, ""), bar(pb, "b")));
      }
      const sl = splitLoss("holdout");
      const ROLE = ["first col", "FROM", "table", "WHERE", "filter col", "=", "value", ";"];
      const Wd = 560, Hd = 210, ml = 40, mb = 30, mt = 10, bw = (Wd - ml - 10) / 8, ymax = 7;
      const Y = (v) => mt + (1 - Math.min(v, ymax) / ymax) * (Hd - mt - mb);
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "chart", role: "img", "aria-label": "Held-out versus seen loss by position for the model that never saw the held-out queries" });
      [0, 2, 4, 6].forEach((v) => { svg.appendChild(s("line", { x1: ml, x2: Wd - 6, y1: Y(v), y2: Y(v), class: "grid" })); svg.appendChild(s("text", { x: ml - 6, y: Y(v) + 4, class: "tick", "text-anchor": "end" }, String(v))); });
      for (let i = 0; i < 8; i++) {
        const x = ml + i * bw + bw * 0.14, w = bw * 0.34;
        svg.appendChild(s("rect", { x, y: Y(sl.seen[i]), width: w, height: Math.max(0.5, Y(0) - Y(sl.seen[i])), rx: 2, class: "ho-seen", "data-tip": `${ROLE[i]}: seen queries ${sl.seen[i].toFixed(3)}` }));
        svg.appendChild(s("rect", { x: x + w + 2, y: Y(sl.held[i]), width: w, height: Math.max(0.5, Y(0) - Y(sl.held[i])), rx: 2, class: "ho-held", "data-tip": `${ROLE[i]}: held-out queries ${sl.held[i].toFixed(3)}` }));
        svg.appendChild(s("text", { x: x + w + 1, y: Hd - mb + 16, class: "tick", "text-anchor": "middle" }, ROLE[i]));
      }
      svg.appendChild(s("line", { x1: ml, x2: Wd - 6, y1: Y(0), y2: Y(0), class: "axis" }));
      el.querySelector(".ho-chart").append(svg, h("div", { class: "legend" },
        h("span", { class: "key" }, h("span", { class: "sw ho-seen-sw" }), "queries it trained on"),
        h("span", { class: "key" }, h("span", { class: "sw ho-held-sw" }), "held-out queries")));
      const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
      el.querySelector(".ho-out").append(stat(mean(sl.seen).toFixed(2), "loss on queries it saw"), stat(mean(sl.held).toFixed(2), "loss on held-out queries"), stat(fmt.p(B.probs[6][ids[7]]), "p(7) for our query"));
    },
  };
})();
