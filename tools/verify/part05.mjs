// Part 5: every number stated on the transformer sheets, recomputed.
// Model numbers come from TinyLM (the same weights the page runs); formulas and
// published configurations are recomputed here independently of the content file.
// Interactive custom figures are mounted in jsdom and driven to their extremes.
import { claim, near, equal, ok, r, int, TinyLM, Deck, window, text, done } from "./lib.mjs";

const M = TinyLM, D = M.data, C = M.config;
const IDS = M.runningIds.slice(0, 8), ALL = M.runningIds, TOK = M.running;
const tr = M.forward(IDS);
const W = M.weights();
const sq = (x) => x * x;
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (v) => Math.sqrt(dot(v, v));
const mx = (a) => Math.max(...a);
const mn = (a) => Math.min(...a);
const sum = (a) => a.reduce((s, v) => s + v, 0);
const softmax = (z) => { const m = mx(z); const e = z.map((v) => Math.exp(v - m)); const t = sum(e); return e.map((v) => v / t); };
const minus = (s) => String(s).replace(/^-/, "−");
const R = (x, d) => minus(r(x, d)); // as printed in prose, with a true minus sign
const id = (t) => M.idOf(t);
const Q = M.allQueries().map((q) => ({ ...q, ids: M.encode(q.text).map(M.idOf) }));

// Llama 3 shapes from the paper (Table 3) and Meta's reference code (vocab 128,256 rows,
// head size d / heads, SwiGLU with three d × d_ff matrices, untied output, RMSNorm gains).
const LLAMA = {
  "8B": { L: 32, d: 4096, ff: 14336, H: 32, KV: 8 },
  "70B": { L: 80, d: 8192, ff: 28672, H: 64, KV: 8 },
  "405B": { L: 126, d: 16384, ff: 53248, H: 128, KV: 8 },
};
const VOCAB = 128256;
function llamaCount(c) {
  const hd = c.d / c.H;
  const attn = c.d * c.d * 2 + 2 * c.d * c.KV * hd;
  const mlp = 3 * c.d * c.ff;
  const layer = attn + mlp + 2 * c.d;
  return { hd, attn, mlp, layer, layerW: attn + mlp, emb: VOCAB * c.d, total: c.L * layer + 2 * VOCAB * c.d + c.d };
}
const L8 = llamaCount(LLAMA["8B"]), L70 = llamaCount(LLAMA["70B"]), L405 = llamaCount(LLAMA["405B"]);

// jsdom mount for custom figures
function mount(type, spec) {
  const el = window.document.createElement("div");
  window.document.body.appendChild(el);
  const inst = Deck.figs[type](el, spec, { strip: { set() {}, reset() {} }, slide: {} });
  return { el, inst };
}
const tiles = (el) => [...el.querySelectorAll(".f-tile-v")].map((e) => e.textContent);
const setRange = (el, k, v) => { const inp = el.querySelectorAll('input[type="range"]')[k]; inp.value = String(v); inp.dispatchEvent(new window.Event("input")); };
const pressSeg = (el, group, value) => { const g = el.querySelectorAll(".ctl-seg")[group]; [...g.querySelectorAll(".seg-btn")].find((b) => b.dataset.v === String(value)).click(); };

// ------------------------------------------------------------------ structure
ok("part 5 exists", Deck.parts.some((p) => p.n === 5));
const p5 = Deck.parts.find((p) => p.n === 5);
ok(`part 5 has 7 chapters (has ${p5.chapters.length})`, p5.chapters.length === 7 || process.env.PARTIAL);

// ------------------------------------------------------------------ 5.1 Why attention
// every query has 9 tokens and FROM at position 2, so x0 at position 2 is the same everywhere
ok("all 270 queries are 9 tokens", Q.every((q) => q.ids.length === 9));
ok("FROM at position 2 in every query", Q.every((q) => q.ids[2] === id(" FROM")));
{
  const x2 = tr.x0[2];
  ok("x0 at position 2 identical across queries", Q.every((q) => M.forward(q.ids.slice(0, 3)).x0[2].every((v, c) => v === x2[c])));
}
claim("tf-need", "all 270 queries");
ok("name only in users", Object.entries(D.schema).filter(([, c]) => c.includes("name")).map(([t]) => t).join() === "users");
ok("total only in orders", Object.entries(D.schema).filter(([, c]) => c.includes("total")).map(([t]) => t).join() === "orders");
{
  const p4 = tr.probs[4];
  for (const [t, want] of [[" id", "0.346"], [" email", "0.345"], [" age", "0.307"], [" name", "0.001"]]) { equal(`p(${t} | WHERE)`, r(p4[id(t)], 3), want); claim("tf-need", want); }
}
claim("tf-rnn", "131,072"); near("128K", 128 * 1024, 131072);
claim("tf-rnn", "8 dependent steps");
claim("tf-rnn", "3 more updates"); near("positions 1 to 4", 4 - 1, 3);
equal("n^2 d", int(8 * 8 * 16), "1,024"); claim("tf-rnn", "n²·d = 1,024");
equal("n d^2", int(8 * 16 * 16), "2,048"); claim("tf-rnn", "n·d² = 2,048");
{
  const a2 = tr.layers[0].heads[1].att[4], a1 = tr.layers[0].heads[0].att[4];
  equal("L1H2 WHERE->users", r(a2[3], 2), "1.00"); claim("tf-attn-idea", "1.00 on <code>users</code>".replace(/<[^>]+>/g, " ").replace(/\s+/g, " "));
  ok("L1H2 WHERE other weights < 0.005", a2.every((v, j) => j === 3 || v < 0.005)); claim("tf-attn-idea", "less than 0.005");
  const o = tr.layers[0].concat[4].slice(8, 16), v = tr.layers[0].v[3].slice(8, 16);
  ok("L1H2 output at WHERE = v(users) to 3 dp", o.every((x, c) => r(x, 3) === r(v[c], 3))); claim("tf-attn-idea", "to 3 decimals");
  equal("L1H1 WHERE->name", r(a1[1], 2), "0.98"); claim("tf-attn-idea", "0.98");
  near("rows sum to 1", sum(a1.slice(0, 5)), 1, 1e-12);
}
equal("8 x 113", int(8 * 113), "904"); claim("tf-plan", "904");
equal("8 x 16", int(8 * 16), "128"); claim("tf-plan", "128 = 8 × 16");
claim("tf-plan", "16 × 113");

// ------------------------------------------------------------------ 5.2 Self-attention step by step
equal("WHERE id", id(" WHERE"), 52); claim("tf-input", "Row 52");
equal("W_E shape", W.wte.shape.join("x"), "113x16"); claim("tf-input", "(113 × 16)");
equal("W_P shape", W.wpe.shape.join("x"), "8x16"); claim("tf-input", "(8 × 16)");
{
  const a = tr.tokEmb[4][0], b = tr.posEmb[4][0], x = tr.x0[4][0];
  near("tokEmb + posEmb = x0", a + b, x, 1e-12);
  claim("tf-input", `${R(a, 3)} + ${R(b, 3)} = ${R(x, 3)}`);
  near("displayed sum consistent", +r(a, 3) + +r(b, 3), +r(x, 3), 1e-9);
  ok("x0[4] is wte[52] + wpe[4]", tr.x0[4].every((v, c) => Math.abs(v - (W.wte.data[52 * 16 + c] + W.wpe.data[4 * 16 + c])) < 1e-12));
}
{
  const q = tr.layers[0].q[4].slice(8, 16), k = tr.layers[0].k[3].slice(8, 16);
  const raw = dot(q, k);
  claim("tf-scores", R(raw, 2)); equal("raw dot", r(raw, 2), "40.28");
  const s = tr.layers[0].heads[1].scores[4];
  near("scaled score from trace", s[3], raw / Math.sqrt(8), 1e-12);
  for (const j of [0, 1, 2, 3, 4]) claim("tf-scores", R(s[j], 2));
  equal("score users", r(s[3], 2), "14.24");
  ok("users has the largest score", s.slice(0, 5).every((v, j) => j === 3 || v < s[3]));
  claim("tf-scores", "8 multiplications and 7 additions");
}
// √d: the seeded experiment (independent implementation of the same procedure)
function scaleExp(d, seed = 11, rows = 250, keys = 8) {
  const rand = M.mulberry32(seed);
  let spare = null;
  const normal = () => {
    if (spare !== null) { const z = spare; spare = null; return z; }
    let u = 0; while (u === 0) u = rand();
    const v = rand(), rr = Math.sqrt(-2 * Math.log(u));
    spare = rr * Math.sin(2 * Math.PI * v); return rr * Math.cos(2 * Math.PI * v);
  };
  const dots = []; let a = 0, b = 0, sat = 0;
  for (let i = 0; i < rows; i++) {
    const q = Array.from({ length: d }, normal), sc = [];
    for (let j = 0; j < keys; j++) { const k = Array.from({ length: d }, normal); const x = dot(q, k); sc.push(x); dots.push(x); }
    const pr = mx(softmax(sc)), ps = mx(softmax(sc.map((x) => x / Math.sqrt(d))));
    a += pr; b += ps; if (pr > 0.99) sat++;
  }
  const mean = sum(dots) / dots.length, sd = Math.sqrt(sum(dots.map((x) => sq(x - mean))) / dots.length);
  return { sd, sds: sd / Math.sqrt(d), maxRaw: a / rows, maxScaled: b / rows, sat: sat / rows, dots };
}
{
  const e = scaleExp(128);
  claim("tf-sqrt-d", r(e.sd, 2)); equal("sd at 128", r(e.sd, 2), "11.40");
  claim("tf-sqrt-d", r(e.maxRaw, 3));
  claim("tf-sqrt-d", `${Math.round(e.sat * 100)}%`);
  claim("tf-sqrt-d", r(e.sds, 3));
  claim("tf-sqrt-d", r(e.maxScaled, 3));
  claim("tf-sqrt-d", "250 queries");
  near("2,000 dots", e.dots.length, 2000);
  equal("p(1-p) at 0.99", r(0.99 * 0.01, 4), "0.0099"); claim("tf-sqrt-d", "0.0099");
  equal("p(1-p) at 0.5", r(0.25, 2), "0.25"); claim("tf-sqrt-d", "0.25 at p = 0.5");
  // across all head sizes the scaled std stays within 5% of 1, the unscaled within 5% of sqrt(d)
  for (const d of [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]) {
    const x = scaleExp(d);
    ok(`sd ≈ sqrt(${d})`, Math.abs(x.sd / Math.sqrt(d) - 1) < 0.05);
  }
  // the figure, at its default and both extremes
  const { el } = mount("tf-scale", { d: 128, seed: 11 });
  const want = (d) => { const x = scaleExp(d); return [r(x.sd, 2), r(Math.sqrt(d), 2), r(x.sds, 3), r(x.maxRaw, 3), r(x.maxScaled, 3), `${Math.round(x.sat * 100)}%`]; };
  equal("tf-scale default tiles", tiles(el), want(128));
  setRange(el, 0, 0); equal("tf-scale d=1 tiles", tiles(el), want(1));
  setRange(el, 0, 9); equal("tf-scale d=512 tiles", tiles(el), want(512));
  const over = el.querySelector(".f-note").textContent;
  const outside = scaleExp(512).dots.filter((x) => x < -30 || x >= 30).length;
  ok(`tf-scale overflow note (${over})`, over.startsWith(outside.toLocaleString("en-US")));
}
// the causal mask
{
  const L = tr.layers[0], q = L.q[4].slice(0, 8);
  const S = L.k.map((k) => dot(q, k.slice(0, 8)) / Math.sqrt(8));
  claim("tf-mask", R(S[5], 2)); equal("future score", r(S[5], 2), "5.69");
  ok("pos 5 scores higher than every visible position", S.slice(0, 5).every((v) => v < S[5]));
  const a = softmax(S);
  claim("tf-mask", r(a[5], 3)); equal("unmasked weight on id", r(a[5], 3), "0.896");
  ok("position 5 is id", TOK[5] === " id");
  // figure: stage 3 for every row, including the extremes
  const { el } = mount("tf-mask", { head: 0, stage: 0, row: 4 });
  ok("tf-mask stage 1 shows all 64 scores", el.querySelectorAll("td.tf-mask-c").length === 64 && el.querySelectorAll("td.masked").length === 0);
  pressSeg(el, 1, 1);
  ok("tf-mask stage 2 masks 28 entries", el.querySelectorAll("td.masked").length === 28);
  pressSeg(el, 1, 2);
  const sums = [...el.querySelectorAll("td.tf-mask-sum")].map((e) => e.textContent);
  equal("tf-mask row sums", sums, new Array(8).fill("1.00"));
  const expl = () => el.querySelector(".tf-expl").textContent;
  ok("tf-mask row 4 explanation", expl().includes(r(a.slice(5).reduce((x, y) => x + y, 0), 3)));
  for (const [hh, l] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    pressSeg(el, 0, l * 2 + hh);
    const att = tr.layers[l].heads[hh].att;
    const cells = [...el.querySelectorAll("tr")].slice(1).map((row) => [...row.querySelectorAll("td.tf-mask-c")].map((c) => c.textContent));
    ok(`tf-mask weights L${l + 1}H${hh + 1}`, cells.every((row, i) => row.every((c, j) => c === (j > i ? "0" : minus(r(att[i][j], 2)).replace(/^−?0\.00$/, "0.00")))));
  }
  const rows = el.querySelectorAll("tr");
  rows[1].click(); ok("tf-mask row 0", expl().includes("Row 0") && expl().includes("positions 1–7"));
  rows[8].click(); ok("tf-mask row 7", expl().includes("Row 7") && expl().includes("Nothing was masked"));
}
{
  const s = tr.layers[0].heads[1].scores[4].slice(0, 5);
  const gap = s[3] - mx(s.filter((_, j) => j !== 3));
  claim("tf-mix", r(gap, 2)); equal("gap", r(gap, 2), "15.29");
  equal("e^-gap", Math.exp(-gap).toExponential(0), "2e-7"); claim("tf-mix", String.raw`e^{-15.29} \approx 2 \times 10^{-7}`);
  claim("tf-mix", "−1.05"); equal("second-best score", r(mx(s.filter((_, j) => j !== 3)), 2), "-1.05");
  equal("weight on users", r(tr.layers[0].heads[1].att[4][3], 3), "1.000"); claim("tf-mix", "1.000");
  claim("tf-mix", "weight 0.98");
}
{
  const proj = 3 * 8 * 16 * 16, score = 2 * 8 * 8 * 8, mix = 2 * 8 * 8 * 8, out = 8 * 16 * 16;
  equal("projections", int(proj), "6,144"); claim("tf-matrix-form", "6,144");
  equal("scores", int(score), "1,024"); claim("tf-matrix-form", "1,024");
  equal("W_O", int(out), "2,048"); claim("tf-matrix-form", "2,048");
  equal("total", int(proj + score + mix + out), "10,240"); claim("tf-matrix-form", "10,240");
  equal("masked pairs", 64 - 8 * 9 / 2, 28); claim("tf-matrix-form", "discards 28");
  claim("tf-matrix-form", "64 dot products of length 8");
}
{
  const A = tr.layers[0].heads[0].att;
  for (const [i, j, want] of [[4, 1, "0.98"], [2, 0, "0.60"], [3, 0, "0.61"], [2, 2, "0.39"], [3, 2, "0.35"]]) { equal(`A1[${i}][${j}]`, r(A[i][j], 2), want); claim("tf-att-matrix", want); }
  equal("row 7 min", r(mn(A[7]), 2), "0.08"); equal("row 7 max", r(mx(A[7]), 2), "0.16"); claim("tf-att-matrix", "between 0.08 and 0.16");
  ok("; always follows the digit", Q.every((q) => q.ids[8] === id(";")));
}
{
  equal("our ratio", r(2 * 8 * 16 / 3072, 3), "0.083"); claim("tf-att-cost", "0.083"); claim("tf-att-cost", "8%");
  equal("our layer params", 4 * 16 * 16 + 2 * 16 * 64, 3072); claim("tf-att-cost", "3,072");
  equal("Llama 3 8B layer weights", L8.layerW, 218103808); claim("tf-att-cost", int(L8.layerW));
  equal("ratio 8K", r(2 * 8192 * 4096 / L8.layerW, 2), "0.31"); claim("tf-att-cost", "0.31 at T = 8,192");
  equal("ratio 128K", r(2 * 131072 * 4096 / L8.layerW, 2), "4.92"); claim("tf-att-cost", "4.92 at T = 131,072");
  // the calculator at every option
  const spec = Deck.slides.find((s) => s.id === "tf-att-cost").fig;
  for (const m of [0, 1]) for (const T of [8, 1024, 8192, 32768, 131072]) {
    const v = { m, T }, d = [16, 4096][m], P = [3072, L8.layerW][m];
    near(`calc T² m${m} T${T}`, spec.outputs[0].f(v), 4 * T * T * d, 0);
    near(`calc W m${m} T${T}`, spec.outputs[1].f(v), 2 * T * P, 0);
    near(`calc ratio m${m} T${T}`, spec.outputs[2].f(v), 2 * T * d / P, 1e-12);
  }
}
{
  equal("8192^2", int(8192 * 8192), "67,108,864"); claim("tf-attn-check", "67,108,864");
  equal("sqrt 128", r(Math.sqrt(128), 1), "11.3"); claim("tf-attn-check", "11.3");
  equal("8192 x 128", int(8192 * 128), "1,048,576"); claim("tf-attn-check", "1,048,576");
  equal("unmasked entries at 8,192", int(8192 * 8193 / 2), "33,558,528"); claim("tf-attn-check", "T(T + 1)/2 = 33,558,528");
  equal("K, V width with 8 KV heads", 8 * L8.hd, 1024); claim("tf-attn-check", "K and V of 8,192 × 1,024"); claim("tf-attn-check", "Q of 8,192 × 4,096");
  ok("Llama 3 8B head size 128", L8.hd === 128);
}


// ------------------------------------------------------------------ 5.3 Multi-head attention
{
  const A = (l, hh) => tr.layers[l].heads[hh].att;
  claim("tf-heads-split", "0.98 on <code>name</code>".replace(/<[^>]+>/g, ""));
  claim("tf-heads-split", "1.00 on <code>users</code>".replace(/<[^>]+>/g, ""));
  equal("4096 / 32", L8.hd, 128); claim("tf-heads-split", "32 heads of 128");
  for (const [l, hh, i, j, want] of [[0, 1, 3, 2, "0.98"], [0, 1, 4, 3, "1.00"], [0, 1, 5, 4, "0.97"], [0, 0, 4, 1, "0.98"], [1, 1, 2, 1, "0.95"], [1, 1, 5, 1, "0.88"]]) {
    equal(`L${l + 1}H${hh + 1} ${i}->${j}`, r(A(l, hh)[i][j], 2), want); claim("tf-heads", want);
  }
  const oq = "SELECT total FROM orders WHERE status = 3;", oi = M.encode(oq).map(M.idOf), ot = M.forward(oi.slice(0, 8));
  ok("orders query tokens", oi.length === 9 && M.encode(oq)[1] === " total" && M.encode(oq)[3] === " orders");
  equal("orders: WHERE->total L1H1", r(ot.layers[0].heads[0].att[4][1], 2), "0.99");
  equal("orders: WHERE->orders L1H2", r(ot.layers[0].heads[1].att[4][3], 2), "1.00");
  equal("orders: FROM->total L2H2", r(ot.layers[1].heads[1].att[2][1], 2), "0.95");
  claim("tf-heads", "(layer 1, head 1, 0.99)"); claim("tf-heads", "(layer 1, head 2, 1.00)"); claim("tf-heads", "(layer 2, head 2, 0.95)");
  claim("tf-heads", "SELECT total FROM orders WHERE status = 3;");
  { const { el } = mount("tinylm", { widget: "heads" }); ok("heads widget offers the orders query", [...el.querySelectorAll("option")].some((o) => o.textContent === oq)); }
}
{
  const Wo = W["l0.wo"], cat = tr.layers[0].concat[4];
  const part = (hh) => { const o = new Array(16).fill(0); for (let rr = 0; rr < 8; rr++) for (let c = 0; c < 16; c++) o[c] += cat[8 * hh + rr] * Wo.data[(8 * hh + rr) * 16 + c]; return o; };
  const p1 = part(0), p2 = part(1), out = tr.layers[0].attnOut[4];
  ok("head terms sum to attnOut", out.every((v, c) => Math.abs(v - p1[c] - p2[c]) < 1e-12));
  claim("tf-concat-wo", r(norm(p1), 2)); equal("head1 norm", r(norm(p1), 2), "10.86");
  claim("tf-concat-wo", r(norm(p2), 2)); claim("tf-concat-wo", r(norm(out), 2));
  claim("tf-concat-wo", `${R(p1[0], 2)} + ${R(p2[0], 2)} = ${R(out[0], 2)}`);
  near("displayed entry-0 sum", +r(p1[0], 2) + +r(p2[0], 2), +r(out[0], 2), 1e-9);
}
{
  equal("4d^2", int(4 * 16 * 16), "1,024"); claim("tf-attn-params", "1,024"); claim("tf-attn-params", "2,048 for our two layers");
  equal("4 x 4096^2", int(4 * 4096 * 4096), "67,108,864"); claim("tf-attn-params", "67,108,864");
  equal("Llama 3 8B attention per layer", int(L8.attn), "41,943,040"); claim("tf-attn-params", "41,943,040");
  const spec = Deck.slides.find((x) => x.id === "tf-attn-params").fig;
  for (const d of [16, 4096]) for (const H of [1, 2, 4, 8, 16, 32]) {
    const v = { d, H }, bad = d % H !== 0;
    const o = spec.outputs.map((f) => f.f(v));
    ok(`attn-params calc d${d} H${H}`, bad ? o.every(Number.isNaN) && spec.note(v).includes("cannot be split") : o[0] === d / H && o[1] === d * d / H && o[2] === 4 * d * d);
  }
}
{
  // mean weights by selected column, recomputed
  const cols = ["id", "name", "email", "age", "user_id", "total", "status", "title", "price"];
  const acc = {}, wsum = {};
  for (const q of Q) {
    const t = M.forward(q.ids.slice(0, 8)), a = (l, hh, i, j) => t.layers[l].heads[hh].att[i][j];
    const v = [a(0, 0, 4, 1), a(0, 0, 4, 3), a(0, 1, 4, 1), a(0, 1, 4, 3), a(0, 1, 2, 1), a(1, 1, 2, 1)];
    acc[q.c1] = (acc[q.c1] || new Array(6).fill(0)).map((x, k) => x + q.p * v[k]); wsum[q.c1] = (wsum[q.c1] || 0) + q.p;
  }
  const m = Object.fromEntries(cols.map((c) => [c, acc[c].map((x) => x / wsum[c])]));
  const spec = Deck.slides.find((x) => x.id === "tf-head-roles").fig;
  ok("head-roles matrix matches", spec.matrix.every((row, i) => row.every((v, k) => Math.abs(v - m[cols[i]][k]) < 1e-12)));
  ok("L1H1 WHERE->col >= 0.98 except title, price", cols.filter((c) => c !== "title" && c !== "price").every((c) => +r(m[c][0], 2) >= 0.98));
  equal("title col", r(m.title[0], 2), "0.25"); equal("price col", r(m.price[0], 2), "0.03");
  equal("title table", r(m.title[1], 2), "0.74"); equal("price table", r(m.price[1], 2), "0.95");
  for (const x of ["0.25", "0.03", "0.74", "0.95"]) claim("tf-head-roles", x);
  ok("L1H2 WHERE->table >= 0.999 except price", cols.filter((c) => c !== "price").every((c) => m[c][3] >= 0.999)); claim("tf-head-roles", "0.999 or more");
  equal("price: L1H2 reads the column", r(m.price[2], 2), "1.00"); claim("tf-head-roles", "reads the column (1.00)"); claim("tf-head-roles", "six attention weights");
  const l1from = ["id", "title", "price", "user_id", "total", "status"];
  ok("L1H2 FROM->col >= 0.995 for six columns", l1from.every((c) => m[c][4] >= 0.995)); claim("tf-head-roles", "0.995 or more");
  ok("L1H2 FROM->col low for name, email, age", ["name", "email", "age"].every((c) => m[c][4] < 0.1));
  const l2from = ["name", "email", "age", "user_id", "total", "status"];
  ok("those six are the users/orders columns other than id", l2from.every((c) => (D.schema.users.includes(c) || D.schema.orders.includes(c)) && c !== "id") && [...new Set([...D.schema.users, ...D.schema.orders])].filter((c) => c !== "id").length === 6);
  equal("L2H2 FROM->col min", r(mn(l2from.map((c) => m[c][5])), 2), "0.91"); equal("L2H2 FROM->col max", r(mx(l2from.map((c) => m[c][5])), 2), "0.95"); claim("tf-head-roles", "(0.91 to 0.95)");
  ok("every column read at FROM by some head", cols.every((c) => Math.max(m[c][4], m[c][5]) > 0.9));
  claim("tf-head-roles", "with weight 0.95");
}

// ------------------------------------------------------------------ 5.4 The transformer block
{
  const n = [norm(tr.x0[4]), norm(tr.layers[0].mid[4]), norm(tr.layers[0].post[4]), norm(tr.layers[1].mid[4]), norm(tr.layers[1].post[4])];
  equal("stream norms", n.map((x) => r(x, 2)), ["5.06", "15.81", "22.84", "37.39", "61.97"]);
  for (const k of [0, 1, 2, 4]) claim("tf-residual", n[k].toFixed(1));
  ok("tf-residual does not quote the stream after layer 2's attention", !text("tf-residual").includes("37.39") && !text("tf-residual").includes("37.4"));
  const bw = mount("tinylm", { widget: "block" }).el;
  bw.querySelectorAll(".pos-btn")[4].click();
  const shown = [...bw.querySelectorAll(".vec-note")].map((e) => e.textContent);
  for (const k of [0, 1, 2, 4]) ok(`block widget shows ${n[k].toFixed(1)} at WHERE`, shown.includes(`size ${n[k].toFixed(1)}`));
  equal("Llama 3 70B layers", LLAMA["70B"].L, 80); claim("tf-residual", "Llama 3 70B stacks 80");
}
{
  const post = tr.layers[0].post, x = post[4], g = Array.from(W["l1.n1"].data);
  const lens = post.map(norm);
  equal("stream at WHERE entering layer 2", r(lens[4], 2), "22.84"); equal("at SELECT", r(lens[0], 2), "591.03");
  claim("tf-rmsnorm", "22.84"); claim("tf-rmsnorm", "591.03");
  equal("spread", Math.round(mx(lens) / mn(lens)), 26); ok("WHERE is the shortest", mn(lens) === lens[4]); claim("tf-rmsnorm", "26-fold");
  const ms = sum(x.map(sq)) / 16, rms = Math.sqrt(ms + 1e-5);
  equal("mean square", r(ms, 2), "32.60"); claim("tf-rmsnorm", "32.60"); equal("rms", r(rms, 3), "5.710"); claim("tf-rmsnorm", "5.710");
  const xh = x.map((v) => v / rms), out = tr.layers[1].h1[4];
  ok("h1 = x/rms * g", out.every((v, c) => Math.abs(v - xh[c] * g[c]) < 1e-12));
  for (const c of [0, 6, 7]) {
    claim("tf-rmsnorm", R(xh[c], 3)); claim("tf-rmsnorm", r(g[c], 3)); claim("tf-rmsnorm", R(out[c], 3));
    near(`displayed product entry ${c}`, +r(+r(xh[c], 3) * +r(g[c], 3), 3), +r(out[c], 3), 1e-9);
  }
  equal("gain 4", r(g[4], 3), "0.000"); claim("tf-rmsnorm", "gain 4 is 0.000");
  ok("x/RMS rows all 4.000", post.every((v) => r(norm(v.map((z) => z / Math.sqrt(sum(v.map(sq)) / 16 + 1e-5))), 3) === "4.000"));
  claim("tf-rmsnorm", "√16 = 4");
}
{
  equal("MLP share", Math.round(2048 / 3104 * 100), 66); claim("tf-mlp", "66%"); claim("tf-mlp", "2,048 of each layer's 3,104");
  const pre = tr.layers[0].pre[4], act = tr.layers[0].act[4];
  equal("range", [r(mn(pre), 2), r(mx(pre), 2)], ["-6.45", "3.53"]); claim("tf-mlp", "from −6.45 to 3.53");
  equal("positive", pre.filter((z) => z > 0).length, 5); claim("tf-mlp", "5 of 64 are positive");
  equal("> 0.1", act.filter((a) => a > 0.1).length, 3); claim("tf-mlp", "only 3 produce more than 0.1");
  equal("positive at id", tr.layers[0].pre[5].filter((z) => z > 0).length, 39); claim("tf-mlp", "39 are positive");
  ok("act = gelu(pre)", act.every((a, u) => Math.abs(a - 0.5 * pre[u] * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (pre[u] + 0.044715 * pre[u] ** 3)))) < 1e-12));
  // figure at every layer and position
  const { el } = mount("tf-mlp", { pos: 4, layer: 0 });
  for (const l of [0, 1]) for (let i = 0; i < 8; i++) {
    pressSeg(el, 0, l); pressSeg(el, 1, i);
    const p = tr.layers[l].pre[i], a = tr.layers[l].act[i];
    equal(`tf-mlp tiles L${l + 1} pos ${i}`, tiles(el), [String(p.filter((z) => z > 0).length), String(a.filter((v) => v > 0.1).length), `${R(mn(p), 2)} to ${R(mx(p), 2)}`, r(norm(tr.layers[l].mlpOut[i]), 2)]);
    const dots = el.querySelectorAll("circle.dot").length;
    ok(`tf-mlp draws 64 units L${l + 1} pos ${i}`, dots === 64);
  }
}
claim("tf-block-diagram", "3,104"); equal("block params", 2 * 16 + 4 * 256 + 2 * 1024, 3104);
{
  // independent ablation forward
  function abl(ids, on) {
    let x = ids.map((t, i) => Array.from({ length: 16 }, (_, c) => W.wte.data[t * 16 + c] + W.wpe.data[i * 16 + c]));
    const mm = (v, Mx) => { const [a, b] = Mx.shape, o = new Array(b).fill(0); for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) o[j] += v[i] * Mx.data[i * b + j]; return o; };
    const rn = (v, gg) => { const rr = 1 / Math.sqrt(sum(v.map(sq)) / v.length + 1e-5); return v.map((z, c) => z * rr * gg.data[c]); };
    const gelu = (z) => 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z ** 3)));
    for (let l = 0; l < 2; l++) {
      const P = (k) => W[`l${l}.${k}`];
      if (on[2 * l]) {
        const hn = x.map((v) => rn(v, P("n1"))), q = hn.map((v) => mm(v, P("wq"))), k = hn.map((v) => mm(v, P("wk"))), v = hn.map((u) => mm(u, P("wv")));
        const cat = x.map(() => new Array(16).fill(0));
        for (let hh = 0; hh < 2; hh++) for (let i = 0; i < x.length; i++) {
          const a = softmax(Array.from({ length: i + 1 }, (_, j) => dot(q[i].slice(8 * hh, 8 * hh + 8), k[j].slice(8 * hh, 8 * hh + 8)) / Math.sqrt(8)));
          for (let j = 0; j <= i; j++) for (let c = 0; c < 8; c++) cat[i][8 * hh + c] += a[j] * v[j][8 * hh + c];
        }
        x = x.map((u, i) => { const o = mm(cat[i], P("wo")); return u.map((z, c) => z + o[c]); });
      }
      if (on[2 * l + 1]) x = x.map((u) => { const o = mm(mm(rn(u, P("n2")), P("w1")).map(gelu), P("w2")); return u.map((z, c) => z + o[c]); });
    }
    return x.map((u) => softmax(mm(rn(u, W.nf), W.wu)));
  }
  const evalAbl = (on) => { const per = new Array(8).fill(0); for (const q of Q) { const p = abl(q.ids.slice(0, 8), on); for (let i = 0; i < 8; i++) per[i] += -Math.log(p[i][q.ids[i + 1]]) * q.p; } return { per, mean: sum(per) / 8, ours: abl(IDS, on) }; };
  const full = evalAbl([1, 1, 1, 1]);
  near("ablation all-on equals exact loss", full.mean, M.exactLoss().mean, 1e-9);
  const no2 = evalAbl([1, 1, 0, 0]), noA2 = evalAbl([1, 1, 0, 1]);
  equal("loss without layer 2", r(no2.mean, 4), "3.3274"); claim("tf-stack", "0.6942 to 3.3274");
  const pOurs = (res, i) => res.ours[i][ALL[i + 1]];
  equal("p(FROM) after layer 1", r(pOurs(no2, 1), 3), "1.000"); equal("p(WHERE) after layer 1", r(pOurs(no2, 3), 3), "1.000"); equal("p(=) after layer 1", r(pOurs(no2, 5), 3), "0.990");
  claim("tf-stack", "(p = 1.000, 1.000 and 0.990)");
  equal("p(users) after layer 1", r(pOurs(no2, 2), 3), "0.051"); claim("tf-stack", "only 0.051");
  equal("loss without layer 2 attention", r(noA2.mean, 4), "0.7384"); claim("tf-stack", "(0.7384)");
  ok("almost all of it at WHERE", (noA2.per[4] - full.per[4]) / sum(noA2.per.map((v, i) => v - full.per[i])) > 0.95);
  // the figure, through all 16 switch settings
  const { el } = mount("tf-ablate", {});
  const btns = [...el.querySelectorAll(".tf-tog")];
  let state = [1, 1, 1, 1];
  for (let code = 15; code >= 0; code--) {
    const want = [8, 4, 2, 1].map((b) => (code & b ? 1 : 0));
    want.forEach((w, k) => { if (w !== state[k]) { btns[k].click(); state[k] = w; } });
    const res = evalAbl(want);
    equal(`tf-ablate tiles ${want.join("")}`, tiles(el).slice(0, 2), [r(res.mean, 4), r(full.mean, 4)]);
    const shown = [...el.querySelectorAll("table.tf-abl tr")].slice(1).map((row) => row.querySelectorAll(".tf-abl-p .num")[1].textContent);
    equal(`tf-ablate p ${want.join("")}`, shown, res.ours.map((p, i) => { const v = p[ALL[i + 1]]; return v >= 0.9995 ? "1.000" : v.toFixed(3); }));
  }
}
{
  const z = tr.logits[4];
  for (const [t, want] of [[" id", "7.621"], [" email", "7.617"], [" age", "7.502"], [" name", "1.994"]]) { equal(`logit ${t}`, r(z[id(t)], 3), want); claim("tf-logits", want); }
  const top3 = [z[id(" id")], z[id(" email")], z[id(" age")]];
  ok("top 3 within 0.12", mx(top3) - mn(top3) < 0.12); claim("tf-logits", "within 0.12");
  equal("name below top", r(z[id(" id")] - z[id(" name")], 2), "5.63"); claim("tf-logits", "5.63 below the top");
  ok("these are the top 4 logits", [...z].sort((a, b) => b - a).slice(0, 4).every((v, k) => v === [z[id(" id")], z[id(" email")], z[id(" age")], z[id(" name")]][k]));
  for (const want of ["0.346", "0.345", "0.307", "0.001"]) claim("tf-logits", want);
  equal("W_U shape", W.wu.shape.join("x"), "16x113");
}
{
  const n = Object.fromEntries(C.shapes.map(([k, sh]) => [k, sh.reduce((a, b) => a * b, 1)]));
  equal("sum of shapes", sum(Object.values(n)), 9968);
  const L = 2, d = 16, V = 113, T = 8;
  equal("12Ld^2", 12 * L * d * d, 6144); equal("2Vd", 2 * V * d, 3616); equal("Td", T * d, 128); equal("(2L+1)d", (2 * L + 1) * d, 80);
  equal("formula total", 12 * L * d * d + 2 * V * d + T * d + (2 * L + 1) * d, C.params);
  for (const x of ["6{,}144 + 3{,}616 + 128 + 80", "&= 9{,}968", "1,808", "128", "3,104", "6,208"]) claim("tf-params", x);
  equal("per layer", n["l0.n1"] + n["l0.wq"] + n["l0.wk"] + n["l0.wv"] + n["l0.wo"] + n["l0.n2"] + n["l0.w1"] + n["l0.w2"], 3104);
  equal("vocab share", Math.round(3616 / 9968 * 100), 36); claim("tf-params", "36%");
  equal("Llama 3 8B total", L8.total, 8030261248);
  equal("Llama 3 8B vocab share", Math.round(2 * L8.emb / L8.total * 100), 13); claim("tf-params", "13%");
}
{
  const perTok = 2 * (2 * 3072 + 1808);
  equal("per token", int(perTok), "15,904"); claim("tf-flops", "2 × (2 × 3,072 + 1,808) = 15,904");
  const attn = sum(Array.from({ length: 8 }, (_, t) => 4 * 16 * (t + 1)));
  equal("attention per layer", int(attn), "2,304"); claim("tf-flops", "2,304 per layer"); claim("tf-flops", "4,608");
  equal("forward", int(8 * perTok + 2 * attn), "131,840"); claim("tf-flops", "8 × 15,904 + 4,608 = 131,840");
  const comp = [2 * 8 * 2 * 1024, 2 * 8 * 2 * 2048, 2 * 8 * 1808, 2 * attn];
  equal("components", comp, [32768, 65536, 28928, 4608]); equal("components sum", sum(comp), 131840);
  const spec = Deck.slides.find((x) => x.id === "tf-flops").fig;
  equal("bars values", spec.items.map((i) => i.value), comp);
  const w = 2 * (L8.total - L8.emb);
  equal("Llama 3 8B weight FLOPs", w.toExponential(2), "1.50e+10"); claim("tf-flops", String.raw`\(1.50 \times 10^{10}\)`);
  equal("Kaplan 2N (both vocabulary matrices left out)", (2 * (L8.total - 2 * L8.emb)).toExponential(2), "1.40e+10"); claim("tf-flops", String.raw`\(1.40 \times 10^{10}\)`);
  claim("tf-flops", "Their N leaves out both vocabulary matrices"); claim("tf-flops", "twice the 32,768 of the Q, K, V, O projections");
  equal("with embedding", (2 * L8.total).toExponential(1), "1.6e+10");
  equal("6N", (6 * L8.total).toExponential(1), "4.8e+10");
  equal("forgets the add", L8.total.toExponential(1), "8.0e+9");
  claim("tf-flops", "8,030,261,248"); claim("tf-flops", "525,336,576"); equal("Llama 3 8B embedding", L8.emb, 525336576);
}


// ------------------------------------------------------------------ 5.5 Positions
{
  // independent layer-1 attention output at the last position
  function last(order, usePos) {
    const mm = (v, Mx) => { const [a, b] = Mx.shape, o = new Array(b).fill(0); for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) o[j] += v[i] * Mx.data[i * b + j]; return o; };
    const rn = (v, gg) => { const rr = 1 / Math.sqrt(sum(v.map(sq)) / v.length + 1e-5); return v.map((z, c) => z * rr * gg.data[c]); };
    const P = (k) => W[`l0.${k}`];
    const X = order.map((i, t) => Array.from({ length: 16 }, (_, c) => W.wte.data[IDS[i] * 16 + c] + (usePos ? W.wpe.data[t * 16 + c] : 0)));
    const hn = X.map((v) => rn(v, P("n1"))), q = mm(hn[7], P("wq")), k = hn.map((v) => mm(v, P("wk"))), v = hn.map((u) => mm(u, P("wv")));
    const cat = new Array(16).fill(0);
    for (let hh = 0; hh < 2; hh++) { const a = softmax(k.map((kv) => dot(q.slice(8 * hh, 8 * hh + 8), kv.slice(8 * hh, 8 * hh + 8)) / Math.sqrt(8))); a.forEach((wgt, j) => { for (let c = 0; c < 8; c++) cat[8 * hh + c] += wgt * v[j][8 * hh + c]; }); }
    return mm(cat, P("wo"));
  }
  const base = [0, 1, 2, 3, 4, 5, 6, 7], b0 = last(base, false), b1 = last(base, true);
  const rnd = M.mulberry32(12345);
  let worst = 0;
  for (let n = 0; n < 1000; n++) {
    const a = base.slice(0, 7); for (let i = 6; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    const o = last(a.concat([7]), false); worst = Math.max(worst, ...o.map((x, c) => Math.abs(x - b0[c])));
  }
  ok(`1,000 shuffles without positions change the output by < 1e-15 (worst ${worst})`, worst < 1e-15); claim("tf-perm", "less than 10⁻¹⁵");
  // the plate's first shuffle (seed 3)
  const rand = M.mulberry32(3), a = base.slice(0, 7);
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  const first = a.concat([7]), o1 = last(first, true), d1 = mx(o1.map((x, c) => Math.abs(x - b1[c])));
  equal("first shuffle, with positions", r(d1, 2), "1.94"); claim("tf-perm", "changes an output entry by 1.94");
  const { el } = mount("tf-perm", { seed: 3 });
  const bt = el.querySelectorAll("button.btn");
  bt[0].click();
  equal("tf-perm order after one shuffle", tiles(el)[1], first.join(" "));
  ok(`tf-perm no-position change tiny (${tiles(el)[0]})`, tiles(el)[0] === "0" || /× 10⁻¹[⁵⁶⁷⁸⁹]$/.test(tiles(el)[0]));
  pressSeg(el, 0, 1);
  equal("tf-perm with positions", tiles(el)[0], r(d1, 3));
  bt[1].click(); equal("tf-perm original order", tiles(el)[0], "0");
}
{
  const P = Array.from({ length: 8 }, (_, t) => M.row(W.wpe, t)), cos = (a, b) => dot(a, b) / (norm(a) * norm(b));
  equal("cos rows 4,5", r(cos(P[4], P[5]), 2), "-0.34"); claim("tf-learned-pos", "−0.34");
  equal("cos rows 0,4", r(cos(P[0], P[4]), 2), "0.35"); claim("tf-learned-pos", "0.35");
  equal("W_P rows", W.wpe.shape[0], 8); claim("tf-learned-pos", "no row 8");
  claim("tf-learned-pos", "512-token sequences"); claim("tf-learned-pos", "1,024");
}
{
  const om = (i) => Math.pow(10000, (-2 * i) / 16);
  equal("wavelength pair 0", r(2 * Math.PI / om(0), 2), "6.28"); claim("tf-sinusoid", "6.28 positions");
  equal("wavelength pair 7", int(2 * Math.PI / om(7)), "19,869"); claim("tf-sinusoid", "19,869 positions");
  // the offset is a rotation independent of t
  const k = 3, i = 2, t = 5;
  near("PE(t+k) = R(kω) PE(t)", Math.sin((t + k) * om(i)), Math.sin(t * om(i)) * Math.cos(k * om(i)) + Math.cos(t * om(i)) * Math.sin(k * om(i)), 1e-12);
  const spec = Deck.slides.find((x) => x.id === "tf-sinusoid").fig;
  for (let ii = 0; ii <= 7; ii++) { near(`sinusoid omega ${ii}`, spec.readouts[0].f({ i: ii }), om(ii), 1e-15); near(`sinusoid wavelength ${ii}`, spec.readouts[1].f({ i: ii }), 2 * Math.PI / om(ii), 1e-9); }
}
{
  const th63 = Math.pow(500000, -126 / 128);
  equal("theta_63", th63.toExponential(2), "2.46e-6"); claim("tf-rope", "2.46 × 10⁻⁶");
  equal("one turn", r(2 * Math.PI / th63 / 1e6, 2), "2.56"); claim("tf-rope", "2.56 million positions");
  // Meta's apply_scaling (llama-models, models/llama3/model.py): factor 8, low 1, high 4, original context 8,192
  const scale = (f) => { const wl = 2 * Math.PI / f, lowW = 8192, highW = 8192 / 4; if (wl < highW) return f; if (wl > lowW) return f / 8; const sm = (8192 / wl - 1) / 3; return (1 - sm) * f / 8 + sm * f; };
  const fr = Array.from({ length: 64 }, (_, i) => Math.pow(500000, -2 * i / 128)), sc = fr.map(scale);
  const same = sc.map((f, i) => f === fr[i]), div8 = sc.map((f, i) => Math.abs(f * 8 / fr[i] - 1) < 1e-12);
  equal("unchanged pairs", [same.indexOf(false), same.lastIndexOf(true)], [29, 28]);
  equal("divided by 8", [div8.indexOf(true), div8.filter(Boolean).length], [35, 29]);
  ok("pairs 29 to 34 slowed by less than 8", [29, 30, 31, 32, 33, 34].every((i) => sc[i] < fr[i] && sc[i] > fr[i] / 8));
  claim("tf-rope", "pairs 35 to 63 by a factor of 8 and pairs 29 to 34 by less");
  equal("scaled pair 63", sc[63].toExponential(2), "3.07e-7"); claim("tf-rope", "3.07 × 10⁻⁷ radians per position");
  equal("one turn, scaled", r(2 * Math.PI / sc[63] / 1e6, 1), "20.5"); claim("tf-rope", "one turn every 20.5 million");
  const turns = (fs) => fs.filter((f) => 2 * Math.PI / f <= 131072).length;
  equal("pairs completing a turn within 128K: base 10,000", turns(Array.from({ length: 64 }, (_, i) => Math.pow(10000, -2 * i / 128))), 64);
  equal("pairs completing a turn within 128K: base 500,000", turns(fr), 49);
  equal("pairs completing a turn within 128K: Llama 3.1", turns(sc), 39);
  claim("tf-long", "all 64 pairs of a 128-number head turn a full circle within 128K positions"); claim("tf-long", "only 49 do"); claim("tf-long", "rescaling 39");
  equal("theta_0", Math.pow(500000, 0), 1); claim("tf-rope", "Pair 0 turns 1 radian");
  ok("2.56 million > 131,072", 2 * Math.PI / th63 > 131072);
  const rot = (v, a) => [Math.cos(a) * v[0] - Math.sin(a) * v[1], Math.sin(a) * v[0] + Math.cos(a) * v[1]];
  const q = [0.6, 0.8], k = [1, 0];
  near("relative-position identity", dot(rot(q, 3 * 0.5), rot(k, 1 * 0.5)), dot(q, rot(k, (1 - 3) * 0.5)), 1e-12);
  // figure: default, shift, extremes
  const { el } = mount("tf-rope", { q, k, theta: 0.5, m: 3, n: 1 });
  const score = (m, n, th) => r(dot(rot(q, m * th), rot(k, n * th)), 3);
  equal("tf-rope default score", tiles(el)[2], minus(score(3, 1, 0.5)));
  const shiftBtn = [...el.querySelectorAll("button.btn")].find((b) => b.textContent.includes("Shift"));
  shiftBtn.click(); equal("tf-rope after shift", tiles(el)[2], minus(score(4, 2, 0.5)));
  equal("shift keeps the score", score(4, 2, 0.5), score(3, 1, 0.5));
  setRange(el, 0, 12); setRange(el, 1, 0); setRange(el, 2, 1.5);
  equal("tf-rope extreme score", tiles(el)[2], minus(score(12, 0, 1.5)));
  equal("tf-rope extreme angle", tiles(el)[3], r(12 * 1.5, 2));
  ok("tf-rope shift disabled at 12", shiftBtn.disabled);
  setRange(el, 0, 0); setRange(el, 1, 12); setRange(el, 2, 0.05);
  equal("tf-rope other extreme", tiles(el)[2], minus(score(0, 12, 0.05)));
}
{
  const rot = (v, a) => [Math.cos(a) * v[0] - Math.sin(a) * v[1], Math.sin(a) * v[0] + Math.cos(a) * v[1]];
  const q = [0.6, 0.8], k = [1, 0];
  equal("cos 1.5", r(Math.cos(1.5), 4), "0.0707"); equal("sin 1.5", r(Math.sin(1.5), 4), "0.9975");
  const rq = rot(q, 1.5), rk = rot(k, 0.5);
  equal("R(1.5)q", [r(rq[0], 4), r(rq[1], 4)], ["-0.7556", "0.6551"]);
  equal("R(0.5)k", [r(rk[0], 4), r(rk[1], 4)], ["0.8776", "0.4794"]);
  equal("displayed product", r(-0.7556 * 0.8776 + 0.6551 * 0.4794, 3), "-0.349");
  equal("exact score", r(dot(rq, rk), 3), "-0.349");
  for (const x of ["x = 0.0707 \\cdot 0.6 - 0.9975 \\cdot 0.8\\) \\(= -0.7556", "y = 0.9975 \\cdot 0.6 + 0.0707 \\cdot 0.8\\) \\(= 0.6551", "(0.8776,\\ 0.4794)", "-0.7556 \\cdot 0.8776 + 0.6551 \\cdot 0.4794\\) \\(= -0.349"]) claim("tf-rope-worked", x);
  near("x of R(1.5)q from displayed factors", +r(0.0707 * 0.6 - 0.9975 * 0.8, 4), -0.7556, 1e-9); near("y of R(1.5)q from displayed factors", +r(0.9975 * 0.6 + 0.0707 * 0.8, 4), 0.6551, 1e-9);
  equal("angle of q", r(Math.atan2(0.8, 0.6), 3), "0.927"); claim("tf-rope-worked", "cos(0.927 + 1.5 − 0.5) = cos 1.927 = −0.349");
  equal("cos 1.927", r(Math.cos(Math.atan2(0.8, 0.6) + 1), 3), "-0.349");
  const rq2 = rot(q, 2.5), rk2 = rot(k, 1.5);
  equal("R(2.5)q", [r(rq2[0], 4), r(rq2[1], 4)], ["-0.9595", "-0.2818"]); equal("R(1.5)k", [r(rk2[0], 4), r(rk2[1], 4)], ["0.0707", "0.9975"]);
  equal("score at (5,3)", r(dot(rq2, rk2), 3), "-0.349");
  claim("tf-rope-worked", "(-0.9595,\\ -0.2818)"); claim("tf-rope-worked", "(0.0707,\\ 0.9975)");
  near("unrotated", dot(q, k), 0.6, 1e-12); claim("tf-rope-worked", "q · k = 0.6");
}


// ------------------------------------------------------------------ 5.6 Training the transformer
{
  const ec = D.exactCurve, at = (st) => ec.find((x) => x[0] === st)[1];
  for (const [st, want] of [[0, "4.7217"], [50, "2.3086"], [100, "0.8193"], [200, "0.7179"], [6000, "0.6942"]]) { equal(`exact loss at ${st}`, r(at(st), 4), want); claim("tf-train", want); }
  equal("ln 113", r(Math.log(113), 4), "4.7274"); claim("tf-train", "ln 113 = 4.7274");
  equal("snapshots", M.snapSteps.length, 13); claim("tf-train", "13 saved snapshots");
  equal("floor", r(sum(D.floor) / 8, 4), "0.6937"); claim("tf-train", "0.6937");
  equal("recipe", [C.betas, C.lr, C.warmup, C.batch, C.steps], [[0.9, 0.95], 0.01, 100, 128, 6000]);
  claim("tf-train", "β = (0.9, 0.95)"); claim("tf-train", "100 warmup steps"); claim("tf-train", "128 freshly sampled");
  near("final exact loss from the weights", M.exactLoss().mean, 0.6942, 5e-5);
  // the floor from the grammar, exactly
  const H = (ps) => -sum(ps.filter((p) => p > 0).map((p) => p * Math.log(p)));
  const tabs = Object.keys(D.schema), cols = [...new Set(Object.values(D.schema).flat())];
  const pc1 = cols.map((c) => sum(tabs.map((t) => (D.schema[t].includes(c) ? 1 / 3 / D.schema[t].length : 0))));
  const hTable = sum(cols.map((c, ci) => { const w = tabs.map((t) => (D.schema[t].includes(c) ? 1 / 3 / D.schema[t].length : 0)); return pc1[ci] * H(w.map((x) => x / pc1[ci])); }));
  const hC2 = sum(tabs.map((t) => (1 / 3) * Math.log(D.schema[t].length - 1)));
  const floor = [H(pc1), 0, hTable, 0, hC2, 0, Math.log(9), 0];
  ok("data floor matches the grammar", floor.every((v, i) => Math.abs(v - D.floor[i]) < 1e-5));
}
{
  const ec = D.exactCurve.filter((x) => x[0] >= 10);
  const firstBelow = (i, thr) => ec.find((x) => x[2][i] - D.floor[i] < thr)[0];
  const staysFrom = (i, thr) => { let k = ec.length - 1; while (k >= 0 && ec[k][2][i] - D.floor[i] < thr) k--; return ec[k + 1][0]; };
  const fixedFree = [0, 1, 3, 5, 6, 7];
  const f = fixedFree.map((i) => firstBelow(i, 0.05)), st = fixedFree.map((i) => staysFrom(i, 0.05));
  equal("fixed and free positions: first within 0.05", [mn(f), mx(f)], [90, 100]); equal("and stay", st, f);
  claim("tf-learn-order", "by steps 90 to 100 and stay there");
  equal("table first/stay", [firstBelow(2, 0.05), staysFrom(2, 0.05)], [140, 140]); claim("tf-learn-order", "follows at step 140");
  equal("filter column first", firstBelow(4, 0.05), 340); equal("filter column stays", staysFrom(4, 0.05), 560);
  claim("tf-learn-order", "at step 340"); claim("tf-learn-order", "from step 560");
  const gap = M.exactLoss().perPos.map((v, i) => v - D.floor[i]);
  ok("WHERE has the largest final gap", gap.every((g, i) => i === 4 || g < gap[4]));
}
{
  const p = tr.probs[4][id(" id")], loss = -Math.log(p);
  equal("p(id)", r(p, 3), "0.346"); equal("loss", r(loss, 3), "1.061"); equal("p - 1", r(p - 1, 3), "-0.654");
  claim("tf-backprop", "−ln 0.346 = 1.061"); claim("tf-backprop", "p − 1 = −0.654");
  equal("column of id", id(" id"), 58); claim("tf-backprop", "column 58");
  claim("tf-backprop", "9,968");
}
{
  const per = M.exactLoss().perPos, fl = D.floor, gap = per.map((v, i) => v - fl[i]);
  equal("mean gap", r(sum(gap) / 8, 4), "0.0005"); claim("tf-floor", "0.0005 nats per token");
  equal("relative", r(sum(gap) / sum(fl) * 100, 2), "0.07"); claim("tf-floor", "0.07%");
  ok("fixed tokens < 1e-7", [1, 3, 5, 7].every((i) => per[i] < 1e-7)); claim("tf-floor", "less than 10⁻⁷");
  equal("gap at WHERE", r(gap[4], 4), "0.0026"); equal("summed gap", r(sum(gap), 4), "0.0039"); claim("tf-floor", "0.0026 of the 0.0039");
  claim("tf-floor", "0.6942"); claim("tf-floor", "0.6937");
  const spec = Deck.slides.find((x) => x.id === "tf-floor").fig;
  ok("floor table rows", spec.rows.length === 9 && spec.rows[4][4] === gap[4].toFixed(5));
}
{
  const held = new Set(D.holdout.queries);
  equal("held-out count", held.size, 27); claim("tf-heldout", "27, ours included"); claim("tf-heldout", "243 queries"); equal("seen count", 270 - 27, 243);
  ok("our query is held out", held.has(D.running));
  const prefix = "SELECT name FROM users WHERE id = ";
  equal("held-out queries with our prefix", [...held].filter((q) => q.startsWith(prefix)).map((q) => q.slice(prefix.length)), ["7;"]);
  const B = M.forward(IDS, "holdout"), pb = (i) => B.probs[i][ALL[i + 1]];
  equal("holdout p FROM, users, WHERE", [r(pb(1), 3), r(pb(2), 3), r(pb(3), 3)], ["1.000", "1.000", "1.000"]); claim("tf-heldout", "p = 1.000");
  equal("holdout p(id)", r(pb(4), 3), "0.348"); claim("tf-heldout", "0.348");
  equal("holdout p(7)", r(pb(6), 3), "0.003"); claim("tf-heldout", "0.003");
  const acc = { held: new Array(8).fill(0), seen: new Array(8).fill(0) }, w = { held: 0, seen: 0 };
  for (const q of Q) { const k = held.has(q.text) ? "held" : "seen", t = M.forward(q.ids.slice(0, 8), "holdout"); w[k] += q.p; for (let i = 0; i < 8; i++) acc[k][i] += -Math.log(t.probs[i][q.ids[i + 1]]) * q.p; }
  const seen6 = acc.seen[6] / w.seen, held6 = acc.held[6] / w.held;
  equal("seen loss after =", r(seen6, 2), "2.10"); equal("held loss after =", r(held6, 2), "6.76");
  for (const id_ of ["tf-heldout", "tf-train-check"]) { claim(id_, "2.10"); claim(id_, "6.76"); }
  equal("ln 9", r(Math.log(9), 3), "2.197"); claim("tf-heldout", "ln 9 = 2.197");
  ok("seen loss below ln 9", seen6 < Math.log(9));
  near("holdout mean losses (data)", sum(acc.held) / 8 / w.held, D.holdout.lossB.held, 1e-4);
  const copies = Object.fromEntries(Q.map((q) => [q.t, C.steps * C.batch * q.p]));
  equal("expected copies of a users or orders query", [int(copies.users), int(copies.orders)], ["2,370", "2,370"]); equal("expected copies of a products query", int(copies.products), "4,741");
  equal("our query is a users query", Q.find((q) => q.text === D.running).t, "users");
  claim("tf-heldout", "2,370 times"); claim("tf-heldout", "4,741 times");
}
{
  equal("d ratio", 16384 / 16, 1024); claim("tf-real", "1,024-fold"); equal("L ratio", 126 / 2, 63); claim("tf-real", "63-fold");
  equal("12 L d^2", (12 * 126 * 16384 * 16384).toExponential(2), "4.06e+11"); claim("tf-real", "4.06 × 10¹¹");
  ok("about 40 million-fold", Math.round(L405.total / C.params / 1e6) === 41 && Math.round(405e9 / C.params / 1e6) === 41); claim("tf-real", "about 40 million-fold");
  ok("12Ld² within 0.01% of the 405B count", Math.abs(12 * 126 * 16384 ** 2 / L405.total - 1) < 1e-4); claim("tf-real", "within 0.01%");
  equal("405B blocks in units of L·d²", (2 * 16384 * 16384 + 2 * 16384 * 1024 + 3 * 16384 * 53248) / 16384 ** 2, 11.875); claim("tf-real", "11.875·L·d²");
  ok("vocabulary tables nearly make up the difference", Math.abs((12 - 11.875) * 126 * 16384 ** 2 - 2 * VOCAB * 16384) / ((12 - 11.875) * 126 * 16384 ** 2) < 0.01);
  equal("our 12Ld²", 12 * 2 * 16 * 16, 6144); claim("tf-real", "only 6,144 of 9,968");
  equal("token ratio", r(15.6e12 / (C.steps * C.batch * C.T) / 1e6, 1), "2.5"); claim("tf-real", "2.5 million-fold");
  equal("our tokens", r(C.steps * C.batch * C.T / 1e6, 1), "6.1"); claim("tf-real", "6.1 million");
}
claim("tf-train-check", "4.7217"); claim("tf-train-check", "0.6942"); claim("tf-train-check", "0.6937");

// ------------------------------------------------------------------ 5.7 Modern variants
{
  equal("SwiGLU params, Llama 3 8B", int(3 * 4096 * 14336), "176,160,768"); claim("tf-swiglu", "176,160,768");
  equal("3.5 d", 14336 / 4096, 3.5); claim("tf-swiglu", "14,336 = 3.5d");
  equal("share of layer weights", Math.round(L8.mlp / L8.layerW * 100), 81); claim("tf-swiglu", "81%"); claim("tf-swiglu", "218,103,808");
  equal("Shazeer 2/3", 3072 * 2 / 3, 2048); claim("tf-swiglu", "2,048 instead of 3,072 at d = 768");
  equal("equal params", 3 * 768 * 2048, 2 * 768 * 3072);
  const spec = Deck.slides.find((x) => x.id === "tf-swiglu").fig;
  const silu = (a) => a / (1 + Math.exp(-a));
  for (const b of [-2, -1, 0, 0.5, 2]) { near(`swiglu unit b=${b}`, spec.fns[0].f(1.3, { b }), silu(1.3) * b, 1e-12); near(`readout b=${b}`, spec.readouts[1].f({ b }), silu(2) * b, 1e-12); }
}
{
  equal("70B K/V width", 8 * 128, 1024); claim("tf-gqa", "8,192 × 1,024");
  equal("KV per token GQA", int(2 * 80 * 8 * 128 * 2), "327,680"); claim("tf-gqa", "327,680 bytes");
  equal("KV per token MHA", int(2 * 80 * 64 * 128 * 2), "2,621,440"); claim("tf-gqa", "2,621,440");
  equal("query heads per KV head", 64 / 8, 8); claim("tf-gqa", "serves 8 query heads");
  ok("Llama 3 70B config", LLAMA["70B"].H === 64 && LLAMA["70B"].KV === 8 && LLAMA["70B"].L === 80 && llamaCount(LLAMA["70B"]).hd === 128);
  const { el } = mount("tf-gqa", { kv: 8, ctx: 131072 });
  for (const G of [64, 32, 16, 8, 4, 2, 1]) for (const ctx of [8192, 32768, 131072]) {
    pressSeg(el, 0, G); pressSeg(el, 1, ctx);
    const per = 2 * 80 * G * 128 * 2;
    equal(`tf-gqa G${G} ctx${ctx}`, tiles(el), [String(64 / G), int(2 * 8192 * G * 128), `${int(per)} B`, `${(per * ctx / 1e9).toFixed(1)} GB`]);
    ok(`tf-gqa draws ${G} KV heads`, el.querySelectorAll("rect.tf-gqa-kv").length === G && el.querySelectorAll("rect.tf-gqa-q").length === 64);
  }
}
{
  equal("router", int(4096 * 8), "32,768"); claim("tf-moe-params", "router 32,768"); claim("tf-moe-params", "8 × 176,160,768");
  claim("tf-moe", "E = 8 and k = 2 in each of its 32 layers"); claim("tf-moe", "factor × T·k/E"); claim("tf-moe", "never drop any");
  // Mixtral 8x7B from the paper's Table 1: dim 4096, 32 layers, head_dim 128, hidden 14336, 32 heads, 8 KV heads, vocab 32000, 8 experts, top-2
  const mx_ = { d: 4096, L: 32, hd: 128, ff: 14336, H: 32, KV: 8, V: 32000, E: 8, k: 2 };
  const attn = 2 * mx_.d * mx_.H * mx_.hd + 2 * mx_.d * mx_.KV * mx_.hd, expert = 3 * mx_.d * mx_.ff, router = mx_.d * mx_.E, norms = 2 * mx_.d;
  equal("Mixtral attention per layer", attn, 41943040); equal("Mixtral expert", expert, 176160768);
  const vocab = 2 * mx_.V * mx_.d;
  equal("Mixtral vocab tables", vocab, 262144000);
  const total = mx_.L * (attn + mx_.E * expert + router + norms) + vocab + mx_.d;
  const active = mx_.L * (attn + mx_.k * expert + router + norms) + vocab + mx_.d;
  equal("Mixtral total", int(total), "46,702,792,704"); equal("Mixtral active", int(active), "12,879,925,248");
  for (const x of ["41,943,040", "8 × 176,160,768", "32,768", "8,192", "262,144,000", "4,096", "46,702,792,704", "12,879,925,248"]) claim("tf-moe-params", x);
  equal("46.7B", r(total / 1e9, 1), "46.7"); equal("12.9B", r(active / 1e9, 1), "12.9"); claim("tf-moe-params", "46.7B"); claim("tf-moe-params", "12.9B");
  equal("paper rounding", [Math.round(total / 1e9), Math.round(active / 1e9)], [47, 13]); claim("tf-moe-params", "47B and 13B");
  claim("tf-moe-params", "671B total and 37B active"); claim("tf-moe-params", "256 routed experts plus 1 shared expert");
  const spec = Deck.slides.find((x) => x.id === "tf-moe-params").fig;
  for (const E of [2, 4, 8, 16, 32, 64]) for (const k of [1, 2, 4]) {
    const v = { E, k }, kk = Math.min(k, E);
    const T_ = mx_.L * (attn + E * expert + mx_.d * E + norms) + vocab + mx_.d, A_ = mx_.L * (attn + kk * expert + mx_.d * E + norms) + vocab + mx_.d;
    near(`moe calc total E${E} k${k}`, spec.outputs[0].f(v), T_, 0); near(`moe calc active E${E} k${k}`, spec.outputs[1].f(v), A_, 0); near(`moe calc share E${E} k${k}`, spec.outputs[2].f(v), A_ / T_, 1e-12);
    ok(`moe note E${E} k${k}`, spec.note(v).includes(`Experts per token: ${kk}`) && (k > E) === spec.note(v).includes("cannot exceed"));
  }
}
{
  equal("Mistral span", int(32 * 4096), "131,072"); claim("tf-long", "W = 4,096 in all 32 layers"); claim("tf-long", "about 131K tokens");
  // the figure: pairs and reach at the extremes
  const { el } = mount("tf-swa", { T: 16, W: 4, L: 2 });
  for (const [T, Wn, Ly] of [[16, 4, 2], [4, 1, 1], [32, 32, 6], [32, 1, 6], [5, 32, 1], [32, 3, 6], [17, 6, 3]]) {
    setRange(el, 0, T); setRange(el, 1, Wn); setRange(el, 2, Ly);
    const w = Math.min(Wn, T);
    let causal = 0, win = 0;
    for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) { causal++; if (j > i - w) win++; }
    const reach = Math.min(T, Ly * (w - 1) + 1);
    equal(`tf-swa T${T} W${Wn} L${Ly}`, tiles(el), [int(causal), int(win), `${T} vs ${w}`, `${reach} of ${T}`]);
    // brute-force reach: positions the last token depends on through Ly windowed layers
    let dep = new Set([T - 1]);
    for (let l = 0; l < Ly; l++) { const nx = new Set(); for (const i of dep) for (let j = Math.max(0, i - w + 1); j <= i; j++) nx.add(j); dep = nx; }
    equal(`tf-swa brute-force reach T${T} W${Wn} L${Ly}`, dep.size, reach);
    equal(`tf-swa strip T${T} W${Wn} L${Ly}`, el.querySelectorAll(".tf-swa-strip span.on").length, reach);
  }
}
{
  equal("tie saving", r(1808 / 9968 * 100, 1), "18.1"); claim("tf-tying", "1,808 of its 9,968 parameters, 18.1%");
  equal("8B share", r(L8.emb / L8.total * 100, 1), "6.5"); equal("70B share", r(L70.emb / L70.total * 100, 1), "1.5"); equal("405B share", r(L405.emb / L405.total * 100, 1), "0.5");
  claim("tf-tying", "6.5% of the 8B model, 1.5% of the 70B and 0.5% of the 405B");
  equal("70B total", int(L70.total), "70,553,706,496"); equal("405B total", int(L405.total), "405,853,388,800");
  claim("tf-tying", "8,030,261,248, 70,553,706,496 and 405,853,388,800");
  const spec = Deck.slides.find((x) => x.id === "tf-tying").fig;
  near("bars 8B", spec.items[1].value, L8.emb / L8.total, 1e-15); near("bars 70B", spec.items[2].value, L70.emb / L70.total, 1e-15); near("bars 405B", spec.items[3].value, L405.emb / L405.total, 1e-15);
}
{
  const spec = Deck.slides.find((x) => x.id === "tf-llama3").fig, row = (name) => spec.rows.find((x) => x[0] === name).slice(1, 4);
  equal("layers", row("Layers"), ["32", "80", "126"]); equal("d", row("Model dimension"), ["4,096", "8,192", "16,384"]);
  equal("ffn", row("FFN dimension"), ["14,336", "28,672", "53,248"]); equal("heads", row("Attention heads"), ["32", "64", "128"]);
  equal("kv", row("Key/value heads"), ["8", "8", "8"]); equal("head size", row("Head size"), ["128", "128", "128"]);
  equal("params", row("Parameters"), [int(L8.total), int(L70.total), int(L405.total)]);
  ok("LLAMA constants match the table", ["8B", "70B", "405B"].every((k, i) => int(LLAMA[k].L) === row("Layers")[i] && int(LLAMA[k].d) === row("Model dimension")[i] && int(LLAMA[k].ff) === row("FFN dimension")[i] && int(LLAMA[k].H) === row("Attention heads")[i]));
  // SwiGLU width rule in Meta's code: 2/3 of 4d, times ffn_dim_multiplier, rounded up to multiple_of
  const ffw = (d, mult, of) => { let x = Math.floor(2 * 4 * d / 3); x = Math.floor(mult * x); return of * Math.ceil(x / of); };
  equal("8B d_ff rule", ffw(4096, 1.3, 1024), 14336); equal("70B d_ff rule", ffw(8192, 1.3, 4096), 28672); equal("405B d_ff rule", ffw(16384, 1.2, 4096), 53248);
  claim("tf-llama3", "128,000 tokens (128,256 rows"); claim("tf-llama3", "500,000"); claim("tf-llama3", "8K-token sequences"); claim("tf-llama3", "128K tokens");
}
{
  claim("tf-vs-llama", "9,968"); claim("tf-vs-llama", int(L8.total));
  equal("blocks", int(32 * L8.layer), "6,979,584,000"); claim("tf-vs-llama", "6,979,584,000"); equal("per block", int(L8.layer), "218,112,000"); claim("tf-vs-llama", "218,112,000");
  equal("vocab tables", int(2 * L8.emb), "1,050,673,152"); claim("tf-vs-llama", "1,050,673,152");
  equal("sum", 32 * L8.layer + 2 * L8.emb + 4096, L8.total);
  equal("KV per token 8B", int(2 * 32 * 8 * 128 * 2), "131,072"); claim("tf-vs-llama", "131,072 bytes");
  equal("MQA", int(2 * 32 * 1 * 128 * 2), "16,384"); claim("tf-vs-llama", "16,384 bytes");
  equal("our eps", C.eps, 1e-5); claim("tf-vs-llama", "ε = 10⁻⁵");
  equal("our per block", 3104, 2 * 16 + 4 * 256 + 2048); claim("tf-vs-llama", "3,104");
}


// ------------------------------------------------------------------ wording added in review
{
  equal("unmasked pairs", 8 * 9 / 2, 36); claim("tf-matrix-form", "compute 36");
  equal("L1H2 name->SELECT", r(tr.layers[0].heads[1].att[1][0], 2), "1.00"); claim("tf-heads", "reads <code>SELECT</code> (1.00)".replace(/<[^>]+>/g, ""));
  const prevTok = [1, 2, 3, 4, 5, 6, 7].filter((i) => tr.layers[0].heads[1].att[i][i - 1] > 0.9);
  equal("rows of L1H2 that read the previous token", prevTok, [1, 3, 4, 5]); claim("tf-heads", "four positions read the token just before them");
  equal("Llama 3 8B blocks share", Math.round(32 * L8.layer / L8.total * 100), 87); claim("tf-params", "the 32 blocks the other 87%");
}


// ------------------------------------------------------------------ legacy widgets on these sheets, driven to their extremes
{
  const W_ = (w) => mount("tinylm", { widget: w }).el;
  const fmtp = (v) => (v >= 0.9995 ? "1.000" : v.toFixed(3));
  // qkv: every head and asking position; scores shown to 2 decimals equal the trace
  const q = W_("qkv"), HEADS = [[0, 0], [0, 1], [1, 0], [1, 1]];
  HEADS.forEach(([l, hh], hi) => {
    [...q.querySelectorAll(".seg-btn")].find((b) => b.dataset.v === String(hi)).click();
    for (let pos = 0; pos < 8; pos++) {
      q.querySelectorAll(".pos-btn")[pos].click();
      const shown = [...q.querySelectorAll(".kv-row:not(.kv-head) .kv-score .num")].map((e) => e.textContent);
      equal(`qkv L${l + 1}H${hh + 1} pos ${pos}`, shown, tr.layers[l].heads[hh].scores[pos].slice(0, pos + 1).map((v) => minus(v.toFixed(2))));
    }
  });
  // mix: weights for every head and position
  const mixEl = W_("mix");
  HEADS.forEach(([l, hh], hi) => {
    [...mixEl.querySelectorAll(".seg-btn")].find((b) => b.dataset.v === String(hi)).click();
    for (let pos = 0; pos < 8; pos++) {
      mixEl.querySelectorAll(".pos-btn")[pos].click();
      const shown = [...mixEl.querySelectorAll(".mx-row:not(.mx-head) .w-cell .num")].map((e) => e.textContent);
      equal(`mix L${l + 1}H${hh + 1} pos ${pos}`, shown, tr.layers[l].heads[hh].att[pos].map((v, j) => (j > pos ? "0" : v.toFixed(2))));
    }
  });
  // block: stream sizes at every position
  const bl = W_("block");
  for (let pos = 0; pos < 8; pos++) {
    bl.querySelectorAll(".pos-btn")[pos].click();
    const sizes = [...bl.querySelectorAll(".vec-note")].map((e) => e.textContent);
    equal(`block sizes pos ${pos}`, sizes[0], `size ${norm(tr.x0[pos]).toFixed(1)}`);
    equal(`block final size pos ${pos}`, sizes[sizes.length - 1], `size ${norm(tr.layers[1].post[pos]).toFixed(1)}`);
  }
  // logits: temperature, top-k and top-p at their extremes
  const lg = W_("logits"), z = tr.logits[4];
  const order = z.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 8).map(([, i]) => i);
  const sliders = lg.querySelectorAll('input[type="range"]');
  const set = (k, v) => { sliders[k].value = String(v); sliders[k].dispatchEvent(new window.Event("input")); };
  for (const [T, K, P] of [[1, 0, 1], [0.1, 0, 1], [3, 0, 1], [1, 1, 1], [1, 8, 1], [1, 0, 0.05], [0.1, 1, 0.05], [3, 8, 1]]) {
    set(0, T); set(1, K); set(2, P);
    const pf = M.filterProbs(M.softmax(z, T), { topK: K, topP: P });
    const shown = [...lg.querySelectorAll(".lg-row:not(.lg-head) .lg-p .num")].map((e) => e.textContent);
    equal(`logits T${T} k${K} p${P}`, shown, order.map((i) => (pf[i] > 0 ? fmtp(pf[i]) : "0")));
    ok(`logits note T${T} k${K} p${P}`, lg.querySelector(".lg-note").textContent.startsWith(`${pf.filter((x) => x > 0).length} of 113`));
  }
  // train: every saved step; the cards show p(true next) for our query at that step
  const trn = W_("train"), sl = trn.querySelector('input[type="range"]');
  M.snapSteps.forEach((st, k) => {
    sl.value = String(k); sl.dispatchEvent(new window.Event("input"));
    const t = M.forward(IDS, st), shown = [...trn.querySelectorAll(".tr-ours b")].map((e) => e.textContent);
    equal(`train step ${st}`, shown, t.probs.map((pp, i) => fmtp(pp[ALL[i + 1]])));
  });
  // heads: every preset query
  const hd = W_("heads"), sel = hd.querySelector("select");
  [...sel.options].forEach((o, k) => {
    sel.value = o.value; sel.dispatchEvent(new window.Event("change"));
    ok(`heads preset ${k} draws 4 maps`, hd.querySelectorAll("svg.attn-map").length === 4);
  });
  // need: every column and table, including the combinations the grammar never produces
  const nd = W_("need"), c1 = nd.querySelector("#need-c1"), tb = nd.querySelector("#need-t");
  for (const c of [...c1.options].map((o) => o.value)) for (const t of [...tb.options].map((o) => o.value)) {
    c1.value = c; c1.dispatchEvent(new window.Event("change")); tb.value = t; tb.dispatchEvent(new window.Event("change"));
    const ids = ["SELECT", " " + c, " FROM", " " + t, " WHERE"].map(M.idOf), pr = M.forward(ids).probs;
    const topWhere = [...pr[4]].map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])[0];
    const shown = nd.querySelectorAll(".need-where .bar-row .bar-val")[0].textContent;
    equal(`need ${c}/${t}`, shown, fmtp(topWhere[0]));
  }
  // backprop: forward then backward
  const bp = W_("backprop"), bb = bp.querySelectorAll(".bp-ctl button");
  bb[0].click(); ok("backprop forward shows p", bp.textContent.includes(`p(id) = ${r(tr.probs[4][58], 3)}`));
  bb[1].click(); ok("backprop backward shows p - 1", bp.textContent.includes(minus((tr.probs[4][58] - 1).toFixed(3))));
  // heldout, real, moe: render and their stated numbers
  const ho = W_("heldout"), hs = [...ho.querySelectorAll(".ho-out .stat-v")].map((e) => e.textContent);
  equal("heldout stats", hs, [r(D.holdout.lossB.seen, 2), r(D.holdout.lossB.held, 2), fmtp(M.forward(IDS, "holdout").probs[6][ALL[7]])]);
  const rl = W_("real"), ratios = () => [...rl.querySelectorAll(".ratio-v")].map((e) => e.textContent);
  equal("real 405B ratios", ratios(), ["40.6 million×", "2.5 million×"]);
  const mo = W_("moe"), ms = mo.querySelectorAll(".ctl-seg");
  for (const k of [1, 2]) for (const cap of [0, 1.25, 1]) {
    [...ms[0].querySelectorAll(".seg-btn")].find((b) => b.dataset.v === String(k)).click();
    [...ms[1].querySelectorAll(".seg-btn")].find((b) => b.dataset.v === String(cap)).click();
    for (const skew of [0, 3]) {
      const inp = mo.querySelector('input[type="range"]'); inp.value = String(skew); inp.dispatchEvent(new window.Event("input"));
      const stats = [...mo.querySelectorAll(".moe-out .stat-v")].map((e) => e.textContent);
      const placed = [...mo.querySelectorAll(".moe-exp .num")].reduce((a, e) => a + +e.textContent, 0), dropped = +stats[2];
      equal(`moe k${k} cap${cap} skew${skew}: every token copy placed or dropped`, placed + dropped, 32 * k);
      equal(`moe k${k} work`, stats[3], `${k} of 8`);
    }
  }
}


// ------------------------------------------------------------------ findings of the independent review
{
  equal("projections at n = 8, d = 16", int(4 * 8 * 16 * 16), "8,192"); claim("tf-rnn", "4·n·d² = 8,192");
  claim("tf-att-cost", "3,072 for ours; its 32 norm gains act elementwise");
  equal("our layer's matrix weights", 4 * 256 + 2 * 16 * 64, 3072);
  claim("tf-mask", String.raw`\mathbf{A}_h = \mathrm{softmax}\left(\frac{\mathbf{Q}_h\mathbf{K}_h^\top}`);
  claim("tf-matrix-form", String.raw`\(\mathbf{Q}_h\): columns \(8(h-1)\) to \(8h-1\)`);
  ok("head h uses columns 8(h-1)..8h-1", [1, 2].every((hh) => tr.layers[0].heads[hh - 1].scores[4][3] === tr.layers[0].heads[hh - 1].scores[4][3] && (() => { const q = tr.layers[0].q[4].slice(8 * (hh - 1), 8 * hh), k = tr.layers[0].k[3].slice(8 * (hh - 1), 8 * hh); return Math.abs(dot(q, k) / Math.sqrt(8) - tr.layers[0].heads[hh - 1].scores[4][3]) < 1e-12; })()));
  claim("tf-qkv-maps", "multiplied entry by entry by 16 learned gains");
  equal("smallest gain is in layer 2", r(Math.abs(W["l1.n1"].data[4]), 5), "0.00014");
  claim("tf-part", "mixture-of-experts layers of Mixtral and DeepSeek-V3");
  claim("tf-need", "The input vector at a position does not determine its prediction");
  claim("tf-rmsnorm", "Xiong et al. (2020)");
  claim("tf-gqa", "never change and are kept instead of recomputed");
  // the sinusoid legend is drawn at every pair
  const sp = Deck.slides.find((x) => x.id === "tf-sinusoid").fig;
  ok("sinusoid legend above the curves", sp.y[1] >= 1.7 && sp.yticks.every((t) => t <= 1));
  // the SwiGLU plot keeps every curve inside its y-range at the slider's extremes
  const sw = Deck.slides.find((x) => x.id === "tf-swiglu").fig;
  ok("SwiGLU curve within the y-range for b in [-2, 2]", [-2, 2].every((b) => [-4, -1.28, 0, 4].every((a) => Math.abs(sw.fns[0].f(a, { b })) <= sw.y[1])));
  equal("SwiGLU slider label", sw.params[0].label, "second projection b = x · w₃");
  // block diagram: every node is connected and the attention sum feeds the MLP row
  const bd = Deck.slides.find((x) => x.id === "tf-block-diagram").fig;
  ok("block diagram has the a1 -> x + attn edge", bd.edges.some((e) => e.from === "a1" && e.to === "xm"));
  ok("every block-diagram node has an edge", bd.nodes.every((n) => bd.edges.some((e) => e.from === n.id || e.to === n.id)));
  // the moe widget's capacity rule is capacity factor × T·k/E, and no expert exceeds it
  const mo = mount("tinylm", { widget: "moe" }).el, segs = mo.querySelectorAll(".ctl-seg");
  for (const k of [1, 2]) for (const cap of [1.25, 1]) {
    [...segs[0].querySelectorAll(".seg-btn")].find((b) => b.dataset.v === String(k)).click();
    [...segs[1].querySelectorAll(".seg-btn")].find((b) => b.dataset.v === String(cap)).click();
    const inp = mo.querySelector('input[type="range"]'); inp.value = "3"; inp.dispatchEvent(new window.Event("input"));
    const limit = Math.ceil(cap * 32 * k / 8), loads = [...mo.querySelectorAll(".moe-exp .num")].map((e) => +e.textContent);
    ok(`moe k${k} cap${cap}: no expert above ${limit}`, loads.every((x) => x <= limit));
    ok(`moe k${k} cap${cap}: drop note names the limit`, !mo.querySelector(".moe-drop") || mo.querySelector(".moe-drop").textContent.includes(`capacity of ${limit}`));
  }
}

done();
