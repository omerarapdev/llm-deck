// Recomputes every model-derived number the slides quote and checks the text.
// Run: node legacy/tools/check_text.mjs
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

// v1 deck: checks legacy/index.html against the same trained model
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const root = path.join(repo, "legacy");
const ctx = { window: {}, atob: (s) => Buffer.from(s, "base64").toString("binary"), Math, console };
vm.createContext(ctx);
for (const f of ["data/model-data.js", "legacy/js/model.js"]) vm.runInContext(fs.readFileSync(path.join(repo, f), "utf8"), ctx);
const M = ctx.window.TinyLM, D = M.data, V = M.vocab, C = M.config;
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const facts = [...html.matchAll(/data-fact="([^"]*)"/g)].map((x) => x[1]).join(" ");
const clean = (s) => s.replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/\s+/g, " ");
const text = clean(html) + " " + clean(facts);

let fails = 0, passes = 0;
const ok = (cond, msg) => { if (cond) passes++; else { fails++; console.log("FAIL", msg); } };
const p3 = (x) => (x >= 0.9995 ? "1.000" : x.toFixed(3));
const ids = M.runningIds.slice(0, 8);
const fin = M.forward(ids);

// ---------------------------------------------------------------- data-check spans
function sampleValidity(T) {
  const valid = new Set(M.allQueries().map((q) => q.text));
  const rand = M.mulberry32(42);
  let good = 0;
  for (let k = 0; k < 2000; k++) {
    const cur = [M.runningIds[0]];
    while (cur.length < 9) { const t = M.sampleFrom(M.nextDist(cur, { temperature: T }), rand); cur.push(t); if (V[t] === ";") break; }
    if (valid.has(cur.map((i) => V[i]).join(""))) good++;
  }
  return `${((good / 2000) * 100).toFixed(1)}%`;
}
const floorMean = D.floor.reduce((a, b) => a + b, 0) / D.floor.length;
let holdCache = null;
function holdSplit() {
  if (holdCache) return holdCache;
  const held = new Set(D.holdout.queries);
  const acc = { held: new Array(8).fill(0), seen: new Array(8).fill(0) }, w = { held: 0, seen: 0 };
  for (const q of M.allQueries()) {
    const qi = M.encode(q.text).map(M.idOf), key = held.has(q.text) ? "held" : "seen";
    const tr = M.forward(qi.slice(0, 8), "holdout");
    w[key] += q.p;
    for (let i = 0; i < 8; i++) acc[key][i] += -Math.log(tr.probs[i][qi[i + 1]]) * q.p;
  }
  holdCache = { held: acc.held.map((x) => x / w.held), seen: acc.seen.map((x) => x / w.seen) };
  return holdCache;
}
const expect = (key) => {
  const [k, ...a] = key.split(":");
  if (k === "params") return C.params.toLocaleString("en-US");
  if (k === "vocab") return String(V.length);
  if (k === "floor") return floorMean.toFixed(4);
  if (k === "exact") return M.exactLoss(a[0] === "final" ? M.finalStep : +a[0]).mean.toFixed(4);
  if (k === "p") { const tr = a[0] === "final" ? fin : M.forward(ids, +a[0]); return p3(tr.probs[+a[1]][+a[2]]); }
  if (k === "att") return fin.layers[+a[0] - 1].heads[+a[1] - 1].att[+a[2]][+a[3]].toFixed(2);
  if (k === "valid") return sampleValidity(+a[0]);
  if (k === "hold") {
    const B = M.forward(ids, "holdout");
    if (a[0] === "p7") return p3(B.probs[6][M.idOf(" 7")]);
    const sl = holdSplit();
    if (a[0] === "held6") return sl.held[6].toFixed(2);
    if (a[0] === "seen6") return sl.seen[6].toFixed(2);
  }
  throw new Error("unknown key " + key);
};
const re = /data-check="([^"]+)"[^>]*>([^<]*)</g;
let m;
while ((m = re.exec(html))) {
  const want = expect(m[1]);
  ok(m[2].trim() === want, `data-check ${m[1]}: text "${m[2].trim()}" but model gives "${want}"`);
}

// ---------------------------------------------------------------- claims in running text
const has = (s) => text.includes(s);
const claim = (s, cond, why) => { ok(has(s), `text no longer contains: "${s}"`); ok(cond, `claim "${s}" is wrong: ${why}`); };

claim("36 characters to 9 tokens", D.running.length === 36 && M.running.length === 9, `${D.running.length} chars, ${M.running.length} tokens`);
claim("Our tokenizer made 74 merges over 2,000 queries", D.merges.length === 74 && D.corpusSize === 2000, "merge/corpus counts");
claim("Ours only knows the 39 characters", D.baseCount === 39, `base ${D.baseCount}`);
claim("contain only 269 distinct ones", D.corpusDistinct === 269, `distinct ${D.corpusDistinct}`);
claim("These rules allow exactly 270 different queries", M.allQueries().length === 270, "query count");
claim("Token id 63, users", M.idOf(" users") === 63, `id ${M.idOf(" users")}`);
const idsTxt = M.runningIds.join(" ");
ok(html.includes(`ids ${idsTxt}`), `fact with token ids should read "${idsTxt}"`);
claim("768,000", C.steps * C.batch === 768000, "steps × batch");
claim("6 × 9,968 × 6,144,000 ≈ 3.7 × 10¹¹", Math.abs(6 * C.params * C.steps * C.batch * 8 / 3.7e11 - 1) < 0.01, "6ND for our model");
claim("about ln 113 = 4.73", Math.log(V.length).toFixed(2) === "4.73", "ln 113");
claim("ln 9 = 2.197", Math.log(9).toFixed(3) === "2.197", "ln 9");
claim("2 × 2 × 2 × 8 × 4 = 256 bytes per token", 2 * C.L * C.H * (C.d / C.H) * 4 === 256, "our KV bytes");
claim("64 numbers in all", C.L * C.H * 2 * (C.d / C.H) === 64, "per-token cache entries");
claim("Our gradient is 39.9 KB per step", (C.params * 4 / 1000).toFixed(1) === "39.9", "FP32 gradient bytes");
claim("159 KB in total", Math.round(C.params * 16 / 1000) === 159, "16 bytes × params");
claim("stacks 2 blocks", C.L === 2, "layers");
claim("16 numbers to 64", C.d === 16 && C.dff === 64, "MLP width");
claim("a 16 × 113 matrix", C.d === 16 && V.length === 113, "unembedding");
claim("one row of 16 numbers for each of the 113 tokens", V.length === 113 && C.d === 16, "embedding table");
claim("peak rate 0.01, β₁ = 0.9, β₂ = 0.95", C.lr === 0.01 && C.betas[0] === 0.9 && C.betas[1] === 0.95, "optimizer config");
claim("100 warmup steps", C.warmup === 100, "warmup");

// claims that live in widget code
{
  const js = fs.readFileSync(path.join(root, "js/w-start.js"), "utf8");
  const distinctCols = new Set(Object.values(D.schema).flat()).size;
  ok(js.includes("Nine different column names") && distinctCols === 9, `predict note: ${distinctCols} distinct column names`);
  const p0 = fin.probs[0], top = p0.indexOf(Math.max(...p0));
  ok(js.includes("<code>id</code> leads") && V[top] === " id", `top first column is ${V[top]}`);
  ok(Math.abs(p0[M.idOf(" name")] - 1 / 12) < 0.005, `p(name) ${p0[M.idOf(" name")].toFixed(3)} vs 1/12`);
}

// greedy decoding from "SELECT"
{
  const cur = [M.runningIds[0]];
  while (cur.length < 9) { const p = M.nextDist(cur); let b = 0; p.forEach((v, i) => { if (v > p[b]) b = i; }); cur.push(b); if (V[b] === ";") break; }
  const greedy = cur.map((i) => V[i]).join("");
  claim(`gives ${greedy}`, true, "greedy output");
}

// training dynamics claims
const at = (step) => D.exactCurve.find((r) => r[0] === step)[2];
const s100 = at(100), s150 = at(150);
claim("cost almost nothing by step 100", [1, 3, 5, 7].every((i) => s100[i] < 0.03), `step-100 losses ${[1, 3, 5, 7].map((i) => s100[i].toFixed(3))}`);
claim("follows by step 150", s150[2] - D.floor[2] < 0.05, `pos-2 loss ${s150[2].toFixed(3)} vs floor ${D.floor[2].toFixed(3)}`);
const lastConv = (i) => { const r = D.exactCurve.find((row) => row[2][i] - D.floor[i] < 0.01); return r ? r[0] : Infinity; };
claim("converges last", lastConv(4) >= Math.max(lastConv(1), lastConv(2), lastConv(3), lastConv(5), lastConv(7)), `pos 4 within 0.01 at step ${lastConv(4)}`);
claim("Four never reach zero because the data itself is uncertain there", [0, 2, 4, 6].every((i) => D.floor[i] > 0.1) && [1, 3, 5, 7].every((i) => D.floor[i] === 0), "nonzero floors");

// held-out experiment
claim("trained a second copy with the same recipe on only 243 of them. The other 27, including our query", D.holdout.queries.length === 27 && D.holdout.queries.includes(D.running) && 270 - 27 === 243, "held-out set");
claim("saw each of the 270 queries about 2,800 times", Math.round(C.steps * C.batch / 270 / 100) === 28, "samples per query");
{
  const sl = holdSplit();
  const struct = [0, 2, 4].every((i) => Math.abs(sl.held[i] - sl.seen[i]) < 0.15);
  claim("it still gets the structure right", struct, `held vs seen at positions 0,2,4: ${[0, 2, 4].map((i) => sl.held[i].toFixed(2) + "/" + sl.seen[i].toFixed(2))}`);
  claim("below the true floor of 2.197", sl.seen[6] < D.floor[6], `seen digit loss ${sl.seen[6].toFixed(3)}`);
  ok(html.includes("its loss on held-out queries is " + (sl.held.reduce((a, b) => a + b) / 8).toFixed(2) + " vs. " + (sl.seen.reduce((a, b) => a + b) / 8).toFixed(2)), "held-out fact numbers");
}

// speculative decoding acceptance (draft = step 150)
const draft = M.forward(ids, 150).probs;
const acc = fin.probs.map((p, i) => p.reduce((a, v, j) => a + Math.min(v, draft[i][j]), 0));
claim("93% of drafted digits are accepted", Math.round(acc[6] * 100) === 93, `digit acceptance ${(acc[6] * 100).toFixed(1)}%`);
claim("accepted only 67% of the time", Math.round(acc[4] * 100) === 67, `WHERE acceptance ${(acc[4] * 100).toFixed(1)}%`);

// quantization claims (per-channel)
function quantize(Wt, bits) {
  const out = {}, qmax = Math.pow(2, bits - 1) - 1;
  for (const [name, t] of Object.entries(Wt)) {
    if (t.shape.length < 2) { out[name] = t; continue; }
    const [r, c] = t.shape, data = new Float64Array(t.data.length);
    for (let j = 0; j < c; j++) {
      let mx = 0;
      for (let i = 0; i < r; i++) mx = Math.max(mx, Math.abs(t.data[i * c + j]));
      const s = mx / qmax || 1;
      for (let i = 0; i < r; i++) data[i * c + j] = Math.max(-qmax, Math.min(qmax, Math.round(t.data[i * c + j] / s))) * s;
    }
    out[name] = { shape: t.shape, data };
  }
  return out;
}
const Wt = M.weights(), base = M.exactLoss().mean;
const ql = {};
for (const b of [8, 6, 5, 4]) {
  const Wq = quantize(Wt, b);
  ql[b] = { loss: M.exactLoss(M.finalStep, { weights: Wq }).mean, pu: M.forward(ids, M.finalStep, { weights: Wq }).probs[2][M.idOf(" users")] };
}
claim("8, 6 and 5 bits barely move the loss", [8, 6, 5].every((b) => ql[b].loss - base < 0.002), `losses ${[8, 6, 5].map((b) => ql[b].loss.toFixed(4))}`);
ok(html.includes(`(${base.toFixed(4)} → ${ql[5].loss.toFixed(4)})`), `quant fact should say (${base.toFixed(4)} → ${ql[5].loss.toFixed(4)})`);
claim("falls from 1.000 to 0.000", p3(fin.probs[2][M.idOf(" users")]) === "1.000" && ql[4].pu < 0.0005, `4-bit p(users) ${ql[4].pu}`);

// RL base rate
const pre = ["SELECT", " email", " FROM", " users", " WHERE"].map(M.idOf);
const pc = M.forward(pre).probs[4];
const pid = pc[M.idOf(" id")] / [" id", " name", " age"].reduce((a, c) => a + pc[M.idOf(c)], 0);
const p7 = M.forward(pre.concat([M.idOf(" id"), M.idOf(" =")])).probs[6][M.idOf(" 7")];
claim("about 1 sample in 26", Math.round(1 / (pid * p7)) === 26, `1/${(1 / (pid * p7)).toFixed(1)}`);

// arithmetic about real systems
const near = (a, b, tol = 0.01) => Math.abs(a / b - 1) < tol;
claim("6 × 405 × 10⁹ × 15.6 × 10¹² ≈ 3.8 × 10²⁵", near(6 * 405e9 * 15.6e12, 3.8e25, 0.01), "6ND 405B");
claim("about 295 operations for every byte", Math.round(989.4 / 3.35) === 295, "ridge");
claim("67 million scores per head, 128 MiB in BF16", Math.round(8192 ** 2 / 1e6) === 67 && 8192 ** 2 * 2 / 2 ** 20 === 128, "score matrix");
claim("2 × 80 × 8 × 128 × 2 = 327,680 bytes, or 320 KiB per token", 2 * 80 * 8 * 128 * 2 === 327680 && 327680 / 1024 === 320, "KV 70B");
claim("needs 2.5 GiB; 64 of them at once need 160 GiB", 327680 * 8192 / 2 ** 30 === 2.5 && 64 * 2.5 === 160, "KV per conversation");
claim("would need 20 GiB", 8 * 2.5 === 20, "no GQA");
claim("that is 1.12 TB", near(70e9 * 16, 1.12e12), "16 bytes × 70B");
claim("BF16 weights alone, 140 GB", 70e9 * 2 === 140e9, "70B weights");
claim("64 GPUs at 17.5 GB each", 70e9 * 16 / 64 === 17.5e9, "ZeRO-3");
claim("26.8 TB/s together, a step takes at least 5.2 ms", near(8 * 3.35e12, 26.8e12) && (140e9 / 26.8e12 * 1000).toFixed(1) === "5.2", "decode bound");
claim("8 × 16 × 128 = 16,384 GPUs", 8 * 16 * 128 === 16384, "4D layout");
claim("about every 3 hours", Math.round(54 * 24 / 419) === 3, "MTBF");
claim("roughly 1,900 per parameter", Math.round(15e12 / 8e9 / 100) === 19, "8B tokens/param");
claim("525 million numbers", Math.round(128256 * 4096 / 1e6) === 525, "8B embedding");
claim("about 68 days on 16,384 H100s", Math.round(3.79e25 / (0.4 * 989e12) / 16384 / 86400) === 68, "405B calendar time");

console.log(`${passes} checks passed, ${fails} failed`);
process.exit(fails ? 1 : 0);
