// Part 8: every number stated on the serving sheets, recomputed from first principles.
// Hardware: NVIDIA H100 SXM datasheet / whitepaper. Models: Llama 3.1 config.json files.
import { claim, near, equal, ok, r, int, TinyLM, done, slide, text, window } from "./lib.mjs";

const M = TinyLM, V = M.vocab;
const fx = (x, d) => r(x, d);                       // fixed decimals as written
const GBs = (b, d = 1) => fx(b / 1e9, d);           // decimal GB

// ------------------------------------------------------------ independent constants
// HBM: an "80 GB" H100 has 80 GiB; CUDA reports 79.32 GiB as the device total (PyTorch OOM report on an H100 80GB);
// vLLM's budget is gpu_memory_utilization (0.92) x the reported total.
const GIB = 2 ** 30, CUDA_TOTAL = 79.32 * GIB, UTIL = 0.92, USABLE = (g) => g * CUDA_TOTAL * UTIL;
const HW = { mem: 80 * GIB, bw: 3.35e12, bf16: 989e12, fp8: 1979e12 };
const SERVER = { gpus: 8, bw: 8 * 3.35e12, peak: 8 * 989e12, mem: USABLE(8) };
const cfg = {
  "8B": { L: 32, d: 4096, ffn: 14336, H: 32, KV: 8, hd: 128, V: 128256 },
  "70B": { L: 80, d: 8192, ffn: 28672, H: 64, KV: 8, hd: 128, V: 128256 },
  "405B": { L: 126, d: 16384, ffn: 53248, H: 128, KV: 8, hd: 128, V: 128256 },
};
function count(c) {
  const q = c.d * c.H * c.hd, kv = 2 * c.d * c.KV * c.hd, o = c.H * c.hd * c.d, mlp = 3 * c.d * c.ffn, norms = 2 * c.d;
  const layer = q + kv + o + mlp + norms;
  const emb = c.V * c.d, head = c.V * c.d;
  return { attn: q + kv + o, mlp, norms, layer, emb, head, total: emb + c.L * layer + head + c.d, read: c.L * layer + head + c.d };
}
const P8 = count(cfg["8B"]), P70 = count(cfg["70B"]), P405 = count(cfg["405B"]);
// Hugging Face model metadata (safetensors parameter counts) for the three checkpoints
near("8B params", P8.total, 8030261248);
near("70B params", P70.total, 70553706496);
near("405B params", P405.total, 405853388800);
const kvTok = (c, b = 2) => 2 * c.L * c.KV * c.hd * b;
const attn1 = (c) => 4 * c.L * c.H * c.hd;           // attention FLOPs per (query token, cached position)
const c70 = cfg["70B"];
const W70read = P70.read * 2, W70 = P70.total * 2;
const step70 = (B, ctx) => {
  const tm = (W70read + B * ctx * kvTok(c70)) / SERVER.bw;
  const tc = (B * (2 * P70.read + attn1(c70) * ctx)) / SERVER.peak;
  return { t: Math.max(tm, tc), tm, tc, mem: W70 + B * ctx * kvTok(c70) };
};

// ================================================================ 8.1 Two phases
{
  const ids = M.runningIds, tr = M.forward(ids.slice(0, 8));
  const pu = tr.probs[2][M.idOf(" users")];
  ok("p(users | SELECT name FROM) rounds to 1.000", pu >= 0.9995);
  claim("serve-phases", "(p = 1.000)");
  claim("serve-phases", "Our 6 output tokens take 6 passes: 1 prefill and 5 decodes");
  // attention scores: prefill 3 tokens causal = 6; decodes 4..8; total 36 = 8*9/2
  const sc = 6 + [4, 5, 6, 7, 8].reduce((a, b) => a + b, 0);
  near("scores total", sc, 36); near("one full pass", (8 * 9) / 2, 36);
}
{
  const ridge = HW.bf16 / HW.bw;
  near("ridge", ridge, 295.22, 0.01);
  claim("serve-bound", "295 FLOPs per byte");
  claim("serve-bound", `1/295 = ${fx(100 / ridge, 2)}%`);
  claim("serve-bound", `reads ${GBs(W70read)} GB of weights`);
  const tRead = W70read / SERVER.bw;
  claim("serve-bound", `at least ${fx(tRead * 1e3, 2)} ms`);
  const pf = 2048 * 2 * P70.read;
  claim("serve-bound", `2,048 × ${fx((2 * P70.read) / 1e9, 1)} GFLOP = ${fx(pf / 1e12, 1)} TFLOP`);
  claim("serve-bound", `${fx((pf / SERVER.peak) * 1e3, 1)} ms at the peak of 8 GPUs`);
  near("prefill/read ratio ~7", Math.round(pf / SERVER.peak / tRead), 7);
  claim("serve-bound", "seven times the read");
  claim("serve-bound", `${fx(((2 * P70.read) / SERVER.peak) * 1e6, 1)} µs of math against ${fx(tRead * 1e3, 2)} ms`);
}
{
  // metrics: 2,048-token prompt, 500-token answer, alone on 8 GPUs
  const P = 2048, n = 500;
  const pfFlops = 2 * P70.read * P + (attn1(c70) * P * (P + 1)) / 2;
  claim("serve-metrics", `prefill needs ${fx(pfFlops / 1e12, 1)} TFLOP including attention`);
  const ttft = Math.max(pfFlops / SERVER.peak, W70read / SERVER.bw);
  claim("serve-metrics", `TTFT ≥ ${fx(ttft * 1e3, 1)} ms`);
  claim("serve-metrics", `reads ${GBs(W70read)} GB of weights and up to ${GBs((P + n - 1) * kvTok(c70), 2)} GB of cache`);
  let e2e = ttft, first = 0;
  for (let i = 1; i <= n - 1; i++) { const t = step70(1, P + i).t; if (i === 1) first = t; e2e += t; }
  claim("serve-metrics", `TPOT ≥ ${fx(first * 1e3, 2)} ms`);
  claim("serve-metrics", `takes at least ${fx(ttft * 1e3, 1)} ms + 499 steps of about ${fx(((e2e - ttft) / 499) * 1e3, 2)} ms = ${fx(e2e, 2)} s`);
}
{
  const c = count(c70);
  near("attention per layer", c.attn, 150994944);
  near("MLP per layer", c.mlp, 704643072);
  near("norms per layer", c.norms, 16384);
  near("layer", c.layer, 855654400);
  near("read per step", c.read, 69503033344);
  for (const s of ["150{,}994{,}944", "704{,}643{,}072", "16{,}384", "855,654,400", "69{,}503{,}033{,}344"]) claim("serve-min-step", s);
  claim("serve-min-step", `<b>${GBs(c.read * 2)} GB</b>`.replace(/<[^>]+>/g, ""));
  const perGpu = (c.read * 2) / 8;
  claim("serve-min-step", `${fx(perGpu / 1e9, 2)} GB`);
  const t = perGpu / HW.bw;
  claim("serve-min-step", `= ${fx(t * 1e3, 2)}\\ \\text{ms}`);
  claim("serve-min-step", `at most ${int(1 / t)} tokens/s`);
  claim("serve-min-step", `${int(1 / t)} tokens per second for one request`);
  claim("serve-min-step", `${fx((2 * c.read) / 1e9, 1)}\\) GFLOP, \\(${fx(((2 * c.read) / SERVER.peak) * 1e6, 1)}\\ \\mu\\text{s}`);
  claim("serve-min-step", `${fx((100 * (2 * c.read) / SERVER.peak) / t, 2)}% of the step`);
  claim("serve-min-step", `In BF16, Llama 3.1 70B on 8 H100s needs at least ${fx(t * 1e3, 2)} ms per step`);
}
{
  claim("serve-batch", `${int(kvTok(c70))} bytes per token`);
  claim("serve-batch", `${fx((2048 * kvTok(c70)) / 1e9, 2)} GB for a 2,048-token request`);
  const bKV = Math.floor(W70read / (2048 * kvTok(c70))) + 1;
  claim("serve-batch", `From ${bKV} such requests on`);
  // smallest B at which a 128-token-context step is compute-bound
  let bC = 1; while (step70(bC, 128).tc < step70(bC, 128).tm) bC++;
  claim("serve-batch", `compute-bound at ${bC} requests`);
  // context above which no batch size is compute-bound: per-request compute < per-request cache read
  const ctxStar = ((2 * P70.read) / SERVER.peak) / (kvTok(c70) / SERVER.bw - attn1(c70) / SERVER.peak);
  claim("serve-batch", `at ${int(Math.ceil(ctxStar))} tokens of context or more it never does`);
  ok("no B is compute-bound at ctx 1,478", [1, 16, 256, 1024, 1e6].every((b) => step70(b, Math.ceil(ctxStar)).tc < step70(b, Math.ceil(ctxStar)).tm));
  claim("serve-batch", `With 92% of the memory CUDA reports usable (8.2), ${GBs(USABLE(8))} GB minus ${GBs(W70)} GB of weights leaves ${GBs(USABLE(8) - W70)} GB for caches: ${Math.floor((USABLE(8) - W70) / (2048 * kvTok(c70)))} requests of 2,048 tokens`);
  near("79.32 GiB in bytes", CUDA_TOTAL, 85.17e9, 0.005e9);
  const s64 = step70(64, 2048);
  claim("serve-batch", `64 requests get ${int(1 / s64.t)} tokens/s each and ${int(64 / s64.t)} tokens/s together`);
  const s1 = step70(1, 2048), s2 = step70(2, 2048);
  claim("serve-batch", `${fx(s1.t * 1e3, 2)} ms becomes ${fx(s2.t * 1e3, 2)} ms`);
  claim("serve-batch", `from ${int(1 / s1.t)} to ${int(2 / s2.t)} tokens/s`);
  claim("serve-batch", `rises by about ${fx(100 * (s2.t / s1.t - 1), 1)}%`);
  claim("serve-batch", "7,912 TFLOP/s"); near("server peak", SERVER.peak / 1e12, 7912);
  claim("serve-batch", "26.8 TB/s"); near("server bw", SERVER.bw / 1e12, 26.8, 1e-9);
}

// ================================================================ 8.2 The KV cache
{
  const ids = M.runningIds;
  for (let t = 3; t <= 8; t++) {
    const inc = M.forwardIncremental(ids.slice(0, t)), full = M.forward(ids.slice(0, t));
    let diff = 0; full.logits.forEach((row, i) => row.forEach((v, j) => { diff = Math.max(diff, Math.abs(v - inc.logits[i][j])); }));
    near(`cached vs full logits, ${t} tokens`, diff, 0, 0);
  }
  const C = M.config;
  near("numbers per token entry", 2 * C.L * C.d, 64);
  claim("serve-kvcache", "each of the 2 layers: 64 numbers");
  const without = 3 + [4, 5, 6, 7, 8].reduce((a, b) => a + b, 0);
  claim("serve-kvcache", `computes 8 entries with the cache and ${without} without (3 + 4 + 5 + 6 + 7 + 8)`);
}
{
  const ours = 2 * M.config.L * M.config.H * (M.config.d / M.config.H) * 4;
  near("ours per token", ours, 256);
  claim("serve-kvsize", "2 × 2 layers × 2 heads × 8 × 4 bytes = 256 bytes per token");
  claim("serve-kvsize", `${int(ours * 8)} bytes for the 8 positions`);
  claim("serve-kvsize", `2 × 80 × 8 × 128 × 2 = ${int(kvTok(c70))} bytes (${kvTok(c70) / 1024} KiB)`);
  const full = kvTok(c70) * 131072;
  claim("serve-kvsize", `needs ${GBs(full)} GB (${full / 2 ** 30} GiB)`);
  near("40 GiB is half of 80 GiB", full / HW.mem, 0.5, 1e-12);
  claim("serve-kvsize", "(40 GiB): half of an H100's 80 GiB");
  claim("serve-kvsize", `8B needs ${int(kvTok(cfg["8B"]))} bytes per token and 405B needs ${int(kvTok(cfg["405B"]))}`);
  claim("serve-kvsize", `${fx(kvTok(c70) / 1e6, 2)} MB per token, ${GBs(full)} GB at 128K`);
}
{
  const mha = 2 * 80 * 64 * 128 * 2, gqa = kvTok(c70), mqa = 2 * 80 * 1 * 128 * 2, mla80 = (512 + 64) * 80 * 2, mla = (512 + 64) * 61 * 2;
  claim("serve-gqa", `2 × 64 × 128 = ${int(2 * 64 * 128)} numbers in a 70B-shaped model. Over 80 layers in BF16 that is ${int(mha)} bytes per token, and ${GBs(mha * 131072)} GB (${mha * 131072 / GIB} GiB) for one 131,072-token request: all the HBM of four H100s`);
  near("MHA request = 4 × 80 GiB", mha * 131072, 4 * HW.mem, 0);
  claim("serve-gqa", `so ${int(2 * 8 * 128)} numbers, 8 times fewer`); near("MHA/GQA", mha / gqa, 8);
  claim("serve-gqa", `one key/value head: ${2 * 1 * 128} numbers`);
  claim("serve-gqa", `576 numbers, ${fx(2048 / 576, 1)} times fewer than GQA`);
  claim("serve-gqa", `Over DeepSeek-V3's 61 layers in BF16: ${int(mla)} bytes per token`);
  equal("bars per layer", slide("serve-gqa").fig.items.map((x) => [x.value, x.text]), [[16384, "16,384"], [2048, "2,048"], [576, "576"], [256, "256"]]);
  claim("serve-gqa", `MHA ${int(mha / 1000)} KB, GQA ${int(gqa / 1000)} KB, MLA ${int(mla80 / 1000)} KB, MQA ${int(mqa / 1000)} KB`);
  claim("serve-gqa", `${GBs(mha * 131072)} GB, ${GBs(gqa * 131072)} GB, ${GBs(mla80 * 131072)} GB and ${GBs(mqa * 131072)} GB per request`);
}
{
  claim("serve-kv-budget", `An "80 GB" H100 has 80 GiB of HBM3 (${fx(HW.mem / 1e9, 1)} × 10⁹ bytes), of which CUDA reports 79.32 GiB (${fx(CUDA_TOTAL / 1e9, 2)} × 10⁹)`);
  const w8 = P8.total * 2, use8 = USABLE(1), kv8 = use8 - w8;
  claim("serve-kv-budget", `0.92 × ${fx(CUDA_TOTAL / 1e9, 2)} GB = ${GBs(use8, 2)} GB; minus ${GBs(w8, 2)} GB of BF16 weights leaves ${GBs(kv8, 2)} GB`);
  const tok8 = Math.floor(kv8 / kvTok(cfg["8B"]));
  claim("serve-kv-budget", `${int(kvTok(cfg["8B"]))} bytes per token that holds ${int(tok8)} tokens: ${Math.floor(tok8 / 8192)} requests of 8,192 tokens, or ${Math.floor(tok8 / 131072)} of the full 131,072`);
  const kv70 = USABLE(8) - W70, tok70 = Math.floor(kv70 / kvTok(c70));
  claim("serve-kv-budget", `${GBs(USABLE(8))} GB minus ${GBs(W70)} GB of weights leaves ${GBs(kv70)} GB, which is ${int(tok70)} tokens: ${Math.floor(tok70 / 8192)} requests of 8,192 tokens`);
}
{
  // prefix caching: 12 queries sampled with mulberry32(1), each whole query used as a prompt
  const rand = M.mulberry32(1);
  const reqs = Array.from({ length: 16 }, () => M.encode(M.sampleQuery(rand)).map(M.idOf));
  const trie = new Set(); let comp12 = 0;
  reqs.slice(0, 12).forEach((t) => {
    let m = 0; while (m < t.length && trie.has(t.slice(0, m + 1).join(","))) m++;
    comp12 += t.length - Math.min(m, t.length - 1);
    for (let i = 1; i <= t.length; i++) trie.add(t.slice(0, i).join(","));
  });
  claim("serve-prefix", `${comp12} of their 108 tokens need computing`);
  claim("serve-prefix", "up to 6.4 times the throughput");
}
{
  const T = 32768, cache = kvTok(c70) * T, perGpu = cache / 8, copy = perGpu / 64e9;
  claim("serve-kv-offload", `cache is ${GBs(cache, 2)} GB, or ${GBs(perGpu, 2)} GB per GPU`);
  claim("serve-kv-offload", `${fx(copy * 1e3, 1)} ms to copy back`);
  const fl = 2 * P70.read * T + (attn1(c70) * T * (T + 1)) / 2;
  claim("serve-kv-offload", `prefill of ${int(fl / 1e12)} TFLOP: ${fx((fl / SERVER.peak) * 1e3, 1)} ms at the peak of 8 GPUs, ${Math.round(fl / SERVER.peak / copy)} times longer`);
  claim("serve-kv-offload", "rated at 128 GB/s, 64 GB/s in each direction");
  claim("serve-kv-offload", `= ${int(kvTok(c70))} bytes per token, × 32,768 tokens = ${GBs(cache)} GB`);
  claim("serve-kv-offload", `${GBs(cache / 2)} GB`);
  claim("serve-kv-offload", `${GBs(2 * 80 * 64 * 128 * 2 * T)} GB`);
}

// ================================================================ 8.3 Batching and scheduling
function batchSim(lens, slots, mode) {
  const finish = new Array(lens.length).fill(0); let t = 0, next = 0; const active = new Array(slots).fill(null);
  if (mode === "static") {
    while (next < lens.length) { const b = []; for (let k = 0; k < slots && next < lens.length; k++) b.push(next++); b.forEach((r) => { finish[r] = t + lens[r]; }); t += Math.max(...b.map((r) => lens[r])); }
  } else {
    while (next < lens.length || active.some((a) => a)) {
      for (let k = 0; k < slots; k++) if (!active[k] && next < lens.length) active[k] = { r: next++, done: 0 };
      for (let k = 0; k < slots; k++) { const a = active[k]; if (!a) continue; if (++a.done === lens[a.r]) { finish[a.r] = t + 1; active[k] = null; } }
      t++;
    }
  }
  return { T: t, util: lens.reduce((x, y) => x + y, 0) / (t * slots), mean: finish.reduce((x, y) => x + y, 0) / lens.length };
}
{
  const first = [5, 2, 8, 3];
  claim("serve-static", `of its 32 slot-steps, ${first.reduce((a, b) => a + b, 0)} produce tokens, ${Math.round((100 * 18) / 32)}%`);
  near("first batch span", Math.max(...first), 8);
  const L = [5, 2, 8, 3, 6, 2, 4, 7, 3, 5], a = batchSim(L, 4, "static"), b = batchSim(L, 4, "continuous");
  claim("serve-continuous", `finishes in ${b.T} steps instead of ${a.T}, keeps ${Math.round(100 * b.util)}% of the slots busy instead of ${Math.round(100 * a.util)}%, and returns the average request at step ${fx(b.mean, 1)} instead of ${fx(a.mean, 1)}`);
  claim("serve-continuous", "36.9 times the throughput of FasterTransformer");
  claim("serve-paged", "20.4%, 26.8% and 38.2%"); claim("serve-paged", "96.3%"); claim("serve-paged", "(16 by default)"); claim("serve-paged", "2 to 4 times the throughput");
}
{
  // chunked prefill: 64 decodes at 2,048 context + an 8,192-token prompt, Llama 3.1 70B on 8 H100s
  const B = 64, CTX = 2048, P = 8192;
  const iter = (c, o) => {
    const fl = 2 * P70.read * (B + c) + B * attn1(c70) * CTX + attn1(c70) * (c * o + (c * (c + 1)) / 2);
    const by = W70read + (B * CTX + o) * kvTok(c70);
    return { t: Math.max(by / SERVER.bw, fl / SERVER.peak), fl };
  };
  const whole = iter(P, 0), base = step70(B, CTX).t;
  claim("serve-chunked", `needs ${int(whole.fl / 1e12)} TFLOP: ${fx(whole.t * 1e3, 1)} ms at the peak of 8 GPUs`);
  // a 128-token budget: 64 prompt tokens per step
  let o2 = 0, n2 = 0, t2 = 0, reread = 0;
  while (o2 < P) { const ci = Math.min(128 - B, P - o2); t2 += iter(ci, o2).t; reread += (o2 * kvTok(c70)) / SERVER.bw; o2 += ci; n2++; }
  claim("serve-chunked", `at least ${fx(base * 1e3, 2)} ms here. With a 128-token budget the prompt takes ${n2} steps and its first token arrives after ${fx(t2 * 1e3, 1)} ms, of which re-reading earlier chunks' caches is only ${fx(reread * 1e3, 1)} ms`);
  ok("the per-step floor dominates the 128-token plan", n2 * base > 0.99 * t2 - reread);
  claim("serve-chunked", `jumps from ${fx(base * 1e3, 2)} ms to ${fx(whole.t * 1e3, 1)} ms`);
  let o = 0, n = 0, worst = 0, ttft = 0; const c = 512 - B;
  while (o < P) { const ci = Math.min(c, P - o); const it = iter(ci, o); worst = Math.max(worst, it.t); ttft += it.t; o += ci; n++; }
  claim("serve-chunked", `the prompt takes ${n} steps of at most ${fx(worst * 1e3, 2)} ms, and its first token arrives after ${fx(ttft * 1e3, 1)} ms instead of ${fx(whole.t * 1e3, 1)} ms`);
  claim("serve-chunked", "2.6 times vLLM's serving capacity for Mistral-7B on one A100 and up to 5.6 times for Falcon-180B");
  claim("serve-disagg", `Its ${int(whole.fl / 1e12)} TFLOP keep 8 H100s busy for ${fx(whole.t * 1e3, 1)} ms`);
  claim("serve-disagg", `instead of the ${fx(base * 1e3, 2)} ms of a decode step alone`);
  claim("serve-disagg", `about 156 ms instead of ${fx(base * 1e3, 1)} ms`);
}
{
  // preemption: independent re-simulation of the stated rules
  function sim(cap, BS = 4) {
    const reqs = [{ n: "A", p: 6, o: 9 }, { n: "B", p: 5, o: 14 }, { n: "C", p: 7, o: 6 }, { n: "D", p: 4, o: 12 }];
    const st = reqs.map((r, i) => ({ ...r, i, g: 0, s: "w", b: 0 })), wq = st.slice(); let run = [], free = cap, t = 0; const ev = [];
    const nb = (x) => Math.ceil(x / BS);
    for (; t < 80 && st.some((r) => r.s !== "d"); t++) {
      for (const r of [...run].sort((x, y) => x.i - y.i)) {
        if (r.s !== "r") continue;
        const add = nb(r.p + r.g + 1) - r.b;
        while (add > free && r.s === "r") { const v = [...run].sort((x, y) => y.i - x.i)[0]; v.s = "w"; free += v.b; v.b = 0; run = run.filter((x) => x !== v); wq.unshift(v); ev.push({ k: "pre", n: v.n, t: t + 1, g: v.g }); }
        if (r.s !== "r") continue;
        free -= add; r.b += add; if (++r.g === r.o) { r.s = "d"; free += r.b; r.b = 0; run = run.filter((x) => x !== r); ev.push({ k: "done", n: r.n, t: t + 1 }); }
      }
      while (wq.length) { const r = wq[0], b = nb(r.p + r.g + 1); if (b > free) break; wq.shift(); free -= b; r.b = b; r.s = "r"; run.push(r); if (r.g) ev.push({ k: "re", n: r.n, t: t + 1, x: r.p + r.g }); if (++r.g === r.o) { r.s = "d"; free += r.b; r.b = 0; run = run.filter((x) => x !== r); } }
    }
    return { steps: t, ev };
  }
  const s11 = sim(11), pre = s11.ev.filter((e) => e.k === "pre"), re = s11.ev.filter((e) => e.k === "re");
  equal("one preemption with 11 blocks", pre.map((e) => [e.n, e.t, e.g]), [["D", 5, 4]]);
  equal("D recomputes 8 at step 6", re.map((e) => [e.n, e.t, e.x]), [["D", 6, 8]]);
  equal("C done at 6", s11.ev.filter((e) => e.k === "done" && e.n === "C").map((e) => e.t), [6]);
  equal("D done at 13", s11.ev.filter((e) => e.k === "done" && e.n === "D").map((e) => e.t), [13]);
  equal("D done at 12 without preemption", sim(16).ev.filter((e) => e.k === "done" && e.n === "D").map((e) => e.t), [12]);
  equal("10 blocks: D preempted twice", sim(10).ev.filter((e) => e.k === "pre").map((e) => e.n), ["D", "D"]);
  ok("13+ blocks: no preemption", [13, 14, 15, 16].every((c) => !sim(c).ev.some((e) => e.k === "pre")));
  ok("12 blocks: preemption", sim(12).ev.some((e) => e.k === "pre"));
  claim("serve-preempt", "At step 5, D, the last to arrive, is preempted after generating 4 tokens. When C finishes at step 6, D is readmitted and recomputes its 8 tokens in one prefill; it finishes at step 13 instead of 12. With 10 blocks, D is preempted twice; with 13 or more, never.");
  // the figure's own simulation must agree
  const fig = window.Deck.slides.find((x) => x.id === "serve-preempt");
  ok("figure registered", !!fig && fig.fig.type === "serve-preempt");
}
{
  const cache = 2048 * kvTok(c70);
  const pf = Math.max((2 * P70.read * 2048 + (attn1(c70) * 2048 * 2049) / 2) / SERVER.peak, W70read / SERVER.bw);
  claim("serve-disagg", `leaves ${GBs(cache, 2)} GB of keys and values. With 8-way tensor parallelism each GPU holds ${fx(cache / 8 / 1e6, 1)} MB of it`);
  claim("serve-disagg", `so the 8 transfers take ${fx((cache / 8 / 50e9) * 1e3, 2)} ms in parallel, against a prefill of at least ${fx(pf * 1e3, 1)} ms`);
  near("400 Gb/s = 50 GB/s", 400e9 / 8, 50e9);
  claim("serve-disagg", "7.4 times more requests than DeepSpeed-MII, or a 12.6 times tighter SLO than vLLM, with over 90% of requests");
  claim("serve-disagg", "1.4 times the throughput at 20% lower cost");
  claim("serve-disagg", "about 5 ms left unhidden");
}

// ================================================================ 8.4 Making decode faster
{
  const ids = M.runningIds;
  const P8 = M.forward(ids.slice(0, 8)).probs, Q8 = M.forward(ids.slice(0, 8), 150).probs;
  const overlap = (i) => P8[i].reduce((a, v, j) => a + Math.min(v, Q8[i][j]), 0);
  claim("serve-speculative", `overlap ${fx(overlap(4), 3)}`);
  claim("serve-speculative", `1 to 9: overlap ${fx(overlap(6), 3)}`);
  const id = M.idOf(" id"), em = M.idOf(" email"), ag = M.idOf(" age"), nm = M.idOf(" name");
  const p = P8[4], q = Q8[4];
  void ag;
  const strip = (h) => h.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
  claim("serve-speculative", strip(`puts ${fx(q[id], 3)} on <code>id</code> and still gives <code>name</code>, the column already selected, ${fx(q[nm], 3)}; the final model gives <code>name</code> ${fx(p[nm], 3)}`));
  const top2 = q.map((v, i) => [v, i]).sort((x, y) => y[0] - x[0]).slice(0, 2).map((x) => V[x[1]]);
  equal("draft's two most likely tokens after WHERE", top2, [" id", " email"]);
  const Ex = (1 - 0.8 ** 5) / 0.2, sp = Ex / (4 * 0.1 + 1);
  claim("serve-speculative", `At α = 0.8, k = 4 and c = 0.1: ${fx(Ex, 2)} tokens per pass, ${fx(sp, 2)} times faster`);
  claim("serve-speculative", "2 to 3 times faster T5-XXL decoding"); claim("serve-speculative", "2 to 2.5 times for Chinchilla 70B");
  claim("serve-spec-exact", `<code>id</code> ${fx(p[id], 4)}, <code>email</code> ${fx(p[em], 4)}, <code>age</code> ${fx(p[ag], 4)}, <code>name</code> ${fx(p[nm], 4)}. Draft q: ${fx(q[id], 4)}, ${fx(q[em], 4)}, ${fx(q[ag], 4)}, ${fx(q[nm], 4)}`.replace(/<[^>]+>/g, " ").replace(/\s+/g, " "));
  const beta = overlap(4), rej = 1 - beta, res = p.reduce((a, v, j) => a + Math.max(0, v - q[j]), 0);
  near("rejection = residual mass", rej, res, 1e-12);
  claim("serve-spec-exact", `= ${fx(beta, 4)}\\): a draft is accepted ${fx(100 * beta, 2)}% of the time; a rejection happens with probability ${fx(rej, 4)}`);
  claim("serve-spec-exact", `emitted with \\(${fx(q[id], 4)} \\times ${fx(p[id], 4)} / ${fx(q[id], 4)} = ${fx(p[id], 4)}\\)`);
  near("id via acceptance", q[id] * Math.min(1, p[id] / q[id]), p[id], 1e-15);
  const r = (p[em] - q[em]) / rej;
  claim("serve-spec-exact", `After a rejection (${fx(rej, 4)}) it is resampled with probability \\((${fx(p[em], 4)} - ${fx(q[em], 4)}) / ${fx(rej, 4)}\\), which adds \\(${fx(p[em], 4)} - ${fx(q[em], 4)} = ${fx(rej * r, 4)}\\). Total \\(${fx(q[em], 4)} + ${fx(rej * r, 4)} = ${fx(p[em], 4)}\\)`);
  near("email total", Number(fx(q[em], 4)) + Number(fx(rej * r, 4)), Number(fx(p[em], 4)), 1e-9);
  const out = p.map((pi, j) => Math.min(pi, q[j]) + (rej * Math.max(0, pi - q[j])) / res);
  const md = Math.max(...out.map((v, j) => Math.abs(v - p[j])));
  ok(`speculative output equals p (max diff ${md})`, md < 3e-16);
  claim("serve-spec-exact", "\\(3 \\times 10^{-16}\\)");
  const E = (a, k) => (1 - a ** (k + 1)) / (1 - a);
  claim("serve-spec-exact", `(1 − 0.8⁵)/(1 − 0.8) = ${fx(1 - 0.8 ** 5, 3)}/0.2 = ${fx(E(0.8, 4), 2)}`);
  claim("serve-spec-exact", `The ${fx(E(0.8, 4) - 1, 2)} counts only accepted drafts (0.8 + 0.64 + 0.512 + ${fx(0.8 ** 4, 2)})`);
}
{
  claim("serve-medusa", "Over 2.2× faster"); claim("serve-medusa", "2.3 to 2.8× trained jointly"); claim("serve-medusa", "2.7 to 3.5× lower latency on LLaMA2-Chat 70B");
  // quantization of our model with per-output-channel and per-matrix scales (independent implementation)
  const W0 = M.weights();
  const quant = (bits, perCh) => {
    const out = {}, qmax = 2 ** (bits - 1) - 1;
    for (const [k, t] of Object.entries(W0)) {
      if (t.shape.length < 2) { out[k] = t; continue; }
      const [rr, cc] = t.shape, d = new Float64Array(t.data.length);
      let sc = [];
      if (perCh) for (let j = 0; j < cc; j++) { let m = 0; for (let i = 0; i < rr; i++) m = Math.max(m, Math.abs(t.data[i * cc + j])); sc.push(m / qmax); }
      else { let m = 0; for (const v of t.data) m = Math.max(m, Math.abs(v)); sc = new Array(cc).fill(m / qmax); }
      for (let i = 0; i < rr; i++) for (let j = 0; j < cc; j++) d[i * cc + j] = Math.max(-qmax, Math.min(qmax, Math.round(t.data[i * cc + j] / sc[j]))) * sc[j];
      out[k] = { shape: t.shape, data: d };
    }
    return out;
  };
  const loss = (Wt) => M.exactLoss(M.finalStep, { weights: Wt }).mean;
  const base = M.exactLoss().mean;
  claim("serve-quant", `give losses ${fx(loss(quant(8, true)), 4)}, ${fx(loss(quant(6, true)), 4)} and ${fx(loss(quant(5, true)), 4)} (unquantized ${fx(base, 4)})`);
  const kvB = 64 * 8192 * kvTok(c70);
  claim("serve-quant", `4-bit weights move the bound only from ${fx(((W70read + kvB) / SERVER.bw) * 1e3, 2)} to ${fx(((P70.read / 2 + kvB) / SERVER.bw) * 1e3, 2)} ms`);
  const q4 = quant(4, true), tr4 = M.forward(M.runningIds.slice(0, 8), M.finalStep, { weights: q4 }), pr = tr4.probs[2], top = pr.indexOf(Math.max(...pr));
  equal("4-bit top token after FROM", V[top], " FROM");
  ok("4-bit p(FROM) rounds to 1.000", pr[top] >= 0.9995);
  ok("4-bit p(users) rounds to 0.000", pr[M.idOf(" users")] < 0.0005);
  claim("serve-quant", "it predicts <code>FROM</code> again with p = 1.000".replace(/<[^>]+>/g, " ").replace(/\s+/g, " "));
  claim("serve-quant", `fails already at 5 bits (loss ${fx(loss(quant(5, false)), 4)})`);
  // the legacy widget's own quantizer agrees with this one
  const wq = window.Widgets._quantizeModel(W0, 4, true);
  ok("widget quantizer = ours", Object.keys(wq).every((k) => wq[k].data.every((v, i) => v === q4[k].data[i])));
  claim("serve-quant", `${GBs(W70)} GB in BF16, ${GBs(P70.total)} GB in 8 bits, ${GBs(P70.total / 2)} GB in 4 bits`);
  claim("serve-quant", `at batch 1 the decode bound, counting the integers alone, drops from ${fx((W70read / SERVER.bw) * 1e3, 2)} ms to ${fx((P70.read / SERVER.bw) * 1e3, 2)} and ${fx((P70.read / 2 / SERVER.bw) * 1e3, 2)} ms`);
  claim("serve-quant", "AWQ's 128 weights per scale");
}
{
  // GPTQ vs RTN: independent implementation (block size 1 form of GPTQ Algorithm 1)
  const W0 = M.weights();
  const calRand = M.mulberry32(5);
  const Q = Array.from({ length: 32 }, () => ({ ids: M.encode(M.sampleQuery(calRand)).map(M.idOf).slice(0, 8), p: 1 / 32 }));
  const mats = ["l0.wq", "l0.wk", "l0.wv", "l0.wo", "l0.w1", "l0.w2", "l1.wq", "l1.wk", "l1.wv", "l1.wo", "l1.w1", "l1.w2"];
  const clone = (Wt) => Object.fromEntries(Object.entries(Wt).map(([k, t]) => [k, { shape: t.shape, data: Float64Array.from(t.data) }]));
  const scales = (t, qmax) => Array.from({ length: t.shape[1] }, (_, j) => { let m = 0; for (let i = 0; i < t.shape[0]; i++) m = Math.max(m, Math.abs(t.data[i * t.shape[1] + j])); return m / qmax || 1; });
  const rq = (x, sc, qmax) => Math.max(-qmax, Math.min(qmax, Math.round(x / sc))) * sc;
  const rtn = (t, bits) => { const qmax = 2 ** (bits - 1) - 1, sc = scales(t, qmax); return { shape: t.shape, data: Float64Array.from(t.data, (v, k) => rq(v, sc[k % t.shape[1]], qmax)) }; };
  const inputsOf = (Wt, name) => { const m = name.match(/^l(\d)\.(\w+)$/), X = [], w = []; for (const q of Q) { const L = M.forward(q.ids, M.finalStep, { weights: Wt }).layers[+m[1]]; for (const row of { wq: L.h1, wk: L.h1, wv: L.h1, wo: L.concat, w1: L.h2, w2: L.act }[m[2]]) { X.push(row); w.push(q.p); } } return { X, w }; };
  function chol(A) { const n = A.length, L = A.map(() => new Array(n).fill(0)); for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let a = A[i][j]; for (let k = 0; k < j; k++) a -= L[i][k] * L[j][k]; L[i][j] = i === j ? Math.sqrt(a) : a / L[j][j]; } return L; }
  function inv(A) { // Gauss-Jordan, independent of the figure's Cholesky inverse
    const n = A.length, M2 = A.map((r, i) => r.concat(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))));
    for (let c = 0; c < n; c++) { let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(M2[r][c]) > Math.abs(M2[piv][c])) piv = r; [M2[c], M2[piv]] = [M2[piv], M2[c]]; const d = M2[c][c]; for (let j = 0; j < 2 * n; j++) M2[c][j] /= d; for (let r = 0; r < n; r++) if (r !== c) { const f = M2[r][c]; for (let j = 0; j < 2 * n; j++) M2[r][j] -= f * M2[c][j]; } }
    return M2.map((r) => r.slice(n));
  }
  function gptq(t, X, w, bits) {
    const qmax = 2 ** (bits - 1) - 1, [nin, nout] = t.shape, sc = scales(t, qmax);
    const H = Array.from({ length: nin }, (_, i) => Array.from({ length: nin }, (_, j) => X.reduce((a, x, k) => a + 2 * w[k] * x[i] * x[j], 0)));
    const md = H.reduce((a, r, i) => a + r[i], 0) / nin; for (let i = 0; i < nin; i++) { if (H[i][i] === 0) H[i][i] = 1; H[i][i] += 0.01 * md; }
    const Lc = chol(inv(H)); // H^-1 = L L^T, so U = L^T is its upper Cholesky factor
    const W = Array.from({ length: nin }, (_, i) => Array.from(t.data.subarray(i * nout, (i + 1) * nout)));
    for (let i = 0; i < nin; i++) { const e = W[i].map((v, o) => { const qv = rq(v, sc[o], qmax); W[i][o] = qv; return (v - qv) / Lc[i][i]; }); for (let j = i + 1; j < nin; j++) for (let o = 0; o < nout; o++) W[j][o] -= e[o] * Lc[j][i]; }
    return { shape: t.shape, data: Float64Array.from(W.flat()) };
  }
  const evalW = (Wt) => { const pr = M.forward(M.runningIds.slice(0, 8), M.finalStep, { weights: Wt }).probs[2]; return { loss: M.exactLoss(M.finalStep, { weights: Wt }).mean, pu: pr[M.idOf(" users")] }; };
  const res = {};
  for (const bits of [4, 3]) {
    const R = clone(W0), G = clone(W0);
    for (const n of mats) R[n] = rtn(W0[n], bits);
    for (const n of mats) { const { X, w } = inputsOf(G, n); G[n] = gptq(G[n], X, w, bits); }
    res[bits] = { rtn: evalW(R), gptq: evalW(G) };
    const fig = slide("serve-gptq").fig.run(bits);
    near(`${bits}-bit RTN loss: figure vs script`, fig.rtn.loss, res[bits].rtn.loss, 1e-9);
    near(`${bits}-bit GPTQ loss: figure vs script`, fig.gptq.loss, res[bits].gptq.loss, 1e-6);
    near(`${bits}-bit GPTQ p(users): figure vs script`, fig.gptq.pu, res[bits].gptq.pu, 1e-6);
  }
  claim("serve-gptq", `round to nearest gives loss ${fx(res[4].rtn.loss, 4)} and p(users) = ${fx(res[4].rtn.pu, 3)}. GPTQ, calibrated on 32 queries sampled with mulberry32(5) and evaluated on all 270, gives ${fx(res[4].gptq.loss, 4)} and ${fx(res[4].gptq.pu, 3)}`);
  ok("3-bit GPTQ improves on RTN", res[3].gptq.loss < res[3].rtn.loss);
  claim("serve-gptq", "about four GPU hours, at 3 or 4 bits per weight"); claim("serve-gptq", "about 1% of the weights"); claim("serve-gptq", "groups hold 128 weights");
}
{
  const c405 = cfg["405B"], P405c = count(c405);
  claim("serve-fp8", "1,979 TFLOP/s dense"); near("FP8 = 2 × BF16 (dense)", HW.fp8 / HW.bf16, 2, 0.002);
  // Meta's FP8 mix for 405B: attention (all layers), MLPs of the first and last layers, embeddings and output layer in BF16
  const attnL = P405c.attn, mlpL = P405c.mlp;
  const bytes = 126 * attnL * 2 + 124 * mlpL * 1 + 2 * mlpL * 2 + 2 * c405.V * c405.d * 2 + (2 * 126 + 1) * c405.d * 2;
  claim("serve-fp8", `those weights take ${GBs(bytes)} GB, against ${GBs(P405c.total * 2)} GB in BF16`);
  ok("405B FP8 mix fits in 640 GB", bytes < 640e9);
  claim("serve-fp8", "up to 50% higher prefill throughput"); claim("serve-fp8", "capped at 1,200"); claim("serve-fp8", "about half of inference time");
  const lin = 2048 * 2 * P70.read;
  const fp8rows = [["Dense peak per H100, TFLOP/s", "989", "1,979"], ["Llama 3.1 70B weights, GB", GBs(W70), GBs(P70.total)],
    ["70B decode step bound on 8 GPUs, ms", fx((W70read / SERVER.bw) * 1e3, 2), fx((P70.read / SERVER.bw) * 1e3, 2)],
    ["70B prefill of 2,048 tokens, linear layers, ms", fx((lin / SERVER.peak) * 1e3, 1), fx((lin / (8 * HW.fp8)) * 1e3, 1)],
    ["Llama 3.1 405B weights, GB", GBs(P405c.total * 2), GBs(bytes)]];
  equal("FP8 table rows", slide("serve-fp8").fig.rows, fp8rows);
}
{
  claim("serve-kv-quant", `needs ${int(kvTok(c70, 1))} bytes per token`);
  const budget = USABLE(8) - W70;
  claim("serve-kv-quant", `with ${GBs(budget)} GB for the cache, 8,192-token requests that fit go from ${Math.floor(budget / (8192 * kvTok(c70)))} to ${Math.floor(budget / (8192 * kvTok(c70, 1)))}`);
  const kvB = 64 * 8192 * kvTok(c70), kvF = 64 * 8192 * kvTok(c70, 1);
  const tB = Math.max((W70read + kvB) / SERVER.bw, (64 * (2 * P70.read + attn1(c70) * 8192)) / SERVER.peak);
  const tF = Math.max((W70read + kvF) / SERVER.bw, (64 * (2 * P70.read + attn1(c70) * 8192)) / SERVER.peak);
  claim("serve-kv-quant", `reads ${GBs(kvB)} GB of cache in BF16 and ${GBs(kvF)} GB in FP8, so its bound drops from ${fx(tB * 1e3, 2)} ms to ${fx(tF * 1e3, 2)} ms`);
  claim("serve-kv-quant", "2⁻⁴ = 6.25%"); near("E4M3 half-ulp", 2 ** -4, 0.0625);
}
{
  const tr = M.forward(M.runningIds.slice(0, 8)), sc = tr.layers[0].heads[1].scores[7];
  const m1 = Math.max(...sc.slice(0, 4)), m2 = Math.max(...sc.slice(4, 8));
  claim("serve-flashdecoding", `largest score of ${fx(m1, 3)} and the second half ${fx(m2, 3)}`);
  claim("serve-flashdecoding", `e<sup>${fx(m2, 3)} − ${fx(m1, 3)}</sup> = ${fx(Math.exp(m2 - m1), 3)}`.replace(/<[^>]+>/g, " ").replace(/\s+/g, " "));
  // split-and-merge equals full attention for every head and every split
  let worst = 0;
  for (const [l, hh] of [[0, 0], [0, 1], [1, 0], [1, 1]]) for (const parts of [1, 2, 4, 8]) {
    const s7 = tr.layers[l].heads[hh].scores[7], Vv = tr.layers[l].v.map((r) => r.slice(hh * 8, hh * 8 + 8)), size = 8 / parts;
    const ch = Array.from({ length: parts }, (_, c) => { const js = Array.from({ length: size }, (_, k) => c * size + k); const m = Math.max(...js.map((j) => s7[j])); const e = js.map((j) => Math.exp(s7[j] - m)); const l2 = e.reduce((a, b) => a + b, 0); return { m, l: l2, o: Array.from({ length: 8 }, (_, d) => js.reduce((a, j, k) => a + e[k] * Vv[j][d], 0) / l2) }; });
    const Mx = Math.max(...ch.map((c) => c.m)), wts = ch.map((c) => Math.exp(c.m - Mx) * c.l), Ls = wts.reduce((a, b) => a + b, 0);
    const merged = Array.from({ length: 8 }, (_, d) => ch.reduce((a, c, k) => a + wts[k] * c.o[d], 0) / Ls);
    const att = tr.layers[l].heads[hh].att[7], full = Array.from({ length: 8 }, (_, d) => att.reduce((a, x, j) => a + x * Vv[j][d], 0));
    worst = Math.max(worst, ...merged.map((x, d) => Math.abs(x - full[d])));
  }
  ok(`split attention = full attention (max diff ${worst})`, worst < 1e-12);
  claim("serve-flashdecoding", "up to 8 times faster generation"); claim("serve-flashdecoding", "up to 50 times faster");
  claim("serve-flashdecoding", `at batch 1 has ${64 / 8} heads per GPU, ${64 / 8} thread blocks for 132 SMs`);
  const l1 = sc.slice(0, 4).reduce((a, x) => a + Math.exp(x - m1), 0), l2 = sc.slice(4, 8).reduce((a, x) => a + Math.exp(x - m2), 0);
  const w1 = l1, w2 = Math.exp(m2 - m1) * l2;
  claim("serve-flashdecoding", `(${fx(l1, 3)} and ${fx(l2, 3)}) the halves contribute ${Math.round((100 * w1) / (w1 + w2))}% and ${Math.round((100 * w2) / (w1 + w2))}% of the result`);
}
{
  const l70 = (count(c70).layer * 2) / 8, l8 = (count(cfg["8B"]).layer * 2) / 8;
  claim("serve-cuda-graphs", `reads ${fx(l70 / 1e6, 1)} MB of weights per GPU per layer, ${fx((l70 / HW.bw) * 1e6, 1)} µs of HBM time; Llama 3.1 8B on 8 GPUs, ${fx(l8 / 1e6, 1)} MB and ${fx((l8 / HW.bw) * 1e6, 1)} µs`);
  const sizes = [1, 2, 4].concat(Array.from({ length: 31 }, (_, i) => 8 * (i + 1)), Array.from({ length: 17 }, (_, i) => 256 + 16 * i));
  // vLLM v0.32: [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 512 + 1, 16))
  const py = [1, 2, 4]; for (let x = 8; x < 256; x += 8) py.push(x); for (let x = 256; x <= 512; x += 16) py.push(x);
  equal("capture sizes", sizes, py);
  const g = (b) => py.find((c) => c >= b);
  claim("serve-cuda-graphs", `a batch of 13 runs in the graph for ${g(13)}, leaving ${g(13) - 13} slots empty, and a batch of 260 in the graph for ${g(260)}`);
  equal("no graph above 512", g(513), undefined);
  claim("serve-cuda-graphs", `The bound is ${GBs(W70read)} GB of weight reads plus ${GBs(2048 * kvTok(c70), 2)} GB of cache, at 26.8 TB/s`);
  claim("serve-cuda-graphs", `An FP8 cache saves ${GBs(1024 * kvTok(c70), 2)} GB of ${GBs(W70read + 2048 * kvTok(c70))}`);
}

// ================================================================ 8.5 Parallelism for inference
{
  for (const [n, t] of [[2, "20.7"], [4, "10.4"]]) claim("serve-tp", `on ${n} at least ${fx((W70read / n / HW.bw) * 1e3, 1)} ms`.replace("on 2 at", "on 2 GPUs a step takes at"));
  ok("70B BF16 does not fit one H100", W70 > 80e9);
  claim("serve-tp", `on 8 at least ${fx((W70read / 8 / HW.bw) * 1e3, 2)} ms`);
  claim("serve-tp", `${int(c70.d * 2)} bytes per request`);
  claim("serve-tp", `takes ${Math.round(((2 * 7) / 8) * 16384 / 450e9 * 1e9)} ns`);
  claim("serve-tp", `2 per layer, ${2 * c70.L} per decode step`);
  claim("serve-tp", `the 160 calls would add ${fx(160 * 10e-6 * 1e3, 1)} ms to the ${fx((W70read / SERVER.bw) * 1e3, 2)} ms step`);
  near("NVLink each way", 900e9 / 2, 450e9);
}
{
  const c = cfg["405B"], cc = count(c);
  const s1 = (63 * cc.layer * 2) / 8 / HW.bw, s2 = ((63 * cc.layer + cc.head + c.d) * 2) / 8 / HW.bw;
  claim("serve-pp", `${GBs(cc.total * 2)} GB of BF16 weights`); ok("405B BF16 > 640 GB", cc.total * 2 > 640e9);
  claim("serve-pp", `reads about ${GBs((cc.read * 2) / 16)} GB per token: ${fx(s1 * 1e3, 2)} ms for the first stage and ${fx(s2 * 1e3, 2)} ms for the second`);
  claim("serve-pp", `That is ${fx((s1 + s2) * 1e3, 2)} ms through both, at most ${Math.floor(1 / (s1 + s2))} tokens per second`);
  claim("serve-pp", `${fx(s1 * 1e3, 2)} ms for layers 1–63, ${fx(s2 * 1e3, 2)} ms for layers 64–126`);
  near("63 + 63 = 126 layers", 63 * 2, c.L);
  ok("timeline fits 91 ms window for 3 tokens", 3 * (s1 + s2) * 1e3 < 91);
}
{
  claim("serve-ep", "671B parameters, 37B active per token"); claim("serve-ep", "256 routed experts, a token uses 8 of them plus 1 shared expert");
  claim("serve-ep", "8/256 = 1/32"); near("8/256", 8 / 256, 1 / 32);
  claim("serve-ep", "units of 4 nodes, 32 GPUs"); claim("serve-ep", "units of 40 nodes, 320 GPUs"); claim("serve-ep", "with 64 GPUs holding redundant copies");
  near("40 nodes × 8", 40 * 8, 320); near("4 nodes × 8", 4 * 8, 32); near("256 experts + 64", 256 + 64, 320);
}
{
  const room = (n, wb) => USABLE(1) - (P70.total * wb) / n;
  claim("serve-replica", `on 2, with 92% of the reported 79.32 GiB usable, ${fx(room(2, 2) / 1e9, 2)} GB per GPU remain for the cache: ${Math.floor(room(2, 2) / ((kvTok(c70) / 2) * 2048))} requests of 2,048 tokens`);
  claim("serve-replica", `On 4 GPUs, ${fx(room(4, 2) / 1e9, 2)} GB per GPU: ${Math.floor(room(4, 2) / ((kvTok(c70) / 4) * 2048))} such requests`);
  claim("serve-replica", `fits on one GPU at ${fx((P8.read * 2) / HW.bw * 1e3, 2)} ms per step`);
  ok("70B FP8 fits on one GPU", room(1, 1) > 0);
  claim("serve-replica", `with room for ${Math.floor(room(1, 1) / (kvTok(c70) * 2048))} such requests, at ${fx((P70.read / HW.bw) * 1e3, 2)} ms per step; on 2 GPUs the bound is ${fx((P70.read / 2 / HW.bw) * 1e3, 2)} ms, on 4, ${fx((P70.read / 4 / HW.bw) * 1e3, 2)} ms`);
  // wider replicas: requests that fit at 2,048 tokens and tokens/s per GPU at that batch
  const fitB = (n) => Math.floor(room(n, 2) / ((kvTok(c70) / n) * 2048));
  const stepN = (n, B) => Math.max((W70read + B * 2048 * kvTok(c70)) / (n * HW.bw), (B * (2 * P70.read + attn1(c70) * 2048)) / (n * HW.bf16));
  const b4 = fitB(4), b8 = fitB(8);
  claim("serve-replica", `4 GPUs fit ${b4} requests and 8 fit ${b8}; at those batches the bound gives ${int(b4 / stepN(4, b4) / 4)} and ${int(b8 / stepN(8, b8) / 8)} tokens/s per GPU`);
  // 405B FP8 as released: attention, first and last MLPs, embeddings and output layer in BF16
  const c4 = cfg["405B"], cc4 = count(c4);
  const mix = 126 * cc4.attn * 2 + 124 * cc4.mlp + 2 * cc4.mlp * 2 + 2 * c4.V * c4.d * 2 + (2 * 126 + 1) * c4.d * 2;
  near("405B FP8 mix bytes (GB)", mix / 1e9, 487.2, 0.05);
  near("405B FP8 mix: BF16 params (B)", (126 * cc4.attn + 2 * cc4.mlp + 2 * c4.V * c4.d + (2 * 126 + 1) * c4.d) / 1e9, 81.3, 0.05);
  near("405B FP8 mix: FP8 params (B)", (124 * cc4.mlp) / 1e9, 324.5, 0.05);
  claim("serve-replica", "Meta's released mix for 405B (487.2 GB)");
  const out = slide("serve-replica").fig.outputs, v405 = { m: "405B", wb: 1, n: 8, lc: 11, lb: 5 };
  near("replica calc: 405B FP8 weights per GPU", out[0].f(v405), mix / 8, 1);
  equal("replica calc: 405B FP8 requests at 2,048 on 8 GPUs", out[1].f(v405), Math.floor((USABLE(1) - mix / 8) / ((kvTok(c4) / 8) * 2048)));
  equal("replica calc: 70B BF16 on 4 GPUs", out[1].f({ m: "70B", wb: 2, n: 4, lc: 11, lb: 5 }), b4);
}
{
  const rand = M.mulberry32(28), R = 4, N = 32;
  const lens = Array.from({ length: N }, () => 32 * 2 ** Math.floor(rand() * 6));
  const assign = (pol) => { const load = new Array(R).fill(0); lens.forEach((L, i) => { const k = pol === "rr" ? i % R : load.indexOf(Math.min(...load)); load[k] += L; }); return load; };
  const tot = lens.reduce((a, b) => a + b, 0), mean = tot / R, rr = assign("rr"), ll = assign("ll");
  claim("serve-lb", `busiest replica ${int(Math.max(...rr))} tokens to generate, ${Math.round(100 * (Math.max(...rr) / mean - 1))}% above an even share of ${int(mean)} and ${fx(Math.max(...rr) / Math.min(...rr), 1)} times the least busy one`);
  claim("serve-lb", `cuts the busiest to ${int(Math.max(...ll))}, ${Math.round(100 * (Math.max(...ll) / mean - 1))}% above`);
  // medians over 40 draws
  const meds = [[], []];
  for (let seed = 1; seed <= 40; seed++) {
    const rd = M.mulberry32(seed), L2 = Array.from({ length: N }, () => 32 * 2 ** Math.floor(rd() * 6)), m2 = L2.reduce((a, b) => a + b, 0) / R;
    const as = (pol) => { const load = new Array(R).fill(0); L2.forEach((L, i) => { const k = pol === "rr" ? i % R : load.indexOf(Math.min(...load)); load[k] += L; }); return Math.max(...load) / m2 - 1; };
    meds[0].push(as("rr")); meds[1].push(as("ll"));
  }
  const median = (a) => { const b = a.slice().sort((x, y) => x - y); return (b[19] + b[20]) / 2; };
  claim("serve-lb", `Over 40 random draws, the medians are ${Math.round(100 * median(meds[0]))}% and ${Math.round(100 * median(meds[1]))}%`);
  const s64 = step70(64, 2048).t, s128 = step70(128, 2048).t;
  claim("serve-lb", `a 70B step for 128 requests takes ${fx(s128 * 1e3, 2)} ms against ${fx(s64 * 1e3, 2)} ms for 64, so every one of its requests streams ${Math.round(100 * (s128 / s64 - 1))}% slower`);
  claim("serve-lb", `Each decode step takes ${fx(s128 * 1e3, 2)} ms instead of ${fx(s64 * 1e3, 2)} ms, so their tokens arrive ${Math.round(100 * (s128 / s64 - 1))}% slower`);
  claim("serve-lb", `128 caches of ${GBs(2048 * kvTok(c70), 2)} GB, ${GBs(W70read + 128 * 2048 * kvTok(c70))} GB in all, ${fx(s128 * 1e3, 2)} ms at 26.8 TB/s; with 64 requests it reads ${GBs(W70read + 64 * 2048 * kvTok(c70))} GB in ${fx(s64 * 1e3, 2)} ms`);
  ok("lengths span 32..1024", Math.min(...lens) >= 32 && Math.max(...lens) <= 1024);
  claim("serve-pp", "252 per decode step"); near("126 layers × 2", 126 * 2, 252);
}

// ================================================================ 8.6 Serving systems
{
  equal("prompt ids", ["SELECT", " name", " FROM"].map((t) => M.idOf(t)), [43, 93, 47]);
  equal("users id", M.idOf(" users"), 63); equal("; id", M.idOf(";"), 10);
  const row = slide("serve-path").fig.rows;
  ok("path table has ids 43, 93, 47", row[0][2].includes("43, 93, 47"));
  ok("path table users id 63", row[3][2].includes("id 63") && row[3][2].includes("p = 1.000"));
  ok("3 tokens fit in 1 block of 16", Math.ceil(3 / 16) === 1 && row[1][2] === "3 tokens: 1 block of 16");
}
// ---- grammar-constrained decoding: independent implementation
const GQ = M.allQueries().map((q) => ({ t: M.encode(q.text).map(M.idOf), p: q.p, text: q.text }));
const validSet = new Set(GQ.map((q) => q.t.join(",")));
const trieAllowed = (pre) => { const set = new Set(); for (const q of GQ) if (pre.length < 9 && pre.every((x, i) => q.t[i] === x)) set.add(q.t[pre.length]); return set; };
const soft = (z, tau) => { const m = Math.max(...z.map((v) => v / tau)); const e = z.map((v) => Math.exp(v / tau - m)); const s2 = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s2); };
const lmemo = new Map();
const logitsOf = (pre) => { const k = pre.join(","); if (!lmemo.has(k)) lmemo.set(k, M.forward(pre).logits[pre.length - 1]); return lmemo.get(k); };
function gsample(rand, tau, mask) {
  const out = [M.idOf("SELECT")];
  while (out.length < 9) {
    let p = soft(logitsOf(out), tau);
    if (mask) { const A = trieAllowed(out); let z = 0; for (let i = 0; i < p.length; i++) if (A.has(i)) z += p[i]; p = p.map((v, i) => (A.has(i) ? v / z : 0)); }
    const x = M.sampleFrom(p, rand); out.push(x); if (x === M.idOf(";")) break;
  }
  return out;
}
function grun(tau, mask, n = 1000, seed = 1) { const rand = M.mulberry32(seed); let v = 0; const bad = []; for (let i = 0; i < n; i++) { const t = gsample(rand, tau, mask); if (validSet.has(t.join(","))) v++; else bad.push(t.map((x) => V[x]).join("")); } return { v, bad }; }
{
  // character-level check with the 9-token window: a token is allowed if the text stays a prefix of a valid query
  // and the rest can still be written in the tokens left. Same sets as the token trie at every prefix.
  const minTok = (str) => { const best = new Array(str.length + 1).fill(Infinity); best[0] = 0; for (let i = 0; i < str.length; i++) { if (best[i] === Infinity) continue; for (const tok of V) if (tok.length && str.startsWith(tok, i)) best[i + tok.length] = Math.min(best[i + tok.length], best[i] + 1); } return best[str.length]; };
  const texts = GQ.map((q) => q.text);
  let mismatches = 0, prefixes = 0;
  const seen = new Set();
  for (const q of GQ) for (let n = 1; n < 9; n++) {
    const pre = q.t.slice(0, n), key = pre.join(","); if (seen.has(key)) continue; seen.add(key); prefixes++;
    const str = pre.map((x) => V[x]).join("");
    const charSet = new Set();
    V.forEach((tok, id) => { const s2 = str + tok; const rest = texts.filter((t) => t.startsWith(s2)).map((t) => t.slice(s2.length)); if (rest.some((r) => minTok(r) <= 9 - (n + 1))) charSet.add(id); });
    const tr = trieAllowed(pre);
    if (charSet.size !== tr.size || [...tr].some((x) => !charSet.has(x))) mismatches++;
  }
  ok(`character-level mask = token mask at all ${prefixes} prefixes`, mismatches === 0);
  let maxAllowed = 0; for (const q of GQ) for (let n = 1; n < 9; n++) maxAllowed = Math.max(maxAllowed, trieAllowed(q.t.slice(0, n)).size);
  near("most tokens allowed at any step", maxAllowed, 9);
  claim("serve-grammar", "the 9 column tokens"); claim("serve-grammar", "at most 9 of the 113 tokens remain");
  const z = M.forward(M.runningIds.slice(0, 3)).logits[2];
  claim("serve-grammar", `gives <code>users</code> ${fx(soft(z, 3)[M.idOf(" users")], 3)} at temperature 3 and ${fx(soft(z, 4)[M.idOf(" users")], 3)} at temperature 4`.replace(/<[^>]+>/g, " ").replace(/\s+/g, " "));
  equal("only users allowed after SELECT name FROM", [...trieAllowed(M.runningIds.slice(0, 3))], [M.idOf(" users")]);
}
{
  const full = GQ.map((q) => M.forward(q.t.slice(0, 8)).logits);
  const exactValid = (tau) => GQ.reduce((acc, q, qi) => { let pr = 1; for (let i = 1; i < 9; i++) pr *= soft(full[qi][i - 1], tau)[q.t[i]]; return acc + pr; }, 0);
  claim("serve-grammar-rates", `${fx(exactValid(1), 4)} at temperature 1, ${fx(exactValid(2), 4)} at 2, ${fx(exactValid(3), 4)} at 3 and ${fx(exactValid(4), 4)} at 4`);
  const r1 = grun(1, false), r2 = grun(2, false), r3 = grun(3, false);
  claim("serve-grammar-rates", `give ${r1.v} valid queries at temperature 1, ${r2.v} at 2 and ${r3.v} at 3`);
  for (const tau of [0.25, 1, 2, 3, 4]) equal(`masked run at ${tau}: all valid`, grun(tau, true).v, 1000);
  // the figure's own sampler agrees with this one
  const fig = slide("serve-grammar-rates").fig;
  for (const tau of [1, 2, 3]) equal(`figure sampler = script sampler at ${tau}`, fig.run(tau, false).valid, grun(tau, false).v);
  near("figure exact = script exact at 3", fig.exact(3), exactValid(3), 1e-12);
  const bads = r1.bad.concat(r2.bad, r3.bad);
  ok("an invalid sample repeats the selected column", bads.includes("SELECT name FROM users WHERE name = 4;"));
  claim("serve-grammar-rates", "a filter column equal to the selected one"); claim("serve-grammar-rates", "SELECT name FROM users WHERE name = 4;");
  const schema = M.data.schema;
  const parse = (t) => t.match(/^SELECT (\w+) FROM (\w+) WHERE (\w+) = (\w+);$/);
  ok("an invalid sample uses another table's column", bads.some((t) => { const m = parse(t); return m && schema[m[2]] && !schema[m[2]].includes(m[1]) === false && !schema[m[2]].includes(m[3]); }));
  ok("an invalid sample has a table name where a digit belongs", bads.some((t) => { const m = parse(t); return m && Object.keys(schema).includes(m[4]); }));
  // total-variation distance of masked sampling from the grammar
  const allowedMemo = new Map(); const al = (pre) => { const k = pre.join(","); if (!allowedMemo.has(k)) allowedMemo.set(k, trieAllowed(pre)); return allowedMemo.get(k); };
  const tv = (tau) => 0.5 * GQ.reduce((acc, q, qi) => { let pm = 1; for (let i = 1; i < 9; i++) { const pr = soft(full[qi][i - 1], tau), A = al(q.t.slice(0, i)); let zz = 0; for (const j of A) zz += pr[j]; pm *= pr[q.t[i]] / zz; } return acc + Math.abs(pm - q.p); }, 0);
  claim("serve-grammar-rates", `at a total-variation distance (½ Σ |p − q| over the queries) of ${fx(tv(3), 3)} from the grammar's distribution, against ${fx(tv(1), 3)} at temperature 1`);
}
{
  claim("serve-json", "all 128,256 tokens of Llama 3"); claim("serve-json", "1,134 of the 128K tokens, under 1%"); ok("1,134 / 128,256 < 1%", 1134 / 128256 < 0.01);
  claim("serve-json", "up to 100 times lower per-token cost");
}
{
  const price = 2, perHour = price * 8;
  claim("serve-cost", `costs $${fx(perHour, 2)} per hour`);
  const s64 = step70(64, 2048), tps = 64 / s64.t;
  claim("serve-cost", `gives ${int(tps)} tokens/s, ${fx((tps * 3600) / 1e6, 1)} million per hour: $${fx((perHour / 3600 / tps) * 1e6, 2)} per million output tokens`);
  const s1 = step70(1, 2048), tps1 = 1 / s1.t;
  claim("serve-cost", `At batch 1, ${int(tps1)} tokens/s: $${fx((perHour / 3600 / tps1) * 1e6, 2)} per million`);
  const pf = SERVER.peak / (2 * P70.read);
  claim("serve-cost", `${fx((2 * P70.read) / 1e9, 1)} GFLOP each. At the peak of 8 GPUs that is ${int(pf)} tokens/s, $${fx((perHour / 3600 / pf) * 1e6, 3)} per million`);
  const ratio = ((perHour / 3600 / pf) * 1e6) / ((perHour / 3600 / tps) * 1e6);
  ok(`a prompt token costs about a sixth of an output token (${ratio.toFixed(3)})`, Math.round(1 / ratio) === 6);
  // the figure's helper agrees
  const sp = slide("serve-cost").fig;
  ok("cost calc defaults", sp.inputs.find((i) => i.id === "price").value === 2 && sp.inputs.find((i) => i.id === "lb").value === 6 && sp.inputs.find((i) => i.id === "lc").value === 11);
}
{
  // capacity planning: steady state with chunked prefill, Llama 3.1 70B on 8 H100s, P = 2,048, O = 256, TPOT 50 ms
  const P = 2048, O = 256, ctx = P + O / 2, room = USABLE(8) - W70;
  const stepT = (B) => { const c = (B * P) / O; const fl = 2 * P70.read * (B + c) + B * attn1(c70) * ctx + c * attn1(c70) * (P / 2); const by = W70read + B * ctx * kvTok(c70); return Math.max(by / SERVER.bw, fl / SERVER.peak); };
  let B = 0; while (stepT(B + 1) <= 0.05 && (B + 1) * (P + O) * kvTok(c70) <= room) B++;
  const t = stepT(B), mu = B / (O * t), reps = Math.ceil(50 / mu);
  claim("serve-slo", `that is ${P / O} prompt tokens per decode`);
  ok("compute-bound at the planned batch", (2 * P70.read * (B + (B * P) / O)) / SERVER.peak > (W70read + B * ctx * kvTok(c70)) / SERVER.bw);
  claim("serve-slo", `${B} requests per step take ${fx(t * 1e3, 2)} ms. Each replica completes ${B} ÷ (256 × ${fx(t * 1e3, 2)} ms) = ${fx(mu, 1)} requests/s, so 50 requests/s need ${reps} replicas, ${8 * reps} GPUs`);
}
{
  claim("serve-fleet", "prefill on units of 32 GPUs and decode on units of 320");
  // the consolidating request
  const B = 64, CTX = 2048, P = 2048;
  const iter = (c, o) => Math.max((W70read + (B * CTX + o) * kvTok(c70)) / SERVER.bw, (2 * P70.read * (B + c) + B * attn1(c70) * CTX + attn1(c70) * (c * o + (c * (c + 1)) / 2)) / SERVER.peak);
  let o = 0, n = 0, ttft = 0, worst = 0; while (o < P) { const ci = Math.min(512 - B, P - o); const t = iter(ci, o); ttft += t; worst = Math.max(worst, t); o += ci; n++; }
  let dec = 0, first = 0, last = 0;
  for (let i = 1; i <= 255; i++) { const t = Math.max((W70read + (64 * 2048 + 2048 + i) * kvTok(c70)) / SERVER.bw, (2 * P70.read * 65 + attn1(c70) * (64 * 2048 + 2048 + i)) / SERVER.peak); if (i === 1) first = t; last = t; dec += t; }
  claim("serve-timeline", `enters in ${n} chunks of up to ${512 - B} tokens`);
  { const parts = []; let oo = 0; while (oo < P) { const ci = Math.min(512 - B, P - oo); const tm = (W70read + (B * CTX + oo) * kvTok(c70)) / SERVER.bw, tc = (2 * P70.read * (B + ci) + B * attn1(c70) * CTX + attn1(c70) * (ci * oo + (ci * (ci + 1)) / 2)) / SERVER.peak; parts.push({ ci, tm, tc }); oo += ci; }
    ok("first four chunks compute-bound, last memory-bound", parts.slice(0, 4).every((x) => x.tc > x.tm) && parts[4].tc < parts[4].tm);
    claim("serve-timeline", `The four 448-token steps are compute-bound (${fx(parts[0].tc * 1e3, 2)} to ${fx(parts[3].tc * 1e3, 2)} ms); the last, with ${parts[4].ci} prompt tokens, is bound by its ${fx(parts[4].tm * 1e3, 2)} ms of reads`);
    equal("timeline limit column", slide("serve-timeline").fig.rows[0][3], "compute; last chunk HBM reads"); }
  claim("serve-timeline", `The first token arrives after ${fx(ttft * 1e3, 1)} ms`);
  claim("serve-timeline", `255 decode steps of ${fx(first * 1e3, 2)} ms each`);
  ok("decode step barely grows", fx(first * 1e3, 2) === fx(last * 1e3, 2) || Math.abs(last - first) < 2e-5);
  claim("serve-timeline", `complete after ${fx(ttft + dec, 2)} s`);
  const rows = slide("serve-timeline").fig.rows;
  equal("timeline table", rows.map((r) => r.slice(0, 3)), [["Prefill in chunks, beside 64 decodes", String(n), `${fx(ttft * 1e3, 1)} ms`], ["First token (TTFT)", "", `${fx(ttft * 1e3, 1)} ms`], ["Decode, batch of 65", "255", `${fx(dec, 2)} s`], ["Time per output token", "", `${fx(first * 1e3, 2)} ms → ${fx(last * 1e3, 2)} ms`], ["Answer complete", String(n + 255), `${fx(ttft + dec, 2)} s`]]);
  // the check: whole prompt in one step (budget 2,112)
  const one = iter(P, 0);
  claim("serve-timeline", `a 2,112-token budget the prompt runs in one step of about ${Math.round(one * 1e3)} ms instead of 5 chunked steps totalling ${fx(ttft * 1e3, 1)} ms`);
  claim("serve-timeline", `waits about ${Math.round(one * 1e3)} ms for that step's token instead of about ${Math.round(worst * 1e3)} ms`);
  near("2,048 + 64 = 2,112", 2048 + 64, 2112);
}

done();
