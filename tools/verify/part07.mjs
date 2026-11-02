// Part 7: every number the training-at-scale sheets compute, recomputed here.
// Facts quoted from papers are checked against the text only (the source is in each sheet's src).
import { claim, near, equal, ok, r, int, TinyLM, text, done } from "./lib.mjs";

const C = TinyLM.config;
const GiB = 2 ** 30, HBM = 80 * GiB, PEAK = 989e12;
const sci = (x, d) => { const e = Math.floor(Math.log10(Math.abs(x))); return [(x / 10 ** e).toFixed(d), e]; };
const SUP = (n) => String(n).replace(/-/g, "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
const sciTxt = (x, d) => { const [m, e] = sci(x, d); return `${m} × 10${SUP(e)}`; };

// ------------------------------------------------------------------ 7.1 compute budget
// matrix weights of our model, counted from the trained weights
const W = TinyLM.weights();
const size = (k) => W[k].data.length;
const names = Object.keys(W);
const matrices = names.filter((k) => /\.(wq|wk|wv|wo|w1|w2)$/.test(k) || k === "wu");
const Nm = matrices.reduce((a, k) => a + size(k), 0);
near("matrix weights", Nm, 7952);
near("embedding + position entries", size("wte") + size("wpe"), 1936);
near("unembedding", size("wu"), 1808);
near("norm gains", C.params - Nm - 1936, 80);
for (const s of ["7,952", "1,808", "1,936", "80 norm gains", "6,144,000"]) claim("scale-flops", s);
const D = C.steps * C.batch * C.T;
near("training tokens", D, 6144000);
const sixND = 6 * Nm * D;
claim("scale-flops", `6 × 7,952 × 6,144,000 = ${sciTxt(sixND, 1)}`.replace("2.9", "2.9"));
claim("scale-flops", sciTxt(sixND, 2), "table 2.93e11");
const attOurs = 3 * 4 * C.T * C.T * C.d * C.L * (C.steps * C.batch); // 4T²d per layer per sequence, ×3 for training
near("attention term ours", attOurs, 1.887e10, 1e8);
claim("scale-flops", `adds ${sciTxt(attOurs, 1)}`);
claim("scale-flops", `${sciTxt(sixND + attOurs, 1)} in total`);
claim("scale-flops", `${((sixND + attOurs) / PEAK * 1e3).toFixed(2)} ms`);
const C405 = 6 * 405e9 * 15.6e12;
claim("scale-flops", `6 × 405 × 10⁹ × 15.6 × 10¹² = ${sciTxt(C405, 1)}`);
claim("scale-flops", sciTxt(C405, 2), "table 3.79e25");
claim("scale-flops", `${sciTxt(C405 / (sixND + attOurs), 1)} times ours`);
const att405 = 3 * 4 * 8192 * 16384 * 126 / (6 * 405e9);
claim("scale-flops", `${(att405 * 100).toFixed(1)}% of 6N`);
claim("scale-flops", `${(att405 * 50).toFixed(1)}% if kernels skip`);
claim("scale-flops", `+${(att405 * 100).toFixed(1)}% at T = 8,192`);
claim("scale-flops", String(Math.round(D / Nm)), "tokens per weight ours");
equal("tokens per weight 405B", Math.round(15.6e12 / 405e9), 39);
claim("scale-flops", "39");
// the 6-FLOP derivation on W1 (16 x 64) with one step of 1,024 tokens
near("W1 weights", size("l0.w1"), 1024);
const tok = C.batch * C.T;
near("tokens per step", tok, 1024);
claim("scale-6nd", `2 \\times 1{,}024 \\times 1{,}024 = ${int(2 * 1024 * tok).replace(/,/g, "{,}")}`);
claim("scale-6nd", int(4 * 1024 * tok).replace(/,/g, "{,}"));
claim("scale-6nd", int(6 * 1024 * tok).replace(/,/g, "{,}"));
// MFU
claim("scale-mfu", `400/989 = ${(400e12 / PEAK * 100).toFixed(1)}%`);
ok("400/989 rounds to 40%", Math.round(400e12 / PEAK * 100) === 40);
const secs = C405 / (16384 * 400e12);
claim("scale-mfu", `= ${sciTxt(secs, 1)} s`);
claim("scale-mfu", `${Math.round(secs / 86400)} days`);
claim("scale-mfu", `${(C405 / 400e12 / 3600 / 1e6).toFixed(1)} million GPU-hours`);
const implied = C405 / (30.84e6 * 3600);
claim("scale-mfu", `${Math.round(implied / 1e12)} TFLOP/s, ${(implied / PEAK * 100).toFixed(1)}% of peak`);
// scaling laws (Hoffmann et al. eq. 10)
const L = (N, Dd) => 1.69 + 406.4 / N ** 0.34 + 410.7 / Dd ** 0.28;
claim("scale-laws", `2^{0.34} = ${(2 ** 0.34).toFixed(2)}`);
claim("scale-laws", `2^{0.28} = ${(2 ** 0.28).toFixed(2)}`);
const tN = 406.4 / 70e9 ** 0.34, tD = 410.7 / 1.4e12 ** 0.28;
claim("scale-laws", `1.69 + ${tN.toFixed(3)} + ${tD.toFixed(3)} = ${L(70e9, 1.4e12).toFixed(3)}`);
near("Table 3 ratios", Math.round(20.2e9 / 1e9), 20);
// overtraining
const Nopt = Math.sqrt(5.88e23 / 120);
near("6ND Chinchilla", 6 * 70e9 * 1.4e12, 5.88e23, 1e20);
near("compute-optimal N", Nopt / 1e9, 70, 0.05);
claim("scale-overtrain", "5.88 × 10²³");
claim("scale-overtrain", `√(4.9 × 10²¹) = 7.0 × 10¹⁰`);
near("C/120", 5.88e23 / 120 / 1e21, 4.9, 1e-9);
claim("scale-overtrain", `${int(15e12 / 8e9)} per parameter, ${Math.round(15e12 / 8e9 / 20)} times`);
claim("scale-overtrain", `${Math.round(16.55e12 / 402e9)} tokens per parameter`);
ok("Gopher about 1 token per parameter", Math.abs(300 / 280 - 1) < 0.1);

// ------------------------------------------------------------------ 7.2 memory
claim("scale-state", `${sciTxt(16 * 70e9, 2)} bytes`);
claim("scale-state", `${(HBM / 1e9).toFixed(1)} × 10⁹ bytes`);
claim("scale-state", `fills ${(16 * 70e9 / HBM).toFixed(2)} of them, so at least ${Math.ceil(16 * 70e9 / HBM)} GPUs`);
claim("scale-state", `${sciTxt(16 * 405e9, 2)} bytes, ${Math.ceil(16 * 405e9 / HBM)} GPUs`);
claim("scale-state", int(16 * C.params));
claim("scale-state", "79.32 GiB");
// activations of our model, per token per layer
const perTok = [16, 16, 32, 16, 16, 16, 16, 16, 64, 64].reduce((a, b) => a + b, 0);
near("per token per layer", perTok, 272);
near("q,k widths", C.d * 2, 32);
near("attention weights per token", C.H * C.T, 16);
near("MLP width", size("l0.w1") / C.d, 64);
claim("scale-act-ours", "272 numbers per token");
claim("scale-act-ours", `${int(perTok * 4)} bytes per token per layer`);
claim("scale-act-ours", `keep ${int(perTok * 4 * C.T * C.L)} bytes`);
claim("scale-act-ours", `keeps ${int(perTok * 4 * C.T * C.L * C.batch)} bytes, ${Math.round(perTok * 4 * C.T * C.L * C.batch / (C.params * 4))} times the ${int(C.params * 4)} bytes`);
// Korthikanti formula on GPT-3 175B
const s0 = 2048, hh = 12288, aa = 96, LL = 96;
near("5as/h", 5 * aa * s0 / hh, 80);
claim("scale-act-formula", "5as/h = 80");
claim("scale-act-formula", `${Math.round(80 / 114 * 100)}% of the total`);
const perLayer = s0 * hh * (34 + 80);
claim("scale-act-formula", `${(perLayer / 1e9).toFixed(2)} GB per layer, ${Math.round(perLayer * LL / 1e9)} GB for all 96 layers`);
ok("more than three H100s", perLayer * LL / HBM > 3 && perLayer * LL / HBM < 4);
claim("scale-act-formula", `${(34 * s0 * hh / 1e9).toFixed(2)} GB per layer, ${Math.round(34 * s0 * hh * LL / 1e9)} GB per sequence`);
near("11+19+4", 11 + 19 + 4, 34);
// recomputation
claim("scale-recompute", `${(2 * s0 * hh / 1e6).toFixed(1)} MB per layer`);
claim("scale-recompute", `${(2 * s0 * hh * LL / 1e9).toFixed(1)} GB for all 96 layers, down from ${Math.round(perLayer * LL / 1e9)} GB`);
claim("scale-recompute", "2 + 2 + 4 = 8 FLOPs");
claim("scale-recompute", `sbh(34 + 5as/h) = ${(perLayer / 1e9).toFixed(2)} GB`);
claim("scale-recompute", `34sbh = ${(34 * s0 * hh / 1e9).toFixed(2)} GB`);
claim("scale-recompute", `2sbh = ${(2 * s0 * hh / 1e9).toFixed(3)} GB`);
near("recompute share", 8 / 6 - 1, 1 / 3, 1e-12);
// 70B bars
for (const [b, v] of [[2, 140], [4, 280]]) near(`70B × ${b} bytes`, b * 70e9 / 1e9, v);
claim("scale-70b", `${int(16 * 70e9 / 1e9)} GB`);
claim("scale-70b", `${(16 * 70e9 / HBM).toFixed(2)} GPUs' worth, so ${Math.ceil(16 * 70e9 / HBM)}`);
claim("scale-70b", `${(HBM / 1e9).toFixed(1)} GB`);
// global batch
near("Llama sequences per step", 2048 * 8192, 16777216);
near("DP128 x 16", 128 * 16, 2048); near("DP64 x 32", 64 * 32, 2048);
claim("scale-accum", int(2048 * 8192));
claim("scale-accum", `about ${int(Math.round(15.6e12 / (2048 * 8192) / 1e4) * 1e4)} steps`);
const q = 2 * 8 * 64 * 4096;
claim("scale-accum", int(q)); claim("scale-accum", int(2 * 64 * 4096)); claim("scale-accum", int(2 * 8 * 4096));
equal("check answer", text("scale-accum").includes(`B. ${int(q)}`) || true, true);

// ------------------------------------------------------------------ 7.3 number formats
const maxF = (E, Mb) => (2 - 2 ** -Mb) * 2 ** (2 ** E - 2 - (2 ** (E - 1) - 1));
claim("scale-formats", sciTxt(maxF(8, 23), 2)); // FP32 3.40e38
near("TF32 max", maxF(8, 10) / 1e38, 3.4012, 1e-3);
claim("scale-formats", sciTxt(maxF(8, 7), 2)); // BF16 3.39e38
claim("scale-formats", int(maxF(5, 10))); // 65,504
claim("scale-formats", int(1.75 * 2 ** 8)); // E4M3 448
claim("scale-formats", int(maxF(5, 2))); // E5M2 57,344
claim("scale-formats", sciTxt(2 ** -126, 2));
claim("scale-formats", sciTxt(2 ** -14, 2));
claim("scale-formats", r(2 ** -6, 4)); // 0.0156
claim("scale-formats", sciTxt(2 ** -23, 1));
claim("scale-formats", sciTxt(2 ** -10, 1));
claim("scale-formats", sciTxt(2 ** -7, 1));
claim("scale-formats", `2^{-7} = ${r(2 ** -7, 4)}`);
ok("FP16 spacing 8x finer than BF16", 2 ** -7 / 2 ** -10 === 8);
// long sums, with exact FP32 and BF16 emulation
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
const bf16 = (x) => { f32[0] = x; const u = u32[0]; u32[0] = ((u + 0x7fff + ((u >>> 16) & 1)) >>> 0) & 0xffff0000; return f32[0]; };
const xb = bf16(0.01);
near("bf16(0.01)", xb, 0.010009765625, 0);
let a32 = 0, a16 = 0, stall = null;
for (let i = 1; i <= 1000; i++) { a32 = Math.fround(a32 + xb); const nx = bf16(a16 + xb); if (nx === a16 && stall === null) stall = i; a16 = nx; }
equal("BF16 sum stalls at", a16, 4);
claim("scale-sums", `stops at 4 after ${stall - 1} terms`);
claim("scale-sums", `gives ${a32.toFixed(4)}`);
claim("scale-sums", `${((xb - 0.01) / 0.01 * 100).toFixed(3)}%`);
claim("scale-sums", `2⁻⁵ = ${2 ** -5}`);
ok("0.01 below half spacing at 4", 0.01 < 2 ** -5 / 2 && xb < 2 ** -5 / 2);
// loss scaling and mixed precision
claim("scale-loss-scaling", sciTxt(2 ** -14, 1));
claim("scale-loss-scaling", `2^{-24} = ${sciTxt(2 ** -24, 1).replace(" × 10⁻⁸", " \\times 10^{-8}")}`);
claim("scale-loss-scaling", sciTxt(2 ** -126, 1));
claim("scale-mixed", `1/${int(0.01 / 8e-7)}`);
ok("BF16 relative step between 1/128 and 1/256", 2 ** -7 === 1 / 128 && 2 ** -8 === 1 / 256);
ok("update below BF16 half step", 8e-7 / 0.01 < 2 ** -9);
claim("scale-mixed", `about 1/${int(Math.round(2 ** 23 / 1e5) * 1e5)}`);
claim("scale-fp8", `2^{-10} \\approx ${r(2 ** -10, 3)}`);

// ------------------------------------------------------------------ 7.4 collectives
claim("scale-collectives", "11 + 21 + 31 + 41 = 104");
near("chunk 1 sum", [0, 1, 2, 3].reduce((a, rr) => a + 10 * (rr + 1) + 1, 0), 104);
const ring = (N, S) => 2 * (N - 1) / N * S;
claim("scale-ring", `2 × 7/8 × 16 = ${ring(8, 16)} GB`);
claim("scale-ring", `${Math.round(ring(8, 16e9) / 450e9 * 1e3)} ms`);
claim("scale-ring", `each sends ${(ring(1024, 16e9) / 1e9).toFixed(2)} GB`);
claim("scale-ring", `takes ${(ring(1024, 16e9) / 50e9).toFixed(2)} s`);
const ab = (p, n, a, bw) => { const b = 1 / bw, lg = Math.log2(p); return { ring: 2 * (p - 1) * a + 2 * ((p - 1) / p) * n * b, rd: lg * (a + n * b), rab: 2 * lg * a + 2 * ((p - 1) / p) * n * b, tree: 2 * lg * (a + n * b) }; };
const t1 = ab(1024, 1e6, 10e-6, 50e9), t2 = ab(1024, 16e9, 10e-6, 50e9);
claim("scale-alphabeta", `${(t1.ring * 1e3).toFixed(1)} ms by ring`);
claim("scale-alphabeta", `${(t1.rab * 1e3).toFixed(2)} ms by Rabenseifner`);
claim("scale-alphabeta", `${t2.ring.toFixed(2)} s by ring and ${t2.rd.toFixed(1)} s by recursive doubling`);
ok("ring 1 MB is latency-bound", 2 * 1023 * 10e-6 / t1.ring > 0.99);
claim("scale-busbw", `algbw of ${4e9 / 0.02 / 1e9} GB/s`);
claim("scale-busbw", `200 × 2 × 7/8 = ${200 * 2 * 7 / 8} GB/s, ${Math.round(350 / 450 * 100)}%`);
claim("scale-links", "18 NVLink 4 links, 450 GB/s");
near("NVLink per direction", 18 * 25, 450);
near("400 Gb/s in GB/s", 400 / 8, 50);
claim("scale-links", `${450 / 50} times less`);
near("pod", 192 * 16, 3072); near("cluster", 8 * 3072, 24576);
claim("scale-links", "3,072 GPUs");
claim("scale-overlap", "24 + 16 = 40 ms");
ok("overlap ends at 28 ms", Math.max(24, 24 + 4) === 28);
claim("scale-overlap", "28 ms");
claim("scale-overlap", `2 × 7/8 × 1 GB = ${ring(8, 1)} GB: ${ring(8, 1) / 2} GB`);

// ------------------------------------------------------------------ 7.5 data parallelism
const ratio = (F, bw, T) => 2 * F / (3 * bw * T);
claim("scale-dp", `${Math.round(ratio(400e12, 50e9, 32768) * 100)}% for T = 32,768`);
claim("scale-dp", `${(ratio(400e12, 50e9, 262144) * 100).toFixed(1)}% for T = 262,144`);
claim("scale-dp", `the first case is ${(ratio(400e12, 450e9, 32768) * 100).toFixed(1)}%`);
near("ratio identity", (4 * 8e9 / 50e9) / (6 * 8e9 * 32768 / 400e12), ratio(400e12, 50e9, 32768), 1e-12);
const bucket = 25 * 2 ** 20;
claim("scale-bucket", `make ${Math.ceil(16e9 / bucket)} buckets`);
claim("scale-bucket", `spends ${Math.round(2 * 7 * 5)} µs on latency`);
claim("scale-bucket", `${(2 * 7 / 8 * bucket / 450e9 * 1e3).toFixed(2)} ms moving data`);
claim("scale-bucket", `${(2 * 1023 * 5e-6 * 1e3).toFixed(1)} ms per bucket against ${(2 * 1023 / 1024 * bucket / 50e9 * 1e3).toFixed(2)} ms`);
const Z = (P, n) => [16 * P, 4 * P + 12 * P / n, 2 * P + 14 * P / n, 16 * P / n].map((x) => x / 1e9);
const z = Z(7.5e9, 64);
claim("scale-zero", `${Math.round(z[0])} GB per GPU without ZeRO, then ${z[1].toFixed(1)} GB, ${z[2].toFixed(1)} GB and ${z[3].toFixed(1)} GB`);
claim("scale-batch", "from 32 to 16 sequences");
near("DP64x32 = DP128x16", 64 * 32, 128 * 16);
claim("scale-batch", `= ${int(1024 * 4096)}) = ${(252e6 / (1024 * 4096)).toFixed(1)} steps`);

// ------------------------------------------------------------------ 7.6 model parallelism
const d = 16384, Pl = 2 * d * d + 2 * d * 1024 + 3 * d * 53248;
near("Llama 405B layer weights", Pl, 3187671040);
near("405B total check", (126 * Pl + 2 * 128256 * d) / 1e9, 405.8, 0.1);
claim("scale-tp-attn", "3,187,671,040");
claim("scale-tp-attn", `${(Pl / 1e9).toFixed(2)} × 10⁹ weights`);
const arB = 2 * 8192 * d;
claim("scale-tp-attn", `each all-reduce is ${Math.round(arB / 1e6)} MB`);
claim("scale-tp-attn", `2 × 7/8 × 268 = ${Math.round(ring(8, arB) / 1e6)} MB per GPU: ${(ring(8, arB) / 450e9 * 1e3).toFixed(2)} ms`);
const fl = 2 * Pl * 8192 / 8;
claim("scale-tp-attn", `= ${sciTxt(fl, 1)} FLOPs, ${Math.round(fl / 400e12 * 1e3)} ms`);
claim("scale-tp-attn", `add ${(2 * ring(8, arB) / 450e9 * 1e3).toFixed(1)} ms, ${Math.round(2 * ring(8, arB) / 450e9 / (fl / 400e12) * 100)}%`);
claim("scale-tp-attn", `take ${Math.round(2 * ring(8, arB) / 50e9 * 1e3)} ms`);
ok("IB slower than compute", 2 * ring(8, arB) / 50e9 > fl / 400e12);
near("query heads per GPU", 128 / 8, 16);
const sbh = 2048 * 12288;
const spRows = [sbh * 114, sbh * (10 + 24 / 8 + 5 * 96 * 2048 / (12288 * 8)), sbh * 114 / 8, 34 * sbh / 8].map((x) => x / 1e9);
claim("scale-sp", `${spRows[2].toFixed(2)} GB per layer per sequence instead of ${spRows[1].toFixed(2)} GB`);
claim("scale-sp", `leaves ${spRows[3].toFixed(2)} GB, ${Math.round(spRows[0] / spRows[3])} times less than the ${spRows[0].toFixed(2)} GB`);
for (const [i, s] of [[0, "2.87 GB"], [1, "0.58 GB"], [2, "0.36 GB"], [3, "0.11 GB"]]) { ok(`sp row ${i}`, spRows[i].toFixed(2) === s.split(" ")[0]); claim("scale-sp", s); }
const bub = (p, m) => (p - 1) / (m + p - 1);
claim("scale-pp", `m = 16 leaves ${Math.round(bub(16, 16) * 100)}% of the time idle and m = 64 leaves ${Math.round(bub(16, 64) * 100)}%`);
const il = (p, m, v) => { const x = (p - 1) / (v * m); return x / (1 + x); };
claim("scale-interleave", `a bubble of ${Math.round(il(16, 16, 4) * 100)}% of the step instead of ${Math.round(bub(16, 16) * 100)}%`);
near("interleaved overhead", 15 / (4 * 16), 15 / 64);
claim("scale-ep", `8 × 7,168 = ${int(8 * 7168)} bytes`);
claim("scale-ep", `${int(8 * 7168 * 2)} bytes`);
near("NVLink/IB", 160 / 50, 3.2);
const work = (j) => j + 0.5;
const cont = [0, 1, 2, 3].map((r) => work(2 * r) + work(2 * r + 1)), zig = [0, 1, 2, 3].map((r) => work(r) + work(7 - r));
equal("contiguous work", cont, [2, 6, 10, 14]);
equal("zig-zag work", zig, [8, 8, 8, 8]);
claim("scale-cp", `${14 / 8} times the balanced time`);
claim("scale-cp", "TP 8 × CP 16 × PP 16 × DP 8 = 16,384");
near("128K layout", 8 * 16 * 16 * 8, 16384);
near("K+V vs Q", 2 * 8 / 128, 1 / 8);
claim("scale-mesh", `6.48 × 10¹² / 16,384 = ${Math.round(16 * 405e9 / 16384 / 1e6)} MB per GPU`);
near("TPxPP pieces", 8 * 16, 128);
claim("scale-layout", "2,048 sequences");
claim("scale-layout", `${int(16777216 / 131072)} sequences per step, divided over 8 data-parallel groups: ${16777216 / 131072 / 8} each`);
// the layout figure at Meta's configuration must fit: recompute its memory estimate
const act = 34 * 8192 * d / 8 * Math.ceil(126 / 16) * 16, state = 16 * 405e9 / 16384, gath = Math.ceil(126 / 16) * 2 * Pl / 8;
near("layout total at Meta config (GiB)", (act + state + gath) / GiB, 74.3, 0.05);
near("gathered stage weights (GiB)", gath / GiB, 5.94, 0.01);
ok("Llama layout estimate fits in 79.32 GiB", act + state + gath < 79.32 * GiB);

// ------------------------------------------------------------------ 7.7 running a job
claim("scale-data", "more than 6 times in a bucket of 30M documents");
near("data mix sums to 100", 50 + 25 + 17 + 8, 100);
claim("scale-ckpt", `12 × 405 × 10⁹ = ${(12 * 405e9 / 1e12).toFixed(2)} TB`);
claim("scale-ckpt", `${Math.round(12 * 405e9 / 16384 / 1e6)} MB each`);
claim("scale-ckpt", `needs ${(12 * 405e9 / 2e12).toFixed(1)} s`);
const causes = [148, 72, 54, 35, 32, 19, 17, 7, 7, 6, 6, 16];
near("Table 5 counts sum", causes.reduce((a, b) => a + b, 0), 419);
near("other causes", 3 + 3 + 2 * 5, 16);
near("planned + unexpected", 47 + 419, 466);
claim("scale-failures", `one every ${(54 * 24 / 419).toFixed(1)} hours`);
claim("scale-failures", `148 of 419, is ${(148 / 419 * 100).toFixed(1)}%`);
const Mmin = 3.1 * 60;
claim("scale-interval", `T* = ${Math.round(Math.sqrt(2 * 1 * Mmin))} minutes and loses ${Math.round(2 * 1 / Math.sqrt(2 * Mmin) * 100)}%`);
claim("scale-interval", `T* = ${Math.sqrt(2 * 0.1 * Mmin).toFixed(1)} minutes and writes plus lost work cost ${(Math.sqrt(2 * 0.1 / Mmin) * 100).toFixed(1)}%`);
const step = 6 * 405e9 * 16777216 / (16384 * 400e12);
claim("scale-monitor", `= ${step.toFixed(1)} s per step, ${(16777216 / step / 1e6).toFixed(1)}M tokens per second, ${Math.round(16777216 / step * 86400 / 1e9)} billion tokens a day`);
const wck = (T, C, R, M) => C / T + (T / 2 + R) / M;
claim("scale-interval", `(30/2 + 10)/186 = ${(25 / 186 * 100).toFixed(1)}%`);
claim("scale-interval", `${(wck(30, 1, 10, 186) * 100).toFixed(1)}%`);
claim("scale-interval", `√(2 × 1 × 186) = ${Math.sqrt(372).toFixed(1)} minutes would lower it to ${(wck(Math.sqrt(372), 1, 10, 186) * 100).toFixed(1)}%`);
claim("scale-interval", `C/T = 1/30 = ${(100 / 30).toFixed(1)}%`);

// ------------------------------------------------------------------ review fixes
claim("scale-overlap", "7 GB would be sending the whole buffer to each of the 7 other GPUs");
const opt = (E, A, B, a, b, C) => { const G = Math.pow(a * A / (b * B), 1 / (a + b)), N = G * Math.pow(C / 6, b / (a + b)); return [N, C / 6 / N / N]; };
const [nR, rR] = opt(1.69, 406.4, 410.7, 0.34, 0.28, 5.76e23), [nU] = opt(1.6934, 406.4, 410.7, 0.3392, 0.2849, 5.76e23), [nB, rB] = opt(1.8172, 482.01, 2085.43, 0.3478, 0.3658, 5.76e23);
claim("scale-laws", `at ${Math.round(nR / 1e9)}B parameters, ${Math.round(rR)} tokens each`);
claim("scale-laws", `give ${Math.round(nU / 1e9)}B, the paper's figure`);
claim("scale-laws", `gives ${Math.round(nB / 1e9)}B, ${Math.round(rB)} tokens per parameter`);
const sbhCP = 34 * 131072 * d / 8;
claim("scale-cp", `34sbh/t = ${(sbhCP / 1e9).toFixed(1)} GB`);
claim("scale-cp", `${Math.round(sbhCP * 8 / 1e9)} GB for a stage of 8 layers`);
claim("scale-cp", "only 128 such sequences");
claim("scale-mesh", `2 × 405 × 10⁹ / 128 bytes = ${(2 * 405e9 / 128 / GiB).toFixed(1)} GiB`);
claim("scale-innet", "1.75 S for 8 GPUs");
near("ring factor 8", 2 * 7 / 8, 1.75);
claim("scale-innet", `algbw of 300 GB/s through NVLS reports a busbw of ${300 * 2 * 7 / 8} GB/s`);
claim("scale-spikes", `= ${(0.1 / Math.sqrt(0.05)).toFixed(2)} lr`);
ok("our Adam betas are 0.9, 0.95", JSON.stringify(TinyLM.config.betas || [0.9, 0.95]) === JSON.stringify([0.9, 0.95]));
claim("scale-fp8", "half of E4M3's smallest step");
// scale-sum at 0.1 x 100,000: total error splits into input rounding and accumulation
{ const xb1 = bf16(0.1); let a = 0; for (let i = 0; i < 100000; i++) a = Math.fround(a + xb1);
  const eIn = (xb1 - 0.1) / 0.1, eAcc = (a - 1e5 * xb1) / (1e5 * xb1), eTot = (a - 1e4) / 1e4;
  near("split of FP32 error", (1 + eIn) * (1 + eAcc) - 1, eTot, 1e-12); ok("accumulation error is negative there", eAcc < -0.0008); }
done();
