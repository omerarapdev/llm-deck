// Part 6: GPUs and CUDA. Recomputes every number the sheets state, and checks
// every custom figure's model (window.GPU6) against an independent brute-force
// implementation over the whole input range the figure allows.
import { claim, near, equal, ok, r, int, window, done } from "./lib.mjs";

const G = window.GPU6;
ok("window.GPU6 is exported by content/06-gpu.js", !!G);

// ---------------------------------------------------------------- hardware constants (H100 SXM5, sources in the sheets)
const SMS = 132, BF16 = 989.4e12, FP32 = 66.9e12, BW = 3.35e12, BW_WP = 3352e9, CLK = 1.98e9, CLK_TC = 1.83e9;
const PCIE = 64e9, NVL = 450e9;
// Luo et al. measured latencies on an H800 PCIe (HBM2e), whose maximum clock is 1.755 GHz (their Table III):
// cycles / 1.755 GHz is the shortest the latency can have been in nanoseconds.
const H800_CLK = 1.755e9, HBM_CYC = 478.8, HBM_NS = Math.round((HBM_CYC / H800_CLK) * 1e9);
const ms = (s) => s * 1e3, us = (s) => s * 1e6;

// ================================================================ 6.1 Why GPUs
{
  const d = 4096, kv = 1024, ff = 14336, heads = 32;
  const weights = 2 * d * d + 2 * d * kv + 2 * d * kv + 2 * d * d + 3 * 2 * d * ff;
  near("Llama 3 8B weight FLOPs per token per layer", weights, 436207616);
  equal("GPU6.llamaLayer weights", G.llamaLayer(4096).weights, weights);
  claim("gpu-flops", "2 × 4,096 × (4,096 + 1,024 + 1,024 + 4,096 + 3 × 14,336)");
  near("bracket", 2 * 4096 * (4096 + 1024 + 1024 + 4096 + 3 * 14336), weights);
  claim("gpu-flops", int(weights));
  const attn = 2 * 2 * 4096 * heads * 128;
  near("attention FLOPs at T = 4096", attn, 67108864);
  equal("GPU6.llamaLayer attn", G.llamaLayer(4096).attn, attn);
  claim("gpu-flops", "67.1 million");
  const other = 2 * 4 * d + 3 * (d + kv) + 5 * ff + 2 * d + 6 * heads * 4096;
  equal("GPU6.llamaLayer other", G.llamaLayer(4096).other, other);
  near("other ≈ 0.9 M", Math.round(other / 1e5) / 10, 0.9);
  claim("gpu-flops", "about 0.9 million");
  const share = (weights + attn) / (weights + attn + other);
  claim("gpu-flops", r(share * 100, 2) + "%");
  claim("gpu-flops", r(100 - share * 100, 2) + "%");
  near("share 99.82", +r(share * 100, 2), 99.82);
  near("128,000 constant", 2 * 4 * d + 3 * (d + kv) + 5 * ff + 2 * d, 128000);
  // calc at every option stays a valid fraction and the note's formula matches
  for (const T of [512, 2048, 4096, 8192, 32768, 131072]) {
    const L = G.llamaLayer(T);
    near(`other at T=${T}`, L.other, 128000 + 192 * T);
    ok(`share at T=${T} in (0.98, 1)`, L.share > 0.98 && L.share < 1);
  }
}
{
  // Grace vs H100
  const grace = 72 * 32 * 3.1e9, superFP64 = 144 * 16 * 3.1e9;
  near("Grace FP32 peak", grace / 1e12, 7.1424, 1e-4);
  near("Superchip FP64 peak ≈ 7.1", +r(superFP64 / 1e12, 1), 7.1);
  claim("gpu-cpu", "72 cores × 32 × 3.1 GHz = 7.1 TFLOP/s");
  near("H100 FP32 from lanes", 132 * 256 * CLK / 1e12, 66.90816, 1e-5);
  claim("gpu-cpu", "66.9 TFLOP/s");
  claim("gpu-cpu", r(FP32 / grace, 1) + " times Grace's FP32 peak");
  claim("gpu-cpu", r(BF16 / grace, 0) + " times that peak");
  claim("gpu-cpu", r(BW / 512e9, 1) + " times its memory bandwidth");
  claim("gpu-cpu", "128 FP32 lanes (256 FLOPs per clock)");
  claim("gpu-cpu", "2,048 threads");
}
{
  claim("gpu-h100", r(BF16 / BW, 0) + " FLOPs per byte");
  near("dense = sparse / 2", 1978.9 / 2, 989.45, 1e-9);
  claim("gpu-h100", "989.4 TFLOP/s");
  near("8 GPCs × 9 TPCs × 2", 8 * 9 * 2, 144);
  claim("gpu-h100", "of 144 on the full GH100 die");
}
{
  const n = 2 ** 26, bytes = 12 * n, t = bytes / BW;
  near("1/12", +r(1 / 12, 3), 0.083);
  claim("gpu-intensity", "1/12 ≈ 0.083 FLOP/byte");
  near("805 MB", +r(bytes / 1e6, 0), 805);
  claim("gpu-intensity", "moves 805 MB");
  claim("gpu-intensity", `takes ${r(us(t), 0)} µs`);
  const flops = n / t;
  claim("gpu-intensity", `${r(flops / 1e9, 0)} GFLOP/s`);
  claim("gpu-intensity", `${r((flops / FP32) * 100, 2)}%`);
  claim("gpu-intensity", `66.9 / 3.35 ≈ ${r(FP32 / BW, 0)}`);
  claim("gpu-intensity", `989.4 / 3.35 ≈ ${r(BF16 / BW, 0)}`);
  // first integer FLOPs/element at which arithmetic time exceeds memory time
  let k = 1;
  while ((k * n) / FP32 <= bytes / BW) k++;
  equal("crossover k", k, 240);
  claim("gpu-intensity", "at 240 FLOPs per element");
  // check
  claim("gpu-intensity", "3/8 = 0.375 FLOP/byte");
  claim("gpu-intensity", int(8 * n) + " bytes");
  claim("gpu-intensity", `${r(us((8 * n) / BW), 0)} µs at 3.35 TB/s`);
  claim("gpu-intensity", `would take ${r(us((3 * n) / FP32), 0)} µs`);
}

// ================================================================ 6.2 Inside an H100
{
  near("SXM SMs", 66 * 2, SMS);
  near("bus", 10 * 512, 5120);
  claim("gpu-chip", "66 TPCs");
  claim("gpu-chip", "5,120-bit bus");
  near("registers per block", 16384 * 4, 65536);
  claim("gpu-sm", "64 KB register file (16,384 registers of 32 bits)");
  claim("gpu-sm", "65,536 registers (256 KB)");
  near("4 × 64 KB", 4 * 64, 256);
}
{
  near("FP32 FLOPs/clk/SM", 128 * 2, 256);
  near("tensor FMA/clk/SM = 2 × A100's 1,024", 2 * 1024, 2048);
  near("16×", 4096 / 256, 16);
  claim("gpu-tensor", "132 × 256 × 1.98 GHz = 66.9 TFLOP/s");
  near("SFU rate is 1/8 of the FP32 FMA rate", 16 / 128, 1 / 8);
  claim("gpu-tensor", "16 results per clock per SM, an eighth of the FP32 FMA rate");
  near("16 per clock per SM × 132 × 1.83 GHz ≈ FlashAttention-3's 3.9 TFLOP/s", +r((132 * 16 * CLK_TC) / 1e12, 1), 3.9);
  claim("gpu-tensor", "about 3.9 TFLOP/s on an H100 SXM5");
  claim("gpu-tensor", "132 × 4,096 × 1.83 GHz = 989.4 TFLOP/s");
  near("BF16 peak from parts", +r((132 * 4096 * CLK_TC) / 1e12, 1), 989.4);
  near("TF32 half", +r((132 * 2048 * CLK_TC) / 1e12, 1), 494.7);
  near("FP8 double", +r((132 * 8192 * CLK_TC) / 1e12, 1), 1978.9);
  near("FP64 lanes", +r((132 * 64 * 2 * CLK) / 1e12, 1), 33.5);
  near("FP64 tensor", +r((132 * 128 * 2 * CLK) / 1e12, 1), 66.9);
  near("BF16 lanes = 2 × FP32", +r((2 * 132 * 256 * CLK) / 1e12, 1), 133.8);
}
{
  const lat = { shared: 29.0, l1: 40.7, l2: 263.0, hbm: 478.8 };
  claim("gpu-hierarchy", `${r(lat.hbm / lat.shared, 1)} times a shared-memory load`);
  equal("HBM latency lower bound, ns", HBM_NS, 273);
  claim("gpu-hierarchy", `${HBM_NS} ns or more, since the clock was at most 1.755 GHz`);
  claim("gpu-hierarchy", "479 (HBM2e)");
  const smemBW = SMS * 128 * CLK;
  claim("gpu-hierarchy", `about ${r(smemBW / 1e12, 0)} TB/s`);
  near("smem / HBM ≈ 10", +r(smemBW / BW, 0), 10);
  claim("gpu-hierarchy", "ten times");
  for (const [k, v] of Object.entries({ "29": lat.shared, "41": lat.l1, "263": lat.l2, "479": lat.hbm })) claim("gpu-hierarchy", k, `latency ${v}`);
  near("32 banks × 4 B", 32 * 4, 128);
}
{
  const h100 = 5120 * 2 * 2619e6 / 8, v100 = 4096 * 2 * 877.5e6 / 8, a100 = 5120 * 2 * 1215e6 / 8;
  claim("gpu-hbm", `5,120 × 5.238 / 8 = ${int(h100 / 1e9)} GB/s`);
  near("per-pin H100", 2 * 2.619, 5.238, 1e-12);
  near("H100 bw", +r(h100 / 1e9, 0), 3352);
  claim("gpu-hbm", `= ${r(v100 / 1e9, 0)} GB/s`);
  near("A100 40GB bw", +r(a100 / 1e9, 0), 1555);
  const rows = { V: [125, 900], A40: [312, 1555], A80: [312, 2039], H: [989.4, 3352] };
  equal("ratios", Object.values(rows).map(([f, b]) => r(f / (b / 1e3), 0)), ["139", "201", "153", "295"]);
  claim("gpu-hbm", `${r(989.4 / 125, 1)} times and bandwidth ${r(3352 / 900, 1)} times`);
  claim("gpu-hbm", "from 139 to 295");
  claim("gpu-hbm", "5 HBM3 stacks of 16 GB");
  claim("gpu-hbm", "5,120 pins at a modest 5.2 Gb/s each");
  near("16 GB per stack", 80 / 5, 16);
  near("1024 bits per stack", 5120 / 5, 1024);
}
{
  near("peak as product", 132 * 4 * 1024 * CLK_TC / 1e12, 989.43, 0.01);
  claim("gpu-peak", "132 SMs × 4 tensor cores × 1,024 FLOPs per tensor core per clock × 1.83 GHz");
  claim("gpu-peak", r((728.5 / 756.5) * 100, 0) + "%");
  claim("gpu-peak", r((665.4 / 756.5) * 100, 0) + "%");
  claim("gpu-peak", r((740 / 989.4) * 100, 0) + "%");
  claim("gpu-peak", `about ${r((1 - 665.4 / 728.5) * 100, 0)}% slower`);
}
{
  near("NVLink per direction", 18 * 25, 450);
  claim("gpu-nvlink", "450 GB/s out and 450 GB/s in, the 900 GB/s");
  claim("gpu-nvlink", `${r(ms(1e9 / BW), 2)} ms from a GPU's own HBM`);
  claim("gpu-nvlink", `${r(ms(1e9 / NVL), 1)} ms to another GPU`);
  claim("gpu-nvlink", `${r(ms(1e9 / PCIE), 1)} ms to or from the host`);
  near("NVLink / PCIe ≈ 7", +r(NVL / PCIE, 0), 7);
  claim("gpu-nvlink", "one seventh of NVLink");
}
{
  // B200 per GPU = HGX B200 8-GPU totals / 8; dense = sparse / 2 where marked
  const b200 = { fp16: 36e15 / 8 / 2, fp8: 72e15 / 8 / 2, fp4: 72e15 / 8, fp32: 600e12 / 8, fp64: 296e12 / 8, mem: 1440 / 8, bw: 64e12 / 8 };
  equal("B200 per-GPU", [b200.fp16 / 1e12, b200.fp8 / 1e12, b200.fp4 / 1e12, b200.fp32 / 1e12, b200.fp64 / 1e12, b200.mem, b200.bw / 1e12], [2250, 4500, 9000, 75, 37, 180, 8]);
  for (const x of ["2,250", "4,500", "9,000", "180 GB HBM3e", "8 TB/s", "281"]) claim("gpu-datasheet", x);
  near("B200 ratio", +r(2250 / 8, 0), 281);
  claim("gpu-datasheet", `grew ${r(2250 / 125, 0)} times, HBM bandwidth ${r(8000 / 900, 1)} times and FP32 throughput without tensor cores ${r(75 / 15.7, 1)} times`);
  claim("gpu-datasheet", `${r(989.4 / 312, 1)} times while bandwidth grew 3,352 / 2,039 = ${r(3352 / 2039, 1)} times`);
  claim("gpu-datasheet", `312 / 2.039 = ${r(312 / 2.039, 0)}`);
  claim("gpu-datasheet", `989.4 / 3.352 = ${r(989.4 / 3.352, 0)}`);
}

// ================================================================ 6.3 The CUDA programming model
{
  claim("gpu-host", `${r(BW / PCIE, 0)} times less`);
  const g = G.grid1d(1e6, 256);
  equal("1e6 / 256", [g.grid, g.threads, g.idle], [3907, 1000192, 192]);
  claim("gpu-index", "3,907 blocks and 1,000,192 threads, and the last 192");
  claim("gpu-copycost", "PCIe Gen 4 x16, 32 GB/s");
  // exhaustive: the figure's 1-D model against brute force
  for (const B of [4, 8, 16, 32]) for (let n = 1; n <= 48; n++) {
    const m = G.grid1d(n, B);
    let blocks = 0; while (blocks * B < n) blocks++;
    let idle = 0; for (let b = 0; b < blocks; b++) for (let t = 0; t < B; t++) if (b * B + t >= n) idle++;
    if (m.grid !== blocks || m.idle !== idle || m.threads !== blocks * B) { ok(`grid1d n=${n} B=${B}`, false); }
  }
  ok("grid1d exhaustive (n 1..48, B 4..32)", true);
  for (const bx of [2, 4, 8]) for (const by of [2, 4, 8]) for (let R = 1; R <= 12; R++) for (let C = 1; C <= 16; C++) {
    const m = G.grid2d(R, C, bx, by);
    let gx = 0; while (gx * bx < C) gx++;
    let gy = 0; while (gy * by < R) gy++;
    let idle = 0, covered = new Set();
    for (let byi = 0; byi < gy; byi++) for (let bxi = 0; bxi < gx; bxi++) for (let ty = 0; ty < by; ty++) for (let tx = 0; tx < bx; tx++) {
      const col = bxi * bx + tx, row = byi * by + ty;
      if (row < R && col < C) covered.add(row * C + col); else idle++;
    }
    if (m.gx !== gx || m.gy !== gy || m.idle !== idle || covered.size !== R * C) ok(`grid2d ${R}x${C} ${bx}x${by}`, false);
  }
  ok("grid2d exhaustive (R 1..12, C 1..16, blocks 2..8)", true);
}
{
  const n = 2 ** 26;
  claim("gpu-vecadd", "2 26 floats");
  claim("gpu-vecadd", int(n) + " elements");
  claim("gpu-vecadd", int(4 * n) + " bytes");
  claim("gpu-vecadd", `three ${r(4 * n / 1e6, 0)} MB arrays`);
  equal("grid", Math.floor((n + 255) / 256), 262144);
  claim("gpu-vecadd", "262,144 blocks");
  claim("gpu-vecadd", `3 × 256 + 5 = ${3 * 256 + 5}`);
  claim("gpu-vecadd", `${r(12 * n / 1e6, 0)} MB for the three`);
}
{
  const n = 2 ** 26, tin = (8 * n) / PCIE, tout = (4 * n) / PCIE, tk = (12 * n) / BW;
  claim("gpu-copycost", `${r(8 * n / 1e6, 0)} MB in (${r(ms(tin), 1)} ms) and ${r(4 * n / 1e6, 0)} MB out (${r(ms(tout), 1)} ms)`);
  claim("gpu-copycost", `takes ${r(us(tk), 0)} µs`);
  claim("gpu-copycost", `${r(BW / PCIE, 0)} times as much`);
  claim("gpu-copycost", `${r((tk / (tin + tout + tk)) * 100, 1)}% of the end-to-end time`);
  near("copies / kernel", (tin + tout) / tk, BW / PCIE, 1e-9);
}
{
  claim("gpu-2d", "128 × 512 grid, 65,536 blocks");
  near("grid 4096/32 × 4096/8", (4096 / 32) * (4096 / 8), 65536);
}
{
  const perSM = 2048 / 256, total = perSM * SMS;
  equal("blocks resident", total, 1056);
  claim("gpu-sync", "1,056 resident blocks, 8 per SM");
  claim("gpu-sync", `about ${Math.round(262144 / total)} waves`);
}
{
  const n = 2 ** 26, tin = (8 * n) / PCIE, tout = (4 * n) / PCIE, tk = (12 * n) / BW;
  const serial = tin + tk + tout, four = tin + tk / 4 + tout / 4;
  claim("gpu-streams", `${r(ms(serial), 1)} ms to ${r(ms(four), 1)} ms`);
  claim("gpu-streams", `${r(ms(tin), 2)} + ${r(ms(tk), 2)} + ${r(ms(tout), 2)} = ${r(ms(serial), 2)} ms`);
  claim("gpu-streams", `${r(ms(tin), 2)} + ${r(ms(tk / 4), 2)} + ${r(ms(tout / 4), 2)} = ${r(ms(four), 2)} ms`);
  // the 4-stream schedule is feasible: each chunk's D2H finishes before the next chunk's D2H starts
  ok("D2H chunks do not overlap", tout / 4 + tk / 4 <= tin / 4);
  const blocks = Math.floor((10000 + 255) / 256);
  equal("check: blocks", blocks, 40);
  equal("check: idle", blocks * 256 - 10000, 240);
}

// ================================================================ 6.4 How an SM executes threads
{
  equal("100-thread block: warps", Math.ceil(100 / 32), 4);
  equal("100-thread block: active lanes in last warp", 100 - 3 * 32, 4);
  claim("gpu-warps", "the last has 4 active threads and 28 idle");
  claim("gpu-warps", "A 256-thread block is 8 warps");
}
{
  // warp scheduler model: simulated steady state equals min(1, W(c+1)/(L+c)) for every input class
  let bad = 0, n = 0;
  for (let W = 1; W <= 16; W++) for (let c = 1; c <= 16; c++) for (const L of [4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 20, 25, 29, 30, 31, 40, 50, 64, 100, 128, 200, 263, 300, 479, 500, 600]) {
    const P = L + c, T = W * (c + 1) + 6 * P;
    const sim = G.warpsim(W, c, L, T);
    let busy = 0; for (let t = T - P; t < T; t++) if (sim.issue[t] >= 0) busy++;
    n++;
    if (Math.abs(busy / P - G.warpUtil(W, c, L)) > 1e-12) bad++;
    // the scheduler never issues two instructions in one cycle, and a warp never issues while waiting
    for (let w = 0; w < W && !bad; w++) for (let t = 0; t < T; t++) if ((sim.kind[w][t] === "L" || sim.kind[w][t] === "C") !== (sim.issue[t] === w)) { bad++; break; }
  }
  ok(`warp scheduler simulation matches U = min(1, W(c+1)/(L+c)) in ${n} cases`, bad === 0);
  equal("warps needed L=30 c=4", G.warpsNeeded(4, 30), 7);
  const cycH100 = Math.round(HBM_NS * 1e-9 * CLK);
  equal("273 ns at 1.98 GHz, cycles (rounded)", cycH100, 541);
  ok("about 540 cycles", Math.abs((HBM_CYC / H800_CLK) * CLK - 540) < 1);
  equal("warps needed at 540 cycles, c = 4", G.warpsNeeded(4, 540), 109);
  equal("warps needed at 541 cycles, c = 4", G.warpsNeeded(4, 541), 109);
  claim("gpu-hiding", "seven warps suffice");
  claim("gpu-hiding", `${HBM_NS} ns or more (measured on an H800 PCIe), about 540 cycles at the H100's 1.98 GHz: a scheduler would need 109 warps`);
  claim("gpu-hiding", "at most 16");
  near("64 slots / 4 schedulers", 64 / 4, 16);
  near("HBM latency rounds to 479", Math.round(HBM_CYC), 479);
}
{
  const inflight = BW * HBM_NS * 1e-9, perSM = inflight / SMS;
  claim("gpu-little", `= ${r(inflight / 1e3, 0)} KB of loads`);
  claim("gpu-little", `${r(perSM / 1e3, 1)} KB per SM`);
  claim("gpu-little", `3.35 × 10¹² × ${HBM_NS} × 10⁻⁹`);
  equal("128-byte warp loads per SM", Math.ceil(perSM / 128), 55);
  equal("512-byte warp loads per SM", Math.ceil(perSM / 512), 14);
  claim("gpu-little", "needs 55 of them in flight");
  claim("gpu-little", `55 resident warps, ${r((55 / 64) * 100, 0)}% occupancy`);
  claim("gpu-little", "it is 14");
  ok("title: about 900 KB", Math.abs(inflight / 1e3 - 900) < 20);
  claim("gpu-little", "keep about 900 KB in flight");
  claim("gpu-ilp", `${HBM_NS} ns of latency`);
  claim("gpu-ilp", `55 warps per SM, ${r((55 / 64) * 100, 0)}% occupancy`);
  claim("gpu-ilp", `14 warps, ${r((14 / 64) * 100, 0)}%`);
  ok("littleBW: 55 warps saturate, 54 do not", G.littleBW(55, 4, 1, HBM_NS) === BW && G.littleBW(54, 4, 1, HBM_NS) < BW);
  ok("littleBW: ILP 4 saturates at 14 warps", G.littleBW(14, 4, 4, HBM_NS) === BW && G.littleBW(13, 4, 4, HBM_NS) < BW);
  ok("littleBW: float4 saturates at 14 warps", G.littleBW(14, 16, 1, HBM_NS) === BW && G.littleBW(13, 16, 1, HBM_NS) < BW);
  claim("gpu-ilp", "87% of peak memory bandwidth at 8% occupancy");
}
{
  // divergence: per-warp cost = A·[any lane takes if] + B·[any lane takes else], checked against a direct count
  let bad = 0;
  for (const cond of ["even", "lt", "warp", "rand"]) for (let t = 0; t <= 128; t += cond === "lt" ? 1 : 128) for (const [A, B] of [[1, 1], [12, 8], [32, 1], [5, 32]]) for (const seed of cond === "rand" ? [11, 12, 13] : [11]) {
    const takes = G.divergeLanes(cond, t, seed), m = G.diverge(takes, A, B);
    let total = 0, useful = 0, div = 0;
    for (let w = 0; w < 4; w++) {
      let i1 = 0; for (let l = 0; l < 32; l++) { const tid = 32 * w + l; const tk = takes[tid]; if (tk) { i1++; useful += A; } else useful += B;
        // the lane's condition, recomputed from its thread index
        if (cond === "even" && tk !== (tid % 2 === 0)) bad++;
        if (cond === "lt" && tk !== (tid < t)) bad++;
        if (cond === "warp" && tk !== (Math.floor(tid / 32) % 2 === 0)) bad++; }
      total += (i1 ? A : 0) + (i1 < 32 ? B : 0); if (i1 && i1 < 32) div++;
    }
    if (m.total !== total || m.divergent !== div || Math.abs(m.eff - useful / (32 * total)) > 1e-12) bad++;
  }
  ok("divergence model matches a direct count for every condition", bad === 0);
  const d = G.diverge(G.divergeLanes("even", 0, 11), 12, 8);
  equal("even lanes: every warp pays A + B", d.warps.map((w) => w.cost), [20, 20, 20, 20]);
  near("P(all 32 random lanes agree)", 2 * 0.5 ** 32, 2 * 2 ** -32, 0);
  claim("gpu-diverge", "is 2 × 2 −32");
}
{
  // occupancy: an independent block-placement simulation (warps of each block spread round-robin over the
  // 4 sub-partitions' register pools) must agree with GPU6.occupancy for every slider position
  function place(T, R, S) {
    const w = Math.ceil(T / 32), rpw = Math.ceil((R * 32) / 256) * 256, sm = Math.ceil((S + 1024) / 128) * 128;
    if (R > 255 || rpw * w > 65536 || rpw * Math.ceil(w / 4) * 4 > 65536 || sm > 227 * 1024 + 1024) return 0;
    const regs = [16384, 16384, 16384, 16384];
    let blocks = 0, warps = 0, smem = 228 * 1024, next = 0;
    for (;;) {
      if (blocks + 1 > 32 || warps + w > 64 || smem < sm) break;
      const trial = regs.slice(); let fits = true, nx = next;
      for (let k = 0; k < w; k++) { if (trial[nx] < rpw) { fits = false; break; } trial[nx] -= rpw; nx = (nx + 1) % 4; }
      if (!fits) break;
      regs.splice(0, 4, ...trial); next = nx; blocks++; warps += w; smem -= sm;
    }
    return blocks;
  }
  let bad = 0, n = 0;
  for (let T = 32; T <= 1024; T += 32) for (let R = 16; R <= 255; R += (R < 64 ? 1 : 3)) for (const KB of [0, 1, 7, 16, 31, 48, 63, 100, 113, 114, 160, 227]) {
    n++; if (G.occupancy(T, R, KB * 1024).blocks !== place(T, R, KB * 1024)) bad++;
  }
  ok(`occupancy calculator agrees with block placement in ${n} cases`, bad === 0);
  // NVIDIA's own example (Best Practices Guide, CC 7.0 has the same register file and warp limits):
  // 37 registers per thread: 128-thread blocks reach 75%, 320-thread blocks 63% (4 blocks)
  near("BPG example, 128 threads, 37 regs", G.occupancy(128, 37, 0).occ, 0.75, 0);
  equal("BPG example, 320 threads, 37 regs: blocks", G.occupancy(320, 37, 0).blocks, 4);
  near("BPG example, 320 threads: 62.5%", G.occupancy(320, 37, 0).occ, 0.625, 0);
  const a = G.occupancy(256, 64, 16384), b = G.occupancy(256, 32, 16384), c = G.occupancy(256, 128, 0);
  equal("text: 256/64/16KB", [a.blocks, a.warps, a.occ, a.limiter], [4, 32, 0.5, "registers"]);
  equal("text: 256/32/16KB", [b.blocks, b.warps, b.occ], [8, 64, 1]);
  claim("gpu-occupancy", "the registers allow 4 blocks, 32 warps: 50% occupancy");
  claim("gpu-occupancy", "With 32 registers per thread, 8 blocks, 100%");
  equal("check: 128 regs", [c.warps, c.blocks, c.occ], [16, 2, 0.25]);
  claim("gpu-occupancy", "128 × 32 = 4,096 registers");
  claim("gpu-occupancy", "room for 4 warps, so the SM holds 16 warps: 2 blocks of 8 warps, 25% occupancy");
  // reduction check: atomics are one per block whatever n is
  equal("atomics per launch = blocks in the grid = 8 per SM × 132 SMs", 8 * SMS, 1056);
  equal("one per 256 elements", [2 ** 26 / 256, 2 ** 20 / 256], [262144, 4096]);
  equal("one per 32 elements", [2 ** 26 / 32, 2 ** 20 / 32], [2097152, 32768]);
  claim("gpu-reduce", "1,056 and 1,056");
  claim("gpu-reduce", "262,144 and 4,096");
  claim("gpu-reduce", "2,097,152 and 32,768");
  equal("1024 threads × 65 regs fails", G.occupancy(1024, 65, 0).blocks, 0);
  equal("1024 threads × 64 regs: 1 block", G.occupancy(1024, 64, 0).blocks, 1);
  equal("227 KB fits once", G.occupancy(128, 32, 227 * 1024).blocks, 1);
}
{
  const rand = window.TinyLM.mulberry32(32);
  const vals = Array.from({ length: 32 }, () => Math.floor(rand() * 10));
  const st = G.shfl(vals);
  equal("figure: lane 0 after 5 shuffles = sum", st[5][0], vals.reduce((x, y) => x + y, 0));
  let bad = 0;
  for (let seed = 1; seed <= 200; seed++) {
    const rr = window.TinyLM.mulberry32(seed), v = Array.from({ length: 32 }, () => Math.floor(rr() * 1000) - 500);
    const t = G.shfl(v);
    if (t[5][0] !== v.reduce((x, y) => x + y, 0)) bad++;
    // after j steps, lane l < 32 >> j holds the sum of lanes l, l + (32 >> j), ...
    for (let j = 1; j <= 5; j++) { const live = 32 >> j; for (let l = 0; l < live; l++) { let e = 0; for (let m = l; m < 32; m += live) e += v[m]; if (t[j][l] !== e) bad++; } }
  }
  ok("shuffle tree: partial sums correct at every step (200 random warps)", bad === 0);
}
{
  const thr = 1056 * 256;
  equal("reduction threads", thr, 270336);
  claim("gpu-reduce", "each of the 270,336 threads adds about 248 elements");
  near("elements per thread", Math.round(2 ** 26 / thr), 248);
  claim("gpu-reduce", "1,056 atomics instead of 67 million");
  near("67 million", Math.floor(2 ** 26 / 1e6), 67);
  claim("gpu-reduce", `268 MB at 3.35 TB/s is ${r(us((4 * 2 ** 26) / BW), 0)} µs`);
}

// ================================================================ 6.5 Memory access
{
  // sectors: brute force over every byte each lane touches, for every slider position
  let bad = 0;
  for (const size of [4, 8, 16]) for (let stride = 1; stride <= 32; stride++) for (let offset = 0; offset <= 31; offset++) {
    const m = G.sectors(size, stride, offset);
    const sec = new Set(), lines = new Set();
    for (let k = 0; k < 32; k++) { const a = (offset + k * stride) * size; for (let b = a; b < a + size; b++) { sec.add(b >> 5); lines.add(b >> 7); } }
    if (m.sectors.length !== sec.size || m.lines.length !== lines.size || m.moved !== 32 * sec.size || Math.abs(m.eff - (32 * size) / (32 * sec.size)) > 1e-12) bad++;
  }
  ok("coalescing figure: sector and line counts match a byte-level count (3,072 inputs)", bad === 0);
  equal("consecutive floats: 4 sectors", G.sectors(4, 1, 0).sectors.length, 4);
  equal("misaligned by one float: 5 sectors", G.sectors(4, 1, 1).sectors.length, 5);
  equal("stride 2: 8 sectors", G.sectors(4, 2, 0).sectors.length, 8);
  equal("stride 8: 32 sectors", G.sectors(4, 8, 0).sectors.length, 32);
  claim("gpu-coalesce", "exactly 4 sectors");
  claim("gpu-coalesce", `straddles 5 sectors, ${r((128 / 160) * 100, 0)}%`);
  claim("gpu-coalesce", "touches 8 sectors, 50%");
  claim("gpu-coalesce", "32 sectors for 128 useful bytes, 12.5%");
  claim("gpu-coalesce", "takes 8 times as long, only if no other warp uses the rest of those sectors");
  near("float4 warp load = 16 sectors, 100%", G.sectors(16, 1, 0).eff, 1, 0);
}
{
  const N = 8192;
  claim("gpu-transpose", "32 × 32 = 1,024 bytes of sector traffic to store 128 bytes: 8 times");
  near("8x", 1024 / 128, 8);
  claim("gpu-transpose", `matrix is ${r((N * N * 4) / 1e6, 0)} MB`);
  equal("warp count", (N * N) / 32, 2097152);
  claim("gpu-transpose", "2,097,152 warp loads");
  near("strided write sectors (N ≥ 8)", G.sectors(4, 8, 0).sectors.length, 32);
  claim("gpu-smem", "4,224 bytes per block");
  equal("tile bytes", 32 * 33 * 4, 4224);
  claim("gpu-smem", "On an H800 PCIe a load from it took about 29 cycles, against 479 for HBM");
}
{
  // banks: brute-force distinct words per bank
  const brute = (words) => { const m = new Map(); words.forEach((w) => { const b = w % 32; if (!m.has(b)) m.set(b, new Set()); m.get(b).add(w); }); return Math.max(...[...m.values()].map((x) => x.size)); };
  let bad = 0;
  for (const pattern of ["row", "col", "stride", "bcast"]) for (const pad of [0, 1, 2]) for (let k = 1; k <= 33; k++) {
    const w = G.bankWords(pattern, pad, k), d = G.banks(w).degree;
    if (d !== brute(w)) bad++;
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    if (pattern === "stride" && d !== gcd(k, 32)) bad++;
    if (pattern === "col" && d !== (pad === 0 ? 32 : gcd(pad, 32))) bad++;
    if ((pattern === "row" || pattern === "bcast") && d !== 1) bad++;
  }
  ok("bank figure: conflict degree = brute force = gcd rule, for every input", bad === 0);
  equal("column of [32][32]: 32-way", G.banks(G.bankWords("col", 0, 0)).degree, 32);
  equal("column of [32][33]: none", G.banks(G.bankWords("col", 1, 0)).degree, 1);
  equal("column of [32][34]: 2-way", G.banks(G.bankWords("col", 2, 0)).degree, 2);
  claim("gpu-banks", "a 32-way conflict at 1/32 of the bandwidth");
  claim("gpu-banks", "word 33l + c, in bank (l + c) mod 32");
  claim("gpu-banks", "gcd(k, 32)-way conflict");
}
{
  const ridge = BF16 / BW;
  claim("gpu-roofline", `989.4 / 3.35 = ${r(ridge, 0)} FLOPs per byte`);
  const I = (B) => (8192 * B) / (8192 + 2 * B);
  near("intensity formula = FLOPs/bytes", I(37), (2 * 37 * 8192 ** 2) / (2 * (8192 ** 2 + 2 * 37 * 8192)), 1e-9);
  claim("gpu-roofline", `One token gives ${r(I(1), 1)} FLOP per byte, ${r((Math.min(BF16, I(1) * BW) / BF16) * 100, 2)}% of peak`);
  claim("gpu-roofline", `B = 256 (${r(I(256), 0)} FLOPs per byte) and B = 512 (${r(I(512), 0)})`);
  ok("crossing between 256 and 512", I(256) < ridge && I(512) > ridge);
  // the legacy widget's own numbers (elementwise add 1/6, softmax 1.25, GELU 2.5) are below 3 FLOP/byte
  ok("elementwise ops ≤ 2.5 FLOP/byte", [1 / 6, 1.25, 2.5].every((x) => x <= 2.5));
}
{
  const n = 2 ** 26;
  claim("gpu-ops", `0.17 FLOP/byte. On 2 26 elements, ${r((6 * n) / 1e6, 0)} MB: at least ${r(us((6 * n) / BW), 0)} µs`);
  const gvF = 2 * 4096 ** 2, gvB = 2 * (4096 ** 2 + 2 * 4096);
  claim("gpu-ops", `${r(gvF / 1e6, 1)} MFLOP over ${r(gvB / 1e6, 1)} MB, ${r(gvF / gvB, 1)} FLOP/byte. It needs at least ${r(us(gvB / BW), 1)} µs; the arithmetic alone, ${r(us(gvF / BF16), 2)} µs`);
  const linF = 2 * 512 * 1024 * 4096, linB = 2 * (512 * 1024 + 1024 * 4096 + 512 * 4096);
  near("linear layer intensity = NVIDIA's 315", +r(linF / linB, 0), 315);
  claim("gpu-ops", "315 FLOP/byte, just above the ridge");
  ok("315 > ridge", linF / linB > BF16 / BW);
  const gF = 2 * 4096 ** 3, gB = 3 * 2 * 4096 ** 2;
  claim("gpu-ops", `${r(gF / 1e9, 1)} GFLOP over ${r(gB / 1e6, 1)} MB, ${int(gF / gB)} FLOP/byte. It needs at least ${r(us(gF / BF16), 0)} µs at peak; moving its bytes, at least ${r(us(gB / BW), 0)} µs`);
  ok("add and LayerNorm are 2–3 orders of magnitude below the ridge", (BF16 / BW) / (1 / 6) > 1000 && (BF16 / BW) / 2 > 100);
}
{
  const n = 2 ** 26;
  claim("gpu-bound", `needs at least ${r(us((4 * n) / BW), 0)} µs in BF16 and ${r(us((8 * n) / BW), 0)} µs in FP32`);
  claim("gpu-bound", "time ≥ bytes ÷ 3.35 TB/s");
  claim("gpu-bound", "time ≥ FLOPs ÷ peak");
  claim("gpu-bound", `${r(ms((2 * 4096 ** 3) / FP32), 2)} ms on the FP32 lanes and ${r(us((2 * 4096 ** 3) / BF16), 0)} µs on the BF16 tensor cores`);
  const B = 64, bytes = 2 * (4096 ** 2 + 2 * B * 4096), flops = 2 * B * 4096 ** 2;
  claim("gpu-bound", `= ${r(bytes / 1e6, 1)} MB, at least ${r(us(bytes / BW), 1)} µs at 3.35 TB/s`);
  claim("gpu-bound", `to ${r(flops / 1e9, 2)} GFLOP, an intensity of ${r(flops / bytes, 0)}`);
  claim("gpu-bound", `(${r(us(flops / BF16), 1)} µs at peak)`);
  claim("gpu-bound", `${r(bytes / (2 * (4096 ** 2 + 2 * 4096)), 2)} times the bound of one`);
  claim("gpu-bound", "barely moves, to 10.3 µs: intensity rises to 62");
}

// ================================================================ 6.6 Matrix multiplication on a GPU
{
  const N = 4096;
  claim("gpu-mm-naive", "4,096 multiply-adds (8,192 FLOPs) and 8,192 loads of 4 bytes, 32,768 bytes");
  near("per-thread intensity", (2 * N) / (8 * N), 0.25);
  const bytes = N * N * 2 * N * 4;
  claim("gpu-mm-naive", `move ${r(bytes / 1e9, 0)} GB and need at least ${r(ms(bytes / BW), 0)} ms`);
  claim("gpu-mm-naive", `needs at least ${r(ms((2 * N ** 3) / FP32), 2)} ms on the FP32 lanes`);
  equal("naive grid blocks", (N / 32) ** 2, 16384);
  claim("gpu-mm-naive", "(4,096 / 32)² = 16,384 blocks");
  claim("gpu-mm-naive", "each of the 4,096 threads that needs a given element of B");
}
{
  claim("gpu-mm-tiled", "from 0.25 to T / 4 FLOPs per byte: 8 for T = 32");
  equal("two tiles", 2 * 32 * 32 * 4, 8192);
  claim("gpu-mm-tiled", "2 × 32 × 32 × 4 = 8,192 bytes");
  // one warp-FMA instruction needs 2 shared-memory passes (As broadcast + Bs row); one pass per clock per SM
  const fmaPerClock = 32 / 2;
  equal("FMAs per clock under the shared-memory limit", fmaPerClock, 16);
  claim("gpu-mm-tiled", `at most 16 FMAs per clock, ${r((fmaPerClock / 128) * 100, 1)}% of its 128 FP32 lanes`);
}
{
  // walk every phase of the figure for every option: each (C tile, t) pair appears once, loads = 2N³/T
  let bad = 0;
  for (const N of [8, 12, 16]) for (const T of [1, 2, 4]) {
    const P = G.tiledPhases(N, T), seen = new Set();
    for (let p = 0; p < P; p++) { const q = G.tiledPhase(N, T, p); seen.add(`${q.bi},${q.bj},${q.t}`); if (q.bi >= N / T || q.bj >= N / T || q.t >= N / T) bad++; }
    if (seen.size !== P || P !== (N / T) ** 3 || G.tiledLoads(N, T, P) !== (2 * N ** 3) / T) bad++;
    // the loads also equal a direct count: every output tile reads its row band of A and column band of B once
    if ((N / T) ** 2 * (2 * T * N) !== (2 * N ** 3) / T) bad++;
  }
  ok("tiled figure: phases cover every (tile, t) once; loads = 2N³/T", bad === 0);
  equal("N=16 untiled", 2 * 16 ** 3, 8192);
  equal("N=16 T=4", (2 * 16 ** 3) / 4, 2048);
  claim("gpu-tiles", "8,192 loads untiled and 2,048 with T = 4, for the same 2N³ = 8,192 FLOPs");
  const L = ((2 * 4096 ** 3) / 32) * 4;
  claim("gpu-tiles", `issues ${r(L / 1e9, 1)} GB of global loads instead of 550 GB: at least ${r(ms(L / BW), 1)} ms`);
  // shared-memory ceiling of the T = 32 kernel (gpu-mm-tiled): 16 FMAs per clock per SM
  const smemFlops = SMS * 16 * 2 * CLK, tSmem = (2 * 4096 ** 3) / smemFlops;
  claim("gpu-tiles", `needs at least ${r(ms(tSmem), 1)} ms, ${r(tSmem / ((2 * 4096 ** 3) / FP32), 0)} times the 2.05 ms of FP32 arithmetic`);
  ok("shared memory binds before HBM", tSmem > L / BW);
  near("smem bound / HBM bound ≈ 3.2", +r(tSmem / (L / BW), 1), 3.2);
}
{
  const w = (t) => (2 * t) / (t * t);
  near("8x8 words per FMA", w(8), 0.25);
  near("1x1 words per FMA", w(1), 2);
  near("SM budget words per FMA", 32 / 128, 0.25);
  claim("gpu-regtile", "16 words per 64 FMAs, 0.25 per FMA");
  claim("gpu-regtile", "at least 80 registers per thread");
  near("128x128 block tile intensity", (128 * 128) / (2 * 256), 32);
  claim("gpu-regtile", "32 FLOPs per byte in FP32, above the FP32 lanes' ridge of 20");
  claim("gpu-regtile", `${r(0.25 / w(1) * 100, 1)}% ceiling`.replace("12.5% ceiling", "12.5% ceiling"));
}
{
  near("m16n8k16 FLOPs", 2 * 16 * 8 * 16, 4096);
  equal("fragments per thread A, B, C", [256 / 32, 128 / 32, 128 / 32], [8, 4, 4]);
  claim("gpu-tensorcore", "8 elements of A, 4 of B and 4 of C");
  near("m64n256k16 FLOPs", 2 * 64 * 256 * 16, 524288);
  near("cycles at 4,096 FLOP/clk/SM", 524288 / 4096, 128);
  claim("gpu-tensorcore", "524,288 FLOPs, 128 clocks");
  equal("table FLOPs", [2 * 16 * 8 * 8, 2 * 64 * 256 * 8, 2 * 64 * 256 * 32], [2048, 262144, 1048576]);
  for (const x of ["2,048", "up to 262,144", "up to 524,288", "up to 1,048,576"]) claim("gpu-tensorcore", x);
}
{
  const tiles = (M, N, a, b) => Math.ceil(M / a) * Math.ceil(N / b);
  equal("1408 x 1536", tiles(1408, 1536, 128, 128), 132);
  equal("1408 x 1544", tiles(1408, 1544, 128, 128), 143);
  const waves = Math.ceil(143 / 132);
  equal("waves", waves, 2);
  equal("last wave", 143 - 132, 11);
  claim("gpu-waves", "143 tiles in two waves, the second keeping 11 of 132 SMs busy");
  claim("gpu-waves", `falls to ${r(((1408 * 1544) / (2 * 132 * 128 * 128)) * 100, 1)}%`);
  claim("gpu-waves", `${r((1544 / 1536 - 1) * 100, 1)}% more work`);
  equal("A100 example", [tiles(2304, 1536, 256, 128), tiles(2304, 1544, 256, 128)], [108, 117]);
  claim("gpu-waves", "N = 1,536 gives 108 tiles, N = 1,544 gives 117");
  claim("gpu-waves", "80 to 90% of an H100's peak");
  // the calculator at every slider position: useful share in (0, 1], waves × 132 ≥ tiles
  let bad = 0;
  for (let M = 128; M <= 8192; M += 64) for (let N = 128; N <= 8192; N += 8) for (const [a, b] of [[128, 128], [128, 256], [256, 128]]) {
    const t = tiles(M, N, a, b), wv = Math.ceil(t / 132), u = (M * N) / (wv * 132 * a * b);
    if (!(u > 0 && u <= 1) || wv * 132 < t || t - (wv - 1) * 132 < 1 || t - (wv - 1) * 132 > 132) bad++;
  }
  ok("wave calculator well-formed at every slider position", bad === 0);
  near("check: 133 tiles", 133 / 264, 0.5038, 1e-4);
  claim("gpu-toosmall", "133 / 264 = 50.4%");
}
{
  claim("gpu-cutlass", "32 KB per stage: four stages fill 128 KB");
  equal("stage bytes", (128 * 64 + 64 * 128) * 2, 32768);
  equal("accumulator bytes", 128 * 128 * 4, 65536);
  claim("gpu-cutlass", "take 64 KB of registers per block");
}
{
  const C = window.TinyLM.config, T = C.T, d = C.d, H = C.H, hd = d / H, ff = C.dff, V = C.V, L = C.L;
  const qkv = 3 * 2 * T * d * d, sc = H * 2 * T * T * hd, mix = H * 2 * T * T * hd, o = 2 * T * d * d, mlp = 2 * 2 * T * d * ff;
  const layer = qkv + sc + mix + o + mlp, un = 2 * T * d * V, total = L * layer + un;
  equal("tiny model products", [qkv, sc, mix, o, mlp, layer, L * layer, un, total], [12288, 2048, 2048, 4096, 32768, 53248, 106496, 28928, 135424]);
  claim("gpu-toosmall", "gives 135,424 FLOPs");
  claim("gpu-toosmall", `that is ${r((total / BF16) * 1e9, 2)} ns`);
  claim("gpu-toosmall", `fills ${r(((T * V) / (128 * 128)) * 100, 1)}% of a single 128 × 128 tile`);
  ok("weight shapes from the model", C.shapes.some(([n, s]) => n === "wu" && s[0] === 16 && s[1] === 113) && C.shapes.some(([n, s]) => n === "l0.w1" && s[1] === 64));
}

// ================================================================ 6.7 Kernels for transformers
{
  const n = 8192 * 16384;
  equal("n", n, 134217728);
  claim("gpu-fusion", "3 × (2n + 2n) = 12n bytes");
  claim("gpu-fusion", `${r((12 * n) / 1e9, 2)} GB and ${r(us((12 * n) / BW), 0)} µs unfused, ${r((4 * n) / 1e6, 0)} MB and ${r(us((4 * n) / BW), 0)} µs fused`);
  claim("gpu-fusion", "n is 134 million");
}
{
  claim("gpu-online", `above x = ${r(Math.log(3.4028234663852886e38), 1)}`);
  claim("gpu-online", `above x = ${r(Math.log(65504), 1)}`);
  const x = [2, 1, 5, 3, 8, 6, 9, 4], o = G.online(x);
  equal("final max", o.m, 9);
  near("final sum = Σ e^(x − 9)", o.d, x.reduce((a, v) => a + Math.exp(v - 9), 0), 1e-12);
  near("probabilities sum to 1", o.y.reduce((a, v) => a + v, 0), 1, 1e-12);
  ok("online = two-pass to 1e-15", o.y.every((v, i) => Math.abs(v - o.y2[i]) < 1e-15));
  const sh = G.online(x.map((v) => v + 100));
  ok("shifted: online unchanged", sh.y.every((v, i) => Math.abs(v - o.y[i]) < 1e-12));
  ok("shifted: FP32 direct formula gives NaN", sh.y32.every((v) => Number.isNaN(v)));
  ok("unshifted: FP32 direct formula finite and close", o.y32.every((v, i) => Math.abs(v - o.y[i]) < 1e-6));
  // every recurrence step against its definition
  let m = -Infinity, dd = 0, bad = 0;
  x.forEach((v, j) => { const m2 = Math.max(m, v); dd = (j ? dd * Math.exp(m - m2) : 0) + Math.exp(v - m2); m = m2; if (Math.abs(o.steps[j].d - dd) > 1e-12 || o.steps[j].m !== m) bad++; });
  ok("online recurrence matches step by step", bad === 0);
  claim("gpu-online", "count 4 memory accesses per element");
  claim("gpu-online", "3 accesses per element");
}
{
  claim("gpu-flash", `${r((8192 * 8192) / 1e6, 0)} million entries, ${r((8192 * 8192 * 2) / 1e6, 0)} MB in BF16`);
  for (const x of ["4.4 GB instead of 40.3 GB", "7.3 ms instead of 41.7 ms", "75.2 instead of 66.6 GFLOPs", "Θ(Td + T²)", "Θ(T²d²/M)", "Θ(Td/M) passes of Θ(Td) values"]) claim("gpu-flash", x);
  equal("FlashAttention passes at T = 8,192, 128-row blocks", 8192 / 128, 64);
  claim("gpu-flash", "At T = 8,192 with 128-row blocks, that is 64 passes");
  // FlashAttention-2 order: every query block streams all of K and V of its head
  const kvHead = 2 * 8192 * 128 * 2;
  claim("gpu-flash23", `each of a head's 64 query blocks reads all of K and V, ${r(kvHead / 1e6, 1)} MB in BF16: ${r((64 * kvHead) / 1e6, 0)} MB of loads`);
  ok("K, V of one head fit in the 50 MB L2", kvHead < 50e6);
}
{
  near("A100 matmul / non-matmul", 312 / 19.5, 16);
  claim("gpu-flash23", "16 times a tensor-core FLOP on an A100 (19.5 against 312 TFLOP/s)");
  near("matmul FLOPs per score, d = 128", 2 * 128 + 2 * 128, 512);
  near("exp time / matmul time per score", (1 / 3.9e12) / (512 / 989e12), 0.495, 0.001);
  claim("gpu-flash23", "3.9 against 989 TFLOP/s: half the matmul time");
  claim("gpu-flash23", "740 TFLOP/s in FP16 (75%)");
}
{
  claim("gpu-triton", "reads of 5MN + 2M elements and writes of 3MN + 2M");
  near("tutorial's expected speedup (8MN + 4M) / 2MN, large N", (8 * 1e3 * 4096 + 4 * 1e3) / (2 * 1e3 * 4096), 4, 0.001);
}
{
  equal("five kernels", [5 * 9.6, 5 * 3.8, 5 * 3.4, 5 * 2.9].map((v) => r(v, 1)), ["48.0", "19.0", "17.0", "14.5"]);
  claim("gpu-graphs", "Five kernels take 48.0, 19.0 and 17.0 µs of wall time, of which 14.5 µs is work");
}
{
  claim("gpu-profile", "reached 91% of theoretical");
  near("Luo: 1861.5 / 2039", 1861.5 / 2039, 0.913, 0.001);
}
{
  const b = G.budget(2048), one = G.budget(1), b128 = G.budget(128);
  ok("T=2048: all four weight products compute-bound", b.ops.filter((o) => o.kind === "mm").every((o) => o.bound === "compute"));
  ok("T=1: every operation memory-bound", one.ops.every((o) => o.bound === "memory"));
  claim("gpu-budget", `take ${r(b.mm * 100, 0)}% of the time, attention ${r(b.attn * 100, 1)}%, and the memory-bound elementwise operations ${r(b.elem * 100, 0)}%`);
  const wbytes = 2 * (4096 * (4096 + 2048) + 4096 * 4096 + 3 * 4096 * 14336);
  equal("weight bytes per layer", wbytes, 436207616);
  claim("gpu-budget", `reading the ${r(wbytes / 1e6, 0)} MB of weights: ${r(us(one.total), 0)} µs per layer, ${r(ms(one.total * 32), 1)} ms for 32 layers`);
  claim("gpu-budget", `The total bound is ${r(us(b128.total), 1)} µs, ${r(b128.total / one.total, 1)} times`);
  ok("T=128 weight products memory-bound, intensity ≈ 120", b128.ops.filter((o) => o.kind === "mm").every((o) => o.bound === "memory" && o.F / o.B > 115 && o.F / o.B < 125));
  claim("gpu-budget", "38 to 43% model FLOPs utilization");
  // KV-cache reads of a decode step at context L (8 KV heads × 128 dims = 1,024 values per K and per V)
  const wb = 436207616, kv = (L) => 2 * L * 1024 * 2;
  claim("gpu-budget", `${r(kv(8192) / 1e6, 1)} MB (${r((kv(8192) / wb) * 100, 1)}% more time) at L = 8,192, and ${r(kv(131072) / 1e6, 0)} MB, ${r(kv(131072) / wb, 2)} times the weights, at L = 131,072`);
  for (const L of [8192, 32768, 131072]) {
    const g = G.budget(1, L), g0 = G.budget(1, 0), a = g.ops.find((o) => o.kind === "attn");
    near(`budget(1, ${L}): attention bytes = Q, O, and K, V of L + 1 tokens`, a.B, 2 * (2 * 4096 + 2 * (L + 1) * 1024));
    near(`budget(1, ${L}): extra time over L = 0 is exactly the KV-cache bytes / bandwidth`, g.total - g0.total, kv(L) / BW, 1e-15);
    near(`budget(1, ${L}): FLOPs = 4d(L + 1)`, a.F, 4 * 4096 * (L + 1));
  }
  // with context, causal attention FLOPs = Σ over new tokens i of 4·d·(L + i + 1)
  near("budget(16, 100) attention FLOPs by direct sum", G.budget(16, 100).ops.find((o) => o.kind === "attn").F, Array.from({ length: 16 }, (_, i) => 4 * 4096 * (100 + i + 1)).reduce((x, y) => x + y, 0));
  // FLOP formulas per op against the per-token counts of sheet gpu-flops
  const T = 2048, L8 = G.llamaLayer(T);
  near("budget weight FLOPs = T × llamaLayer weights", b.ops.filter((o) => o.kind === "mm").reduce((a, o) => a + o.F, 0), T * L8.weights);
  near("causal attention = Σ over positions of 4·d·(i + 1)", b.ops.find((o) => o.kind === "attn").F, Array.from({ length: T }, (_, i) => 4 * 4096 * (i + 1)).reduce((a, v) => a + v, 0));
}

// ================================================================ computed values that appear on more than one sheet, or only in checks
{
  const n = 2 ** 26, t = (12 * n) / BW;
  ok("vector add: arithmetic units idle more than 99% of the time", 1 - n / t / FP32 > 0.99);
  claim("gpu-intensity", "more than 99% of the time");
  claim("gpu-intensity", "3 / 12 = 0.25");
  claim("gpu-tensor", "2,048 FMAs, 4,096 FLOPs per clock per SM");
  claim("gpu-hierarchy", "128 bytes per clock per SM (32 banks of 4 bytes)");
  ok("200 FLOP/byte: compute-bound on V100, memory-bound on H100", 200 > 125 / 0.9 && 200 < BF16 / BW);
  claim("gpu-hbm", "A kernel at 200 FLOPs per byte was limited by arithmetic on a V100");
  claim("gpu-copycost", `moves ${r((12 * n) / 1e6, 0)} MB through HBM, which takes ${r(us(t), 0)} µs`);
  claim("gpu-index", "(10,000 + 255) / 256 = 40 blocks in integer division, so 40 × 256 = 10,240 threads run and the last 10,240 − 10,000 = 240");
  {
    // streams check: 8 chunks
    const n = 2 ** 26, tin = (8 * n) / PCIE, tk = (12 * n) / BW, tout = (4 * n) / PCIE, eight = tin + tk / 8 + tout / 8;
    ok("8-chunk schedule feasible: each copy-out ends before the next chunk's kernel needs the engine", tout / 8 + tk / 8 <= tin / 8);
    claim("gpu-streams", `${r(ms(eight), 2)} ms`);
    claim("gpu-streams", `before the last chunk's kernel can start: ${r(ms(tin), 2)} ms`);
    claim("gpu-streams", `(0.24 / 8 = ${r(ms(tk / 8), 2)} ms) and its copy out (4.19 / 8 = ${r(ms(tout / 8), 2)} ms): ${r(ms(eight), 2)} ms`);
    near("distractor 4.75 ms = half of the 4-stream 9.50 ms", +r(ms(tin + tk / 4 + tout / 4) / 2, 2), 4.75);
    claim("gpu-streams", "4.75 ms: twice the overlap");
  }
  claim("gpu-little", "moves 128 bytes");
  near("32 × 4 bytes", 32 * 4, 128);
  const gv = (2 * (4096 ** 2 + 2 * 4096)) / BW;
  claim("gpu-bound", `${r(us(64 * gv), 0)} µs`);
  claim("gpu-bound", `The weight (${r((2 * 4096 ** 2) / 1e6, 1)} MB)`);
  claim("gpu-bound", `at least ${r(us(gv), 1)} µs on an H100`);
  claim("gpu-mm-naive", "2N FLOPs for 8N bytes: 0.25 FLOP per byte");
  claim("gpu-tiles", "goes from 0.25 to 1 FLOP per byte");
  near("T=4 intensity", 4 / 4, 1);
  claim("gpu-tiles", `${r(ms((2 * 4096 ** 3) / FP32), 2)} ms of FP32 arithmetic`);
  const one = G.budget(1), b128 = G.budget(128);
  near("displayed single-token bound rounds to 130 µs", +r(us(one.total), 0), 130);
  claim("gpu-budget", `128 × 130 µs = ${r((128 * 130) / 1e3, 1)} ms`);
  claim("gpu-budget", `About ${r(us(b128.total), 0)} µs`);
  claim("gpu-budget", "an intensity of about 120 FLOPs per byte");
  claim("gpu-toosmall", "131 SMs wait");
  claim("gpu-little", `took ${HBM_NS} ns or more on the H800 PCIe`);
}

{
  // register-tiling calculator: every option combination
  const rows = [];
  for (const t of [1, 2, 4, 8]) for (const bm of [32, 64, 128]) {
    const w = (2 * t) / (t * t), reach = Math.min(1, 0.25 / w), regs = t * t + 2 * t, thr = (bm / t) ** 2, I = (bm * bm) / (2 * (bm + bm));
    rows.push([t, bm, w, reach, regs, thr, I]);
    // intensity of a BM × BM block tile in FP32 equals BM / 4, the same T / 4 rule as the shared-memory tile
    if (Math.abs(I - bm / 4) > 1e-12) ok(`regtile intensity bm=${bm}`, false);
  }
  equal("regtile reach by t", [1, 2, 4, 8].map((t) => Math.min(1, 0.25 / ((2 * t) / (t * t)))), [0.125, 0.25, 0.5, 1]);
  ok("regtile calculator rows computed", rows.length === 12);
}

done();
