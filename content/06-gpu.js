/* Part 6 — GPUs and CUDA. The H100 SXM in detail, the CUDA programming model,
 * how an SM executes warps, memory access patterns, matrix multiplication, and
 * the kernels behind transformers.
 *
 * Hardware numbers come from NVIDIA's datasheets, whitepapers and CUDA documents,
 * or from cited measurements. Every number computed here (and every count a
 * figure shows) is recomputed in tools/verify/part06.mjs, which calls the pure
 * functions exported on window.GPU6. */
(function () {
  "use strict";
  const { h, s, clear, slider, segmented, button } = window.UI;
  const int = (x) => Math.round(x).toLocaleString("en-US");
  const fx = (x, d) => (Math.abs(x) < 0.5 * Math.pow(10, -d) ? 0 : x).toFixed(d).replace(/^-/, "−");
  const pct = (x, d = 1) => `${fx(x * 100, d)}%`;
  const tile = (v, l) => h("div", { class: "f-tile" }, h("div", { class: "f-tile-v" }, v), h("div", { class: "f-tile-l" }, l));
  const ctlRow = (...kids) => h("div", { class: "f-ctl sliders" }, ...kids);
  const optList = (vals, lab = (v) => String(v)) => vals.map((v) => ({ value: v, label: lab(v) }));

  // ------------------------------------------------------------ the GPU every sheet uses: H100 SXM5
  // NVIDIA H100 whitepaper V1.04, Table 3; H100 product page; CUDA Programming Guide, CC 9.0.
  const HW = {
    sms: 132,
    bf16: 989.4e12, // dense BF16/FP16 tensor FLOP/s (1.83 GHz)
    fp32: 66.9e12, // FP32 without tensor cores (1.98 GHz)
    bw: 3.35e12, // HBM3 bytes/s (datasheet; whitepaper 3,352 GB/s)
    clockTensor: 1.83e9,
    clock: 1.98e9,
    pcie: 64e9, // PCIe Gen 5 x16, per direction
    nvlink: 450e9, // per direction
    // compute capability 9.0 limits
    maxWarpsSM: 64, maxBlocksSM: 32, regsSM: 65536, regsBlock: 65536, maxRegsThread: 255,
    subPartitions: 4, regUnit: 256, smemSM: 233472, smemBlockOptin: 232448, smemReserved: 1024, smemUnit: 128,
  };

  // ------------------------------------------------------------ pure functions (also used by the verifier)
  const G = {
    HW,
    grid1d(n, B) {
      const grid = Math.ceil(n / B);
      return { grid, threads: grid * B, idle: grid * B - n };
    },
    grid2d(R, C, bx, by) {
      const gx = Math.ceil(C / bx), gy = Math.ceil(R / by);
      return { gx, gy, blocks: gx * gy, threads: gx * gy * bx * by, idle: gx * gy * bx * by - R * C };
    },
    // Llama 3 8B, one layer, one token, attending to T tokens (Dubey et al. 2024, Table 3)
    llamaLayer(T) {
      const d = 4096, kv = 1024, ff = 14336, heads = 32;
      const weights = 2 * d * (d + kv + kv + d) + 2 * 3 * d * ff;
      const attn = 2 * 2 * T * d; // scores q·k and mixing p·v, 32 heads × 128 dims
      // elementwise, counting exp, division, sqrt as 1 FLOP each:
      // 2 RMSNorms × 4/elem, RoPE 3/elem on q and k, SwiGLU gate 5/elem, softmax 6/score, 2 residual adds
      const other = 2 * 4 * d + 3 * (d + kv) + 5 * ff + 2 * d + 6 * heads * T;
      return { weights, attn, other, share: (weights + attn) / (weights + attn + other) };
    },
    // One warp scheduler, W warps. Each warp repeats: one load, then c instructions that need its value.
    // A load's value is usable L cycles after it issues. Policy: greedy-then-oldest; keep issuing from the
    // warp that issued last cycle while it is ready, otherwise take the lowest-numbered ready warp.
    warpsim(W, c, L, T) {
      const st = Array.from({ length: W }, () => ({ next: "load", readyAt: 0, left: 0 }));
      const kind = Array.from({ length: W }, () => new Array(T).fill(""));
      const issue = new Array(T).fill(-1);
      let last = -1;
      for (let t = 0; t < T; t++) {
        let pick = last >= 0 && st[last].readyAt <= t ? last : -1;
        if (pick < 0) for (let w = 0; w < W; w++) if (st[w].readyAt <= t) { pick = w; break; }
        last = pick;
        for (let w = 0; w < W; w++) {
          const x = st[w];
          if (w === pick) {
            if (x.next === "load") { kind[w][t] = "L"; x.readyAt = t + L; x.next = "comp"; x.left = c; }
            else { kind[w][t] = "C"; x.left--; x.readyAt = t + 1; if (x.left === 0) x.next = "load"; }
          } else kind[w][t] = x.readyAt <= t ? "r" : "w";
        }
        issue[t] = pick;
      }
      return { kind, issue };
    },
    warpUtil: (W, c, L) => Math.min(1, (W * (c + 1)) / (L + c)),
    warpsNeeded: (c, L) => Math.ceil((L + c) / (c + 1)),
    // Branch divergence in a 128-thread block (4 warps) running if (cond) A cycles; else B cycles.
    divergeLanes(cond, t, seed) {
      const rand = window.TinyLM.mulberry32(seed);
      const x = Array.from({ length: 128 }, () => rand() * 2 - 1);
      return Array.from({ length: 128 }, (_, i) =>
        cond === "even" ? i % 2 === 0 : cond === "lt" ? i < t : cond === "warp" ? Math.floor(i / 32) % 2 === 0 : x[i] > 0);
    },
    diverge(takes, A, B) {
      const warps = [0, 1, 2, 3].map((w) => {
        const lanes = takes.slice(32 * w, 32 * w + 32), nIf = lanes.filter(Boolean).length;
        const cost = (nIf > 0 ? A : 0) + (nIf < 32 ? B : 0);
        return { lanes, nIf, cost, divergent: nIf > 0 && nIf < 32 };
      });
      const total = warps.reduce((a, w) => a + w.cost, 0);
      const useful = warps.reduce((a, w) => a + w.nIf * A + (32 - w.nIf) * B, 0);
      return { warps, total, useful, eff: useful / (32 * total), divergent: warps.filter((w) => w.divergent).length };
    },
    // Resident blocks per SM for compute capability 9.0, following the CUDA Toolkit's cuda_occupancy.h:
    // registers are allocated per warp in units of 256 within each of 4 sub-partitions (16,384 each);
    // shared memory per block = static + dynamic + 1 KB reserved, rounded up to 128 bytes, out of 228 KB.
    occupancy(T, R, S) {
      const w = Math.ceil(T / 32);
      const out = { warpsPerBlock: w, regsPerWarp: Math.ceil((R * 32) / HW.regUnit) * HW.regUnit, smemPerBlock: Math.ceil((S + HW.smemReserved) / HW.smemUnit) * HW.smemUnit, fail: null };
      out.limWarps = Math.floor(HW.maxWarpsSM / w);
      out.limBlocks = HW.maxBlocksSM;
      const perSub = Math.floor(HW.regsSM / HW.subPartitions / out.regsPerWarp);
      out.warpsPerSubByRegs = perSub;
      out.limRegs = Math.floor((perSub * HW.subPartitions) / w);
      out.limSmem = Math.floor(HW.smemSM / out.smemPerBlock);
      if (R > HW.maxRegsThread) out.fail = `more than ${HW.maxRegsThread} registers per thread`;
      else if (out.regsPerWarp * Math.ceil(w / HW.subPartitions) * HW.subPartitions > HW.regsBlock || out.regsPerWarp * w > HW.regsBlock) out.fail = "the block needs more than 65,536 registers";
      else if (out.smemPerBlock > HW.smemBlockOptin + HW.smemReserved) out.fail = "the block needs more than 227 KB of shared memory";
      out.blocks = out.fail ? 0 : Math.min(out.limWarps, out.limBlocks, out.limRegs, out.limSmem);
      out.warps = out.blocks * w;
      out.occ = out.warps / HW.maxWarpsSM;
      out.limiter = out.fail ? "launch fails" : ["warp slots", "block slots", "registers", "shared memory"][[out.limWarps, out.limBlocks, out.limRegs, out.limSmem].indexOf(out.blocks)];
      return out;
    },
    // __shfl_down_sync tree sum: after each step, lane l adds the value of lane l + d (its own value if l + d > 31)
    shfl(vals) {
      const steps = [vals.slice()];
      for (let d = 16; d >= 1; d /= 2) {
        const v = steps[steps.length - 1];
        steps.push(v.map((x, l) => x + (l + d < 32 ? v[l + d] : x)));
      }
      return steps;
    },
    // Little's law for HBM: bandwidth an SM array reaches with a given number of bytes in flight
    littleBW: (warps, bytes, ilp, latNs) => Math.min(HW.bw, (HW.sms * warps * 32 * bytes * ilp) / (latNs * 1e-9)),
    // One warp load: lane k reads `size` bytes at byte address (offset + k * stride) * size from a 128-byte-aligned base.
    // Memory moves whole 32-byte sectors; four sectors make a 128-byte cache line.
    sectors(size, stride, offset) {
      const addrs = Array.from({ length: 32 }, (_, k) => (offset + k * stride) * size);
      const sec = new Set(), lines = new Set();
      addrs.forEach((a) => { for (let b = a; b < a + size; b++) { sec.add(Math.floor(b / 32)); lines.add(Math.floor(b / 128)); } });
      const req = 32 * size, uniq = new Set(); addrs.forEach((a) => { for (let b = a; b < a + size; b++) uniq.add(b); });
      return { addrs, sectors: [...sec].sort((a, b) => a - b), lines: [...lines].sort((a, b) => a - b), req, useful: uniq.size, moved: 32 * sec.size, eff: uniq.size / (32 * sec.size) };
    },
    // Shared memory: 32 banks of 4-byte words; word w is in bank w mod 32. A warp needs one pass per
    // distinct word in its most-used bank (threads reading the same word share it: broadcast).
    bankWords(pattern, pad, k) {
      return Array.from({ length: 32 }, (_, l) => (pattern === "row" ? l : pattern === "col" ? l * (32 + pad) : pattern === "stride" ? l * k : 0));
    },
    banks(words) {
      const per = Array.from({ length: 32 }, () => new Map());
      words.forEach((w, l) => { const m = per[w % 32]; if (!m.has(w)) m.set(w, []); m.get(w).push(l); });
      const degree = Math.max(...per.map((m) => m.size));
      return { per, degree, passes: degree, banksUsed: per.filter((m) => m.size > 0).length };
    },
    // Tiled C = A·B, N × N, tiles T × T: output tiles in row-major order, each with N/T phases.
    // Phase p (0-based, over the whole multiply) loads one T × T tile of A and one of B.
    tiledPhase(N, T, p) {
      const n = N / T, tileIdx = Math.floor(p / n);
      return { bi: Math.floor(tileIdx / n), bj: tileIdx % n, t: p % n };
    },
    tiledLoads: (N, T, phasesDone) => phasesDone * 2 * T * T,
    tiledPhases: (N, T) => (N / T) ** 3,
    // Online softmax (Milakov & Gimelshein 2018) next to the two-pass safe softmax and the direct formula in FP32.
    online(x) {
      const f32 = Math.fround;
      let m = -Infinity, d = 0;
      const steps = x.map((v, j) => {
        const mNew = Math.max(m, v), scale = m === -Infinity ? 0 : Math.exp(m - mNew);
        d = d * scale + Math.exp(v - mNew);
        m = mNew;
        return { j, x: v, m, scale, d, first: j === 0 };
      });
      const y = x.map((v) => Math.exp(v - m) / d);
      const mx = Math.max(...x), d2 = x.reduce((a, v) => a + Math.exp(v - mx), 0), y2 = x.map((v) => Math.exp(v - mx) / d2);
      const e32 = x.map((v) => f32(Math.exp(v))), s32 = e32.reduce((a, v) => f32(a + v), 0), y32 = e32.map((v) => f32(v / s32));
      return { steps, m, d, y, y2, y32 };
    },
    // One Llama 3 8B layer in BF16 on T new tokens after L earlier ones in the KV cache:
    // FLOPs, minimum bytes and the roofline time of each operation.
    budget(T, L = 0) {
      const d = 4096, kv = 1024, ff = 14336, b = 2;
      const ops = [
        ["RMSNorm", 4 * T * d, b * (2 * T * d + d), "elem"],
        ["Q, K, V projection", 2 * T * d * (d + 2 * kv), b * (T * d + d * (d + 2 * kv) + T * (d + 2 * kv)), "mm"],
        ["RoPE on q and k", 3 * T * (d + kv), b * 2 * T * (d + kv), "elem"],
        ["attention, causal", 4 * d * T * (L + (T + 1) / 2), b * (2 * T * d + 2 * (L + T) * kv), "attn"],
        ["output projection", 2 * T * d * d, b * (2 * T * d + d * d), "mm"],
        ["residual add", T * d, b * 3 * T * d, "elem"],
        ["RMSNorm", 4 * T * d, b * (2 * T * d + d), "elem"],
        ["gate and up projections", 2 * T * d * 2 * ff, b * (T * d + d * 2 * ff + T * 2 * ff), "mm"],
        ["SiLU(gate) × up", 5 * T * ff, b * 3 * T * ff, "elem"],
        ["down projection", 2 * T * ff * d, b * (T * ff + ff * d + T * d), "mm"],
        ["residual add", T * d, b * 3 * T * d, "elem"],
      ].map(([name, F, B, kind]) => ({ name, F, B, kind, t: Math.max(F / HW.bf16, B / HW.bw), bound: F / HW.bf16 >= B / HW.bw ? "compute" : "memory" }));
      const total = ops.reduce((a, o) => a + o.t, 0);
      const share = (k) => ops.filter((o) => o.kind === k).reduce((a, o) => a + o.t, 0) / total;
      return { ops, total, mm: share("mm"), attn: share("attn"), elem: share("elem") };
    },
  };
  window.GPU6 = G;

  // ------------------------------------------------------------ fig: thread index grid (1-D, 2-D, warps of a block)
  Deck.fig("gpu-grid", (el, sp) => {
    const st = Object.assign({ mode: "1d", n: 20, B: 8, pick: 13, R: 6, C: 10, bx: 4, by: 4, shape: "32x8" }, sp);
    const ctl = ctlRow();
    const host = h("div", { class: "scroll-x gpu-grid-host" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const out = h("div", { class: "f-tiles" });
    el.append(ctl, host, expl, out);
    let sel = null; // selected cell in 2-D mode: [row, col] of the padded grid

    function svgFor(W, H, label) {
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "gpu-svg", role: "img", "aria-label": label, style: `max-width:${W}px` });
      const pid = window.UI.nextId("gpu-hatch");
      svg.appendChild(s("defs", null, s("pattern", { id: pid, width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, s("line", { x1: 0, y1: 0, x2: 0, y2: 5, class: "gpu-hatch" }))));
      svg.hatch = `url(#${pid})`;
      return svg;
    }

    function draw1d() {
      const { n, B } = st;
      st.pick = Math.min(st.pick, n - 1);
      const g = G.grid1d(n, B);
      const cw = B <= 8 ? 30 : B <= 16 ? 24 : 19, chh = 22, lab = 64, W = lab + B * cw + 6, H = g.grid * (chh + 6) + 4;
      const svg = svgFor(W, H, `${g.grid} blocks of ${B} threads covering ${n} elements`);
      svg.style.minWidth = `${Math.min(W, 520)}px`;
      for (let b = 0; b < g.grid; b++) {
        const y = 2 + b * (chh + 6);
        svg.appendChild(s("text", { x: lab - 8, y: y + 15, class: "gpu-lab", "text-anchor": "end" }, `block ${b}`));
        for (let t = 0; t < B; t++) {
          const i = b * B + t, idle = i >= n, x = lab + t * cw;
          const gEl = s("g", { class: "gpu-cell" + (idle ? " idle" : "") + (i === st.pick ? " pick" : ""), "data-tip": `blockIdx.x = ${b}, threadIdx.x = ${t}: i = ${b} × ${B} + ${t} = ${i}` + (idle ? ` ≥ n = ${n}, so this thread does nothing` : "") });
          gEl.appendChild(s("rect", { x: x + 1, y, width: cw - 2, height: chh, style: idle ? `fill:${svg.hatch}` : null }));
          if (!idle) gEl.appendChild(s("text", { x: x + cw / 2, y: y + 15, class: "gpu-cell-t", "text-anchor": "middle" }, String(i)));
          svg.appendChild(gEl);
        }
      }
      clear(host).appendChild(svg);
      const b = Math.floor(st.pick / B), t = st.pick % B;
      expl.innerHTML = `Element <b>${st.pick}</b> belongs to block <code>blockIdx.x = ${b}</code>, thread <code>threadIdx.x = ${t}</code>: ${b} × ${B} + ${t} = ${st.pick}. Hatched threads have i ≥ n and fail the bounds check.`;
      clear(out).append(tile(`⌈${n} / ${B}⌉ = ${g.grid}`, "blocks in the grid"), tile(int(g.threads), "threads launched"), tile(int(g.idle), "idle threads"));
    }

    function draw2d() {
      const { R, C, bx, by } = st;
      const g = G.grid2d(R, C, bx, by);
      const cols = g.gx * bx, rows = g.gy * by;
      const cs = 26, top = 18, left = 22, W = left + cols * cs + 4, H = top + rows * cs + 4;
      const svg = svgFor(W, H, `${g.gx} × ${g.gy} blocks over a ${R} × ${C} matrix`);
      if (!sel || sel[0] >= rows || sel[1] >= cols) sel = [R - 1, C - 1];
      for (let c = 0; c < cols; c++) svg.appendChild(s("text", { x: left + c * cs + cs / 2, y: 12, class: "gpu-axis", "text-anchor": "middle" }, String(c)));
      for (let r = 0; r < rows; r++) svg.appendChild(s("text", { x: left - 5, y: top + r * cs + 17, class: "gpu-axis", "text-anchor": "end" }, String(r)));
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const bX = Math.floor(c / bx), bY = Math.floor(r / by), tx = c % bx, ty = r % by;
        const idle = r >= R || c >= C, idx = r * C + c;
        const tipTxt = `block (${bX}, ${bY}), thread (${tx}, ${ty}): col = ${bX}·${bx} + ${tx} = ${c}, row = ${bY}·${by} + ${ty} = ${r}` + (idle ? ": outside the matrix, idle" : `, idx = ${r}·${C} + ${c} = ${idx}`);
        const gEl = s("g", { class: `gpu-cell b${(bX + bY) % 2}` + (idle ? " idle" : "") + (sel[0] === r && sel[1] === c ? " pick" : ""), "data-tip": tipTxt });
        gEl.appendChild(s("rect", { x: left + c * cs + 1, y: top + r * cs + 1, width: cs - 2, height: cs - 2, style: idle ? `fill:${svg.hatch}` : null }));
        if (!idle) gEl.appendChild(s("text", { x: left + c * cs + cs / 2, y: top + r * cs + 17, class: "gpu-cell-t", "text-anchor": "middle" }, String(idx)));
        gEl.addEventListener("click", () => { sel = [r, c]; draw(); });
        svg.appendChild(gEl);
      }
      for (let gy = 0; gy < g.gy; gy++) for (let gx = 0; gx < g.gx; gx++) svg.appendChild(s("rect", { x: left + gx * bx * cs, y: top + gy * by * cs, width: bx * cs, height: by * cs, class: "gpu-blk" }));
      svg.appendChild(s("rect", { x: left, y: top, width: C * cs, height: R * cs, class: "gpu-mat" }));
      clear(host).appendChild(svg);
      const [r, c] = sel, bX = Math.floor(c / bx), bY = Math.floor(r / by), tx = c % bx, ty = r % by;
      expl.innerHTML = `Thread (x = ${tx}, y = ${ty}) of block (${bX}, ${bY}): col = ${bX} × ${bx} + ${tx} = ${c}, row = ${bY} × ${by} + ${ty} = ${r}` +
        (r < R && c < C ? `, so it handles element ${r} × ${C} + ${c} = <b>${r * C + c}</b>.` : `: outside the ${R} × ${C} matrix, so the bounds check <code>if (row &lt; ${R} &amp;&amp; col &lt; ${C})</code> skips it.`) + " Click any cell.";
      clear(out).append(tile(`${g.gx} × ${g.gy}`, "grid (blocks in x × y)"), tile(int(g.threads), "threads launched"), tile(int(g.idle), "idle threads"));
    }

    function drawWarps() {
      const [bx, by] = st.shape.split("x").map(Number);
      const cs = Math.min(16, Math.floor(384 / Math.max(bx, by))), W = bx * cs + 4, H = by * cs + 4;
      const svg = svgFor(W, H, `a ${bx} × ${by} block split into warps`);
      for (let y = 0; y < by; y++) for (let x = 0; x < bx; x++) {
        const k = x + y * bx, w = Math.floor(k / 32), lane = k % 32;
        const gEl = s("g", { class: `gpu-cell w${w % 2}`, "data-tip": `threadIdx = (${x}, ${y}): linear id ${x} + ${y}·${bx} = ${k}, warp ${w}, lane ${lane}` });
        gEl.appendChild(s("rect", { x: 2 + x * cs, y: 2 + y * cs, width: cs, height: cs }));
        svg.appendChild(gEl);
      }
      const nw = (bx * by) / 32;
      for (let w = 0; w < nw; w++) {
        const k0 = w * 32, x0 = k0 % bx, y0 = Math.floor(k0 / bx);
        const span = Math.max(1, 32 / bx); // rows per warp when bx ≤ 32
        const cxx = bx >= 32 ? 2 + (x0 + 16) * cs : 2 + (bx * cs) / 2, cyy = 2 + (y0 + span / 2) * cs + 4;
        svg.appendChild(s("text", { x: cxx, y: cyy, class: "gpu-warp-t", "text-anchor": "middle" }, `warp ${w}`));
      }
      clear(host).appendChild(svg);
      const rowsPerWarp = 32 / bx;
      expl.innerHTML = `Threads are numbered x + y × ${bx} and cut into warps of 32 consecutive numbers. With <code>blockDim = (${bx}, ${by})</code> each warp covers ` +
        (bx >= 32 ? `${32 / bx === 1 ? "one full row" : "part of a row"} of the block.` : `${rowsPerWarp} rows of ${bx} threads.`) + " Hover a thread for its warp and lane.";
      clear(out).append(tile(String(nw), "warps per block"), tile(bx >= 32 ? "1" : String(rowsPerWarp), "block rows per warp"), tile(String(Math.min(HW.maxBlocksSM, Math.floor(HW.maxWarpsSM / nw))), "such blocks per SM, at most"));
    }

    const draw = () => (st.mode === "1d" ? draw1d() : st.mode === "2d" ? draw2d() : drawWarps());
    if (st.mode === "1d") {
      const pick = slider({ label: "element i", min: 0, max: st.n - 1, step: 1, value: st.pick, onInput: (v) => { st.pick = v; draw(); } });
      ctl.append(
        slider({ label: "elements n", min: 1, max: 48, step: 1, value: st.n, onInput: (v) => { st.n = v; pick.input.max = String(v - 1); if (st.pick > v - 1) pick.set(v - 1); st.pick = Math.min(st.pick, v - 1); draw(); } }),
        segmented({ label: "threads per block", options: optList([4, 8, 16, 32]), value: st.B, onChange: (v) => { st.B = v; draw(); } }),
        pick);
    } else if (st.mode === "2d") {
      ctl.append(
        slider({ label: "rows", min: 1, max: 12, step: 1, value: st.R, onInput: (v) => { st.R = v; draw(); } }),
        slider({ label: "columns", min: 1, max: 16, step: 1, value: st.C, onInput: (v) => { st.C = v; draw(); } }),
        segmented({ label: "blockDim.x", options: optList([2, 4, 8]), value: st.bx, onChange: (v) => { st.bx = v; draw(); } }),
        segmented({ label: "blockDim.y", options: optList([2, 4, 8]), value: st.by, onChange: (v) => { st.by = v; draw(); } }));
    } else {
      ctl.append(segmented({ label: "blockDim (x, y), 256 threads", options: [{ value: "32x8", label: "32 × 8" }, { value: "16x16", label: "16 × 16" }, { value: "8x32", label: "8 × 32" }], value: st.shape, onChange: (v) => { st.shape = v; draw(); } }));
    }
    draw();
  });

  // ------------------------------------------------------------ fig: a DGX H100 (GPUs, NVSwitch fabric, host CPUs)
  Deck.fig("gpu-dgx", (el) => {
    const W = 640, H = 284;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "gpu-svg gpu-dgx", role: "img", "aria-label": "Eight H100 GPUs connected to an NVSwitch fabric and two host CPUs" });
    const box = (x, y, w, hh, l1, l2, cls, tip) => {
      const g = s("g", { class: `gpu-box ${cls || ""}`, "data-tip": tip || null });
      g.appendChild(s("rect", { x, y, width: w, height: hh, rx: 2 }));
      g.appendChild(s("text", { x: x + w / 2, y: y + (l2 ? hh / 2 - 2 : hh / 2 + 4), class: "gpu-box-l", "text-anchor": "middle" }, l1));
      if (l2) g.appendChild(s("text", { x: x + w / 2, y: y + hh / 2 + 12, class: "gpu-box-s", "text-anchor": "middle" }, l2));
      svg.appendChild(g);
    };
    const gx = (i) => 12 + i * 78, gw = 66, gy = 112, gh = 42;
    // PCIe from the CPUs (drawn first, under the boxes)
    [0, 1].forEach((c) => {
      const cx = c === 0 ? 170 : 470;
      for (let i = 4 * c; i < 4 * c + 4; i++) svg.appendChild(s("line", { x1: cx, y1: 48, x2: gx(i) + gw / 2, y2: gy, class: "gpu-pcie" }));
    });
    for (let i = 0; i < 8; i++) svg.appendChild(s("line", { x1: gx(i) + gw / 2, y1: gy + gh, x2: gx(i) + gw / 2, y2: 206, class: "gpu-nvl" }));
    box(90, 10, 160, 38, "Xeon 8480C", "56 cores, host CPU 0", "", "Intel Xeon 8480C, 56 cores: PCIe Gen 5 x16 to each of its 4 GPUs, 64 GB/s per direction");
    box(390, 10, 160, 38, "Xeon 8480C", "56 cores, host CPU 1", "", "Intel Xeon 8480C, 56 cores: PCIe Gen 5 x16 to each of its 4 GPUs, 64 GB/s per direction");
    for (let i = 0; i < 8; i++) box(gx(i), gy, gw, gh, `H100 ${i}`, "80 GB", "gpu", `H100 SXM ${i}: 18 NVLink 4 links × 25 GB/s = 450 GB/s each way`);
    svg.appendChild(s("rect", { x: 8, y: 206, width: 624, height: 72, class: "gpu-fabric" }));
    svg.appendChild(s("text", { x: 16, y: 221, class: "gpu-lab" }, "NVSwitch fabric: any GPU to any GPU at 450 GB/s each way"));
    for (let j = 0; j < 4; j++) box(22 + j * 152, 230, 136, 40, `NVSwitch ${j}`, "3rd gen, 64 NVLink ports", "sw", "third-generation NVSwitch chip: 64 ports of NVLink 4");
    svg.appendChild(s("text", { x: 320, y: 80, class: "gpu-note", "text-anchor": "middle" }, "PCIe Gen 5 x16: 64 GB/s each way"));
    svg.appendChild(s("text", { x: gx(0) + gw / 2 + 6, y: 186, class: "gpu-lab" }, "18 links"));
    el.appendChild(h("div", { class: "scroll-x" }, svg));
    el.appendChild(h("p", { class: "f-note" }, "Eight GPUs × 900 GB/s = 7.2 TB/s of total GPU-to-GPU bandwidth. NVIDIA does not document how each GPU's 18 links divide among the four switch chips, so the fabric is drawn as one block. The PCIe side is simplified to one x16 link per GPU."));
  });

  // ------------------------------------------------------------ fig: one warp scheduler hiding memory latency
  Deck.fig("gpu-warpsched", (el, sp) => {
    const st = { W: sp.W || 4, c: sp.c || 4, L: sp.L || 30 };
    const host = h("div", { class: "scroll-x" });
    const out = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.append(ctlRow(
      slider({ label: "warps on this scheduler, W", min: 1, max: 16, step: 1, value: st.W, onInput: (v) => { st.W = v; draw(); } }),
      slider({ label: "load latency L, cycles", min: 4, max: 600, step: 1, value: st.L, onInput: (v) => { st.L = v; draw(); } }),
      slider({ label: "instructions per load, c", min: 1, max: 16, step: 1, value: st.c, onInput: (v) => { st.c = v; draw(); } })), host, expl, out,
      h("div", { class: "legend" }, [["s2", "load issued"], ["s1", "instruction issued"], ["rule", "waiting for memory"], ["s4", "ready, not picked"]].map(([c, l]) => h("span", { class: "key" }, h("span", { class: `sw gpu-sw-${c}` }), l))));
    function draw() {
      const { W, c, L } = st;
      const T = Math.min(2400, W * (c + 1) + 2 * (L + c) + 8);
      const sim = G.warpsim(W, c, L, T);
      const lab = 58, Wd = 640, rh = 14, gap = 3, top = 6, plotW = Wd - lab - 8;
      const Hd = top + (W + 1) * (rh + gap) + 34;
      const X = (t) => lab + (t / T) * plotW;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "gpu-svg gpu-ws", role: "img", "aria-label": `${W} warps on one scheduler over ${T} cycles` });
      const cls = { L: "gpu-ws-L", C: "gpu-ws-C", w: "gpu-ws-w", r: "gpu-ws-r" };
      const name = { L: "issues its load", C: "issues an instruction", w: "waits for memory", r: "is ready but another warp issues" };
      for (let w = 0; w < W; w++) {
        const y = top + w * (rh + gap);
        svg.appendChild(s("text", { x: lab - 6, y: y + rh - 3, class: "gpu-axis", "text-anchor": "end" }, `warp ${w}`));
        const k = sim.kind[w];
        let t0 = 0;
        for (let t = 1; t <= T; t++) {
          if (t === T || k[t] !== k[t0]) {
            if (k[t0]) svg.appendChild(s("rect", { x: X(t0), y, width: Math.max(0.8, X(t) - X(t0)), height: rh, class: cls[k[t0]], "data-tip": `warp ${w}, cycles ${t0}–${t - 1}: ${name[k[t0]]}` }));
            t0 = t;
          }
        }
      }
      const yI = top + W * (rh + gap) + 2;
      svg.appendChild(s("text", { x: lab - 6, y: yI + rh - 3, class: "gpu-axis", "text-anchor": "end" }, "issue"));
      let t0 = 0;
      for (let t = 1; t <= T; t++) {
        if (t === T || (sim.issue[t] >= 0) !== (sim.issue[t0] >= 0)) {
          if (sim.issue[t0] >= 0) svg.appendChild(s("rect", { x: X(t0), y: yI, width: Math.max(0.8, X(t) - X(t0)), height: rh, class: "gpu-ws-busy", "data-tip": `cycles ${t0}–${t - 1}: an instruction issues every cycle` }));
          t0 = t;
        }
      }
      svg.appendChild(s("line", { x1: lab, x2: lab + plotW, y1: yI + rh + 3, y2: yI + rh + 3, class: "gpu-ws-axis" }));
      const step = [10, 20, 50, 100, 200, 250, 500].find((v) => T / v <= 8) || 500;
      for (let t = 0; t <= T; t += step) {
        svg.appendChild(s("line", { x1: X(t), x2: X(t), y1: yI + rh + 3, y2: yI + rh + 7, class: "gpu-ws-axis" }));
        svg.appendChild(s("text", { x: X(t), y: yI + rh + 18, class: "gpu-axis", "text-anchor": "middle" }, String(t)));
      }
      svg.appendChild(s("text", { x: lab + plotW, y: Hd - 2, class: "gpu-axis", "text-anchor": "end" }, "clock cycles →"));
      clear(host).appendChild(svg);
      const U = G.warpUtil(W, c, L), need = G.warpsNeeded(c, L);
      expl.innerHTML = `One warp issues ${c + 1} instructions every ${L + c} cycles, so ${W} warp${W > 1 ? "s" : ""} keep the issue slot busy min(1, ${W} × ${c + 1} / ${L + c}) = <b>${pct(U, 1)}</b> of the time once the start-up is over. Full use needs ⌈${L + c} / ${c + 1}⌉ = <b>${need}</b> warps.`;
      clear(out).append(tile(pct(U, 1), "issue slot busy, steady state"), tile(String(need), "warps needed per scheduler"), tile(`${4 * need}${4 * need > 64 ? " > 64" : ""}`, "per SM (4 schedulers, 64 max)"));
    }
    draw();
  });

  // ------------------------------------------------------------ fig: branch divergence in a 128-thread block
  Deck.fig("gpu-diverge", (el, sp) => {
    const st = { cond: sp.cond || "even", t: 40, A: 12, B: 8, seed: 11 };
    const host = h("div", { class: "scroll-x" });
    const out = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const tSl = slider({ label: "t", min: 0, max: 128, step: 1, value: st.t, onInput: (v) => { st.t = v; draw(); } });
    const reseed = button("New random x", () => { st.seed++; draw(); });
    el.append(
      h("div", { class: "f-ctl" }, segmented({ label: "condition", options: [{ value: "even", label: "threadIdx.x % 2 == 0" }, { value: "lt", label: "threadIdx.x < t" }, { value: "warp", label: "(threadIdx.x / 32) % 2 == 0" }, { value: "rand", label: "x[i] > 0" }], value: st.cond, onChange: (v) => { st.cond = v; draw(); } })),
      ctlRow(slider({ label: "if-path cycles, A", min: 1, max: 32, step: 1, value: st.A, onInput: (v) => { st.A = v; draw(); } }), slider({ label: "else-path cycles, B", min: 1, max: 32, step: 1, value: st.B, onInput: (v) => { st.B = v; draw(); } }), tSl, reseed),
      host, expl, out);
    function draw() {
      tSl.hidden = st.cond !== "lt"; reseed.hidden = st.cond !== "rand";
      const takes = G.divergeLanes(st.cond, st.t, st.seed), r = G.diverge(takes, st.A, st.B);
      const cw = 13, lab = 64, right = 120, Wd = lab + 32 * cw + right, rowH = 13, gap = 5;
      let y = 4;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${4 + 4 * (2 * rowH + gap + 12)}`, class: "gpu-svg gpu-dv", role: "img", "aria-label": "lanes of four warps on the two sides of a branch" });
      r.warps.forEach((w, wi) => {
        svg.appendChild(s("text", { x: 0, y: y + rowH + 4, class: "gpu-lab" }, `warp ${wi}`));
        [[true, "if", st.A], [false, "else", st.B]].forEach(([side, nm, cyc], k) => {
          const yy = y + k * (rowH + 1);
          const runs = w.lanes.some((x) => x === side);
          svg.appendChild(s("text", { x: lab - 6, y: yy + rowH - 3, class: "gpu-axis", "text-anchor": "end" }, nm));
          w.lanes.forEach((x, l) => {
            const on = x === side;
            svg.appendChild(s("rect", { x: lab + l * cw + 0.5, y: yy, width: cw - 1, height: rowH, class: !runs ? "gpu-dv-skip" : on ? (side ? "gpu-dv-if" : "gpu-dv-else") : "gpu-dv-off", "data-tip": `thread ${32 * wi + l} (lane ${l}): ${x ? "takes the if side" : "takes the else side"}` + (runs && !on ? `, masked off for these ${cyc} cycles` : "") }));
          });
          svg.appendChild(s("text", { x: lab + 32 * cw + 8, y: yy + rowH - 3, class: "gpu-axis" }, runs ? `${cyc} cycles` : "skipped"));
        });
        svg.appendChild(s("text", { x: Wd - 2, y: y + rowH + 4, class: "gpu-dv-cost", "text-anchor": "end" }, `= ${w.cost}`));
        y += 2 * rowH + gap + 12;
      });
      clear(host).appendChild(svg);
      expl.innerHTML = `${r.divergent} of 4 warps diverge. Each divergent warp pays A + B = ${st.A + st.B} cycles; a uniform warp pays only its own side. The block's warps spend <b>${r.total}</b> cycles in the branch.`;
      clear(out).append(tile(`${r.divergent} of 4`, "divergent warps"), tile(String(r.total), "warp-cycles in the branch"), tile(pct(r.eff, 1), "lane-cycles doing useful work"));
    }
    draw();
  });

  // ------------------------------------------------------------ fig: occupancy calculator, compute capability 9.0
  Deck.fig("gpu-occ", (el, sp) => {
    const st = { T: sp.T || 256, R: sp.R || 64, S: sp.S ?? 16 };
    const bars = h("div", { class: "gpu-occ-bars" });
    const out = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.append(ctlRow(
      slider({ label: "threads per block", min: 32, max: 1024, step: 32, value: st.T, onInput: (v) => { st.T = v; draw(); } }),
      slider({ label: "registers per thread", min: 16, max: 255, step: 1, value: st.R, onInput: (v) => { st.R = v; draw(); } }),
      slider({ label: "shared memory per block, KB", min: 0, max: 227, step: 1, value: st.S, onInput: (v) => { st.S = v; draw(); } })), bars, out, expl);
    function draw() {
      const o = G.occupancy(st.T, st.R, st.S * 1024);
      const rows = [
        ["warp slots", o.limWarps, `⌊64 / ${o.warpsPerBlock} warps⌋`],
        ["block slots", o.limBlocks, "32 per SM"],
        ["registers", o.limRegs, `⌊${o.warpsPerSubByRegs} warps × 4 / ${o.warpsPerBlock}⌋`],
        ["shared memory", o.limSmem, `⌊233,472 / ${int(o.smemPerBlock)} B⌋`],
      ];
      clear(bars);
      rows.forEach(([nm, v, how]) => {
        const lim = !o.fail && v === o.blocks;
        bars.appendChild(h("div", { class: "gpu-occ-row" + (lim ? " lim" : "") },
          h("span", { class: "gpu-occ-l" }, nm),
          h("span", { class: "gpu-occ-t" }, h("span", { class: "gpu-occ-f", style: { width: `${(Math.min(v, 32) / 32) * 100}%` } })),
          h("span", { class: "gpu-occ-v" }, `${int(v)} blocks`),
          h("span", { class: "gpu-occ-how" }, how)));
      });
      clear(out).append(tile(o.fail ? "0" : String(o.blocks), "resident blocks per SM"), tile(o.fail ? "0" : `${o.warps} of 64`, "resident warps"), tile(o.fail ? "–" : pct(o.occ, 1), "occupancy"), tile(o.limiter, "limited by"));
      expl.innerHTML = o.fail ? `The launch fails: ${o.fail}.` :
        `Registers: ${st.R} × 32 = ${int(st.R * 32)} per warp, rounded up to ${int(o.regsPerWarp)}; each processing block has 16,384, enough for ${o.warpsPerSubByRegs} warps, so ${4 * o.warpsPerSubByRegs} per SM. Shared memory: ${int(st.S * 1024)} B + 1,024 reserved, rounded to ${int(o.smemPerBlock)} B per block.`;
    }
    draw();
  });

  // ------------------------------------------------------------ fig: warp shuffle tree sum
  Deck.fig("gpu-shfl", (el, sp) => {
    const rand = window.TinyLM.mulberry32(sp.seed || 32);
    const vals = Array.from({ length: 32 }, () => Math.floor(rand() * 10));
    const steps = G.shfl(vals), total = vals.reduce((a, b) => a + b, 0);
    let k = sp.start ?? 1;
    const host = h("div", { class: "scroll-x" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const prev = button("◂ Back", () => { k = Math.max(0, k - 1); draw(); });
    const next = button("Next shuffle ▸", () => { k = Math.min(5, k + 1); draw(); }, { kind: "primary" });
    el.append(host, expl, h("div", { class: "f-ctl" }, button("Reset", () => { k = 0; draw(); }), prev, next, button("All 5", () => { k = 5; draw(); })));
    function draw() {
      const cw = 19, lab = 66, Wd = lab + 32 * cw + 4, rh = 22, gapY = 26;
      const Hd = 16 + (k + 1) * (rh + gapY) - gapY + 6;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "gpu-svg gpu-sh", role: "img", "aria-label": `warp sum after ${k} shuffles` });
      const uid = window.UI.nextId("gpu-ah");
      svg.appendChild(s("defs", null, s("marker", { id: uid, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, s("path", { d: "M0,0 L10,5 L0,10 z", class: "gpu-ah" }))));
      for (let l = 0; l < 32; l += 1) svg.appendChild(s("text", { x: lab + l * cw + cw / 2, y: 10, class: "gpu-axis", "text-anchor": "middle" }, String(l)));
      for (let j = 0; j <= k; j++) {
        const y = 16 + j * (rh + gapY), d = j === 0 ? 0 : 32 >> j, live = 32 >> j;
        svg.appendChild(s("text", { x: 0, y: y + rh - 7, class: "gpu-lab" }, j === 0 ? "start" : `d = ${d}`));
        steps[j].forEach((v, l) => {
          const used = l < live;
          const src = Array.from({ length: 1 << j }, (_, m) => l + m * live);
          const srcTxt = src.length <= 4 ? src.join(", ") : `${src[0]}, ${src[1]}, …, ${src[src.length - 1]}`;
          const g = s("g", { class: "gpu-sh-c" + (used ? "" : " dead") + (j === k && used ? " cur" : ""), "data-tip": used ? `lane ${l}: ${v}` + (j ? ` = start values of lanes ${srcTxt}` : "") : `lane ${l}: ${v}, no longer needed by the sum` });
          g.appendChild(s("rect", { x: lab + l * cw + 1, y, width: cw - 2, height: rh }));
          g.appendChild(s("text", { x: lab + l * cw + cw / 2, y: y + rh - 7, "text-anchor": "middle" }, String(v)));
          svg.appendChild(g);
        });
        if (j > 0) for (const l of [...new Set([0, 1, live - 1])].filter((q) => q < live)) {
          const xs = lab + (l + d) * cw + cw / 2, xe = lab + l * cw + cw / 2;
          svg.appendChild(s("path", { d: `M${xs},${y - gapY + 2} L${xe},${y - 3}`, class: "gpu-sh-a" + (j === k ? " cur" : ""), "marker-end": `url(#${uid})` }));
        }
      }
      clear(host).appendChild(svg);
      prev.disabled = k === 0; next.disabled = k === 5;
      expl.innerHTML = k === 0 ? `Each lane starts with one value; their sum is ${total}. Press <b>Next shuffle</b>.` :
        k < 5 ? `Shuffle ${k}, d = ${32 >> k}: every lane l adds the value of lane l + ${32 >> k} (arrows for three of them). Lanes 0 to ${(32 >> k) - 1} now hold sums of ${1 << k} values each; grey lanes hold values the sum no longer needs.` :
          `After 5 shuffles lane 0 holds <b>${steps[5][0]}</b>, the sum of all 32 values (${total}).`;
    }
    draw();
  });

  // ------------------------------------------------------------ fig: coalescing, sectors per warp request
  Deck.fig("gpu-coalesce", (el, sp) => {
    const st = { size: sp.size || 4, stride: sp.stride || 1, offset: sp.offset || 0 };
    const host = h("div", { class: "scroll-x" });
    const out = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.append(ctlRow(
      segmented({ label: "element", options: [{ value: 4, label: "float, 4 B" }, { value: 8, label: "double, 8 B" }, { value: 16, label: "float4, 16 B" }], value: st.size, onChange: (v) => { st.size = v; draw(); } }),
      slider({ label: "stride between lanes, elements", min: 1, max: 32, step: 1, value: st.stride, onInput: (v) => { st.stride = v; draw(); } }),
      slider({ label: "offset of lane 0, elements", min: 0, max: 31, step: 1, value: st.offset, onInput: (v) => { st.offset = v; draw(); } })), host, expl, out);
    function draw() {
      const { size, stride, offset } = st;
      const r = G.sectors(size, stride, offset);
      const secW = 34, lineW = 4 * secW, gapX = 14, perRow = 4, rowH = 44, Wd = perRow * lineW + (perRow - 1) * gapX + 4;
      const rows = Math.ceil(r.lines.length / perRow), Hd = rows * rowH + 4;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "gpu-svg gpu-co", role: "img", "aria-label": `${r.sectors.length} sectors in ${r.lines.length} cache lines` });
      const lanesAt = (a0, a1) => r.addrs.map((a, k) => [a, k]).filter(([a]) => a < a1 && a + size > a0).map(([, k]) => k);
      r.lines.forEach((ln, i) => {
        const x0 = 2 + (i % perRow) * (lineW + gapX), y0 = 2 + Math.floor(i / perRow) * rowH;
        const gap = i > 0 && ln !== r.lines[i - 1] + 1;
        svg.appendChild(s("text", { x: x0, y: y0 + 10, class: "gpu-axis" }, `${gap ? "… " : ""}byte ${int(ln * 128)}`));
        for (let q = 0; q < 4; q++) {
          const sec = ln * 4 + q, a0 = sec * 32, used = r.sectors.includes(sec), lanes = lanesAt(a0, a0 + 32);
          const usedB = lanes.reduce((acc, k) => acc + Math.min(r.addrs[k] + size, a0 + 32) - Math.max(r.addrs[k], a0), 0);
          const tipTxt = used ? `sector at bytes ${int(a0)}–${int(a0 + 31)}: lane${lanes.length > 1 ? "s" : ""} ${lanes.length > 4 ? `${lanes[0]}–${lanes[lanes.length - 1]}` : lanes.join(", ")}, ${usedB} of 32 bytes used` : `sector at bytes ${int(a0)}–${int(a0 + 31)}: not moved`;
          const g = s("g", { class: "gpu-co-sec" + (used ? " on" : ""), "data-tip": tipTxt });
          g.appendChild(s("rect", { x: x0 + q * secW, y: y0 + 15, width: secW, height: 22 }));
          lanes.forEach((k) => {
            const a = Math.max(r.addrs[k], a0), b = Math.min(r.addrs[k] + size, a0 + 32);
            g.appendChild(s("rect", { x: x0 + q * secW + ((a - a0) / 32) * secW + 0.4, y: y0 + 18, width: Math.max(0.6, ((b - a) / 32) * secW - 0.8), height: 16, class: "gpu-co-b" }));
          });
          svg.appendChild(g);
        }
      });
      clear(host).appendChild(svg);
      expl.innerHTML = `Lane k reads ${size} bytes at byte (${offset} + k × ${stride}) × ${size}. The warp asks for ${r.req} bytes; it touches <b>${r.sectors.length}</b> sectors of 32 bytes = ${int(r.moved)} bytes in ${r.lines.length} cache line${r.lines.length > 1 ? "s" : ""} of 128 bytes, each fetched from L2 (from HBM on a miss) unless it is already in L1. Shown: only the lines the warp touches.`;
      clear(out).append(tile(String(r.sectors.length), "32-byte sectors"), tile(`${int(r.moved)} B`, "bytes in those sectors"), tile(pct(r.eff, 1), "of those bytes used"), tile(String(r.lines.length), "128-byte lines"));
    }
    draw();
  });

  // ------------------------------------------------------------ fig: shared-memory bank conflicts
  Deck.fig("gpu-banks", (el, sp) => {
    const st = { pattern: sp.pattern || "col", pad: sp.pad || 0, k: sp.k || 2 };
    const host = h("div", { class: "scroll-x" });
    const out = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const padCtl = segmented({ label: "tile width (padding)", options: [{ value: 0, label: "[32][32]" }, { value: 1, label: "[32][33]" }, { value: 2, label: "[32][34]" }], value: st.pad, onChange: (v) => { st.pad = v; draw(); } });
    const kCtl = slider({ label: "stride k, words", min: 1, max: 33, step: 1, value: st.k, onInput: (v) => { st.k = v; draw(); } });
    el.append(h("div", { class: "f-ctl" }, segmented({ label: "thread l reads", options: [{ value: "row", label: "row: tile[0][l]" }, { value: "col", label: "column: tile[l][0]" }, { value: "stride", label: "s[l × k]" }, { value: "bcast", label: "s[0]" }], value: st.pattern, onChange: (v) => { st.pattern = v; draw(); } })), ctlRow(padCtl, kCtl), host, expl, out);
    function draw() {
      padCtl.hidden = !(st.pattern === "row" || st.pattern === "col"); kCtl.hidden = st.pattern !== "stride";
      const words = G.bankWords(st.pattern, st.pad, st.k), b = G.banks(words);
      const cw = 18, lab = 50, Wd = lab + 32 * cw + 4, ch = Math.max(7, Math.min(16, Math.floor(300 / b.degree)));
      const top = 50, Hd = top + b.degree * ch + 22;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "gpu-svg gpu-bk", role: "img", "aria-label": `${b.degree}-way bank conflict` });
      svg.appendChild(s("text", { x: lab - 6, y: 12, class: "gpu-axis", "text-anchor": "end" }, "lane"));
      svg.appendChild(s("text", { x: lab - 6, y: 30, class: "gpu-axis", "text-anchor": "end" }, "bank"));
      for (let l = 0; l < 32; l++) {
        svg.appendChild(s("text", { x: lab + l * cw + cw / 2, y: 12, class: "gpu-axis", "text-anchor": "middle" }, String(l)));
        const g = s("g", { class: "gpu-bk-lb", "data-tip": `lane ${l} reads word ${words[l]}: bank ${words[l] % 32}` });
        g.appendChild(s("rect", { x: lab + l * cw + 1, y: 18, width: cw - 2, height: 16 }));
        g.appendChild(s("text", { x: lab + l * cw + cw / 2, y: 30, "text-anchor": "middle" }, String(words[l] % 32)));
        svg.appendChild(g);
      }
      svg.appendChild(s("text", { x: lab - 6, y: top + 10, class: "gpu-axis", "text-anchor": "end" }, "words"));
      b.per.forEach((m, bank) => {
        [...m.entries()].forEach(([w, lanes], j) => {
          const g = s("g", { class: "gpu-bk-w" + (m.size > 1 ? " hot" : ""), "data-tip": `bank ${bank}, word ${w}: read by lane${lanes.length > 1 ? "s" : ""} ${lanes.length > 3 ? `${lanes[0]}–${lanes[lanes.length - 1]} (${lanes.length})` : lanes.join(", ")}` });
          g.appendChild(s("rect", { x: lab + bank * cw + 1, y: top + j * ch, width: cw - 2, height: ch - 1 }));
          svg.appendChild(g);
        });
      });
      for (let bank = 0; bank < 32; bank += 4) svg.appendChild(s("text", { x: lab + bank * cw + cw / 2, y: Hd - 6, class: "gpu-axis", "text-anchor": "middle" }, String(bank)));
      svg.appendChild(s("text", { x: lab + 32 * cw, y: Hd - 6, class: "gpu-axis", "text-anchor": "end" }, "bank →"));
      clear(host).appendChild(svg);
      const what = st.pattern === "row" ? `tile[0][l] is word l` : st.pattern === "col" ? `tile[l][0] is word ${32 + st.pad}l` : st.pattern === "stride" ? `s[l × ${st.k}] is word ${st.k}l` : "every lane reads word 0";
      expl.innerHTML = `${what}, in bank (word mod 32). ` + (b.degree === 1 ? (b.banksUsed === 1 ? "All 32 lanes read the same word: one broadcast, <b>no conflict</b>." : `${b.banksUsed} different banks: <b>no conflict</b>, one pass.`) : `The busiest bank holds <b>${b.degree}</b> different words the warp needs: a ${b.degree}-way conflict, ${b.degree} passes.`);
      clear(out).append(tile(b.degree === 1 ? "none" : `${b.degree}-way`, "bank conflict"), tile(String(b.passes), "passes through the banks"), tile(b.degree === 1 ? "100%" : `1/${b.degree} = ${pct(1 / b.degree, 1)}`, "of shared-memory bandwidth"));
    }
    draw();
  });

  // ------------------------------------------------------------ fig: tiled matrix multiply, phase by phase
  Deck.fig("gpu-tiles", (el, sp) => {
    const st = { N: sp.N || 8, T: sp.T || 2, done: 1 };
    const host = h("div", { class: "scroll-x" });
    const out = h("div", { class: "f-tiles" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    const nextP = button("Next phase ▸", () => { st.done = Math.min(G.tiledPhases(st.N, st.T), st.done + 1); draw(); }, { kind: "primary" });
    const nextT = button("Finish this tile", () => { const n = st.N / st.T; st.done = Math.min(G.tiledPhases(st.N, st.T), Math.ceil(st.done / n) * n + (st.done % n === 0 ? n : 0)); draw(); });
    el.append(ctlRow(
      segmented({ label: "matrix size N", options: optList([8, 12, 16]), value: st.N, onChange: (v) => { st.N = v; st.done = 1; draw(); } }),
      segmented({ label: "tile width T", options: [{ value: 1, label: "1 (no tiling)" }, { value: 2, label: "2" }, { value: 4, label: "4" }], value: st.T, onChange: (v) => { st.T = v; st.done = 1; draw(); } })),
      host, expl, out, h("div", { class: "f-ctl" }, button("Reset", () => { st.done = 1; draw(); }), nextP, nextT, button("Run to the end", () => { st.done = G.tiledPhases(st.N, st.T); draw(); })));
    function draw() {
      const { N, T, done } = st, n = N / T, P = G.tiledPhases(N, T);
      const cur = G.tiledPhase(N, T, done - 1), tilesDone = Math.floor((done - 1) / n) + ((done % n === 0) ? 1 : 0);
      const cs = Math.floor(176 / N), mw = N * cs, gap = 34, top = 18, Wd = 3 * mw + 2 * gap + 8, Hd = top + mw + 6;
      const svg = s("svg", { viewBox: `0 0 ${Wd} ${Hd}`, class: "gpu-svg gpu-tl", role: "img", "aria-label": `tiled multiply, phase ${done} of ${P}` });
      const mat = (ox, name, cellCls) => {
        svg.appendChild(s("text", { x: ox + mw / 2, y: 12, class: "gpu-lab", "text-anchor": "middle" }, name));
        for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) svg.appendChild(s("rect", { x: ox + c * cs, y: top + r * cs, width: cs, height: cs, class: "gpu-tl-c " + cellCls(r, c) }));
        svg.appendChild(s("rect", { x: ox, y: top, width: mw, height: mw, class: "gpu-tl-frame" }));
      };
      const inTile = (r, c, br, bc) => Math.floor(r / T) === br && Math.floor(c / T) === bc;
      const xA = 4, xB = 4 + mw + gap, xC = 4 + 2 * (mw + gap);
      mat(xA, "A", (r, c) => (inTile(r, c, cur.bi, cur.t) ? "ld" : Math.floor(r / T) === cur.bi ? "band" : ""));
      mat(xB, "B", (r, c) => (inTile(r, c, cur.t, cur.bj) ? "ld" : Math.floor(c / T) === cur.bj ? "band" : ""));
      mat(xC, "C", (r, c) => {
        const ti = Math.floor(r / T) * n + Math.floor(c / T), curIdx = cur.bi * n + cur.bj;
        return ti < curIdx || (ti === curIdx && done % n === 0) ? "fin" : ti === curIdx ? "acc" : "";
      });
      svg.appendChild(s("text", { x: xA + mw + gap / 2, y: top + mw / 2 + 5, class: "gpu-tl-op", "text-anchor": "middle" }, "×"));
      svg.appendChild(s("text", { x: xB + mw + gap / 2, y: top + mw / 2 + 5, class: "gpu-tl-op", "text-anchor": "middle" }, "→"));
      svg.appendChild(s("rect", { x: xC + cur.bj * T * cs, y: top + cur.bi * T * cs, width: T * cs, height: T * cs, class: "gpu-tl-cur" }));
      clear(host).appendChild(svg);
      const rr = (b) => (T === 1 ? `${b}` : `${b * T}–${b * T + T - 1}`);
      expl.innerHTML = `Phase ${cur.t + 1} of ${n} for the C tile at rows ${rr(cur.bi)}, columns ${rr(cur.bj)}: load A rows ${rr(cur.bi)}, columns ${rr(cur.t)} and B rows ${rr(cur.t)}, columns ${rr(cur.bj)}, ${2 * T * T} element${2 * T * T > 1 ? "s" : ""}, into shared memory; then each of the ${T * T} thread${T > 1 ? "s" : ""} adds ${T} product${T > 1 ? "s" : ""}.`;
      clear(out).append(tile(`${int(done)} of ${int(P)}`, "phases done"), tile(int(G.tiledLoads(N, T, done)), "global loads so far"), tile(`2N³/T = ${int((2 * N ** 3) / T)}`, "loads for the whole product"), tile(`2N³ = ${int(2 * N ** 3)}`, "loads without tiling"), tile(fx(T / 4, 2), "FLOPs per byte, FP32"));
      nextP.disabled = done >= P; nextT.disabled = done >= P;
    }
    draw();
  });

  // ------------------------------------------------------------ fig: online softmax, element by element
  Deck.fig("gpu-online", (el, sp) => {
    const base = sp.x || [2, 1, 5, 3, 8, 6, 9, 4];
    const st = { k: 3, shift: 0 };
    const host = h("div", { class: "scroll-x" });
    const expl = h("p", { class: "f-mm-expl", "aria-live": "polite" });
    el.append(h("div", { class: "f-ctl" }, segmented({ label: "inputs", options: [{ value: 0, label: "x" }, { value: 100, label: "x + 100" }], value: 0, onChange: (v) => { st.shift = v; draw(); } })),
      host, expl, h("div", { class: "f-ctl" }, button("Reset", () => { st.k = 1; draw(); }), button("Next element ▸", () => { st.k = Math.min(base.length + 1, st.k + 1); draw(); }, { kind: "primary" }), button("All", () => { st.k = base.length + 1; draw(); })));
    const g4 = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3).replace("e+", "e") : fx(v, 4)) : v > 0 ? "∞" : Number.isNaN(v) ? "NaN" : "−∞");
    function draw() {
      const x = base.map((v) => v + st.shift), r = G.online(x), shown = Math.min(st.k, x.length);
      const t = h("table", { class: "f-table dense gpu-on" });
      t.appendChild(h("thead", null, h("tr", null, ["j", "x<sub>j</sub>", "running max m<sub>j</sub>", "rescale e<sup>m<sub>j−1</sub> − m<sub>j</sub></sup>", "running sum d<sub>j</sub>"].map((c, i) => h("th", { class: i ? "r" : "", html: c })))));
      const tb = h("tbody");
      r.steps.slice(0, shown).forEach((s0, i) => tb.appendChild(h("tr", { class: i === shown - 1 && st.k <= x.length ? "hi" : "" },
        h("td", null, String(s0.j + 1)), h("td", { class: "r" }, String(s0.x)), h("td", { class: "r" }, String(s0.m)),
        h("td", { class: "r" }, s0.first ? "–" : s0.scale === 1 ? "1 (max unchanged)" : g4(s0.scale)), h("td", { class: "r" }, g4(s0.d)))));
      t.appendChild(tb);
      clear(host).appendChild(t);
      if (st.k <= x.length) {
        const s0 = r.steps[shown - 1];
        expl.innerHTML = shown === 1 ? `Element 1: m = ${s0.x}, d = e^(${s0.x} − ${s0.x}) = 1.` :
          (s0.scale === 1 ? `Element ${shown}: x = ${s0.x} does not raise the maximum, so d grows by e^(${s0.x} − ${s0.m}) = ${g4(Math.exp(s0.x - s0.m))}.` :
            `Element ${shown}: x = ${s0.x} raises the maximum to ${s0.m}, so the old sum is multiplied by e^(${r.steps[shown - 2].m} − ${s0.m}) = ${g4(s0.scale)} before adding e^0 = 1.`);
      } else {
        const maxDiff = Math.max(...r.y.map((v, i) => Math.abs(v - r.y2[i])));
        expl.innerHTML = `Done in one pass: m = ${r.m}, d = ${g4(r.d)}. Softmax = e^(x − m) / d, e.g. ${g4(r.y[6])} for x = ${x[6]}; the two-pass safe softmax ${maxDiff === 0 ? "agrees exactly" : `agrees to ${maxDiff.toExponential(1)}`}. ` +
          (st.shift ? `The direct formula in FP32 computes e^${x[6]} = ${g4(Math.fround(Math.exp(x[6])))}, and ∞ / ∞ = <b>${g4(r.y32[6])}</b>.` : `The direct formula in FP32 also works here (${g4(r.y32[6])}), because e^9 is small.`);
      }
    }
    draw();
  });

  // ------------------------------------------------------------ fig: the time of one Llama 3 8B layer on an H100
  Deck.fig("gpu-budget", (el, sp) => {
    const st = { T: sp.T || 2048, L: sp.L || 0 };
    const host = h("div", { class: "scroll-x" });
    const out = h("div", { class: "f-tiles" });
    const key = h("div", { class: "legend" }, [["s1", "weight matrix products"], ["s2", "attention"], ["s3", "elementwise operations"]].map(([c, l]) => h("span", { class: "key" }, h("span", { class: `sw sw-${c}` }), l)));
    el.append(h("div", { class: "f-ctl" },
      segmented({ label: "new tokens processed together, T", options: optList([1, 128, 512, 2048, 8192], int), value: st.T, onChange: (v) => { st.T = v; draw(); } }),
      segmented({ label: "earlier tokens in the KV cache, L", options: optList([0, 8192, 32768, 131072], int), value: st.L, onChange: (v) => { st.L = v; draw(); } })), host, key, out);
    const fmtT = (sec) => (sec < 1e-6 ? `${fx(sec * 1e9, 0)} ns` : `${sep(sec * 1e6, sec < 1e-5 ? 2 : 1)} µs`);
    const sep = (v, d) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
    function draw() {
      const b = G.budget(st.T, st.L), mx = Math.max(...b.ops.map((o) => o.t));
      const t = h("table", { class: "f-table dense gpu-bud" });
      t.appendChild(h("thead", null, h("tr", null, ["Operation", "GFLOP", "MB moved", "Bound", "Time", ""].map((c, i) => h("th", { class: i > 0 && i < 5 ? "r" : "" }, c)))));
      const tb = h("tbody");
      b.ops.forEach((o) => tb.appendChild(h("tr", null,
        h("td", null, o.name), h("td", { class: "r" }, sep(o.F / 1e9, o.F < 1e8 ? 4 : 1)), h("td", { class: "r" }, sep(o.B / 1e6, o.B < 1e7 ? 2 : 1)),
        h("td", { class: "r" }, o.bound), h("td", { class: "r" }, fmtT(o.t)),
        h("td", { class: "gpu-bud-bar" }, h("span", { class: `gpu-bud-f ${o.kind}`, style: { width: `${(o.t / mx) * 100}%` }, "data-tip": `${pct(o.t / b.total, 1)} of the layer` })))));
      tb.appendChild(h("tr", { class: "hi" }, h("td", null, "layer total"), h("td", { class: "r" }, sep(b.ops.reduce((a, o) => a + o.F, 0) / 1e9, 1)), h("td", { class: "r" }, sep(b.ops.reduce((a, o) => a + o.B, 0) / 1e6, 1)), h("td", null, ""), h("td", { class: "r" }, fmtT(b.total)), h("td", null, "")));
      t.appendChild(tb);
      clear(host).appendChild(t);
      clear(out).append(tile(pct(b.mm, 1), "weight matrix products"), tile(pct(b.attn, 1), "attention"), tile(pct(b.elem, 1), "elementwise operations"), tile(`${fx(b.total * 32 * 1e3, 2)} ms`, "32 layers"));
    }
    draw();
  });

  // ================================================================== Part 6
  Deck.part({ n: 6, title: "GPUs and CUDA", short: "GPUs" });

  Deck.chapter({ title: "Why GPUs" });

  Deck.slide({
    id: "gpu-part",
    layout: "part",
    title: "GPUs and CUDA",
    body: `<p>Parts 1 to 5 built the model as arithmetic. This part is about the machine that does the arithmetic: an NVIDIA H100, which multiplies BF16 matrices at 989 trillion operations per second but reads its memory at 3.35 trillion bytes per second.</p>
<p>You will take the chip apart down to the register file, write CUDA kernels and count their memory traffic in 32-byte sectors, and work out why a matrix multiply, a softmax and FlashAttention each run as fast as they do. Hardware numbers come from NVIDIA's datasheets, whitepapers and CUDA documentation, and from published measurements.</p>`,
    take: "Predict whether a kernel is limited by arithmetic or by memory, read and write CUDA kernels, and explain what tiling, tensor cores, fusion and FlashAttention change on an H100.",
  });

  Deck.slide({
    id: "gpu-flops",
    title: "Matrix multiplies are almost all of a transformer's arithmetic",
    body: `<p>Count the <b>floating-point operations</b> (FLOPs) one Llama 3 8B layer spends on one token. A multiply-add counts as 2 FLOPs, so a vector of length <i>m</i> times an <i>m</i> × <i>n</i> matrix costs 2<i>mn</i>.</p>
<p>The layer has d = 4,096, 32 query heads and 8 key/value heads of 128 dimensions, and a SwiGLU MLP with three 4,096 × 14,336 matrices. Its seven weight matrices cost 2 × 4,096 × (4,096 + 1,024 + 1,024 + 4,096 + 3 × 14,336) = <span class="num">436,207,616</span> FLOPs per token. Attention adds two products, the scores and the mixing, of 2 × 4,096 FLOPs per token attended to each: 67.1 million at a context of 4,096.</p>
<p>Everything else (RMSNorm, rotary embeddings, the SwiGLU gate, softmax, residual adds) is about 0.9 million FLOPs, counting an exp or a division as one. Matrix products are 99.82% of the arithmetic.</p>
<p>The other 0.18% is not free. Those operations read and write whole tensors for a few FLOPs each, and chapter 6.5 shows that memory bandwidth sets their speed.</p>`,
    fig: {
      type: "calc", caption: "FLOPs per token in one Llama 3 8B layer",
      inputs: [{ id: "T", label: "tokens attended to (context)", options: optList([512, 2048, 4096, 8192, 32768, 131072], int), value: 4096 }],
      outputs: [
        { label: "weight matrices, MFLOP", f: (v) => G.llamaLayer(v.T).weights / 1e6, fmt: "f1" },
        { label: "attention products, MFLOP", f: (v) => G.llamaLayer(v.T).attn / 1e6, fmt: "f1" },
        { label: "everything else, MFLOP (approx.)", f: (v) => G.llamaLayer(v.T).other / 1e6, fmt: "f2" },
        { label: "share in matrix products", f: (v) => G.llamaLayer(v.T).share, fmt: (x) => pct(x, 2) },
      ],
      note: (v) => `Everything else = RMSNorm 2 × 4 × 4,096 + RoPE 3 × 5,120 + SwiGLU gate 5 × 14,336 + residuals 2 × 4,096 + softmax 6 × 32 heads × ${int(v.T)} scores = ${int(G.llamaLayer(v.T).other)} FLOPs.`,
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Table 3: 8B model dimensions)"],
  });

  Deck.slide({
    id: "gpu-cpu",
    title: "A CPU core runs one thread fast; an SM keeps 2,048 threads in flight",
    body: `<p>A CPU core is built to finish one instruction stream quickly. Each of the 72 Arm Neoverse V2 cores in NVIDIA's Grace CPU executes out of order behind 64 KB of L1 data cache and 1 MB of L2, and has four 128-bit SVE2 vector pipelines: 32 FP32 FLOPs per clock with fused multiply-adds.</p>
<p>A <b>streaming multiprocessor</b> (SM), the GPU's unit of execution, has 128 FP32 lanes (256 FLOPs per clock) and 4 tensor cores, and holds up to 2,048 threads at once. It does not make any one thread fast: instructions issue in order, with no branch prediction or speculation. When a thread waits for memory, the SM issues another thread's instruction instead.</p>
<p>Over the whole chip, the H100 SXM has 9.4 times Grace's FP32 peak, 139 times that peak in BF16 on its tensor cores (units that multiply small matrices in one instruction, chapter 6.2), and 6.5 times its memory bandwidth. The price is latency: one GPU thread is far slower than one CPU thread, so a GPU only wins on work that splits into many thousands of independent pieces.</p>`,
    fig: {
      type: "table", caption: "One Grace CPU and one H100 SXM, side by side",
      cols: ["", "Grace CPU", "H100 SXM"], rowHeads: true,
      rows: [
        ["Execution units", "72 cores", "132 SMs"],
        ["FP32 FLOPs per clock, per core or SM", "32", "256 (4,096 BF16 on the tensor cores)"],
        ["Clock", "3.1 GHz (3.0 GHz all-core SIMD)", "1.98 GHz (1.83 GHz for tensor cores)"],
        ["Peak FP32", "7.1 TFLOP/s", "66.9 TFLOP/s"],
        ["Peak BF16, tensor cores, dense", "–", "989.4 TFLOP/s"],
        ["Fast storage per core or SM", "64 KB L1D + 1 MB L2", "256 KB L1/shared + 256 KB registers"],
        ["Last-level cache", "114 MB L3", "50 MB L2"],
        ["Memory", "up to 480 GB LPDDR5X, up to 512 GB/s", "80 GB HBM3, 3.35 TB/s"],
      ],
      note: "Grace peak: 72 cores × 32 × 3.1 GHz = 7.1 TFLOP/s, the same arithmetic that gives NVIDIA's 7.1 TFLOP/s FP64 for the 144-core Grace Superchip. Grace bandwidth is 512 GB/s with 120 or 240 GB of memory and 384 GB/s with 480 GB.",
    },
    terms: [{ term: "Streaming multiprocessor (SM)", def: "The GPU's unit of execution: schedulers, register file, FP32/INT32/FP64 lanes, tensor cores and 256 KB of L1/shared memory. An H100 SXM has 132." }],
    src: ["NVIDIA Grace CPU Superchip datasheet and whitepaper", "Arm Neoverse V2 Core Telemetry Specification (in-order frontend, out-of-order backend)", "NVIDIA H100 Tensor Core GPU Architecture whitepaper V1.04 (Table 3)", "CUDA C++ Programming Guide, Advanced Kernel Programming (SMs issue in order, no branch prediction or speculation)"],
  });

  Deck.slide({
    id: "gpu-h100",
    title: "The H100 SXM in eight numbers",
    body: `<p>Every calculation in this part uses one GPU: the NVIDIA H100 in its <b>SXM</b> form, the module that mounts on an HGX baseboard of 4 or 8 GPUs joined by NVLink, with a 700 W power budget. The PCIe card of the same name has 114 SMs, 2.0 TB/s of HBM2e and 350 W, so its numbers differ.</p>
<p>NVIDIA's datasheet lists 1,979 TFLOP/s for BF16 with the footnote "with sparsity". That figure assumes <b>2:4 structured sparsity</b>: two of every four weights are zero and the hardware skips them. Dense matrices, the normal case in training and most inference, get half, 989.4 TFLOP/s. This set always quotes dense figures.</p>
<p>The ratio of the two headline numbers, 989.4 TFLOP/s to 3.35 TB/s, is 295 FLOPs per byte. Most of this part is about that ratio.</p>`,
    fig: {
      type: "specs", caption: "NVIDIA H100 SXM",
      items: [
        { v: "132", label: "SMs", note: "of 144 on the full GH100 die" },
        { v: "989 TFLOP/s", label: "BF16 tensor, dense", note: "1,979 with 2:4 sparsity" },
        { v: "67 TFLOP/s", label: "FP32, no tensor cores", note: "66.9 in the whitepaper" },
        { v: "80 GB", label: "HBM3", note: "5 stacks, 5,120-bit bus" },
        { v: "3.35 TB/s", label: "HBM bandwidth", note: "3,352 GB/s in the whitepaper" },
        { v: "50 MB", label: "L2 cache" },
        { v: "900 GB/s", label: "NVLink 4", note: "18 links, both directions" },
        { v: "700 W", label: "TDP", note: "configurable" },
      ],
    },
    terms: [
      { term: "SXM", def: "NVIDIA's mezzanine module form of a data-center GPU, mounted on a baseboard with NVLink. Higher power and different specifications from the PCIe card of the same name." },
      { term: "2:4 structured sparsity", def: "A weight layout with two zeros in every group of four, which tensor cores can skip for twice the throughput. Datasheet figures marked 'with sparsity' assume it." },
    ],
    src: ["NVIDIA H100 product page and datasheet", "NVIDIA H100 Tensor Core GPU Architecture whitepaper V1.04, 2023 (Table 3)", "NVIDIA HGX H100 and HGX H200 datasheet (4-GPU and 8-GPU forms)"],
  });

  Deck.slide({
    id: "gpu-intensity",
    title: "Arithmetic intensity: FLOPs per byte moved",
    body: `<p><b>Arithmetic intensity</b> is the number of FLOPs a kernel does per byte it moves to or from memory. FP32 vector addition, <code>c[i] = a[i] + b[i]</code>, does 1 FLOP per element and moves 12 bytes (two 4-byte loads, one 4-byte store): 1/12 ≈ 0.083 FLOP/byte.</p>
<p>At 3.35 TB/s, adding two vectors of 2<sup>26</sup> elements moves 805 MB and takes 240 µs. That is 279 GFLOP/s, 0.42% of the 66.9 TFLOP/s FP32 peak: the arithmetic units wait for memory more than 99% of the time.</p>
<p>To keep the FP32 lanes busy a kernel needs 66.9 / 3.35 ≈ 20 FLOPs per byte; to keep the tensor cores busy, 989.4 / 3.35 ≈ 295. Raise the FLOPs per element on the plate: the kernel stops being limited by memory at 240 FLOPs per element.</p>`,
    fig: {
      type: "calc", caption: "An elementwise FP32 kernel over 2²⁶ elements on an H100",
      inputs: [{ id: "k", label: "FLOPs per element (12 bytes moved)", min: 1, max: 480, step: 1, value: 1, fmt: "int" }],
      outputs: [
        { label: "FLOPs per byte", f: (v) => v.k / 12, fmt: "f3" },
        { label: "time", f: (v) => Math.max((12 * 2 ** 26) / HW.bw, (v.k * 2 ** 26) / HW.fp32), fmt: "time" },
        { label: "achieved TFLOP/s", f: (v) => (v.k * 2 ** 26) / Math.max((12 * 2 ** 26) / HW.bw, (v.k * 2 ** 26) / HW.fp32) / 1e12, fmt: "f3" },
        { label: "share of FP32 peak", f: (v) => (v.k * 2 ** 26) / Math.max((12 * 2 ** 26) / HW.bw, (v.k * 2 ** 26) / HW.fp32) / HW.fp32, fmt: (x) => pct(x, x < 0.1 ? 2 : 1) },
        { label: "limited by", f: (v) => ((12 * 2 ** 26) / HW.bw >= (v.k * 2 ** 26) / HW.fp32 ? "memory" : "arithmetic"), fmt: (x) => x },
      ],
      note: (v) => `Memory time 805,306,368 B / 3.35 TB/s = 240.4 µs; arithmetic time ${int(v.k)} × 2²⁶ / 66.9 TFLOP/s = ${window.UI.format((v.k * 2 ** 26) / HW.fp32, "time")}. The kernel takes the larger of the two.`,
    },
    check: {
      q: "An FP32 kernel computes <code>y[i] = 3 * x[i] * x[i] + 2</code>: two multiplies and an add per element, one 4-byte load and one 4-byte store. What is its arithmetic intensity, and what limits it on an H100?",
      options: ["3 / 12 = 0.25 FLOP/byte, memory", "3 / 8 = 0.375 FLOP/byte, memory", "3 / 4 = 0.75 FLOP/byte, arithmetic", "3 FLOP/byte, arithmetic"],
      answer: 1,
      why: "It moves 8 bytes per element (one load, one store), so 3/8 = 0.375 FLOP/byte, far below the 20 the FP32 lanes need. Over 2²⁶ elements it moves 536,870,912 bytes: 160 µs at 3.35 TB/s, while the arithmetic alone would take 3 µs.",
    },
    terms: [{ term: "Arithmetic intensity", def: "FLOPs performed per byte moved between the processor and memory. Compared with peak FLOP/s ÷ bandwidth, it says whether a kernel is limited by arithmetic or by memory." }],
    src: ["NVIDIA H100 Tensor Core GPU Architecture whitepaper V1.04 (Table 3)"],
  });

  // ================================================================== 6.2
  Deck.chapter({ title: "Inside an H100" });

  Deck.slide({
    id: "gpu-chip",
    title: "The chip: 132 SMs in 8 clusters around a 50 MB L2",
    layout: "wide",
    body: `<p>The full GH100 die has 8 <b>GPCs</b> (GPU processing clusters), each with 9 TPCs (texture processing clusters) of 2 SMs: 144 SMs. Around them sit a 60 MB L2 cache and 12 memory controllers of 512 bits each, wired to 6 HBM stacks. The H100 SXM enables part of the die: 66 TPCs (<span class="num">132</span> SMs), 50 MB of L2, and 10 controllers driving 5 HBM3 stacks over a 5,120-bit bus. The die has 80 billion transistors on 814 mm², made on TSMC's 4N process.</p>
<p>The L2 sits between the SMs and HBM and is partitioned: each partition caches data for the SMs in the GPCs connected to it. The GPC matters to programmers on Hopper because the blocks of a thread block cluster (chapter 6.3) are scheduled onto SMs of one GPC.</p>`,
    fig: {
      type: "diagram", caption: "GH100 as configured in the H100 SXM; the full die supports a 6th HBM stack (dashed box) that the SXM does not use", w: 24, h: 9.4, unit: 40,
      groups: [{ x: 6.15, y: 1.55, w: 11.7, h: 7.6, label: "8 GPCs (9 TPCs = 18 SMs each on the die)" }],
      nodes: [
        { id: "pcie", x: 0.2, y: 0.15, w: 11.3, h: 0.95, label: "PCIe Gen 5 x16: 64 GB/s each way to the host" },
        { id: "nvl", x: 12.5, y: 0.15, w: 11.3, h: 0.95, label: "NVLink 4: 18 links, 450 GB/s each way" },
        ...[0, 1, 2, 3].map((r) => ({ id: `g${2 * r}`, x: 6.45, y: 2.1 + r * 1.75, w: 5.3, h: 1.5, label: `GPC ${2 * r}` })),
        ...[0, 1, 2, 3].map((r) => ({ id: `g${2 * r + 1}`, x: 12.25, y: 2.1 + r * 1.75, w: 5.3, h: 1.5, label: `GPC ${2 * r + 1}` })),
        { id: "l2a", x: 3.0, y: 2.1, w: 2.7, h: 6.75, label: "L2\npartition", kind: "blue" },
        { id: "l2b", x: 18.3, y: 2.1, w: 2.7, h: 6.75, label: "L2\npartition", kind: "blue" },
        { id: "m0", x: 0.2, y: 2.1, w: 2.2, h: 1.6, label: "HBM3", sub: "16 GB", kind: "dark" },
        { id: "m1", x: 0.2, y: 4.67, w: 2.2, h: 1.6, label: "HBM3", sub: "16 GB", kind: "dark" },
        { id: "m2", x: 0.2, y: 7.25, w: 2.2, h: 1.6, label: "HBM3", sub: "16 GB", kind: "dark" },
        { id: "m3", x: 21.6, y: 2.1, w: 2.2, h: 1.6, label: "HBM3", sub: "16 GB", kind: "dark" },
        { id: "m4", x: 21.6, y: 4.67, w: 2.2, h: 1.6, label: "HBM3", sub: "16 GB", kind: "dark" },
        { id: "m5", x: 21.6, y: 7.25, w: 2.2, h: 1.6, label: "6th", sub: "stack", kind: "ghost" },
      ],
      edges: [
        ...[0, 1, 2].map((i) => ({ from: `m${i}`, to: "l2a", both: true })), ...[3, 4].map((i) => ({ from: `m${i}`, to: "l2b", both: true })),
      ],
      note: "The SMs reach the L2 partitions through a crossbar; which GPC sits next to which partition is drawn schematically. Each HBM stack connects through two 512-bit memory controllers. The die has 60 MB of L2 and 6 stacks; the H100 SXM enables 50 MB and 5.",
    },
    terms: [{ term: "GPC", def: "GPU processing cluster: a group of TPCs (pairs of SMs) on the die. GH100 has 8; the SMs of a thread block cluster are always in one GPC." }],
    note: "66 TPCs do not divide evenly among 8 GPCs, so the GPCs of an H100 SXM differ in their number of SMs.",
    src: ["NVIDIA H100 Tensor Core GPU Architecture whitepaper V1.04 (GH100 units, L2 cache, Table 3)"],
  });

  Deck.slide({
    id: "gpu-sm",
    title: "Inside an SM: four schedulers sharing 256 KB of on-chip memory",
    layout: "wide",
    body: `<p>An H100 SM is split into four <b>processing blocks</b> (SM sub-partitions). Each has a warp scheduler and dispatch unit that issue one instruction per clock for a warp (a group of 32 threads; chapter 6.4), a 64 KB register file (16,384 registers of 32 bits), 32 FP32 lanes, 16 INT32 lanes, 16 FP64 lanes, one tensor core, 8 load/store units and a special-function unit (SFU), which NVIDIA describes as executing transcendental instructions such as sine, cosine, reciprocal and square root.</p>
<p>The four blocks share a 256 KB SRAM split between L1 data cache and <b>shared memory</b>, a scratchpad that the program addresses explicitly; up to 228 KB can be shared memory. They also share the Tensor Memory Accelerator (TMA), which copies tiles between global and shared memory. Per SM that makes 65,536 registers (256 KB), 128 FP32 lanes and 4 tensor cores: the register file is as large as the L1 and shared memory together.</p>`,
    fig: {
      type: "diagram", caption: "One H100 SM: four processing blocks and the memory they share", w: 24, h: 9.2, unit: 40,
      groups: [0, 1, 2, 3].map((p) => ({ x: 0.2 + p * 5.95, y: 0.15, w: 5.65, h: 6.75, label: `Processing block ${p}` })),
      nodes: [0, 1, 2, 3].flatMap((p) => {
        const x = 0.45 + p * 5.95, w = 5.15;
        return [
          { id: `ws${p}`, x, y: 0.75, w, h: 0.9, label: "Warp scheduler + dispatch" },
          { id: `rf${p}`, x, y: 1.85, w, h: 1.1, label: "Register file 64 KB", sub: "16,384 × 32-bit", kind: "hi" },
          { id: `f32${p}`, x, y: 3.15, w, h: 0.8, label: "32 FP32 lanes" },
          { id: `int${p}`, x, y: 4.1, w, h: 0.8, label: "16 INT32 · 16 FP64" },
          { id: `tc${p}`, x, y: 5.05, w, h: 0.8, label: "Tensor core (4th gen)", kind: "blue" },
          { id: `ld${p}`, x, y: 6.0, w, h: 0.7, label: "8 LD/ST · SFU" },
        ];
      }).concat([
        { id: "smem", x: 0.2, y: 7.3, w: 17.1, h: 1.6, label: "256 KB L1 data cache / shared memory", sub: "up to 228 KB as shared memory", kind: "hi" },
        { id: "tma", x: 17.9, y: 7.3, w: 5.9, h: 1.6, label: "Tensor Memory", sub: "Accelerator (TMA)", kind: "blue" },
      ]),
      edges: [0, 1, 2, 3].map((p) => ({ from: `ld${p}`, to: "smem", both: true })).concat([{ from: "tma", to: "smem", both: true }]),
    },
    terms: [
      { term: "Processing block (SM sub-partition)", def: "One quarter of an SM: a scheduler that issues one instruction per clock for a group of 32 threads (a warp), a 64 KB register file, 32 FP32 lanes and one tensor core." },
      { term: "Shared memory", def: "On-chip SRAM inside each SM that the program reads and writes explicitly. Allocated per thread block; up to 228 KB per SM on H100." },
    ],
    src: ["NVIDIA H100 Tensor Core GPU Architecture whitepaper V1.04 (SM diagram, Combined L1 Data Cache and Shared Memory, Table 3)", "NVIDIA Fermi Compute Architecture whitepaper (Special Function Units)"],
  });

  Deck.slide({
    id: "gpu-tensor",
    title: "Tensor cores do 16 times the work per clock of the FP32 lanes",
    body: `<p>The FP32 lanes, which NVIDIA calls CUDA cores, each complete one <b>fused multiply-add</b> (FMA, a·b + c in one rounding) per clock: 128 FMAs, 256 FLOPs per SM per clock. A <b>tensor core</b> works on small matrices instead: each instruction computes D = A·B + C for a tile. NVIDIA states that an H100 SM's tensor cores have twice the dense rate of an A100 SM, which does 1,024 FP16 FMAs per clock. That gives 2,048 FMAs, 4,096 FLOPs per clock per SM in FP16 or BF16: 16 times the FP32 lanes.</p>
<p>A peak is SMs × FLOPs per clock × clock. With 132 SMs, the FP32 peak at the 1.98 GHz boost clock is 132 × 256 × 1.98 GHz = 66.9 TFLOP/s. NVIDIA rates the tensor cores at a lower clock, 1.83 GHz: 132 × 4,096 × 1.83 GHz = 989.4 TFLOP/s in BF16. TF32 runs at half that rate and FP8 at twice.</p>
<p>Only matrix products can use the tensor cores. Additions and normalizations run on the lanes; exponentials use the special-function units, at 16 results per clock per SM, an eighth of the FP32 FMA rate: about 3.9 TFLOP/s on an H100 SXM5.</p>`,
    fig: {
      type: "bars", caption: "H100 SXM peak throughput by unit and format, dense, TFLOP/s",
      items: [
        { label: "FP64, lanes", value: 33.5, text: "33.5" },
        { label: "FP32, lanes", value: 66.9, text: "66.9" },
        { label: "FP64, tensor", value: 66.9, text: "66.9" },
        { label: "BF16, lanes", value: 133.8, text: "133.8" },
        { label: "TF32, tensor", value: 494.7, text: "494.7" },
        { label: "BF16, tensor", value: 989.4, text: "989.4", hi: true },
        { label: "FP8, tensor", value: 1978.9, text: "1,978.9" },
      ],
      note: "Figures from NVIDIA's whitepaper (Table 3). Lane rates use 1.98 GHz, tensor rates for TF32, BF16 and FP8 use 1.83 GHz; FP64 tensor uses 1.98 GHz.",
    },
    terms: [
      { term: "Tensor core", def: "A unit that executes small matrix multiply-accumulates (D = A·B + C) as single instructions. Four per H100 SM; 4,096 dense BF16 FLOPs per clock per SM." },
      { term: "Fused multiply-add (FMA)", def: "a × b + c computed with one rounding at the end. Counts as 2 FLOPs." },
    ],
    src: ["NVIDIA H100 Tensor Core GPU Architecture whitepaper V1.04 (Table 3; H100 Tensor Core Architecture)", "NVIDIA A100 Tensor Core GPU Architecture whitepaper (1,024 dense FP16 FMA per clock per SM)", "CUDA C++ Best Practices Guide (throughput of native arithmetic instructions)", "Shah et al., FlashAttention-3, 2024 (3.9 TFLOP/s of special functions on H100 SXM5)"],
  });

  Deck.slide({
    id: "gpu-hierarchy",
    title: "The memory hierarchy: registers, shared memory, L2, HBM",
    body: String.raw`<p>Each level is larger and slower than the one above it. The capacities are the H100 SXM's. The latencies were measured by Luo et al. with pointer-chasing microbenchmarks on an H800 PCIe, a Hopper GPU with the same SM but HBM2e memory and a maximum clock of 1.755 GHz; read them as approximate.</p>
<p>On that GPU a load that went to HBM took 478.8 clock cycles, 16.5 times a shared-memory load (29.0 cycles): 273 ns or more, since the clock was at most 1.755 GHz. Chapter 6.4 shows how the SM stays busy during that time.</p>
<p>Bandwidth falls with distance too. Shared memory delivers 128 bytes per clock per SM (32 banks of 4 bytes), about 33 TB/s over 132 SMs at 1.98 GHz: ten times the 3.35 TB/s of HBM. Every fast kernel in this part moves data from HBM once into shared memory or registers and reuses it there.</p>`,
    fig: {
      type: "table", caption: "Capacity, bandwidth and measured latency of each level", align: ["l", "l", "l", "r"],
      cols: ["Level", "Capacity (H100 SXM)", "Bandwidth", "Latency, cycles"],
      rows: [
        ["Registers", "256 KB per SM (65,536 × 32-bit)", "operands for every lane, every clock", "–"],
        ["Shared memory", "up to 228 KB per SM", "128 B per clock per SM", "29"],
        ["L1 data cache", "the rest of the SM's 256 KB", "same SRAM", "41"],
        ["L2 cache", "50 MB per chip", "≈ 4,470 B per clock, chip (H800 PCIe)", "263"],
        ["HBM", "80 GB of HBM3", "3.35 TB/s", "479 (HBM2e)"],
      ],
      note: "Latency and L2 bandwidth: Luo et al. 2024, Tables III to V, measured on an H800 PCIe (HBM2e, maximum clock 1.755 GHz): L1 40.7, shared 29.0, L2 263.0, global 478.8 cycles. On-chip sizes use 1 KB = 1,024 bytes, as NVIDIA writes them.",
    },
    src: ["Luo et al., Benchmarking and Dissecting the Nvidia Hopper GPU Architecture, IPDPS 2024 (Tables IV, V)", "NVIDIA H100 whitepaper V1.04", "CUDA C++ Programming Guide (shared memory banks)"],
  });

  Deck.slide({
    id: "gpu-hbm",
    title: "HBM: a 5,120-bit bus, and bandwidth that grows slower than FLOP/s",
    body: `<p><b>HBM</b> (high-bandwidth memory) is DRAM in stacks of dies placed beside the GPU die on the same package. The H100 SXM has 5 HBM3 stacks of 16 GB on a 5,120-bit bus, 1,024 bits per stack. NVIDIA gives a data rate of 2,619 MHz DDR, so each pin carries 5.238 Gb/s and the bus 5,120 × 5.238 / 8 = 3,352 GB/s, which the datasheet rounds to 3.35 TB/s.</p>
<p>The bandwidth comes from width: 5,120 pins at a modest 5.2 Gb/s each. And it has grown slower than compute: from V100 to H100, dense FP16 tensor throughput grew 7.9 times and bandwidth 3.7 times. The FLOPs per byte a kernel needs to keep the tensor cores busy went from 139 to 295.</p>
<p>A kernel at 200 FLOPs per byte was limited by arithmetic on a V100. On an H100 it is limited by memory.</p>`,
    fig: {
      type: "table", dense: true, caption: "Memory interface and FLOPs per byte, three generations (SXM)", align: ["l", "l", "r", "r", "r", "r"],
      cols: ["GPU", "Memory", "Data rate, MHz DDR", "GB/s", "FP16 tensor TFLOP/s", "FLOPs per byte"],
      rows: [
        ["V100 SXM2", "HBM2, 4 stacks, 4,096-bit", "877.5", "900", "125", "139"],
        ["A100 SXM4 40 GB", "HBM2, 5 stacks, 5,120-bit", "1,215", "1,555", "312", "201"],
        ["A100 SXM4 80 GB", "HBM2e", "–", "2,039", "312", "153"],
        ["H100 SXM5", "HBM3, 5 stacks, 5,120-bit", "2,619", "3,352", "989.4", "295"],
      ],
      highlight: [3],
      note: "Bandwidth = bus bits × 2 × data rate / 8. V100: 4,096 × 2 × 877.5 MHz / 8 = 899 GB/s, rated 900. Dense FP16 tensor throughput; the last column is TFLOP/s ÷ TB/s.",
    },
    terms: [{ term: "HBM", def: "High-bandwidth memory: stacked DRAM on the GPU package with a very wide bus (1,024 bits per stack). The H100 SXM's 5 HBM3 stacks give 3.35 TB/s." }],
    src: ["NVIDIA Tesla V100 GPU Architecture whitepaper (2017) and V100 datasheet", "NVIDIA A100 Tensor Core GPU Architecture whitepaper (Table 4) and A100 datasheet", "NVIDIA H100 whitepaper V1.04 (Table 3)"],
  });

  Deck.slide({
    id: "gpu-peak",
    title: "Peak FLOP/s multiplies four numbers; real kernels lose some of each",
    body: `<p>989.4 TFLOP/s is a product: 132 SMs × 4 tensor cores × 1,024 FLOPs per tensor core per clock × 1.83 GHz. A kernel reaches it only if every factor holds at once.</p>
<ul><li><b>Every SM busy.</b> A grid with fewer blocks than SMs, or a last wave that fills only some of them, leaves SMs idle (chapter 6.6).</li>
<li><b>Every tensor core fed every clock.</b> Operands must already sit in registers or shared memory. A clock spent waiting for a load, a barrier, or a softmax between two matrix products is lost.</li>
<li><b>The full clock.</b> 1.83 GHz is a boost clock, held only within the power limit, and power depends on the data. On a 350 W H800 PCIe, Luo et al. measured the same tensor-core loop about 9% slower on random inputs than on zeros.</li></ul>
<p>The plate lists measured fractions of peak on Hopper GPUs, from one instruction in a loop to a 16,384-GPU training run.</p>`,
    fig: {
      type: "table", caption: "Measured throughput on Hopper GPUs, as a fraction of each GPU's dense peak", align: ["l", "l", "r", "r"],
      cols: ["Workload", "GPU", "Achieved", "Of peak"],
      rows: [
        ["Hopper's warpgroup matrix-multiply instruction (<code>wgmma</code>, chapter 6.6) on 64 × 256 × 16 tiles in a loop, FP16 in, FP32 accumulate, zero inputs", "H800 PCIe", "728.5 TFLOP/s", "96%"],
        ["the same with random inputs", "H800 PCIe", "665.4 TFLOP/s", "88%"],
        ["FlashAttention-3 forward pass, FP16", "H100 SXM5", "740 TFLOP/s", "75%"],
        ["Llama 3 405B pre-training, BF16, 16,384 GPUs", "H100, 700 W", "400 TFLOP/s per GPU", "41% (reported model FLOPs utilization)"],
      ],
      note: "Peaks: 756.5 TFLOP/s FP16 for the H800 PCIe as stated by Luo et al.; 989.4 TFLOP/s for the H100 SXM5. Model FLOPs utilization (MFU) is the FLOP/s the model's own arithmetic needs, divided by the peak.",
    },
    src: ["Luo et al., Benchmarking and Dissecting the Nvidia Hopper GPU Architecture, IPDPS 2024 (Table VIII)", "Shah et al., FlashAttention-3, 2024", "Dubey et al., The Llama 3 Herd of Models, 2024 (Table 4)", "NVIDIA H100 whitepaper V1.04"],
  });

  Deck.slide({
    id: "gpu-nvlink",
    title: "Eight H100s share an NVLink switch; the CPUs sit behind PCIe",
    body: `<p>In a DGX H100 server, eight H100 SXM GPUs connect through four third-generation NVSwitch chips. Each GPU has 18 NVLink 4 links of 25 GB/s per direction: 450 GB/s out and 450 GB/s in, the 900 GB/s on the datasheet. Through the switches, every GPU reaches every other at that rate.</p>
<p>The two host CPUs, Intel Xeon 8480C with 56 cores each, reach the GPUs over PCIe Gen 5 x16: 64 GB/s per direction, one seventh of NVLink.</p>
<p>Moving 1 GB takes 0.30 ms from a GPU's own HBM, 2.2 ms to another GPU over NVLink and 15.6 ms to or from the host over PCIe. Training keeps weights and activations on the GPUs and exchanges gradients and activations between them; Part 7 builds its parallelism on these numbers.</p>`,
    fig: { type: "gpu-dgx", caption: "A DGX H100: 8 GPUs, an NVSwitch fabric of 4 chips, 2 host CPUs" },
    terms: [{ term: "NVLink", def: "NVIDIA's GPU-to-GPU link. On H100: 18 links of 25 GB/s per direction, 900 GB/s per GPU counting both directions, switched by NVSwitch chips within a server." }],
    src: ["NVIDIA H100 whitepaper V1.04 (NVLink, PCIe Gen 5, DGX H100)", "NVIDIA DGX H100 User Guide (CPUs)"],
  });

  Deck.slide({
    id: "gpu-datasheet",
    title: "Four generations of data-center GPUs",
    body: `<p>The table compares the SXM form of each generation with dense throughput. From V100 (2017) to B200, dense FP16 tensor throughput grew 18 times, HBM bandwidth 8.9 times and FP32 throughput without tensor cores 4.8 times. Each recent generation also added a narrower format at twice the rate of the previous one: FP8 on H100, FP4 on B200.</p>
<p>The B200 figures are NVIDIA's totals for the 8-GPU HGX B200 divided by 8. NVIDIA's pages disagree on B200 memory: 180 GB per GPU for HGX and DGX B200, 192 GB in a Blackwell table on its developer blog. The table uses 180 GB.</p>`,
    fig: {
      type: "table", caption: "V100, A100, H100 and B200, SXM forms, dense", dense: true, rowHeads: true, align: ["l", "r", "r", "r", "r"],
      cols: ["", "V100 SXM2", "A100 SXM4 80 GB", "H100 SXM5", "B200 (HGX)"],
      rows: [
        ["FP16/BF16 tensor, TFLOP/s", "125 (FP16 only)", "312", "989.4", "2,250"],
        ["FP8 tensor, TFLOP/s", "–", "–", "1,978.9", "4,500"],
        ["FP4 tensor, TFLOP/s", "–", "–", "–", "9,000"],
        ["FP32 lanes, TFLOP/s", "15.7", "19.5", "66.9", "75"],
        ["FP64, TFLOP/s", "7.8", "9.7", "33.5", "37"],
        ["Memory", "32 GB HBM2", "80 GB HBM2e", "80 GB HBM3", "180 GB HBM3e"],
        ["Bandwidth", "900 GB/s", "2,039 GB/s", "3,352 GB/s", "8 TB/s"],
        ["FP16 FLOPs per byte", "139", "153", "295", "281"],
        ["L2 cache", "6 MB", "40 MB", "50 MB", "–"],
        ["NVLink per GPU", "300 GB/s", "600 GB/s", "900 GB/s", "1,800 GB/s"],
        ["Power (TDP)", "300 W", "400 W", "700 W", "1,000 W"],
        ["Transistors", "21.1 B", "54.2 B", "80 B", "208 B, 2 dies"],
      ],
      note: "Sparse (2:4) tensor rates are twice the dense ones on A100, H100 and B200; V100 has no sparsity support. B200: FP64 is NVIDIA's 'FP64/FP64 Tensor Core' figure; power is the configurable maximum per GPU in HGX/DGX B200.",
    },
    check: {
      q: "From A100 SXM4 80 GB to H100 SXM5, what happened to the FLOPs per byte a BF16 kernel needs to be limited by arithmetic rather than memory?",
      options: ["It stayed near 150, because FLOP/s and bandwidth grew together", "It roughly doubled, from 153 to 295", "It halved, because HBM3 is faster than HBM2e", "It did not change: the ratio is a property of the tensor cores"],
      answer: 1,
      why: "Dense BF16 grew 989.4 / 312 = 3.2 times while bandwidth grew 3,352 / 2,039 = 1.6 times, so the ratio went from 312 / 2.039 = 153 to 989.4 / 3.352 = 295. A kernel at 200 FLOPs per byte moved from arithmetic-limited to memory-limited.",
    },
    src: ["NVIDIA V100 datasheet and Volta whitepaper", "NVIDIA A100 datasheet and Ampere whitepaper", "NVIDIA H100 whitepaper V1.04", "NVIDIA HGX platform page and DGX B200 page (B200)", "NVIDIA Blackwell architecture page", "NVIDIA Technical Blog, Inside NVIDIA Blackwell Ultra, 2025 (Table 2)"],
  });

  // ================================================================== 6.3
  Deck.chapter({ title: "The CUDA programming model" });

  Deck.slide({
    id: "gpu-host",
    title: "Host and device: two processors with separate memories",
    body: String.raw`<p>A CUDA program runs on the <b>host</b>, the CPU and its memory, and launches work on the <b>device</b>, the GPU and its HBM. The memories are separate: a pointer returned by <code>cudaMalloc</code> addresses HBM, and the CPU cannot dereference it. Data crosses the link as explicit copies, or as direct loads from page-locked host memory, which CUDA maps into the device's address space (zero-copy access). A GH200 goes further: its Grace CPU and H100 share coherent memory over NVLink-C2C.</p>
<p>The host runs the control flow. It allocates device memory, copies inputs in, launches kernels and copies results out. A launch places the kernel in a queue and returns at once; the CPU continues while the GPU works and waits only when it needs a result.</p>
<p>The link is the slowest path in the system. PCIe Gen 5 x16 carries 64 GB/s per direction, 52 times less than the 3.35 TB/s between an H100 and its own HBM. Data that crosses it should cross once.</p>`,
    fig: {
      type: "diagram", caption: "The host and the device, and the bandwidth of each path", w: 16, h: 7.4, unit: 40,
      groups: [{ x: 0.2, y: 0.5, w: 5.2, h: 6.7, label: "Host" }, { x: 9.0, y: 0.5, w: 6.8, h: 6.7, label: "Device: H100 SXM" }],
      nodes: [
        { id: "cpu", x: 0.5, y: 1.2, w: 4.6, h: 1.4, label: "CPU", sub: "runs main()" },
        { id: "dram", x: 0.5, y: 5.2, w: 4.6, h: 1.6, label: "Host memory", sub: "malloc'd arrays" },
        { id: "sm", x: 9.3, y: 1.2, w: 6.2, h: 1.4, label: "132 SMs", sub: "run the kernels", kind: "hi" },
        { id: "l2", x: 9.3, y: 3.2, w: 6.2, h: 1.2, label: "L2 cache, 50 MB" },
        { id: "hbm", x: 9.3, y: 5.2, w: 6.2, h: 1.6, label: "HBM3, 80 GB", sub: "cudaMalloc'd arrays", kind: "dark" },
      ],
      edges: [
        { from: "cpu", to: "sm", dashed: true, label: "launch, queued" },
        { from: "dram", to: "hbm", both: true, label: "PCIe 64 GB/s" },
        { from: "sm", to: "l2", both: true },
        { from: "l2", to: "hbm", both: true, label: "3.35 TB/s", dx: 40, dy: 8 },
        { from: "cpu", to: "dram", both: true },
      ],
    },
    terms: [
      { term: "Host", def: "The CPU and its memory in a CUDA program. It allocates device memory, copies data and launches kernels." },
      { term: "Device", def: "The GPU and its own memory (HBM). Kernels run here; device pointers address HBM." },
    ],
    note: "Managed (unified) memory hides the copies, migrating pages on demand, but the bytes still cross the same link.",
    src: ["CUDA C++ Programming Guide (programming model, device memory, mapped page-locked host memory)", "NVIDIA H100 whitepaper V1.04 (PCIe Gen 5: 64 GB/s per direction)", "NVIDIA GH200 Grace Hopper Superchip whitepaper (coherent memory over NVLink-C2C)"],
  });

  Deck.slide({
    id: "gpu-launch",
    title: "A kernel launch creates a grid of thread blocks",
    layout: "wide",
    body: `<p>A <b>kernel</b> is a function marked <code>__global__</code>. The host calls it with a launch configuration, <code>add&lt;&lt;&lt;grid, block&gt;&gt;&gt;(a, b, c, n)</code>, which creates a <b>grid</b> of <code>grid</code> <b>thread blocks</b>, each of <code>block</code> threads. Every thread runs the same function body. Threads differ only in their coordinates: <code>threadIdx</code> (position within the block) and <code>blockIdx</code> (position of the block in the grid), with the sizes <code>blockDim</code> and <code>gridDim</code>; each has up to three dimensions. A block holds at most 1,024 threads, and a grid can have up to 2<sup>31</sup> − 1 blocks along x.</p>
<p>The hardware hands blocks to SMs as resources free up. A block runs entirely on one SM, so its threads can share that SM's shared memory. An H100 SM holds at most 32 blocks and 2,048 threads at once; the rest of the grid waits for earlier blocks to finish.</p>`,
    fig: {
      type: "diagram", caption: "<code>add&lt;&lt;&lt;262,144, 256&gt;&gt;&gt;</code> on an H100: which blocks can run at once", w: 24, h: 8.6, unit: 40,
      groups: [{ x: 0.2, y: 0.5, w: 9.4, h: 5.8, label: "Grid: gridDim.x = 262,144" }, { x: 0.2, y: 6.8, w: 9.4, h: 1.6, label: "Block 0: blockDim.x = 256" }],
      nodes: [
        { id: "r0", x: 0.5, y: 1.1, w: 8.8, h: 1.0, label: "block 0, block 1, …", kind: "hi" },
        { id: "r1", x: 0.5, y: 2.45, w: 8.8, h: 1.0, label: "… up to 1,056 blocks at once" },
        { id: "r2", x: 0.5, y: 3.8, w: 8.8, h: 1.0, label: "the remaining blocks", kind: "ghost" },
        { id: "r3", x: 0.5, y: 5.15, w: 8.8, h: 0.9, label: "…, block 262,143", kind: "ghost" },
        ...[0, 1, 2, 3].map((t) => ({ id: `t${t}`, x: 0.5 + t * 1.15, y: 7.35, w: 0.95, h: 0.8, label: String(t) })),
        { id: "tdots", x: 5.1, y: 7.35, w: 1.4, h: 0.8, label: "…", kind: "ghost" },
        { id: "t255", x: 6.8, y: 7.35, w: 1.6, h: 0.8, label: "255" },
        { id: "sms", x: 14.2, y: 1.1, w: 9.6, h: 2.35, label: "132 SMs", sub: "each holds up to 8 such blocks (2,048 threads)", kind: "blue" },
        { id: "wait", x: 14.2, y: 3.8, w: 9.6, h: 2.25, label: "waiting", sub: "start as running blocks finish", kind: "ghost" },
      ],
      edges: [{ from: "r1", to: "sms" }, { from: "r2", to: "wait", dashed: true }],
      notes: [{ x: 14.2, y: 7.3, text: "the hardware picks the SM and the order" }],
    },
    terms: [
      { term: "Kernel", def: "A function marked __global__ that runs on the GPU, launched from the host with &lt;&lt;&lt;grid, block&gt;&gt;&gt;; every thread of the grid executes it." },
      { term: "Thread block", def: "A group of up to 1,024 threads that run on one SM, can share its shared memory and synchronize with __syncthreads()." },
      { term: "Grid", def: "All the thread blocks of one kernel launch." },
    ],
    src: ["CUDA C++ Programming Guide (kernels, thread hierarchy, compute capability 9.0 limits)"],
  });

  Deck.slide({
    id: "gpu-index",
    title: "Each thread finds its element from blockIdx, blockDim and threadIdx",
    body: String.raw`<p>In a 1-D launch, thread <code>threadIdx.x</code> of block <code>blockIdx.x</code> handles element i below. Block b covers elements b·B to b·B + B − 1, where B is <code>blockDim.x</code>.</p>
<p>Covering n elements takes ⌈n / B⌉ blocks, written in integer arithmetic as <code>(n + B - 1) / B</code>. The last block usually runs past the end of the array, and its extra threads must not touch memory: that is why kernels start with the bounds check <code>if (i &lt; n)</code>.</p>
<p>For n = 1,000,000 and 256 threads per block, the grid has 3,907 blocks and 1,000,192 threads, and the last 192 do nothing. The plate uses small sizes so every thread is visible; pick an element to see which thread owns it.</p>`,
    math: String.raw`i = \texttt{blockIdx.x} \times \texttt{blockDim.x} + \texttt{threadIdx.x}`,
    fig: { type: "gpu-grid", mode: "1d", n: 20, B: 8, pick: 13, caption: "A 1-D grid: each row is one block, each cell one thread and the element it handles" },
    check: {
      q: "A kernel is launched as <code>k&lt;&lt;&lt;(n + 255) / 256, 256&gt;&gt;&gt;</code> with n = 10,000. How many threads fail the bounds check <code>if (i &lt; n)</code>?",
      options: ["0", "16", "240", "256"],
      answer: 2,
      why: "(10,000 + 255) / 256 = 40 blocks in integer division, so 40 × 256 = 10,240 threads run and the last 10,240 − 10,000 = 240 skip the work.",
    },
    src: ["CUDA C++ Programming Guide (thread hierarchy)"],
  });

  Deck.slide({
    id: "gpu-vecadd",
    title: "Vector add, line by line",
    body: `<p>The listing is a complete CUDA program for c = a + b on 2<sup>26</sup> floats. The kernel is three lines. The rest is the host's work: allocate three 268 MB arrays in HBM, copy the inputs in, launch, copy the result out, free.</p>
<p>The launch uses 256 threads per block and (n + 255) / 256 = 262,144 blocks. Here n is a multiple of 256, so no thread is idle, but the bounds check stays: the kernel must be correct for any n.</p>
<p>Each <code>cudaMemcpy</code> waits for the work queued before it in the default stream, so the copy on line 15 also waits for the kernel. Every CUDA call returns an error code that real code checks; the checks are left out here. Click a line number or step through the notes.</p>`,
    fig: {
      type: "code", lang: "cuda", caption: "c = a + b: kernel and host program",
      code: `__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {                            // ha, hb, hc: host arrays
  int n = 1 << 26;                      // 67,108,864 elements
  size_t bytes = n * sizeof(float);     // 268,435,456 bytes
  float *a, *b, *c;                     // device pointers
  cudaMalloc(&a, bytes); cudaMalloc(&b, bytes); cudaMalloc(&c, bytes);
  cudaMemcpy(a, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(b, hb, bytes, cudaMemcpyHostToDevice);
  int block = 256, grid = (n + block - 1) / block;   // 262,144
  add<<< grid, block >>>(a, b, c, n);
  cudaMemcpy(hc, c, bytes, cudaMemcpyDeviceToHost);  // waits
  cudaFree(a); cudaFree(b); cudaFree(c);
}`,
      notes: {
        1: "<code>__global__</code>: runs on the device and is launched from the host. The pointers hold device (HBM) addresses.",
        2: "The thread's global index. Thread 5 of block 3 gets i = 3 × 256 + 5 = 773.",
        3: "The bounds check, then two 4-byte loads and one 4-byte store: 12 bytes for 1 FLOP.",
        7: "1 &lt;&lt; 26 = 67,108,864 elements.",
        8: "4 bytes each: 268,435,456 bytes per array, 805 MB for the three.",
        10: "<code>cudaMalloc</code> reserves HBM and writes the device address into the pointer.",
        11: "Host to device over PCIe: 268 MB per input.",
        13: "Integer division rounds the block count up: (67,108,864 + 255) / 256 = 262,144.",
        14: "The launch queues the kernel and returns to the CPU immediately.",
        15: "Device to host. In the default stream it starts only after the kernel finishes, so it doubles as the synchronization.",
        16: "Frees the device memory.",
      },
      highlight: [2, 3],
    },
    src: ["CUDA C++ Programming Guide (kernels, device memory, default stream)"],
  });

  Deck.slide({
    id: "gpu-copycost",
    title: "The PCIe copies take 52 times as long as the kernel",
    body: `<p>The vector add moves 805 MB through HBM, which takes 240 µs at 3.35 TB/s. The copies move the same bytes over PCIe Gen 5 x16 at 64 GB/s per direction: 537 MB in (8.4 ms) and 268 MB out (4.2 ms). A byte costs 52 times as much over PCIe as from HBM, so the kernel is 1.9% of the end-to-end time.</p>
<p>These are nominal link rates; measured copies are slower. Copies from ordinary pageable host memory are slower again, because the driver stages them through a page-locked buffer. Host arrays allocated with <code>cudaMallocHost</code> (pinned memory) avoid the staging and allow asynchronous copies.</p>
<p>So data should stay on the GPU between kernels. A model's weights are copied in once and stay in HBM for the whole run; only inputs and results cross the link.</p>`,
    fig: {
      type: "calc", caption: "End-to-end time of the vector add, by size and link",
      inputs: [
        { id: "n", label: "elements n", options: [["2²⁰", 20], ["2²⁴", 24], ["2²⁶", 26], ["2²⁸", 28]].map(([l, e]) => ({ label: l, value: 2 ** e })), value: 2 ** 26 },
        { id: "link", label: "host link, per direction", options: [{ label: "PCIe Gen 4 x16, 32 GB/s", value: 32e9 }, { label: "PCIe Gen 5 x16, 64 GB/s", value: 64e9 }, { label: "NVLink-C2C (GH200), 450 GB/s", value: 450e9 }], value: 64e9 },
      ],
      outputs: [
        { label: "copy a and b in", f: (v) => (8 * v.n) / v.link, fmt: "time" },
        { label: "kernel (HBM)", f: (v) => (12 * v.n) / HW.bw, fmt: "time" },
        { label: "copy c out", f: (v) => (4 * v.n) / v.link, fmt: "time" },
        { label: "kernel share of total", f: (v) => ((12 * v.n) / HW.bw) / ((12 * v.n) / v.link + (12 * v.n) / HW.bw), fmt: (x) => pct(x, 1) },
      ],
      note: (v) => `${int(v.n)} floats: ${int(8 * v.n)} bytes in, ${int(4 * v.n)} out, ${int(12 * v.n)} through HBM. Per byte the link costs ${fx(HW.bw / v.link, 1)} times as much as HBM.`,
    },
    note: "GH200 joins a Grace CPU to an H100 with NVLink-C2C: 900 GB/s total, 450 GB/s per direction.",
    src: ["NVIDIA H100 whitepaper V1.04 (PCIe Gen 5)", "NVIDIA A100 datasheet (PCIe Gen 4: 64 GB/s in both directions)", "NVIDIA GH200 Grace Hopper Superchip whitepaper (NVLink-C2C)", "CUDA C++ Programming Guide (page-locked host memory)"],
  });

  Deck.slide({
    id: "gpu-2d",
    title: "2-D grids: one thread per matrix element, row-major",
    body: String.raw`<p>For a matrix, launch a 2-D grid of 2-D blocks with <code>dim3</code> sizes, for example:</p>
<pre><code>dim3 block(32, 8);
dim3 grid((cols + 31) / 32, (rows + 7) / 8);</code></pre>
<p>Thread (x, y) of block (bx, by) then handles the element below.</p>
<p>C and CUDA store matrices <b>row-major</b>: element (row, col) of a matrix with C columns sits at offset row × C + col, so walking along a row walks through consecutive addresses. That is why x, the fastest-changing thread coordinate, should index the column: neighbouring threads then touch neighbouring addresses, which is what memory wants (chapter 6.5).</p>
<p>A 4,096 × 4,096 matrix with 32 × 8 blocks needs a 128 × 512 grid, 65,536 blocks of 256 threads. The plate shows a small case, including the padding threads that fall outside the matrix.</p>`,
    math: [String.raw`\text{col} = b_x B_x + x,\qquad \text{row} = b_y B_y + y`, String.raw`\text{idx} = \text{row} \times C + \text{col}`],
    fig: { type: "gpu-grid", mode: "2d", R: 6, C: 10, bx: 4, by: 4, caption: "A 2-D grid over a matrix: block boundaries in bold, the matrix outlined in red, offsets in the cells" },
    terms: [{ term: "Row-major", def: "Matrix layout where each row is contiguous: element (r, c) of an R × C matrix is at offset r·C + c. The C, CUDA and PyTorch default." }],
    src: ["CUDA C++ Programming Guide (thread hierarchy, dim3)"],
  });

  Deck.slide({
    id: "gpu-sync",
    title: "Blocks run in any order; __syncthreads() orders one block",
    body: String.raw`<p>CUDA requires thread blocks to be independent: they may run in any order, in parallel or one after another, on any SM. A kernel whose blocks spin-wait for a flag set by the last block can hang: if the waiting blocks fill every slot on every SM, the last block never becomes resident. The vector add's 262,144 blocks run in about 248 waves of at most 1,056 resident blocks, 8 per SM.</p>
<p>Within a block, the barrier <code>__syncthreads()</code> holds every thread until all have reached it, and makes the global- and shared-memory accesses they made before it visible to the whole block. In the listing, without it, thread 0 could read <code>s[255]</code> before thread 255, in another warp (another group of 32 threads), has written it.</p>
<p>In an ordinary launch, the only barrier across blocks is the end of the kernel. A cooperative launch guarantees that all blocks are resident at once (here at most 1,056), so <code>this_grid().sync()</code> can be a grid-wide barrier. Hopper adds programmatic dependent launch, which lets the next kernel start early, and <b>thread block clusters</b>: up to 8 blocks (16 with an opt-in) co-scheduled on one GPC, which can wait for each other and read each other's shared memory.</p>`,
    fig: {
      type: "code", lang: "cuda", caption: "Reversing each 256-element chunk through shared memory",
      code: `__global__ void reverse_chunks(float* d) {   // 256 threads per block
  __shared__ float s[256];                    // one copy per block
  int t = threadIdx.x;
  int base = blockIdx.x * 256;
  s[t] = d[base + t];
  __syncthreads();                            // all 256 writes are done
  d[base + t] = s[255 - t];
}`,
      notes: {
        2: "<code>__shared__</code>: 1 KB of the SM's shared memory, allocated for each block for its lifetime and visible to all its threads.",
        5: "Each thread copies one element from global memory into shared memory.",
        6: "The barrier. Threads that arrive early wait here until all 256 have arrived.",
        7: "Thread t reads the element written by thread 255 − t. Thread 0 reads what thread 255 wrote, in warp 7.",
      },
      highlight: [6],
    },
    terms: [{ term: "Thread block cluster", def: "On Hopper, a group of up to 8 thread blocks (16 with an opt-in) scheduled together on SMs of one GPC; its blocks can synchronize and access each other's shared memory (distributed shared memory)." }],
    src: ["CUDA C++ Programming Guide (thread hierarchy: blocks execute independently; synchronization functions; cooperative groups; programmatic dependent launch; thread block clusters)", "NVIDIA Hopper Tuning Guide (cluster size 8 portable, 16 non-portable)", "NVIDIA H100 whitepaper V1.04 (thread block clusters, distributed shared memory)"],
  });

  Deck.slide({
    id: "gpu-streams",
    title: "Streams let copies and kernels overlap",
    body: `<p>A <b>stream</b> is a queue of GPU work that runs in order; work in different streams may run concurrently. A GPU whose <code>asyncEngineCount</code> property is 2 can run a host-to-device copy, a device-to-host copy and kernels at the same time, if the host memory is pinned. Split the vector add into 4 chunks, each in its own stream: copy in with <code>cudaMemcpyAsync</code>, run the kernel on the chunk, copy out. While chunk 0 is computed, chunk 1 is copied in; while chunk 1 is copied out, chunk 2 is copied in.</p>
<p>The total falls from 12.8 ms to 9.5 ms: the time to copy all inputs in, plus one chunk's kernel and copy out. The copies still dominate. Streams hide shorter work behind the longest; they do not make PCIe faster.</p>`,
    fig: {
      type: "timeline", caption: "The vector add on 2²⁶ elements over PCIe Gen 5, with two copy engines: one stream, then four", unit: "ms", total: 13,
      rows: ["1 stream: in", "1 stream: add", "1 stream: out", "4 streams: in", "4 streams: add", "4 streams: out"],
      items: (() => {
        const inT = (8 * 2 ** 26) / HW.pcie * 1e3, kT = (12 * 2 ** 26) / HW.bw * 1e3, outT = (4 * 2 ** 26) / HW.pcie * 1e3;
        const it = [
          { row: 0, start: 0, len: inT, label: "copy a, b in", kind: "s2", tip: `${fx(inT, 2)} ms` },
          { row: 1, start: inT, len: kT, label: "", kind: "s1", tip: `kernel ${fx(kT * 1e3, 1)} µs` },
          { row: 2, start: inT + kT, len: outT, label: "copy c out", kind: "s3", tip: `${fx(outT, 2)} ms` },
        ];
        for (let k = 0; k < 4; k++) {
          const s0 = (k + 1) * (inT / 4);
          it.push({ row: 3, start: k * (inT / 4), len: inT / 4, label: `in ${k}`, kind: "s2", tip: `chunk ${k}: ${fx(inT / 4, 2)} ms` });
          it.push({ row: 4, start: s0, len: kT / 4, label: "", kind: "s1", tip: `kernel on chunk ${k}: ${fx((kT / 4) * 1e3, 1)} µs` });
          it.push({ row: 5, start: s0 + kT / 4, len: outT / 4, label: `out ${k}`, kind: "s3", tip: `chunk ${k}: ${fx(outT / 4, 2)} ms` });
        }
        return it;
      })(),
      note: "Kernels (blue) are 240 µs in total and barely visible at this scale. Totals: 8.39 + 0.24 + 4.19 = 12.82 ms in one stream; 8.39 + 0.06 + 1.05 = 9.50 ms in four.",
    },
    check: {
      q: "Split the vector add into 8 chunks in 8 streams instead of 4. What is the total time?",
      options: ["4.75 ms: twice the overlap", "8.94 ms", "9.50 ms: unchanged", "12.82 ms: the copies serialize"],
      answer: 1,
      why: "Every input byte must cross PCIe before the last chunk's kernel can start: 8.39 ms. After it come one chunk's kernel (0.24 / 8 = 0.03 ms) and its copy out (4.19 / 8 = 0.52 ms): 8.94 ms. More chunks shrink only that tail; the input copy is the floor.",
    },
    terms: [{ term: "Stream", def: "An in-order queue of GPU operations (copies, kernels). Operations in different streams may overlap." }],
    src: ["CUDA C++ Programming Guide (streams, asynchronous concurrent execution, page-locked memory)"],
  });
  // ================================================================== 6.4
  Deck.chapter({ title: "How an SM executes threads" });

  Deck.slide({
    id: "gpu-warps",
    title: "A warp is 32 threads that issue one instruction together",
    body: String.raw`<p>An SM cuts each block into <b>warps</b> of 32 threads with consecutive thread numbers, x + y × blockDim.x (+ z × blockDim.x × blockDim.y), and each scheduler issues one instruction for a whole warp at a time: <b>SIMT</b>, single instruction, multiple threads. A 256-thread block is 8 warps; an H100 SM holds at most 64. A 100-thread block occupies 4 warps, and the last has 4 active threads and 28 idle.</p>
<p>SIMT differs from SIMD vector code in what the programmer writes. Each thread is scalar code with its own registers (and, since Volta, its own program counter); the hardware forms the 32-wide vector and a mask of active lanes. Every lane computes its own address, so every load is a gather, which the hardware merges into as few 32-byte sectors as it can (chapter 6.5). Lanes may take different branches, at the cost of running both paths. Units narrower than 32 lanes, such as a processing block's 16 FP64 lanes, take two clocks per warp instruction.</p>`,
    fig: { type: "gpu-grid", mode: "warps", shape: "16x16", caption: "A block of 256 threads, cut into 8 warps of 32 consecutive thread numbers" },
    terms: [
      { term: "Warp", def: "32 threads with consecutive thread numbers in a block. The SM issues one instruction at a time for all 32 (SIMT)." },
      { term: "SIMT", def: "Single instruction, multiple threads: one instruction is issued for the 32 threads of a warp, each with its own registers and (since Volta) its own program counter." },
    ],
    src: ["CUDA C++ Programming Guide (SIMT architecture, independent thread scheduling, CC 9.0 limits)"],
  });

  Deck.slide({
    id: "gpu-hiding",
    title: "Schedulers hide latency by issuing from another warp",
    body: String.raw`<p>When a warp's next instruction needs a value still on its way from memory, that warp stalls. Its scheduler does not wait: in the same clock it issues an instruction from another ready warp. A warp stays on one of the SM's four schedulers from launch to completion, so the candidates are that scheduler's warps, at most 16.</p>
<p>The plate models one scheduler. Each warp repeats a load and then c instructions that need the loaded value; a load returns after L cycles. The scheduler keeps issuing from one warp until it stalls, then switches to the lowest-numbered ready warp. One warp issues c + 1 instructions every L + c cycles, so W warps keep the scheduler busy a fraction U of the time. The model leaves out arithmetic latency (the c instructions issue on consecutive clocks) and independent loads within a warp.</p>
<p>With L = 30 cycles, about one shared-memory load, and c = 4, seven warps suffice. An HBM load takes 273 ns or more (measured on an H800 PCIe), about 540 cycles at the H100's 1.98 GHz: a scheduler would need 109 warps. Such loops need more independent work per warp, the subject of the sheet on instruction-level parallelism below.</p>`,
    math: String.raw`U = \min\!\left(1,\ \frac{W\,(c+1)}{L + c}\right)`,
    fig: { type: "gpu-warpsched", W: 4, c: 4, L: 30, caption: "One warp scheduler: who issues in each clock cycle" },
    terms: [{ term: "Latency hiding", def: "Keeping execution units busy while some threads wait for memory, by issuing from other warps (or other independent instructions) in the meantime." }],
    src: ["CUDA C++ Programming Guide (hardware multithreading)", "Nsight Compute Profiling Guide (a warp resides on one SM sub-partition from launch to completion)", "Luo et al., IPDPS 2024 (latencies on an H800 PCIe at up to 1.755 GHz: global 478.8 cycles, shared 29.0)"],
  });

  Deck.slide({
    id: "gpu-little",
    title: "Little's law: to saturate HBM, keep about 900 KB in flight",
    body: String.raw`<p>Little's law says that the average number of items in a system equals their arrival rate times the time each spends inside. For memory, bytes in flight = bandwidth × latency: the <b>bandwidth-delay product</b>. An HBM load took 273 ns or more on the H800 PCIe of the previous sheets; the H100 SXM's HBM3 may differ, so use it as an estimate. Reading HBM at 3.35 TB/s then needs 3.35 × 10¹² × 273 × 10⁻⁹ = 915 KB of loads outstanding at every moment: 6.9 KB per SM.</p>
<p>A warp load of one 4-byte float per thread moves 128 bytes, so each SM needs 55 of them in flight. With one outstanding load per warp that is 55 resident warps, 86% occupancy; with 16-byte loads (<code>float4</code>) it is 14. Under load the latency grows, so real kernels need more still.</p>
<p>Set the plate's inputs to see how many warps and how many bytes per load a kernel needs before HBM bandwidth, not latency, becomes the limit.</p>`,
    fig: {
      type: "calc", caption: "Bytes in flight and achievable HBM bandwidth on an H100",
      inputs: [
        { id: "lat", label: "memory latency, ns", min: 150, max: 1000, step: 1, value: 273, fmt: "int" },
        { id: "b", label: "bytes per thread per load", options: optList([4, 8, 16]), value: 4 },
        { id: "ilp", label: "loads in flight per thread", options: optList([1, 2, 4]), value: 1 },
        { id: "w", label: "resident warps per SM", min: 1, max: 64, step: 1, value: 48, fmt: "int" },
      ],
      outputs: [
        { label: "bytes in flight needed per SM", f: (v) => (HW.bw * v.lat * 1e-9) / HW.sms, fmt: (x) => `${int(x)} B` },
        { label: "bytes in flight per SM", f: (v) => v.w * 32 * v.b * v.ilp, fmt: (x) => `${int(x)} B` },
        { label: "achievable bandwidth", f: (v) => G.littleBW(v.w, v.b, v.ilp, v.lat), fmt: (x) => `${fx(x / 1e12, 2)} TB/s` },
        { label: "share of 3.35 TB/s", f: (v) => G.littleBW(v.w, v.b, v.ilp, v.lat) / HW.bw, fmt: (x) => pct(x, 1) },
      ],
      note: (v) => { const need = Math.ceil((HW.bw * v.lat * 1e-9) / HW.sms / (32 * v.b * v.ilp)); return `Achievable = min(3.35 TB/s, 132 SMs × ${v.w} warps × 32 threads × ${v.b} B × ${v.ilp} / ${v.lat} ns). Full bandwidth needs ${need} warps per SM with these loads${need > 64 ? " (> 64: unreachable)" : ""}.`; },
    },
    terms: [{ term: "Little's law", def: "Average occupancy = throughput × latency. For memory: bytes in flight = bandwidth × latency, the bandwidth-delay product." }],
    src: ["Luo et al., IPDPS 2024 (global-memory latency 478.8 cycles on an H800 PCIe, maximum clock 1.755 GHz)", "NVIDIA H100 whitepaper V1.04"],
  });

  Deck.slide({
    id: "gpu-diverge",
    title: "Divergent branches run both paths, with lanes switched off",
    body: `<p>If the threads of a warp take different sides of a data-dependent branch, the warp executes each side in turn and disables the threads that are not on it. A divergent if/else costs the sum of both paths, and all 32 lanes pay it.</p>
<p>The pattern matters more than the fraction. With <code>if (threadIdx.x % 2 == 0)</code> every warp has even and odd threads, so every warp pays A + B. With <code>if ((threadIdx.x / 32) % 2 == 0)</code> half the threads also take each side, but each warp is uniform and pays only A or B. A test such as <code>x[i] &gt; 0</code> on data of random sign diverges in nearly every warp: the chance that all 32 lanes agree is 2 × 2<sup>−32</sup>.</p>
<p>The plate runs a 128-thread block (4 warps) through an if/else whose two sides take A and B cycles. For short branches the compiler may use predication instead of a jump: every lane issues the instructions of both sides, and only lanes whose condition holds write results.</p>`,
    fig: { type: "gpu-diverge", cond: "even", caption: "Four warps through an if/else: which lanes run each side" },
    terms: [{ term: "Branch divergence", def: "Threads of one warp taking different sides of a branch. The warp executes each side in turn with the other lanes masked off." }],
    src: ["CUDA C++ Programming Guide (SIMT architecture: divergence)"],
  });

  Deck.slide({
    id: "gpu-occupancy",
    title: "Occupancy: what limits the warps resident on an SM",
    body: String.raw`<p><b>Occupancy</b> is the number of warps resident on an SM divided by the maximum, 64 on an H100. A warp keeps its registers, and its block its shared memory, until the block finishes, so a kernel's resource use caps how many fit.</p>
<p>The rules for compute capability 9.0, as NVIDIA's occupancy calculator applies them: registers are allocated per warp in units of 256, and each of the 4 processing blocks has 16,384. Shared memory is allocated per block in units of 128 bytes, plus 1 KB that CUDA reserves per block, out of 228 KB; more than 48 KB per block requires dynamic shared memory and an explicit opt-in (<code>cudaFuncSetAttribute</code>). At most 32 blocks and 64 warps fit. The resident block count is the smallest of the four limits.</p>
<p>With 256 threads, 64 registers per thread and 16 KB of shared memory, the registers allow 4 blocks, 32 warps: 50% occupancy. With 32 registers per thread, 8 blocks, 100%. The compiler chooses the register count; <code>__launch_bounds__</code> or <code>-maxrregcount</code> caps it, and values that no longer fit spill to local memory, which lives in HBM behind the caches.</p>`,
    fig: { type: "gpu-occ", T: 256, R: 64, S: 16, caption: "Resident blocks per SM on an H100 (compute capability 9.0)" },
    check: {
      q: "A kernel uses 128 registers per thread, 256-thread blocks and no shared memory. How many warps can be resident on one H100 SM?",
      options: ["64", "32", "16", "8"],
      answer: 2,
      why: "Each warp needs 128 × 32 = 4,096 registers. A processing block has 16,384, room for 4 warps, so the SM holds 16 warps: 2 blocks of 8 warps, 25% occupancy.",
    },
    terms: [{ term: "Occupancy", def: "Resident warps per SM divided by the maximum (64 on H100). Limited by registers, shared memory, block size and the 32-block limit." }],
    src: ["CUDA C++ Programming Guide (compute capability 9.0 limits; shared memory above 48 KB; local memory)", "CUDA C++ Best Practices Guide (occupancy; register allocation in units of 256 per warp; register pressure)", "CUDA Toolkit cuda_occupancy.h (allocation granularities, 4 sub-partitions)", "NVIDIA Hopper Tuning Guide (1 KB of shared memory reserved per block)"],
  });

  Deck.slide({
    id: "gpu-ilp",
    title: "Occupancy is a means: independent loads per warp hide latency too",
    body: `<p>What hides latency is independent work in flight, and it can come from more warps or from more independent instructions per warp: <b>instruction-level parallelism</b> (ILP). A thread that issues four loads before using any of them has four requests outstanding, so a quarter as many warps keep memory just as busy.</p>
<p>The plate applies Little's law to the whole GPU streaming from HBM. With 4-byte loads, 273 ns of latency and one load in flight per thread, full bandwidth needs 55 warps per SM, 86% occupancy; with four loads in flight, 14 warps, 22%.</p>
<p>Volkov measured this in 2010 on a GTX 480: 87% of peak memory bandwidth at 8% occupancy. Fast matrix-multiply kernels run at low occupancy on purpose: in CUTLASS the accumulators take at least half of each thread's registers, so few warps fit, and each warp has many independent multiply-adds to issue.</p>`,
    fig: {
      type: "plot", caption: "Share of 3.35 TB/s reached, by resident warps per SM and loads in flight per thread",
      x: [0, 64], y: [0, 1.08], xlabel: "resident warps per SM",
      xticks: [0, 8, 16, 24, 32, 40, 48, 56, 64], yticks: [0, 0.25, 0.5, 0.75, 1], yaxis: { fmt: (v) => `${Math.round(v * 100)}%` },
      extra: (p) => ({ after(svg, X, Y) {
        [1, 2, 4].forEach((ilp, k) => {
          const need = Math.ceil((HW.bw * p.lat * 1e-9) / HW.sms / (32 * [4, 8, 16][p.bi] * ilp)), y = Y(0.36 - 0.11 * k);
          svg.appendChild(s("line", { x1: X(38), x2: X(41.5), y1: y - 4, y2: y - 4, class: `gpu-key s${k + 1}` }));
          svg.appendChild(s("text", { x: X(42.5), y, class: "gpu-plot-l" }, `${ilp} load${ilp > 1 ? "s" : ""}: ${need > 64 ? "unreachable" : `full at ${need} warps`}`));
        });
      } }),
      params: [{ name: "lat", label: "latency, ns", min: 150, max: 800, step: 1, value: 273, fmt: "int" }, { name: "bi", label: "bytes per load", min: 0, max: 2, step: 1, value: 0, fmt: (v) => String([4, 8, 16][v]) }],
      fns: [1, 2, 4].map((ilp, k) => ({ f: (x, p) => G.littleBW(x, [4, 8, 16][p.bi], ilp, p.lat) / HW.bw, cls: ["s1", "s2", "s3"][k], samples: 256 })),
      readouts: [1, 2, 4].map((ilp) => ({ label: `warps for full bandwidth, ILP ${ilp}`, f: (p) => Math.ceil((HW.bw * p.lat * 1e-9) / HW.sms / (32 * [4, 8, 16][p.bi] * ilp)), fmt: (x) => (x > 64 ? `${x} (> 64: unreachable)` : String(x)) })),
    },
    terms: [{ term: "Instruction-level parallelism (ILP)", def: "Independent instructions within one thread that can be in flight at the same time, such as several loads issued before any result is used." }],
    src: ["Volkov, Better Performance at Lower Occupancy, GTC 2010 (slide 38)", "CUTLASS documentation, Efficient GEMM", "Luo et al., IPDPS 2024 (latency on an H800 PCIe)"],
  });

  Deck.slide({
    id: "gpu-shuffle",
    title: "Warp shuffles add 32 values in 5 steps without shared memory",
    body: `<p>The threads of a warp can read each other's registers with shuffle instructions. <code>__shfl_down_sync(mask, v, d)</code> returns the value of <code>v</code> held by the lane d places higher; a lane whose source would lie past lane 31 gets its own value back.</p>
<p>Summing a warp's 32 values takes five shuffles, with d = 16, 8, 4, 2, 1:</p>
<pre><code>for (int d = 16; d &gt; 0; d /= 2)
  v += __shfl_down_sync(0xffffffff, v, d);</code></pre>
<p>After the first, lanes 0 to 15 each hold a sum of two values; after the second, lanes 0 to 7 hold sums of four; after the fifth, lane 0 holds the total. That is log₂ 32 = 5 steps with no shared memory and no <code>__syncthreads()</code>: each exchange happens for all the lanes named in the mask, here all 32, at once. The mask exists because of independent thread scheduling: since Volta the lanes of a warp need not be executing together, so the <code>_sync</code> shuffles name the lanes that must take part.</p>`,
    fig: { type: "gpu-shfl", seed: 32, start: 1, caption: "Tree sum of 32 lane values with <code>__shfl_down_sync</code>" },
    terms: [{ term: "Warp shuffle", def: "An instruction (__shfl_sync, __shfl_down_sync, …) by which a lane reads another lane's register within the same warp, without shared memory." }],
    src: ["CUDA C++ Programming Guide (warp shuffle functions)"],
  });

  Deck.slide({
    id: "gpu-reduce",
    title: "A full reduction: grid-stride loop, warp shuffles, one atomic per block",
    body: `<p>The listing sums n floats. Each thread first adds up elements one grid-width apart, a <b>grid-stride loop</b>, so a fixed grid of 1,056 blocks of 256 threads (the most an H100 holds at once) covers any n. For n = 2<sup>26</sup> each of the 270,336 threads adds about 248 elements, and every warp's loads are coalesced.</p>
<p>Each warp then reduces its 32 partial sums with shuffles. Lane 0 of each warp writes its sum to shared memory, and after a barrier the first warp reduces those 8 values. One <code>atomicAdd</code> per block adds the block's total to the result: 1,056 atomics instead of 67 million.</p>
<p>The kernel does 1 FLOP per 4 bytes read, so HBM sets its time: 268 MB at 3.35 TB/s is 80 µs. The order of the additions depends on scheduling and on the atomics, and floating-point addition is not associative, so the last bits of the result can change from run to run.</p>`,
    fig: {
      type: "code", lang: "cuda", caption: "Sum of n floats: one pass, one atomic per block",
      code: `__global__ void sum(const float* x, float* out, int n) {
  int stride = gridDim.x * blockDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = 0.f;
  for (; i < n; i += stride) v += x[i];         // grid-stride loop
  for (int d = 16; d > 0; d /= 2)
    v += __shfl_down_sync(0xffffffff, v, d);    // lane 0: warp sum
  __shared__ float warp_sum[32];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (lane == 0) warp_sum[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < blockDim.x / 32) ? warp_sum[lane] : 0.f;
    for (int d = 16; d > 0; d /= 2)
      v += __shfl_down_sync(0xffffffff, v, d);
    if (lane == 0) atomicAdd(out, v);           // one per block
  }
}
// launch: sum<<< 1056, 256 >>>(x, out, n), with *out = 0`,
      notes: {
        2: "The total number of threads in the grid: 1,056 × 256 = 270,336.",
        5: "Thread i reads x[i], x[i + 270,336], … In each step the 32 threads of a warp read 32 consecutive floats: 128 bytes, fully coalesced.",
        7: "Five shuffles: lane 0 ends up with the sum of the warp's 32 partial sums.",
        8: "One slot per warp; a 256-thread block uses 8 of the 32.",
        11: "Every warp's sum must be in shared memory before warp 0 reads them.",
        13: "Warp 0 loads the 8 warp sums (lanes 8 to 31 use 0) and reduces them the same way.",
        16: "One atomic add to global memory per block: 1,056 in total.",
      },
      highlight: [5, 7, 16],
    },
    check: {
      q: "The kernel runs as 1,056 blocks of 256 threads. How many <code>atomicAdd</code> calls does it make for n = 2²⁶, and for n = 2²⁰?",
      options: ["1,056 and 1,056", "262,144 and 4,096", "2,097,152 and 32,768", "67,108,864 and 1,048,576"],
      answer: 0,
      why: "One atomic per block, made by lane 0 of warp 0 after the block's reduction, and the grid has 1,056 blocks whatever n is: the grid-stride loop gives each thread more elements instead. One atomic per 256 elements would be 262,144 and 4,096; one per warp of elements, 2,097,152 and 32,768.",
    },
    terms: [{ term: "Grid-stride loop", def: "A loop in which each thread handles elements i, i + G, i + 2G, … where G is the total number of threads, so a fixed-size grid covers any n." }],
    src: ["CUDA C++ Programming Guide (warp shuffle functions, atomic functions)"],
  });
  // ================================================================== 6.5
  Deck.chapter({ title: "Memory access" });

  Deck.slide({
    id: "gpu-coalesce",
    title: "Coalescing: a warp's 32 loads become as few 32-byte sectors as possible",
    body: `<p>Global memory moves data in 32-byte <b>sectors</b>, four to a 128-byte cache line. When a warp executes a load, the hardware collects its 32 addresses and fetches every sector that any of them touches: the request is <b>coalesced</b>. On compute capability 6.0 and later, a warp needs as many transactions as the 32-byte sectors required to serve it.</p>
<p>32 threads reading 32 consecutive floats ask for 128 bytes and get exactly 4 sectors: 100% of the moved bytes are used. Shift the start by one float and the same request straddles 5 sectors, 80%. A stride of 2 floats touches 8 sectors, 50%; a stride of 8 floats or more puts every lane in its own sector, 32 sectors for 128 useful bytes, 12.5%.</p>
<p>Sectors are what the request moves between L1 and L2. A stride-8 load moves 8 times the bytes of a coalesced one through L2 for the same data. HBM moves 8 times the bytes too, and the load takes 8 times as long, only if no other warp uses the rest of those sectors while they are still cached.</p>`,
    fig: { type: "gpu-coalesce", size: 4, stride: 1, offset: 0, caption: "One warp load: the bytes each lane wants (filled) and the 32-byte sectors the request touches (outlined)" },
    terms: [
      { term: "Sector", def: "An aligned 32-byte chunk of memory, the unit in which global memory is moved. A 128-byte L1/L2 cache line has four." },
      { term: "Coalescing", def: "Combining the 32 addresses of a warp's memory instruction into as few 32-byte sectors as possible." },
    ],
    src: ["CUDA C++ Best Practices Guide (coalesced access: simple, misaligned and strided patterns)", "CUDA C++ Programming Guide (coalesced global memory access)", "Nsight Compute Profiling Guide (sectors and cache lines)"],
  });

  Deck.slide({
    id: "gpu-transpose",
    title: "A naive transpose reads rows and writes columns",
    body: `<p>Transpose an N × N matrix of floats with one thread per element: <code>out[x * N + y] = in[y * N + x]</code>, where x = blockIdx.x × 32 + threadIdx.x is the column. Within a warp, x takes 32 consecutive values and y is fixed.</p>
<p>The read is coalesced: 32 consecutive floats, 128 bytes, 4 sectors. The write is not. Consecutive lanes write addresses N floats apart, so the warp's store touches 32 different sectors, 32 × 32 = 1,024 bytes of sector traffic to store 128 bytes: 8 times the transactions of a coalesced store. Exchanging the roles so the read is strided only moves the problem.</p>
<p>For N = 8,192 the matrix is 268 MB, read by 2,097,152 warp loads of 4 sectors each and written by as many warp stores of 32 sectors each. The fix stages each tile through shared memory so that both the global read and the global write go along rows (next sheet).</p>`,
    fig: {
      type: "table", caption: "Sector traffic per warp in a transpose of FP32 data (N ≥ 8)", align: ["l", "l", "r", "r", "r"],
      cols: ["Access", "Byte address of lane k", "Sectors", "Bytes moved", "Used"],
      rows: [
        ["read <code>in[y·N + x]</code>", "base + 4k", "4", "128", "100%"],
        ["write <code>out[x·N + y]</code>", "base + 4kN", "32", "1,024", "12.5%"],
        ["write through a shared-memory tile", "base + 4k", "4", "128", "100%"],
      ],
      highlight: [1],
      note: "With N ≥ 8, consecutive lanes of the strided write are at least 32 bytes apart, so each lands in its own sector.",
    },
    src: ["CUDA C++ Best Practices Guide (strided accesses)"],
  });

  Deck.slide({
    id: "gpu-smem",
    title: "Shared memory: a 228 KB scratchpad the kernel fills itself",
    body: `<p>Shared memory is the part of the SM's 256 KB SRAM that a program addresses directly. A block declares it with <code>__shared__</code>; it exists as long as the block and is visible to all of the block's threads. On an H800 PCIe a load from it took about 29 cycles, against 479 for HBM, and each SM reads 128 bytes from it per clock.</p>
<p>The tiled transpose uses it as a staging buffer. A block of 32 × 8 threads reads a 32 × 32 tile of the input row by row (coalesced) into shared memory. After the barrier, each thread reads the tile down a column from shared memory, where address order matters only through bank conflicts, and writes the output row by row (coalesced again). Both global accesses are now 4 sectors per warp.</p>
<p>The tile is declared 32 × 33, one column wider than the data; the next sheet shows why.</p>`,
    note: "Best Practices Guide, C = AAᵀ on a V100: 12.8 GB/s reading Aᵀ by columns, 140.2 through shared tiles, 199.4 with the padded tile.",
    fig: {
      type: "code", lang: "cuda", caption: "Transpose through a shared-memory tile",
      code: `#define TILE 32
__global__ void transpose(const float* in, float* out, int N) {
  __shared__ float tile[TILE][TILE + 1];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += 8)
    tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += 8)
    out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
}
// launch: dim3 block(32, 8), grid(N / 32, N / 32)`,
      notes: {
        3: "32 × 33 floats = 4,224 bytes per block. The extra column removes bank conflicts (next sheet).",
        4: "x is the column: the 32 threads of a warp (same threadIdx.y) cover 32 consecutive columns.",
        7: "Four passes of 8 rows copy the 32 × 32 tile. Each warp reads 32 consecutive floats of one row: 4 sectors.",
        8: "Every element of the tile must be in shared memory before any thread reads it back transposed.",
        9: "Swap the block coordinates: this block writes the transposed tile at block position (blockIdx.y, blockIdx.x).",
        12: "A warp writes 32 consecutive floats of one output row (4 sectors) and reads one column of the tile from shared memory.",
        14: "(N / 32)² blocks of 256 threads, for N a multiple of 32.",
      },
      highlight: [3, 7, 12],
    },
    src: ["CUDA C++ Best Practices Guide (shared memory, C = AAᵀ example, Table 3)", "Luo et al., IPDPS 2024 (latencies)", "CUDA C++ Programming Guide (shared memory)"],
  });

  Deck.slide({
    id: "gpu-banks",
    title: "Bank conflicts: 32 banks, one 4-byte word each per clock",
    body: `<p>Shared memory is divided into 32 <b>banks</b>. Successive 4-byte words go to successive banks, so word w lives in bank w mod 32, and each bank delivers one word per clock. A warp's request finishes in one pass if its threads touch 32 different banks, or the same word (a broadcast). If they touch n different words in one bank, it takes n passes: an n-way <b>bank conflict</b>.</p>
<p>In <code>float tile[32][32]</code>, element [r][c] is word 32r + c, in bank c. A row read (thread l reads [r][l]) hits 32 banks. A column read (thread l reads [l][c]) hits bank c 32 times: a 32-way conflict at 1/32 of the bandwidth. The transpose reads columns.</p>
<p>Declaring <code>tile[32][33]</code> puts [l][c] at word 33l + c, in bank (l + c) mod 32: 32 different banks and no conflict, for the cost of 32 unused words. In general a stride of k words between threads gives a gcd(k, 32)-way conflict, so odd strides are conflict-free.</p>`,
    fig: { type: "gpu-banks", pattern: "col", pad: 0, k: 2, caption: "Which bank each lane's word falls in, and the words each bank must deliver" },
    terms: [{ term: "Bank conflict", def: "Several threads of a warp accessing different 4-byte words in the same shared-memory bank. An n-way conflict takes n passes." }],
    src: ["CUDA C++ Programming Guide (shared memory banks)", "CUDA C++ Best Practices Guide (shared memory and memory banks)"],
  });

  Deck.slide({
    id: "gpu-roofline",
    title: "The roofline: attainable FLOP/s = min(peak, intensity × bandwidth)",
    example: "tokens ids",
    body: String.raw`<p>Williams, Waterman and Patterson's <b>roofline model</b> bounds a kernel by two numbers of the machine. A kernel with arithmetic intensity I, FLOPs per byte of DRAM traffic, can run no faster than P below. On log-log axes this is a sloped line (memory-bound) meeting a flat one (compute-bound). For the H100 SXM in BF16 they meet at the <b>ridge point</b>, I = 989.4 / 3.35 = 295 FLOPs per byte.</p>
<p>The plate multiplies B token vectors by an 8,192 × 8,192 BF16 weight matrix: 2B × 8,192² FLOPs over 2 × (8,192² + 2B × 8,192) bytes, an intensity of 8,192B / (8,192 + 2B). One token gives 1.0 FLOP per byte, 0.34% of peak. The product crosses the ridge between B = 256 (241 FLOPs per byte) and B = 512 (455).</p>
<p>Elementwise operations stay below 3 FLOPs per byte at any size, two orders of magnitude left of the ridge.</p>`,
    math: String.raw`P = \min\left(P_{\text{peak}},\ I \times B_{\text{mem}}\right)`,
    fig: { type: "tinylm", widget: "gpu", caption: "Roofline of an H100 SXM: attainable TFLOP/s against FLOPs per byte" },
    terms: [{ term: "Roofline model", def: "Attainable FLOP/s = min(peak FLOP/s, arithmetic intensity × memory bandwidth). The two meet at the ridge point, peak ÷ bandwidth." }],
    src: ["Williams, Waterman, Patterson, Roofline: An Insightful Visual Performance Model for Multicore Architectures, CACM 52(4), 2009", "NVIDIA H100 whitepaper V1.04 and product page"],
  });

  Deck.slide({
    id: "gpu-ops",
    title: "Five operations on the H100 roofline",
    body: `<p>To place an operation you need only its FLOPs and the bytes it must move at least once, here in BF16 at 2 bytes per number.</p>
<ul><li><b>Elementwise add</b> of two tensors: 1 FLOP per 6 bytes, 0.17 FLOP/byte. On 2<sup>26</sup> elements, 403 MB: at least 120 µs.</li>
<li><b>LayerNorm</b>: about 8 FLOPs per element for 4 bytes read and written, 2 FLOP/byte. Softmax is similar, about 5 FLOPs per element.</li>
<li><b>Decode matrix–vector product</b>, one token through a 4,096 × 4,096 weight: 33.6 MFLOP over 33.6 MB, 1.0 FLOP/byte. It needs at least 10.0 µs; the arithmetic alone, 0.03 µs.</li>
<li><b>Linear layer</b>, 1,024 inputs to 4,096 outputs for 512 tokens: 315 FLOP/byte, just above the ridge.</li>
<li><b>4,096³ matrix multiply</b>: 137.4 GFLOP over 100.7 MB, 1,365 FLOP/byte. It needs at least 139 µs at peak; moving its bytes, at least 30 µs.</li></ul>
<p>Everything except large matrix multiplies sits two to three orders of magnitude below the ridge.</p>`,
    fig: {
      type: "chart", caption: "Operations on the H100 SXM BF16 roofline (log-log)",
      x: { type: "log", domain: [0.1, 10000], ticks: [0.1, 1, 10, 100, 1000, 10000], fmt: (v) => (v < 1 ? String(v) : int(v)), label: "FLOPs per byte" },
      y: { type: "log", domain: [0.1, 3000], ticks: [0.1, 1, 10, 100, 1000], fmt: (v) => String(v), label: "TFLOP/s" },
      series: [{ points: [[0.1, 0.335], [HW.bf16 / HW.bw, HW.bf16 / 1e12], [10000, HW.bf16 / 1e12]], cls: "ink" }],
      vlines: [{ x: HW.bf16 / HW.bw, label: "ridge 295" }],
      dots: [
        { x: 1 / 6, y: (HW.bw / 6) / 1e12, label: "add", cls: "s1", dx: 9, dy: 15, tip: "elementwise add: 0.167 FLOP/byte → 0.56 TFLOP/s" },
        { x: 1.0, y: HW.bw / 1e12, label: "decode GEMV", cls: "s2", dx: 9, dy: 14, tip: "4,096 × 4,096 matrix–vector: 1.0 FLOP/byte → 3.35 TFLOP/s" },
        { x: 2, y: (2 * HW.bw) / 1e12, label: "LayerNorm", cls: "s3", dx: -8, dy: -10, anchor: "end", tip: "LayerNorm: ≈ 2 FLOP/byte → 6.7 TFLOP/s" },
        { x: 315.08, y: HW.bf16 / 1e12, label: "linear, 512 tokens", cls: "s4", dx: -10, dy: -8, anchor: "end", tip: "1,024 → 4,096 linear layer, 512 tokens: 315 FLOP/byte → 989.4 TFLOP/s" },
        { x: 1365.3, y: HW.bf16 / 1e12, label: "4,096³ GEMM", cls: "s5", dx: 0, dy: 20, anchor: "middle", tip: "4,096³ matrix multiply: 1,365 FLOP/byte → 989.4 TFLOP/s" },
      ],
    },
    src: ["Williams et al., CACM 2009", "NVIDIA GPU Performance Background User's Guide (Table 1: linear layer 315 FLOPS/B, layer normalization below 10)"],
  });

  Deck.slide({
    id: "gpu-bound",
    title: "Memory-bound kernels need fewer bytes; compute-bound ones need tensor cores",
    body: String.raw`<p>The roofline gives a lower bound on time, and it tells you which resource to work on. Below 295 FLOPs per byte, time ≥ bytes ÷ 3.35 TB/s; the levers are fewer bytes, and enough bytes in flight to reach the bandwidth. Above it, time ≥ FLOPs ÷ peak; the levers raise the FLOP/s actually delivered.</p>
<p>LayerNorm on 2<sup>26</sup> elements needs at least 80 µs in BF16 and 160 µs in FP32; faster arithmetic changes neither. A 4,096³ matrix multiply needs at least 2.05 ms on the FP32 lanes and 139 µs on the BF16 tensor cores; moving fewer bytes changes neither.</p>
<p>The remaining sheets apply both kinds of lever: tiling and tensor cores for matrix multiplication, fusion and FlashAttention for the memory-bound rest of a transformer.</p>`,
    fig: {
      type: "compare", caption: "What helps on each side of the ridge",
      left: { title: "Memory-bound: below 295 FLOPs per byte", items: ["time ≥ bytes ÷ 3.35 TB/s", "fuse kernels so intermediates stay on chip", "BF16 instead of FP32: half the bytes; FP8 a quarter", "tile through shared memory: read each byte from HBM once", "batch requests: one weight read serves many tokens", "keep enough loads in flight to reach the bandwidth (6.4)"] },
      right: { title: "Compute-bound: above 295 FLOPs per byte", items: ["time ≥ FLOPs ÷ peak", "tensor cores: 989.4 instead of 66.9 TFLOP/s", "FP8 tensor cores: 1,978.9 TFLOP/s", "keep them fed: asynchronous copies, deep pipelines", "keep every SM busy: tile and wave quantization (6.6)"] },
    },
    check: {
      q: "A decode step multiplies a 4,096 × 4,096 BF16 weight by one token vector: at least 10.0 µs on an H100. What happens to that bound when 64 requests are batched, so 64 vectors share the weight?",
      options: ["It grows 64 times, to 641 µs", "It barely moves, to 10.3 µs: intensity rises to 62 FLOPs per byte, still memory-bound", "It becomes compute-bound at 295 FLOPs per byte", "It halves, because the weight is cached"],
      answer: 1,
      why: "The weight (33.6 MB) dominates the bytes: 2 × (4,096² + 2 × 64 × 4,096) = 34.6 MB, at least 10.3 µs at 3.35 TB/s. The FLOPs grow 64 times to 2.15 GFLOP, an intensity of 62, so the arithmetic (2.2 µs at peak) still fits under the memory time: 64 tokens for 1.03 times the bound of one.",
    },
    src: ["Williams et al., CACM 2009", "NVIDIA H100 whitepaper V1.04 (Table 3)"],
  });
  // ================================================================== 6.6
  Deck.chapter({ title: "Matrix multiplication on a GPU" });

  Deck.slide({
    id: "gpu-mm-naive",
    title: "The naive matrix multiply: one thread per output, 2N loads each",
    body: `<p>The direct kernel gives each thread one element of C = A·B for N × N matrices and lets it walk a row of A and a column of B. For N = 4,096, each thread does 4,096 multiply-adds (8,192 FLOPs) and 8,192 loads of 4 bytes, 32,768 bytes. Counted per thread, that is 2N FLOPs for 8N bytes: 0.25 FLOP per byte.</p>
<p>If every load went to HBM, the 4,096³ product would move 550 GB and need at least 164 ms. The hardware does better than that count: the 32 threads of a warp share one row of A, so <code>A[row * N + k]</code> is a single broadcast, their B reads are 32 consecutive floats, and the caches absorb part of the reuse. But nothing in the kernel keeps reused data close; each of the 4,096 threads that needs a given element of B fetches it for itself.</p>
<p>The arithmetic alone needs at least 2.05 ms on the FP32 lanes. Getting near that, and then onto the tensor cores, means loading each element from HBM once and reusing it many times on chip.</p>`,
    fig: {
      type: "code", lang: "cuda", caption: "One thread per element of C",
      code: `__global__ void mm(const float* A, const float* B, float* C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float acc = 0.f;
    for (int k = 0; k < N; ++k)
      acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}
// launch: dim3 block(32, 32), grid(N / 32, N / 32)`,
      notes: {
        2: "Each thread owns C[row][col]. col follows threadIdx.x, so a warp covers 32 consecutive columns of one row.",
        6: "N iterations: N multiply-adds and 2N loads per thread.",
        7: "A[row][k] is the same address for the whole warp (one broadcast); B[k][col] is 32 consecutive floats across the warp (coalesced).",
        8: "One 4-byte store per thread.",
        11: "1,024 threads per block; (4,096 / 32)² = 16,384 blocks for N = 4,096.",
      },
      highlight: [7],
    },
    src: ["CUDA C++ Best Practices Guide (shared memory in matrix multiplication)"],
  });

  Deck.slide({
    id: "gpu-mm-tiled",
    title: "Tiling: each block stages T × T tiles of A and B in shared memory",
    body: `<p>Split C into T × T tiles and give each to a block of T × T threads. The block walks along k in steps of T. In each step its threads copy one tile of A and one of B into shared memory, one element each, synchronize, and do T multiply-adds each from shared memory. Every element loaded from HBM is now used T times.</p>
<p>Per output, global loads fall from 2N to 2N / T, so the intensity rises from 0.25 to T / 4 FLOPs per byte: 8 for T = 32. The two tiles take 2 × 32 × 32 × 4 = 8,192 bytes of shared memory. The inner loop has no bank conflicts: <code>As[ty][k]</code> is one word for the whole warp (a broadcast) and <code>Bs[k][tx]</code> is 32 consecutive words.</p>
<p>The next limit is shared memory itself. With one 4-byte load per operand, each warp FMA instruction (32 FMAs) needs two shared-memory requests, <code>As[ty][k]</code> and <code>Bs[k][tx]</code>. An SM's shared memory serves one request of 128 bytes per clock, so it sustains at most 16 FMAs per clock, 12.5% of its 128 FP32 lanes.</p>`,
    fig: {
      type: "code", lang: "cuda", caption: "Tiled matrix multiply through shared memory",
      code: `#define T 32
__global__ void mm_tiled(const float* A, const float* B,
                         float* C, int N) {
  __shared__ float As[T][T], Bs[T][T];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int t = 0; t < N / T; ++t) {          // N a multiple of T
    As[ty][tx] = A[row * N + t * T + tx];
    Bs[ty][tx] = B[(t * T + ty) * N + col];
    __syncthreads();
    for (int k = 0; k < T; ++k)
      acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * N + col] = acc;
}`,
      notes: {
        4: "Two 32 × 32 float tiles: 8,192 bytes of shared memory per block of 1,024 threads.",
        8: "N / T steps along k, each covering T values of k.",
        9: "Each thread loads one element of the A tile; a warp (fixed ty) reads 32 consecutive floats of a row of A: coalesced.",
        10: "And one element of the B tile, also 32 consecutive floats per warp.",
        11: "Both tiles must be complete before any thread uses them.",
        13: "T multiply-adds from shared memory. As[ty][k] is one word for the whole warp, Bs[k][tx] is 32 consecutive words: no bank conflicts.",
        14: "No thread may overwrite the tiles with the next step's data while others are still reading them.",
      },
      highlight: [9, 10, 13],
    },
    src: ["CUDA C++ Best Practices Guide (shared memory in matrix multiplication)", "CUDA C++ Programming Guide (shared memory banks)"],
  });

  Deck.slide({
    id: "gpu-tiles",
    title: "Stepping through a tiled multiply: global loads fall by the tile width",
    body: `<p>The plate multiplies two N × N matrices tile by tile. For each T × T tile of C, the block runs N / T phases; in phase t it loads tile (row block, t) of A and tile (t, column block) of B, 2T² elements, and every output in the tile accumulates T products.</p>
<p>Count the loads: (N / T)² output tiles, each with N / T phases of 2T² loads, is 2N³ / T in total. The naive kernel is the case T = 1, with 2N³. For N = 16 that is 8,192 loads untiled and 2,048 with T = 4, for the same 2N³ = 8,192 FLOPs, so the FP32 intensity goes from 0.25 to 1 FLOP per byte.</p>
<p>Real tiles are larger. With T = 32, a 4,096³ product issues 17.2 GB of global loads instead of 550 GB: at least 5.1 ms if all of them reached HBM, and the 50 MB L2 absorbs part of them. Shared memory binds first: at 16 FMAs per clock per SM (previous sheet) the product needs at least 16.4 ms, 8 times the 2.05 ms of FP32 arithmetic. Register tiling, next, removes that limit.</p>`,
    fig: { type: "gpu-tiles", N: 8, T: 2, caption: "C = A·B tile by tile: the tiles loaded in the current phase (red), the C tile being accumulated (outlined), finished tiles (blue)" },
    terms: [{ term: "Tiling", def: "Splitting a computation into blocks that fit in fast on-chip memory, so each element loaded from DRAM is reused many times. For matrix multiply, T × T tiles cut global loads by a factor of T." }],
    src: ["CUDA C++ Best Practices Guide (shared memory in matrix multiplication)"],
  });

  Deck.slide({
    id: "gpu-regtile",
    title: "Register tiling: each thread computes a block of outputs",
    body: `<p>To get past the shared-memory limit, give each thread a TM × TN block of outputs kept in registers. At each k it loads TM values of A and TN values of B from shared memory into registers and does TM × TN FMAs with them: TM + TN words for TM·TN FMAs.</p>
<p>With 8 × 8 outputs per thread that is 16 words per 64 FMAs, 0.25 per FMA, exactly what an SM's 32 words per clock can feed to 128 FP32 lanes (assuming conflict-free accesses). With 1 × 1 it is 2 words per FMA, the tiled kernel's 12.5% ceiling. The price is registers: 64 accumulators and 16 operands, so at least 80 registers per thread, and lower occupancy.</p>
<p>The block tile sets the HBM side. A 128 × 128 block tile loads (128 + 128) × BK floats per step of BK and does 128 × 128 × BK FMAs: 32 FLOPs per byte in FP32, above the FP32 lanes' ridge of 20.</p>`,
    fig: {
      type: "calc", caption: "A register-tiled FP32 matrix multiply: shared-memory traffic, registers, HBM intensity",
      inputs: [
        { id: "t", label: "outputs per thread, TM × TN", options: [1, 2, 4, 8].map((v) => ({ value: v, label: `${v} × ${v}` })), value: 8 },
        { id: "bm", label: "block tile, BM × BN", options: [32, 64, 128].map((v) => ({ value: v, label: `${v} × ${v}` })), value: 128 },
      ],
      outputs: [
        { label: "shared-memory words per FMA", f: (v) => (2 * v.t) / (v.t * v.t), fmt: "f3" },
        { label: "FP32 peak reachable (shared-memory limit)", f: (v) => Math.min(1, 0.25 / ((2 * v.t) / (v.t * v.t))), fmt: (x) => pct(x, 1) },
        { label: "registers per thread, at least", f: (v) => v.t * v.t + 2 * v.t, fmt: "int" },
        { label: "threads per block", f: (v) => (v.bm / v.t) ** 2, fmt: (x) => (x > 1024 ? `${int(x)} (> 1,024: too many)` : int(x)) },
        { label: "FLOPs per HBM byte", f: (v) => (v.bm * v.bm) / (2 * (v.bm + v.bm)), fmt: "f0" },
      ],
      note: (v) => `Words per FMA = (TM + TN) / (TM × TN) = ${2 * v.t} / ${v.t * v.t}; the SM supplies 32 words per clock for 128 FMAs, 0.25 per FMA. HBM intensity = 2·BM·BN·BK / (4·(BM + BN)·BK) = ${int(v.bm * v.bm)} / ${int(2 * 2 * v.bm)} FLOPs per byte.`,
    },
    terms: [{ term: "Register tiling", def: "Each thread computes a small block of outputs in registers, so every value it loads from shared memory feeds several FMAs." }],
    src: ["CUTLASS documentation, Efficient GEMM (threadblock, warp and thread tiles; accumulators in registers)", "CUDA C++ Programming Guide (shared memory bandwidth)"],
  });

  Deck.slide({
    id: "gpu-tensorcore",
    title: "Tensor cores: one instruction multiplies whole tiles",
    body: `<p>A tensor-core instruction is issued by a warp, or on Hopper by a <b>warpgroup</b> of 4 warps, and multiplies small matrices held in registers or shared memory. Since Ampere, <code>mma.sync.m16n8k16</code> has a warp compute D = A·B + C with A of 16 × 16 and B of 16 × 8 in BF16 and C, D in FP32: 4,096 FLOPs in one instruction, each of the 32 threads holding 8 elements of A, 4 of B and 4 of C in registers.</p>
<p>Hopper adds <code>wgmma.mma_async</code>: a warpgroup computes a 64 × N × 16 product, N up to 256, with B (and optionally A) read straight from shared memory. The m64n256k16 shape is 524,288 FLOPs, 128 clocks of an SM's 4,096 BF16 FLOPs per clock, and Luo et al. measured it at 128 cycles. The instruction is asynchronous: the warpgroup issues it and continues until it waits for the result.</p>
<p>The TMA feeds it: one thread starts the copy of a whole tile (up to 5-D) from global to shared memory, and a barrier that counts the arriving bytes signals completion.</p>`,
    fig: {
      type: "table", caption: "Tensor-core instructions in PTX (dense)", dense: true, align: ["l", "l", "l", "l", "r"],
      cols: ["Instruction", "Issued by", "M × N × K", "Inputs → accumulator", "FLOPs"],
      rows: [
        ["<code>mma.sync.m16n8k8</code>", "warp", "<span class=\"gpu-nw\">16 × 8 × 8</span>", "BF16/FP16 → FP32", "2,048"],
        ["<code>mma.sync.m16n8k16</code>", "warp", "<span class=\"gpu-nw\">16 × 8 × 16</span>", "BF16/FP16 → FP32", "4,096"],
        ["<code>wgmma</code> m64nNk8", "warpgroup", "<span class=\"gpu-nw\">64 × N × 8</span>", "TF32 → FP32", "up to 262,144"],
        ["<code>wgmma</code> m64nNk16", "warpgroup", "<span class=\"gpu-nw\">64 × N × 16</span>", "BF16/FP16 → FP32", "up to 524,288"],
        ["<code>wgmma</code> m64nNk32", "warpgroup", "<span class=\"gpu-nw\">64 × N × 32</span>", "FP8 E4M3/E5M2 → FP32", "<span class=\"gpu-nw\">up to 1,048,576</span>"],
      ],
      highlight: [3],
      note: "A warp is 32 threads, a warpgroup 128. N runs from 8 to 256 in steps of 8. mma.sync with BF16 needs sm_80 (A100) or later; wgmma needs sm_90a (H100). FP16 and FP8 inputs can also accumulate in FP16.",
    },
    terms: [{ term: "Warpgroup", def: "Four contiguous warps (128 threads) that issue a Hopper wgmma instruction together." }],
    src: ["NVIDIA PTX ISA 9.4 (warp-level matrix shapes; wgmma.mma_async)", "NVIDIA H100 whitepaper V1.04 (TMA, asynchronous transaction barrier)", "Luo et al., IPDPS 2024 (Table VIII)"],
  });

  Deck.slide({
    id: "gpu-waves",
    title: "Tile and wave quantization: shapes that leave SMs idle",
    body: `<p>A GEMM kernel splits C into output tiles, one per thread block. Two roundings waste work. <b>Tile quantization</b>: if M or N is not a multiple of the tile, the edge tiles are partly empty but cost as much as full ones. <b>Wave quantization</b>: with one tile per SM at a time, tiles run in waves of 132, and a last wave with few tiles leaves most SMs idle.</p>
<p>With 128 × 128 tiles, M = 1,408 and N = 1,536 make exactly 132 tiles: one full wave. N = 1,544, 0.5% more work, adds a 13th column of tiles: 143 tiles in two waves, the second keeping 11 of 132 SMs busy. The product takes twice as long, and the useful share of SM time falls to 50.3%.</p>
<p>NVIDIA measured the same on an A100 (108 SMs, 256 × 128 tiles, M = 2,304): N = 1,536 gives 108 tiles, N = 1,544 gives 117, and throughput roughly halves.</p>
<p>Libraries soften both effects by choosing a tile shape per problem and by splitting the K dimension across blocks (split-K; Stream-K divides the multiply-add iterations evenly over the SMs). Optimized GEMM kernels reach 80 to 90% of an H100's peak.</p>`,
    fig: {
      type: "calc", caption: "Tiles, waves and useful SM time on 132 SMs, one tile per SM at a time",
      inputs: [
        { id: "M", label: "M (rows of C)", min: 128, max: 8192, step: 64, value: 1408, fmt: "int" },
        { id: "N", label: "N (columns of C)", min: 128, max: 8192, step: 8, value: 1544, fmt: "int" },
        { id: "tile", label: "tile, rows × columns", options: [{ value: "128x128", label: "128 × 128" }, { value: "128x256", label: "128 × 256" }, { value: "256x128", label: "256 × 128" }], value: "128x128" },
      ],
      outputs: [
        { label: "tiles", f: (v) => { const [a, b] = v.tile.split("x").map(Number); return Math.ceil(v.M / a) * Math.ceil(v.N / b); }, fmt: "int" },
        { label: "waves of 132", f: (v) => { const [a, b] = v.tile.split("x").map(Number); return Math.ceil((Math.ceil(v.M / a) * Math.ceil(v.N / b)) / HW.sms); }, fmt: "int" },
        { label: "SMs busy in the last wave", f: (v) => { const [a, b] = v.tile.split("x").map(Number); const t = Math.ceil(v.M / a) * Math.ceil(v.N / b); return t - (Math.ceil(t / HW.sms) - 1) * HW.sms; }, fmt: "int" },
        { label: "useful share of SM time", f: (v) => { const [a, b] = v.tile.split("x").map(Number); const t = Math.ceil(v.M / a) * Math.ceil(v.N / b); return (v.M * v.N) / (Math.ceil(t / HW.sms) * HW.sms * a * b); }, fmt: (x) => pct(x, 1) },
      ],
      note: (v) => { const [a, b] = v.tile.split("x").map(Number); return `⌈${int(v.M)} / ${a}⌉ × ⌈${int(v.N)} / ${b}⌉ = ${Math.ceil(v.M / a)} × ${Math.ceil(v.N / b)} tiles. Useful share = M × N / (waves × 132 × ${a} × ${b}).`; },
    },
    terms: [
      { term: "Tile quantization", def: "Work wasted when a matrix dimension is not a multiple of the kernel's tile size: edge tiles are partly empty." },
      { term: "Wave quantization", def: "Work wasted when the number of tiles is not a multiple of the number of tiles the GPU runs at once: the last wave leaves SMs idle." },
    ],
    src: ["NVIDIA Matrix Multiplication Background User's Guide (tile and wave quantization, A100-SXM4-80GB)", "CUTLASS documentation, Efficient GEMM (split-K)", "Osama, Merrill, Cecka, Garland, Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU, 2023", "Shah et al., FlashAttention-3, 2024 (optimized GEMMs: 80–90% of H100 peak)"],
  });

  Deck.slide({
    id: "gpu-cutlass",
    title: "A production GEMM: CUTLASS's hierarchy of tiles and pipelines",
    layout: "wide",
    body: String.raw`<p>NVIDIA's open-source CUTLASS library builds a matrix multiply as a hierarchy of tiles. A thread block owns an output tile of C, for example 128 × 128, and walks along K in steps of 64; within it, warps compute warp tiles with tensor-core instructions. With Ampere's <code>mma.sync</code> every operand sits in registers. Hopper's <code>wgmma</code> reads B, and often A, straight from shared memory, and only the accumulators live in registers.</p>
<p>The Hopper kernels split warpgroups by role. A producer warpgroup issues TMA copies of the next A and B tiles into a ring of shared-memory stages, and consumer warpgroups run <code>wgmma</code> on stages already loaded, so copies and math overlap. The kernels are persistent: only as many blocks as the SMs hold are launched, and each computes output tile after output tile. In the cooperative design two consumers split one tile along M; in the ping-pong design they take different tiles, so one's epilogue overlaps the other's math.</p>
<p>One 128 × 64 tile of A and one 64 × 128 tile of B in BF16 are 32 KB per stage: four stages fill 128 KB of the 228 KB of shared memory. The 128 × 128 FP32 accumulators take 64 KB of registers per block.</p>`,
    fig: {
      type: "diagram", caption: "A cooperative Hopper GEMM in CUTLASS: one 128 × 128 output tile per block", w: 24, h: 7.6, unit: 40,
      nodes: [
        { id: "hbm", x: 0.3, y: 0.4, w: 4.4, h: 2.6, label: "HBM", sub: "A, B and C", kind: "dark" },
        { id: "prod", x: 6.2, y: 0.4, w: 5.8, h: 1.2, label: "Producer warpgroup", sub: "issues TMA copies" },
        { id: "tma", x: 6.2, y: 1.85, w: 5.8, h: 1.25, label: "TMA", sub: "copies A and B tiles", kind: "blue" },
        { id: "smem", x: 13.4, y: 0.4, w: 10.3, h: 2.6, label: "Shared memory: 4 stages", sub: "each A 128 × 64 + B 64 × 128, BF16, 32 KB", kind: "hi" },
        { id: "c1", x: 13.4, y: 3.7, w: 4.95, h: 1.5, label: "Consumer warpgroup 1", sub: "wgmma, rows 0–63" },
        { id: "c2", x: 18.75, y: 3.7, w: 4.95, h: 1.5, label: "Consumer warpgroup 2", sub: "wgmma, rows 64–127" },
        { id: "acc", x: 13.4, y: 5.9, w: 10.3, h: 1.3, label: "Accumulators in registers", sub: "128 × 128 FP32 = 64 KB" },
        { id: "epi", x: 6.2, y: 5.9, w: 5.8, h: 1.3, label: "Epilogue", sub: "scale, convert, store C" },
      ],
      edges: [
        { from: "hbm", to: "tma" }, { from: "prod", to: "tma", dashed: true }, { from: "tma", to: "smem" },
        { from: "smem", to: "c1" }, { from: "smem", to: "c2" }, { from: "c1", to: "acc" }, { from: "c2", to: "acc" },
        { from: "acc", to: "epi" }, { from: "epi", to: "hbm", label: "C" },
      ],
    },
    terms: [{ term: "Epilogue", def: "The last part of a GEMM kernel: it converts the accumulators, applies scaling, bias or activation, and stores the output tile." }],
    src: ["CUTLASS documentation (efficient_gemm.md: persistent, warp-specialized cooperative and ping-pong kernels; profiler tile 128 × 128 × 64)", "NVIDIA PTX ISA 9.4 (mma.sync and wgmma operands)"],
  });

  Deck.slide({
    id: "gpu-toosmall",
    title: "Our model's matrix multiplies are far too small for an H100",
    example: "tokens ids",
    body: `<p>One forward pass of the running example's model over 8 tokens multiplies matrices no larger than 8 × 64 by 64 × 16, or 8 × 16 by 16 × 113 for the output layer. Counting every product, the plate's table, gives 135,424 FLOPs.</p>
<p>At 989 TFLOP/s that is 0.14 ns of tensor-core time. The largest output, 8 × 113 logits, fills 5.5% of a single 128 × 128 tile on one of the 132 SMs. A kernel launch alone costs microseconds: NVIDIA measured a 2.9 µs kernel at 3.8 µs per launch when launched back to back on a V100, and 9.6 µs with a synchronization after each launch.</p>
<p>On a GPU, this model's forward pass would be almost entirely launch overhead. Matrix multiplies approach the tensor cores' peak only when M, N and K are in the thousands, which is why serving systems batch requests (Part 8) and training uses large batches (Part 7).</p>`,
    fig: {
      type: "table", caption: "Matrix products in one forward pass of our model (8 tokens, d = 16, 2 heads of 8, MLP 64)", align: ["l", "l", "r"],
      cols: ["Product", "Shapes", "FLOPs"],
      rows: [
        ["Q, K, V projections", "3 × (8 × 16)·(16 × 16)", "12,288"],
        ["attention scores, 2 heads", "2 × (8 × 8)·(8 × 8)", "2,048"],
        ["attention mixing, 2 heads", "2 × (8 × 8)·(8 × 8)", "2,048"],
        ["output projection", "(8 × 16)·(16 × 16)", "4,096"],
        ["MLP", "(8 × 16)·(16 × 64) and (8 × 64)·(64 × 16)", "32,768"],
        ["one layer", "", "53,248"],
        ["two layers", "", "106,496"],
        ["unembedding", "(8 × 16)·(16 × 113)", "28,928"],
        ["forward pass", "", "<b>135,424</b>"],
      ],
      highlight: [8],
      note: "Attention counted over all 8 × 8 query–key pairs, as the kernels compute them before masking.",
    },
    check: {
      q: "A GEMM with 128 × 128 tiles has 133 tiles. On a 132-SM H100 running one tile per SM at a time, about what share of the SMs' time does useful work?",
      options: ["About 99%: only one tile is extra", "About 50%: the second wave runs 1 tile while 131 SMs wait", "About 75%", "100%: each SM runs two tiles in parallel"],
      answer: 1,
      why: "⌈133 / 132⌉ = 2 waves, so the product takes 2 × 132 = 264 tile-slots of SM time for 133 tiles of work: 133 / 264 = 50.4%.",
    },
    src: ["NVIDIA Technical Blog, Getting Started with CUDA Graphs (Gray, 2019)", "Running-example model configuration (Part 0)"],
  });

  // ================================================================== 6.7
  Deck.chapter({ title: "Kernels for transformers" });

  Deck.slide({
    id: "gpu-fusion",
    title: "Kernel fusion: one read and one write instead of three of each",
    body: `<p>An unfused chain of elementwise operations runs one kernel per operation, and each kernel reads its input from HBM and writes its output back. Bias add, GELU and <b>dropout</b> (in training, each value is set to zero with probability p and the rest are multiplied by 1/(1 − p)) on an activation tensor of n BF16 values move 3 × (2n + 2n) = 12n bytes. A fused kernel reads x once, computes all three in registers, and writes once: 4n bytes, a third. A 1-byte dropout mask saved for the backward pass costs the same in both.</p>
<p>For a GPT-style MLP with 16,384 hidden units and 8,192 tokens, n is 134 million: 1.61 GB and 481 µs unfused, 537 MB and 160 µs fused. Every kernel in the chain is memory-bound, so time follows bytes, and a chain of k operations shrinks by a factor of k. The fused kernel also saves two launches.</p>
<p>Compilers fuse elementwise chains automatically: <code>torch.compile</code> generates fused Triton kernels. Attention needs the hand-built fusion of the next sheets.</p>`,
    fig: {
      type: "calc", caption: "An elementwise chain on a BF16 activation tensor, unfused and fused",
      inputs: [
        { id: "T", label: "tokens", options: optList([2048, 8192, 32768], int), value: 8192 },
        { id: "w", label: "hidden units", options: optList([4096, 14336, 16384], int), value: 16384 },
        { id: "k", label: "elementwise operations in the chain", min: 1, max: 6, step: 1, value: 3, fmt: "int" },
      ],
      outputs: [
        { label: "bytes, unfused", f: (v) => 4 * v.k * v.T * v.w, fmt: "bytes" },
        { label: "time, unfused", f: (v) => (4 * v.k * v.T * v.w) / HW.bw, fmt: "time" },
        { label: "bytes, fused", f: (v) => 4 * v.T * v.w, fmt: "bytes" },
        { label: "time, fused", f: (v) => (4 * v.T * v.w) / HW.bw, fmt: "time" },
      ],
      note: (v) => `n = ${int(v.T)} × ${int(v.w)} = ${int(v.T * v.w)} values. Unfused: ${v.k} kernels × (2n read + 2n written) bytes. Fused: 2n read + 2n written.`,
    },
    terms: [{ term: "Dropout", def: "A training-time operation that sets each value to zero with probability p and multiplies the others by 1/(1 − p), so the expected value is unchanged." }, { term: "Kernel fusion", def: "Combining several operations into one kernel so intermediate results stay in registers or shared memory instead of going through HBM." }],
    src: ["PyTorch documentation, torch.compile and TorchInductor (Triton code generation)"],
  });

  Deck.slide({
    id: "gpu-online",
    title: "Online softmax: the maximum and the sum in one pass",
    body: String.raw`<p>Softmax divides \(e^{x_i}\) by \(\sum_j e^{x_j}\). Computed directly it overflows: \(e^x\) exceeds the largest FP32 number above x = 88.7, and the largest FP16 number above x = 11.1. Subtracting the row maximum m first gives the same result with every exponent at most 0. This safe softmax reads the row three times (find m, sum \(e^{x_j - m}\), normalize); Milakov and Gimelshein count 4 memory accesses per element.</p>
<p>Online softmax keeps a running maximum and a running sum, and when a new element raises the maximum it rescales the old sum by \(e^{m_{j-1} - m_j}\). One pass computes both, so normalizing takes a second: 3 accesses per element. The same rescaling lets FlashAttention fold in keys one tile at a time without seeing the whole row. Step through it on the plate, then shift the inputs by 100: the direct formula overflows to ∞ / ∞ in FP32, while the online one does not change.</p>`,
    math: [String.raw`m_j = \max(m_{j-1},\, x_j)`, String.raw`d_j = d_{j-1}\, e^{\,m_{j-1} - m_j} + e^{\,x_j - m_j}`],
    fig: { type: "gpu-online", caption: "Online softmax over eight values, one element at a time" },
    terms: [{ term: "Online softmax", def: "Computing the softmax normalizer in one pass with a running maximum and a running sum that is rescaled whenever the maximum grows." }],
    src: ["Milakov and Gimelshein, Online normalizer calculation for softmax, 2018"],
  });

  Deck.slide({
    id: "gpu-flash",
    title: "FlashAttention: exact attention without the T × T matrix in HBM",
    example: "tokens ids focus:4",
    body: String.raw`<p>Standard attention for one head writes the T × T score matrix S = QKᵀ to HBM, reads it for the softmax, writes the probabilities P and reads them again for PV: Θ(Td + T²) HBM accesses, with d the head dimension. At T = 8,192 one head's S has 67 million entries, 134 MB in BF16.</p>
<p><b>FlashAttention</b> (Dao et al., 2022) never writes S or P. As published, its outer loop loads a block of K and V into on-chip memory; the inner loop brings in each block of Q with its running output O, sum ℓ and maximum m, computes those scores, rescales with the online softmax and writes O, ℓ, m back. K and V cross HBM once, Q and O once per K, V block. On-chip memory of size M holds blocks of Θ(M/d) rows, so there are Θ(Td/M) passes of Θ(Td) values: Θ(T²d²/M) accesses, below T² when d² is below M. At T = 8,192 with 128-row blocks, that is 64 passes.</p>
<p>The result is exact. For GPT-2 medium's attention, forward and backward (T = 1,024, d = 64), on an A100 it moved 4.4 GB instead of 40.3 GB and took 7.3 ms instead of 41.7 ms, for 75.2 instead of 66.6 GFLOPs.</p>`,
    fig: { type: "tinylm", widget: "flash", caption: "FlashAttention's online softmax over our model's real attention scores, keys in tiles" },
    terms: [{ term: "FlashAttention", def: "An exact attention kernel that tiles Q, K and V through on-chip memory and uses the online softmax, so the N × N score matrix is never written to HBM." }],
    src: ["Dao, Fu, Ermon, Rudra, Ré, FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, 2022 (Algorithm 1, Theorem 2, Figure 2)"],
  });

  Deck.slide({
    id: "gpu-flash23",
    title: "FlashAttention-2 and -3: better work partitioning, then Hopper's asynchrony",
    body: String.raw`<p>FlashAttention reached 25 to 40% of an A100's peak. <b>FlashAttention-2</b> swapped its loops: each thread block owns one block of queries, keeps its O, ℓ and m on chip and streams K and V past them, so O is written once and the query blocks of a head run in parallel. At T = 8,192, d = 128 and 128-row blocks, each of a head's 64 query blocks reads all of K and V, 4.2 MB in BF16: 268 MB of loads, which reach HBM only if the 50 MB L2 does not keep K and V between blocks. It also cut non-matmul FLOPs, which cost 16 times a tensor-core FLOP on an A100 (19.5 against 312 TFLOP/s). It is about 2 times faster, at 50 to 73% of peak.</p>
<p><b>FlashAttention-3</b> targets the H100, where FlashAttention-2 reaches 35%. Producer warps load tiles with the TMA while consumer warpgroups run <code>wgmma</code>, and two warpgroups alternate so that one's softmax overlaps the other's matrix products. With d = 128 a score costs 512 matmul FLOPs and one exponential, and exponentials run at 3.9 against 989 TFLOP/s: half the matmul time. It reaches 740 TFLOP/s in FP16 (75%).</p>`,
    fig: {
      type: "table", caption: "Three generations of FlashAttention", dense: true,
      cols: ["Version", "GPU", "Main changes", "Reported"],
      rows: [
        ["FlashAttention (2022)", "A100", "outer loop over K, V blocks; online softmax; recomputation in the backward pass", "GPT-2 medium attention 7.3 ms vs 41.7 ms; 25–40% of peak"],
        ["FlashAttention-2 (2023)", "A100", "outer loop over query blocks, parallel over the sequence; fewer non-matmul FLOPs; Q split across warps", "≈ 2× FlashAttention; 50–73% of peak; 225 TFLOP/s per GPU training GPT"],
        ["FlashAttention-3 (2024)", "H100 SXM5", "TMA and wgmma warp specialization; softmax and GEMM overlap between warpgroups; FP8 block quantization and incoherent processing", "1.5–2.0× FlashAttention-2 forward; 740 TFLOP/s FP16 (75%); ≈ 1.2 PFLOP/s FP8"],
      ],
      note: "Incoherent processing multiplies Q and K by the same random orthogonal matrix (random ±1 signs times a Hadamard matrix) before FP8 quantization: outliers spread across dimensions, and QKᵀ is unchanged.",
    },
    terms: [{ term: "Incoherent processing", def: "Multiplying Q and K by the same random orthogonal matrix before quantizing them to FP8, which spreads outlier values across dimensions and leaves QKᵀ unchanged (FlashAttention-3)." }],
    src: ["Dao, FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, 2023 (§3.1, §3.2: loop order, sequence parallelism)", "Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision, 2024"],
  });

  Deck.slide({
    id: "gpu-triton",
    title: "Triton: a tiled kernel written in Python",
    body: `<p><b>Triton</b> is a language and compiler for GPU kernels whose programs operate on whole blocks of data: the compiler decides how elements map to threads and warps, coalesces the loads, places data in registers or shared memory, and generates the PTX. Tillet, Kung and Cox introduced it in 2019 as Triton-C, a C-like language with an LLVM-based intermediate representation; the Python form with <code>@triton.jit</code> shown here arrived with Triton 1.0 in July 2021.</p>
<p>The listing is a fused softmax. One program per row loads the row into registers, subtracts its maximum, exponentiates, normalizes and stores, so each element crosses HBM once in each direction. The official tutorial counts what an unfused PyTorch softmax moves on an M × N input: reads of 5MN + 2M elements and writes of 3MN + 2M, against MN of each fused, and expects about a 4× speedup.</p>
<p>The tutorial's current kernel adds a persistent loop over rows; this is the one-row-per-program form.</p>`,
    fig: {
      type: "code", lang: "python", caption: "Fused softmax in Triton, one row per program",
      code: `import triton
import triton.language as tl

@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)                  # one program per row
    offs = tl.arange(0, BLOCK)              # BLOCK column indices
    mask = offs < n_cols
    ptrs = in_ptr + row * in_stride + offs
    x = tl.load(ptrs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)               # subtract the row maximum
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + offs, y, mask=mask)

# one program per row; BLOCK = smallest power of 2 >= n_cols
BLOCK = triton.next_power_of_2(n_cols)
softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols,
                          BLOCK=BLOCK, num_warps=8)`,
      notes: {
        4: "<code>@triton.jit</code> compiles the function for the GPU the first time it is called with a given set of constant arguments.",
        7: "A program, not a thread: <code>program_id(0)</code> is this instance's row.",
        8: "A vector of BLOCK offsets. The compiler spreads it over the program's warps (8 here).",
        11: "Masked load: columns past n_cols read −∞, which adds e^(−∞) = 0 to the sum.",
        12: "Block-wide reductions such as tl.max and tl.sum are written as one call.",
        15: "One read and one write per element: the fused kernel's whole HBM traffic.",
        19: "The launch grid is a tuple of program counts, here one per row; num_warps sets the warps per program.",
      },
    },
    terms: [{ term: "Triton", def: "A language and compiler for GPU kernels whose programs operate on blocks of data; the compiler handles threads, coalescing and shared memory. Introduced as the C-like Triton-C (2019); written in Python since Triton 1.0 (2021)." }],
    src: ["Tillet, Kung, Cox, Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations, MAPL 2019 (Triton-C, Triton-IR, Triton-JIT)", "Triton 1.0 release, PyPI release history (28 July 2021)", "Triton documentation (triton-lang.org): Python API and tutorial Fused Softmax"],
  });

  Deck.slide({
    id: "gpu-graphs",
    title: "CUDA graphs remove the launch overhead between small kernels",
    body: `<p>Every kernel launch passes through the CUDA runtime and driver before the GPU starts it. For long kernels this is noise; for short ones it dominates. In NVIDIA's measurement on a V100, a kernel that runs for 2.9 µs took 9.6 µs per launch when the host synchronized after each one, 3.8 µs when launches were queued back to back in a stream, and 3.4 µs when the sequence was captured once as a <b>CUDA graph</b> and replayed with one launch. Creating the graph took about 400 µs, once.</p>
<p>A graph records kernels and copies with their dependencies, so replaying it skips the per-launch work on the CPU. In PyTorch, <code>torch.compile</code> fuses elementwise operations into fewer Triton kernels, and its "reduce-overhead" mode replays the result as CUDA graphs. Both matter most for decode, where each step runs several kernels per layer on only a few tokens.</p>`,
    fig: {
      type: "timeline", caption: "Five launches of a 2.9 µs kernel on a V100, three ways (NVIDIA's measured per-kernel times)", unit: "µs", total: 50,
      rows: ["sync each launch", "stream", "CUDA graph"],
      items: [[0, 9.6], [1, 3.8], [2, 3.4]].flatMap(([row, per]) => Array.from({ length: 5 }, (_, k) => [
        { row, start: k * per, len: 2.9, label: "", kind: "s1", tip: `kernel ${k + 1}: 2.9 µs of work` },
        { row, start: k * per + 2.9, len: per - 2.9, label: "", kind: "idle", tip: `${fx(per - 2.9, 1)} µs of launch overhead per kernel` },
      ]).flat()),
      note: "Five kernels take 48.0, 19.0 and 17.0 µs of wall time, of which 14.5 µs is work.",
    },
    terms: [{ term: "CUDA graph", def: "A recorded sequence of GPU operations with their dependencies that can be launched as a unit, removing per-kernel launch overhead." }],
    src: ["NVIDIA Technical Blog, Getting Started with CUDA Graphs (Gray, 2019)", "PyTorch documentation, torch.compile (mode reduce-overhead)"],
  });

  Deck.slide({
    id: "gpu-profile",
    title: "Profiling: Nsight Systems finds the gaps, Nsight Compute explains a kernel",
    body: `<p>Two NVIDIA tools answer different questions. <b>Nsight Systems</b> records a timeline of the whole program: CPU threads and CUDA API calls, every kernel and copy per stream, NCCL communication kernels, and NVTX ranges that the program marks itself. It shows where the GPU sits idle: gaps between kernels (launch overhead or CPU work), copies that do not overlap computation, communication waiting on computation. <b>Nsight Compute</b> replays one kernel and reports why it runs at its speed. Its "GPU Speed Of Light Throughput" section gives compute and memory throughput as percentages of peak; "Memory Workload Analysis" counts bytes and sectors per request at each level; "Occupancy" compares theoretical and achieved warps; "Warp State Statistics" says what stalled warps were waiting for.</p>
<p>A memory-bound kernel near 90% of DRAM bandwidth is done: Luo et al.'s microbenchmark reached 91% of theoretical on an H800 PCIe. NVIDIA's triage guide treats throughput below about 60% of peak on both compute and memory as a sign of latency limits: check occupancy and stall reasons next.</p>`,
    fig: {
      type: "timeline", caption: "Schematic of an Nsight Systems timeline for the backward pass of a training step (illustrative times)", unit: "µs", total: 100,
      rows: ["CPU API", "compute", "NCCL", "copies"],
      items: [
        ...[0, 4, 8, 30, 34, 60, 64, 68].map((t, k) => ({ row: 0, start: t, len: 2, label: "", kind: "s4", tip: `launch ${k + 1}` })),
        { row: 1, start: 3, len: 22, label: "GEMM bwd L3", kind: "s1", tip: "backward matrix multiplies of layer 3, compute-bound" },
        { row: 1, start: 25, len: 5, label: "", kind: "idle", tip: "gap: the GPU waits for the next launch" },
        { row: 1, start: 30, len: 11, label: "softmax", kind: "s3", tip: "softmax backward of layer 3, memory-bound" },
        { row: 1, start: 41, len: 20, label: "GEMM bwd L2", kind: "s1", tip: "backward matrix multiplies of layer 2" },
        { row: 1, start: 61, len: 5, label: "", kind: "idle", tip: "gap" },
        { row: 1, start: 66, len: 20, label: "GEMM bwd L1", kind: "s1", tip: "backward matrix multiplies of layer 1" },
        { row: 2, start: 41, len: 30, label: "all-reduce L3", kind: "s2", tip: "NCCL all-reduce of layer 3's gradients, overlapping layer 2's backward pass" },
        { row: 2, start: 71, len: 15, label: "L2", kind: "s2", tip: "all-reduce of layer 2's gradients, overlapping layer 1's backward pass" },
        { row: 2, start: 86, len: 10, label: "L1", kind: "s2", tip: "all-reduce of layer 1's gradients, not overlapped: the step waits" },
        { row: 3, start: 0, len: 12, label: "H→D batch", kind: "s5", tip: "next batch copied in, overlapping compute" },
      ],
    },
    src: ["NVIDIA Nsight Systems and Nsight Compute documentation (Profiling Guide: sections; Compute Triage Guide)", "Luo et al., IPDPS 2024 (Table V: 91% of theoretical global-memory bandwidth on H800 PCIe)"],
  });

  Deck.slide({
    id: "gpu-budget",
    title: "Where the time of one Llama 3 8B layer goes on an H100",
    layout: "wide",
    body: String.raw`<p>Put the part together on one layer of Llama 3 8B in BF16, processing T new tokens after L earlier ones held in the KV cache. For each operation, take the larger of FLOPs ÷ 989.4 TFLOP/s and bytes ÷ 3.35 TB/s: a roofline lower bound with perfect kernels, no overlap and no launch overhead. Attention is counted as its two matrix products in a fused, causal FlashAttention kernel that reads Q, K and V once (the cached ones too) and writes the output once.</p>
<p>At T = 2,048 and L = 0, a prefill, the four weight matrix products are compute-bound and take 86% of the time, attention 3.3%, and the memory-bound elementwise operations 11%. At T = 1 and L = 0 every operation is memory-bound and the time is reading the 436 MB of weights: 130 µs per layer, 4.2 ms for 32 layers. A real decode step also reads its KV cache, 2 × L × 1,024 × 2 bytes per layer: 33.6 MB (7.7% more time) at L = 8,192, and 537 MB, 1.23 times the weights, at L = 131,072. Real kernels reach a fraction of these bounds (FlashAttention-3: 75% of peak; Llama 3 405B training: 38 to 43% model FLOPs utilization).</p>`,
    fig: { type: "gpu-budget", T: 2048, caption: "Roofline time of each operation in one Llama 3 8B layer, BF16, on an H100 SXM" },
    check: {
      q: "At T = 1 and L = 0 the layer's bound is 130 µs. Roughly how long does T = 128 take, and why?",
      options: ["128 × 130 µs = 16.6 ms, because work grows with tokens", "About 144 µs: the weights are read once for all 128 tokens and every product is still memory-bound", "About 1 µs, because the tensor cores are 128 times busier", "The same 130 µs exactly, because only weights cost time"],
      answer: 1,
      why: "At T = 128 the weight products have an intensity of about 120 FLOPs per byte, still below 295, so they stay memory-bound: their time is set by the 436 MB of weights plus a few MB of activations. The total bound is 143.5 µs, 1.1 times the single-token time for 128 times the tokens.",
    },
    src: ["Dubey et al., The Llama 3 Herd of Models, 2024 (Table 3, Table 4)", "NVIDIA H100 whitepaper V1.04", "Shah et al., FlashAttention-3, 2024"],
  });
})();
