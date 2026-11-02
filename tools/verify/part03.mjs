// Part 3: every number stated on the neural-network sheets, recomputed.
// PyTorch-specific behavior (autograd output, saved tensors, optimizer steps) is
// recomputed by part03_torch.py; its printed results are pasted in TORCH below and
// cross-checked here against independent JavaScript recomputations.
import { near, equal, ok, r, int, text, TinyLM, window, done } from "./lib.mjs";

// The sheets wrap unary minus signs in TeX braces, {-0.7}, so KaTeX's MathML renders them
// as prefix minus signs. Compare against the text with those braces removed; normalize the
// needle the way lib.mjs normalizes sheet text.
const norm = (s) => String(s).replace(/−/g, "-").replace(/[\u00a0\u2009\u202f]/g, " ").replace(/\s+/g, " ");
const unbrace = (t) => t.replace(/(?<![\^_])\{(-[0-9.]+)\}/g, "$1").replace(/\{(-[a-z])\}/g, "$1").replace(/\{-f\(x\)\}/g, "-f(x)");
function claim(id, needle, label = "") {
  let t = "";
  try { t = unbrace(text(id)); } catch (e) { ok(`${id}: ${e.message}`, false); return; }
  ok(`${id}: text does not contain "${needle}"${label ? ` (${label})` : ""}`, t.includes(norm(needle)));
}

const NN3 = window.NN3;
const C = TinyLM.config;

// Printed by `python tools/verify/part03_torch.py` (torch 2.15.0, CPU, one thread).
const TORCH = {
  mlp_print: "tensor([ 0.5300, -0.2700,  3.5600])",
  twolayer_L: 0.22089999999999999,
  twolayer_grad: [-0.564, 0.376, 0.0, -1.128, 0.752, 0.0, -0.564, 0.376, 0.0, -0.94, -0.282, -0.0, -0.94],
  relu_grad_at_0: [0.0, 0.0, 1.0],
  relu_saves: ["_saved_result"], relu_saves_output: true, gelu_saves: ["_saved_approximate", "_saved_self"],
  mlp_saved_shapes: [[1024, 16], [1024, 64], [1024, 64]], mlp_saved_numbers: 147456, mlp_saved_bytes: 589824, mlp_saved_numbers_relu: 81920,
  tensor_shape: "torch.Size([3, 2])", tensor_dtype: "torch.float32", tensor_bytes: 24, matmul_shape: "torch.Size([3, 3])",
  broadcast_shape: "torch.Size([3, 3])", bf16_size: 2, wte_bytes: 7232,
  broadcast_31_13: [3, 3], broadcast_3_33: [3, 3], broadcast_32_3_error: "The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1",
  inplace_error: "a leaf Variable that requires grad is being used in an in-place operation.", outofplace_is_leaf: false, outofplace_grad_is_none: true,
  autograd_L_repr: "tensor(0.3600, grad_fn=<PowBackward0>)", autograd_next: [["SubBackward0", 0]], autograd_next2: [["AddBackward0", 0], [null, 0]],
  autograd_grads: ["tensor(-1.8000)", "tensor(-1.2000)"],
  sgd_manual: [0.9800000190734863, 0.3199999928474426], sgd_loss_repr: "tensor(0.0441, grad_fn=<PowBackward0>)", sgd_optim: [0.9799999594688416, 0.3199999928474426],
  accum_twice: "tensor(-3.6000)", zero_grad_sets_none: true,
  accum_max_abs_diff: 7.450580596923828e-09, accum_max_abs_grad: 0.030413124710321426, accum_max_abs_diff_f64: 1.214306433183765e-17,
  n_params: 9968, adam_state_numbers: 19936, adam_state_keys: ["exp_avg", "exp_avg_sq", "step"],
  momentum_3: [[-3.2800000000000002, -1.2], [-2.3296, -0.39], [-1.4350720000000001, 0.717]],
  gd_3: [[-3.2800000000000002, -1.2], [-2.6896000000000004, 0.96], [-2.2054720000000003, -0.7679999999999999]],
  adam_3: [[-3.900000000125, 1.4000000000333332], [-3.800042184370188, 1.300152563972872], [-3.700156261663697, 1.2005791528186043]],
  adam_3_m: [-2.110008436896537, 7.550305128005742], adam_3_v: [8.665964120783848, 111.66043379387808],
  steps_gd: 26, steps_mom: { "0.3": 14, "0.4": 12, "0.5": 15, "0.7": 30, "0.9": 75 }, steps_k100_gd: 307, steps_k100_mom09: 92,
  l2_g0_100: 0.12509830785123485, l2_g0_100_small: 0.12510842229500468, adamw_g0_100: 0.9047921471137096, l2_g0_1: 0.9900000009999999, adamw_g0_1: 0.999, l2_g10_1: 0.990000000009901, adam_g10_1: 0.99000000001, adamw_g10_1: 0.98900000001,
  clip_total: 2.5, clip_after: [0.5999997854232788, 0.7999997138977051],
  torch_depth: {
    "linear_0.500": [0.49846098557347085, 0.2486935874520974, 0.12439460877333751, 0.06234797399036718, 0.031051213088328946, 0.015458819656306128, 0.00775135658911423, 0.003903082641880317, 0.0019432513770353127, 0.0009770355896510307],
    "relu_1.000": [0.9969219711469417, 0.7107335072210368, 0.5003428650883427, 0.3669976272770669, 0.26107569779354517, 0.19823934810842203, 0.14109794126905634, 0.09944283073966223, 0.06612802915125333, 0.046944392208595066],
    "relu_1.414": [1.4098605722237243, 1.4214670144420734, 1.4151833312890925, 1.467990509108268, 1.476867170502605, 1.585914784867377, 1.5963409773249762, 1.591085291834597, 1.496306490859295, 1.5022205506750428],
  },
};
const sig = (z) => 1 / (1 + Math.exp(-z));
const relu = (z) => Math.max(0, z);
const GK = Math.sqrt(2 / Math.PI);
const gelu = (z) => 0.5 * z * (1 + Math.tanh(GK * (z + 0.044715 * z ** 3)));
const dgelu = (z, e = 1e-6) => (gelu(z + e) - gelu(z - e)) / (2 * e);
// row vector times matrix
const vm = (x, W) => W[0].map((_, j) => x.reduce((acc, xi, i) => acc + xi * W[i][j], 0));
const mm = (A, B) => A.map((row) => vm(row, B));
const close = (a, b, tol = 1e-9) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= tol);

// ------------------------------------------------------------------ 3.1 Neurons
{ // nn-xor: no line separates XOR; the best is 3 of 4
  const P = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
  let best = 0;
  for (let t = 0; t < 3600; t++) {
    const th = (t / 10) * Math.PI / 180, n = [Math.cos(th), Math.sin(th)];
    for (let ci = -300; ci <= 300; ci++) {
      const c = ci / 100;
      const okc = P.filter(([a, b, y]) => ((n[0] * a + n[1] * b - c) > 0 ? 1 : 0) === y).length;
      best = Math.max(best, okc);
    }
  }
  equal("XOR best linear accuracy over 2.16M lines", best, 3);
  claim("nn-xor", "the best is 3 of 4");
  const n = [Math.cos(Math.PI / 4), Math.sin(Math.PI / 4)];
  equal("default line gets 3", P.filter(([a, b, y]) => ((n[0] * a + n[1] * b - 0.5) > 0 ? 1 : 0) === y).length, 3);
}
{ // nn-neuron
  const z = 0.6 * 1.0 - 0.2 * 2.0 + 0.1;
  near("neuron z", z, 0.3, 1e-12);
  claim("nn-neuron", "0.6 \\cdot 1.0 - 0.2 \\cdot 2.0 + 0.1 = 0.3");
  claim("nn-neuron", `a = ${r(sig(z), 3)}`);
}
{ // nn-activations
  claim("nn-activations", `at most ${r(sig(0) * (1 - sig(0)), 2)}`); // 0.25
  const s5 = sig(5) * (1 - sig(5));
  claim("nn-activations", `${r(s5, 4)} at \\(z = 5\\)`);
  const t5 = 1 - Math.tanh(5) ** 2;
  claim("nn-activations", `at most 1, and ${r(t5, 4)} at \\(z = 5\\)`);
  // GELU minimum (tanh form)
  let mn = [0, 0], mxs = [0, -Infinity];
  for (let i = 0; i <= 600000; i++) {
    const z = -4 + (8 * i) / 600000, g = gelu(z), d = NN3.ACT.gelu.d(z);
    if (g < mn[1]) mn = [z, g];
    if (d > mxs[1]) mxs = [z, d];
  }
  claim("nn-activations", `dips to ${r(mn[1], 2)} at \\(z = ${r(mn[0], 2)}\\)`);
  near("analytic GELU slope matches numeric", NN3.ACT.gelu.d(0.7), dgelu(0.7), 1e-8);
  // the plate's tile says "1.13 at z = 1.42"
  near("GELU largest slope", mxs[1], 1.13, 0.005);
  near("GELU largest slope location", mxs[0], 1.42, 0.005);
  const src = window.Deck.figs["nn-act"].toString();
  ok("nn-act tile states GELU max slope 1.13 at z = 1.42", src.includes('gelu: "1.13 at z = 1.42"'));
  ok("nn-act tile states sigmoid max slope 0.25", src.includes('sigmoid: "0.25 at z = 0"'));
}
{ // nn-collapse
  const W1 = [[1, 2], [0, 1]], W2 = [[2, 0], [1, -1]], x = [1, 2];
  const a = vm(x, W1), b = vm(a, W2), W12 = mm(W1, W2), c = vm(x, W12);
  equal("xW1", a, [1, 4]); equal("(xW1)W2", b, [6, -4]); equal("W1W2", W12, [[4, -2], [1, -1]]); equal("x(W1W2)", c, [6, -4]);
  claim("nn-collapse", "= (1, 4)"); claim("nn-collapse", "= (6, -4)");
  claim("nn-collapse", "\\begin{pmatrix} 4 & -2 \\\\ 1 & -1 \\end{pmatrix}");
  const f = (x) => vm(vm(x, W1).map(relu), W2);
  equal("ReLU net at (1,-1)", f([1, -1]), [3, -1]);
  equal("ReLU net at (-1,1)", f([-1, 1]).map((v) => v + 0), [0, 0]);
  equal("linear net at (-1,1)", vm(vm([-1, 1], W1), W2), [-3, 1]);
  claim("nn-collapse", "gives (3, −1) but \\(x = (-1, 1)\\) gives (0, 0)");
  claim("nn-collapse", "= (0, 0) \\ne (-3, 1)");
}
{ // nn-xor-relu
  const net = (x1, x2) => { const h1 = relu(x1 + x2), h2 = relu(x1 + x2 - 1); return [h1, h2, h1 - 2 * h2]; };
  equal("XOR table", [[0, 0], [0, 1], [1, 0], [1, 1]].map(([a, b]) => net(a, b)), [[0, 0, 0], [1, 0, 1], [1, 0, 1], [2, 1, 0]]);
  const h = net(1, 1);
  equal("with output weight -1 at (1,1)", h[0] - h[1], 1);
}

// ------------------------------------------------------------------ 3.2 Layers and networks
const N = NN3.net231;
{ // nn-layer: Z = [X | 1][W; b]
  const X = [[1, 2], [-1, 1], [2, -1]];
  const Z = X.map((x) => vm(x, N.W1).map((v, j) => v + N.b1[j]));
  ok("Z rows", close(Z[0], [1.0, 0.3, -0.7]) && close(Z[1], [-0.2, 0.8, -1.7]) && close(Z[2], [0.9, -1.7, 3.3]));
  const H = Z.map((row) => row.map(relu));
  ok("H rows", close(H[0], [1.0, 0.3, 0]) && close(H[1], [0, 0.8, 0]) && close(H[2], [0.9, 0, 3.3]));
  claim("nn-layer", "rows (1.0, 0.3, 0), (0, 0.8, 0) and (0.9, 0, 3.3)");
  near("one batch, first MLP matrix", 1024 * 16 * 64, 1048576);
  near("tokens per batch", C.batch * C.T, 1024);
  claim("nn-layer", "1{,}024 \\times 16 \\times 64 = 1{,}048{,}576");
  // the plate's matrices are the network's
  const fig = window.Deck.slides.find((s) => s.id === "nn-layer").fig;
  equal("plate B = [W1; b1]", fig.B, [...N.W1, N.b1]);
  // nn-mlp-code outputs
  const Y = H.map((h) => h[0] * N.W2[0] + h[1] * N.W2[1] + h[2] * N.W2[2] + N.b2);
  ok("Y", close(Y, [0.53, -0.27, 3.56], 1e-12));
  claim("nn-mlp-code", "0.53, −0.27 and 3.56");
  claim("nn-mlp-code", "tensor([ 0.5300, -0.2700,  3.5600])");
  claim("nn-mlp-code", "1.0 × 0.6 + 0.3 × (−0.4) + 0 × 0.9 + 0.05 = 0.53");
}
{ // nn-forward
  const f = NN3.loss231(NN3.flat231(N));
  ok("z", close(f.z, [1.0, 0.3, -0.7], 1e-12)); ok("h", close(f.h, [1.0, 0.3, 0], 1e-12)); near("yhat", f.yh, 0.53, 1e-12);
  claim("nn-forward", "z_1 = 0.5 + 0.4 + 0.1 = 1.0"); claim("nn-forward", "z_2 = -0.5 + 1.0 - 0.2 = 0.3"); claim("nn-forward", "z_3 = 1.0 - 2.0 + 0.3 = -0.7");
  claim("nn-forward", "= 0.6 - 0.12 + 0 + 0.05 = 0.53");
  // unit 3: z3 = 1.0 x1 - 1.0 x2 + 0.3 > 0  <=>  x1 - x2 > -0.3
  equal("unit 3 weights", [N.W1[0][2], N.W1[1][2], N.b1[2]], [1.0, -1.0, 0.3]);
  claim("nn-forward", "x_1 - x_2 \\gt -0.3");
  near("weights in the 2-3-1 network", 2 * 3 + 3 * 1, 9);
  claim("nn-forward", "9 multiply-adds for 9 weights");
}
{ // nn-relu-sum
  const { f, a, b } = NN3.bumpTarget;
  const errs = [1, 4, 8].map((k) => NN3.bumps(f, a, b, k).maxErr);
  // independent recomputation: piecewise-linear interpolation through the centers
  const interp = (k) => {
    const cs = k === 1 ? [1.5] : Array.from({ length: k }, (_, i) => (3 * i) / (k - 1));
    const hw = k === 1 ? 1.5 : 3 / (k - 1);
    let m = 0;
    for (let i = 0; i <= 30000; i++) {
      const x = (3 * i) / 30000;
      const g = cs.reduce((acc, c) => acc + f(c) * Math.max(0, 1 - Math.abs(x - c) / hw), 0);
      m = Math.max(m, Math.abs(f(x) - g));
    }
    return m;
  };
  [1, 4, 8].forEach((k, i) => near(`bump error k=${k}`, errs[i], interp(k), 2e-4));
  claim("nn-relu-sum", `from ${r(errs[0], 3)} with one bump to ${r(errs[1], 3)} with four and ${r(errs[2], 3)} with eight`);
  const bound = ((3 / 7) ** 2 * 4) / 8;
  claim("nn-relu-sum", `which is ${r(bound, 3)} for eight bumps`);
  ok("error below the bound", errs[2] <= bound + 1e-12);
  equal("units for k bumps", NN3.bumps(f, a, b, 5).units, 7);
  claim("nn-relu-sum", "0 outside \\([c - s, c + s]\\)"); claim("nn-relu-sum", "on a closed and bounded region");
}
{ // nn-params
  near("2-3-1 params", 2 * 3 + 3 + 3 * 1 + 1, 13);
  claim("nn-params", "2 \\cdot 3 + 3 + 3 \\cdot 1 + 1 = 13");
  const sh = Object.fromEntries(C.shapes.map(([n, s]) => [n, s.reduce((a, b) => a * b, 1)]));
  const mlp = sh["l0.w1"] + sh["l0.w2"] + sh["l1.w1"] + sh["l1.w2"];
  const attn = ["wq", "wk", "wv", "wo"].reduce((a, k) => a + sh[`l0.${k}`] + sh[`l1.${k}`], 0);
  const norms = sh["l0.n1"] + sh["l0.n2"] + sh["l1.n1"] + sh["l1.n2"] + sh.nf;
  near("mlp per block", sh["l0.w1"] + sh["l0.w2"], 2048); near("mlp total", mlp, 4096);
  near("attention", attn, 2048); near("wte", sh.wte, 1808); near("wu", sh.wu, 1808); near("wpe", sh.wpe, 128); near("norms", norms, 80);
  near("sum", mlp + attn + sh.wte + sh.wu + sh.wpe + norms, C.params);
  claim("nn-params", "16 \\times 64 = 1{,}024"); claim("nn-params", "2,048 per block and 4,096 in both blocks");
  claim("nn-params", `${Math.round((100 * mlp) / C.params)}% of the 9,968 parameters`);
  claim("nn-params", "attention (2,048), the token embeddings and the unembedding (1,808 each), positions (128) and the RMSNorm gains (80)");
}
{ // nn-depth-width
  for (let L = 1; L <= 6; L++) {
    const pts = NN3.saw(L);
    ok(`saw L=${L} alternates 0/1 at its ${2 ** L + 1} breakpoints`, pts.every(([, v], j) => Math.abs(v - (j % 2)) < 1e-12));
    // between breakpoints it is linear: check a midpoint
    const x = 0.5 / 2 ** L; let v = x; for (let l = 0; l < L; l++) v = NN3.tent(v);
    near(`saw L=${L} midpoint`, v, 0.5, 1e-12);
  }
  claim("nn-depth-width", "64 pieces from 12 ReLU units"); claim("nn-depth-width", "at least 63 units");
}

// ------------------------------------------------------------------ 3.3 Backpropagation
{ // nn-bp-tape and nn-bp-graph: L = (w x + b - y)^2 at w 0.8, x 1.5, b 0.2, y 2
  const w = 0.8, x = 1.5, b = 0.2, y = 2;
  const m = w * x, z = m + b, e = z - y, L = e * e;
  near("m", m, 1.2, 1e-12); near("z", z, 1.4, 1e-12); near("e", e, -0.6, 1e-12); near("L", L, 0.36, 1e-12);
  ["1.2", "1.4", "−0.6", "0.36", "2e = -1.2"].forEach((t) => claim("nn-bp-tape", t));
  near("dL/dw", 2 * e * x, -1.8, 1e-12); near("dL/dx", 2 * e * w, -0.96, 1e-12);
  claim("nn-bp-graph", "\\partial L/\\partial e = 2e = -1.2"); claim("nn-bp-graph", "-1.2 \\times x = -1.8"); claim("nn-bp-graph", "-1.2 \\times w = -0.96");
  const Lw = (ww) => (ww * x + b - y) ** 2;
  claim("nn-bp-graph", `L(w + 0.001) = ${Lw(0.801).toFixed(8)}`); claim("nn-bp-graph", `L(w - 0.001) = ${Lw(0.799).toFixed(8)}`);
  near("central difference", (Lw(0.801) - Lw(0.799)) / 0.002, -1.8, 1e-9);
  claim("nn-bp-graph", `${r(Lw(0.801) - Lw(0.799), 4)} / 0.002 = -1.8`);
  claim("nn-bp-graph", "(-1.2)(1)(1)(1.5) = -1.8");
}
{ // nn-bp-neuron
  const z = 0.3, a = sig(z), L = (a - 1) ** 2, dLda = 2 * (a - 1), dadz = a * (1 - a), dz = dLda * dadz;
  claim("nn-bp-neuron", `a = \\sigma(0.3) = ${r(a, 4)}`); claim("nn-bp-neuron", `L = (a - 1)^2 = ${r(L, 4)}`);
  claim("nn-bp-neuron", `= ${r(dLda, 4)}`); claim("nn-bp-neuron", `${r(dLda, 4)} \\times ${r(dadz, 4)} = ${r(dz, 4)}`);
  claim("nn-bp-neuron", `2.0 = ${r(2 * dz, 4)}`);
  const w1 = 0.6 - 0.5 * dz, w2 = -0.2 - 0.5 * 2 * dz, b = 0.1 - 0.5 * dz, z2 = w1 + 2 * w2 + b, a2 = sig(z2), L2 = (a2 - 1) ** 2;
  claim("nn-bp-neuron", `w_1 = ${r(w1, 4)},\\ w_2 = ${r(w2, 4)},\\ b = ${r(b, 4)}`);
  claim("nn-bp-neuron", `z = ${r(z2, 4)},\\ a = ${r(a2, 4)},\\ L = ${r(L2, 4)}`);
  claim("nn-bp-neuron", `from ${r(L, 4)} to ${r(L2, 4)}`);
}
{ // nn-bp-layer
  const x = [1, 2], dl = [0.5, -1, 2];
  const outer = x.map((xi) => dl.map((dj) => xi * dj));
  equal("outer", outer, [[0.5, -1, 2], [1, -2, 4]]);
  claim("nn-bp-layer", "\\begin{pmatrix} 0.5 & -1 & 2 \\\\ 1 & -2 & 4 \\end{pmatrix}");
  const dx = N.W1.map((row) => row.reduce((acc, w, j) => acc + w * dl[j], 0));
  ok("dW^T", close(dx, [2.75, -2.4], 1e-12));
  claim("nn-bp-layer", "(0.25 + 0.5 + 2,\\ 0.1 - 0.5 - 2) \\\\ &= (2.75, -2.4)");
}
const G = NN3.grad231(NN3.flat231(N));
{ // nn-bp-relu and nn-bp-2layer
  ok("dh", close(G.dh, [-0.564, 0.376, -0.846], 1e-12)); ok("dz", close(G.dz, [-0.564, 0.376, 0], 1e-12));
  claim("nn-bp-relu", "(-0.564, 0.376, -0.846)"); claim("nn-bp-relu", "(-0.564, 0.376, 0)");
  equal("PyTorch ReLU slope at 0", TORCH.relu_grad_at_0[1], 0);
  ok("PyTorch ReLU saves its output", TORCH.relu_saves_output && TORCH.relu_saves.length === 1);
  ok("PyTorch GELU saves its input", TORCH.gelu_saves.includes("_saved_self"));
  claim("nn-bp-relu", "PyTorch also uses 0 at exactly");
  const gm = NN3.ACT.gelu.d(-1.5);
  claim("nn-bp-relu", `it is ${r(gm, 2)} at \\(z = -1.5\\)`);
  near("2-layer loss", G.f.L, 0.2209, 1e-12); near("torch loss", TORCH.twolayer_L, G.f.L, 1e-15);
  ok("backprop = torch autograd (float64)", close(G.grad, TORCH.twolayer_grad, 1e-15));
  claim("nn-bp-2layer", "L = 0.2209."); claim("nn-bp-2layer", "2(\\hat y - y) = -0.94");
  claim("nn-bp-2layer", "h^\\top(-0.94) = (-0.94, -0.282, 0)"); claim("nn-bp-2layer", "-0.94\\,W_2^\\top = (-0.564, 0.376, -0.846)");
  claim("nn-bp-2layer", "rows \\((-0.564, 0.376, 0)\\) and \\((-1.128, 0.752, 0)\\)");
  const p = NN3.flat231(N);
  const maxd = (eps, central) => Math.max(...NN3.fd231(p, eps, central).map((v, k) => Math.abs(v - G.grad[k])));
  const fmtSci = (v) => { const e = Math.floor(Math.log10(v)); return `${(v / 10 ** e).toFixed(1)} × 10${String(e).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d])}`; };
  claim("nn-bp-2layer", `largest disagreement is ${fmtSci(maxd(1e-4, true))}`);
  claim("nn-bp-2layer", `(${fmtSci(maxd(1e-4, false))} at \\(\\varepsilon = 10^{-4}\\))`);
  const fd5 = NN3.fd231(p, 0.5, true), wrongIdx = fd5.map((v, k) => (Math.abs(v - G.grad[k]) > 1e-6 ? k : -1)).filter((k) => k >= 0);
  equal("parameters wrong at eps 0.5", wrongIdx.map((k) => NN3.names231[k]), ["W₁[1,2]", "W₁[2,2]", "W₁[2,3]", "b₁[2]"]);
  // why: at x = (1, 2), z = (1.0, 0.3, -0.7); a nudge of 0.5 carries z₂ or z₃ across 0
  const zAt = (k, dd) => { const q = p.slice(); q[k] += dd; return NN3.loss231(q).z; };
  ok("W₁[1,2], W₁[2,2], b₁[2] nudged down by 0.5 switch unit 2 off", [1, 4, 7].every((k) => zAt(k, -0.5)[1] < 0 && zAt(k, 0.5)[1] > 0));
  ok("W₁[2,3] nudged up by 0.5 switches unit 3 on", zAt(5, 0.5)[2] > 0 && zAt(5, -0.5)[2] < 0);
  near("true gradient of W₁[2,3]", G.grad[5], 0, 0);
  claim("nn-bp-2layer", "three gradients come out wrong because unit 2 switches off (W₁[1,2], W₁[2,2], b₁[2]) and one because unit 3 switches on");
  claim("nn-bp-2layer", `(${r(fd5[5], 3)} for W₁[2,3] instead of 0)`);
  ok("eps 0.1 central is exact to rounding", maxd(0.1, true) < 1e-14);
}
{ // nn-bp-softmax and the nn-pgrad plate
  const tr = TinyLM.forward(TinyLM.runningIds.slice(0, 8)), pr = tr.probs[4], V = TinyLM.vocab;
  const P = (t) => pr[TinyLM.idOf(t)];
  claim("nn-bp-softmax", `<code>id</code> ${r(P(" id"), 4)}`.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim());
  claim("nn-bp-softmax", `email ${r(P(" email"), 4)}`); claim("nn-bp-softmax", `age ${r(P(" age"), 4)}`);
  const rest = 1 - P(" id") - P(" email") - P(" age");
  claim("nn-bp-softmax", `${r(rest, 4)} to the other ${V.length - 3} tokens together`);
  claim("nn-bp-softmax", `gradient is ${r(P(" id") - 1, 4)}`); claim("nn-bp-softmax", `+${r(P(" email"), 4)} and +${r(P(" age"), 4)}`);
  near("p - y sums to 0", pr.reduce((a, v) => a + v, 0) - 1, 0, 1e-12);
  ok("position 1 is certain", tr.probs[1][TinyLM.idOf(" FROM")] > 0.9995);
  claim("nn-bp-softmax", "position 1, FROM at 1.000");
  near("final vector size", tr.final[4].length, 16); near("vocab", V.length, 113);
}
{ // nn-bp-cost: 6ND with N the weights of matrix multiplies, plus attention's own products
  const sh = Object.fromEntries(C.shapes.map(([n, s]) => [n, s.reduce((a, b) => a * b, 1)]));
  const sum = (f) => Object.keys(sh).filter(f).reduce((a, n) => a + sh[n], 0);
  const mat = sum((n) => /\.(wq|wk|wv|wo|w1|w2)$/.test(n) || n === "wu"), look = sh.wte + sh.wpe, gains = sum((n) => /\.(n1|n2)$/.test(n) || n === "nf");
  equal("matrix weights", mat, 7952); equal("looked-up embedding entries", look, 1936); equal("norm gains", gains, 80); equal("all parameters", mat + look + gains, C.params);
  claim("nn-bp-cost", "Our model has 7,952 of them; its 1,936 embedding entries are looked up and its 80 norm gains multiply element by element");
  const D = C.steps * C.batch * C.T; near("tokens trained", D, 6144000);
  // per token and layer: scores against T positions (T·d multiply-adds) and the weighted sum of values (T·d):
  // 4·T·d FLOPs forward, times 3 with the backward pass
  const fMat = 6 * mat * D, fAttn = 3 * C.L * 4 * C.T * C.d * D, fAll = 6 * C.params * D;
  claim("nn-bp-cost", `6 \\times 7{,}952 \\times 6{,}144{,}000 \\approx ${r(fMat / 1e11, 1)} \\times 10^{11}`);
  claim("nn-bp-cost", `plus \\(${r(fAttn / 1e11, 2)} \\times 10^{11}\\) for attention's \\(8 \\times 8\\) score and mixing products, \\(\\approx ${r((fMat + fAttn) / 1e11, 1)} \\times 10^{11}\\)`);
  claim("nn-bp-cost", `Counting all 9,968 parameters gives \\(${r(fAll / 1e11, 1)} \\times 10^{11}\\), ${Math.round(100 * (fAll / (fMat + fAttn) - 1))}% too many`);
  equal("calc default forward", 2 * 1024 * 16 * 64, 2097152);
}
{ // nn-bp-memory
  const T = C.batch * C.T, saved = T * (16 + 64 + 64), w = 2 * 16 * 64;
  equal("saved numbers = PyTorch", saved, TORCH.mlp_saved_numbers); equal("saved bytes = PyTorch", 4 * saved, TORCH.mlp_saved_bytes);
  equal("saved shapes", TORCH.mlp_saved_shapes, [[1024, 16], [1024, 64], [1024, 64]]);
  equal("ReLU variant", T * (16 + 64), TORCH.mlp_saved_numbers_relu);
  claim("nn-bp-memory", `${int(saved)} numbers, ${int(4 * saved)} bytes`); claim("nn-bp-memory", `${int(w)} numbers, ${int(4 * w)} bytes`);
  claim("nn-bp-memory", `${saved / w} times larger`); claim("nn-bp-memory", `${int(2 * saved)} at 2,048`);
}

// ------------------------------------------------------------------ 3.4 Autograd and PyTorch
{
  [TORCH.tensor_shape, TORCH.tensor_dtype, TORCH.matmul_shape].forEach((t) => claim("nn-tensors", t));
  claim("nn-tensors", `# ${TORCH.tensor_bytes}: 6 numbers × 4 bytes`); claim("nn-tensors", `element_size() # ${TORCH.bf16_size}`);
  claim("nn-tensors", `element_size() # ${TORCH.wte_bytes}`);
  equal("wte bytes", 113 * 16 * 4, TORCH.wte_bytes);
  claim("nn-tensors", `${int(TORCH.wte_bytes)} bytes each`); claim("nn-tensors", `${int(4 * C.params)} bytes`);
  near("405B in bf16, GB", (405e9 * 2) / 1e9, 810);
  equal("model tensors", C.shapes.length, 20); claim("nn-tensors", "Our model is 20 tensors");
  // broadcasting: the stated rule, implemented here, against PyTorch
  const bshape = (a, b) => {
    const out = [];
    for (let i = 1; i <= Math.max(a.length, b.length); i++) {
      const x = a[a.length - i] ?? 1, y = b[b.length - i] ?? 1;
      if (x !== y && x !== 1 && y !== 1) return null;
      out.unshift(Math.max(x, y));
    }
    return out;
  };
  equal("(3,1) + (1,3) = PyTorch", bshape([3, 1], [1, 3]), TORCH.broadcast_31_13); equal("(3,3) + (3) = PyTorch", bshape([3, 3], [3]), TORCH.broadcast_3_33);
  ok("(3,2) + (3) refused by the rule and by PyTorch", bshape([3, 2], [3]) === null && /non-singleton dimension 1/.test(TORCH.broadcast_32_3_error));
  claim("nn-tensors", "shapes are compared from the last dimension, and each pair of sizes must be equal or one of them 1, a missing dimension counting as 1");
  claim("nn-autograd", TORCH.autograd_L_repr); equal("next_functions", TORCH.autograd_next[0][0], "SubBackward0");
  claim("nn-autograd", "((<SubBackward0 ...>, 0),)"); claim("nn-autograd", `(${TORCH.autograd_grads[0]}, ${TORCH.autograd_grads[1]})`);
  equal("next of sub", TORCH.autograd_next2[0][0], "AddBackward0");
  claim("nn-loop", "128 fresh queries, which give 1,024"); near("batch", C.batch, 128); near("steps", C.steps, 6000);
  claim("nn-loop", "lr=0.01, betas=(0.9, 0.95)"); equal("betas", C.betas, [0.9, 0.95]); near("lr", C.lr, 0.01); near("wd", C.wd, 0);
  near("SGD w", TORCH.sgd_manual[0], 0.98, 1e-6); near("SGD b", TORCH.sgd_manual[1], 0.32, 1e-6);
  near("SGD optim w", TORCH.sgd_optim[0], 0.98, 1e-6); near("SGD optim b", TORCH.sgd_optim[1], 0.32, 1e-6);
  near("loss after step", (0.98 * 1.5 + 0.32 - 2) ** 2, 0.0441, 1e-12);
  claim("nn-step", TORCH.sgd_loss_repr); claim("nn-step", "falls from 0.36 to 0.0441");
  near("residual after the step", 0.98 * 1.5 + 0.32 - 2, -0.21, 1e-12); claim("nn-step", "(0.98 × 1.5 + 0.32 − 2)² = (−0.21)² = 0.0441");
  equal("Adam state", TORCH.adam_state_numbers, 2 * C.params); equal("torch params", TORCH.n_params, C.params);
  claim("nn-step", `AdamW holds ${int(TORCH.adam_state_numbers)} numbers`);
  // mixed precision: bf16 weights 2 + bf16 grads 2 + fp32 master 4 + fp32 m 4 + fp32 v 4 = 16 bytes; optimizer state is 12 of them
  equal("bytes per parameter", 2 + 2 + 4 + 4 + 4, 16); ok("optimizer state is most of it", (4 + 4 + 4) / 16 > 0.5);
  claim("nn-step", "most of the 16 bytes per parameter");
  claim("nn-step", `"${TORCH.inplace_error.replace(/\.$/, "")}"`);
  ok("out of place, w is no longer a leaf and its .grad stays None", TORCH.outofplace_is_leaf === false && TORCH.outofplace_grad_is_none === true);
  claim("nn-accum", TORCH.accum_twice); claim("nn-accum", "is −3.6, twice −1.8");
  ok("zero_grad sets None", TORCH.zero_grad_sets_none);
  claim("nn-accum", `at most ${(TORCH.accum_max_abs_diff * 1e9).toFixed(1)} × 10⁻⁹ in float32 (largest gradient entry ${r(TORCH.accum_max_abs_grad, 2)})`);
  claim("nn-accum", `${(TORCH.accum_max_abs_diff_f64 * 1e17).toFixed(1)} × 10⁻¹⁷ in float64`);
}

// ------------------------------------------------------------------ 3.5 Making deep networks train
{ // nn-init-symmetry
  const half = NN3.symNet("half"), Lh = NN3.symTrain(half, 2000);
  claim("nn-init-symmetry", `every first-layer weight is ${r(half.W[0][0], 4)}`); claim("nn-init-symmetry", `the loss is ${r(Lh, 4)}`);
  ok("all six first-layer weights equal", Array.from(half.W[0]).every((v) => Math.abs(v - half.W[0][0]) < 1e-12));
  ok("biases and outgoing weights equal too", [half.b[0], half.W[1]].every((a) => Array.from(a).every((v) => Math.abs(v - a[0]) < 1e-12)));
  // equal weights on x₁ and x₂: each unit, and so the output, is one monotone (tanh) function of x₁ + x₂ = 0, 1, 1, 2.
  // The best such outputs for targets 0, 1, 1, 0, searched on a grid:
  let best = Infinity;
  for (let i = -50; i <= 150; i++) for (let j = -50; j <= 150; j++) for (let k = -50; k <= 150; k++) {
    const f0 = i / 100, f1 = j / 100, f2 = k / 100;
    if ((f0 <= f1 && f1 <= f2) || (f0 >= f1 && f1 >= f2)) best = Math.min(best, (f0 * f0 + 2 * (f1 - 1) ** 2 + f2 * f2) / 4);
  }
  near("lowest loss of a monotone function of x₁ + x₂", best, 1 / 6, 1e-3);
  claim("nn-init-symmetry", "can never go below 1/6"); claim("nn-init-symmetry", "one increasing or decreasing function of \\(x_1 + x_2\\)");
  ok("1/6 bound respected at step 2,000", Lh >= 1 / 6);
  const L5k = NN3.symTrain(half, 3000);
  ok(`the loss keeps falling: ${L5k.toFixed(4)} at step 5,000, still above 1/6`, L5k < Lh && L5k >= 1 / 6);
  const zero = NN3.symNet("zero"), Lz = NN3.symTrain(zero, 2000);
  ok("zero init: weights stay exactly 0", [...zero.W[0], ...zero.W[1], ...zero.b[0]].every((v) => v === 0));
  near("zero init output bias", zero.b[1][0], 0.5, 1e-12); near("zero init loss", Lz, 0.25, 1e-12);
  claim("nn-init-symmetry", "only the output bias moves (to 0.5) and the loss stays at 0.25");
  const rnd = NN3.symNet("random"), Lr = NN3.symTrain(rnd, 500);
  ok("random init loss < 1e-7 after 500 steps", Lr < 1e-7); claim("nn-init-symmetry", "below 10⁻⁷ after 500 steps");
}
const DS = (act) => NN3.depthStats(act);
{ // nn-init-depth and nn-grad-depth
  const lin = DS("linear"), relu = DS("relu"), sg = DS("sigmoid");
  claim("nn-init-depth", `\\(g = 0.5\\) ends at ${r(lin[0].fwd[10], 4)}`); claim("nn-init-depth", `\\(g = 2\\) at ${int(lin[3].fwd[10])}`);
  near("sqrt 128", Math.sqrt(128), 11.3, 0.05); claim("nn-init-depth", "g = \\sqrt{128} = 11.3");
  const big = lin[4].fwd[10], e = Math.floor(Math.log10(big));
  claim("nn-init-depth", `\\(${r(big / 10 ** e, 1)} \\times 10^{${e}}\\)`);
  claim("nn-init-depth", `(${r(lin[1].fwd[10], 2)} after 10 layers)`);
  claim("nn-init-depth", `\\((1/\\sqrt2)^9 = ${r(Math.SQRT1_2 ** 9, 3)}\\)`);
  claim("nn-init-depth", `this draw of weights gives ${r(relu[1].fwd[10], 3)}, PyTorch at width 512 gives ${r(TORCH.torch_depth["relu_1.000"][9], 3)}`);
  const perLayer = Math.pow(relu[1].fwd[10] / relu[1].fwd[1], 1 / 9);
  near("ReLU g=1 shrinks about sqrt 2 per layer", perLayer, Math.SQRT1_2, 0.05);
  // theory: linear std = g^l; checks that the measurement follows it
  near("linear g=0.5 vs 0.5^10", lin[0].fwd[10] / Math.pow(0.5, 10), 1, 0.1);
  near("linear g=2 vs 2^10", lin[3].fwd[10] / 1024, 1, 0.1);
  // PyTorch, width 512, its own random numbers: same behavior
  near("torch linear 0.5 vs theory", TORCH.torch_depth["linear_0.500"][9] / Math.pow(0.5, 10), 1, 0.02);
  near("torch relu g=1 vs theory (1/√2)^9", TORCH.torch_depth["relu_1.000"][9] / Math.pow(Math.SQRT1_2, 9), 1, 0.1);
  ok("torch relu √2 stays within 1.3 to 1.7", TORCH.torch_depth["relu_1.414"].every((v) => v > 1.3 && v < 1.7));
  // backward
  const sgr = sg[1].bwd[1] / sg[1].bwd[10], rr2 = relu[2].bwd[1] / relu[2].bwd[10], rr4 = relu[3].bwd[1] / relu[3].bwd[10];
  claim("nn-grad-depth", `\\(${r(sgr * 1e6, 1)} \\times 10^{-6}\\) of the gradient at layer 10`);
  claim("nn-grad-depth", `keeps the ratio at ${r(rr2, 2)}`); claim("nn-grad-depth", `by ${Math.round(rr4)} on its way down`);
}
{ // nn-init-schemes: our model's rules, and the step-0 weights follow them
  const W0 = TinyLM.weights(0);
  const sd = (n) => { const a = Array.from(W0[n].data), m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
  const target = { wte: 0.5, wpe: 0.5, "l0.wq": 0.25, "l0.wk": 0.25, "l0.wv": 0.25, "l1.wq": 0.25, "l1.wk": 0.25, "l1.wv": 0.25, "l0.wo": 0.125, "l1.wo": 0.125, "l0.w1": 0.25, "l1.w1": 0.25, "l0.w2": 0.0625, "l1.w2": 0.0625, wu: 0.02 };
  for (const [n, t] of Object.entries(target)) ok(`step-0 ${n} std ${sd(n).toFixed(4)} within 7% of ${t}`, Math.abs(sd(n) / t - 1) < 0.07);
  near("1/sqrt(16)", 1 / Math.sqrt(16), 0.25); near("wo rule", 1 / Math.sqrt(16) / Math.sqrt(2 * C.L), 0.125); near("w2 rule", 1 / Math.sqrt(64) / Math.sqrt(2 * C.L), 0.0625);
  claim("nn-init-schemes", "\\(\\sqrt{2L} = 2\\)");
  ok("norm gains are exactly 1 at step 0", ["l0.n1", "l0.n2", "l1.n1", "l1.n2", "nf"].every((n) => Array.from(W0[n].data).every((v) => v === 1)));
}
// The stream at position 4 leaving layer 2 (the final RMSNorm's input) at steps 0, 400 and 6,000.
const DRIFT = [0, 400, 6000].map((st) => {
  const tr = TinyLM.forward(TinyLM.runningIds.slice(0, 8), st), x = tr.layers[1].post[4], g = Array.from(TinyLM.weights(st).nf.data);
  const ms = x.reduce((a, v) => a + v * v, 0) / x.length, y = x.map((v) => v / Math.sqrt(ms + C.eps));
  ok(`step ${st}: post of layer 2, normalized and scaled, is the forward pass's final vector`, y.every((v, i) => Math.abs(v * g[i] - tr.final[4][i]) < 1e-12));
  near(`step ${st}: RMS after the norm, before the gains`, Math.sqrt(y.reduce((a, v) => a + v * v, 0) / y.length), 1, 1e-4);
  return Math.sqrt(ms);
});
{ // nn-norm-kinds, nn-rmsnorm
  const gains = C.shapes.filter(([n]) => /n1|n2|nf/.test(n));
  equal("RMSNorms", gains.length, 5); ok("16 gains each", gains.every(([, sh]) => sh[0] === 16));
  claim("nn-norm-kinds", "5 RMSNorms with 16 gains each");
  claim("nn-norm-kinds", `leaving layer 2 has RMS ${r(DRIFT[0], 2)} at step 0, ${r(DRIFT[1], 2)} at step 400 and ${r(DRIFT[2], 1)} at step 6,000, ${Math.round(DRIFT[2] / DRIFT[0])} times larger`);
  claim("nn-norm-kinds", "the RMSNorm after it still passes on a vector of RMS 1 before the gains");
  const tr = TinyLM.forward(TinyLM.runningIds.slice(0, 8)), x = tr.x0[4];
  claim("nn-rmsnorm", `between ${r(Math.min(...x), 2)} and ${r(Math.max(...x), 2)}`);
  const ms = x.reduce((a, v) => a + v * v, 0) / 16, rr = Math.sqrt(ms + C.eps);
  claim("nn-rmsnorm", `averages their squares, ${r(ms, 4)}`); claim("nn-rmsnorm", `the RMS is ${r(rr, 4)}`);
  claim("nn-rmsnorm", `mean square is ${r(x.reduce((a, v) => a + (v / rr) ** 2, 0) / 16, 4)}`);
  const g = Array.from(TinyLM.weights()["l0.n1"].data);
  claim("nn-rmsnorm", `between ${r(Math.min(...g), 4)} and ${r(Math.max(...g), 2)}`);
  equal("switched-off feature", g.indexOf(Math.min(...g)), 13); equal("largest gain feature", g.indexOf(Math.max(...g)), 12);
  claim("nn-rmsnorm", "feature 13 is all but switched off"); claim("nn-rmsnorm", `feature 12 passes at ${r(Math.max(...g), 2)}`);
  const y = x.map((v, i) => (v / rr) * g[i]);
  ok("worked output equals TinyLM h1", y.every((v, i) => Math.abs(v - tr.layers[0].h1[4][i]) < 1e-12));
  near("eps", C.eps, 1e-5);
}
{ // nn-residual
  const sg = NN3.residStats("sigmoid", 20), re = NN3.residStats("relu", 20);
  const ratio = (a) => a[0] / a[20];
  const sciT = (v) => { const e = Math.floor(Math.log10(v)); return `\\(${r(v / 10 ** e, 1)} \\times 10^{${e}}\\)`; };
  claim("nn-residual", `the gradient at the input is ${sciT(ratio(sg.plain))} of the gradient at the top`);
  claim("nn-residual", `the ratio is ${r(ratio(sg.resid), 1)}`);
  claim("nn-residual", `ReLU blocks: ${sciT(ratio(re.plain))} against ${r(ratio(re.resid), 1)}`);
  claim("nn-residual", "152 layers deep");
}
{ // nn-momentum
  near("GD factor per step", 1 - 2 * 0.09, 0.82, 1e-12); claim("nn-momentum", "shrinks by only 18% per step");
  equal("GD steps", NN3.stepsTo(0.09, 0), 26); equal("torch GD steps", TORCH.steps_gd, 26); claim("nn-momentum", "takes 26 steps");
  const P = NN3.momentumPath(0.09, 0.5, 3);
  ok("momentum path = PyTorch SGD(momentum=0.5)", P.slice(1).every((w, i) => close(w, TORCH.momentum_3[i], 1e-12)));
  const G = NN3.momentumPath(0.09, 0, 3);
  ok("GD path = PyTorch SGD", G.slice(1).every((w, i) => close(w, TORCH.gd_3[i], 1e-12)));
  const gs = P.slice(0, 3).map((w) => 2 * w[0]);
  claim("nn-momentum", `gradients are ${r(gs[0], 0)}, ${r(gs[1], 2)} and ${r(gs[2], 2)}`);
  let v = 0; const vs = gs.map((g) => (v = 0.5 * v + g));
  claim("nn-momentum", `velocities are ${r(vs[0], 0)}, ${r(vs[1], 2)} and ${r(vs[2], 2)}`);
  near("the velocity is a decaying sum: v₃ = g₃ + βg₂ + β²g₁", vs[2], gs[2] + 0.5 * gs[1] + 0.25 * gs[0], 1e-12);
  claim("nn-momentum", "each older gradient is multiplied by \\(\\beta\\) once more per step");
  claim("nn-momentum", `\\(w_1 = ${r(P[3][0], 2)}\\) instead of ${r(G[3][0], 2)}`);
  equal("momentum 0.5 steps", NN3.stepsTo(0.09, 0.5), 15); equal("torch", TORCH.steps_mom["0.5"], 15); claim("nn-momentum", "takes 15 steps");
  equal("momentum 0.9 steps", NN3.stepsTo(0.09, 0.9), 75); equal("torch 0.9", TORCH.steps_mom["0.9"], 75); claim("nn-momentum", "needs 75 steps here");
  equal("kappa 100, momentum", NN3.stepsTo(0.0099, 0.9, 1e-3, 100, 20000), 92); equal("kappa 100 GD", NN3.stepsTo(0.0099, 0, 1e-3, 100, 20000), 307);
  equal("torch kappa 100", [TORCH.steps_k100_mom09, TORCH.steps_k100_gd], [92, 307]);
  claim("nn-momentum", "needs 92 steps where plain descent needs 307");
  ok("eta 0.0099 below the stability limit 2/200", 0.0099 < 2 / 200);
}
{ // nn-rmsprop and its plate: SGD and RMSProp (ρ = 0.95, ε = 10⁻⁸) on ½(25a² + b²) from (−1, 1), recomputed here
  const Lb = (a, b) => 0.5 * (25 * a * a + b * b);
  const sgdRun = (eta, n) => { let a = -1, b = 1; const out = [{ a, b }]; for (let t = 0; t < n; t++) { a -= eta * 25 * a; b -= eta * b; out.push({ a, b }); if (!(Lb(a, b) < 1e12)) break; } return out; };
  const rmsRun = (eta, n) => {
    let a = -1, b = 1, sa = 0, sb = 0; const out = [{ a, b, sa, sb }];
    for (let t = 0; t < n; t++) {
      const ga = 25 * a, gb = b;
      sa = 0.95 * sa + 0.05 * ga * ga; sb = 0.95 * sb + 0.05 * gb * gb;
      a -= (eta * ga) / (Math.sqrt(sa) + 1e-8); b -= (eta * gb) / (Math.sqrt(sb) + 1e-8);
      out.push({ a, b, sa, sb });
      if (!(Lb(a, b) < 1e12)) break;
    }
    return out;
  };
  const stepsTo = (run, eta) => { const P = run(eta, 5000), i = P.findIndex((q) => Lb(q.a, q.b) < 1e-3); return i >= 0 ? String(i) : P.length <= 5000 ? "diverges" : "> 5,000"; };
  near("SGD limit", 2 / 25, 0.08, 1e-12);
  ok("SGD converges at 0.079, diverges at 0.081", Lb(...Object.values(sgdRun(0.079, 1000)[1000])) < 1 && sgdRun(0.081, 1000).length < 1001);
  claim("nn-rmsprop", "diverges above \\(\\eta = 0.08\\), where the factor \\(1 - 25\\eta\\) passes -1");
  const S = sgdRun(0.07, 10); near("SGD flat coordinate after 10 steps", S[10].b, 0.93 ** 10, 1e-12);
  claim("nn-rmsprop", `shrinks by only 7% per step: from 1 to ${r(S[10].b, 3)} after 10 steps`);
  const Rr = rmsRun(0.07, 60), E = NN3.bowlPath("rms", 0.07, 60), Es = NN3.bowlPath("sgd", 0.07, 60), Sg = sgdRun(0.07, 60);
  ok("plate's RMSProp path = recomputation", E.every((q, i) => Math.abs(q.w[0] - Rr[i].a) < 1e-12 && Math.abs(q.w[1] - Rr[i].b) < 1e-12 && Math.abs(q.s[0] - Rr[i].sa) < 1e-9 && Math.abs(q.s[1] - Rr[i].sb) < 1e-12));
  ok("plate's SGD path = recomputation", Es.every((q, i) => Math.abs(q.w[0] - Sg[i].a) < 1e-12 && Math.abs(q.w[1] - Sg[i].b) < 1e-12));
  const ra = 0.07 / Math.sqrt(Rr[1].sa), rb = 0.07 / Math.sqrt(Rr[1].sb);
  claim("nn-rmsprop", `after one step the rates are ${r(ra, 4)} for \\(a\\) and ${r(rb, 3)} for \\(b\\), a ratio of ${r(rb / ra, 0)}, the ratio of the two gradients`);
  near("ratio of the first gradients", 25 / 1, rb / ra, 1e-6);
  const stepsR = stepsTo(rmsRun, 0.07);
  ok("RMSProp moves along the diagonal |a| = |b| until it gets there", Rr.slice(0, Number(stepsR) + 1).every((q) => Math.abs(Math.abs(q.a) - Math.abs(q.b)) < 1e-6));
  claim("nn-rmsprop", `after 10 steps \\(b = ${r(Rr[10].b, 3)}\\)`);
  equal("RMSProp steps to L < 1e-3", stepsR, "16"); equal("SGD steps to L < 1e-3", stepsTo(sgdRun, 0.07), "43");
  claim("nn-rmsprop", "\\(L \\lt 10^{-3}\\) takes 16 steps against SGD's 43");
  near("RMSProp first step = η/√0.05", Math.abs(Rr[1].a + 1), 0.07 / Math.sqrt(0.05), 1e-9); near("1/√0.05", 1 / Math.sqrt(0.05), 4.47, 0.005);
  claim("nn-rmsprop", "step 1 is \\(\\eta/\\sqrt{0.05} = 4.5\\eta\\)");
  // the plate at every slider position (0.01 to 0.1 by 0.005): its step counts are the recomputed ones
  for (let i = 0; i <= 18; i++) {
    const eta = 0.01 + 0.005 * i;
    equal(`plate at η = ${eta.toFixed(3)}: steps (SGD, RMSProp)`, [NN3.bowlStepsTo("sgd", eta), NN3.bowlStepsTo("rms", eta)], [stepsTo(sgdRun, eta), stepsTo(rmsRun, eta)]);
  }
  const fig = window.Deck.slides.find((x) => x.id === "nn-rmsprop").fig;
  equal("plate defaults", [fig.eta, fig.show], [0.07, 10]);
}
{ // nn-adam: three steps recomputed here, against the table and against torch.optim.Adam
  const A = NN3.adamValley(0.1, 3);
  ok("JS Adam = torch.optim.Adam", A.every((row, i) => close(row.w, TORCH.adam_3[i], 1e-12)));
  ok("m, v after 3 steps = torch state", close(A[2].m, TORCH.adam_3_m, 1e-12) && close(A[2].v, TORCH.adam_3_v, 1e-9));
  near("gradient ratio", 30 / 8, 3.75); claim("nn-adam", "whose two gradients differ by a factor of 3.75");
  let w = [-4, 1.5], m = [0, 0], v = [0, 0];
  const sgn4 = (x) => (x > 0 ? "+" : "") + r(x, 4);
  for (let t = 1; t <= 3; t++) {
    const g = [2 * w[0], 20 * w[1]];
    m = m.map((x, i) => 0.9 * x + 0.1 * g[i]); v = v.map((x, i) => 0.95 * x + 0.05 * g[i] * g[i]);
    const mh = m.map((x) => x / (1 - 0.9 ** t)), vh = v.map((x) => x / (1 - 0.95 ** t)), d = mh.map((x, i) => (-0.1 * x) / (Math.sqrt(vh[i]) + 1e-8));
    w = w.map((x, i) => x + d[i]);
    ok(`step ${t}: both coordinates move by about η`, d.every((x) => Math.abs(Math.abs(x) - 0.1) < 0.001));
    claim("nn-adam", `${t} \\(w_1\\) ${r(g[0], 2)} ${r(mh[0], 3)} ${r(vh[0], 1)} ${sgn4(d[0])} \\(w_2\\) ${r(g[1], 2)} ${r(mh[1], 3)} ${r(vh[1], 1)} ${sgn4(d[1])}`, `table rows of step ${t}`);
    if (t === 3) ok("step 3 m̂ = torch's m / (1 − 0.9³)", close(mh, TORCH.adam_3_m.map((x) => x / (1 - 0.9 ** 3)), 1e-12));
  }
  near("step 1 without bias correction", (0.1 * 0.1) / Math.sqrt(0.05), 0.045, 5e-4);
  claim("nn-adam", "Without bias correction step 1 would be \\(0.1 \\cdot 0.1/\\sqrt{0.05} = 0.045\\)");
  // the plate (the optim widget): Adam with β₂ = 0.999 on the previous sheet's bowl
  const wsrc = window.Widgets.optim.init.toString();
  ok("optim widget: bowl ½(25a² + b²), Adam β = (0.9, 0.999)", wsrc.includes("const K = 25") && wsrc.includes("0.9 * x + 0.1 * g[i]") && wsrc.includes("0.999 * x + 0.001 * g[i] * g[i]"));
  // the check: steady gradients 0.001 and 1,000, 100 steps at η = 0.01
  const moved = (g) => { let mm = 0, vv = 0, x = 0; for (let t = 1; t <= 100; t++) { mm = 0.9 * mm + 0.1 * g; vv = 0.95 * vv + 0.05 * g * g; x += (0.01 * (mm / (1 - 0.9 ** t))) / (Math.sqrt(vv / (1 - 0.95 ** t)) + 1e-8); } return x; };
  ok(`check: both move by about 1 (${moved(0.001).toFixed(5)}, ${moved(1000).toFixed(5)})`, Math.abs(moved(0.001) - 1) < 0.001 && Math.abs(moved(1000) - 1) < 1e-6);
  equal("betas", C.betas, [0.9, 0.95]);
}
{ // nn-adamw: recompute PyTorch's Adam (L2 in the gradient) and AdamW (decay on the weight) in JS
  const run = (kind, g, steps, lam) => {
    let w = 1, m = 0, v = 0;
    for (let t = 1; t <= steps; t++) {
      let gg = g;
      if (kind === "l2") gg += lam * w;
      if (kind === "w") w -= 0.01 * lam * w;
      m = 0.9 * m + 0.1 * gg; v = 0.95 * v + 0.05 * gg * gg;
      w -= (0.01 * (m / (1 - 0.9 ** t))) / (Math.sqrt(v / (1 - 0.95 ** t)) + 1e-8);
    }
    return w;
  };
  near("L2 g0 100 = torch", run("l2", 0, 100, 0.1), TORCH.l2_g0_100, 1e-12);
  near("L2 small lambda = torch", run("l2", 0, 100, 0.001), TORCH.l2_g0_100_small, 1e-12);
  near("AdamW g0 100 = torch", run("w", 0, 100, 0.1), TORCH.adamw_g0_100, 1e-12);
  near("AdamW = 0.999^100", TORCH.adamw_g0_100, 0.999 ** 100, 1e-12);
  near("L2 g10 1", run("l2", 10, 1, 0.1), TORCH.l2_g10_1, 1e-12); near("Adam g10 1", run("a", 10, 1, 0), TORCH.adam_g10_1, 1e-12); near("AdamW g10 1", run("w", 10, 1, 0.1), TORCH.adamw_g10_1, 1e-12);
  near("L2 g0 1", run("l2", 0, 1, 0.1), TORCH.l2_g0_1, 1e-12); near("AdamW g0 1", run("w", 0, 1, 0.1), TORCH.adamw_g0_1, 1e-12);
  claim("nn-adamw", `after 100 steps it is at ${r(TORCH.l2_g0_100, 3)} (the note below shows the same for a 100 times smaller \\(\\lambda\\))`);
  near("the note's smaller λ gives the same", TORCH.l2_g0_100_small, TORCH.l2_g0_100, 5e-4);
  claim("learn-regularize", "\\lambda \\sum_{j} c_j^2", "Part 2's penalty"); claim("nn-adamw", "Part 2's penalty \\(\\lambda \\sum c_j^2\\) has gradient \\(2\\lambda c_j\\)");
  claim("nn-adamw", `at ${r(TORCH.adamw_g0_100, 3)} after 100 steps`);
  const rowsT = [["1.000", r(TORCH.l2_g0_1, 3), r(TORCH.adamw_g0_1, 3)], ["1.000", r(TORCH.l2_g0_100, 3), r(TORCH.adamw_g0_100, 3)], [r(TORCH.adam_g10_1, 3), r(TORCH.l2_g10_1, 3), r(TORCH.adamw_g10_1, 3)]];
  equal("AdamW table", window.Deck.slides.find((x) => x.id === "nn-adamw").fig.rows.map((row) => row.slice(1)), rowsT);
  claim("nn-adamw", "λ = 0.001 also leaves the unused weight at 0.125");
  equal("our model's weight decay", C.wd, 0);
}
{ // nn-schedule
  near("lr step 0", NN3.lrAt(0), 0.0001, 1e-15); near("lr step 99", NN3.lrAt(99), 0.01, 1e-15); near("lr step 100", NN3.lrAt(100), 0.01, 1e-15);
  near("lr step 3050", NN3.lrAt(3050), 0.0055, 1e-12); near("lr at 6000", NN3.lrAt(6000), 0.001, 1e-12); near("lr at 5999", NN3.lrAt(5999), 0.001, 1e-8);
  claim("nn-schedule", "from 0.0001 at step 0 to 0.01 at step 99"); claim("nn-schedule", "from 0.01 at step 100 to 0.001 at step 6,000");
  claim("nn-schedule", "at step 3,050, it is 0.0055");
  // Adam's first step is η·g/(|g| + ε): at most η for any gradient
  const rnd = TinyLM.mulberry32(3);
  ok("Adam's first step never exceeds eta", Array.from({ length: 1000 }, () => (rnd() - 0.5) * 10 ** (8 * rnd() - 4)).every((g) => Math.abs((NN3.lrAt(0) * g) / (Math.abs(g) + 1e-8)) <= NN3.lrAt(0)));
  claim("nn-schedule", "no weight moves by more than 0.0001"); near("peak / step-0 rate", C.lr / NN3.lrAt(0), 100, 1e-9); claim("nn-schedule", "about 0.01, a hundred times more");
  equal("warmup and steps", [C.warmup, C.steps, C.lr], [100, 6000, 0.01]);
  claim("nn-schedule", "\\tfrac{t - 100}{5{,}900}");
  // the displayed formula is the code's formula
  const shown = (t) => 0.01 * (0.1 + 0.45 * (1 + Math.cos(Math.PI * (t - 100) / 5900)));
  ok("displayed cosine formula equals lr_at for t >= 100", [100, 777, 3050, 5000, 5999].every((t) => Math.abs(shown(t) - NN3.lrAt(t)) < 1e-15));
  claim("nn-schedule", "0.01\\,\\big(0.1 + 0.45\\,(1 + \\cos\\theta_t)\\big)"); claim("nn-schedule", "\\theta_t &= \\pi\\,\\tfrac{t - 100}{5{,}900}");
}
{ // nn-clip
  equal("torch total norm", TORCH.clip_total, Math.hypot(1.5, 2)); ok("torch clipped grads", close(TORCH.clip_after, [0.6, 0.8], 1e-6));
  claim("nn-clip", "gradients 1.5 and 2 have global norm 2.5"); claim("nn-clip", `scaled by ${1 / 2.5}, to 0.6 and 0.8`);
  const ang = (Math.atan2(2, 1.5) - Math.atan2(1, 1)) * 180 / Math.PI;
  claim("nn-clip", `a direction ${r(ang, 1)}° off`);
  const b = TinyLM.data.batchLoss10; let up = 0;
  for (let i = 30; i < b.length; i++) up = Math.max(up, b[i] - Math.min(...b.slice(Math.max(0, i - 10), i)));
  ok(`largest rise after step 300 is ${up.toFixed(4)} <= 0.007`, up <= 0.007);
  claim("nn-clip", "never rose more than 0.007");
}

{ // nn-recipe restates numbers from the chapter: recompute each
  const half = NN3.symNet("half"); const Lh = NN3.symTrain(half, 2000);
  claim("nn-recipe", `XOR loss never below 1/6 from equal weights (${r(Lh, 4)} at step 2,000); below 10⁻⁷ from random ones`);
  claim("nn-recipe", "initial weights with a fixed standard deviation of 0.02 instead of one set by the fan-in, the weights of residual layers further scaled by \\(1/\\sqrt{N}\\)");
  const lin = NN3.depthStats("linear");
  claim("nn-recipe", `0.0009 at g = 0.5, ${int(lin[3].fwd[10])} at g = 2, ${r(lin[1].fwd[10], 2)} at g = 1`); near("0.0009", lin[0].fwd[10], 0.0009, 0.00005);
  claim("nn-recipe", `position 4 leaving layer 2: RMS ${r(DRIFT[0], 2)} at step 0, ${r(DRIFT[2], 1)} at step 6,000; RMS 1 again after the norm, before the gains`);
  const sg = NN3.residStats("sigmoid", 20);
  claim("nn-recipe", `20 sigmoid layers: ${r(sg.plain[0] / sg.plain[20] * 1e13, 1)} × 10⁻¹³ without, ${r(sg.resid[0] / sg.resid[20], 1)} with`);
  claim("nn-recipe", `${NN3.stepsTo(0.09, 0)} steps to L < 10⁻³ without, ${NN3.stepsTo(0.09, 0.5)} with β = 0.5`);
  const A = NN3.adamValley(0.1, 1); ok("Adam first steps are 0.1", A[0].d.every((d) => Math.abs(Math.abs(d) - 0.1) < 1e-9));
  claim("nn-recipe", "steps of 0.1 for gradients of −8 and 30 alike");
  claim("nn-recipe", `${r(TORCH.l2_g0_100, 3)} with L2 in Adam, ${r(TORCH.adamw_g0_100, 3)} with AdamW`);
  claim("nn-recipe", "η from 0.0001 to 0.01 by step 99, then down to 0.001 at step 6,000");
  claim("nn-recipe", `norm ${Math.hypot(1.5, 2)} scaled to 1, direction kept`);
}

// ------------------------------------------------------------------ 3.6 A network trained in the page
{ // the general hand-written backward pass agrees with central differences
  for (const [act, loss] of [["tanh", "bce"], ["relu", "mse"], ["sigmoid", "bce"], ["gelu", "mse"]]) {
    const net = NN3.mlp.create([2, 5, 4, 1], act, null, 4), rand = TinyLM.mulberry32(9);
    const B = 7, X = Float64Array.from({ length: 2 * B }, () => 2 * rand() - 1), Y = Array.from({ length: B }, () => (rand() < 0.5 ? 0 : 1));
    const g = NN3.mlp.grads(net, X, Y, B, loss), eps = 1e-6;
    let worst = 0;
    [...net.W, ...net.b].forEach((P, k) => {
      const G = k < net.W.length ? g.gW[k] : g.gb[k - net.W.length];
      for (let i = 0; i < P.length; i++) {
        const v = P[i];
        P[i] = v + eps; const Lu = NN3.mlp.loss(net, X, Y, B, loss).loss;
        P[i] = v - eps; const Ld = NN3.mlp.loss(net, X, Y, B, loss).loss;
        P[i] = v;
        worst = Math.max(worst, Math.abs((Lu - Ld) / (2 * eps) - G[i]));
      }
    });
    ok(`MLP backward (${act}, ${loss}) matches finite differences: worst ${worst.toExponential(1)}`, worst < 1e-7);
  }
}
const SP = NN3.data.spirals();
const pct = (a) => (a === 1 ? "100%" : `${(100 * a).toFixed(1)}%`);
const fmtL = (v) => { if (v >= 0.01) return v.toFixed(3); if (v >= 0.001) return v.toFixed(4); const e = Math.floor(Math.log10(v)); return `${r(v / 10 ** e, 1)} × 10${String(e).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d])}`; };
{ // nn-train-demo
  const run = NN3.trainer(SP, 16, 2, 0.01), at = {};
  for (const m of [200, 400, 600]) { while (run.steps < m) run.step(); at[m] = run.eval(); }
  claim("nn-train-demo", `loss ${r(at[200].loss, 3)}, ${pct(at[200].acc)} of the points`);
  claim("nn-train-demo", `After 400 steps the loss is ${r(at[400].loss, 3)}`); equal("all correct at 400", at[400].acc, 1);
  claim("nn-train-demo", `by 600 it is ${r(at[600].loss, 4)}`);
  equal("params 2-16-16-1", (2 * 16 + 16) + (16 * 16 + 16) + (16 + 1), 337); claim("nn-train-demo", "337 parameters");
  equal("200 points", SP.B, 200);
}
{ // nn-train-knobs: every table row recomputed
  const rows = window.Deck.slides.find((x) => x.id === "nn-train-knobs").fig.rows;
  const cfg = [[2, 1, 0.01], [4, 1, 0.01], [8, 1, 0.01], [16, 1, 0.01], [32, 1, 0.01], [8, 2, 0.01], [8, 3, 0.01], [16, 2, 0.001], [16, 2, 0.01], [16, 2, 0.1], [16, 2, 1]];
  const params = (w, d) => 2 * w + w + (d - 1) * (w * w + w) + w + 1;
  cfg.forEach(([w, d, lr], i) => {
    const e = NN3.trainTo(SP, w, d, lr, 1000).eval();
    equal(`knobs row ${i + 1}`, rows[i].slice(1), [int(params(w, d)), fmtL(e.loss), pct(e.acc)]);
  });
  claim("nn-train-knobs", "2 units end at 56% accuracy");
  const d82 = NN3.trainTo(SP, 8, 2, 0.01, 1000).eval(), d321 = NN3.trainTo(SP, 32, 1, 0.01, 1000).eval(), d161 = NN3.trainTo(SP, 16, 1, 0.01, 4000).eval();
  claim("nn-train-knobs", `Two layers of 8 units (${params(8, 2)} parameters) reach a loss of ${fmtL(d82.loss)} by step 1,000, one layer of 32 (${params(32, 1)} parameters) ${fmtL(d321.loss)}`);
  ok("both classify every point by step 1,000", d82.acc === 1 && d321.acc === 1); claim("nn-train-knobs", "Both classify every point by then");
  claim("nn-train-knobs", `(width 16: ${fmtL(d161.loss)} at step 4,000)`);
  claim("nn-train-knobs", "At 0.001 the network is at 77.5% after 1,000 steps");
  const fast = NN3.trainTo(SP, 16, 2, 0.1, 200).eval(); equal("lr 0.1 fits by step 200", fast.acc, 1);
  const r1 = NN3.trainer(SP, 16, 2, 1);
  for (let k = 0; k < 10; k++) r1.step();
  const h1 = NN3.mlp.forward(r1.net, SP.X, SP.B).hs[1];
  const sat = h1.filter((v) => Math.abs(v) > 0.99).length / h1.length;
  claim("nn-train-knobs", `after 10 steps ${Math.round(100 * sat)}% of first-layer outputs are beyond ±0.99`);
  let lo = Infinity, hi = -Infinity, Lmin = Infinity, Lmax = -Infinity, spread = 0;
  const accs = new Set();
  while (r1.steps < 1000) {
    r1.step();
    if (r1.steps < 300) continue;
    const o = NN3.mlp.forward(r1.net, SP.X, SP.B).hs[3], e = r1.eval();
    spread = Math.max(spread, Math.max(...o) - Math.min(...o)); lo = Math.min(lo, o[0]); hi = Math.max(hi, o[0]);
    Lmin = Math.min(Lmin, e.loss); Lmax = Math.max(Lmax, e.loss); accs.add(e.acc);
  }
  ok(`lr 1, steps 300 to 1,000: one logit for every point (largest spread ${spread.toExponential(1)})`, spread < 1e-6);
  equal("lr 1, steps 300 to 1,000: accuracy", [...accs], [0.5]);
  claim("nn-train-knobs", `From step 300 on every point gets the same logit, which Adam's steps keep swinging between ${r(lo, 2)} and +${r(hi, 2)}: the loss cycles between ${r(Lmin, 3)} and ${r(Lmax, 3)} and the accuracy stays at 50%`);
  equal("100 points per class", Array.from(SP.Y).filter((y) => y === 1).length, 100);
  const Lz = (z) => (Math.log(1 + Math.exp(-z)) + Math.log(1 + Math.exp(z))) / 2; // one shared logit z, two equal classes
  ok("from step 300 the loss never falls below ln 2", Lmin >= Math.log(2) - 1e-12); near("ln 2", Math.log(2), 0.693, 5e-4);
  near("the cycle's top is the loss of the largest swing", Lz(Math.max(-lo, hi)), Lmax, 1e-3);
  claim("nn-train-knobs", "the loss is \\(-(\\ln p + \\ln(1 - p))/2\\), smallest at \\(p = 0.5\\)");
  claim("nn-train-knobs", `the swings of the logit to ±${r(hi, 2)} give ${r(Lz(hi), 3)}`);
  ok("step 1,000 is near the bottom of the cycle", r1.eval().loss - Lmin < 1e-3);
  claim("nn-train-knobs", `At η = 1 the loss cycles between ${r(Lmin, 3)} and ${r(Lmax, 3)} from step 300 on; step 1,000 falls near the bottom of the cycle`);
}
{ // nn-overfit
  const tr = NN3.data.moons(15, 0.25, 21), te = NN3.data.moons(500, 0.25, 99);
  equal("30 training points", tr.B, 30); equal("1,000 test points", te.B, 1000);
  const run = NN3.trainer(tr, 32, 2, 0.01, te);
  while (run.steps < 1000) run.step();
  const e = run.eval(), t = run.evalTest();
  equal("params 2-32-32-1", (2 * 32 + 32) + (32 * 32 + 32) + (32 + 1), 1185); claim("nn-overfit", "1,185 parameters");
  equal("training accuracy", e.acc, 1); claim("nn-overfit", `training loss is \\(${r(e.loss * 1e9, 0)} \\times 10^{-9}\\)`);
  claim("nn-overfit", `accuracy is ${(100 * t.acc).toFixed(1)}% and the loss ${r(t.loss, 2)}`);
  const o = NN3.mlp.forward(run.net, te.X, te.B).hs[3];
  let wrong = 0, conf = 0;
  for (let i = 0; i < te.B; i++) { const p = 1 / (1 + Math.exp(-o[i])); if ((p > 0.5 ? 1 : 0) !== te.Y[i]) { wrong++; if (Math.max(p, 1 - p) > 0.99) conf++; } }
  claim("nn-overfit", `${conf} of the ${wrong} wrong predictions`); near("ln 100", Math.log(100), 4.6, 0.01);
  const best = run.testCurve.reduce((b, q) => (q[1] < b[1] ? q : b), [0, Infinity]);
  claim("nn-overfit", `lowest, ${r(best[1], 3)}, at step ${best[0]}`);
  const small = NN3.trainer(tr, 4, 1, 0.01, te); while (small.steps < 3000) small.step();
  const bs = small.testCurve.reduce((b, q) => (q[1] < b[1] ? q : b), [0, Infinity]);
  claim("nn-overfit", `${r(bs[1], 3)} at step ${bs[0]}`); equal("4-unit params", 2 * 4 + 4 + 4 + 1, 17);
  ok("the 4-unit network overfits later", small.testCurve[small.testCurve.length - 1][1] > bs[1] * 2);
  const ex = TinyLM.exactLoss().mean, floor = TinyLM.data.floor.reduce((a, b) => a + b, 0) / TinyLM.data.floor.length;
  claim("nn-overfit", `its loss, ${r(ex, 4)}, sits just above the ${r(floor, 4)} floor`);
  equal("270 queries", TinyLM.allQueries().length, 270);
}
{ // nn-same-loop
  const rows = window.Deck.slides.find((x) => x.id === "nn-same-loop").fig.rows;
  equal("parameters row", rows[0].slice(1, 3), ["337", int(C.params)]);
  equal("our batch", C.batch * C.T, 1024); claim("nn-same-loop", "128 queries × 8 positions = 1,024 predictions");
  claim("nn-same-loop", "100 steps, cosine to 0.001"); claim("nn-same-loop", "cross-entropy over 113 tokens"); equal("vocab", C.V, 113);
  const fl = 6 * 405e9 * 16e6; near("one 16M-token step of Llama 3 405B, FLOPs", fl / 1e19, 3.888, 1e-9);
  claim("nn-same-loop", `about \\(${Math.round(fl / 1e19)} \\times 10^{19}\\) floating-point operations (6 × 405 billion × 16 million)`);
  equal("the spirals plate's batch is all 200 points", SP.B, 200);
  claim("nn-same-loop", "with one fixed batch of all 200 points and a constant learning rate");
}

{ // sheets referred to by name
  const S = (id) => window.Deck.slides.find((x) => x.id === id), at = (id) => window.Deck.slides.indexOf(S(id));
  ok("nn-bp-relu names nn-bp-2layer", S("nn-bp-2layer").title.startsWith("Backpropagation through the 2-3-1 network"));
  claim("nn-bp-relu", "the sheet Backpropagation through the 2-3-1 network shows where it comes from");
  ok("nn-autograd names nn-bp-graph, in chapter 3.3", S("nn-bp-graph").title.startsWith("The backward pass multiplies local derivatives") && S("nn-bp-graph").chapter.code === "3.3");
  claim("nn-autograd", "the graph plate of chapter 3.3, The backward pass multiplies local derivatives");
  ok("nn-adam names nn-schedule", S("nn-schedule").title.startsWith("Our learning rate")); claim("nn-adam", "explained on the sheet Our learning rate");
  ok("nn-rmsprop: Adam is the next sheet; nn-adam: the bowl is on the previous sheet", at("nn-adam") === at("nn-rmsprop") + 1);
  claim("nn-rmsprop", "Adam (next sheet)"); claim("nn-adam", "on the previous sheet's bowl");
}
{ // the nn-depth plates' numbers, recomputed from the same random draws (mulberry32(2024), width 128, batch 64)
  const N = 128, B = 64, L = 10, rand = TinyLM.mulberry32(2024);
  const Ws = Array.from({ length: L }, () => Array.from({ length: N * N }, () => NN3.gauss(rand)));
  const X = Array.from({ length: B * N }, () => NN3.gauss(rand)), Gtop = Array.from({ length: B * N }, () => NN3.gauss(rand));
  const sd = (a) => { const m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
  const run = (f, df, gain) => {
    const sc = gain / Math.sqrt(N), zs = [], fwd = [], bwd = []; let hc = X, sat = 0;
    for (let l = 0; l < L; l++) {
      const z = new Array(B * N).fill(0);
      for (let b = 0; b < B; b++) for (let i = 0; i < N; i++) { const v = hc[b * N + i] * sc; for (let j = 0; j < N; j++) z[b * N + j] += v * Ws[l][i * N + j]; }
      zs.push(z); fwd.push(sd(z)); hc = z.map(f);
      sat += hc.filter((a) => a < 0.02 || a > 0.98).length;
    }
    let gh = Gtop;
    for (let l = L - 1; l >= 0; l--) {
      const gz = gh.map((g, k) => g * df(zs[l][k]));
      bwd[l] = sd(gz);
      gh = new Array(B * N).fill(0);
      for (let b = 0; b < B; b++) for (let i = 0; i < N; i++) { let acc = 0; for (let j = 0; j < N; j++) acc += gz[b * N + j] * Ws[l][i * N + j]; gh[b * N + i] = acc * sc; }
    }
    return { fwd, bwd, sat: sat / (L * B * N) };
  };
  const sg = (z) => 1 / (1 + Math.exp(-z));
  const cases = [["sigmoid", 4, sg, (z) => sg(z) * (1 - sg(z))], ["sigmoid", 1, sg, (z) => sg(z) * (1 - sg(z))], ["relu", 1, relu, (z) => (z > 0 ? 1 : 0)], ["relu", 2, relu, (z) => (z > 0 ? 1 : 0)]];
  for (const [act, gi, f, df] of cases) {
    const mine = run(f, df, NN3.GAINS[gi]), plate = NN3.depthStats(act)[gi];
    ok(`${act}, g = ${NN3.GAINS[gi].toFixed(3)}: plate forward and backward = recomputation`, mine.fwd.every((x, l) => Math.abs(x / plate.fwd[l + 1] - 1) < 1e-9) && mine.bwd.every((x, l) => Math.abs(x / plate.bwd[l + 1] - 1) < 1e-9));
    if (act === "sigmoid" && gi === 4) {
      near("σ = 1 line: saturated fraction = recomputation", plate.sat, mine.sat, 1e-12);
      ok(`σ = 1 line: ${Math.round(100 * mine.sat)}% saturated, and its backward ratio ${(mine.bwd[0] / mine.bwd[9]).toFixed(2)} stays near 1`, mine.sat > 0.5 && mine.bwd[0] / mine.bwd[9] > 0.5 && mine.bwd[0] / mine.bwd[9] < 2);
    }
  }
  near("σ = 1 weights over g = 1 weights", NN3.GAINS[4] / NN3.GAINS[1], 11.3, 0.05);
  ok("nn-depth note: 11 times the g = 1 line", window.Deck.figs["nn-depth"].toString().includes("its weights are 11 times those of the g = 1 line"));
}

done();
