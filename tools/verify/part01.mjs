// Part 1: recompute every number the math sheets state and check the text says it.
import { claim, near, equal, ok, r, int, TinyLM, Deck, done } from "./lib.mjs";
import { strings } from "../lib/deck-env.mjs";

const C = TinyLM.config;
const sq = (x) => x * x;

// ---------------------------------------------------------------- TeX lint for this part
// The deck renders TeX as native MathML. The core (ui.js fixMathML) adds the space after
// function names and removes binary spacing from unary minus; Part 1 also braces unary
// minus ({-2}), puts a thin space before a function name that follows a symbol, and keeps
// trailing punctuation inside inline math so a line cannot break before it.
{
  const mine = Deck.slides.filter((s) => s.part.n === 1);
  let n = 0;
  for (const sl of mine) {
    for (const [where, str] of strings(sl)) {
      const tex = [...str.matchAll(/\\\(([\s\S]+?)\\\)|\\\[([\s\S]+?)\\\]/g)].map((m) => m[1] || m[2]);
      if (where === "math" || where.startsWith("math[")) tex.push(str);
      for (const t of tex) {
        if (/\\(ln|log|cos|sin|exp|max|min)(_\{?\w+\}?)?\\,/.test(t)) ok(`${sl.id} ${where}: \\, after a function name doubles the space the core adds, in "${t}"`, false);
        if (/(^|[=(,\[]|\\approx|\\times|\\cdot|\\lt|\\gt|\\le|\\ge|\\to|\\pm)\s*-\d/.test(t)) ok(`${sl.id} ${where}: unbraced unary minus in "${t}"`, false);
        if (/[=(,]\s*\\pm/.test(t)) ok(`${sl.id} ${where}: unbraced unary \\pm in "${t}"`, false);
        if (/(^|[=(,]|\\to)\s*-\s*(\\[A-Za-z]|[A-Za-z])/.test(t)) ok(`${sl.id} ${where}: unbraced unary minus before a symbol in "${t}"`, false);
        n++;
      }
      if (/\\\)[,.;]/.test(str)) ok(`${sl.id} ${where}: punctuation after inline math`, false);
      if (/^fig\.(caption|items\[\d+\]\.note)$/.test(where) && /\\\(/.test(str) && !/^<span>/.test(str)) ok(`${sl.id} ${where}: caption with TeX must be wrapped in <span>`, false);
    }
  }
  ok("TeX lint ran", n > 0);
}

// ---------------------------------------------------------------- opener and model facts used in Part 1
near("params", C.params, 9968);
claim("math-part", int(C.params));
near("vocab", C.V, 113); near("context", C.T, 8); near("d", C.d, 16); near("layers", C.L, 2);

// ---------------------------------------------------------------- 1.1 Functions
{
  const f = (x) => (x - 3) ** 2;
  near("f(1)", f(1), 4); near("f(3)", f(3), 0);
  claim("math-function", "(1 - 3)^2 = 4");
  const table = [0, 1, 2, 3, 4, 5, 6].map(f);
  equal("table f(0..6)", table, [9, 4, 1, 0, 1, 4, 9]);
  for (const v of table) claim("math-function", String(v));
  claim("math-function", "up to 8 token ids"); claim("math-function", "113 probabilities"); claim("math-function", "9,968 parameters");

  const line = (w, b) => (x) => w * x + b;
  equal("2x+1 at 0..3", [0, 1, 2, 3].map(line(2, 1)), [1, 3, 5, 7]);
  claim("math-linear", "y = 1, 3, 5, 7");
  near("rise over run", (line(2, 1)(3) - line(2, 1)(1)) / (3 - 1), 2);
  claim("math-linear", "(7 - 3)/(3 - 1) = 2");
  near("zero crossing w=-0.5 b=3", -3 / -0.5, 6);
  near("line at 6", line(-0.5, 3)(6), 0);
  claim("math-linear", "crosses \\(y = 0\\) at \\(x = 6.\\)");

  const g = (x) => 3 * x + 1, F = (u) => u * u;
  near("g(2)", g(2), 7); near("f(g(2))", F(g(2)), 49); near("g(f(2))", g(F(2)), 13);
  claim("math-compose", "u = 3 \\cdot 2 + 1 = 7"); claim("math-compose", "f(7) = 49"); claim("math-compose", "g(f(2)) = 3 \\cdot 4 + 1 = 13");
  claim("math-compose", "16 numbers per token"); claim("math-compose", "2 layers"); claim("math-compose", "113 probabilities");

  const f2 = (x, y) => x * x + 3 * y * y;
  near("f(1,1)", f2(1, 1), 4); near("f(2,0)", f2(2, 0), 4); near("f(0,0)", f2(0, 0), 0);
  claim("math-surface", "f(1, 1) = 1 + 3 = 4"); claim("math-surface", "f(2, 0) = 4");
  near("sqrt(4)", Math.sqrt(4), 2); near("sqrt(4/3)", +r(Math.sqrt(4 / 3), 3), 1.155);
  claim("math-surface", "\\pm 1.155");

  // five functions
  claim("math-recurring", "e = 2.71828"); near("e", +r(Math.E, 5), 2.71828);
  claim("math-recurring", "multiplies it by 2.718");
  near("ln 0.1", +r(Math.log(0.1), 3), -2.303); claim("math-recurring", "0.1 = {-2.303}");
  const sig = (x) => 1 / (1 + Math.exp(-x));
  near("sigma(0)", sig(0), 0.5);
  near("sigma(2)", +r(sig(2), 3), 0.881); claim("math-recurring", "\\sigma(2) = 0.881"); claim("math-recurring", "σ(2) = 0.881");
  near("sigma(-2)", +r(sig(-2), 3), 0.119); claim("math-recurring", "\\sigma({-2}) = 0.119"); claim("math-recurring", "σ(−2) = 0.119");
  claim("math-recurring", "3^2 = 9"); claim("math-recurring", "10^2 = 100");
  near("e^-1", +r(Math.exp(-1), 3), 0.368); claim("math-recurring", "e⁻¹ = 0.368");
  near("e^2", +r(Math.exp(2), 3), 7.389); claim("math-recurring", "e² = 7.389");
  near("ln 7.389", +r(Math.log(7.389), 3), 2.0); claim("math-recurring", "ln 7.389 = 2.000");
  claim("math-recurring", "ln 0.1 = −2.303");
  // check: f(g(-1)) with g = x + 2, f = ln
  near("ln(g(-1))", Math.log(-1 + 2), 0);
}

// ---------------------------------------------------------------- 1.2 Derivatives
{
  // secant slopes of x^2
  near("secant 3..5", (sq(5) - sq(3)) / (5 - 3), 8); claim("math-secant", "(25 - 9)/(5 - 3) = 16/2 = 8");
  near("secant 3..4", (sq(4) - sq(3)) / 1, 7); claim("math-secant", "(16 - 9)/1 = 7");
  for (const [a, b] of [[3, 5], [-1, 6], [0.3, 2.7]]) near(`secant = x1 + x2 at ${a},${b}`, (sq(b) - sq(a)) / (b - a), a + b, 1e-12);
  // limit table: ((3+h)^2 - 9)/h = 6 + h
  for (const [h, want] of [[1, "7"], [0.1, "6.1"], [0.01, "6.01"], [0.001, "6.001"]]) {
    const v = (sq(3 + h) - 9) / h;
    near(`secant slope h=${h}`, v, 6 + h, 1e-9);
    claim("math-limit", `<td class="r">${want}</td>`.replace(/<[^>]+>/g, " ").trim());
  }
  equal("2x at 1, 3, -2", [1, 3, -2].map((x) => 2 * x), [2, 6, -4]);
  claim("math-limit", "2 at \\(x = 1,\\) 6 at 3, −4 at −2");
  // tangent of x^2 at 3
  near("tangent at 3.1", 9 + 6 * 0.1, 9.6, 1e-12); near("3.1^2", sq(3.1), 9.61, 1e-12);
  near("gap at 0.1", sq(3.1) - 9.6, 0.01, 1e-12); near("gap at 0.01", sq(3.01) - (9 + 6 * 0.01), 0.0001, 1e-12);
  claim("math-tangent", "9 + 6 \\cdot 0.1 = 9.6"); claim("math-tangent", "3.1^2 = 9.61"); claim("math-tangent", "The gap is 0.01"); claim("math-tangent", "0.0001 at a distance of 0.01");
  // rules on 3x^2 + 2x
  const f = (x) => 3 * x * x + 2 * x, df = (x) => 6 * x + 2;
  near("f'(1)", df(1), 8); near("f(1.001)", f(1.001), 5.008003, 1e-12); near("f(1)", f(1), 5);
  near("forward diff at 1", +r((f(1.001) - f(1)) / 0.001, 3), 8.003);
  near("excess = 3h", (f(1.001) - f(1)) / 0.001 - 8, 3 * 0.001, 1e-9);
  for (const t of ["5.008003", "8.003", "6x + 2", "That is 8".replace("That is", "that is")]) claim("math-rules", t);
  // e^x and ln x
  near("(e^1.001 - e)/0.001", +r((Math.exp(1.001) - Math.E) / 0.001, 4), 2.7196); claim("math-exp-ln", "2.7196");
  near("e to 4dp", +r(Math.E, 4), 2.7183); claim("math-exp-ln", "e = 2.7183");
  near("(ln 2.001 - ln 2)/0.001", +r((Math.log(2.001) - Math.log(2)) / 0.001, 4), 0.4999); claim("math-exp-ln", "= 0.4999");
  near("1/0.1", 1 / 0.1, 10); claim("math-exp-ln", "At \\(x = 0.1\\) the slope is 10");
  // finite differences of e^x at 1
  const F32 = Math.fround;
  const FD = {
    f32: (h) => { const x = F32(1), hh = F32(h); return F32(F32(F32(Math.exp(F32(x + hh))) - F32(Math.exp(x))) / hh); },
    c32: (h) => { const x = F32(1), hh = F32(h); return F32(F32(F32(Math.exp(F32(x + hh))) - F32(Math.exp(F32(x - hh)))) / F32(2 * hh)); },
    f64: (h) => (Math.exp(1 + h) - Math.exp(1)) / h,
    c64: (h) => (Math.exp(1 + h) - Math.exp(1 - h)) / (2 * h),
  };
  claim("math-finite-diff", r(Math.E, 6));
  const fw1 = FD.f64(0.1), ce1 = FD.c64(0.1), fw2 = FD.f64(0.05), ce2 = FD.c64(0.05);
  for (const v of [fw1, fw1 - Math.E, fw2 - Math.E, ce1, ce1 - Math.E, ce2 - Math.E]) claim("math-finite-diff", r(v, 6));
  ok("forward error about halves", Math.abs((fw1 - Math.E) / (fw2 - Math.E) - 2) < 0.1);
  ok("central error divides by 4", Math.abs((ce1 - Math.E) / (ce2 - Math.E) - 4) < 0.05);
  // round-off: same grid as the figure
  const H = Array.from({ length: 241 }, (_, i) => Math.pow(10, -8 + (8 * i) / 240));
  // the curves are jagged, so describe the smoothed minimum: geometric mean over 15 grid points (half a decade)
  const smin = (fn) => {
    const e = H.map((h) => Math.max(Math.abs(fn(h) - Math.E), 1e-16));
    let b = [0, Infinity];
    for (let i = 7; i < H.length - 7; i++) { let s = 0; for (let j = -7; j <= 7; j++) s += Math.log10(e[i + j]); if (s / 15 < b[1]) b = [H[i], s / 15]; }
    return b;
  };
  const bf32 = smin(FD.f32), bc32 = smin(FD.c32), bf64 = smin(FD.f64), bc64 = smin(FD.c64);
  ok(`float32 forward dip at h ${bf32[0]} near 2e-4`, bf32[0] >= 1.5e-4 && bf32[0] <= 3.5e-4);
  ok(`float32 forward dip error 10^${bf32[1]} about 1e-3`, bf32[1] > -3.5 && bf32[1] < -2.5);
  ok(`float32 central dip at h ${bc32[0]} near 3e-3`, bc32[0] >= 2e-3 && bc32[0] <= 4.5e-3);
  ok(`float32 central dip error 10^${bc32[1]} near 1e-5`, bc32[1] > -5.5 && bc32[1] < -4.5);
  const lowerF = bf32[1] - bf64[1], lowerC = bc32[1] - bc64[1];
  ok(`float64 forward ${lowerF.toFixed(2)} orders lower: four to six`, lowerF > 3.5 && lowerF < 6.5);
  ok(`float64 central ${lowerC.toFixed(2)} orders lower: four to six`, lowerC > 3.5 && lowerC < 6.5);
  claim("math-roundoff", "near \\(h = 2 \\times 10^{-4},\\) where its error is about \\(10^{-3};\\)");
  claim("math-roundoff", "near \\(h = 3 \\times 10^{-3},\\) with an error near \\(10^{-5}.\\)");
  claim("math-roundoff", "four to six orders of magnitude lower");
  near("machine eps float32", +(Math.pow(2, -23)).toPrecision(3), 1.19e-7); claim("math-roundoff", "2^{-23} \\approx 1.19 \\times 10^{-7}");
  ok("1 + 2^-23 is the next float32 after 1", F32(1 + Math.pow(2, -23)) === 1 + Math.pow(2, -23) && F32(1 + Math.pow(2, -24)) === 1);
  near("2^-24", +(Math.pow(2, -24)).toPrecision(1), 6e-8); claim("math-roundoff", "2^{-24} \\approx 6 \\times 10^{-8}");
  ok("forward estimate is 0 below 2^-24", FD.f32(5e-8) === 0 && FD.f32(1e-8) === 0);
  near("machine eps float64", +(Math.pow(2, -52)).toPrecision(2), 2.2e-16); ok("Number.EPSILON", Number.EPSILON === Math.pow(2, -52));
  near("float32 digits", +Math.log10(Math.pow(2, 24)).toFixed(1), 7.2); claim("math-roundoff", "about 7 significant digits");
  near("float64 digits", +Math.log10(Math.pow(2, 53)).toFixed(1), 16.0); claim("math-roundoff", "about 16 digits");
  // sign of the derivative on (x - 3)^2 + 1
  const g = (x) => sq(x - 3) + 1, dg = (x) => 2 * (x - 3);
  near("g'(1)", dg(1), -4); near("g(1.1)", g(1.1), 4.61, 1e-12); near("g(1)", g(1), 5); near("g'(5)", dg(5), 4); near("g(3)", g(3), 1); near("g'(3)", dg(3), 0);
  for (const t of ["f' = {-4}", "f(1.1) = 4.61", "f(1) = 5", "f' = 4", "f(3) = 1"]) claim("math-minimum", t);
  // second derivative on x^3 - 3x
  const c = (x) => x ** 3 - 3 * x, dc = (x) => 3 * x * x - 3, ddc = (x) => 6 * x;
  near("c'(1)", dc(1), 0); near("c'(-1)", dc(-1), 0); near("c''(1)", ddc(1), 6); near("c''(-1)", ddc(-1), -6); near("c(1)", c(1), -2); near("c(-1)", c(-1), 2);
  for (const t of ["f'' = 6", "f(1) = {-2}", "f'' = {-6}", "f({-1}) = 2"]) claim("math-second", t);
  // the step-size limit on (x - 3)^2 + 1: x - 3 <- (1 - 2 eta)(x - 3), stable iff eta < 2/f'' = 1
  for (const eta of [0.1, 0.9, 1.1]) { const x1 = 5 - eta * dg(5); near(`factor 1 - 2eta at eta=${eta}`, (x1 - 3) / (5 - 3), 1 - 2 * eta, 1e-12); }
  claim("math-second", "1 - 2\\eta"); claim("math-second", "\\eta \\lt 2/f''");
  // check question: f = x^2 - 4x
  const q = (x) => x * x - 4 * x;
  near("q'(1)", 2 * 1 - 4, -2); near("q(1.1)", q(1.1), -3.19, 1e-12); near("q(1)", q(1), -3); near("argmin", 4 / 2, 2);
  claim("math-second", "f(1.1) = {-3.19}");
}

// ---------------------------------------------------------------- 1.3 The chain rule
{
  // (3x + 1)^2
  const y = (x) => sq(3 * x + 1);
  for (const x of [-1, 0, 2, 2.5]) near(`chain = expanded at ${x}`, 2 * (3 * x + 1) * 3, 18 * x + 6, 1e-12);
  for (const x of [-1, 0, 2]) near(`expansion at ${x}`, y(x), 9 * x * x + 6 * x + 1, 1e-12);
  near("u at 2", 3 * 2 + 1, 7); near("dy/du at 2", 2 * 7, 14); near("dy/dx at 2", 14 * 3, 42);
  near("forward diff at 2", +r((y(2.001) - 49) / 0.001, 3), 42.009);
  for (const t of ["18x + 6", "9x^2 + 6x + 1", "u = 7,", "dy/du = 14,", "du/dx = 3,", "dy/dx = 42.", "= 42.009"]) claim("math-chain-rule", t);
  // three stages at x = 1
  const u = 3 * 1 + 1, v = u * u, yy = Math.log(v);
  near("u", u, 4); near("v", v, 16); near("ln 16", +r(yy, 4), 2.7726);
  near("local slopes product", 3 * (2 * u) * (1 / v), 1.5, 1e-15); near("6/(3x+1)", 6 / 4, 1.5);
  near("numeric slope of ln((3x+1)^2) at 1", (Math.log(sq(3 * 1.00001 + 1)) - Math.log(sq(3 * 0.99999 + 1))) / 2e-5, 1.5, 1e-8);
  for (const t of ["u = 4,", "v = 16", "\\ln 16 = 2.7726,", "2u = 8", "1/v = 0.0625.", "3 \\times 8 \\times 0.0625 = 1.5.", "6/4 = 1.5."]) claim("math-chain-three", t);
  // the graph: y = ln((w x + b)^2), w = 3, x = 1, b = 1
  const w = 3, x = 1, b = 1, m = w * x, uu = m + b, vv = uu * uu;
  const gv = 1 / vv, gu = gv * 2 * uu, gm = gu, gb = gu, gw = gm * x, gx = gm * w;
  near("m", m, 3); near("u", uu, 4); near("v", vv, 16);
  near("dy/dv", gv, 0.0625); near("dy/du", gu, 0.5); near("dy/db", gb, 0.5); near("dy/dw", gw, 0.5); near("dy/dx", gx, 1.5);
  for (const t of ["m = 3,", "u = 4,", "v = 16,", "y = 2.7726.", "1 \\times 0.0625 \\times 8 = 0.5", "0.5 \\times x = 0.5", "0.5 \\times w = 1.5,"]) claim("math-graph", t);
  // two paths: x e^x
  const f = (t) => t * Math.exp(t);
  near("2e", +r(2 * Math.E, 4), 5.4366); near("central diff of x e^x", +r((f(1.001) - f(0.999)) / 0.002, 4), 5.4366);
  near("paths add to (1 + x) e^x", Math.exp(1) + 1 * Math.exp(1), 2 * Math.E, 1e-15);
  claim("math-two-paths", "2e = 5.4366."); claim("math-two-paths", "/0.002 = 5.4366.");
  // why: finite differences vs backprop on our model; parameter groups
  const shapes = Object.fromEntries(C.shapes.map(([n, s]) => [n, s.reduce((a, b) => a * b, 1)]));
  const sum = (ns) => ns.reduce((a, n) => a + shapes[n], 0);
  const emb = sum(["wte", "wpe"]), lay = (l) => sum(["n1", "wq", "wk", "wv", "wo", "n2", "w1", "w2"].map((k) => `l${l}.${k}`)), outp = sum(["nf", "wu"]);
  near("embed params", emb, 1936); near("layer 0 params", lay(0), 3104); near("layer 1 params", lay(1), 3104); near("out params", outp, 1824);
  near("groups sum", emb + lay(0) + lay(1) + outp, C.params);
  for (const t of ["1,936 params", "3,104 params", "1,824 params", "113 probabilities", "1,936 derivatives", "3,104 derivatives", "1,824 derivatives", "1,936 + 3,104 + 3,104 + 1,824 = 9,968"]) claim("math-chain-why", t);
  near("forward-difference passes", C.params + 1, 9969); near("central-difference passes", 2 * C.params, 19936);
  claim("math-chain-why", int(C.params + 1) + " forward passes"); claim("math-chain-why", int(2 * C.params) + " with central ones");
  claim("math-chain-why", "16 numbers for each token"); claim("math-chain-why", "Each of its 2 layers");
  // check: (2x + 1)^3 at 1
  near("check (2x+1)^3", 3 * sq(2 * 1 + 1) * 2, 54);
}

// ---------------------------------------------------------------- 1.4 Partial derivatives and the gradient
{
  const f = (x, y) => x * x + 3 * y * y, grad = (x, y) => [2 * x, 6 * y];
  near("f(1.001,1)", f(1.001, 1), 4.002001, 1e-12); near("f(1,1.001)", f(1, 1.001), 4.006003, 1e-12);
  near("fwd dx", +r((f(1.001, 1) - 4) / 0.001, 3), 2.001); near("fwd dy", +r((f(1, 1.001) - 4) / 0.001, 3), 6.003);
  for (const t of ["4.002001", "(4.002001 - 4)/0.001 = 2.001;", "4.006003", "gives 6.003."]) claim("math-partial", t);
  equal("grad(1,1)", grad(1, 1), [2, 6]); equal("grad(2,-1)", grad(2, -1), [4, -6]); equal("grad(0,0)", grad(0, 0).map(Math.abs), [0, 0]);
  claim("math-gradient", "\\((2, 6);\\)"); claim("math-gradient", "\\((4, {-6});\\)"); claim("math-gradient", "\\((2, {-1})\\)"); claim("math-gradient", "(2x,\\ 6y)");
  // steepest: ratio of slopes and the level-curve tangent at (1, 1)
  near("6/2", 6 / 2, 3); claim("math-steepest", "3 times faster");
  near("tangent (3,-1) is perpendicular to (2,6)", 2 * 3 + 6 * -1, 0); near("level through (1,1)", f(1, 1), 4);
  { const c = 4, t = Math.atan2(1 / Math.sqrt(c / 3), 1 / Math.sqrt(c)); const T = [-Math.sqrt(c) * Math.sin(t), Math.sqrt(c / 3) * Math.cos(t)]; near("tangent direction ∝ (3,-1)", T[0] / T[1], -3, 1e-12); }
  claim("math-steepest", "x^2 + 3y^2 = 4,"); claim("math-steepest", "direction \\((3, {-1}),\\)");
  // directional derivative at (1, 1)
  const D = (deg) => { const a = (deg * Math.PI) / 180; return 2 * Math.cos(a) + 6 * Math.sin(a); };
  near("D at 0", D(0), 2, 1e-12); near("D at 90", D(90), 6, 1e-12);
  const peak = (Math.atan2(6, 2) * 180) / Math.PI;
  near("peak angle", +r(peak, 2), 71.57); near("max slope", +r(Math.hypot(2, 6), 3), 6.325); near("D at peak", D(peak), Math.hypot(2, 6), 1e-12);
  near("zero at peak + 90", D(peak + 90), 0, 1e-12); near("161.57", +r(peak + 90, 2), 161.57);
  for (const t of ["71.57^\\circ,", "= 6.325,", "161.57^\\circ"]) claim("math-directional", t);
  // our model: partial derivatives of the query loss w.r.t. the 16 numbers of " name"
  const ids = TinyLM.runningIds.slice(0, C.T);
  const lossOf = (W) => { const tr = TinyLM.forward(ids, undefined, { weights: W }); let s2 = 0; for (let i = 0; i < C.T; i++) s2 -= Math.log(tr.probs[i][TinyLM.runningIds[i + 1]]); return s2 / C.T; };
  const partials = (step, tok) => {
    const W = TinyLM.weights(step), data = Float64Array.from(W.wte.data), Wp = Object.assign({}, W, { wte: { shape: W.wte.shape, data } }), g = [];
    for (let j = 0; j < C.d; j++) { const k = tok * C.d + j, v0 = data[k]; data[k] = v0 + 1e-4; const lp = lossOf(Wp); data[k] = v0 - 1e-4; const lm = lossOf(Wp); data[k] = v0; g.push((lp - lm) / 2e-4); }
    return g;
  };
  const nameId = TinyLM.idOf(" name");
  near("name id", nameId, 93);
  const m6000 = Math.max(...partials(6000, nameId).map(Math.abs)), m400 = Math.max(...partials(400, nameId).map(Math.abs));
  near("largest partial at 6000", +r(m6000, 4), 0.0007); near("largest partial at 400", +r(m400, 4), 0.0750);
  ok(`ratio ${m400 / m6000} about 100`, m400 / m6000 > 80 && m400 / m6000 < 125);
  claim("math-many", "the largest of the 16 is 0.0007"); claim("math-many", "the largest is 0.0750, about 100 times");
  { const W = TinyLM.weights(6000), data = Float64Array.from(W.wte.data), Wp = Object.assign({}, W, { wte: { shape: W.wte.shape, data } }); const k = nameId * C.d + 2, v0 = data[k];
    const est = (hh) => { data[k] = v0 + hh; const a = lossOf(Wp); data[k] = v0 - hh; const b = lossOf(Wp); data[k] = v0; return (a - b) / (2 * hh); };
    near("estimate independent of h (1e-3 vs 1e-5)", est(1e-3), est(1e-5), 1e-8); }
  near("passes for 16", 2 * 16, 32); claim("math-many", "cost 32 forward passes"); claim("math-many", "would cost " + int(2 * C.params));
  claim("math-many", "over its 8 predictions");
  // check: f = x^2 y at (2, 3)
  near("check d/dx", 2 * 2 * 3, 12); near("check d/dy", 2 * 2, 4);
}

// ---------------------------------------------------------------- 1.5 Vectors
{
  const W = TinyLM.weights(), row = (id) => Array.from(TinyLM.row(W.wte, id)), prow = (t) => Array.from(TinyLM.row(W.wpe, t));
  const dot = (a, b) => a.reduce((s2, v, i) => s2 + v * b[i], 0), nrm = (a) => Math.sqrt(dot(a, a));
  const name = row(93);
  equal("token 93 is ' name'", TinyLM.vocab[93], " name");
  for (const [i, want] of [[1, "-0.302"], [6, "1.416"], [10, "-1.455"]]) { equal(`name entry ${i}`, r(name[i - 1], 3), want); claim("math-vector", `= ${want.replace("-", "{-")}${want.startsWith("-") ? "}" : ""}`); }
  claim("math-vector", "each of its 113 tokens as a vector of 16 numbers");
  claim("math-vector", "4,096 numbers per token in Llama 3 8B"); // sourced: Meta llama-models sku_list.py, dim 4096
  // vector arithmetic in 2-D
  const add = (a, b) => a.map((v, i) => v + b[i]), scale = (c, a) => a.map((v) => c * v);
  equal("(2,1)+(1,3)", add([2, 1], [1, 3]), [3, 4]); equal("2(2,1)", scale(2, [2, 1]), [4, 2]); equal("-1(2,1)", scale(-1, [2, 1]), [-2, -1]); equal("0.5(2,1)", scale(0.5, [2, 1]), [1, 0.5]);
  for (const t of ["(2, 1) + (1, 3) = (3, 4).", "2 \\cdot (2, 1) = (4, 2)", "0.5 \\cdot (2, 1) = (1, 0.5)"]) claim("math-vec-ops", t);
  // token + position for " name" at position 1, entries 2..4: the rounded terms add up to the rounded sum
  const pos1 = prow(1);
  for (const j of [2, 3, 4]) {
    const a = name[j - 1], b = pos1[j - 1], c = a + b;
    near(`rounded sum consistent, entry ${j}`, +r(a, 3) + +r(b, 3), +r(c, 3), 1e-9);
    const tex = (v) => (v < 0 ? `{-${r(-v, 3)}}` : r(v, 3));          // braced unary minus, as the sheet writes it
    claim("math-vec-ops", `${tex(a)} + ${b < 0 ? `(${tex(b)})` : tex(b)} = ${tex(c)}`);
  }
  near("x0 of the model = token + position", TinyLM.forward(TinyLM.runningIds.slice(0, 8)).x0[1][1], name[1] + pos1[1], 1e-12);
  // dot product
  near("(2,1).(1,3)", dot([2, 1], [1, 3]), 5); near("|a|", +r(Math.sqrt(5), 3), 2.236); near("|b|", +r(Math.sqrt(10), 3), 3.162);
  near("cos", +r(5 / Math.sqrt(50), 4), 0.7071); near("theta", (Math.acos(5 / Math.sqrt(50)) * 180) / Math.PI, 45, 1e-9);
  for (const t of ["2 \\cdot 1 + 1 \\cdot 3 = 5.", "\\sqrt{5} = 2.236", "\\sqrt{10} = 3.162,", "5/\\sqrt{50} = 0.7071", "\\theta = 45^\\circ."]) claim("math-dot", t);
  near("(2,6).(3,-1)", dot([2, 6], [3, -1]), 0);
  // cosine similarity of real token vectors
  const v7 = row(TinyLM.idOf(" 7")), v3 = row(TinyLM.idOf(" 3"));
  near("|7|", +r(nrm(v7), 3), 4.614); near("|name|", +r(nrm(name), 3), 2.874);
  near("cos(7,3)", +r(dot(v7, v3) / (nrm(v7) * nrm(v3)), 2), 0.99); near("cos(7,name)", +r(dot(v7, name) / (nrm(v7) * nrm(name)), 2), -0.35);
  claim("math-cosine", "has length 4.614"); claim("math-cosine", "2.874."); claim("math-cosine", "cosine similarity 0.99,"); claim("math-cosine", "have −0.35.");
  // all nine digits point nearly the same way
  const digits = [1, 2, 3, 4, 5, 6, 7, 8, 9].map((k) => row(TinyLM.idOf(" " + k)));
  let minCos = 1;
  for (let i = 0; i < 9; i++) for (let j = i + 1; j < 9; j++) minCos = Math.min(minCos, dot(digits[i], digits[j]) / (nrm(digits[i]) * nrm(digits[j])));
  let maxCos = -1, maxOther = -1;
  for (let i = 0; i < 9; i++) for (let j = i + 1; j < 9; j++) maxCos = Math.max(maxCos, dot(digits[i], digits[j]) / (nrm(digits[i]) * nrm(digits[j])));
  for (let id = 0; id < C.V; id++) { if (/^ [1-9]$/.test(TinyLM.vocab[id])) continue; const v = row(id); for (const d of digits) maxOther = Math.max(maxOther, dot(v, d) / (nrm(v) * nrm(d))); }
  near("digit pairs: smallest cosine", +r(minCos, 2), 0.87); near("digit pairs: largest cosine", +r(maxCos, 2), 0.99); near("closest non-digit", +r(maxOther, 2), 0.67);
  claim("math-cosine", "between 0.87 and 0.99; no other token comes closer than 0.67");
  // stacking
  near("9 x 16", 9 * 16, 144); near("113 x 16", C.V * C.d, 1808); equal("wte shape", W.wte.shape, [113, 16]);
  claim("math-stack", "9 × 16, holding 144 numbers"); claim("math-stack", "1,808 numbers"); claim("math-stack", "index 93");
  near("check (1,2).(4,-2)", dot([1, 2], [4, -2]), 0);
}

// ---------------------------------------------------------------- 1.6 Matrices
{
  const shapes = Object.fromEntries(C.shapes.map(([n, s2]) => [n, s2]));
  const size = (n) => shapes[n].reduce((a, b) => a * b, 1);
  const L = [0, 1];
  near("wte", size("wte"), 1808); near("wpe", size("wpe"), 128);
  near("attention", L.reduce((a, l) => a + ["wq", "wk", "wv", "wo"].reduce((b, k) => b + size(`l${l}.${k}`), 0), 0), 2048);
  near("w1", L.reduce((a, l) => a + size(`l${l}.w1`), 0), 2048); near("w2", L.reduce((a, l) => a + size(`l${l}.w2`), 0), 2048);
  equal("w1 shape", shapes["l0.w1"], [16, 64]); equal("w2 shape", shapes["l0.w2"], [64, 16]); equal("wu shape", shapes["wu"], [16, 113]);
  near("wu", size("wu"), 1808); near("norm gains", ["l0.n1", "l0.n2", "l1.n1", "l1.n2", "nf"].reduce((a, n) => a + size(n), 0), 80);
  near("all parameters", C.shapes.reduce((a, [n]) => a + size(n), 0), 9968);
  for (const t of ["113 × 16", "1,808", "8 × 16", "128", "8 × (16 × 16)", "2,048", "2 × (16 × 64)", "2 × (64 × 16)", "16 × 113", "5 × 16", "80", "9,968"]) claim("math-matrix", t);
  const A = [[1, 2, 0], [0, 1, 3]];
  near("A12", A[0][1], 2); near("A23", A[1][2], 3); claim("math-matrix", "A_{12} = 2"); claim("math-matrix", "A_{23} = 3.");
  const mv = (M2, x) => M2.map((row) => row.reduce((s2, v, j) => s2 + v * x[j], 0));
  equal("Ax", mv(A, [1, 2, 1]), [5, 5]); claim("math-matvec", "1 \\cdot 1 + 2 \\cdot 2 + 0 \\cdot 1 = 5"); claim("math-matvec", "0 \\cdot 1 + 1 \\cdot 2 + 3 \\cdot 1 = 5,"); claim("math-matvec", "(5, 5).");
  near("16x16 multiplications", 16 * 16, 256); claim("math-matvec", "256 multiplications");
  // transformations
  const rot = (deg) => { const a = (deg * Math.PI) / 180; return [[Math.cos(a), -Math.sin(a)], [Math.sin(a), Math.cos(a)]]; };
  const e1r = mv(rot(30), [1, 0]); near("cos 30", +r(e1r[0], 3), 0.866); near("sin 30", e1r[1], 0.5, 1e-12);
  claim("math-transform", "(0.866, 0.5).");
  equal("shear (1,1)", mv([[1, 1], [0, 1]], [1, 1]), [2, 1]); claim("math-transform", "(1, 1) \\mapsto (2, 1).");
  equal("scale (1,1)", mv([[2, 0], [0, 0.5]], [1, 1]), [2, 0.5]);
  // products
  const mm = (P, Q) => P.map((row) => Q[0].map((_, j) => row.reduce((s2, v, t) => s2 + v * Q[t][j], 0)));
  const B = [[1, 0], [2, 1], [1, 2]];
  equal("AB", mm(A, B), [[5, 2], [5, 7]]); claim("math-matmul", "C_{11} = 1 \\cdot 1 + 2 \\cdot 2 + 0 \\cdot 1 = 5.");
  equal("BA shape", [mm(B, A).length, mm(B, A)[0].length], [3, 3]); claim("math-matmul", "(3 \\times 2)(2 \\times 3) = 3 \\times 3.");
  // FLOPs and entries
  const row = (m, k, n) => ({ fl: 2 * m * n * k, en: m * k + k * n + m * n });
  const fmt = (x) => { const e = Math.floor(Math.log10(x)); const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹"; return `${(x / 10 ** e).toFixed(2)} × 10${String(e).split("").map((c) => SUP[+c]).join("")}`; };
  const ours = row(8, 16, 113), l1 = row(1, 4096, 128256), l4k = row(4096, 4096, 128256), sqr4 = row(4096, 4096, 4096);
  near("our FLOPs", ours.fl, 28928); near("our entries", ours.en, 2840);
  claim("math-flops", "2 \\cdot 8 \\cdot 113 \\cdot 16 = 28{,}928"); claim("math-flops", "28,928"); claim("math-flops", "2,840"); claim("math-flops", r(ours.fl / ours.en, 1));
  claim("math-flops", fmt(l1.fl)); claim("math-flops", fmt(l1.en)); claim("math-flops", r(l1.fl / l1.en, 1));
  claim("math-flops", "1.05 \\times 10^9");
  claim("math-flops", fmt(l4k.fl)); claim("math-flops", fmt(l4k.en)); claim("math-flops", int(l4k.fl / l4k.en));
  claim("math-flops", fmt(sqr4.fl)); claim("math-flops", fmt(sqr4.en)); claim("math-flops", int(sqr4.fl / sqr4.en));
  ok("about 2 FLOPs per number for one token", Math.abs(l1.fl / l1.en - 2) < 0.05); ok("about 4,000 for 4,096 tokens", Math.abs(l4k.fl / l4k.en - 4000) < 100);
  claim("math-flops", "about 2 FLOPs per number"); claim("math-flops", "about 4,000");
  // transpose
  const At = A[0].map((_, j) => A.map((row2) => row2[j]));
  equal("AAt", mm(A, At), [[5, 2], [2, 10]]); claim("math-transpose", "\\begin{pmatrix} 5 & 2 \\\\ 2 & 10 \\end{pmatrix}");
  // batch
  near("batch rows", 128 * 8, 1024); near("batch FLOPs", 2 * 1024 * 16 * 113, 3702784); near("batch size from config", C.batch, 128);
  claim("math-batch", "128 \\times 8 = 1{,}024"); claim("math-batch", "3,702,784 FLOPs"); claim("math-batch", "28,928 FLOPs");
  // the real product: Z = X W_U reproduces the model's logits
  const tr = TinyLM.forward(TinyLM.runningIds.slice(0, 8)), Wu = TinyLM.weights().wu;
  let maxErr = 0;
  for (let i = 0; i < 8; i++) for (let j = 0; j < 113; j++) { let z = 0; for (let t = 0; t < 16; t++) z += tr.final[i][t] * Wu.data[t * 113 + j]; maxErr = Math.max(maxErr, Math.abs(z - tr.logits[i][j])); }
  ok(`X W_U equals the logits (max error ${maxErr})`, maxErr < 1e-9);
  near("check FLOPs", 2 * 8 * 16 * 16, 4096); near("check multiplications", 8 * 16 * 16, 2048);
}

// ---------------------------------------------------------------- 1.7 Probability
{
  const Q = TinyLM.allQueries();
  near("270 queries", Q.length, 270);
  const byTable = (t) => Q.filter((q) => q.t === t);
  near("users count", byTable("users").length, 108); near("products count", byTable("products").length, 54);
  near("users query p", byTable("users")[0].p, 1 / 324, 1e-15); near("products query p", byTable("products")[0].p, 1 / 162, 1e-15);
  for (const t of ["users", "orders", "products"]) near(`p(${t}) total`, byTable(t).reduce((a, q) => a + q.p, 0), 1 / 3, 1e-12);
  near("all queries sum to 1", Q.reduce((a, q) => a + q.p, 0), 1, 1e-12);
  claim("math-distribution", "1/324,"); claim("math-distribution", "108/324 = 1/3."); claim("math-distribution", "1/162"); claim("math-distribution", "270 queries");
  const tr = TinyLM.forward(TinyLM.runningIds.slice(0, C.T)), p0 = tr.probs[0];
  const cols = [" id", " name", " email", " age", " user_id", " total", " status", " title", " price"];
  const nine = cols.reduce((a, t) => a + p0[TinyLM.idOf(t)], 0);
  near("model mass on 9 columns", +r(nine, 5), 0.99984); near("remaining mass", +r(1 - nine, 5), 0.00016);
  claim("math-distribution", "0.99984"); claim("math-distribution", "0.00016"); near("other tokens", C.V - 9, 104); claim("math-distribution", "the other 104 tokens");
  // conditional: the grammar's distribution after SELECT, from the enumeration
  const after = {};
  for (const q of Q) { const c1 = " " + q.c1; after[c1] = (after[c1] || 0) + q.p; }
  near("p(name|SELECT)", after[" name"], 1 / 12, 1e-12); near("p(id|SELECT)", after[" id"], 5 / 18, 1e-12); near("p(title|SELECT)", after[" title"], 1 / 9, 1e-12);
  near("nine columns sum", Object.values(after).reduce((a, b) => a + b, 0), 1, 1e-12); near("nine columns", Object.keys(after).length, 9);
  near("1/12", +r(1 / 12, 4), 0.0833); near("5/18", +r(5 / 18, 4), 0.2778);
  claim("math-conditional", "1/12 = 0.0833."); claim("math-conditional", "5/18 = 0.2778.");
  near("model p(name|SELECT)", +r(p0[TinyLM.idOf(" name")], 4), 0.0854); near("model p(id|SELECT)", +r(p0[TinyLM.idOf(" id")], 4), 0.2806);
  claim("math-conditional", "0.0854 for"); claim("math-conditional", "0.2806 for");
  // users after SELECT name FROM: every query with c1 = name has t = users
  ok("only users has name", Q.filter((q) => q.c1 === "name").every((q) => q.t === "users"));
  // sequence probability
  const ids = TinyLM.runningIds, mp = ids.slice(1).map((id, i) => tr.probs[i][id]);
  const prodModel = mp.reduce((a, b) => a * b, 1);
  near("grammar p(query)", +r(1 / 324, 6), 0.003086); near("model p(query)", +r(prodModel, 6), 0.003182);
  near("running query in the enumeration", Q.find((q) => q.text === TinyLM.data.running).p, 1 / 324, 1e-15);
  claim("math-seq-prob", "\\(= 1/324 = 0.003086.\\)"); claim("math-seq-prob", "0.003182.");
  near("model p(name)", +r(mp[0], 4), 0.0854); near("model p(id)", +r(mp[4], 4), 0.346); near("model p(7)", +r(mp[6], 4), 0.1077);
  // expectation
  near("E[digit]", [1, 2, 3, 4, 5, 6, 7, 8, 9].reduce((a, v) => a + v / 9, 0), 5, 1e-12);
  near("E[columns]", +r((4 + 4 + 3) / 3, 3), 3.667); near("E[filter columns]", +r(11 / 3 - 1, 3), 2.667);
  near("E[columns] from the enumeration", Q.reduce((a, q) => a + q.p * TinyLM.data.schema[q.t].length, 0), 11 / 3, 1e-12);
  claim("math-expectation", "45/9 = 5."); claim("math-expectation", "11/3 = 3.667,"); claim("math-expectation", "8/3 = 2.667.");
  near("exact loss", +r(TinyLM.exactLoss().mean, 4), 0.6942); claim("math-expectation", "0.6942 per token, measured in nats");
  // sampling
  const cum = []; let acc = 0; for (const t of cols) { cum.push([acc, acc + after[t]]); acc += after[t]; }
  near("id ends at", +r(cum[0][1], 4), 0.2778); near("name ends at", +r(cum[1][1], 4), 0.3611);
  ok("u = 0.3 falls in name's segment", 0.3 >= cum[1][0] && 0.3 < cum[1][1]);
  claim("math-sampling", "\\([0, 0.2778)\\)"); claim("math-sampling", "\\([0.2778, 0.3611),\\)");
  // law of large numbers
  const sigma = Math.sqrt([1, 2, 3, 4, 5, 6, 7, 8, 9].reduce((a, v) => a + (v - 5) ** 2 / 9, 0));
  near("sigma", +r(sigma, 3), 2.582); near("sigma^2 = 80/12", sigma * sigma, 80 / 12, 1e-12);
  near("sigma/sqrt(10)", +r(sigma / Math.sqrt(10), 2), 0.82); near("sigma/sqrt(100)", +r(sigma / 10, 2), 0.26); near("sigma/sqrt(10000)", +r(sigma / 100, 3), 0.026);
  claim("math-lln", "\\sigma = 2.582,"); claim("math-lln", "typically off by 0.82 after 10 draws, 0.26 after 100 and 0.026 after 10,000");
  for (const seed of [1, 2, 3]) {
    const rnd = TinyLM.mulberry32(seed); let sum = 0;
    for (let n = 1; n <= 10000; n++) sum += 1 + Math.floor(rnd() * 9);
    ok(`seed ${seed}: average ${(sum / 10000).toFixed(4)} within 0.015 of 5`, Math.abs(sum / 10000 - 5) < 0.015);
  }
  claim("math-lln", "within 0.015 of 5");
  // the averages after 10 draws, for the three seeds of the plate
  const at10 = [1, 2, 3].map((seed) => { const rnd = TinyLM.mulberry32(seed); let sum = 0; for (let k = 0; k < 10; k++) sum += 1 + Math.floor(rnd() * 9); return r(sum / 10, 1); });
  equal("averages after 10 draws", at10, ["5.9", "5.1", "4.1"]); claim("math-lln", "After 10 draws the averages are 5.9, 5.1 and 4.1;");
  // variances of independent draws add: the average of 100 draws has standard deviation sigma/10
  { const rnd = TinyLM.mulberry32(11), T = 4000, avg = [];
    for (let t = 0; t < T; t++) { let sum = 0; for (let k = 0; k < 100; k++) sum += 1 + Math.floor(rnd() * 9); avg.push(sum / 100); }
    const mu = avg.reduce((a, b) => a + b, 0) / T, sd = Math.sqrt(avg.reduce((a, b) => a + (b - mu) ** 2, 0) / (T - 1));
    ok(`std of 100-draw averages ${sd.toFixed(4)} is sigma/10 = ${(sigma / 10).toFixed(4)} within 5%`, Math.abs(sd / (sigma / 10) - 1) < 0.05); }
  claim("math-lln", "a sum of \\(n\\) draws has variance \\(n\\sigma^2\\)"); near("batch", C.batch, 128); claim("math-lln", "128 random queries");
  // the digit is independent of the earlier choices
  const usersNameId = Q.filter((q) => q.t === "users" && q.c1 === "name" && q.c2 === "id");
  near("p(7 | users, name, id)", usersNameId.filter((q) => q.v === "7").reduce((a, q) => a + q.p, 0) / usersNameId.reduce((a, q) => a + q.p, 0), 1 / 9, 1e-12);
  // check
  near("check: products title price 3", Q.find((q) => q.text === "SELECT title FROM products WHERE price = 3;").p, 1 / 162, 1e-15);
}

// ---------------------------------------------------------------- 1.8 Exponentials and logarithms
{
  near("e^5", +r(Math.exp(5), 1), 148.4); near("e^10", Math.round(Math.exp(10)), 22026); near("e^-10", +Math.exp(-10).toPrecision(3), 0.0000454);
  claim("math-exp-log", "e^{5} = 148.4,"); claim("math-exp-log", "e^{10} = 22{,}026."); claim("math-exp-log", "e^{-10} = 0.0000454,");
  near("ln 2", +r(Math.log(2), 4), 0.6931); near("ln 10", +r(Math.log(10), 4), 2.3026); near("ln 0.1", +r(Math.log(0.1), 4), -2.3026);
  claim("math-exp-log", "2 = 0.6931\\)"); claim("math-exp-log", "10 = 2.3026.\\)"); claim("math-exp-log", "0.1 = {-2.3026}.");
  near("log2 via ln", Math.log(8) / Math.log(2), 3, 1e-12); claim("math-exp-log", "one bit is");
  // log-probability of the query
  const g = [1 / 12, 1 / 3, 1 / 9].map(Math.log);
  near("ln 1/12", +r(g[0], 4), -2.4849); near("ln 1/3", +r(g[1], 4), -1.0986); near("ln 1/9", +r(g[2], 4), -2.1972);
  near("grammar sum", +r(g[0] + g[1] + g[2], 4), -5.7807); near("= ln 1/324", g[0] + g[1] + g[2], Math.log(1 / 324), 1e-12);
  claim("math-log-sum", "\\(= {-2.4849} - 1.0986 - 2.1972\\) \\(= {-5.7807},\\)"); claim("math-log-sum", "−2.4849"); claim("math-log-sum", "−1.0986"); claim("math-log-sum", "−2.1972");
  const ids = TinyLM.runningIds, tr = TinyLM.forward(ids.slice(0, C.T)), lp = ids.slice(1).map((id, i) => Math.log(tr.probs[i][id]));
  near("ln p(name)", +r(lp[0], 4), -2.4605); near("ln p(id)", +r(lp[4], 4), -1.0613); near("ln p(7)", +r(lp[6], 4), -2.2283);
  const others = [1, 2, 3, 5, 7].map((i) => lp[i]);
  ok("five other model terms plus SELECT round to 0 at 4 dp", others.every((v) => Math.abs(v) < 0.00005));
  near("largest small term is users", others.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a)), lp[2]);
  near("users term", +lp[2].toFixed(6), -0.000008);
  const total = lp.reduce((a, b) => a + b, 0);
  near("model total", +r(total, 4), -5.7501); near("e^total", +r(Math.exp(total), 6), 0.003182); near("e^-5.7501", +r(Math.exp(-5.7501), 6), 0.003182);
  near("per-token loss", +r(-total / 8, 4), 0.7188);
  for (const t of ["{-2.4605} - 1.0613 - 2.2283", "−0.000008", "−5.7501", "e^{-5.7501} = 0.003182,", "0.7188 nats", "six terms"]) claim("math-log-sum", t);
  // underflow
  const F = Math.fround;
  near("2^-126", +Math.pow(2, -126).toPrecision(3), 1.18e-38); near("2^-149", +Math.pow(2, -149).toPrecision(2), 1.4e-45); near("2^-1022", +Math.pow(2, -1022).toPrecision(2), 2.2e-308);
  ok("2^-126 is float32's smallest normal: halving it stays nonzero but loses bits", F(Math.pow(2, -126)) === Math.pow(2, -126) && F(Math.pow(2, -127)) === Math.pow(2, -127) && F(Math.pow(2, -149)) === Math.pow(2, -149) && F(Math.pow(2, -151)) === 0);
  ok("2^-1074 is float64's smallest subnormal", Number.MIN_VALUE === Math.pow(2, -1074));
  claim("math-underflow", "2^{-126} \\approx 1.18 \\times 10^{-38};"); claim("math-underflow", "2^{-149} \\approx 1.4 \\times 10^{-45}."); claim("math-underflow", "2^{-1022} \\approx 2.2 \\times 10^{-308},");
  let x = F(1), below = null, zero = null; const q = F(0.1);
  for (let i = 1; i <= 100; i++) { x = F(x * q); if (below === null && x < Math.pow(2, -126)) below = i; if (zero === null && x === 0) zero = i; }
  near("float32 below smallest normal after", below, 38); near("float32 zero after", zero, 46);
  let y = 1, zero64 = null; for (let i = 1; i <= 400; i++) { y *= 0.1; if (zero64 === null && y === 0) zero64 = i; }
  near("float64 zero after", zero64, 324);
  claim("math-underflow", "after 38 factors"); claim("math-underflow", "the 46th gives exactly 0"); claim("math-underflow", "after 324 factors.");
  // round to nearest: values above 2^-150 round up to 2^-149; 2^-150 itself (a tie) rounds to 0
  ok("fround(1e-45) = 2^-149", F(1e-45) === Math.pow(2, -149)); ok("fround(2^-150) = 0", F(Math.pow(2, -150)) === 0); ok("just above 2^-150 rounds up", F(Math.pow(2, -150) * (1 + Math.pow(2, -20))) === Math.pow(2, -149));
  near("2^-150", +Math.pow(2, -150).toPrecision(2), 7.0e-46); claim("math-underflow", "2^{-150} \\approx 7.0 \\times 10^{-46},");
  { let y = F(1); for (let i = 1; i <= 44; i++) y = F(y * q); const raw45 = y * q; near("45th product before rounding", +raw45.toPrecision(2), 9.8e-46); ok("45th product rounds up to 2^-149", F(raw45) === Math.pow(2, -149)); ok("46th product is 0", F(F(raw45) * q) === 0); }
  claim("math-underflow", "9.8 \\times 10^{-46},"); claim("math-underflow", "rounds up to \\(2^{-149};\\)");
  near("1000 ln 0.1", +r(1000 * Math.log(0.1), 3), -2302.585);
  let acc = F(0); const l = F(Math.log(0.1)); for (let i = 0; i < 1000; i++) acc = F(acc + l);
  near("float32 sum of logs", +r(acc, 3), -2302.592);
  claim("math-underflow", "{-2{,}302.585},"); claim("math-underflow", "{-2{,}302.592}.");
  // log-sum-exp
  near("ln(max float64)", +r(Math.log(Number.MAX_VALUE), 2), 709.78); near("ln(max float32)", +r(Math.log((2 - Math.pow(2, -23)) * Math.pow(2, 127)), 2), 88.72);
  ok("float32 max", F(3.4028234663852886e38) === 3.4028234663852886e38 && F(3.4028236e38) === Infinity);
  ok("e^710 overflows float64", Math.exp(710) === Infinity && Number.isFinite(Math.exp(709.78)));
  equal("max float64 to 2 digits", Number.MAX_VALUE.toPrecision(2), "1.8e+308");
  claim("math-lse", "1.8 \\times 10^{308},"); claim("math-lse", "z = 709.78;"); claim("math-lse", "z = 88.72."); claim("math-lse", "\\infty/\\infty,");
  ok("direct softmax of 1000..1002 is NaN", Number.isNaN(Math.exp(1000) / (Math.exp(1000) + Math.exp(1001) + Math.exp(1002))));
  const e3 = [-2, -1, 0].map(Math.exp), S = e3.reduce((a, b) => a + b, 0);
  near("e^-2", +r(e3[0], 4), 0.1353); near("e^-1", +r(e3[1], 4), 0.3679); near("sum", +r(S, 4), 1.5032);
  equal("probabilities", e3.map((v) => r(v / S, 4)), ["0.0900", "0.2447", "0.6652"]);
  near("lse", +r(1002 + Math.log(S), 4), 1002.4076);
  for (const t of ["0.1353 + 0.3679 + 1 = 1.5032.", "0.0900, 0.2447 and 0.6652", "= 1002.4076."]) claim("math-lse", t);
  near("check lse", +r(5 + Math.log(3), 4), 6.0986); claim("math-lse", "6.0986");
}

// ---------------------------------------------------------------- statements added after review
{
  // derivatives exist only when both one-sided limits agree: ReLU at 0
  const relu = (x) => Math.max(0, x);
  for (const h of [1e-3, 1e-6]) { near(`ReLU slope from the right, h=${h}`, (relu(h) - relu(0)) / h, 1, 1e-12); near(`ReLU slope from the left, h=${h}`, (relu(-h) - relu(0)) / -h, 0, 1e-12); }
  claim("math-limit", "the secant slopes are 1 from the right and 0 from the left");
  // local, not global, extrema of x^3 - 3x; f'' = 0 says nothing (x^3 vs x^4)
  const c = (x) => x ** 3 - 3 * x;
  near("x^3 - 3x at -3", c(-3), -18); ok("x = 1 is only a local minimum", c(-3) < c(1) && c(0.9) > c(1) && c(1.1) > c(1));
  claim("math-second", "f({-3}) = {-18}."); claim("math-second", "local minimum"); claim("math-second", "local maximum");
  const d1 = (f, x) => (f(x + 1e-4) - f(x - 1e-4)) / 2e-4, d2 = (f, x) => (f(x + 1e-3) - 2 * f(x) + f(x - 1e-3)) / 1e-6;
  for (const [nm, f] of [["x^3", (x) => x ** 3], ["x^4", (x) => x ** 4]]) { near(`${nm}: f'(0)`, d1(f, 0), 0, 1e-7); near(`${nm}: f''(0)`, d2(f, 0), 0, 1e-5); }
  ok("x^4 has a minimum at 0, x^3 does not", [-0.1, 0.1].every((x) => x ** 4 > 0) && (-0.1) ** 3 < 0 && 0.1 ** 3 > 0);
  claim("math-second", "\\(x^3\\) and \\(x^4\\) both have \\(f'(0) = f''(0) = 0,\\)");
  // equally spaced level curves; chart ranges cover every value the sliders reach
  claim("math-partial", "level curves at 2, 4, 6, 8, 10 and 12");
  const fb = (x, y) => x * x + 3 * y * y;
  near("largest f on the map", fb(3, 2), 21); ok("slice charts reach 22", fb(3, 2) <= 22);
  near("largest |grad f| on the map", +Math.hypot(6, 12).toFixed(2), 13.42); ok("direction chart reaches 14", Math.hypot(6, 12) <= 14);
  // the directional derivative is derived before steepest ascent is asserted
  { const order = Deck.slides.filter((x) => x.part.n === 1).map((x) => x.id); ok("math-directional comes before math-steepest", order.indexOf("math-directional") < order.indexOf("math-steepest") && order.indexOf("math-gradient") < order.indexOf("math-directional")); }
  claim("math-dot", "\\nabla f \\cdot \\mathbf{u} = |\\nabla f|");
  // the drawn central secant passes through both of its points
  for (const hh of [0.01, 0.3, 0.6]) {
    const slope = (Math.exp(1 + hh) - Math.exp(1 - hh)) / (2 * hh), line = (x) => Math.exp(1 - hh) + slope * (x - (1 - hh));
    near(`central secant through 1 - h (h=${hh})`, line(1 - hh), Math.exp(1 - hh), 1e-12); near(`central secant through 1 + h (h=${hh})`, line(1 + hh), Math.exp(1 + hh), 1e-12);
  }
  // the batch product is small: FLOPs per number read or written
  const m = 1024, k = 16, n = 113, per = (2 * m * k * n) / (m * k + k * n + m * n);
  near("FLOPs per number for 1,024 x 16 x 113", Math.round(per), 28); claim("math-batch", "about 28 FLOPs per number");
  // log-sum-exp: underflow for very negative scores, overflow of the sum
  ok("e^-1000 underflows to 0", Math.exp(-1000) === 0 && Math.exp(-1001) === 0 && Math.exp(-1002) === 0);
  ok("direct softmax of -1000..-1002 is 0/0", Number.isNaN(Math.exp(-1000) / (Math.exp(-1000) + Math.exp(-1001) + Math.exp(-1002))));
  ok("ln 0 = -inf", Math.log(0) === -Infinity);
  ok("float64 e^z reaches 0 near z = -745.1", Math.exp(-745.2) === 0 && Math.exp(-745.1) > 0 && Math.abs(-1075 * Math.LN2 - -745.1) < 0.05);
  claim("math-lse", "z \\approx {-745.1}"); claim("math-lse", "\\ln 0 = {-\\infty}.");
  { const zs = [-1000, -1001, -1002], mm = Math.max(...zs), S = zs.reduce((a, z) => a + Math.exp(z - mm), 0); near("stable lse of -1000..-1002", mm + Math.log(S), -1000 + Math.log(1 + Math.exp(-1) + Math.exp(-2)), 1e-12); ok("stable lse is finite", Number.isFinite(mm + Math.log(S))); }
  ok("each e^709.5 is finite but their sum overflows", Number.isFinite(Math.exp(709.5)) && 3 * Math.exp(709.5) === Infinity);
  // signed area ad - bc of the plate's examples
  const det = ([[a, b], [cc, d]]) => a * d - b * cc, r30 = Math.PI / 6;
  for (const [nm, A] of [["shear", [[1, 1], [0, 1]]], ["scale", [[2, 0], [0, 0.5]]], ["rotate 30", [[Math.cos(r30), -Math.sin(r30)], [Math.sin(r30), Math.cos(r30)]]], ["identity", [[1, 0], [0, 1]]]]) near(`det ${nm}`, det(A), 1, 1e-12);
  near("det flatten", det([[1, 2], [0.5, 1]]), 0); near("det of a flip", det([[-1, 0], [0, 1]]), -1);
  claim("math-transform", "1 for each example above"); claim("math-transform", "0 for the flatten example");
  // the logarithm of a square, written unambiguously
  claim("math-chain-three", "\\ln\\big((3x + 1)^2\\big)");
  near("ln((3x+1)^2) = 2 ln(3x+1) at x = 1", Math.log((3 + 1) ** 2), 2 * Math.log(3 + 1), 1e-12);
}

done();
