// Part 2: recomputes every number stated on the "Learning from data" sheets and
// checks that the sheet text contains it. Where a sheet describes what a figure
// shows, the figure is also rendered in jsdom and its readouts are compared.
import { claim, near, equal, ok, r, int, TinyLM, Deck, window, done } from "./lib.mjs";

const C = TinyLM.config;
const doc = window.document;
const round = (x, d) => r(x, d).replace(/^-/, "−"); // as written in prose, true minus sign

// ---------------------------------------------------------------- helpers: render a figure and read it
function mount(id) {
  const sl = Deck.slides.find((s) => s.id === id);
  const el = doc.createElement("div");
  doc.body.appendChild(el);
  Deck.figs[sl.fig.type](el, sl.fig, { slide: sl, strip: { set() {}, reset() {}, stage() {} } });
  return el;
}
const tilesOf = (el) => [...el.querySelectorAll(".f-tile")].map((t) => [t.querySelector(".f-tile-l").textContent.trim(), t.querySelector(".f-tile-v").textContent.trim()]);
const tileV = (el, i) => el.querySelectorAll(".f-tile-v")[i].textContent.trim().replace(/−/g, "-");
const press = (el, label) => { const b = [...el.querySelectorAll("button")].find((x) => x.textContent.trim() === label); if (!b) throw new Error("no button " + label); b.click(); };
const setSlider = (el, i, v) => { const inp = el.querySelectorAll("input[type=range]")[i]; inp.value = String(v); inp.dispatchEvent(new window.Event("input")); };

// ---------------------------------------------------------------- the five points
const X = [1, 2, 3, 4, 5], Y = [3, 3, 6, 7, 8], n = 5;
const sum = (a) => a.reduce((p, q) => p + q, 0);
const L = (w, b) => sum(X.map((x, i) => (w * x + b - Y[i]) ** 2)) / n;
const G = (w, b) => [sum(X.map((x, i) => (2 * (w * x + b - Y[i]) * x) / n)), sum(X.map((x, i) => (2 * (w * x + b - Y[i])) / n))];
const Sx = sum(X), Sy = sum(Y), Sxx = sum(X.map((x) => x * x)), Sxy = sum(X.map((x, i) => x * Y[i])), Syy = sum(Y.map((y) => y * y));
equal("sums", [Sx, Sy, Sxx, Sxy, Syy], [15, 27, 55, 95, 167]);
// least squares by the normal equations, solved with Cramer's rule
const det = Sxx * n - Sx * Sx;
const wStar = (Sxy * n - Sx * Sy) / det, bStar = (Sxx * Sy - Sx * Sxy) / det;
near("w*", wStar, 1.4, 1e-12); near("b*", bStar, 1.2, 1e-12);
const Lstar = L(wStar, bStar);
near("L*", Lstar, 0.32, 1e-12);
const resid = X.map((x, i) => wStar * x + bStar - Y[i]);
near("residuals sum", sum(resid), 0, 1e-12);

// ---------------------------------------------------------------- 2.1
claim("learn-part", "0.6937");
near("2x at 3", 2 * 3 + 0, 6); claim("learn-model", String.raw`it predicts \(\hat y = 6\) at \(x = 3\)`);
near("1.4x+1.2 at 3", 1.4 * 3 + 1.2, 5.4, 1e-12); claim("learn-model", "it predicts 5.4");
near("params", C.params, 9968); claim("learn-model", "9,968 parameters"); claim("learn-model", "9,968");
{
  const tr = TinyLM.forward(TinyLM.runningIds.slice(0, C.T));
  equal("output shape", [tr.probs.length, tr.probs[0].length], [8, 113]);
  claim("learn-model", "8 × 113 probabilities"); claim("learn-model", "8 probability distributions over 113 tokens");
}
claim("learn-model", "175.0 billion"); claim("learn-model", "up to 2,048 token ids"); // Brown et al. 2020, Table 2.1 and Section 2.1

ok("no line through all five points", Math.min(...resid.map(Math.abs)) > 0);
near("smallest miss", Math.min(...resid.map(Math.abs)), 0.2, 1e-12); near("largest miss", Math.max(...resid.map(Math.abs)), 1.0, 1e-12);
claim("learn-data", String.raw`\(\hat y = 1.4x + 1.2\)`); claim("learn-data", "by 0.2 to 1.0");

near("lr", C.lr, 0.01); near("warmup", C.warmup, 100); near("batch", C.batch, 128); near("steps", C.steps, 6000); near("layers", C.L, 2); near("d", C.d, 16);
near("final lr = lr × 0.1 (tools/build_model.py)", C.lr * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI))), 0.001, 1e-15);
claim("learn-hyper", "a learning rate of 0.01, reached after 100 warmup steps and then lowered along a cosine curve to 0.001 (Part 3), with batches of 128 queries and 6,000 steps");
claim("learn-hyper", "learning rate 0.01, 128 queries per batch, 6,000 steps, 2 layers of width 16");
claim("learn-hyper", "3.2 million tokens"); claim("learn-hyper", "0.6 × 10⁻⁴"); claim("learn-hyper", "96 layers of width 12,288"); claim("learn-hyper", "300B tokens");

near("L(0,0)", L(0, 0), 33.4, 1e-12); equal("grad(0,0)", G(0, 0).map((v) => +v.toFixed(10)), [-38, -10.8]);
{
  const [gw, gb] = G(0, 0), w1 = 0 - 0.01 * gw, b1 = 0 - 0.01 * gb;
  near("w1", w1, 0.38, 1e-12); near("b1", b1, 0.108, 1e-12);
  near("L1", L(w1, b1), 19.639904, 1e-9);
  claim("learn-loop", "the loss is 33.4"); claim("learn-loop", "(−38, −10.8)"); claim("learn-loop", String.raw`\(w = 0.38,\ b = 0.108\)`); claim("learn-loop", `falls to ${r(L(w1, b1), 2)}`);
  claim("learn-loop", "L = mean (ŷ − y)² = 33.4"); claim("learn-loop", "w = 0.38, b = 0.108"); claim("learn-loop", "repeat: L = 19.64");
  claim("learn-loop", String.raw`\(w = 0 - 0.01 \times ({-38}) = 0.38\)`); claim("learn-loop", String.raw`\(b = 0 - 0.01 \times ({-10.8}) = 0.108\)`);
}

// ---------------------------------------------------------------- 2.2
{
  const e2x = X.map((x, i) => 2 * x - Y[i]);
  equal("errors of 2x", e2x, [-1, 1, 0, 1, 2]);
  claim("learn-error", "this line's errors are −1, 1, 0, 1, 2");
  claim("learn-error", String.raw`\(e = \hat y - y = 8 - 7 = 1\)`); claim("learn-error", String.raw`the prediction is 2 and \(e = {-1}\)`);
  const flat = X.map((x, i) => 5.4 - Y[i]);
  equal("flat errors", flat.map((v) => +v.toFixed(10)), [2.4, 2.4, -0.6, -1.6, -2.6]);
  near("flat sum", sum(flat), 0, 1e-12); near("best sum", sum(resid), 0, 1e-12);
  near("flat squares", sum(flat.map((v) => v * v)), 21.2, 1e-12); near("best squares", sum(resid.map((v) => v * v)), 1.6, 1e-12);
  claim("learn-squared", "by 2.4, 2.4, −0.6, −1.6 and −2.6, and these sum to exactly 0");
  claim("learn-squared", "squared errors sum to 21.2, the best line's to 1.6");
  claim("learn-squared", "an error of 2 costs 4, an error of 1 costs 1");
  // MSE sheet
  near("MSE 2x", L(2, 0), 1.4, 1e-12); near("MSE flat", L(0, 5.4), 4.24, 1e-12); near("sqrt 1.4", Math.sqrt(1.4), 1.1832, 1e-4);
  claim("learn-mse", "the sum 7 and the MSE \\(7/5 = 1.4\\)"); claim("learn-mse", "\\(21.2/5 = 4.24\\)"); claim("learn-mse", "scores 0.32"); claim("learn-mse", "Its square root, 1.18");
  const best = X.map((x) => 1.4 * x + 1.2), be2 = X.map((x, i) => (1.4 * x + 1.2 - Y[i]) ** 2);
  equal("best predictions", best.map((v) => r(v, 1)), ["2.6", "4.0", "5.4", "6.8", "8.2"]);
  equal("best e²", be2.map((v) => r(v, 2)), ["0.16", "1.00", "0.36", "0.04", "0.04"]);
  const tbl = Deck.slides.find((s) => s.id === "learn-mse").fig.rows;
  equal("MSE table", tbl.slice(0, 5).map((row) => row.slice(0, 6)), X.map((x, i) => [String(x), String(Y[i]), String(2 * x), String((2 * x - Y[i]) ** 2), r(best[i], 1), r(be2[i], 2)]));
  equal("MSE table sums", [tbl[5][3], tbl[5][5]], [String(sum(e2x.map((v) => v * v))), r(sum(be2), 2)]);
}
// MAE vs MSE, model y = w x
{
  const slopeMSE = (ys) => sum(X.map((x, i) => x * ys[i])) / Sxx;
  // MAE: brute force over a fine grid and over the data slopes (the optimum is at a data slope)
  const mae = (w, ys) => sum(X.map((x, i) => Math.abs(w * x - ys[i])));
  const slopeMAE = (ys) => { let best = null; for (const c of X.map((x, i) => ys[i] / x)) { const v = mae(c, ys); if (!best || v < best[1] - 1e-12) best = [c, v]; } return best[0]; };
  const uniqueMAE = (ys) => { const w = slopeMAE(ys), v = mae(w, ys); return [w - 1e-6, w + 1e-6].every((u) => mae(u, ys) > v + 1e-9); };
  near("MSE slope", slopeMSE(Y), 95 / 55, 1e-12); near("MAE slope", slopeMAE(Y), 1.75, 1e-12);
  claim("learn-mae", "the MSE-best slope is 1.727 and the MAE-best slope is 1.750");
  const y20 = Y.slice(); y20[4] = 20;
  near("MSE slope y5=20", slopeMSE(y20), 2.818, 5e-4); near("MAE slope y5=20", slopeMAE(y20), 2, 1e-12);
  claim("learn-mae", "the MSE slope rises to 2.818, while the MAE slope rises only to 2.000, reached at \\(y = 10\\)");
  let okStay = true, okUnique = true;
  for (let y5 = -5; y5 <= 1000; y5 += 0.5) { const ys = Y.slice(); ys[4] = y5; if (y5 >= 10 && Math.abs(slopeMAE(ys) - 2) > 1e-12) okStay = false; if (!uniqueMAE(ys)) okUnique = false; }
  ok("MAE slope stays 2 for y5 ≥ 10", okStay);
  { const ys = Y.slice(); ys[4] = 9.9; ok("MAE slope below 2 just under y5 = 10", slopeMAE(ys) < 2); }
  ok("MAE slope unique for every y5 on the slider and beyond", okUnique);
  // derivative of Σ|w xᵢ − yᵢ| in w is Σ xᵢ sign(eᵢ): independent of how far each point is
  { const ys = Y.slice(); ys[4] = 20; const d1 = sum(X.map((x, i) => x * Math.sign(1.9 * x - ys[i]))); ys[4] = 200; const d2 = sum(X.map((x, i) => x * Math.sign(1.9 * x - ys[i]))); near("MAE derivative ignores distance", d1, d2); }
  // every point, every slider position: figure's weighted-median rule equals brute force
  let agree = true;
  const el = mount("learn-mae");
  for (let k = 0; k < 5; k++) {
    el.querySelectorAll(".seg-btn")[k].click();
    for (let y = -5; y <= 25; y += 0.5) {
      setSlider(el, 0, y);
      const ys = Y.slice(); ys[k] = y;
      if (tileV(el, 0) !== r(slopeMSE(ys), 3) || tileV(el, 1) !== r(slopeMAE(ys), 3)) agree = false;
    }
  }
  ok("outlier figure: MSE and MAE slopes match brute force at every point and slider position", agree);
}
// L(w) for fixed b
{
  near("11", Sxx / 5, 11); near("38", (2 * Sxy) / 5, 38); near("33.4", Syy / 5, 33.4, 1e-12);
  claim("learn-curve", String.raw`\(\sum x_i^2 = 55\), \(\sum x_i y_i = 95\) and \(\sum y_i^2 = 167\)`);
  claim("learn-curve", String.raw`= 11\,w^2 - 38\,w + 33.4`);
  near("L(1,0)", L(1, 0), 6.4, 1e-12); near("slope at 1", G(1, 0)[0], -16, 1e-12);
  claim("learn-curve", String.raw`At \(w = 1\) the loss is 6.4`); claim("learn-curve", "is −16");
  const bw = (b) => (Sxy - b * Sx) / Sxx;
  near("best w b=0", bw(0), 38 / 22, 1e-12); near("L there", L(bw(0), 0), 0.5818, 1e-4);
  claim("learn-curve", String.raw`\(w = 38/22 = 1.727\), with \(L = 0.582\)`);
  near("best w b=1.2", bw(1.2), 1.4, 1e-12); near("L there", L(1.4, 1.2), 0.32, 1e-12); near("best w b=-2", bw(-2), 2.2727, 1e-4);
  claim("learn-curve", String.raw`At \(b = 1.2\) its bottom is at \(w = 1.4\) with \(L = 0.32\)`); claim("learn-curve", String.raw`at \(b = {-2}\) it is at \(w = 2.273\)`);
  // figure readouts at the default and at the slider extremes
  const el = mount("learn-curve");
  equal("curve figure default tiles", [0, 1, 2, 3].map((i) => tileV(el, i)), ["6.400", "-16.00", "1.727", "0.582"]);
  let fine = true;
  for (let w = 0; w <= 3; w += 0.25) for (let b = -2; b <= 4; b += 0.5) {
    setSlider(el, 0, w); setSlider(el, 1, b);
    const want = [[L(w, b), 3], [G(w, b)[0], 2], [bw(b), 3], [L(bw(b), b), 3]];
    if (want.some(([v, d], i) => !(Math.abs(+tileV(el, i) - v) <= 0.5 * 10 ** -d + 1e-9))) fine = false;
  }
  ok("curve figure correct over the slider grid", fine);
}
// the surface
{
  const a = Sxx / n, c = Sx / n, d = 1; // L − L* = 11 dw² + 6 dw db + db²
  const lmax = (a + d) / 2 + Math.sqrt(((a - d) / 2) ** 2 + c * c), lmin = (a + d) / 2 - Math.sqrt(((a - d) / 2) ** 2 + c * c);
  near("A eigen max", lmax, 6 + Math.sqrt(34), 1e-12); near("A eigen min", lmin, 6 - Math.sqrt(34), 1e-12);
  near("curvature ratio", lmax / lmin, 70, 0.05); claim("learn-surface", "differ by a factor of 70");
  const vb = -(lmin - a) / c; // direction (−1, vb): (a − λ)(−1) + c·vb = 0
  near("valley direction", vb, 3.61, 5e-4); claim("learn-surface", "change in the ratio −1 to 3.61"); claim("learn-surface", String.raw`pivots about \(x = 3.61\)`);
  // moving along (−1, vb) changes the prediction at x by −x + vb, zero at x = vb
  near("pivot", vb, 3.6103, 1e-4);
  // L − L* really is that quadratic form
  let qf = true;
  for (const [dw, db] of [[0.3, -1], [-2, 0.5], [1, 1]]) if (Math.abs(L(1.4 + dw, 1.2 + db) - 0.32 - (a * dw * dw + 2 * c * dw * db + d * db * db)) > 1e-9) qf = false;
  ok("L − L* = 11dw² + 6dw·db + db²", qf);
  claim("learn-surface", String.raw`at \(w = 1.4,\ b = 1.2\), where \(L = 0.32\)`);
  const el = mount("learn-surface");
  equal("surface explore default", [0, 1, 2].map((i) => tileV(el, i)), ["2.00", "0.00", r(L(2, 0), 3)]);
}
// normal equations
{
  claim("learn-normal", String.raw`55\,w + 15\,b = 95`); claim("learn-normal", String.raw`15\,w + 5\,b = 27`);
  near("from eq 2", 27 / 5, 5.4, 1e-12); near("10w = 14", 55 - 45, 10); near("rhs", 95 - 15 * 5.4, 14, 1e-12);
  claim("learn-normal", String.raw`b = 5.4 - 3w \;\Rightarrow\; 55\,w + 15\,(5.4 - 3w) = 95 \;\Rightarrow\; 10\,w = 14`);
  claim("learn-normal", String.raw`w = 1.4, \qquad b = 5.4 - 3 \times 1.4 = 1.2`);
  claim("learn-normal", String.raw`\tfrac{1}{5}\left(0.16 + 1 + 0.36 + 0.04 + 0.04\right) = 0.32`);
  claim("learn-normal", "\\(95 / 55 = 1.727\\)"); near("check answer", 95 / 55, 1.727, 5e-4);
  claim("learn-normal", "9,968 equations");
}

// ---------------------------------------------------------------- 2.3
{
  // the update on L(w) = 11w² − 38w + 33.4
  const Lw = (w) => L(w, 0), dLw = (w) => G(w, 0)[0];
  near("L'(0)", dLw(0), -38, 1e-12); near("step", -0.01 * dLw(0), 0.38, 1e-12);
  near("predicted drop", 0.01 * 38 ** 2, 14.44, 1e-12); near("predicted L", 33.4 - 14.44, 18.96, 1e-12);
  near("actual L(0.38)", Lw(0.38), 20.5484, 1e-9); near("actual drop", 33.4 - Lw(0.38), 12.8516, 1e-9); near("curvature term", 11 * 0.38 ** 2, 1.5884, 1e-12);
  claim("learn-update", "the slope is −38"); claim("learn-update", "a drop of 14.44, to 18.96"); claim("learn-update", "is 20.548, a drop of 12.852"); claim("learn-update", String.raw`\(11 \times 0.38^2 = 1.588\)`);
  claim("learn-update", String.raw`33.4 - 0.01 \times 38^2 = 33.4 - 14.44 = 18.96`); claim("learn-update", String.raw`L(0.38) = 20.548 = 18.96 + 11 \times 0.38^2`);
  claim("learn-update", String.raw`L'(0) = {-38}`);
}
{
  // three steps by hand on (w − 3)²
  const ws = [-1];
  for (let t = 0; t < 10; t++) ws.push(ws[t] - 0.1 * 2 * (ws[t] - 3));
  equal("w path", ws.slice(0, 5).map((v) => +v.toFixed(10)), [-1, -0.2, 0.44, 0.952, 1.3616]);
  equal("L path", ws.slice(0, 4).map((v) => +((v - 3) ** 2).toFixed(10)), [16, 10.24, 6.5536, 4.194304]);
  equal("slopes", ws.slice(0, 4).map((v) => +(2 * (v - 3)).toFixed(10)), [-8, -6.4, -5.12, -4.096]);
  const rows = ["0</td><td class=\"r\">−1</td><td class=\"r\">16</td><td class=\"r\">−8</td><td class=\"r\">−0.2", "1</td><td class=\"r\">−0.2</td><td class=\"r\">10.24</td><td class=\"r\">−6.4</td><td class=\"r\">0.44",
    "2</td><td class=\"r\">0.44</td><td class=\"r\">6.5536</td><td class=\"r\">−5.12</td><td class=\"r\">0.952", "3</td><td class=\"r\">0.952</td><td class=\"r\">4.1943</td><td class=\"r\">−4.096</td><td class=\"r\">1.3616"];
  const body = Deck.slides.find((s) => s.id === "learn-three").body;
  rows.forEach((rw, i) => ok(`three-steps table row ${i}`, body.includes(rw)));
  near("distance after 10", Math.abs(ws[10] - 3), 4 * 0.8 ** 10, 1e-12); near("≈0.43", 4 * 0.8 ** 10, 0.43, 5e-3);
  claim("learn-three", "4, 3.2, 2.56, 2.048"); claim("learn-three", String.raw`\(4 \times 0.8^{10} = 0.43\)`);
}
{
  // learning-rate regimes
  near("0.96^10", 0.96 ** 10, 0.66, 5e-3); claim("learn-lr", "factor 0.96; after 10 steps 66%");
  near("η=0.5", 1 - 2 * 0.5, 0); claim("learn-lr", "factor 0; the first step lands exactly on 3");
  near("η=0.9", 1 - 2 * 0.9, -0.8, 1e-12); claim("learn-lr", "factor −0.8");
  near("η=1.05", 1 - 2 * 1.05, -1.1, 1e-12); near("1.1^10", 1.1 ** 10, 2.6, 0.05); claim("learn-lr", "factor −1.1"); claim("learn-lr", "grown 2.6 times");
  near("L''", 2, 2); near("five-point L''", 2 * Sxx / n, 22); near("2/22", 2 / 22, 0.0909, 5e-5);
  claim("learn-lr", String.raw`Here \(L'' = 2\)`); claim("learn-lr", String.raw`\(L'' = 22\) and the limit drops to \(\eta \lt 0.0909\)`);
  // the figure: every slider position gives the closed form 4|1 − 2η|^10, and the right verdict
  const el = mount("learn-lr");
  let fine = true;
  for (let k = 1; k <= 120; k++) {
    const eta = k / 100;
    setSlider(el, 0, eta);
    const f = 1 - 2 * eta, e = 4 * Math.abs(f) ** 10;
    if (tileV(el, 0) !== r(f, 2).replace(/^-0\.00$/, "0.00")) fine = false;
    const shown = tileV(el, 1);
    if (e >= 1e-3 && shown !== r(e, 3)) fine = false;
    const v = tileV(el, 2);
    const want = f === 0 ? "lands on 3 in one step" : f > 0 ? "converges from one side" : f > -1 ? "overshoots, converges" : f === -1 ? "bounces forever" : "diverges";
    if (v !== want) fine = false;
  }
  ok("lr figure: factor, distance and verdict right for η = 0.01 … 1.20", fine);
}
{
  // gradient descent on w and b
  const run = (eta, N) => { let w = 0, b = 0; for (let t = 0; t < N; t++) { const g = G(w, b); w -= eta * g[0]; b -= eta * g[1]; } return [w, b]; };
  const stepsTo = (eta) => { let w = 0, b = 0, t = 0; while (L(w, b) > 0.32 * 1.01 && t < 1e6) { const g = G(w, b); w -= eta * g[0]; b -= eta * g[1]; t++; } return t; };
  const [w10, b10] = run(0.01, 10);
  claim("learn-gd2", `Ten steps reach \\((${r(w10, 3)}, ${r(b10, 3)})\\) with loss ${r(L(w10, b10), 3)}`);
  claim("learn-gd2", `after 100 steps the loss is ${r(L(...run(0.01, 100)), 3)}`);
  near("514", stepsTo(0.01), 514); claim("learn-gd2", "514 steps"); near("64", stepsTo(0.08), 64); claim("learn-gd2", "within 1% in 64 steps");
  { const [w, b] = run(0.085, 3000); ok("η = 0.085 diverges", !(Math.abs(w) < 1e6) || L(w, b) > 1e10); }
  claim("learn-gd2", "At 0.085 it diverges"); claim("learn-gd2", "\\((0.38, 0.108)\\), loss 19.64");
  // the figure after 10 and 100 steps
  const el = mount("learn-gd2");
  press(el, "10 steps");
  equal("descent figure after 10 steps", [tileV(el, 0), tileV(el, 1)], ["10", r(L(w10, b10), 4)]);
  press(el, "100 steps");
  equal("descent figure after 110 steps", [tileV(el, 0), tileV(el, 1)], ["110", r(L(...run(0.01, 110)), 4)]);
  setSlider(el, 0, 0.09); press(el, "100 steps"); press(el, "100 steps");
  ok("descent figure reports divergence at η = 0.09", tileV(el, 1) === "diverged");
}
{
  // narrow valley
  near("limit", 2 / 20, 0.1); near("w2 factor", 1 - 20 * 0.09, -0.8, 1e-12); near("w1 factor", 1 - 2 * 0.09, 0.82, 1e-12);
  near("w1 after 10", -4 * 0.82 ** 10, -0.55, 5e-3);
  claim("learn-valley", String.raw`\(\eta \lt 2/20 = 0.1\)`); claim("learn-valley", "multiplied by −0.8 per step"); claim("learn-valley", "multiplied by 0.82"); claim("learn-valley", "from −4 to −0.55");
  const h1 = 2 * (6 + Math.sqrt(34)), h2 = 2 * (6 - Math.sqrt(34));
  near("H max", h1, 23.66, 5e-3); near("H min", h2, 0.338, 5e-4); near("ratio", h1 / h2, 70, 0.05); near("limit 5pt", 2 / h1, 0.0845, 5e-5);
  claim("learn-valley", String.raw`\(2/23.66 = 0.0845\)`);
  // the Hessian of the 5-point MSE is (2/n)[[Σx², Σx], [Σx, n]]
  near("Hessian trace", h1 + h2, 2 * (Sxx + n) / n, 1e-12); near("Hessian det", h1 * h2, (4 / n ** 2) * (Sxx * n - Sx * Sx), 1e-12);
  equal("Hessian entries", [2 * Sxx / n, 2 * Sx / n, 2], [22, 6, 2]);
  near("12 + √136", 12 + Math.sqrt(136), h1, 1e-12); near("12 − √136", 12 - Math.sqrt(136), h2, 1e-12);
  claim("learn-valley", String.raw`\(L - 0.32 = 11\,dw^2 + 6\,dw\,db + db^2\)`); claim("learn-valley", String.raw`\(12 \pm \sqrt{136}\)`); claim("learn-valley", "curvatures 23.66 and 0.338, a ratio of 70");
}
{
  // minibatches
  const g = X.map((x, i) => 2 * (0 - Y[i]) * x);
  equal("per-example dℓ/dw at 0", g, [-6, -12, -36, -56, -80]);
  claim("learn-sgd", "are −6, −12, −36, −56 and −80");
  const pairs = [];
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) pairs.push((g[i] + g[j]) / 2);
  near("pairs", pairs.length, 10); near("max pair", Math.max(...pairs), -9); near("min pair", Math.min(...pairs), -68); near("mean pairs", sum(pairs) / 10, -38, 1e-12);
  near("distinct estimates of ∂L/∂w", new Set(pairs).size, 9);
  equal("the two that coincide", [(g[1] + g[4]) / 2, (g[2] + g[3]) / 2], [-46, -46]);
  claim("learn-sgd", String.raw`give 10 estimates of \(\partial L / \partial w\), from −9 to −68 (two coincide at −46), and their average is exactly −38`);
  near("queries", C.steps * C.batch, 768000); claim("learn-sgd", "768,000 queries"); claim("learn-sgd", "6,000 steps of 128");
  const el = mount("learn-sgd");
  for (let B = 1; B <= 5; B++) {
    el.querySelectorAll(".seg-btn")[B - 1].click();
    ok(`minibatch figure B=${B}: average of all minibatches is the full gradient`, tileV(el, 1) === "(-38.0, -10.8)");
  }
}
{
  // noise vs batch size: same sampler as the figure
  const perW = (w, b) => X.map((x, i) => 2 * (w * x + b - Y[i]) * x);
  const sd = (a) => { const m = sum(a) / a.length; return Math.sqrt(sum(a.map((v) => (v - m) ** 2)) / a.length); };
  const sample = (at, B) => { const gw = perW(...at.p); const rnd = TinyLM.mulberry32(1000 + B + (at.min ? 500 : 0)); const est = []; for (let k = 0; k < 2000; k++) { let s = 0; for (let j = 0; j < B; j++) s += gw[Math.floor(rnd() * 5)]; est.push(s / B); } return est; };
  const start = { p: [0, 0] }, min = { p: [1.4, 1.2], min: true };
  near("σ start", sd(perW(0, 0)), 27.54, 5e-3); claim("learn-noise", String.raw`\(\sigma = 27.54\)`);
  claim("learn-noise", `standard deviation ${r(sd(sample(start, 1)), 2)}`); claim("learn-noise", `at \\(B = 128\\), ${r(sd(sample(start, 128)), 2)}`);
  claim("learn-noise", `\\(27.54/\\sqrt{128} = ${r(sd(perW(0, 0)) / Math.sqrt(128), 2)}\\)`);
  equal("per-example at min", perW(1.4, 1.2).map((v) => +v.toFixed(9)), [-0.8, 4, -3.6, -1.6, 2]);
  claim("learn-noise", "−0.8, 4, −3.6, −1.6 and 2"); near("σ min", sd(perW(1.4, 1.2)), 2.69, 5e-3); claim("learn-noise", String.raw`\(\sigma = 2.69\)`);
  claim("learn-noise", "from 0.01 to 0.001");
  // sampling without replacement (the minibatch sheet): spread = σ/√B × √((n − B)/(n − 1)), checked on every subset
  {
    const gw = perW(0, 0), sg = sd(gw);
    let fpcOk = true;
    for (let B = 1; B <= 5; B++) {
      const means = [];
      const rec = (s, acc) => { if (acc.length === B) { means.push(sum(acc.map((i) => gw[i])) / B); return; } for (let i = s; i < 5; i++) rec(i + 1, acc.concat(i)); };
      rec(0, []);
      if (Math.abs(sd(means) - (sg / Math.sqrt(B)) * Math.sqrt((5 - B) / 4)) > 1e-9) fpcOk = false;
    }
    ok("finite-population factor matches every subset size", fpcOk);
    near("B=2 without replacement", (sg / Math.sqrt(2)) * Math.sqrt(3 / 4), 16.86, 5e-3); near("B=2 independent", sg / Math.sqrt(2), 19.47, 5e-3);
    claim("learn-noise", String.raw`\(\sqrt{(n - B)/(n - 1)}\): 16.86 instead of 19.47 at \(B = 2\), and 0 at \(B = n\)`);
    const elB = mount("learn-sgd");
    elB.querySelectorAll(".seg-btn")[1].click();
    near("minibatch figure spread at B = 2", +tileV(elB, 2), 16.86, 5e-3);
    elB.querySelectorAll(".seg-btn")[4].click();
    near("minibatch figure spread at B = 5", +tileV(elB, 2), 0, 1e-12);
  }
  // the measured spread follows σ/√B within 5% at every batch size, at both points
  let fine = true;
  for (const at of [start, min]) for (const B of [1, 2, 4, 8, 16, 32, 64, 128]) { const m = sd(sample(at, B)), th = sd(perW(...at.p)) / Math.sqrt(B); if (Math.abs(m / th - 1) > 0.05) fine = false; }
  ok("sampled std within 5% of σ/√B everywhere", fine);
  const el = mount("learn-noise");
  equal("noise figure default tiles", [0, 2, 3].map((i) => tileV(el, i)), ["-38.00", r(sd(sample(start, 1)), 3), r(sd(perW(0, 0)), 3)]);
}
{
  // training curves: same runs as the figure
  const run = ({ B, eta, seed, cosine }, S = 600) => {
    const rnd = TinyLM.mulberry32(seed); let w = 0, b = 0; const out = [L(w, b)];
    for (let t = 0; t < S; t++) {
      const idx = B >= 5 ? [0, 1, 2, 3, 4] : Array.from({ length: B }, () => Math.floor(rnd() * 5));
      let gw = 0, gb = 0;
      for (const i of idx) { const e = w * X[i] + b - Y[i]; gw += (2 * e * X[i]) / idx.length; gb += (2 * e) / idx.length; }
      const lr = cosine ? eta * 0.5 * (1 + Math.cos((Math.PI * t) / S)) : eta;
      w -= lr * gw; b -= lr * gb; out.push(L(w, b));
    }
    return out;
  };
  const gd = run({ B: 5, eta: 0.05, seed: 1 }), sg = run({ B: 1, eta: 0.03, seed: 1 }), co = run({ B: 1, eta: 0.03, seed: 1, cosine: true }), sl = run({ B: 1, eta: 0.01, seed: 1 });
  claim("learn-curves", `33.4 to ${r(gd[5], 2)} in 5 steps`);
  near("GD within 1% at", gd.findIndex((v) => v <= 0.32 * 1.01), 102); claim("learn-curves", "within 1% of the minimum after 102 steps");
  const last = (c) => c.slice(501);
  claim("learn-curves", `averages ${r(sum(last(sg)) / 100, 2)} and ranges from ${r(Math.min(...last(sg)), 2)} to ${r(Math.max(...last(sg)), 2)}`);
  claim("learn-curves", `it ends at ${r(co[600], 3)}`);
  const stop = (c) => { for (let t = 10; t < c.length; t++) if ((c[t - 10] - c[t]) / c[t - 10] < 0.001) return t; return -1; };
  claim("learn-curves", `stops gradient descent at step ${stop(gd)}, loss ${r(gd[stop(gd)], 4)}`);
  claim("learn-curves", `it fires at step ${stop(sg)}, at ${r(sg[stop(sg)], 2)}`);
  ok("SGD fired on an upward step", sg[stop(sg)] > sg[stop(sg) - 10]);
  claim("learn-curves", `drops from ${r(sum(last(sg)) / 100, 2)} to ${r(sum(last(sl)) / 100, 2)}`);
  const el = mount("learn-curves");
  equal("curves figure at step 100", [0, 1, 2].map((i) => tileV(el, i)), [r(gd[100], 3), r(sg[100], 3), r(co[100], 3)]);
}

// ---------------------------------------------------------------- 2.4
const sig = (z) => 1 / (1 + Math.exp(-z));
const softmax = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), S = sum(e); return e.map((v) => v / S); };
const tr = TinyLM.forward(TinyLM.runningIds.slice(0, C.T));
const z4 = tr.logits[4], p4 = tr.probs[4];
const tok = (t) => TinyLM.idOf(t);
const [ID, NAME, EMAIL, AGE] = [" id", " name", " email", " age"].map(tok);
{
  near("id is token 58", ID, 58); near("next token is id", TinyLM.runningIds[5], ID);
  claim("learn-classes", "is class 58, its token id"); claim("learn-classes", "113 entries, all 0 except \\(y_{58} = 1\\)");
  near("V", C.V, 113);
  // the grammar: after users … WHERE the filter column is any other column of users
  const cols = TinyLM.data.schema.users;
  equal("users columns", cols, ["id", "name", "email", "age"]);
}
{
  near("σ(0)", sig(0), 0.5); near("e^-2", Math.exp(-2), 0.1353, 5e-5); near("σ(2)", sig(2), 0.881, 5e-4); near("σ(-2)", sig(-2), 0.119, 5e-4); near("σ(5)", sig(5), 0.9933, 5e-5);
  near("1 − σ(2) = σ(−2)", 1 - sig(2), sig(-2), 1e-15);
  claim("learn-sigmoid", String.raw`\(\sigma(2) = 1/(1 + e^{-2}) = 1/(1 + 0.1353) = 0.881\)`); claim("learn-sigmoid", String.raw`\(\sigma({-2}) = 0.119 = 1 - 0.881\)`); claim("learn-sigmoid", "A logit of 5 gives 0.9933");
  near("odds", sig(2) / (1 - sig(2)), 7.39, 5e-3); near("ln 7.389", Math.log(Math.exp(2)), 2, 1e-12); claim("learn-sigmoid", "the odds are 7.39 to 1");
  const el = mount("learn-sigmoid");
  equal("sigmoid readouts at z = 2", [0, 1, 2].map((i) => tileV(el, i)), ["0.8808", "0.1192", "7.389"]);
}
{
  const z = [2, 1, 0, -1], e = z.map(Math.exp), S = sum(e), p = e.map((v) => v / S);
  equal("exps", e.map((v) => r(v, 3)), ["7.389", "2.718", "1.000", "0.368"]); near("sum", S, 11.475, 5e-4);
  equal("softmax", p.map((v) => r(v, 3)), ["0.644", "0.237", "0.087", "0.032"]);
  claim("learn-softmax", "the exponentials are 7.389, 2.718, 1 and 0.368, their sum is 11.475, and the probabilities are 0.644, 0.237, 0.087 and 0.032");
  near("ratio for +1 logit", p[0] / p[1], Math.E, 1e-12); claim("learn-softmax", String.raw`\(e = 2.718\)`);
  // invariance and the two-class case
  const rnd = TinyLM.mulberry32(5);
  let inv = true, two = true;
  for (let k = 0; k < 100; k++) {
    const zz = [0, 0, 0, 0].map(() => 8 * rnd() - 4), c = 20 * rnd() - 10;
    const a = softmax(zz), b = softmax(zz.map((v) => v + c));
    if (a.some((v, i) => Math.abs(v - b[i]) > 1e-12)) inv = false;
    const q = softmax(zz.slice(0, 2));
    if (Math.abs(q[0] - sig(zz[0] - zz[1])) > 1e-12) two = false;
  }
  ok("softmax unchanged by adding a constant", inv); ok("two-class softmax is the sigmoid of the difference", two);
  const el = mount("learn-softmax");
  equal("softmax calc default", [0, 1, 2, 3, 4].map((i) => tileV(el, i)), ["0.644", "0.237", "0.087", "0.032", "1.000"]);
  setSlider(el, 4, 10);
  equal("softmax calc with c = 10", [0, 1, 2, 3].map((i) => tileV(el, i)), ["0.644", "0.237", "0.087", "0.032"]);
}
{
  claim("learn-logits", `<code>id</code> ${r(z4[ID], 3)}, <code>email</code> ${r(z4[EMAIL], 3)}, <code>age</code> ${r(z4[AGE], 3)} and <code>name</code> ${r(z4[NAME], 3)}`.replace(/<[^>]+>/g, ""));
  claim("learn-logits", `id ${r(p4[ID], 4)}, email ${r(p4[EMAIL], 4)}, age ${r(p4[AGE], 4)} and name ${r(p4[NAME], 4)}`);
  const rest = 1 - p4[ID] - p4[EMAIL] - p4[AGE] - p4[NAME];
  near("rest", rest, 0.0008, 5e-5); claim("learn-logits", "the other 109 tokens share the remaining 0.0008");
  near("probs from logits", softmax(z4)[ID], p4[ID], 1e-12);
  near("gap age", z4[ID] - z4[AGE], 0.119, 5e-4); near("ratio age/id", p4[AGE] / p4[ID], Math.exp(-(z4[ID] - z4[AGE])), 1e-12); near("0.888", p4[AGE] / p4[ID], 0.888, 5e-4);
  claim("learn-logits", "is 0.119 below"); claim("learn-logits", String.raw`\(e^{-0.119} = 0.888\)`);
  near("gap name", z4[ID] - z4[NAME], 5.63, 5e-3); near("ratio name/id", p4[NAME] / p4[ID], 0.0036, 5e-5);
  claim("learn-logits", "is 5.63 below"); claim("learn-logits", String.raw`\(e^{-5.63} = 0.0036\)`);
  const el = mount("learn-logits");
  equal("logits figure tiles", [0, 1, 2].map((i) => tileV(el, i)), [r(p4[ID] + p4[EMAIL] + p4[AGE], 4), r(p4[NAME], 4), r(p4[AGE] / p4[ID], 3)]);
  // raising name's logit lowers every other probability
  setSlider(el, 0, 7.6);
  ok("raising name's logit lowers the other three", +tileV(el, 0) < p4[ID] + p4[EMAIL] + p4[AGE] && +tileV(el, 1) > 0.2);
}
{
  const g = (z) => { const q = sig(z); return 2 * (q - 1) * q * (1 - q); };
  near("fd check of the gradient", g(-1.3), ((sig(-1.3 + 1e-6) - 1) ** 2 - (sig(-1.3 - 1e-6) - 1) ** 2) / 2e-6, 1e-8);
  near("p at -5", sig(-5), 0.0067, 5e-5); near("grad at -5", g(-5), -0.0132, 5e-5); near("grad at -10", g(-10), -9.1e-5, 5e-7); near("grad at 0", g(0), -0.25, 1e-15);
  near("ratio", g(0) / g(-5), 19, 0.5);
  claim("learn-se-probs", String.raw`gets \(p = 0.0067\) and the gradient is only −0.0132`); claim("learn-se-probs", String.raw`it is \(-9.1 \times 10^{-5}\)`); claim("learn-se-probs", "the gradient is −0.25, about 19 times larger");
  // the gradient magnitude only shrinks from z = 0 down to z = -10 once past its peak at p = 1/3
  near("peak of |grad| at p = 1/3", g(Math.log(0.5)), -8 / 27, 1e-12);
}
{
  const v = [0.9, 0.5, 0.1, 0.01].map((q) => r(-Math.log(q), 3));
  equal("−ln p table", v, ["0.105", "0.693", "2.303", "4.605"]);
  const body = Deck.slides.find((s) => s.id === "learn-xent").body;
  ok("xent table cells", v.every((x) => body.includes(`<td class="r">${x}</td>`)));
  near("likelihood", 0.9 * 0.5 * 0.1 * 0.01, 4.5e-4, 1e-18); near("−ln likelihood", -Math.log(4.5e-4), 7.706, 5e-4); near("sum of losses", sum([0.9, 0.5, 0.1, 0.01].map((q) => -Math.log(q))), -Math.log(4.5e-4), 1e-12);
  claim("learn-xent", String.raw`\(-\ln\,(4.5 \times 10^{-4}) = 7.706\)`);
  near("CE grad at -5", sig(-5) - 1, -0.993, 5e-4); near("75 times", (1 - sig(-5)) / Math.abs(2 * (sig(-5) - 1) * sig(-5) * (1 - sig(-5))), 75, 0.5);
  claim("learn-xent", "it is −0.993, 75 times");
  near("check: −ln p(id)", -Math.log(p4[ID]), 1.061, 5e-4); near("check: 1 − p", 1 - 0.346, 0.654, 1e-12);
  claim("learn-xent", String.raw`\(-\ln\, 0.3460 = 1.061\)`);
}
{
  // p − y on the real 113 logits, checked against finite differences of −ln softmax
  const y = new Array(C.V).fill(0); y[ID] = 1;
  const Lz = (z) => -Math.log(softmax(z)[ID]);
  let fd = true;
  for (const i of [ID, NAME, EMAIL, AGE, 0, 112]) { const zp = z4.slice(), zm = z4.slice(); zp[i] += 1e-5; zm[i] -= 1e-5; if (Math.abs((Lz(zp) - Lz(zm)) / 2e-5 - (p4[i] - y[i])) > 1e-6) fd = false; }
  ok("∂L/∂z = p − y by finite differences on the model's logits", fd);
  near("sum of gradient", sum(p4.map((v, i) => v - y[i])), 0, 1e-12);
  claim("learn-pminusy", String.raw`\(0.3460 - 1 = {-0.6540}\)`); near("−0.6540", p4[ID] - 1, -0.6540, 5e-5);
  claim("learn-pminusy", `+${r(p4[EMAIL], 4)} for email`); claim("learn-pminusy", `+${r(p4[AGE], 4)} for age`); claim("learn-pminusy", `+${r(p4[NAME], 4)} for name and +0.0008 in total`);
}
{
  // the gradient widget: the same simulation, and the real widget in jsdom
  const truth = [1 / 3, 0, 1 / 3, 1 / 3];
  const expLoss = (p) => -truth.reduce((a, q, i) => a + (q > 0 ? q * Math.log(p[i]) : 0), 0);
  const simulate = (lr, N) => { let z = [0, 0, 0, 0]; const rnd = TinyLM.mulberry32(11); const draws = [], probs = []; for (let s = 0; s < N; s++) { const p = TinyLM.softmax(z), u = rnd(), tgt = u < 1 / 3 ? 0 : u < 2 / 3 ? 2 : 3; z = z.map((zi, i) => zi - lr * (p[i] - (i === tgt ? 1 : 0))); draws.push(tgt); probs.push(TinyLM.softmax(z)); } return { z, p: TinyLM.softmax(z), draws, probs }; };
  const s1 = simulate(0.5, 1);
  equal("first draw is email", s1.draws[0], 2);
  equal("step-1 logits", s1.z.map((v) => +v.toFixed(12)), [-0.125, -0.125, 0.375, -0.125]);
  claim("learn-gradient", "is (0.25, 0.25, −0.75, 0.25), so at learning rate 0.5 the email logit rises by 0.375 and the others fall by 0.125");
  const s100 = simulate(0.5, 100);
  claim("learn-gradient", `its probability is ${r(s100.p[1], 3)}`);
  claim("learn-gradient", `they are ${r(s100.p[0], 3)}, ${r(s100.p[2], 3)} and ${r(s100.p[3], 3)}, and the expected loss is ${r(expLoss(s100.p), 3)}`);
  near("ln 3", Math.log(3), 1.099, 5e-4); claim("learn-gradient", String.raw`\(\ln\, 3 = 1.099\)`);
  // name's logit falls on every step
  { let z = [0, 0, 0, 0]; const rnd = TinyLM.mulberry32(11); let falls = true; for (let s = 0; s < 100; s++) { const p = TinyLM.softmax(z), u = rnd(), tgt = u < 1 / 3 ? 0 : u < 2 / 3 ? 2 : 3; const nz = z.map((zi, i) => zi - 0.5 * (p[i] - (i === tgt ? 1 : 0))); if (!(nz[1] < z[1])) falls = false; z = nz; } ok("name's logit falls every step", falls); }
  const s5 = simulate(5, 300);
  ok("rate 5: the last token drawn holds ≥ 0.97 after each of 300 steps", s5.draws.every((d, i) => s5.probs[i][d] >= 0.97));
  claim("learn-gradient", "the token just drawn holds at least 0.97");
  // the widget itself
  const el = mount("learn-gradient");
  press(el, "100 steps");
  const vals = [...el.querySelectorAll(".bar-val")].map((x) => x.textContent.trim());
  equal("gradient widget after 100 steps at 0.5", vals, s100.p.map((v) => v.toFixed(3)));
  near("check answer", 0 - 0.5 * 0.25, -0.125, 1e-15);
}

// ---------------------------------------------------------------- 2.5
{
  near("ln 2", Math.LN2, 0.693, 5e-4); near("ln 9", Math.log(9), 2.197, 5e-4); near("ln 113", Math.log(113), 4.727, 5e-4);
  claim("learn-entropy", String.raw`\(H = {-2} \times 0.5\, \ln\, 0.5 = \ln\, 2 = 0.693\)`); claim("learn-entropy", String.raw`\(H = \ln\, 9 = 2.197\)`); claim("learn-entropy", String.raw`\(\ln\, 113 = 4.727\)`);
  claim("learn-entropy", "1 bit = 0.693 nats");
  const el = mount("learn-entropy");
  equal("entropy plot at 0.5", [0, 1].map((i) => tileV(el, i)), ["0.693", "1.000"]);
  setSlider(el, 0, 0); equal("entropy at p = 0", [0, 1].map((i) => tileV(el, i)), ["0.000", "0.000"]);
  setSlider(el, 0, 1); equal("entropy at p = 1", [0, 1].map((i) => tileV(el, i)), ["0.000", "0.000"]);
}
{
  const q = [p4[ID], p4[EMAIL], p4[AGE]];
  const H = Math.log(3), CE = -sum(q.map(Math.log)) / 3, KL = CE - H;
  near("H(p)", H, 1.0986, 5e-5); near("H(p,q)", CE, 1.1021, 5e-5); near("KL", KL, 0.0035, 5e-5);
  // KL from its own definition
  near("KL by definition", sum(q.map((qi) => (1 / 3) * Math.log((1 / 3) / qi))), KL, 1e-12);
  claim("learn-kl", String.raw`\(H(p) = \ln\, 3 = 1.0986\)`); claim("learn-kl", String.raw`\(H(p, q) = 1.1021\)`); claim("learn-kl", String.raw`\(D_{\mathrm{KL}} = 0.0035\)`);
  claim("learn-kl", `of ${r(p4[ID], 4)}, ${r(p4[EMAIL], 4)} and ${r(p4[AGE], 4)} gives`);
  const el = mount("learn-kl");
  equal("KL calc default", [0, 1, 2, 3].map((i) => tileV(el, i)), ["1.0986", r(-(Math.log(0.346) + Math.log(0.3448) + Math.log(0.3072)) / 3, 4), r(-(Math.log(0.346) + Math.log(0.3448) + Math.log(0.3072)) / 3 - Math.log(3), 4), r(1 - 0.346 - 0.3448 - 0.3072 - 0.0012, 4)]);
}
{
  // the floor, recomputed from the grammar by enumeration (independent of TinyLM.data.floor)
  const Q = TinyLM.allQueries().map((q) => ({ p: q.p, ids: TinyLM.encode(q.text).map(TinyLM.idOf) }));
  near("270 queries", Q.length, 270); near("probabilities sum to 1", sum(Q.map((q) => q.p)), 1, 1e-12);
  const H = [];
  for (let i = 0; i < 8; i++) {
    const pre = new Map(), joint = new Map();
    for (const q of Q) { const k = q.ids.slice(0, i + 1).join(","), kj = k + ">" + q.ids[i + 1]; pre.set(k, (pre.get(k) || 0) + q.p); joint.set(kj, (joint.get(kj) || 0) + q.p); }
    let h = 0;
    for (const [kj, pj] of joint) { const k = kj.split(">")[0]; h -= pj * Math.log(pj / pre.get(k)); }
    H.push(h);
  }
  H.forEach((h, i) => near(`floor position ${i}`, h, TinyLM.data.floor[i], 5e-6));
  equal("floor values as written", [H[0], H[2], H[4], H[6]].map((v) => r(v, 4)), ["2.0865", "0.3025", "0.9635", "2.1972"]);
  ok("four certain positions", [1, 3, 5, 7].every((i) => Math.abs(H[i]) < 1e-12));
  claim("learn-floor", "Entropy 2.0865 nats"); claim("learn-floor", "3 tables: 0.3025"); claim("learn-floor", "or 2 for products : 0.9635"); claim("learn-floor", String.raw`\(\ln\, 9 = 2.1972\)`);
  near("sum", sum(H), 5.5497, 5e-5); near("mean", sum(H) / 8, 0.6937, 5e-5);
  claim("learn-floor", String.raw`\(5.5497 / 8 = 0.6937\)`); claim("learn-floor", "a floor of 0.6937 nats per token");
  const pid = sum(Q.filter((q) => q.ids[1] === ID).map((q) => q.p));
  near("p(id after SELECT)", pid, 5 / 18, 1e-12); claim("learn-floor", "probability 5/18");
  const firstCols = new Set(Q.map((q) => q.ids[1]));
  near("9 possible columns", firstCols.size, 9); claim("learn-floor", "9 possible columns");
  const others = [...firstCols].filter((c) => c !== ID).map((c) => sum(Q.filter((q) => q.ids[1] === c).map((q) => q.p)));
  ok("each other column has 1/12 or 1/9", others.every((v) => Math.abs(v - 1 / 12) < 1e-12 || Math.abs(v - 1 / 9) < 1e-12));
  near("model's exact loss", TinyLM.exactLoss().mean, 0.6942, 5e-5); claim("learn-floor", "Our trained model reaches 0.6942");
  claim("learn-part", "0.6937 nats per token");
  // perplexity
  near("e^ln9", Math.exp(Math.log(9)), 9, 1e-12); near("e^2.0865", Math.exp(H[0]), 8.06, 5e-3); near("e^0.9635", Math.exp(H[4]), 2.62, 5e-3); near("e^0.6937", Math.exp(sum(H) / 8), 2.001, 5e-4);
  claim("learn-perplexity", String.raw`\(e^{2.0865} = 8.06\)`); claim("learn-perplexity", "After WHERE , 2.62"); claim("learn-perplexity", String.raw`\(e^{0.6937} = 2.001\)`);
  const c0 = TinyLM.data.exactCurve[0], cN = TinyLM.data.exactCurve[TinyLM.data.exactCurve.length - 1];
  near("step 0", c0[0], 0); near("loss at step 0", c0[1], 4.722, 5e-4); near("ppl at step 0", Math.exp(c0[1]), 112.4, 0.05);
  near("exact loss at step 0 from the weights", TinyLM.exactLoss(0).mean, c0[1], 5e-5);
  near("last step", cN[0], 6000); near("ppl at 6000", Math.exp(TinyLM.exactLoss().mean), 2.002, 5e-4);
  claim("learn-perplexity", "Its loss was 4.722 at step 0, perplexity 112.4"); claim("learn-perplexity", "0.6942 at step 6,000, perplexity 2.002");
  near("check: e^2.303", Math.exp(2.303), 10.0, 5e-3); near("check: ln10 − ln9", Math.log(10) - Math.log(9), 0.105, 5e-4);
  {
    // all probability on the nine digits, a on one and (1 − a)/8 on each other, averaging ln 10: solve for a
    const avg = (a) => -(Math.log(a) + 8 * Math.log((1 - a) / 8)) / 9;
    let lo = 1 / 9, hi = 0.99;
    for (let k = 0; k < 200; k++) { const mid = (lo + hi) / 2; if (avg(mid) < Math.log(10)) lo = mid; else hi = mid; }
    near("uneven digits average ln 10", avg(lo), Math.log(10), 1e-12); near("a", lo, 0.304, 5e-4); near("b", (1 - lo) / 8, 0.087, 5e-4);
    claim("learn-perplexity", "0.304 on one digit and 0.087 on each of the others also averages");
    claim("learn-perplexity", "as uncertain as a uniform choice among 10 options, 0.105 nats above the floor");
  }
  // the floor table and the perplexity chart are labelled by role and show averages over the grammar
  {
    const tbl = Deck.slides.find((s) => s.id === "learn-floor").fig.rows;
    equal("floor table entropies = enumeration", tbl.map((row) => row[3]), H.map((v) => r(v, 4)));
    claim("learn-floor", "averages over all 270 queries, each weighted by its probability");
    claim("learn-perplexity", "averaged over all 270 queries");
  }
  const el = mount("learn-perplexity");
  equal("perplexity figure at the last step", [0, 1, 2, 3].map((i) => tileV(el, i)), ["6,000", r(cN[1], 4), r(Math.exp(cN[1]), 3), r(Math.exp(sum(H) / 8), 3)]);
}

// ---------------------------------------------------------------- 2.6
{
  // regenerate the data the same way, then fit by solving the normal equations with Gaussian elimination
  // (an independent method from the figure's Householder QR)
  const rnd = TinyLM.mulberry32(4);
  const gauss = () => { let u = 0; while (u === 0) u = rnd(); const v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const f = (x) => Math.sin(Math.PI * x);
  const tx = Array.from({ length: 10 }, (_, i) => -1 + (2 * i) / 9), ty = tx.map((x) => f(x) + 0.2 * gauss());
  const hx = Array.from({ length: 101 }, (_, i) => -1 + i / 50), hy = hx.map((x) => f(x) + 0.2 * gauss());
  const solve = (A, b) => { const n = b.length, M = A.map((row, i) => row.concat([b[i]])); for (let c = 0; c < n; c++) { let piv = c; for (let r2 = c + 1; r2 < n; r2++) if (Math.abs(M[r2][c]) > Math.abs(M[piv][c])) piv = r2; [M[c], M[piv]] = [M[piv], M[c]]; for (let r2 = c + 1; r2 < n; r2++) { const k = M[r2][c] / M[c][c]; for (let j = c; j <= n; j++) M[r2][j] -= k * M[c][j]; } } const x = new Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; } return x; };
  const fit = (d, lam = 0) => { const k = d + 1, A = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => sum(tx.map((x) => x ** (i + j))) / 10 + (i === j ? lam : 0))), b = Array.from({ length: k }, (_, i) => sum(tx.map((x, q) => x ** i * ty[q])) / 10); return solve(A, b); };
  const pv = (c, x) => sum(c.map((ci, p) => ci * x ** p));
  const mse = (c, xs, ys) => sum(xs.map((x, i) => (pv(c, x) - ys[i]) ** 2)) / xs.length;
  const rows = Array.from({ length: 10 }, (_, d) => { const c = fit(d); return [mse(c, tx, ty), mse(c, hx, hy), c]; });
  // the true function and the memorizer
  const msef = (g, xs, ys) => sum(xs.map((x, i) => (g(x) - ys[i]) ** 2)) / xs.length;
  const nn = (x) => { let bi = 0; tx.forEach((t, i) => { if (Math.abs(t - x) < Math.abs(tx[bi] - x)) bi = i; }); return ty[bi]; };
  const tTr = msef(f, tx, ty), tHo = msef(f, hx, hy), nHo = msef(nn, hx, hy);
  claim("learn-heldout", `has MSE ${r(tTr, 3)} on the training points and ${r(tHo, 3)} on the held-out ones`);
  near("memorizer train", msef(nn, tx, ty), 0, 1e-15); claim("learn-heldout", `held-out MSE ${r(nHo, 3)}, three times`); near("three times", nHo / tHo, 3, 0.2);
  claim("learn-heldout", `is ${r(nHo, 3)} for the memorizer and ${round(tHo - tTr, 3)} for the true function`);
  // degrees
  claim("learn-overfit", `Degree 1: training MSE ${r(rows[1][0], 3)}, held-out ${r(rows[1][1], 3)}`);
  claim("learn-overfit", `Degree 3: ${r(rows[3][0], 3)} and ${r(rows[3][1], 3)}`);
  ok("degree 9 interpolates", rows[9][0] < 1e-12); claim("learn-overfit", `Training MSE 0, held-out ${r(rows[9][1], 3)}, 15 times worse`); near("15 times", rows[9][1] / rows[3][1], 15, 0.5);
  near("largest coefficient", Math.max(...rows[9][2].map(Math.abs)), 157, 0.5); claim("learn-overfit", "coefficients up to 157 in size");
  let mx = 0; for (let i = 0; i <= 2000; i++) mx = Math.max(mx, Math.abs(pv(rows[9][2], -1 + i / 1000)));
  near("largest |p9(x)|", mx, 2.5, 0.05); near("largest |y|", Math.max(...ty.concat(hy).map(Math.abs)), 1.33, 5e-3); claim("learn-overfit", "values up to 2.5 in size, where no data point exceeds 1.33");
  // underfitting
  ok("degrees 0–2 have training MSE ≥ 0.274", rows.slice(0, 3).every((q) => q[0] >= 0.274 - 5e-4)); near("7 times the noise variance", rows[1][0] / 0.04, 7, 0.2);
  claim("learn-underfit", "their training MSE is 0.274 or more, about 7 times the noise variance of 0.04");
  ok("training MSE falls with every added parameter", rows.every((q, d) => d === 0 || q[0] < rows[d - 1][0]));
  const ho = rows.map((q) => q[1]);
  near("held-out lowest at degree 3", ho.indexOf(Math.min(...ho)), 3); claim("learn-underfit", `is lowest at degree 3 (${r(ho[3], 3)})`);
  ok("held-out changes little to degree 6, then climbs", Math.max(...ho.slice(3, 7)) - Math.min(...ho.slice(3, 7)) < 0.011 && ho[6] < ho[7] && ho[7] < ho[8] && ho[8] < ho[9]);
  claim("learn-underfit", `to ${r(ho[9], 3)} at degree 9`);
  // the figures agree with this independent fit
  const slOver = Deck.slides.find((s) => s.id === "learn-underfit");
  ok("underfit chart series = independent fit", slOver.fig.series[0].points.every((pt, d) => Math.abs(pt[1] - rows[d][0]) < 1e-6) && slOver.fig.series[1].points.every((pt, d) => Math.abs(pt[1] - rows[d][1]) < 1e-6));
  const el = mount("learn-overfit");
  equal("overfit figure at degree 9", [0, 1, 2, 3].map((i) => tileV(el, i)), ["10", "0.000", r(rows[9][1], 3), r(Math.max(...rows[9][2].map(Math.abs)), 1)]);
  let degOk = true;
  for (let d = 0; d <= 9; d++) { setSlider(el, 0, d); if (tileV(el, 1) !== r(rows[d][0], 3) || tileV(el, 2) !== r(rows[d][1], 3)) degOk = false; }
  ok("overfit figure matches the independent fit at every degree", degOk);
  const elH = mount("learn-heldout");
  equal("held-out figure (memorizer)", [0, 1, 2].map((i) => tileV(elH, i)), ["0.000", r(nHo, 3), r(nHo, 3)]);

  // early stopping: plain gradient descent, step by step, at η = 0.3 (independent of the figure's closed form)
  const X = tx.map((x) => Array.from({ length: 10 }, (_, p) => x ** p));
  let c = new Array(10).fill(0);
  const curve = [];
  for (let t = 1; t <= 20000; t++) {
    const g = new Array(10).fill(0);
    for (let i = 0; i < 10; i++) { const e = pv(c, tx[i]) - ty[i]; for (let p = 0; p < 10; p++) g[p] += (2 / 10) * e * X[i][p]; }
    c = c.map((v, p) => v - 0.3 * g[p]);
    curve.push([mse(c, tx, ty), mse(c, hx, hy)]);
  }
  const best = curve.reduce((b, q, i) => (q[1] < b[1] ? [i + 1, q[1]] : b), [0, Infinity]);
  ok("held-out minimum around step 500", best[0] > 400 && best[0] < 600); near("held-out minimum", best[1], 0.041, 5e-4);
  claim("learn-early", "falls to its lowest value, 0.041, around step 500"); claim("learn-early", String.raw`\(\eta = 0.3\)`);
  ok("training loss falls at every step", curve.every((q, i) => i === 0 || q[0] <= curve[i - 1][0] + 1e-15));
  ok("held-out loss rises after the minimum (to step 20,000)", curve.slice(best[0]).every((q, i, a) => i === 0 || q[1] >= a[i - 1][1] - 1e-12));
  // the figure's closed form agrees with direct descent, and reaches the interpolating fit by 10⁹ steps
  const elE = mount("learn-early");
  const atLog = (v) => { setSlider(elE, 0, v); return [tileV(elE, 0), tileV(elE, 1), tileV(elE, 2)]; };
  const t500 = atLog(Math.log10(501));
  equal("early figure at 501 steps = direct descent", t500, ["501", r(curve[500][0], 3), r(curve[500][1], 3)]);
  const t1e4 = atLog(4);
  equal("early figure at 10,000 steps = direct descent", t1e4, ["10,000", r(curve[9999][0], 3), r(curve[9999][1], 3)]);
  const t1e9 = atLog(9);
  equal("early figure at 10⁹ steps = the interpolating fit", t1e9.slice(1), ["0.000", r(rows[9][1], 3)]);
  claim("learn-early", String.raw`by \(10^{9}\) steps`); claim("learn-early", `(held-out ${r(rows[9][1], 3)})`); claim("learn-early", `nearly as good as the degree-3 fit's ${r(rows[3][1], 3)}`);
  // curvatures: eigenvalues of H = (2/n)XᵀX by the characteristic polynomial's extremes (power iteration and inverse iteration)
  const Hm = Array.from({ length: 10 }, (_, i) => Array.from({ length: 10 }, (_, j) => (2 / 10) * sum(X.map((row) => row[i] * row[j]))));
  const mv = (A, v) => A.map((row) => sum(row.map((a, j) => a * v[j])));
  let v = new Array(10).fill(1); for (let k = 0; k < 500; k++) { const w = mv(Hm, v), nrm = Math.hypot(...w); v = w.map((q) => q / nrm); }
  const hmax = sum(mv(Hm, v).map((q, i) => q * v[i]));
  let u = new Array(10).fill(1); for (let k = 0; k < 200; k++) { const w = solve(Hm, u), nrm = Math.hypot(...w); u = w.map((q) => q / nrm); }
  const hmin = sum(mv(Hm, u).map((q, i) => q * u[i]));
  near("largest curvature", hmax, 3.11, 5e-3); ok("η·h_max < 1: every mode decays monotonically", 0.3 * hmax < 1);
  near("smallest curvature", hmin, 1.5e-7, 5e-9); claim("learn-early", String.raw`\(1.5 \times 10^{-7}\)`);
  {
    // all eigenpairs of H by cyclic Jacobi, then the fit through every point projected on them
    const A = Hm.map((row) => row.slice()), V = Hm.map((_, i) => Hm.map((__, j) => (i === j ? 1 : 0)));
    for (let sw = 0; sw < 100; sw++) {
      let off = 0; for (let i = 0; i < 10; i++) for (let j = i + 1; j < 10; j++) off += A[i][j] ** 2;
      if (off < 1e-32) break;
      for (let p = 0; p < 10; p++) for (let q = p + 1; q < 10; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), tt = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), cc = 1 / Math.sqrt(tt * tt + 1), ss = tt * cc;
        for (let k = 0; k < 10; k++) { const a1 = A[k][p], a2 = A[k][q]; A[k][p] = cc * a1 - ss * a2; A[k][q] = ss * a1 + cc * a2; }
        for (let k = 0; k < 10; k++) { const a1 = A[p][k], a2 = A[q][k]; A[p][k] = cc * a1 - ss * a2; A[q][k] = ss * a1 + cc * a2; }
        for (let k = 0; k < 10; k++) { const v1 = V[k][p], v2 = V[k][q]; V[k][p] = cc * v1 - ss * v2; V[k][q] = ss * v1 + cc * v2; }
      }
    }
    const cstar = rows[9][2];
    const modes = A.map((row, k) => [row[k], Math.abs(sum(V.map((vr, i) => vr[k] * cstar[i])))]).sort((a, b) => a[0] - b[0]);
    near("flattest curvature", modes[0][0], 1.45e-7, 5e-9); near("second flattest", modes[1][0], 2.9e-6, 5e-8);
    near("component along the flattest", modes[0][1], 198, 0.5); near("component along the second", modes[1][1], 125, 0.5);
    ok("these two are the largest components", modes.slice(2).every((m) => m[1] < 20));
    near("1/(ηh) flattest", 1 / (0.3 * modes[0][0]), 2.3e7, 5e5); near("1/(ηh) second", 1 / (0.3 * modes[1][0]), 1.2e6, 5e4);
    claim("learn-early", String.raw`components of 198 and 125, at curvatures \(1.5 \times 10^{-7}\) and \(2.9 \times 10^{-6}\)`);
    claim("learn-early", String.raw`\(1/(\eta h) = 2.3 \times 10^{7}\) and \(1.2 \times 10^{6}\) steps`);
    // the coefficient length at step 500, from direct descent
    let c5 = new Array(10).fill(0);
    for (let s5 = 0; s5 < 500; s5++) { const g = new Array(10).fill(0); for (let i = 0; i < 10; i++) { const e = pv(c5, tx[i]) - ty[i]; for (let p = 0; p < 10; p++) g[p] += (2 / 10) * e * X[i][p]; } c5 = c5.map((vv, p) => vv - 0.3 * g[p]); }
    near("‖c‖ at step 500", Math.hypot(...c5), 3.7, 0.05); near("‖c*‖", Math.hypot(...cstar), 234.8, 0.05);
    claim("learn-early", "At step 500 the coefficients have length 3.7, against 234.8");
  }
  // ridge
  const ridge = (lam) => { const cr = fit(9, lam); return [mse(cr, tx, ty), mse(cr, hx, hy), Math.hypot(...cr)]; };
  const r0 = [rows[9][0], rows[9][1], Math.hypot(...rows[9][2])], r3 = ridge(1e-3), r1 = ridge(1);
  claim("learn-regularize", String.raw`\(\lVert \mathbf c \rVert = ${r(r0[2], 1)}\) and held-out MSE ${r(r0[1], 3)}`);
  claim("learn-regularize", `shrinks them to ${r(r3[2], 1)}; the training MSE rises to ${r(r3[0], 3)} and the held-out MSE falls to ${r(r3[1], 3)}`);
  claim("learn-regularize", `held-out ${r(r1[1], 3)}`); claim("learn-regularize", `at \\(\\lambda = 1\\) it is ${r(r1[0], 3)}`);
  const elR = mount("learn-regularize");
  equal("ridge figure at λ = 10⁻³", [1, 2, 3].map((i) => tileV(elR, i)), [r(r3[2], 1), r(r3[0], 3), r(r3[1], 3)]);
  setSlider(elR, 0, -10); equal("ridge figure at λ = 0", [0, 1, 3].map((i) => tileV(elR, i)), ["0", r(r0[2], 1), r(r0[1], 3)]);
  setSlider(elR, 0, 0); equal("ridge figure at λ = 1", [2, 3].map((i) => tileV(elR, i)), [r(r1[0], 3), r(r1[1], 3)]);
  // the L2 gradient step equals multiplying by 1 − 2ηλ before the data step (plain gradient descent)
  { const eta = 0.3, lam = 1e-3, w0 = 2.5, gData = 0.7; near("L2 step = decay step", w0 - eta * (gData + 2 * lam * w0), (1 - 2 * eta * lam) * w0 - eta * gData, 1e-15); }
  claim("learn-regularize", String.raw`under plain gradient descent every step also multiplies each \(c_j\) by \(1 - 2\eta\lambda\)`);
  claim("learn-regularize", "AdamW applies the decay directly");
}
{
  // one pass: GPT-3 Table 2.2 (Brown et al. 2020) and our model's repetition
  near("CC tokens drawn", 0.6 * 300, 180, 1e-12); near("CC passes", 180 / 410, 0.44, 5e-3);
  claim("learn-onepass", "It drew 180 billion of them from its largest source, 410 billion tokens of filtered web text, which is 0.44 of one pass");
  claim("learn-onepass", "repeated up to 3.4 times");
  near("768,000", C.steps * C.batch, 768000); near("27 held out", TinyLM.data.holdout.queries.length, 27);
  {
    const Qs = TinyLM.allQueries(), copies = (t) => Qs.filter((q) => q.t === t).map((q) => C.steps * C.batch * q.p);
    ok("every users and orders query: 2,370 expected copies", copies("users").concat(copies("orders")).every((v) => Math.abs(v - 2370.37) < 0.01));
    ok("every products query: 4,741 expected copies", copies("products").every((v) => Math.abs(v - 4740.74) < 0.01));
    near("mean copies", sum(Qs.map((q) => C.steps * C.batch * q.p)) / Qs.length, 2844.4, 0.05);
  }
  claim("learn-onepass", "768,000 training queries were drawn from only 270 distinct queries");
  claim("learn-onepass", "about 2,370 times and each products query about 4,741 times (2,844 on average)"); claim("learn-onepass", "without 27 of the queries");
  claim("learn-onepass", "no systematic gap opens"); claim("learn-onepass", "Over its last 100 batches");
  near("distinct queries", TinyLM.data.distinctQueries, 270);
}

done();
