// Part 9: every number stated on the post-training sheets, recomputed.
// Model numbers come from TinyLM (the page's own forward pass); the fine-tuning
// experiments come from tools/verify/part09_experiments.py (PyTorch), whose output
// is recorded in EXP below and re-run here when the venv is available.
import { claim, near, equal, ok, r, int, slide, TinyLM as M, Deck, window as W, text, done } from "./lib.mjs";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import { EXP } from "./part09_exp.mjs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ids = (t) => M.encode(t).map(M.idOf);
const sig = (x) => 1 / (1 + Math.exp(-x));
const pct = (x, d) => r(100 * x, d) + "%";
const fig = (id) => Deck.slides.find((s) => s.id === id).fig;
// Render a sheet's figure into a detached element (jsdom) and return its text.
function renderFig(id) {
  const sp = fig(id), el = W.document.createElement("div");
  W.document.body.appendChild(el);
  Deck.figs[sp.type](el, sp, { strip: { set() {}, reset() {} } });
  return el;
}
// p(each answer token | prompt and earlier answer tokens)
function answerProbs(prompt, answer) {
  const pre = ids(prompt), ans = ids(answer), all = pre.concat(ans);
  const tr = M.forward(all.slice(0, Math.min(all.length - 1, M.config.T)));
  return ans.map((id, i) => tr.probs[pre.length - 1 + i][id]);
}

// ------------------------------------------------------------------ 9.1 From base model to assistant
{
  const pa = answerProbs("SELECT name FROM users WHERE", " id = 7;");
  const [pId, pEq, p7, pSemi] = pa;
  near("p(id | ...WHERE) = running-example value", pId, 0.3460, 5e-5);
  near("p(7 | ...id =)", p7, 0.1077, 5e-5);
  ok("p(=) and p(;) round to 1.000", pEq > 0.9995 && pSemi > 0.9995);
  claim("post-base", "probability 0.346");
  claim("post-base", "probability 0.108");
  const prod = pa.reduce((a, b) => a * b, 1);
  near("asked-for query probability", prod, 0.03727, 5e-5);
  claim("post-base", `0.346 × 0.108 = ${r(prod, 3)}`);
  equal("one sample in N", Math.round(1 / prod), 27);
  claim("post-base", "about one sample in 27");
  // the figure: exact line and the first 10 seeded samples
  const el = renderFig("post-base");
  ok("post-continue shows the exact product", el.textContent.includes(`= ${r(prod, 4)}, one sample in 27`));
  ok("post-continue starts with 30 samples", el.querySelectorAll(".post-dot").length === 30);
  // InstructGPT quote is 7 words (copyright limit is 15)
  claim("post-base", `"follow the user's instructions helpfully and safely"`);

  // Llama 3 tokenizer facts quoted on post-template
  claim("post-template", "256 of them");
  const chat = renderFig("post-template");
  equal("chat template special tokens in the prompt", chat.querySelectorAll(".post-chat:first-child .post-seg.sp").length, 1 + 3 * 2 + 2);
  equal("generated part ends with eot", [...chat.querySelectorAll(".post-seg.gen")].pop().textContent, "<|eot_id|>");

  // post-pipeline / post-stages arithmetic
  near("R1 SFT data", 600 + 200, 800);
  claim("post-pipeline", "800K examples (600K reasoning, 200K other)");
  claim("post-pipeline", "12,725");
  claim("post-pipeline", "33,207");
  claim("post-pipeline", "31,144");
  near("V3 total GPU hours", 2664 + 119 + 5, 2788);
  claim("post-stages", "5K of its 2,788K H800 GPU hours on post-training, " + pct(5 / 2788, 2));
  near("R1 GPU hours", 101 + 5 + 41, 147);
  claim("post-stages", "147K, 101K of them for R1-Zero");
}

// ------------------------------------------------------------------ experiments (PyTorch), recorded

// the repo's virtualenv (python3 -m venv .venv && .venv/bin/pip install numpy torch), or $PYTHON
const PY = process.env.PYTHON || path.join(HERE, "../../.venv/bin/python");
if (process.env.RERUN_PY !== "0" && fs.existsSync(PY)) {
  const run = spawnSync(PY, [path.join(HERE, "part09_experiments.py")], { encoding: "utf8", timeout: 600000 });
  if (run.status === 0) {
    const fresh = JSON.parse(run.stdout);
    equal("PyTorch experiments reproduce the recorded numbers", JSON.stringify(fresh), JSON.stringify(EXP));
  } else ok("PyTorch experiments ran: " + (run.stderr || "").split("\n").slice(-3).join(" "), false);
}

// ------------------------------------------------------------------ 9.2 Supervised fine-tuning
{
  claim("post-sft-data", "12,725 prompts");
  near("LIMA examples (Table 1)", 200 + 200 + 200 + 150 + 50 + 200, 1000);
  claim("post-sft-data", "400 from Stack Exchange, 200 from wikiHow, 150 from Reddit's WritingPrompts, 50 from Super-Natural Instructions and 200 written by the authors");
  claim("post-sft-data", "over 2.7 million synthetic examples for code alone");
  claim("post-stages", "10³ to 10⁷ examples");
  claim("post-sft-data", "10 to 30 answers");

  // the sft widget: 18 tokens, 17 predictions, 10 trained with the mask
  const el = W.document.createElement("div");
  Deck.figs.tinylm(el, fig("post-sft-mask"), { strip: { set() {}, reset() {} }, byId: () => -1, slides: [] });
  ok("sft widget: 10 of 17 trained", el.textContent.includes("10 of 17"));
  equal("sft widget: chips", el.querySelectorAll(".chip").length, 18);
  claim("post-sft-mask", "18 tokens, so 17 next-token predictions. Ten are trained: the 9 tokens of our query and the end marker");
  near("token-mean weight of a 100-token answer", 100 / 110, 0.909, 5e-4);
  claim("post-sft-mask", `100-token answer ${pct(100 / 110, 0)} of the weight`);

  // packing: default lengths 5, 3, 8
  const Ls = [5, 3, 8], T = Ls.reduce((a, b) => a + b, 0), mx = Math.max(...Ls);
  near("padding slots", 3 * mx - T, 8);
  claim("post-packing", "8 of 24");
  const per = Ls.map((l) => (l * (l + 1)) / 2);
  equal("per-example pairs", per, [15, 6, 36]);
  claim("post-packing", "15 + 6 + 36 = 57");
  near("causal pairs", (T * (T + 1)) / 2, 136);
  claim("post-packing", "of the 136 the causal mask allows");
  const tilesOf = (mode, lengths) => {
    const sp = Object.assign({}, fig("post-packing"), { mode, lengths });
    const e = W.document.createElement("div");
    Deck.figs["post-pack"](e, sp, {});
    return [...e.querySelectorAll(".f-tile-v")].map((x) => x.textContent);
  };
  equal("pack fig, padded", tilesOf("pad", Ls), ["24", "8 (33%)", "57", "0"]);
  equal("pack fig, causal", tilesOf("causal", Ls), ["16", "0 (0%)", "136", "79"]);
  equal("pack fig, per-example", tilesOf("doc", Ls), ["16", "0 (0%)", "57", "0"]);
  equal("pack fig, extremes 1,1,1", tilesOf("causal", [1, 1, 1]), ["3", "0 (0%)", "6", "3"]);
  equal("pack fig, extremes 8,8,8 padded", tilesOf("pad", [8, 8, 8]), ["24", "0 (0%)", "108", "0"]);

  // LoRA
  near("dk", 4096 * 4096, 16777216);
  near("r(d+k)", 16 * (4096 + 4096), 131072);
  claim("post-lora", "16,777,216 becomes 16 × 8,192 = 131,072, which is " + pct(131072 / 16777216, 2));
  const layer = 2 * (4096 + 4096) + 2 * (4096 + 1024) + 3 * (4096 + 14336);
  const llama8b = 2 * 128256 * 4096 + 32 * (2 * 4096 * 4096 + 2 * 4096 * 1024 + 3 * 4096 * 14336 + 2 * 4096) + 4096;
  near("Llama 3 8B parameters", llama8b, 8030261248);
  near("LoRA r=16 on 224 layers", 32 * 16 * layer, 41943040);
  equal("linear layers", 32 * 7, 224);
  claim("post-lora", `r = 16 trains ${r(32 * 16 * layer / 1e6, 1)}M parameters, ${pct(32 * 16 * layer / llama8b, 2)} of ${r(llama8b / 1e9, 2)}B`);
  const lora = (preset, rr) => {
    const sp = Object.assign({}, fig("post-lora"), { preset, r: rr });
    const e = W.document.createElement("div");
    Deck.figs["post-lora"](e, sp, {});
    return { t: [...e.querySelectorAll(".f-tile-v")].map((x) => x.textContent), x: e.querySelector(".f-mm-expl").textContent };
  };
  equal("lora fig default", lora(1, 16).t, ["16,777,216", "131,072", "0.78%"]);
  ok("lora fig whole-model line", lora(1, 16).x.includes("rank 16 trains 41.9M parameters, 0.52% of 8.03B") && lora(1, 16).x.includes("dk/(d + k) = 2,048."));
  ok("lora fig W_k limit 819.2", lora(2, 4).x.includes("dk/(d + k) = 819.2."));
  equal("lora fig 16x16 r=8 equal", lora(0, 8).t.slice(0, 3), ["256", "256", "100.00%"]);
  ok("lora fig 16x16 r=8 says saves nothing", lora(0, 8).x.includes("as many as"));
  equal("lora fig r clamps to 16 for 16x16", lora(0, 256).t.slice(0, 3), ["256", "512", "200.00%"]);
  equal("lora fig MLP up r=1", lora(3, 1).t.slice(0, 3), ["58,720,256", "18,432", "0.03%"]);
  claim("post-lora", "1.2 TB to 350 GB");

  // QLoRA
  near("FP32 scales per weight", 32 / 64, 0.5);
  const dq = 8 / 64 + 32 / (64 * 256);
  near("double-quantized scales", dq, 0.126953125);
  claim("post-qlora", `8/64 + 32/(64 × 256) = ${r(dq, 3)} bits, a saving of ${r(0.5 - dq, 3)} bits`);
  near("saving on 65B, GB", ((0.5 - dq) * 65e9) / 8 / 1e9, 3.03, 0.01);
  claim("post-qlora", "about 3 GB on a 65B model");
  claim("post-qlora", `16 bytes each needs ${int(16 * 65)} GB, ${Math.ceil((16 * 65) / 80)} GPUs of 80 GB`);
  near("12 bytes x 65B", 12 * 65, 780);
  claim("post-qlora", "more than 780 GB, which is 12 bytes per parameter");
  claim("post-qlora", "45.0 GB for batch size 1, 512 tokens and gradient checkpointing");
  claim("post-qlora", `4.127 / 8 bytes = ${r((65e9 * 4.127) / 8 / 1e9, 1)} GB`);
  const q = fig("post-qlora");
  const at = (N, sh) => q.outputs.map((o) => o.fmt(o.f({ N, s: sh })));
  equal("qlora calc at 65B, 1%", at(6.5e10, 0.01), ["1,040 GB", "140 GB", "43.9 GB"]);
  equal("qlora calc at 1B, 0.1%", at(1e9, 0.001), ["16.0 GB", "2.0 GB", "0.5 GB"]);
  equal("qlora calc at 405B, 5%", at(4.05e11, 0.05), ["6,480 GB", "1,134 GB", "533 GB"]);
}

// ------------------------------------------------------------------ 9.2 forgetting (PyTorch)
{
  const F = EXP.forget;
  const rows = [["base", "Before fine-tuning"], ["narrow", "Orders only"], ["mix25", "Orders + 25% original"]];
  for (const [k] of rows) {
    for (const t of ["orders", "users", "products"]) claim("post-forget", r(F[k][t].mean, 3), `${k} ${t}`);
  }
  claim("post-forget", `fell from ${r(F.base.orders.mean, 3)} to ${r(F.narrow.orders.mean, 3)}`);
  claim("post-forget", `users queries rose from ${r(F.base.users.mean, 3)} to ${r(F.narrow.users.mean, 3)}`);
  claim("post-forget", `products queries from ${r(F.base.products.mean, 3)} to ${r(F.narrow.products.mean, 3)}`);
  const e = Math.floor(Math.log10(F.narrow.p_users)), m = F.narrow.p_users / 10 ** e;
  equal("p(users) after narrow, mantissa and exponent", [m.toFixed(1), e], ["9.3", -7]);
  claim("post-forget", "probability 9.3 × 10⁻⁷");
  ok("p(users) stays 1.000 with mixing", F.base.p_users > 0.9995 && F.mix25.p_users > 0.9995);
  claim("post-forget", `keeps users at ${r(F.mix25.users.mean, 3)} and products at ${r(F.mix25.products.mean, 3)}`);
  claim("post-forget", `brings orders down to ${r(F.mix25.orders.mean, 3)}`);
  equal("setup", F.setup, { steps: 300, batch: 128, lr: 0.001, mix: 0.25 });
  claim("post-forget", "300 steps of Adam (learning rate 10⁻³, 128 queries per step)");
  claim("post-forget", "8 times as many pretraining examples as RL episodes, gradient weight γ = 27.8");
}

// ------------------------------------------------------------------ 9.3 Preferences
{
  const C2 = (n) => (n * (n - 1)) / 2;
  equal("pairs from 4 and 9 answers", [C2(4), C2(9)], [6, 36]);
  claim("post-prefs-data", "up to 36 pairs");
  equal("pair rows in the plate", fig("post-prefs-data").rows.length, 6);

  near("sigma(1.5)", sig(1.5), 0.8176, 5e-5);
  claim("post-bt", `\\(\\Delta = 1.5\\), so \\(p(A \\succ B) = \\sigma(1.5) = ${r(sig(1.5), 3)}\\)`);
  claim("post-bt", `odds of \\(e^{1.5} = ${r(Math.exp(1.5), 2)}\\) to 1`);
  near("shift invariance", 11.2 - 9.7, 1.5, 1e-12);
  claim("post-bt", "give the same 0.818");
  claim("post-bt", "multiplies the odds by e = 2.718");
  const bt = fig("post-bt");
  const btOut = (d) => bt.readouts.map((o) => W.UI.format(o.f({ d }), o.fmt));
  equal("BT readouts at 1.5", btOut(1.5), ["0.818", "4.48 : 1", "0.201"]);
  equal("BT readouts at -6", btOut(-6), ["0.002", "1 : 403.43", "6.002"]);
  equal("BT readouts at 6", btOut(6), ["0.998", "403.43 : 1", "0.002"]);

  const rm = [2, 0.5, 0, -1].map((d) => [r(sig(d), 3), r(-Math.log(sig(d)), 3), r(-sig(-d), 3).replace("-", "−")]);
  equal("RM table", fig("post-rm").rows.slice(0, 4).map((row) => row.slice(2)), rm);
  const mean = [2, 0.5, 0, -1].reduce((a, d) => a - Math.log(sig(d)), 0) / 4;
  near("RM mean loss", mean, 0.6519, 5e-5);
  claim("post-rm", `The mean loss is ${r(mean, 3)}`);
  claim("post-rm", `has the largest loss, ${r(-Math.log(sig(-1)), 3)}, and the largest gradient, −${r(sig(1), 3)}`);
  claim("post-rm", "about 1.1%, doubling the model by about 1.8%");

  near("20 ln 2", 20 * Math.log(2), 13.863, 5e-4);
  claim("post-rlhf", `\\(20\\, \\ln 2 = ${r(20 * Math.log(2), 2)}\\) nats, which costs ${r(0.02 * 20 * Math.log(2), 3)} reward at InstructGPT's β = 0.02`);
  near("trained model state, GB", (7e9 * 16) / 1e9, 112);
  near("four models, GB", 2 * 112 + 2 * 14, 252);
  claim("post-rlhf", "7 × 10⁹ × 16 bytes = 112 GB each and the two frozen ones 14 GB each in BF16: 252 GB before activations, at least 4 GPUs");
  equal("GPUs", Math.ceil(252 / 80), 4);

  const rho = Math.exp(0.4);
  claim("post-ppo", `\\(\\rho = e^{0.4} = ${r(rho, 3)}\\)`);
  claim("post-ppo", `min(${r(2 * rho, 3)}, 1.2 × 2) = 2.4`);
  claim("post-ppo", `min(−${r(rho, 3)}, −1.2) = −${r(rho, 3)}`);
  claim("post-ppo", "0.82, against 0.76 for 0.1, 0.70 for 0.3 and −0.39 without clipping");
  equal("PPO runs per setting", 3 * 7, 21);
  claim("post-ppo", "3 seeds on each of 7 environments, 21 runs per setting");
  const pp = fig("post-ppo");
  const ppOut = (A, eps, rh) => pp.readouts.map((o) => W.UI.format(o.f({ A, eps, rho: rh }), o.fmt));
  equal("PPO default (A=2, 1.49)", ppOut(2, 0.2, 1.49), ["2.400", "0.0", "yes, no gradient"]);
  equal("PPO A=-1 at 1.49", ppOut(-1, 0.2, 1.49), ["−1.490", "−1.0", "no"]);
  equal("PPO A=-1 at 0.6", ppOut(-1, 0.2, 0.6), ["−0.800", "0.0", "yes, no gradient"]);
  equal("PPO A=0", ppOut(0, 0.2, 2.5), ["0.000", "0.0", "no"]);
  equal("PPO rho=0, A=-2, eps=0.5", ppOut(-2, 0.5, 0), ["−1.000", "0.0", "yes, no gradient"]);
  equal("PPO rho=2.5, A=2, eps=0.05", ppOut(2, 0.05, 2.5), ["2.100", "0.0", "yes, no gradient"]);
  equal("PPO rho=2.5, A=-2", ppOut(-2, 0.05, 2.5), ["−5.000", "−2.0", "no"]);
  equal("PPO inside the range", ppOut(1.5, 0.2, 1.1), ["1.650", "1.5", "no"]);

  const lw = Math.log(answerProbs("SELECT email FROM users WHERE", " id = 7;").reduce((a, b) => a * b, 1));
  const ll = Math.log(answerProbs("SELECT email FROM users WHERE", " age = 7;").reduce((a, b) => a * b, 1));
  claim("post-dpo", `\\(\\ln \\pi_{\\text{ref}}(y_w) = ${r(lw, 3)}\\) and \\(\\ln \\pi_{\\text{ref}}(y_l) = ${r(ll, 3)}\\)`);
  near("ln 2", Math.log(2), 0.6931, 5e-5);
  claim("post-dpo", "\\(\\ln 2 = 0.693\\) for every pair");
  claim("post-dpo", "0.05 at β = 0.1");
  near("EXP reference log-prob of chosen matches TinyLM", EXP.dpo.ref.lp_chosen, lw, 2e-4);

  const p0 = Math.exp(lw);
  claim("post-beta", `\\(\\pi_{\\text{ref}} = ${r(p0, 3)}\\)`);
  claim("post-beta", `\\(\\sigma(1)\\) = ${pct(sig(1), 0)}`);
  const qs = (b) => 1 / (1 + ((1 - p0) / p0) * Math.exp(-1 / b));
  const kl = (q) => q * Math.log(q / p0) + (1 - q) * Math.log((1 - q) / (1 - p0));
  claim("post-beta", `at β = 1, ${r(Math.E, 2)}× and \\(\\pi^*\\) = ${r(qs(1), 3)}`);
  claim("post-beta", `at β = 0.5, ${r(Math.exp(2), 2)}× and ${r(qs(0.5), 3)}`);
  claim("post-beta", `at β = 0.1, ${int(Math.exp(10))}× and ${r(qs(0.1), 3)}, at a KL of ${r(kl(qs(0.1)), 2)} nats`);
  const m01 = -0.1 - Math.log(1 - Math.exp(-0.1));
  near("beta*m for loss 0.1", m01, 2.2522, 5e-5);
  near("loss at that margin", -Math.log(sig(m01)), 0.1, 1e-12);
  claim("post-beta", `The loss is 0.1 at \\(\\beta m = ${r(m01, 2)}\\): a margin of ${r(m01 / 0.1, 1)} nats at β = 0.1, ${r(m01 / 0.5, 1)} at β = 0.5`);
  claim("post-beta", "drove rejected answers to about \\(e^{-37}\\)");
  const bp = fig("post-beta");
  const bOut = (b, g) => bp.readouts.map((o) => W.UI.format(o.f({ b, g }), o.fmt));
  equal("beta plot default", bOut(0.1, 1), [r(qs(0.1), 3), "22,026×", r(kl(qs(0.1)), 3), r(m01 / 0.1, 1)]);
  equal("beta plot at 1", bOut(1, 1), [r(qs(1), 3), "2.72×", r(kl(qs(1)), 3), r(m01, 1)]);
  const ex1 = bOut(0.02, 3);
  ok("beta plot at beta 0.02, gap 3: finite and saturated " + ex1.join(" | "), ex1[0] === "1.000" && /× 10/.test(ex1[1]) && !ex1.some((x) => /NaN|∞/.test(x)));
  const ex2 = bOut(2, 0.25);
  ok("beta plot at beta 2, gap 0.25 stays near the reference " + ex2.join(" | "), Math.abs(+ex2[0] - p0) < 0.01);
  ok("beta plot curve finite on its domain", [0.001, 0.01, 0.5, 2].every((x) => Number.isFinite(bp.fns[0].f(x, { g: 3 }))));

  const D = EXP.dpo;
  claim("post-dpo-leak", `from probability ${r(D.ref.p_chosen, 3)} to ${r(D.plain.p_chosen, 3)}`);
  equal("rejected log-prob, rounded", Math.round(D.plain.lp_rej_mean), -37);
  claim("post-dpo-leak", "\\(e^{-37}\\)");
  claim("post-dpo-leak", `now hold only ${r(D.plain.mass27, 3)}: the other ${r(1 - D.plain.mass27, 3)} moved`);
  equal("top leaked string", D.plain.top[1][0], " id = 7 status");
  claim("post-dpo-leak", `led by id = 7 status at ${r(D.plain.top[1][1], 3)}`);
  const rows = fig("post-dpo-leak").rows;
  const refTop = answerProbs("SELECT email FROM users WHERE", " id = 3;").reduce((a, b) => a * b, 1);
  equal("reference top answer", D.ref.top[0][0], " id = 3;");
  equal("leak table numbers", rows.map((x) => [x[1], x[2].replace(/<\/?b>/g, "")]), [
    [r(D.ref.p_chosen, 3), r(D.ref.mass27, 3)], [r(D.plain.p_chosen, 3), r(D.plain.mass27, 3)],
    [r(D.nll.p_chosen, 3), r(D.nll.mass27, 3)], [r(D.mask.p_chosen, 3), r(D.mask.mass27, 3)], [r(D.mask_nll.p_chosen, 3), r(D.mask_nll.mass27, 3)]]);
  equal("leak table top answers", rows.map((x) => x[3].replace(/<[^>]+>/g, "")), [`id = 3; ${r(refTop, 3)}`, `id = 7;, then id = 7 status ${r(D.plain.top[1][1], 3)}`, "id = 7;", "id = 7;", "id = 7;"]);
  ok("leak table: winners are the chosen answer", [D.plain, D.nll, D.mask, D.mask_nll].every((x) => x.top[0][0] === " id = 7;"));
  equal("lr 1e-4: masked and unmasked agree to 4 dp", r(D["plain_lr1e-4"].p_chosen, 4), r(D["mask_lr1e-4"].p_chosen, 4));
  claim("post-dpo-leak", "At learning rate 10⁻⁴ the masked and unmasked runs agree to 4 decimals");
  claim("post-dpo-leak", `masked, ${r(D.mask.mass27, 3)} of the probability stays inside the grammar, and the NLL term raises that to ${r(D.mask_nll.mass27, 3)}`);
  ok("beta ordering in the masked runs: larger beta, smaller KL", D["mask_b0.05"].kl27 > D.mask.kl27 && D.mask.kl27 > D["mask_b0.5"].kl27 && D["mask_b0.5"].kl27 > D.mask_b1.kl27);
}

// the PPO-vs-DPO check: chosen at 2x its reference probability, rejected at 0.5x, beta 0.1
{
  const m = Math.log(2) - Math.log(0.5);
  const opts = slide("post-ppo-dpo").check.options;
  equal("DPO check options", opts, [r(Math.log(2), 3), r(-Math.log(sig(0.1 * m)), 3), r(-Math.log(sig(m)), 3), r(-Math.log(sig(-0.1 * m)), 3)]);
  equal("DPO check answer", opts[slide("post-ppo-dpo").check.answer], r(-Math.log(sig(0.1 * m)), 3));
  claim("post-ppo-dpo", `\\(\\ln 2 - \\ln 0.5 = \\ln 4 = ${r(m, 3)}\\)`);
  claim("post-ppo-dpo", `\\(-\\ln\\sigma(0.1 \\times ${r(m, 3)}) = -\\ln ${r(sig(0.1 * m), 3)} = ${r(-Math.log(sig(0.1 * m)), 3)}\\)`);
  const st = slide("post-stages").check, fg = slide("post-forget").check;
  equal("forgetting check answer", fg.answer, 0);
  ok("forgetting check: the answer is about shared weights", /shared by all contexts/.test(fg.options[fg.answer]));
  ok("forgetting check: name is a users column only", M.data.schema.users.includes("name") && !M.data.schema.orders.includes("name"));
  equal("stages check answer is DPO", st.options[st.answer], "DPO");
}

// ------------------------------------------------------------------ 9.4 RL with verifiable rewards
{
  const PROMPT = "SELECT email FROM users WHERE";
  const cols = [" id", " name", " age"], digits = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  const P = {}, run = W.Widgets._runSQL;
  let gram = 0;
  for (const c of cols) for (const d of digits) {
    const ans = `${c} = ${d};`, pr = answerProbs(PROMPT, ans).reduce((a, b) => a * b, 1);
    P[ans] = pr; gram += pr;
    const res = run(`${PROMPT}${ans}`);
    const reward = res.rows && res.rows.length === 1 && res.rows[0] === "gus@example.com" ? 1 : 0;
    equal(`checker on${ans}`, reward, c === " id" && d === 7 ? 1 : 0);
    if (c !== " id") equal(`no rows for${ans}`, res.rows.length, 0);
  }
  const sum = (f) => Object.entries(P).filter(([k]) => f(k)).reduce((a, [, v]) => a + v, 0);
  const rows = fig("post-verifier").rows.map((x) => x[3]);
  equal("checker table probabilities", rows, [r(P[" id = 7;"], 3), r(sum((k) => k.startsWith(" id") && k !== " id = 7;"), 3), r(sum((k) => k.startsWith(" name")), 3), r(sum((k) => k.startsWith(" age")), 3), r(1 - gram, 3)]);
  near("grammatical share", gram, EXP.dpo.ref.mass27, 1e-6);
  claim("post-verifier", `the reward is 1 with probability ${r(P[" id = 7;"], 3)}, and ${r(gram, 3)} of the samples are grammatical`);
  claim("post-hacking", `grammatical query ${pct(gram, 1)} of the time`);

  // the three-action policy: our model's filter-column distribution, renormalized
  const pre = ids(PROMPT), pc = M.forward(pre).probs[4];
  const raw = cols.map((c) => pc[M.idOf(c)]), Z = raw.reduce((a, b) => a + b, 0), pi = raw.map((x) => x / Z);
  equal("pi", pi.map((x) => r(x, 3)), ["0.352", "0.313", "0.335"]);
  claim("post-reinforce", "Our model's probabilities are 0.352, 0.313 and 0.335");
  const e = [1 - pi[0], -pi[1], -pi[2]], exact = e.map((x) => pi[0] * x);
  const v3 = (v) => `(${v.map((x) => r(x, 3).replace("-", "−")).join(", ")})`;
  claim("post-reinforce", `gives the estimate ${v3(e)}`);
  claim("post-reinforce", `exact gradient ${v3(exact)}`);
  const z1 = [Math.log(pi[0]) + e[0], Math.log(pi[1]) + e[1], Math.log(pi[2]) + e[2]];
  const pid1 = Math.exp(z1[0]) / z1.reduce((a, z) => a + Math.exp(z), 0);
  claim("post-reinforce", `raises π(id) from ${r(pi[0], 3)} to ${r(pid1, 3)}`);
  const stepsTex = fig("post-reinforce").steps.map((x) => x.html).join(" ");
  ok("REINFORCE steps show the same numbers", ["0.352", "0.313", "0.335", "0.648", "0.228", "0.110", "0.118", "0.590"].every((x) => stepsTex.includes(x)));

  // variance of the single-sample estimate, exactly
  const variance = (c, useB) => {
    const rr = [1 + c, c, c], J = pi.reduce((a, p, i) => a + p * rr[i], 0), b = useB ? J : 0;
    const g = pi.map((_, a) => pi.map((p, k) => (rr[a] - b) * ((k === a ? 1 : 0) - p)));
    const Eg = [0, 1, 2].map((k) => pi.reduce((acc, p, a) => acc + p * g[a][k], 0));
    return { v: pi.reduce((acc, p, a) => acc + p * g[a].reduce((q, x) => q + x * x, 0), 0) - Eg.reduce((q, x) => q + x * x, 0), Eg };
  };
  claim("post-baseline", `${r(variance(0, false).v, 3)} without a baseline and ${r(variance(0, true).v, 3)} with b = J`);
  claim("post-baseline", `it is ${r(variance(10, false).v, 1)} without and again ${r(variance(10, true).v, 3)} with`);
  ok("baseline variance is shift-invariant", Math.abs(variance(10, true).v - variance(0, true).v) < 1e-12 && Math.abs(variance(-2, true).v - variance(0, true).v) < 1e-12);
  ok("mean estimate equals exact gradient for every shift", [-2, 0, 3.5, 10].every((c) => [false, true].every((b) => variance(c, b).Eg.every((x, k) => Math.abs(x - exact[k]) < 1e-12))));
  const pgTiles = (shift, baseline) => {
    const el = W.document.createElement("div");
    Deck.figs["post-pg"](el, Object.assign({}, fig("post-baseline"), { shift, baseline }), {});
    return [...el.querySelectorAll(".f-tile-v")].map((x) => x.textContent);
  };
  equal("pg fig default", pgTiles(0, false), [v3(exact), v3(exact), r(variance(0, false).v, 3)]);
  equal("pg fig shift 10, baseline", pgTiles(10, true), [v3(exact), v3(exact), r(variance(10, true).v, 3)]);
  equal("pg fig shift -2, none", pgTiles(-2, false), [v3(exact), v3(exact), r(variance(-2, false).v, 3)]);

  // GRPO advantages and wasted groups
  const m = 1 / 8, sd = Math.sqrt(m * (1 - m));
  claim("post-grpo", `the mean is ${r(m, 3)} and the standard deviation ${r(sd, 3)}, so the correct answer gets A = +${r((1 - m) / sd, 2)} and each wrong one −${r(m / sd, 2)}`);
  const pw = Math.exp(Math.log(pi[0])) * (M.softmax(digits.map((d) => Math.log(M.forward(pre.concat(ids(" id ="))).probs[6][M.idOf(" " + d)])))[6]);
  near("widget's starting p(correct)", pw, 0.0379, 5e-4);
  claim("post-grpo", `starts at p(correct) = ${r(pw, 3)}, so (1 − ${r(pw, 3)})⁸ = ${pct((1 - pw) ** 8, 0)} of first groups of 8 are wasted, and ${pct((1 - pw) ** 64, 0)} of groups of 64`);
  const wel = W.document.createElement("div");
  Deck.figs.tinylm(wel, fig("post-grpo"), { strip: { set() {}, reset() {} }, byId: () => -1, slides: [] });
  ok("rl widget starts at 3.8%", wel.textContent.includes("start 3.8%"));
  [...wel.querySelectorAll("button")].find((b) => b.textContent.includes("Run 10")).click();
  ok("rl widget: Run 10 rounds reaches 55.2%", wel.textContent.includes("55.2%"));
  claim("post-grpo", "raises p(correct) from 3.8% to 55.2%");
  claim("post-grpo", `60 groups took p(correct) from ${pct(EXP.grpo.start, 1)} to ${pct(EXP.grpo.end, 1)}; ${EXP.grpo.zero_groups} groups had no correct answer`);
  equal("full-model GRPO updates", EXP.grpo.updates, 60);
  claim("post-grpo", "8% of groups of 64, the size DeepSeekMath used");

  // reasoning: plate values match the text
  equal("bars", fig("post-reasoning").items.map((x) => x.text), ["9.3%", "13.4%", "79.8%", "86.7%", "90.0%"]);
  claim("post-reasoning", "lifts GPT-4o only from 9.3% to 13.4%, while one reasoning sample of DeepSeek-R1 scores 79.8%");
  claim("post-reasoning", "74% of problems with one sample, 83% by majority vote over 64 samples and 93% by re-ranking 1,000 samples");
  claim("post-reasoning", "from 15.6% to 77.9%");

  // reward hacking: an independent implementation of the run, compared with the figure
  const zc0 = cols.map((c) => Math.log(pc[M.idOf(c)]));
  const pd7 = M.forward(pre.concat(ids(" id ="))).probs[6];
  const zd0 = digits.map((d) => Math.log(pd7[M.idOf(" " + d)]));
  function sim(reward, seed, N = 40) {
    let zc = zc0.slice(), zd = zd0.slice();
    const rand = M.mulberry32(seed), out = [];
    for (let u = 0; u <= N; u++) {
      const a = M.softmax(zc), b = M.softmax(zd);
      out.push([a[0], a[0] * b[6]]);
      if (u === N) break;
      const g = []; for (let i = 0; i < 8; i++) { const c = M.sampleFrom(a, rand), d = M.sampleFrom(b, rand); g.push([c, d, reward(c, d)]); }
      const mu = g.reduce((s2, x) => s2 + x[2], 0) / 8, sdv = Math.sqrt(g.reduce((s2, x) => s2 + (x[2] - mu) ** 2, 0) / 8);
      if (!sdv) continue;
      const gc = [0, 0, 0], gd = digits.map(() => 0);
      for (const [c, d, rw] of g) { const A = (rw - mu) / sdv; gc.forEach((_, i) => { gc[i] += (A * ((i === c) - a[i])) / 8; }); gd.forEach((_, i) => { gd[i] += (A * ((i === d) - b[i])) / 8; }); }
      zc = zc.map((v, i) => v + gc[i]); zd = zd.map((v, i) => v + gd[i]);
    }
    return out;
  }
  const px = sim((c) => +(c === 0), 5), tr = sim((c, d) => +(c === 0 && d === 6), 5);
  claim("post-hacking", `climbs from ${pct(px[0][0], 0)} to ${pct(px[40][0], 1)}, but the share of truly correct queries ends at ${pct(px[40][1], 1)}`);
  claim("post-hacking", `the same run reaches ${pct(tr[40][1], 1)}`);
  near("true-reward run matches the widget after 10 groups", tr[10][1], 0.552, 5e-4);
  const hk = (seed) => {
    const el = W.document.createElement("div");
    Deck.figs["post-hack"](el, Object.assign({}, fig("post-hacking"), { seed }), {});
    return [...el.querySelectorAll(".f-tile-v")].map((x) => x.textContent);
  };
  for (const seed of [5, 6, 7, 8]) {
    const a = sim((c) => +(c === 0), seed), b = sim((c, d) => +(c === 0 && d === 6), seed);
    equal(`hack fig seed ${seed}`, hk(seed), [pct(a[40][0], 1), pct(a[40][1], 1), pct(b[40][1], 1)]);
    ok(`seed ${seed}: proxy reward above 98%, true success below 25%`, a[40][0] > 0.98 && a[40][1] < 0.25);
  }
}

// ------------------------------------------------------------------ 9.5 RL infrastructure
{
  near("R1-Zero rollout tokens, 32K limit", 8192 * 32768, 268435456);
  claim("post-rlinfra", `up to ${Math.round((8192 * 32768) / 1e6)} million tokens`);
  claim("post-rlinfra", `65,536 tokens each, up to ${Math.round((8192 * 65536) / 1e6)} million`);
  claim("post-rlinfra", "over 90% of RLHF and RLVR runtime");
  claim("post-rlinfra", "up to 81.2% in NeMo-Aligner");

  near("70B BF16 bytes", 70e9 * 2, 140e9);
  claim("post-sync", "140 GB per step for a 70B model in BF16");
  near("NVLink each way, GB/s", 18 * 25, 450);
  claim("post-sync", `(50 GB/s each way) takes ${r(140 / 50, 1)} s. Over one H100's NVLink it takes at least ${r(140 / 450, 2)} s`);
  claim("post-sync", "18 links × 25 GB/s = 450 GB/s each way");
  claim("post-sync", "140 GB does not fit on one 80 GB GPU");
  claim("post-sync", "up to 36.4% of an RLHF iteration");
  claim("post-sync", "by up to 71.2% against DeepSpeed-Chat and 89.1% against OpenRLHF");
  const sc = fig("post-sync");
  const scOut = (N, B, bw, n) => sc.outputs.map((o) => W.UI.format(o.f({ N, B, bw, n }), o.fmt));
  equal("sync calc default", scOut(7e10, 2, 50e9, 1), ["140 GB", "140 GB", "2.8 s"]);
  equal("sync calc 70B NVLink", scOut(7e10, 2, 450e9, 1), ["140 GB", "140 GB", "311.1 ms"]);
  equal("sync calc 70B on 8 GPUs over ports", scOut(7e10, 2, 50e9, 8), ["140 GB", "17.5 GB", "350.0 ms"]);
  equal("sync calc 1B FP8 NVLink", scOut(1e9, 1, 450e9, 1), ["1.0 GB", "1.0 GB", "2.22 ms"]);
  equal("sync calc 1T BF16 one port", scOut(1e12, 2, 50e9, 1), ["2,000 GB", "2,000 GB", "40.0 s"]);
  equal("sync link options", sc.inputs[2].options.map((o) => o.value), [50e9, 450e9]);
  equal("sync sender options", sc.inputs[3].options.map((o) => o.value), [1, 2, 4, 8]);

  // timeline: 6 generate, 3 train, 1 sync
  const tl = fig("post-async").items;
  const done = (row) => tl.filter((x) => x.row === row && x.kind === "s2" && x.start + x.len <= 30).length;
  equal("batches trained in 30 units (sync, async)", [done(1), done(3)], [3, 4]);
  near("sync step", 6 + 3 + 1, 10);
  claim("post-async", "a step takes 10 units and the trainer is busy 30% of the time");
  claim("post-async", "standard PPO scored 23.3, the decoupled objective 42.2, synchronous training 42.0");
  near("async speedup", 10 / 6, 1.667, 5e-4);
  claim("post-async", `A step then takes 6 units, ${r(10 / 6, 2)} times the throughput`);
  ok("async generator never idles", tl.filter((x) => x.row === 2).every((x, i, a) => i === 0 || x.start === a[i - 1].start + a[i - 1].len));
  near("R1-Zero mini-batches", 8192 / 16, 512);
  claim("post-async", "as 16 mini-batches, so the last one is used 15 updates after it was sampled");
  claim("post-async", "about 40% faster");
  claim("post-async", "up to 2.77 times");

  near("R1-Zero batch", 32 * 16, 512);
  near("R1-Zero GPU hours", 64 * 8 * 198, 101376);
  claim("post-rlscale", "64 × 8 H800 GPUs for about 198 hours: 101K GPU hours");
  equal("check options", ["12,672", "101,376", "1,584", "811,008"], [int(64 * 198), int(512 * 198), int(8 * 198), int(512 * 198 * 8)]);
  equal("cost bars", fig("post-rlscale").items.map((x) => x.text), ["2,664K", "119K", "5K", "101K", "5K", "41K"]);
  near("R1 total from bars", 101 + 5 + 41, 147);
  near("R1-Zero RL as a share of V3 pretraining", 101 / 2664, 0.0379, 5e-5);
  claim("post-rlscale", `101K GPU hours, ${pct(101 / 2664, 1)} of the 2,664K that pretrained DeepSeek-V3 and ${Math.round(101 / 5)} times its 5K of post-training`);
  claim("post-rlscale", "in the final 400 of 1,700 steps");
  claim("post-rlscale", "98.5% of the time in spot checks, against 84.4%");
  claim("post-grpo", "β = 0.04 in DeepSeekMath, 0.001 for R1-Zero");
}

// ------------------------------------------------------------------ 9.6 Evaluation
{
  const held = new Set(M.data.holdout.queries);
  const loss = (step) => {
    const acc = { held: [0, 0], seen: [0, 0] };
    for (const q of M.allQueries()) {
      const qi = ids(q.text), tr = M.forward(qi.slice(0, 8), step);
      let l = 0; for (let i = 0; i < 8; i++) l -= Math.log(tr.probs[i][qi[i + 1]]);
      const k = held.has(q.text) ? "held" : "seen";
      acc[k][0] += q.p * (l / 8); acc[k][1] += q.p;
    }
    return { held: acc.held[0] / acc.held[1], seen: acc.seen[0] / acc.seen[1] };
  };
  const A = loss(M.finalStep), B = loss("holdout");
  near("main model, held-out loss vs build script", A.held, M.data.holdout.lossA.held, 5e-5);
  near("holdout model, held-out loss vs build script", B.held, M.data.holdout.lossB.held, 5e-5);
  equal("held-out queries", held.size, 27);
  claim("post-bench", `has a loss of ${r(A.held, 3)} nats per token and the one that never saw them ${r(B.held, 3)}: a benchmark made of them would favor the first by ${r(B.held - A.held, 3)}`);
  claim("post-bench", `the two are within ${r(Math.abs(A.seen - B.seen), 3)}`);
  equal("bench table", fig("post-bench").rows.map((x) => x.slice(1)), [[r(A.held, 3), r(A.seen, 3)], [r(B.held, 3), r(B.seen, 3)]]);
  claim("post-bench", "85% of HellaSwag was flagged, with an estimated gain of 14.8 points for the 8B model; 52% of Natural Questions");

  claim("post-judge", "85% of non-tied votes, against 81% agreement between two experts");
  claim("post-judge", "65.0% of the time and favored the first position in 30.0%");
  claim("post-judge", "fooled GPT-4 in 8.7% of 23 cases and Claude-v1 in 91.3%");
  equal("judge table", fig("post-judge").rows, [["GPT-4", "65.0%", "30.0%", "8.7%"], ["GPT-3.5", "46.2%", "50.0%", "91.3%"], ["Claude-v1", "23.8%", "75.0%", "91.3%"]]);

  const ex = M.exactLoss().mean, floor = M.data.floor.reduce((a, b) => a + b, 0) / M.data.floor.length;
  claim("post-life", `loss ${r(ex, 4)} against a floor of ${r(floor, 4)}`);
  claim("post-life", `from ${r(EXP.dpo.ref.p_chosen, 3)} to ${r(EXP.dpo.mask_nll.p_chosen, 3)} with masking and NLL`);
  claim("post-life", "3.8% to 55.2% in 10 groups with a 12-logit policy");
  claim("post-life", `3.8% to ${pct(EXP.grpo.end, 1)} in ${EXP.grpo.updates} groups on all 9,968 weights`);
  equal("tabular policy logits", 3 + 9, 12);
  claim("post-life", `Held-out loss ${r(B.held, 3)}, against ${r(A.held, 3)} when contaminated`);
  claim("post-life", "9,968 random weights, seed 7");
  const tokens = M.config.steps * M.config.batch * M.config.T;
  near("our training tokens", tokens, 6144000);
  equal("parameter ratio, millions", Math.round(405e9 / M.config.params / 1e6), 41);
  claim("post-life", "about 40 million times larger in parameters");
  near("token ratio, millions", 15.6e12 / tokens / 1e6, 2.54, 0.005);
  claim("post-life", "about 2.5 million times larger in training tokens: 15.6 trillion against our 6.1 million");
}

done();
