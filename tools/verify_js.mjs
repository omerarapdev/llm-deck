// Checks js/core/model.js against the PyTorch fixtures written by build_model.py.
// Run: node tools/verify_js.mjs
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ctx = { window: {}, atob: (s) => Buffer.from(s, "base64").toString("binary"), console, Math };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(root, "data/model-data.js"), "utf8"), ctx);
vm.runInContext(fs.readFileSync(path.join(root, "js/core/model.js"), "utf8"), ctx);
const M = ctx.window.TinyLM;
const fx = JSON.parse(fs.readFileSync(path.join(root, "tools/fixtures.json"), "utf8"));

let failures = 0;
const check = (ok, msg) => { if (!ok) { failures++; console.log("FAIL", msg); } };

// tokenizer
for (const [q, toks] of Object.entries(fx.tok_tests)) {
  const got = M.encode(q);
  check(JSON.stringify(got) === JSON.stringify(toks), `encode(${q}) = ${JSON.stringify(got)} want ${JSON.stringify(toks)}`);
}
for (const [k, toks] of Object.entries(fx.tok_upto)) {
  const got = M.encode(M.data.running, Number(k));
  check(JSON.stringify(got) === JSON.stringify(toks), `upto ${k}`);
}
check(JSON.stringify(M.running) === JSON.stringify(fx.running_tokens), "running tokens");
check(JSON.stringify(M.runningIds) === JSON.stringify(fx.running_ids), "running ids");

// forward pass, every snapshot
let worstLogit = 0, worstAtt = 0, worstLoss = 0;
for (const [step, s] of Object.entries(fx.snaps)) {
  const tr = M.forward(M.runningIds.slice(0, 8), Number(step));
  tr.logits.forEach((r, i) => r.forEach((v, j) => { worstLogit = Math.max(worstLogit, Math.abs(v - s.logits[i][j])); }));
  tr.layers.forEach((L, l) => L.heads.forEach((h, hh) => h.att.forEach((r, i) => r.forEach((v, j) => {
    worstAtt = Math.max(worstAtt, Math.abs(v - s.att[l][hh][i][j]));
  }))));
  const ex = M.exactLoss(Number(step));
  worstLoss = Math.max(worstLoss, Math.abs(ex.mean - s.exact));
  ex.perPos.forEach((v, i) => { worstLoss = Math.max(worstLoss, Math.abs(v - s.per_pos[i])); });
}
console.log(`max |logit diff| ${worstLogit.toExponential(2)}, max |attention diff| ${worstAtt.toExponential(2)}, max |loss diff| ${worstLoss.toExponential(2)}`);
check(worstLogit < 1e-4, "logits match");
check(worstAtt < 1e-5, "attention matches");
check(worstLoss < 1e-5, "exact loss matches");

// sampling sanity: filtered distributions sum to 1
const p = M.nextDist(M.runningIds.slice(0, 5), { temperature: 0.7, topK: 3 });
check(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-9, "filtered probs sum to 1");
check(p.filter((v) => v > 0).length === 3, "topK keeps 3");

console.log(failures ? `${failures} FAILURES` : "all checks passed");
process.exit(failures ? 1 : 0);
