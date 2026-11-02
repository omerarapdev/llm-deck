// Prints the model-derived numbers the slides quote. Run: node legacy/tools/facts.mjs
import fs from "node:fs"; import vm from "node:vm";
const ctx = { window: {}, atob: (s) => Buffer.from(s, "base64").toString("binary"), Math, console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(new URL("../../data/model-data.js", import.meta.url), "utf8"), ctx);
vm.runInContext(fs.readFileSync(new URL("../js/model.js", import.meta.url), "utf8"), ctx);
const M = ctx.window.TinyLM, V = M.vocab;
const ids = M.runningIds;
console.log("ids", M.running.map((t,i)=>JSON.stringify(t)+"="+ids[i]).join(" "));
console.log("vocab", V.length, "params", M.config.params, "base", M.data.baseCount);
for (const step of [0, M.finalStep]) {
  const tr = M.forward(ids.slice(0,8), step);
  const row = tr.probs.map((p,i)=>`${M.running[i+1].trim()}:${p[ids[i+1]].toFixed(3)}(${(-Math.log(p[ids[i+1]])).toFixed(3)})`);
  const mean = tr.probs.reduce((s,p,i)=>s - Math.log(p[ids[i+1]]),0)/8;
  console.log("step", step, row.join(" "), "mean", mean.toFixed(4));
}
const ex = M.exactLoss(); console.log("exact final", ex.mean.toFixed(4), ex.perPos.map(x=>x.toFixed(4)).join(" "));
console.log("floor", M.data.floor.map(x=>x.toFixed(4)).join(" "), "mean", (M.data.floor.reduce((a,b)=>a+b)/8).toFixed(4));
const ex0 = M.exactLoss(0); console.log("exact step0", ex0.mean.toFixed(4), "ln V", Math.log(V.length).toFixed(4));
// validity of sampled queries by temperature
const valid = new Set(M.allQueries().map(q=>q.text));
for (const T of [0.25, 0.5, 0.8, 1, 1.5, 2, 3]) {
  const rand = M.mulberry32(42); let ok = 0, n = 2000;
  for (let k=0;k<n;k++){ let cur=[ids[0]];
    while (cur.length < 9) { const p = M.nextDist(cur, {temperature:T}); const t = M.sampleFrom(p, rand); cur.push(t); if (V[t]===";") break; }
    if (valid.has(cur.map(i=>V[i]).join(""))) ok++; }
  console.log("T", T, "valid", (ok/n*100).toFixed(1)+"%");
}
// greedy completion from "SELECT"
let cur=[ids[0]]; while(cur.length<9){ const p=M.nextDist(cur,{temperature:1}); let b=0; p.forEach((v,i)=>{if(v>p[b])b=i}); cur.push(b); if(V[b]===";")break; }
console.log("greedy", cur.map(i=>V[i]).join(""));
// attention cells
const tr = M.forward(ids.slice(0,8));
tr.layers.forEach((L,l)=>L.heads.forEach((h,hh)=>console.log(`L${l+1}H${hh+1}`, h.att.map((r,i)=>r.slice(0,i+1).map(x=>x.toFixed(2)).join(",")).join(" | "))));
