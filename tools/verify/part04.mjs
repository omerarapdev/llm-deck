// Part 4: every number the Language modeling sheets state, recomputed.
// UTF-8 facts come from Node's Buffer, tokenizer facts from TinyLM's data and
// encoder, probabilities from TinyLM.forward and the grammar (TinyLM.allQueries).
// The fixed-window experiment is trained by tools/verify/part04_window.py; its
// printed results are recorded below and checked against the sheets.
import { claim, near, equal, ok, r, int, TinyLM as M, text, done } from "./lib.mjs";

const D = M.data, V = M.vocab, C = M.config;
const qIds = M.runningIds, qToks = M.running;
const hexBytes = (s) => Buffer.from(s, "utf8").toString("hex").toUpperCase().match(/../g).join(" ");
const cp = (c) => "U+" + c.codePointAt(0).toString(16).toUpperCase().padStart(4, "0");
// claim with an HTML snippet as written in the sheet (tags count as spaces, as in the sheet text)
const claimH = (id, html, label) => claim(id, html.replace(/<[^>]+>/g, " "), label);

// The tokenizer corpus: tools/build_model.py draws it with Python's random.Random(1234).
// This is CPython's MT19937 (init_by_array seeding, getrandbits, _randbelow, choice).
function pyRandom(seed) {
  const mt = new Uint32Array(624);
  let mti;
  mt[0] = 19650218;
  for (mti = 1; mti < 624; mti++) mt[mti] = (Math.imul(1812433253, mt[mti - 1] ^ (mt[mti - 1] >>> 30)) + mti) >>> 0;
  const key = [seed >>> 0];
  let i = 1, j = 0;
  for (let k = Math.max(624, key.length); k; k--) {
    mt[i] = ((mt[i] ^ Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1664525)) + key[j] + j) >>> 0;
    i++; j++;
    if (i >= 624) { mt[0] = mt[623]; i = 1; }
    if (j >= key.length) j = 0;
  }
  for (let k = 623; k; k--) {
    mt[i] = ((mt[i] ^ Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1566083941)) - i) >>> 0;
    i++;
    if (i >= 624) { mt[0] = mt[623]; i = 1; }
  }
  mt[0] = 0x80000000;
  mti = 624;
  const next = () => {
    if (mti >= 624) {
      for (let k = 0; k < 624; k++) {
        const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
        mt[k] = mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      mti = 0;
    }
    let y = mt[mti++];
    y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18;
    return y >>> 0;
  };
  const below = (n) => { const k = 32 - Math.clz32(n); let r; do r = next() >>> (32 - k); while (r >= n); return r; };
  return { choice: (a) => a[below(a.length)] };
}
const corpus = (() => {
  const rng = pyRandom(1234), S = D.schema, tables = Object.keys(S), out = [D.running];
  for (let q = 0; q < 1999; q++) {
    const t = rng.choice(tables), c1 = rng.choice(S[t]), c2 = rng.choice(S[t].filter((c) => c !== c1)), v = rng.choice(D.values);
    out.push(`SELECT ${c1} FROM ${t} WHERE ${c2} = ${v};`);
  }
  return out;
})();
equal("corpus head reproduced", corpus.slice(0, 16), D.corpusHead);
{
  const wc = new Map();
  for (const q of corpus) for (const w of M.pretokenize(q)) wc.set(w, (wc.get(w) || 0) + 1);
  ok("corpus word counts reproduced", wc.size === D.wordCounts.length && D.wordCounts.every(([w, c]) => wc.get(w) === c));
  near("corpus distinct queries", new Set(corpus).size, D.corpusDistinct);
}

// ---------------------------------------------------------------- 4.1 Text as numbers
{
  const id = "lm-unicode";
  for (const [c, want] of [["S", "U+0053"], ["é", "U+00E9"], ["日", "U+65E5"], ["😀", "U+1F600"]]) { equal(`code point ${c}`, cp(c), want); claim(id, want); }
  near("code space", 0x10ffff + 1, 1114112);
  claim(id, "1,114,112");
  // UTF-8 length boundaries
  const len = (n) => Buffer.byteLength(String.fromCodePoint(n), "utf8");
  equal("utf8 lengths", [0x0, 0x7f, 0x80, 0x7ff, 0x800, 0xffff, 0x10000, 0x10ffff].map(len), [1, 1, 2, 2, 3, 3, 4, 4]);
  claim(id, "1 byte for U+0000 to U+007F"); claim(id, "2 bytes up to U+07FF"); claim(id, "3 up to U+FFFF");
  const bits = (0xe9).toString(2).padStart(11, "0");
  equal("é bits", `${bits.slice(0, 5)} ${bits.slice(5)}`, "00011 101001");
  claim(id, "00011 101001");
  const b = Buffer.from("é", "utf8");
  equal("é bytes binary", [...b].map((x) => x.toString(2)).join(" "), "11000011 10101001");
  claim(id, "11000011 10101001");
  equal("é hex", hexBytes("é"), "C3 A9"); claim(id, "the bytes C3 A9");
  equal("日 hex", hexBytes("日"), "E6 97 A5"); claim(id, "3 bytes, E6 97 A5");
  equal("😀 hex", hexBytes("😀"), "F0 9F 98 80"); claim(id, "4, F0 9F 98 80");
  near("query chars", Array.from(D.running).length, 36); near("query bytes", Buffer.byteLength(D.running), 36);
  claim(id, "36 characters, 36 bytes");
  const dev = "👩‍💻";
  near("dev code points", Array.from(dev).length, 3); near("dev bytes", Buffer.byteLength(dev), 11);
  claim(id, "three of them and 11 bytes");
}
{
  const id = "lm-vocab";
  near("vocab", V.length, 113); claim(id, "113 tokens");
  near("base", D.baseCount, 39); claim(id, "Ids 0 to 38 are the 39 characters");
  const base = V.slice(0, 39);
  ok("base sorted by code point", base.every((c, i) => i === 0 || c.codePointAt(0) > base[i - 1].codePointAt(0)));
  near("capitals", base.filter((c) => /[A-Z]/.test(c)).length, 11); claim(id, "11 capital letters");
  near("lowercase", base.filter((c) => /[a-z]/.test(c)).length, 15); claim(id, "15 lowercase letters");
  equal("digits", base.filter((c) => /[0-9]/.test(c)).join(""), "123456789"); claim(id, "the digits 1 to 9");
  ok("space ; = _ in base", [" ", ";", "=", "_"].every((c) => base.includes(c)));
  near("merges", D.merges.length, 74); claim(id, "Ids 39 to 112 are 74 longer strings");
  equal("query ids", qIds, [43, 93, 47, 63, 52, 58, 53, 107, 10]);
  claim(id, "43, 93, 47, 63, 52, 58, 53, 107 and 10");
  near("users id", M.idOf(" users"), 63); claim(id, "the single token 63");
  claim(id, "128,256");
}
{
  const id = "lm-onehot";
  claim(id, "113 numbers with a single 1 at index 63");
  near("llama one-hot bytes", 128256 * 4, 513024); claim(id, "513,024 bytes");
  claim(id, r(Math.log2(128256), 2)); claim(id, "about 17 bits");
  claim(id, r(Math.SQRT2, 3));
}
{
  const id = "lm-words";
  near("corpus words", D.wordCounts.length, 26); claim(id, "26 distinct words");
  const words = new Set(D.wordCounts.map(([w]) => w));
  const nq = "SELECT username FROM customers WHERE age = 42;";
  const ws = M.pretokenize(nq);
  near("new query words", ws.length, 9);
  equal("unknown words", ws.filter((w) => !words.has(w)), [" username", " customers", " 42"]);
  claim(id, "three of the nine words never occur");
  // every query of the grammar is 9 words, all in the corpus vocabulary
  ok("grammar words in vocab", M.allQueries().every((q) => { const w = M.pretokenize(q.text); return w.length === 9 && w.every((x) => words.has(x)); }));
  claim(id, "every query is 9 of them");
  claim(id, "47,578"); claim(id, "16,383"); claim(id, "30,000 to 50,000");  // sourced: Bengio et al. 2003, Sennrich et al. 2016
}
{
  const id = "lm-subwords";
  const nq = "SELECT username FROM customers WHERE age = 42;";
  equal("username pieces", M.encode(" username"), [" user", "n", "a", "m", "e"]);
  near("new query tokens", M.encode(nq).length, 20); ok("new query known", M.encode(nq).every((t) => M.idOf(t) >= 0));
  claim(id, "costs 20 tokens and none is unknown");
  near("new query chars", Array.from(nq).length, 46);
  const t10 = M.encode("SELECT name FROM users WHERE id = 10;");
  equal("0 unknown", t10.filter((t) => M.idOf(t) < 0), ["0"]);
  claim(id, "the <code>0</code> in <code>10</code>".replace(/<[^>]+>/g, " "));
  near("scores ratio", (36 * 36) / (9 * 9), 16); claim(id, "1,296"); claim(id, "9 × 9 = 81"); claim(id, "(36 / 9)² = 16");
}

// ---------------------------------------------------------------- 4.2 Tokenization
// BPE training with the tie rule of tools/build_model.py: most frequent pair, ties to the pair seen first.
function bpeTrain(words, counts, maxMerges = Infinity) {
  let splits = words.map((w) => Array.from(w));
  const merges = [];
  while (merges.length < maxMerges) {
    const pc = new Map(), first = new Map();
    splits.forEach((sy, wi) => { for (let i = 0; i < sy.length - 1; i++) { const k = sy[i] + "\u0001" + sy[i + 1]; pc.set(k, (pc.get(k) || 0) + counts[wi]); if (!first.has(k)) first.set(k, [wi, i]); } });
    if (!pc.size) break;
    const best = [...pc].sort((a, b) => b[1] - a[1] || first.get(a[0])[0] - first.get(b[0])[0] || first.get(a[0])[1] - first.get(b[0])[1])[0];
    const [a, b] = best[0].split("\u0001");
    merges.push({ a, b, c: best[1], tied: [...pc.values()].filter((c) => c === best[1]).length });
    splits = splits.map((sy) => { const o = []; for (let i = 0; i < sy.length; i++) { if (i < sy.length - 1 && sy[i] === a && sy[i + 1] === b) { o.push(a + b); i++; } else o.push(sy[i]); } return o; });
  }
  return merges;
}
{
  const id = "lm-bpe";
  const m = bpeTrain([" users", " orders", " user_id"], [4, 3, 2]);
  equal("mini merge 1", [m[0].a, m[0].b, m[0].c], ["e", "r", 9]); claim(id, "4 + 3 + 2 = 9");
  equal("mini merge 2", [m[1].a, m[1].b, m[1].c], ["er", "s", 7]); claim(id, "er s</code> (7)".replace(/<[^>]+>/g, " "));
  equal("mini merge 3 (tie)", [m[2].a, m[2].b, m[2].c, m[2].tied], [" ", "u", 6, 2]); claim(id, "tie at 6");
  // the same trainer, run on the reproduced corpus (words in order of first appearance), gives our 74 merges
  const order = [], cnt = new Map();
  for (const q of corpus) for (const w of M.pretokenize(q)) { if (!cnt.has(w)) order.push(w); cnt.set(w, (cnt.get(w) || 0) + 1); }
  const full = bpeTrain(order, order.map((w) => cnt.get(w)));
  equal("tokenizer retrained from the corpus", full.map((m) => [m.a, m.b, m.c]), D.bpeLog.map((e) => [e.pair[0], e.pair[1], e.count]));
}
{
  const id = "lm-bpe-ours";
  near("corpus", D.corpusSize, 2000); claim(id, "2,000 queries");
  near("distinct words", D.wordCounts.length, 26); claim(id, "26 distinct words");
  near("word occurrences", D.wordCounts.reduce((a, [, c]) => a + c, 0), 18000); claim(id, "18,000 word occurrences");
  const L = D.bpeLog;
  ok("first 15 merges count 2000", L.slice(0, 15).every((e) => e.count === 2000) && L[15].count < 2000);
  equal("first 15 merges make the keywords", D.merges.slice(0, 15).map(([a, b]) => a + b).filter((t) => ["SELECT", " FROM", " WHERE", " ="].includes(t)), ["SELECT", " FROM", " WHERE", " ="]);
  claim(id, "The first 15 merges build the keywords");
  const wc = Object.fromEntries(D.wordCounts);
  equal("merge 16 er", [L[15].pair.join(""), L[15].count], ["er", 1688]); near("er = users+orders+user_id", wc[" users"] + wc[" orders"] + wc[" user_id"], 1688); claim(id, "(1,688 times");
  equal("merge 17 id", [L[16].pair.join(""), L[16].count], ["id", 1441]); near("id = id + user_id", wc[" id"] + wc[" user_id"], 1441); claim(id, "(1,441)");
  equal("merge 18 us", [L[17].pair.join(""), L[17].count], ["us", 1354]); near("us = users+user_id+status", wc[" users"] + wc[" user_id"] + wc[" status"], 1354); claim(id, "(1,354)");
  const last9 = L.slice(-9);
  ok("last 9 merges are space+digit", last9.every((e) => e.pair[0] === " " && /^[1-9]$/.test(e.pair[1])));
  equal("first and last digit merges", [last9[0].pair.join(""), last9[0].count, last9[8].pair.join(""), last9[8].count], [" 8", 239, " 3", 194]);
  claim(id, "(239 times) down to"); claim(id, "(194)");
  near("merges", L.length, 74); claim(id, "After 74 merges");
  ok("every corpus word is one token", D.wordCounts.every(([w]) => M.encode(w).length === 1));
  near("query chars", D.running.length, 36); near("query tokens", qToks.length, 9); claim(id, "from 36 characters to 9 tokens");
}
{
  const id = "lm-encode";
  const RANK = new Map(D.merges.map((m, i) => [m[0] + "\u0001" + m[1], i]));
  let sy = Array.from(" username"); const used = [];
  for (;;) { let b = -1, br = Infinity; for (let i = 0; i < sy.length - 1; i++) { const r = RANK.get(sy[i] + "\u0001" + sy[i + 1]); if (r !== undefined && r < br) { br = r; b = i; } } if (b < 0) break; const a0 = sy[b], b0 = sy[b + 1]; used.push([br + 1, a0, b0]); const o = []; for (let i = 0; i < sy.length; i++) { if (i < sy.length - 1 && sy[i] === a0 && sy[i + 1] === b0) { o.push(a0 + b0); i++; } else o.push(sy[i]); } sy = o; }
  equal("username trace", used, [[16, "e", "r"], [18, "u", "s"], [23, " ", "us"], [49, " us", "er"]]);
  equal("username result", sy, M.encodeWord(" username"));
  claimH(id, "merge 16 (<code>e + r</code>) first, then merge 18 (<code>u + s</code>), merge 23 (<code>␣ + us</code>) and merge 49 (<code>␣us + er</code>)");
  ok("no merge joins user|n or n a m e", ![[" user", "n"], ["n", "a"], ["a", "m"], ["m", "e"]].some(([a, b]) => RANK.has(a + "\u0001" + b)));
  near("username tokens", sy.length, 5); claim(id, "the word ends as 5 tokens");
  near("new query tokens", M.encode("SELECT username FROM customers WHERE age = 42;").length, 20); claim(id, "The whole new query takes 20 tokens");
  ok("26 corpus words one token each", D.wordCounts.length === 26 && D.wordCounts.every(([w]) => M.encode(w).length === 1)); claim(id, "each of the 26 corpus words comes out as exactly one token");
}
{
  const id = "lm-decode";
  equal("decode query", qIds.map((i) => V[i]).join(""), D.running); claim(id, "43 → SELECT , 93 → ␣name"); claim(id, "10 → ;");
  const rt = (t) => M.encode(t).map(M.idOf).filter((i) => i >= 0).map((i) => V[i]).join("");
  equal("two spaces become one", rt("SELECT  name FROM users;"), "SELECT name FROM users;"); claim(id, "two spaces become one");
  equal("line break disappears", rt("SELECT name\n"), "SELECT name"); claim(id, "a line break disappears");
  ok("0 and é have no id", M.idOf("0") < 0 && M.idOf("é") < 0 && M.encode("10").includes("0")); claim(id, "outside its 39");
}
{
  const id = "lm-bytes";
  claim(id, "over 130,000"); claim(id, "32,000 to 64,000");  // Radford et al. 2019, Section 2.2
  near("byte values", 2 ** 8, 256); claim(id, "the 256 byte values");
  equal("é bytes", hexBytes("é"), "C3 A9"); claim(id, "the two tokens C3 and A9");
  near("日 bytes", Buffer.byteLength("日"), 3); claim(id, "<code>日</code> three tokens".replace(/<[^>]+>/g, " "));
  near("😀 bytes", Buffer.byteLength("😀"), 4); claim(id, "😀 four");
  near("gpt-2 merges", 50257 - 256 - 1, 50000); claim(id, "50,257 tokens: 256 bytes, 50,000 merges and one end-of-text marker");
  claim(id, "100K of its 128K tokens"); claim(id, "adds 28K");  // Llama 3 paper, Section 3.2
  equal("日 check bytes", hexBytes("日"), "E6 97 A5"); claim(id, "E6 97 A5");
}
{
  const id = "lm-vocabsize";
  near("tokens ratio", 1 - 3.17 / 3.94, 0.1954, 5e-5); claim(id, "takes " + r((1 - 3.17 / 3.94) * 100, 1) + "% fewer tokens"); claim(id, "3.94 characters per token"); claim(id, "averaged 3.17");
  near("32K tables", 2 * 32000 * 4096, 262144000); claim(id, "2 × 32,000 × 4,096 = 262 million");
  near("128K tables", 2 * 128256 * 4096, 1050673152); claim(id, "2 × 128,256 × 4,096 = 1.05 billion");
  // Llama 3 8B: 32 layers, d 4,096, FFN 14,336, 32 heads, 8 KV heads (head 128), vocabulary 128,256, untied output
  const d = 4096, L = 32, ffn = 14336, kv = 8 * 128;
  const p8b = 2 * 128256 * d + L * (2 * d * d + 2 * d * kv + 3 * d * ffn + 2 * d) + d;
  near("Llama 3 8B params", p8b, 8030261248);
  claim(id, r((2 * 128256 * d) / p8b * 100, 1) + "% of Llama 3 8B");
  near("output flops", 2 * 4096 * 128256 / 1e9, 1.05, 0.005); claim(id, "1.05 GFLOP");
  claim(id, "113 tokens instead of 39 characters cut our query from 36 positions to 9");
}
{
  const id = "lm-quirks";
  near("' users' id", M.idOf(" users"), 63); claim(id, "is the single token 63");
  equal("users without space", M.encode("users"), ["us", "ers"]); claim(id, "two tokens, us + ers");
  ok("no U", !V.includes("U")); claim(id, "has no U at all");
  ok("space-digit tokens", [1, 2, 3, 4, 5, 6, 7, 8, 9].every((k) => V.includes(" " + k))); claim(id, "Ours learned ␣1 to ␣9");
  equal("42", M.encode(" 42"), [" 4", "2"]); claim(id, "␣42 becomes ␣4 + 2");
  equal("llama 3 digit pieces", "2024".match(/\p{N}{1,3}/gu), ["202", "4"]); claim(id, "2024 becomes 202 + 4");
  claim(id, "up to 15 times"); claim(id, "over 4 times");  // Petrov et al. 2023
}
{
  const id = "lm-special";
  claim(id, "256 special ids after its 128,000 BPE tokens");
  near("vocab split", 128256 - 256, 128000);
  claim(id, "is 128,000 and"); claim(id, "is 128,001");
  const sl = M.running.length + 2, s2 = 7;  // BOS + 9 + EOS, and BOS SELECT id FROM orders ; EOS
  near("pairs per row", sl - 1, 10); near("row 2 counted", s2 - 1, 6); near("padded targets", sl - s2, 4);
  claim(id, "10 predictions from our query and 6 from the shorter one; the 4 padded targets");
  near("query tokens", M.running.length, 9); claim(id, "has exactly 9 tokens");
}

// ---------------------------------------------------------------- 4.3 Next-token prediction
const ALLQ = M.allQueries().map((q) => ({ ...q, tok: M.encode(q.text), ids: M.encode(q.text).map(M.idOf) }));
const qTrace = M.forward(qIds.slice(0, 8));
const qP = qTrace.probs.map((p, i) => p[qIds[i + 1]]);
// exact conditional probability of the next token under the grammar
function grammarCond(prefixToks, next) {
  let den = 0, num = 0;
  for (const q of ALLQ) {
    if (prefixToks.every((t, i) => q.tok[i] === t)) { den += q.p; if (q.tok[prefixToks.length] === next) num += q.p; }
  }
  return num / den;
}
{
  const id = "lm-predict";
  near("vocab", V.length, 113); claim(id, "Ours returns 113 numbers that sum to 1");
  near("sum to 1", qTrace.probs[0].reduce((a, b) => a + b, 0), 1, 1e-12);
  ok("p(users) rounds to 1.000", qP[2] > 0.9995); claim(id, "gives <code>users</code> 1.000".replace(/<[^>]+>/g, " "));
  ok("only users has name", Object.entries(D.schema).filter(([, c]) => c.includes("name")).map(([t]) => t).join() === "users");
  const p4 = qTrace.probs[4];
  near("grammar after WHERE", grammarCond(qToks.slice(0, 5), " id"), 1 / 3, 1e-12);
  claimH(id, "<code>id</code> 0." + r(p4[M.idOf(" id")], 3).slice(2) + ", <code>email</code> " + r(p4[M.idOf(" email")], 3) + " and <code>age</code> " + r(p4[M.idOf(" age")], 3));
  const p6 = qTrace.probs[6], dig = [1, 2, 3, 4, 5, 6, 7, 8, 9].map((k) => p6[M.idOf(" " + k)]);
  claim(id, `between ${r(Math.min(...dig), 3)} and ${r(Math.max(...dig), 3)}`); claim(id, "1/9 = 0.111");
  near("grammar id after SELECT", grammarCond(["SELECT"], " id"), 5 / 18, 1e-12); claim(id, "5/18 = 0.278");
  claim(id, "the model gives " + r(qTrace.probs[0][M.idOf(" id")], 3));
}
{
  const id = "lm-chain";
  const g = qToks.slice(1).map((t, i) => grammarCond(qToks.slice(0, i + 1), t));
  equal("grammar factors", g.map((x) => Math.round(x * 324) / 324), [1 / 12, 1, 1, 1, 1 / 3, 1, 1 / 9, 1]);
  near("p(query) grammar", g.reduce((a, b) => a * b, 1), 1 / 324, 1e-15);
  near("p(SELECT) first", ALLQ.every((q) => q.tok[0] === "SELECT") ? 1 : 0, 1);
  const pm = qP.reduce((a, b) => a * b, 1);
  // Part 1's numbers, recalled here
  claim(id, "1/324 = " + r(1 / 324, 6) + " under the grammar, " + r(pm, 6) + " under the model");
  let tot = 0;
  for (const q of ALLQ) { const t = M.forward(q.ids.slice(0, 8)); let pp = 1; for (let i = 0; i < 8; i++) pp *= t.probs[i][q.ids[i + 1]]; tot += pp; }
  near("valid queries", ALLQ.length, 270);
  claim(id, "add up to " + r(tot, 4)); claim(id, "The other " + r(1 - tot, 4) + " goes"); claim(id, "= " + r(tot, 4));
  // where the leak goes: probability that leaves the valid prefixes at each position
  const validPre = new Set(); for (const q of ALLQ) for (let k = 1; k <= 9; k++) validPre.add(q.tok.slice(0, k).join("|"));
  const pre = new Map(); for (const q of ALLQ) for (let k = 1; k <= 8; k++) pre.set(q.tok.slice(0, k).join("|"), q.ids.slice(0, k));
  const leak = new Array(8).fill(0);
  for (const [key, ids] of pre) { const k = ids.length, t = M.forward(ids); let pp = 1; for (let i = 0; i < k - 1; i++) pp *= t.probs[i][ids[i + 1]]; leak[k - 1] += pp * t.probs[k - 1].reduce((a, v, j) => a + (validPre.has(key + "|" + V[j]) ? 0 : v), 0); }
  near("leak adds up", leak.reduce((a, b) => a + b), 1 - tot, 1e-9);
  ok("the filter column leaks most", leak[4] === Math.max(...leak));
  claim(id, r(leak[4], 4) + " of it at the filter column");
  claimH(id, "the column already selected, <code>name</code>, " + r(qTrace.probs[4][M.idOf(" name")], 5));
  const bad = M.encode("SELECT name FROM users WHERE name = 7;").map(M.idOf), tb = M.forward(bad.slice(0, 8));
  let pbad = 1; for (let i = 0; i < 8; i++) pbad *= tb.probs[i][bad[i + 1]];
  ok("invalid query not in the grammar", !ALLQ.some((q) => q.text === "SELECT name FROM users WHERE name = 7;"));
  claim(id, "gets " + r(pbad * 1e5, 1) + " × 10⁻⁵ instead of 0");
}
{
  const id = "lm-pairs";
  near("pairs", qToks.length - 1, 8); claim(id, "Our 9-token query gives 8");
  near("context", C.T, 8); claim(id, "context length T = 8");
  claim(id, "1/113 = " + r(1 / 113, 4));
  const e0 = M.exactLoss(0).mean;
  claim(id, "expected loss is " + r(e0, 2) + " nats"); claim(id, "ln 113 = " + r(Math.log(113), 2));
  ok("determined targets 1.000", [1, 2, 3, 5, 7].every((i) => qP[i] > 0.9995)); claim(id, "the five determined targets get 1.000");
  near("pairs per step", C.batch * 8, 1024); claim(id, "128 queries × 8 = 1,024 pairs");
  near("pairs per run", C.batch * 8 * C.steps, 6144000); claim(id, "6,144,000");
}
{
  const id = "lm-loss";
  const losses = qP.map((p) => -Math.log(p));
  const L = losses.reduce((a, b) => a + b) / 8;
  claim(id, "ln p(query) = −" + r(-Math.log(qP.reduce((a, b) => a * b, 1)), 4) + " under the model, a loss of " + r(-Math.log(qP.reduce((a, b) => a * b, 1)), 4) + " / 8 = " + r(L, 4) + " nats per token");
  near("ideal", Math.log(324) / 8, 0.72259, 1e-5); claim(id, "\\ln 324 / 8 = " + r(Math.log(324) / 8, 4));
  claim(id, r(qP[0], 4) + " on"); claim(id, "(exact: 0.0833)"); claim(id, r(qP[4], 3) + " on"); claim(id, "(exact: 0.333)");
  near("batch preds", C.batch * 8, 1024); claim(id, "1,024 predictions per step");
  const pm = qP.reduce((a, b) => a * b, 1);
  claim(id, `L = ${r(-Math.log(pm), 4)} / 8 = ${r(L, 4)}`); claim(id, "ln p(query) = −" + r(-Math.log(pm), 4) + " under the model (Part 1)");
  near("rounded factors reproduce the product", 0.0854 * 0.346 * 0.1077, pm, 5e-7);
  equal("factors as written", [r(qP[0], 4), r(qP[4], 3), r(qP[6], 4)], ["0.0854", "0.346", "0.1077"]);
  ok("other factors round to 1.000", [1, 2, 3, 5, 7].every((i) => r(qP[i], 3) === "1.000"));
  near("five other factors contribute nothing at 4 dp", losses.reduce((a, b) => a + b) - losses[0] - losses[4] - losses[6], 0, 5e-5);
  near("rest of the losses", losses.reduce((a, b) => a + b) - losses[0] - losses[4] - losses[6], 0, 1e-4);
  claim(id, "\\tfrac{1}{8} \\ln 324 = " + r(Math.log(324) / 8, 4));
}
{
  const id = "lm-floor";
  const ex = M.exactLoss(), fl = D.floor, flm = fl.reduce((a, b) => a + b) / 8;
  claim(id, "all 270 queries: " + r(ex.mean, 4));
  // floor per position from the grammar: entropy of the next token given the full prefix
  const H = new Array(8).fill(0);
  for (let i = 0; i < 8; i++) {
    const groups = new Map();
    for (const q of ALLQ) { const k = q.tok.slice(0, i + 1).join("|"); if (!groups.has(k)) groups.set(k, new Map()); const m = groups.get(k); m.set(q.tok[i + 1], (m.get(q.tok[i + 1]) || 0) + q.p); }
    for (const m of groups.values()) { const tot = [...m.values()].reduce((a, b) => a + b); for (const v of m.values()) H[i] -= v * Math.log(v / tot); }
  }
  H.forEach((x, i) => near(`floor pos ${i}`, x, fl[i], 1e-6));
  claim(id, r(H[0], 3) + " for the first column"); claim(id, r(H[2], 3) + " for the table"); claim(id, r(H[4], 3) + " for the filter column"); claim(id, "ln 9 = " + r(Math.log(9), 3));
  near("table floor = P(id) H(t|id)", (5 / 18) * -(2 * 0.3 * Math.log(0.3) + 0.4 * Math.log(0.4)), H[2], 1e-12); claim(id, "uncertain only after <code>id</code>".replace(/<[^>]+>/g, " "));
  ok("four determined positions", [1, 3, 5, 7].every((i) => H[i] === 0)); claim(id, "0 at the four positions");
  claim(id, "it is " + r(flm, 4)); claim(id, r(ex.mean - flm, 4) + " above it");
  claim(id, "(" + r(qP.map((p) => -Math.log(p)).reduce((a, b) => a + b) / 8, 4) + " against " + r(Math.log(324) / 8, 4) + ", previous sheet) and still sit above the average " + r(ex.mean, 4));
  claimH(id, "<code>name</code> costs ln 12 = " + r(Math.log(12), 3) + " nats against an average of " + r(H[0], 3));
  claimH(id, "<code>id</code> ln 3 = " + r(Math.log(3), 3) + " against " + r(H[4], 3));
  ok("our choices cost more than the position averages", Math.log(12) > H[0] && Math.log(3) > H[4]);
}
{
  const id = "lm-bpb";
  const L = M.exactLoss().mean, perQuery = 8 * L;
  const chars = ALLQ.reduce((a, q) => a + q.p * Array.from(q.text).length, 0), bytes = ALLQ.reduce((a, q) => a + q.p * Buffer.byteLength(q.text), 0);
  near("chars = bytes (ASCII)", chars, bytes, 1e-12); near("average query length", chars, 39 + 2 / 3, 1e-9);
  claim(id, "0.6942 nats for each of the 8 predictions of a query, " + r(perQuery, 3) + " nats per query: a perplexity of " + r(Math.exp(L), 3) + " per token");
  claim(id, "averages " + r(chars, 2) + " characters"); claim(id, "the same " + r(perQuery, 3) + " nats are " + r(perQuery / chars, 4) + " nats per character, a perplexity of " + r(Math.exp(perQuery / chars), 3) + " per character");
  claim(id, r(perQuery, 3) + " / (" + r(chars, 2) + " × ln 2) = " + r(perQuery / bytes / Math.LN2, 3) + " bits per byte");
  claim(id, "enwik8 in bits per byte"); claim(id, "The Pile prefers bits per byte");  // GPT-2 Section 3.1 / Table 3; Gao et al. 2020 Section 3.1
  // the check: per-character cost of two models with different tokenizers
  const a = Math.log(12) / 3.17, b = Math.log(15) / 3.94;
  ok("B is better per character", b < a);
  claim(id, "ln 12 / 3.17 = " + r(a, 3) + " nats and B pays ln 15 / 3.94 = " + r(b, 3) + " nats: " + r(a / Math.LN2, 3) + " against " + r(b / Math.LN2, 3) + " bits per character");
}
// ---------------------------------------------------------------- 4.4 Counting models
const corpusTok = corpus.map((q) => M.encode(q));
const big = new Map(), bigTot = new Map();
for (const t of corpusTok) for (let i = 0; i < 8; i++) { const k = t[i] + "|" + t[i + 1]; big.set(k, (big.get(k) || 0) + 1); bigTot.set(t[i], (bigTot.get(t[i]) || 0) + 1); }
const bc = (a, b) => big.get(a + "|" + b) || 0, bp = (a, b) => bc(a, b) / bigTot.get(a);
// best possible n-gram: the grammar's exact conditional given the last n − 1 tokens
function ngram(n) {
  const ctx = (tok, i) => (n === 1 ? "" : (i + 2 - n <= 0 ? "^" : "") + tok.slice(Math.max(0, i + 2 - n), i + 1).join("|"));
  const mass = new Map();
  for (const q of ALLQ) for (let i = 0; i < 8; i++) { const k = ctx(q.tok, i); if (!mass.has(k)) mass.set(k, new Map()); const m = mass.get(k); m.set(q.tok[i + 1], (m.get(q.tok[i + 1]) || 0) + q.p); }
  const per = new Array(8).fill(0);
  for (const q of ALLQ) for (let i = 0; i < 8; i++) { const m = mass.get(ctx(q.tok, i)); const tot = [...m.values()].reduce((a, b) => a + b); per[i] -= q.p * Math.log(m.get(q.tok[i + 1]) / tot); }
  return { per, mean: per.reduce((a, b) => a + b) / 8, contexts: mass.size };
}
const NG = [1, 2, 3, 4, 5, 9].map(ngram);
{
  const id = "lm-bigram";
  near("pairs", [...big.values()].reduce((a, b) => a + b), 16000); claim(id, "16,000 adjacent pairs, 8 per query");
  const allowed = new Set(); for (const q of ALLQ) for (let i = 0; i < 8; i++) allowed.add(q.tok[i] + "|" + q.tok[i + 1]);
  near("distinct pairs", big.size, 60); near("allowed pairs", allowed.size, 60); ok("same pairs", [...big.keys()].every((k) => allowed.has(k)));
  claim(id, "only 60 distinct ones: exactly the 60 the grammar allows");
  equal("FROM row", [bc(" FROM", " users"), bc(" FROM", " products"), bc(" FROM", " orders"), bigTot.get(" FROM")], [682, 660, 658, 2000]);
  claimH(id, "<code>users</code> 682, <code>products</code> 660 and <code>orders</code> 658 of 2,000");
  claim(id, "p(users | FROM) = " + r(bp(" FROM", " users"), 3));
  equal("id row", [bigTot.get(" id"), bc(" id", " FROM"), bc(" id", " =")], [1093, 577, 516]);
  claim(id, "1,093 times: 577 times"); claim(id, "516 times as the filter column");
  claim(id, `splits its prediction ${r(bp(" id", " FROM"), 3)} to ${r(bp(" id", " ="), 3)}`);
}
{
  const id = "lm-bisample";
  const cols = [...new Set(Object.values(D.schema).flat())].map((c) => " " + c);
  const five = cols.reduce((a, c) => a + bp("SELECT", c) * bp(c, " ="), 0);
  claim(id, r(five * 100, 1) + "% of samples stop after five tokens");
  ok("SELECT id = 4; possible", bc("SELECT", " id") && bc(" id", " =") && bc(" =", " 4") && bc(" 4", ";")); claim(id, "SELECT id = 4;");
  const valid = ALLQ.reduce((a, q) => a + q.tok.slice(0, 8).reduce((pp, t, i) => pp * bp(t, q.tok[i + 1]), 1), 0);
  claim(id, "Only " + r(valid * 100, 1) + "% are valid queries");
  const b2 = NG[1];
  claim(id, "expected loss is " + r(b2.mean, 3) + " nats per token");
  near("bigram after column", b2.per[1], Math.log(2), 1e-12); near("bigram after c2", b2.per[5], Math.log(2), 1e-12); claim(id, "ln 2 = " + r(Math.log(2), 3) + " after each column name");
  near("bigram table", b2.per[2], Math.log(3), 1e-12); claim(id, "ln 3 = " + r(Math.log(3), 3) + " for the table instead of " + r(D.floor[2], 3));
  claim(id, r(b2.per[4], 3) + " instead of " + r(D.floor[4], 3) + " for the filter column");
  claim(id, "The transformer reaches " + r(M.exactLoss().mean, 4) + ", the floor is " + r(D.floor.reduce((a, b) => a + b) / 8, 4));
}
{
  const id = "lm-ngram";
  const [u1, b2, t3, f4, g5, w9] = NG;
  claim(id, r(u1.mean, 3) + " nats per token with no context"); claim(id, r(b2.mean, 3) + " for the bigram"); claim(id, r(t3.mean, 3) + " for the trigram");
  claim(id, r(f4.mean, 3) + " again for the 4-gram"); near("4-gram equals trigram", f4.mean, t3.mean, 1e-12);
  near("5-gram reaches the floor", g5.mean, D.floor.reduce((a, b) => a + b) / 8, 1e-6); claim(id, r(g5.mean, 4) + " for the 5-gram: the floor");
  near("9-gram = 5-gram", w9.mean, g5.mean, 1e-12);
  near("5-gram contexts", g5.contexts, 144); claim(id, "only 144 distinct 4-token contexts");
  near("113^4", 113 ** 4 / 1e8, 1.63, 0.005); claim(id, "113^4 = 1.6 \\times 10^8");
  near("bengio 10^50", Math.log10(100000 ** 10), 50, 1e-9); claim(id, "10^{50} - 1");
}
{
  const id = "lm-smooth";
  near("distinct corpus queries", new Set(corpus).size, 269); claim(id, "contain 269 of the 270 possible ones");
  const missing = ALLQ.filter((q) => !corpus.includes(q.text)).map((q) => q.text);
  equal("missing query", missing, ["SELECT user_id FROM orders WHERE id = 2;"]); claim(id, missing[0]);
  const Vn = 113;
  const ctx = (tok, i, n) => (i + 2 - n <= 0 ? "^" : "") + tok.slice(Math.max(0, i + 2 - n), i + 1).join("|");
  const model = (N, n) => {
    const c = new Map(), tot = new Map();
    for (const tok of corpusTok.slice(0, N)) for (let i = 0; i < 8; i++) { const k = ctx(tok, i, n); tot.set(k, (tot.get(k) || 0) + 1); c.set(k + ">" + tok[i + 1], (c.get(k + ">" + tok[i + 1]) || 0) + 1); }
    return (a) => { let L = 0; for (const q of ALLQ) for (let i = 0; i < 8; i++) { const k = ctx(q.tok, i, n); const p = ((c.get(k + ">" + q.tok[i + 1]) || 0) + a) / ((tot.get(k) || 0) + a * Vn); if (!(p > 0)) return Infinity; L -= (q.p / 8) * Math.log(p); } return L; };
  };
  const whole = model(2000, 9);
  ok("whole prefix infinite at alpha 0", whole(0) === Infinity);
  claim(id, "With α = 0.0001 the whole-prefix model scores " + r(whole(1e-4), 3) + " nats");
  claim(id, "at α = 1 the loss is " + r(whole(1), 3));
  const best = (f) => { let b = Infinity; for (let e = -6; e <= 1; e += 0.002) b = Math.min(b, f(10 ** e)); return b; };
  const tri100 = best(model(100, 3)), five100 = best(model(100, 5));
  claim(id, `the trigram beats the 5-gram, ${r(tri100, 3)} against ${r(five100, 3)}`);
  ok("trigram limit above 5-gram limit", NG[2].mean > NG[4].mean);
  claim(id, "1/113 per token");
}
{
  const id = "lm-share";
  near("whole-prefix rows", NG[5].contexts, 371); claim(id, "number 371, which is 41,923 probabilities"); near("rows x V", 371 * 113, 41923);
  near("113^8", 113 ** 8 / 1e16, 2.66, 0.005); claim(id, "113^8 = 2.7 \\times 10^{16}");
  near("held-out queries", D.holdout.queries.length, 27); claim(id, "trained without 27 of the 270 queries");
  // recompute the held-out model's loss on its held-out queries (per-token mean, weighted by query probability)
  const held = new Set(D.holdout.queries);
  let num = 0, den = 0;
  for (const q of ALLQ) if (held.has(q.text)) { const t = M.forward(q.ids.slice(0, 8), "holdout"); let l = 0; for (let i = 0; i < 8; i++) l -= Math.log(t.probs[i][q.ids[i + 1]]); num += q.p * l / 8; den += q.p; }
  near("held-out loss", num / den, D.holdout.lossB.held, 1e-4); claim(id, "a loss of " + r(num / den, 2) + " nats per token");
  let numS = 0, denS = 0;
  for (const q of ALLQ) if (!held.has(q.text)) { const t = M.forward(q.ids.slice(0, 8), "holdout"); let l = 0; for (let i = 0; i < 8; i++) l -= Math.log(t.probs[i][q.ids[i + 1]]); numS += q.p * l / 8; denS += q.p; }
  near("seen-query loss", numS / denS, D.holdout.lossB.seen, 1e-4); near("seen queries", ALLQ.filter((q) => !held.has(q.text)).length, 243);
  claim(id, "against " + r(numS / denS, 2) + " on the 243 queries it trained on and " + r(Math.log(113), 2) + " for a uniform guess");
  // a whole-prefix count model of the 243 training queries: the first unseen extension of each held-out query has a seen prefix, so its product is 0
  const train = ALLQ.filter((q) => !held.has(q.text));
  ok("count model gives held-out queries probability 0", ALLQ.filter((q) => held.has(q.text)).every((q) => {
    for (let i = 0; i < 8; i++) {
      const pre = train.filter((t) => q.tok.slice(0, i + 1).every((x, j) => t.tok[j] === x));
      if (!pre.some((t) => t.tok[i + 1] === q.tok[i + 1])) return pre.length > 0;
    }
    return false;
  }));
  claim(id, "gives every one of them probability 0");
  near("window params k=4", (113 + 1) * 16 + 4 * 16 * 64 + 64 + 64 * 113 + 113, 13329);
  near("check contexts", 50000 ** 2, 2.5e9); near("check entries", 50000 ** 3, 1.25e14);
  claim(id, "50,000² = 2.5 × 10⁹"); claim(id, "1.25 × 10¹⁴");
}

// ---------------------------------------------------------------- 4.5 Neural language models
// printed by tools/verify/part04_window.py (one JSON line per window size)
const WINDOW = [
  { k: 1, params: 10257, loss: 1.107097, pos4: 2.086894, afterWhere: { id: 0.2846, name: 0.0851, email: 0.0844, age: 0.0814 } },
  { k: 2, params: 11281, loss: 0.735045, pos4: 1.291816, afterWhere: { id: 0.2492, name: 0.2518, email: 0.255, age: 0.2435 } },
  { k: 3, params: 12305, loss: 0.735064, pos4: 1.291919, afterWhere: { id: 0.2501, name: 0.2522, email: 0.2534, age: 0.2438 } },
  { k: 4, params: 13329, loss: 0.694291, pos4: 0.965468, afterWhere: { id: 0.3401, name: 0.0007, email: 0.3501, age: 0.3087 } },
  { k: 8, params: 17425, loss: 0.694374, pos4: 0.9659, afterWhere: { id: 0.3416, name: 0.001, email: 0.3476, age: 0.3093 } },
];
const cosv = (a, b) => { let d = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; } return d / Math.sqrt(na * nb); };
const W0 = M.weights(0), WF = M.weights();
const inputs = [...new Set(ALLQ.flatMap((q) => q.tok.slice(0, 8)))];
{
  const id = "lm-embed";
  equal("wte shape", WF.wte.shape, [113, 16]); claim(id, "Ours is 113 × 16, and token 63");
  near("one-hot product flops", 2 * 113 * 16, 3616); claim(id, "2 × 113 × 16 = 3,616 FLOPs");
  near("input tokens", inputs.length, 25); claim(id, "only 25 of our 113 tokens ever are"); claim(id, "Because our output layer is a separate matrix");
  ok("; never an input", !inputs.includes(";"));
  // rows of tokens that are never inputs differ from the step-0 snapshot only by its float16 rounding
  const f16ulps = (t) => { const a = M.row(W0.wte, M.idOf(t)), b = M.row(WF.wte, M.idOf(t)); let w = 0; for (let i = 0; i < 16; i++) w = Math.max(w, Math.abs(a[i] - b[i]) / 2 ** (Math.floor(Math.log2(Math.abs(b[i]))) - 10)); return w; };
  const unchanged = V.filter((t) => f16ulps(t) <= 0.5);
  near("unchanged rows", unchanged.length, 88); ok("unchanged = non-inputs", unchanged.every((t) => !inputs.includes(t)) && V.filter((t) => !inputs.includes(t)).length === 88);
  ok("input rows all changed by > 100 ulps", inputs.every((t) => f16ulps(t) > 100));
  claim(id, "The other 88 rows end training exactly as they started");
  equal("wpe shape", WF.wpe.shape, [8, 16]); claim(id, "8 × 16 position rows");
  near("grammar tokens", new Set(ALLQ.flatMap((q) => q.tok)).size, 26); claim(id, "the 26 token rows the grammar uses");
  claim(id, "128,256 rows × 4,096 numbers");
}
{
  const id = "lm-cosine";
  const digits = [1, 2, 3, 4, 5, 6, 7, 8, 9].map((k) => " " + k), tables = [" users", " orders", " products"];
  const pairs = (W, A) => { const v = []; for (let i = 0; i < A.length; i++) for (let j = i + 1; j < A.length; j++) v.push(cosv(M.row(W.wte, M.idOf(A[i])), M.row(W.wte, M.idOf(A[j])))); return v; };
  const mean = (a) => a.reduce((x, y) => x + y) / a.length;
  const d0 = pairs(W0, digits), d1 = pairs(WF, digits), t0 = pairs(W0, tables), t1 = pairs(WF, tables);
  claim(id, "average pairwise cosine of " + r(mean(d0), 2));
  claim(id, "the nine digits average " + r(mean(d1), 2) + ", between " + r(Math.min(...d1), 2) + " and " + r(Math.max(...d1), 3));
  ok("digits 1/9 each after =", [1, 2, 3, 4, 5, 6, 7, 8, 9].every((k) => Math.abs(grammarCond(qToks.slice(0, 7), " " + k) - 1 / 9) < 1e-12));
  ok("digits followed only by ;", ALLQ.every((q) => q.tok[8] === ";"));
  const nrm = (W) => mean(digits.map((t) => Math.hypot(...M.row(W.wte, M.idOf(t)))));
  claim(id, "from " + r(nrm(W0), 1) + " to " + r(nrm(WF), 1) + " on average");
  claim(id, "from an average cosine of " + r(mean(t0), 2) + " to " + r(mean(t1), 2));
  const up0 = cosv(M.row(W0.wte, 63), M.row(W0.wte, M.idOf(" products"))), up1 = cosv(M.row(WF.wte, 63), M.row(WF.wte, M.idOf(" products")));
  claimH(id, "<code>users</code> and <code>products</code> go from " + r(up0, 2) + " to " + r(up1, 2));
}
{
  const id = "lm-window";
  near("embedding", 114 * 16, 1824); claim(id, "114 × 16 = 1,824");
  near("hidden", 48 * 64 + 64, 3136); claim(id, "48 × 64 + 64 = 3,136");
  near("output", 64 * 113 + 113, 7345); claim(id, "64 × 113 + 113 = 7,345");
  near("total k=3", 1824 + 3136 + 7345, 12305); claim(id, "12,305 in all");
  near("matches the experiment", 12305, WINDOW.find((w) => w.k === 3).params);
  claim(id, "test perplexity of 252"); claim(id, "312 is 24% higher"); near("bengio ratio", 312 / 252, 1.238, 0.001);
  claim(id, "1.18-million-word");
}
{
  const id = "lm-output";
  near("our output flops", 2 * 16 * 113, 3616); claim(id, "1,808 multiply-adds, 3,616 FLOPs per position");
  const mm = C.L * (4 * 16 * 16 + 2 * 16 * 64) + 16 * 113;
  near("our matmul params", mm, 7952); claim(id, r((16 * 113) / mm * 100, 1) + "% of all the FLOPs");
  const L3 = (L, d, ffn) => L * (2 * d * d + 2 * d * 1024 + 3 * d * ffn);
  const mm8 = L3(32, 4096, 14336) + 4096 * 128256;
  near("8B output gflop", 2 * 4096 * 128256 / 1e9, 1.05, 0.005); claim(id, "1.05 GFLOP per token");
  claim(id, r(4096 * 128256 / mm8 * 100, 1) + "% of the " + r(2 * mm8 / 1e9, 1) + " GFLOP");
  claim(id, "are " + r(8192 * 128256 * 4 / 1e9, 1) + " GB for one sequence");
}
{
  const id = "lm-params";
  near("our tables", 2 * 113 * 16, 3616); claim(id, "3,616 together, " + r(3616 / C.params * 100, 1) + "% of our 9,968");
  const tot = (L, d, ffn) => L * (2 * d * d + 2 * d * 1024 + 3 * d * ffn) + L * 2 * d + d + 2 * d * 128256;
  const t8 = tot(32, 4096, 14336), t70 = tot(80, 8192, 28672), t405 = tot(126, 16384, 53248);
  near("8B", t8, 8030261248); near("70B", t70, 70553706496); near("405B", t405, 405853388800);
  claim(id, "128,256 × 4,096 = 525 million"); claim(id, "1.05 billion of " + r(t8 / 1e9, 2) + " billion, " + r(2 * 4096 * 128256 / t8 * 100, 1) + "%");
  claim(id, r(2 * 8192 * 128256 / t70 * 100, 1) + "% of its parameters"); claim(id, "405B (d = 16,384, 126 layers) " + r(2 * 16384 * 128256 / t405 * 100, 1) + "%");
}
{
  const id = "lm-limits";
  // results printed by tools/verify/part04_window.py
  const ref = { 1: 1.107097, 2: 0.735045, 3: 0.735064, 4: 0.694291, 8: 0.694374 };
  const params = (k) => 114 * 16 + k * 16 * 64 + 64 + 64 * 113 + 113;
  for (const w of WINDOW) { near(`window ${w.k} params`, params(w.k), w.params); near(`window ${w.k} loss`, w.loss, ref[w.k], 1e-9); claim(id, int(w.params)); claim(id, r(w.loss, 4)); }
  const lim = (k) => NG[Math.min(k, 4)].mean;
  ok("each window is at or above its n-gram limit", WINDOW.every((w) => w.loss >= lim(w.k) - 1e-6 && w.loss - lim(w.k) < 0.002));
  claim(id, r(WINDOW[0].loss, 3) + " for k = 1"); claim(id, r(WINDOW[1].loss, 3) + " for k = 2 and again for k = 3"); near("k3 = k2", WINDOW[2].loss, WINDOW[1].loss, 1e-4);
  claim(id, r(WINDOW[3].loss, 4) + " for k = 4, level with the transformer's " + r(M.exactLoss().mean, 4));
  near("8192-window first layer", 8192 * 4096 * 14336 / 1e11, 4.81, 0.01); claim(id, "4.8 × 10¹¹ weights");
  claim(id, "stay at " + r(WINDOW[2].pos4, 2)); claim(id, "floor is " + r(D.floor[4], 3));
  claim(id, "this position costs the k = 3 model " + r(WINDOW[2].pos4, 2) + " nats against the floor's " + r(D.floor[4], 2));
  near("m·h", 16 * 64, 1024); claim(id, "each of the k slots has its own m·h = 1,024 first-layer weights");
  const w3 = WINDOW[2].afterWhere, w4 = WINDOW[3].afterWhere;
  ok("k = 3 gives the four users columns about 1/4", ["id", "name", "email", "age"].every((c) => Math.abs(w3[c] - 0.25) < 0.01));
  claimH(id, "<code>name</code> included (" + r(w3.name, 3) + ")");
  claimH(id, "<code>id</code> costs −ln " + r(w3.id, 3) + " = " + r(-Math.log(w3.id), 2) + " nats, against ln 3 = " + r(Math.log(3), 2));
  claimH(id, "the k = 4 model gives <code>name</code> " + r(w4.name, 4));
}

// ---------------------------------------------------------------- 4.6 Generating text
const QL = ALLQ.map((q) => ({ ids: q.ids, z: M.forward(q.ids.slice(0, 8)).logits }));
const validAt = (T) => QL.reduce((tot, q) => { let pp = 1; for (let i = 0; i < 8; i++) pp *= M.softmax(q.z[i], T)[q.ids[i + 1]]; return tot + pp; }, 0);
const pct2 = (x) => r(x * 100, 2) + "%";
{
  const id = "lm-generate";
  claim(id, "our model needs 8 passes");
  claim(id, "valid with probability " + pct2(validAt(1))); claim(id, "the other " + r((1 - validAt(1)) * 100, 2) + "%");
  claim(id, "at temperature 2 the valid share falls to " + pct2(validAt(2))); claim(id, "at 3 to " + pct2(validAt(3)));
  near("validAt(1) equals the summed model probability", validAt(1), ALLQ.reduce((a, q) => { const t = M.forward(q.ids.slice(0, 8)); let pp = 1; for (let i = 0; i < 8; i++) pp *= t.probs[i][q.ids[i + 1]]; return a + pp; }, 0), 1e-12);
}
const greedyFrom = (ids) => { const cur = ids.slice(), steps = []; while (cur.length < 9) { const p = M.nextDist(cur); const o = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]); steps.push(o); cur.push(o[0][1]); if (V[o[0][1]] === ";") break; } return { text: cur.map((i) => V[i]).join(""), steps }; };
{
  const id = "lm-greedy";
  const g = greedyFrom([qIds[0]]);
  equal("greedy from SELECT", g.text, "SELECT id FROM products WHERE title = 3;"); claim(id, g.text);
  const last = g.steps[6];
  equal("digit step top two", [V[last[0][1]], V[last[1][1]]], [" 3", " 6"]);
  claimH(id, `<code>3</code> ${r(last[0][0], 4)} and <code>6</code> ${r(last[1][0], 4)}`);
  claim(id, "a difference of " + r(last[0][0] - last[1][0], 4));
  near("two-step example greedy", 0.6 / 3, 0.2, 1e-12); claim(id, "probability 0.2; B's sequence has 0.4");
  near("grammar p of greedy query", ALLQ.find((q) => q.text === g.text).p, 1 / 162, 1e-15);
}
{
  const id = "lm-temperature";
  const z = M.forward([qIds[0]]).logits[0];
  const zt = (t) => z[M.idOf(t)];
  claimH(id, `${r(zt(" id"), 3)} for <code>id</code>, ${r(zt(" price"), 3)} for <code>price</code>, ${r(zt(" title"), 3)} for <code>title</code>`);
  const others = [" name", " email", " age", " user_id", " total", " status"].map(zt);
  claim(id, `between ${r(Math.min(...others), 2)} and ${r(Math.max(...others), 2)} for the other six columns`);
  const cols = new Set([" id", " name", " email", " age", " user_id", " total", " status", " title", " price"]);
  const nonCols = V.map((t, i) => [t, i]).filter(([t]) => !cols.has(t));
  near("104 impossible tokens", nonCols.length, 104); ok("all <= -5.1", Math.max(...nonCols.map(([, i]) => z[i])) <= -5.09);
  claim(id, "−5.1 or less for the 104 tokens");
  const p = (T) => M.softmax(z, T);
  claim(id, "At T = 1, p(id) = " + r(p(1)[M.idOf(" id")], 3)); claim(id, "doubles and p(id) = " + r(p(0.5)[M.idOf(" id")], 3)); claim(id, "the gaps halve, p(id) = " + r(p(2)[M.idOf(" id")], 3));
  const inv = (T) => nonCols.reduce((a, [, i]) => a + p(T)[i], 0);
  claim(id, "share " + r(inv(2) * 100, 1) + "% instead of " + r(inv(1) * 100, 3) + "%");
  claim(id, "from " + pct2(validAt(1)) + " at T = 1 to " + pct2(validAt(2)) + " at T = 2 and " + pct2(validAt(3)) + " at T = 3");
}
{
  const id = "lm-topkp";
  const pre = { SELECT: qIds.slice(0, 1), FROM: qIds.slice(0, 3), WHERE: qIds.slice(0, 5), EQ: qIds.slice(0, 7) };
  const allowed = (ids) => { const t = ids.map((i) => V[i]); return new Set(ALLQ.filter((q) => t.every((x, j) => q.tok[j] === x)).map((q) => q.tok[t.length])); };
  equal("allowed counts", Object.values(pre).map((ids) => allowed(ids).size), [9, 1, 3, 9]);
  claimH(id, "allows 9 tokens after <code>SELECT</code>, 1 after <code>SELECT name FROM</code>, 3 after <code>…users WHERE</code> and 9 after <code>…id =</code>");
  const kept = (ids, opt) => { const q = M.filterProbs(M.nextDist(ids), opt), al = allowed(ids); const k = q.map((v, i) => [v, V[i]]).filter(([v]) => v > 0).map(([, t]) => t); return { n: k.length, valid: k.filter((t) => al.has(t)).length, bad: k.filter((t) => !al.has(t)).length }; };
  near("k=3 after SELECT drops six", 9 - kept(pre.SELECT, { topK: 3 }).valid, 6); claim(id, "With k = 3, six valid columns can never start a query");
  near("k=9 after FROM keeps 8 impossible", kept(pre.FROM, { topK: 9 }).bad, 8); claim(id, "with k = 9, eight impossible tokens stay in play");
  ok("p=0.95 keeps exactly the allowed tokens", Object.values(pre).every((ids) => { const k = kept(ids, { topP: 0.95 }); return k.bad === 0 && k.valid === allowed(ids).size; }));
  claim(id, "With p = 0.95 it keeps exactly the allowed tokens at all four positions");
  const k9 = kept(pre.SELECT, { topP: 0.9 }); near("p=0.9 after SELECT", k9.n, 8);
  const ps = M.nextDist(pre.SELECT); const top8 = ps.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 9);
  equal("dropped column", V[top8[8][1]], " status");
  claim(id, "add up to " + r(top8.slice(0, 8).reduce((a, [v]) => a + v, 0), 4));
  near("queries starting with status", ALLQ.filter((q) => q.tok[1] === " status").length, 27); claim(id, "the 27 valid queries that start with it");
}
{
  const id = "lm-cost";
  claim(id, "our model takes 8 passes");
  near("recompute positions", [1, 2, 3, 4, 5, 6, 7, 8].reduce((a, b) => a + b), 36); claim(id, "1 + 2 + … + 8 = 36 positions");
  const L3 = 32 * (2 * 4096 * 4096 + 2 * 4096 * 1024 + 3 * 4096 * 14336) + 4096 * 128256;
  claim(id, "reads its " + r(L3 * 2 / 1e9, 1) + " GB of BF16 weight matrices");
  near("check passes", 1 + 499, 500); claim(id, "500 passes, one after another");
}

{
  // Bengio et al. 2003, Section 3: share of operations in the output units for |V| = 17,964, h = 60, n = 6, m = 100
  const Vb = 17964, hb = 60, nb = 6, mb = 100, out = Vb * (1 + (nb - 1) * mb + hb);
  near("Bengio output share", out / (out + hb * (1 + (nb - 1) * mb) + (nb - 1) * mb), 0.997, 5e-4);
  claim("lm-output", "17,964 words, 60 hidden units) it took 99.7% of the operations");
  claim("lm-unicode", "its code point written in 11 bits is 00011 101001");
  claim("lm-bpe", "Sennrich et al. mark word ends");
}

{
  // Facts stated inside the reused widgets (their text is in js/legacy-widgets, not in the sheet strings).
  // predict (lm-predict): one note per position of our query
  const p0 = qTrace.probs[0], top0 = p0.indexOf(Math.max(...p0));
  near("predict: nine column names can come first", new Set(Object.values(D.schema).flat()).size, 9);
  equal("predict: id leads after SELECT", V[top0], " id");
  near("predict: name about 1/12", p0[M.idOf(" name")], 1 / 12, 0.005);
  ok("predict: FROM, users, WHERE, =, ; get 1.000", [1, 2, 3, 5, 7].every((i) => r(qP[i], 3) === "1.000"));
  ok("predict: three columns each near 1/3", [" id", " email", " age"].every((t) => Math.abs(qTrace.probs[4][M.idOf(t)] - 1 / 3) < 0.03));
  ok("predict: each digit about 0.111", [1, 2, 3, 4, 5, 6, 7, 8, 9].every((k) => Math.abs(qTrace.probs[6][M.idOf(" " + k)] - 1 / 9) < 0.005));
  // embed (lm-embed): "All 26 tokens our grammar uses"; bpe (lm-bpe-ours): "all 74 merges"
  near("embed: 26 tokens", new Set(ALLQ.flatMap((q) => q.tok)).size, 26);
  near("bpe: 74 merges", D.merges.length, 74);
  // learn (lm-pairs): floor tile and the two losses it shows
  near("learn: floor", D.floor.reduce((a, b) => a + b) / 8, 0.6937, 5e-5);
}

{
  // review fixes: lm-words, lm-decode, lm-bytes
  near("words with a leading space", D.wordCounts.filter(([w]) => w.startsWith(" ")).length, 24);
  equal("words without one", D.wordCounts.filter(([w]) => !w.startsWith(" ")).map(([w]) => w).sort(), [";", "SELECT"]);
  claimH("lm-words", "24 of them with a leading space (<code>SELECT</code> and <code>;</code> have none)");
  claim("lm-decode", "Detokenization"); claim("lm-decode", "Byte-level tokenizers such as GPT-2's and Llama 3's are lossless");
  // E6 97 is the first byte pair of exactly the code points U+65C0..U+65FF
  const pair = (cp) => hexBytes(String.fromCodePoint(cp)).slice(0, 5);
  ok("E6 97 covers U+65C0..U+65FF", [...Array(64).keys()].every((i) => pair(0x65c0 + i) === "E6 97") && pair(0x65bf) !== "E6 97" && pair(0x6600) !== "E6 97");
  equal("时", [cp("时"), hexBytes("时")], ["U+65F6", "E6 97 B6"]); equal("日", [cp("日"), hexBytes("日")], ["U+65E5", "E6 97 A5"]);
  claim("lm-bytes", "U+65C0 to U+65FF, such as 时 (E6 97 B6)"); claim("lm-bytes", "At most three tokens, none unknown");
  claim("lm-bytes", "falls back to bytes");  // Llama 2 paper, Section 2.2
}

done();
