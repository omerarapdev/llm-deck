// Static checks on the content: structure, ids, figure types, checks, terms,
// TeX that must parse, balanced HTML, and writing-style warnings.
//
//   node tools/validate.mjs            all parts
//   node tools/validate.mjs --part 6   one part (others are still loaded)
import { loadDeck, strings } from "./lib/deck-env.mjs";

const args = process.argv.slice(2);
const partArg = args.includes("--part") ? args[args.indexOf("--part") + 1].split(",").map(Number) : null;
const { Deck, katex, window: win, errors: loadErrors } = loadDeck();
const errors = [], warns = [];
const E = (id, m) => errors.push([id, m]);
const W = (id, m) => warns.push([id, m]);
for (const e of loadErrors) {
  const m = e.file.match(/^content\/(\d\d)-/);
  if (partArg && m && !partArg.includes(Number(m[1]))) W(e.file, "(another part's file) script error: " + e.msg);
  else E(e.file, "script error: " + e.msg);
}
if (!Deck) { console.log("Deck did not load"); process.exit(1); }

const CORE = new Set(["html", "table", "compare", "specs", "quiz", "steps", "code", "plot", "chart", "bars", "heat", "matmul", "graph", "network", "gd", "diagram", "calc", "timeline", "vectors", "sequence", "tinylm"]);
const VOID = new Set(["br", "hr", "img", "input", "wbr", "col", "area", "base", "link", "meta", "source", "track", "param"]);
const BANNED = [
  [/\bimagine\b/i, "analogy framing ('imagine')"], [/\bthink of (it|this|them) as\b/i, "analogy framing ('think of it as')"],
  [/\bjust like\b/i, "analogy ('just like')"], [/\banalog(y|ous)\b/i, "analogy"], [/\bpicture (this|a)\b/i, "analogy framing"],
  [/\bdelve\b/i, "filler word 'delve'"], [/\bcrucial(ly)?\b/i, "filler word 'crucial'"], [/\bleverag(e|es|ing)\b/i, "say 'use'"],
  [/\bseamless(ly)?\b/i, "filler word"], [/\bit'?s worth noting\b/i, "filler phrase"], [/\blet'?s (dive|explore|take a look)\b/i, "filler phrase"],
  [/\bin this (slide|sheet)\b/i, "self-reference"], [/\bmagic(al)?\b/i, "say what happens"], [/\bsimply\b/i, "'simply' (often hides a step)"],
  [/\bnot (just|only) \w+[^.]{0,40}\bbut\b/i, "'not X but Y' construction"], [/!/, "exclamation mark"],
];
const texOf = (s) => [...s.matchAll(/\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)/g)].map((m) => [m[1] || m[2], !!m[1]]);
function checkTex(id, where, tex, display) {
  try { katex.renderToString(tex, { throwOnError: true, displayMode: display, output: "mathml", strict: "ignore" }); }
  catch (e) { E(id, `TeX in ${where}: ${String(e.message).split("\n")[0]} — ${tex.slice(0, 80)}`); }
}
function checkHtml(id, where, s) {
  const stack = [];
  const re = /<\/?([a-zA-Z][a-zA-Z0-9-]*)\b[^>]*?(\/?)>/g;
  let m;
  const clean = s.replace(/\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)/g, "");
  while ((m = re.exec(clean))) {
    const tag = m[1].toLowerCase();
    if (VOID.has(tag) || m[2] === "/") continue;
    if (m[0][1] === "/") {
      if (stack[stack.length - 1] === tag) stack.pop();
      else { E(id, `HTML in ${where}: </${tag}> closes <${stack[stack.length - 1] || "nothing"}>`); return; }
    } else stack.push(tag);
  }
  if (stack.length) E(id, `HTML in ${where}: unclosed <${stack.join("> <")}>`);
}
const words = (s) => s.replace(/<[^>]+>/g, " ").replace(/\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)/g, " x ").split(/\s+/).filter(Boolean).length;

for (const m of Deck.problems || []) {
  const pm = m.match(/part (\d+)/);
  if (partArg && pm && !partArg.includes(Number(pm[1]))) W("(deck)", m); else E("(deck)", m);
}
const seenFig = new Map();
for (const p of Deck.parts) {
  if (!Number.isInteger(p.n)) E(`part ${p.title}`, "part number must be an integer");
  if (!p.title) E(`part ${p.n}`, "part needs a title");
}
for (const sl of Deck.slides) {
  if (partArg && !partArg.includes(sl.part.n)) continue;
  const id = sl.id;
  if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(id)) E(id, "id must be kebab-case: lowercase letters, digits and single hyphens");
  if (!sl.title) E(id, "missing title");
  else if (sl.title.replace(/<[^>]+>/g, "").length > 80) W(id, `title is ${sl.title.length} characters; aim for under 70`);
  if (!sl.body && sl.layout !== "part") E(id, "missing body");
  if (sl.layout && !["split", "text", "wide", "even", "part"].includes(sl.layout)) E(id, `unknown layout ${sl.layout}`);
  const known = new Set(["id", "title", "body", "math", "take", "note", "fig", "layout", "example", "terms", "src", "check", "part", "chapter", "index", "_strip"]);
  for (const k of Object.keys(sl)) if (!known.has(k)) E(id, `unknown field '${k}'`);
  if (sl.fig) {
    const t = sl.fig.type;
    if (!t) E(id, "figure without type");
    else if (!Deck.figs[t]) E(id, `figure type '${t}' is not registered`);
    else seenFig.set(t, (seenFig.get(t) || 0) + 1);
    if (!sl.fig.caption) W(id, "figure without caption");
    if (t === "tinylm" && !(window_widgets().includes(sl.fig.widget))) E(id, `tinylm widget '${sl.fig.widget}' does not exist`);
  }
  if (sl.check) {
    const c = sl.check;
    if (!c.q || !Array.isArray(c.options) || c.options.length < 2 || c.options.length > 5) E(id, "check needs q and 2–5 options");
    else if (!Number.isInteger(c.answer) || c.answer < 0 || c.answer >= c.options.length) E(id, "check.answer must index an option");
    if (!c.why) E(id, "check needs a 'why'");
  }
  for (const t of sl.terms || []) if (!t.term || !t.def) E(id, "each term needs term and def");
  if (sl.src && !Array.isArray(sl.src)) E(id, "src must be an array of strings");
  if (sl.example && !/^(tokens|text|chars)?( ?(ids|focus:[\d,]+|prefix:\d+|target:\d+|tokens|text|chars))*$/.test(sl.example.trim())) E(id, `bad example spec '${sl.example}'`);
  // strings: TeX, HTML, style
  for (const [where, s] of strings(sl)) {
    if (/(^|\.)code$/.test(where)) continue; // code listings are text, not HTML or TeX
    for (const [tex, display] of texOf(s)) checkTex(id, where, tex, display);
    if (/\\\(|\\\[/.test(s.replace(/\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)/g, ""))) E(id, `unclosed TeX delimiter in ${where}`);
    if (/(^|\.)code$/.test(where)) continue; // code listings are text, not HTML or TeX
    if (/[<>]/.test(s)) checkHtml(id, where, s);
    if (/^(body|take|note|title|check\.|fig\.caption|fig\.note)/.test(where)) {
      for (const [re, why] of BANNED) if (re.test(s.replace(/<[^>]+>/g, " "))) W(id, `${where}: ${why}`);
      if (/—/.test(s)) W(id, `${where}: em dash; use a period, colon or parentheses`);
    }
  }
  if (sl.math) for (const m of [].concat(sl.math)) checkTex(id, "math", typeof m === "string" ? m : m.tex, true);
  const n = words(sl.body || "");
  if (n > 300) E(id, `body is ${n} words; split the sheet (limit 300)`);
  else if (n > 210) W(id, `body is ${n} words; aim for 90–200`);
  if (sl.note && words(sl.note) > 30) W(id, `margin note is ${words(sl.note)} words; keep under 30`);
}
function window_widgets() { return Object.keys(win.Widgets || {}); }

const PREFIX = { 1: "math-", 2: "learn-", 3: "nn-", 4: "lm-", 5: "tf-", 6: "gpu-", 7: "scale-", 8: "serve-", 9: "post-", 10: "end-" };
for (const sl of Deck.slides) {
  if (partArg && !partArg.includes(sl.part.n)) continue;
  const pre = PREFIX[sl.part.n];
  if (pre && !sl.id.startsWith(pre)) E(sl.id, `ids in part ${sl.part.n} must start with '${pre}'`);
}
const termSeen = new Map();
for (const t of Deck.terms) {
  const k = t.term.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  if (termSeen.has(k) && (!partArg || partArg.includes(t.slide.part.n))) W(t.slide.id, `term '${t.term}' is also defined on ${termSeen.get(k)}`);
  else termSeen.set(k, t.slide.id);
}

// summary
const parts = Deck.parts.filter((p) => !partArg || partArg.includes(p.n));
console.log("part  sheets  chapters  title");
for (const p of parts) console.log(`${String(p.n).padStart(4)}  ${String(p.slides.length).padStart(6)}  ${String(p.chapters.length).padStart(8)}  ${p.title}`);
console.log(`total ${Deck.slides.length} sheets, ${Deck.terms.length} terms; figures: ${[...seenFig].map(([k, v]) => `${k}×${v}`).join(" ")}`);
for (const [id, m] of warns) console.log(`  warn  ${id}: ${m}`);
for (const [id, m] of errors) console.log(`  ERROR ${id}: ${m}`);
console.log(errors.length ? `✗ ${errors.length} errors, ${warns.length} warnings` : `✓ no errors, ${warns.length} warnings`);
process.exit(errors.length ? 1 : 0);
