// Helpers for per-part verification scripts (tools/verify/partNN.mjs).
// A verification script recomputes every number a part states and asserts
// that the sheet text contains it, so the text cannot drift from the math.
//
//   import { claim, near, TinyLM, text, done } from "./lib.mjs";
//   const p = 1 / (1 + Math.exp(-2));                 // compute
//   near("sigmoid(2)", p, 0.8808, 5e-5);              // check your arithmetic
//   claim("learn-sigmoid", "0.881", "σ(2) to 3 dp");  // check the sheet says it
//   done();
import { loadDeck, slideText } from "../lib/deck-env.mjs";

const env = loadDeck();
export const Deck = env.Deck;
export const window = env.window;
export const TinyLM = env.window.TinyLM;
export const UI = env.window.UI;
let pass = 0;
const fails = [];
const norm = (s) => String(s).replace(/−/g, "-").replace(/[   ]/g, " ").replace(/\s+/g, " ");

export function slide(id) {
  const sl = Deck.slides.find((s) => s.id === id);
  if (!sl) throw new Error(`no sheet with id '${id}'`);
  return sl;
}
export const text = (id) => norm(slideText(slide(id)));

// The sheet's text (all fields, tags removed, TeX as written) contains `needle`.
export function claim(id, needle, label = "") {
  let ok = false;
  try { ok = text(id).includes(norm(needle)); } catch (e) { fails.push(`${id}: ${e.message}`); return; }
  if (ok) pass++; else fails.push(`${id}: text does not contain "${needle}"${label ? ` (${label})` : ""}`);
}
// |got − want| ≤ tol
export function near(label, got, want, tol = 1e-9) {
  if (Number.isFinite(got) && Math.abs(got - want) <= tol) pass++;
  else fails.push(`${label}: got ${got}, expected ${want} ± ${tol}`);
}
export function equal(label, got, want) {
  if (JSON.stringify(got) === JSON.stringify(want)) pass++;
  else fails.push(`${label}: got ${JSON.stringify(got)}, expected ${JSON.stringify(want)}`);
}
export function ok(label, cond) { if (cond) pass++; else fails.push(label); }
// Round half away from zero to d decimals, as written in the text.
export const r = (x, d) => (Math.round(Math.abs(x) * 10 ** d) / 10 ** d * Math.sign(x)).toFixed(d);
// Thousands separators, as written in the text: 9968 -> "9,968".
export const int = (x) => Math.round(x).toLocaleString("en-US");

export function done() {
  for (const f of fails) console.log("  ✗ " + f);
  console.log(fails.length ? `✗ ${fails.length} failed, ${pass} passed` : `✓ all ${pass} checks passed`);
  process.exit(fails.length ? 1 : 0);
}
