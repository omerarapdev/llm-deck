// Loads the deck's scripts (in index.html order) into a jsdom window without
// starting it, so tools can inspect Deck.parts / Deck.slides as data.
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

export function loadDeck({ page = "index.html" } = {}) {
  const html = fs.readFileSync(path.join(ROOT, page), "utf8");
  const scripts = [...html.matchAll(/<script src="([^"]+)"><\/script>/g)].map((m) => m[1]);
  const dom = new JSDOM("<!doctype html><html><body></body></html>", { url: "http://localhost/", runScripts: "outside-only", pretendToBeVisual: true });
  const w = dom.window;
  w.matchMedia = w.matchMedia || (() => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
  const ctx = dom.getInternalVMContext();
  const errors = [];
  for (const src of scripts) {
    const file = path.join(ROOT, src);
    try {
      new vm.Script(fs.readFileSync(file, "utf8"), { filename: src }).runInContext(ctx);
    } catch (e) {
      errors.push({ file: src, msg: (e && e.stack ? e.stack.split("\n").slice(0, 4).join(" | ") : String(e)) });
    }
  }
  return { window: w, Deck: w.Deck, katex: w.katex, errors, scripts };
}

// Every string inside a value, with the path that leads to it.
export function strings(v, at = "", out = []) {
  if (typeof v === "string") out.push([at, v]);
  else if (Array.isArray(v)) v.forEach((x, i) => strings(x, `${at}[${i}]`, out));
  else if (v && typeof v === "object") {
    for (const [k, x] of Object.entries(v)) {
      if (k === "part" || k === "chapter" || k === "_strip" || k === "slide") continue;
      strings(x, at ? `${at}.${k}` : k, out);
    }
  }
  return out;
}

// Plain text of a slide: all its strings with tags removed, TeX kept as written.
export function slideText(sl) {
  return strings(sl).map(([, s]) => s).join(" \n ").replace(/<[^>]+>/g, " ").replace(/&nbsp;/g, " ").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").replace(/[ \t]+/g, " ");
}
