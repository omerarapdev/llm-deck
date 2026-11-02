// Renders sheets in headless Chrome and reports problems: script errors, figures
// that failed, page or sheet overflow, missing fonts. Optionally saves screenshots.
//
//   node tools/render.mjs                      check every sheet, desktop, light
//   node tools/render.mjs --part 6 --shots     check part 6 and save PNGs to tools/shots/
//   node tools/render.mjs --ids cover,route --mobile --dark --shots
//   node tools/render.mjs --page gallery.html --shots --out tools/shots/gallery
//
// Exit code 1 if any sheet has an error.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import vm from "node:vm";
import puppeteer from "puppeteer-core";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const opt = (k) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : null; };
const flag = (k) => args.includes(k);
const parts = opt("--part") ? opt("--part").split(",").map(Number) : null;
const ids = opt("--ids") ? opt("--ids").split(",") : null;
const mobile = flag("--mobile"), dark = flag("--dark"), shots = flag("--shots");
const outDir = path.resolve(ROOT, opt("--out") || "tools/shots");
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const vp = mobile ? { width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true } : { width: 1440, height: 900, deviceScaleFactor: 1 };
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ["--no-sandbox", "--hide-scrollbars", "--allow-file-access-from-files"] });
const page = await browser.newPage();
await page.setViewport(vp);
await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: dark ? "dark" : "light" }, { name: "prefers-reduced-motion", value: "reduce" }]);

const errors = [];
let currentId = "(load)";
page.on("pageerror", (e) => errors.push({ id: currentId, msg: "pageerror: " + (e.stack || e.message || String(e)).split("\n").slice(0, 3).join(" | ") }));
page.on("console", (m) => { if (m.type() === "error") errors.push({ id: currentId, msg: "console: " + m.text() + (m.location() && m.location().url ? ` @ ${path.basename(m.location().url)}:${m.location().lineNumber}` : "") }); });
page.on("requestfailed", (r) => { if (!/fonts\.(googleapis|gstatic)/.test(r.url())) errors.push({ id: currentId, msg: "request failed: " + r.url() }); });

// Syntax-check every content file first, so a parse error is attributed to its file.
const pageFile = opt("--page") || "index.html";
const syntaxErrors = [];
for (const m of fs.readFileSync(path.join(ROOT, pageFile), "utf8").matchAll(/<script src="((?:content|tools)\/[^"]+)"><\/script>/g)) {
  try { new vm.Script(fs.readFileSync(path.join(ROOT, m[1]), "utf8"), { filename: m[1] }); }
  catch (e) { syntaxErrors.push({ id: "(load)", msg: `syntax error in ${m[1]}: ${e.message}` }); }
}
const url = pathToFileURL(path.join(ROOT, pageFile)).href + "?static";
await page.goto(url, { waitUntil: "load" });
await page.evaluate(() => document.fonts && document.fonts.ready);
await page.evaluate(() => { try { localStorage.clear(); } catch (e) { /* ignore */ } });

const all = await page.evaluate(() => (window.Deck ? window.Deck.slides : []).map((s) => ({ id: s.id, part: s.part ? s.part.n : -1, title: String(s.title || "").replace(/<[^>]+>/g, ""), i: s.index })));
const deckProblems = await page.evaluate(() => (window.Deck && window.Deck.problems) || []);
const targets = all.filter((s) => (!parts || parts.includes(s.part)) && (!ids || ids.includes(s.id)));
if (shots) fs.mkdirSync(outDir, { recursive: true });

const problems = [];
const frame = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 30)))));
for (const s of targets) {
  currentId = s.id;
  await page.evaluate((i) => window.Deck.go(i), s.i);
  await frame();
  const r = await page.evaluate(() => {
    const out = [];
    const sheet = document.querySelector("#sheet-view .sheet.is-on") || document.querySelector("#sheet-view .sheet");
    if (!sheet) return { out: ["no sheet rendered"], h: 0 };
    sheet.querySelectorAll(".fig-missing").forEach((e) => out.push("figure: " + e.textContent));
    if (document.documentElement.scrollWidth > document.documentElement.clientWidth + 1) out.push(`page overflows horizontally by ${document.documentElement.scrollWidth - document.documentElement.clientWidth}px`);
    if (sheet.scrollWidth > sheet.clientWidth + 1) {
      // find the widest offender that is not inside a horizontal scroller
      const right = sheet.getBoundingClientRect().right;
      const bad = Array.from(sheet.querySelectorAll("*")).filter((e) => !e.closest(".scroll-x, .ex-row, .tex-d, .formula") && e.getBoundingClientRect().right > right + 1);
      const leaf = bad.filter((e) => !bad.some((o) => o !== e && e.contains(o)));
      out.push(`sheet overflows by ${sheet.scrollWidth - sheet.clientWidth}px: ` + leaf.slice(0, 3).map((e) => e.tagName.toLowerCase() + (e.className && typeof e.className === "string" ? "." + e.className.split(" ").join(".") : "")).join(", "));
    }
    // text that leaves a plate (common with long SVG labels) or plates wider than their column
    sheet.querySelectorAll(".plate").forEach((p) => {
      const pr = p.getBoundingClientRect();
      const body = p.querySelector(".fig-body");
      if (body && body.scrollWidth > body.clientWidth + 2 && !body.querySelector(".scroll-x")) out.push(`plate content wider than plate by ${body.scrollWidth - body.clientWidth}px`);
      if (pr.width < 40) out.push("plate collapsed");
    });
    // display formulas scroll rather than wrap; on desktop that means the formula is too long
    if (!matchMedia("(max-width: 760px)").matches) {
      sheet.querySelectorAll(".formula, .tex-d").forEach((f) => {
        if (f.scrollWidth > f.clientWidth + 2) out.push(`display formula wider than its box by ${f.scrollWidth - f.clientWidth}px: ${(f.textContent || "").trim().slice(0, 60)}`);
      });
    }
    const words = (sheet.querySelector(".prose") || { innerText: "" }).innerText.split(/\s+/).filter(Boolean).length;
    return { out, h: sheet.getBoundingClientRect().height, words };
  });
  for (const m of r.out) problems.push({ id: s.id, msg: m });
  if (shots) {
    const full = Math.min(5000, Math.ceil(r.h + (mobile ? 150 : 110)));
    await page.setViewport(Object.assign({}, vp, { height: Math.max(vp.height, full) }));
    await frame();
    const file = path.join(outDir, `${String(s.i + 1).padStart(3, "0")}-${s.id}${mobile ? "-m" : ""}${dark ? "-d" : ""}.png`);
    await page.screenshot({ path: file });
    await page.setViewport(vp);
  }
}
currentId = "(end)";
await browser.close();

// When checking some parts, load errors from other parts' files are only notes. Unattributed
// load errors (a syntax error reports no file) are notes too: the syntax pre-pass names the file.
const fileOf = (msg) => { const m = msg.match(/content\/(\d\d)-|css\/parts\/(\d\d)\.css/); return m ? Number(m[1] || m[2]) : null; };
const notes = [];
const loadErr = errors.filter((e) => e.id === "(load)").concat(syntaxErrors);
for (const e of loadErr) {
  const f = fileOf(e.msg);
  if (parts && (f === null ? !e.msg.startsWith("syntax error") : !parts.includes(f))) notes.push(e.msg);
  else problems.push(e);
}
for (const m of deckProblems) problems.push({ id: "(deck)", msg: m });
const issues = problems.concat(errors.filter((e) => e.id !== "(load)" && targets.some((t) => t.id === e.id)));
const byId = new Map();
for (const p of issues) { if (!byId.has(p.id)) byId.set(p.id, []); byId.get(p.id).push(p.msg); }
console.log(`checked ${targets.length} of ${all.length} sheets (${mobile ? "mobile 390px" : "desktop 1440px"}, ${dark ? "blueprint" : "graph paper"})`);
for (const [id, msgs] of byId) { console.log(`✗ ${id}`); for (const m of [...new Set(msgs)]) console.log("    " + m); }
for (const n of [...new Set(notes)]) console.log("  note (other parts): " + n.slice(0, 220));
if (!issues.length) console.log("✓ no problems");
if (shots) console.log("screenshots in " + path.relative(process.cwd(), outDir));
process.exit(issues.length ? 1 : 0);
