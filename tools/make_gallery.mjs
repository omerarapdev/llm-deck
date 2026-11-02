// Writes gallery.html: index.html with the content scripts replaced by tools/gallery.js.
import fs from "node:fs";
const src = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
const out = src
  .replace(/(<script src="content\/[^"]+"><\/script>\n)+/, '<script src="tools/gallery.js"></script>\n')
  .replace("<title>One Query, End to End</title>", "<title>Component Gallery</title>");
if (out === src) throw new Error("content scripts not found in index.html");
fs.writeFileSync(new URL("../gallery.html", import.meta.url), out);
console.log("wrote gallery.html");
