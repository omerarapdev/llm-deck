// Runs every tools/verify/partNN.mjs and reports a combined result.
import { readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
const dir = path.dirname(fileURLToPath(import.meta.url));
let failed = 0;
for (const f of readdirSync(dir).filter((f) => /^part\d\d\.mjs$/.test(f)).sort()) {
  const r = spawnSync(process.execPath, [path.join(dir, f)], { encoding: "utf8" });
  const last = (r.stdout || "").trim().split("\n").pop();
  console.log(`${f}: ${last}`);
  if (r.status !== 0) { failed++; console.log((r.stdout || "") + (r.stderr || "")); }
}
process.exit(failed ? 1 : 0);
