// Fails the build when the bundle carries data. Every number the dashboard shows is fetched from
// the API at run time, so the built files must hold no value of the public ledger, no ledger
// fingerprint, no client identifier (synthetic or shaped like a real one) and no data file.
// Findings name the file and the rule, never the token. Runs after `vite build` in `pnpm build`.
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const dist = resolve(here, "..", "dist");
const ledgerPath = resolve(here, "..", "..", "..", "docs", "ledger", "claims.public.json");

const TEXT = new Set([".html", ".js", ".css", ".svg", ".txt"]);
const BINARY = new Set([".woff", ".woff2", ".ttf", ".ico", ".png"]);
// The privacy guard's fallback identifier shape, and the synthetic generator's prefix.
const IDENTIFIER = /(?<![0-9A-Za-z])[0-9A-Fa-f]{32,40}(?![0-9A-Za-z])/;
const SYNTHETIC_ID = /syn_[0-9A-Za-z]{6,}/;

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path) : [path];
  });
}

function ledgerNeedles() {
  if (!existsSync(ledgerPath)) return null;
  const ledger = JSON.parse(readFileSync(ledgerPath, "utf8"));
  const needles = [ledger.data_fingerprint];
  for (const claim of ledger.claims ?? []) {
    for (const value of Object.values(claim.values ?? {})) {
      const text = String(value);
      // Decimals with five or more significant digits cannot appear in code by accident.
      if (text.includes(".") && text.replace(/[^0-9]/g, "").replace(/^0+/, "").length >= 5) {
        needles.push(text);
      }
    }
  }
  return needles;
}

if (!existsSync(dist)) {
  console.error("check-bundle: no dist folder; run vite build first");
  process.exit(1);
}

const needles = ledgerNeedles();
if (needles === null) console.warn("check-bundle: no public ledger beside the app; shape checks only");

const problems = [];
const files = walk(dist);
for (const file of files) {
  const name = relative(dist, file).replaceAll("\\", "/");
  const suffix = extname(file).toLowerCase();
  if (BINARY.has(suffix)) continue;
  if (!TEXT.has(suffix)) {
    problems.push(`${name}: data-file (only code, styles, fonts and icons belong in the bundle)`);
    continue;
  }
  const text = readFileSync(file, "utf8");
  if (IDENTIFIER.test(text)) problems.push(`${name}: identifier-format`);
  if (SYNTHETIC_ID.test(text)) problems.push(`${name}: synthetic-identifier`);
  if (needles?.some((needle) => text.includes(needle))) problems.push(`${name}: ledger-value`);
}

if (problems.length) {
  console.error(`check-bundle: ${problems.length} problem(s)`);
  for (const problem of problems) console.error(`  ${problem}`);
  process.exit(1);
}
console.log(`check-bundle: ${files.length} files, no data in the bundle`);
