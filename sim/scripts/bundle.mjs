#!/usr/bin/env node
// Bundles sim/src/cli.ts against one engine checkout:
//   node sim/scripts/bundle.mjs --engine-root <dir> --out <file>
// @openfront/* resolve to <engine-root>/packages/<pkg>/src and "resources/*"
// to <engine-root>/resources (aliases.mjs). Third-party packages resolve
// normally from the importing file, i.e. from <engine-root>/node_modules for
// engine code (roadmap §3.2 dependency rule).
import * as esbuild from "esbuild";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { ENGINE_PACKAGES } from "./aliases.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));

function arg(name) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0 || !process.argv[i + 1]) {
    console.error(`usage: bundle.mjs --engine-root <dir> --out <file>`);
    process.exit(2);
  }
  return path.resolve(process.argv[i + 1]);
}

const engineRoot = arg("engine-root");
const outFile = arg("out");
for (const p of ENGINE_PACKAGES) {
  if (!fs.existsSync(path.join(engineRoot, "packages", p, "src"))) {
    console.error(`engine root ${engineRoot} has no packages/${p}/src (pre-split commit?)`);
    process.exit(1);
  }
}

const pkgRe = new RegExp(`^@openfront/(${ENGINE_PACKAGES.join("|")})(/.*)?$`);
function resolveTs(base) {
  for (const cand of [base, `${base}.ts`, `${base}.tsx`, path.join(base, "index.ts")]) {
    if (fs.existsSync(cand) && fs.statSync(cand).isFile()) return cand;
  }
  return null;
}

const enginePlugin = {
  name: "openfront-engine",
  setup(build) {
    build.onResolve({ filter: /^@openfront\// }, (args) => {
      const m = pkgRe.exec(args.path);
      if (!m) return { errors: [{ text: `unknown engine package ${args.path}` }] };
      const src = path.join(engineRoot, "packages", m[1], "src");
      const target = m[2] ? resolveTs(path.join(src, m[2].slice(1))) : resolveTs(path.join(src, "index"));
      if (!target) return { errors: [{ text: `cannot resolve ${args.path} in ${src}` }] };
      return { path: target };
    });
    build.onResolve({ filter: /^resources\// }, (args) => ({ path: path.join(engineRoot, args.path) }));
  },
};

fs.mkdirSync(path.dirname(outFile), { recursive: true });
const t0 = Date.now();
await esbuild.build({
  entryPoints: [path.join(here, "..", "src", "cli.ts")],
  outfile: outFile,
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node24",
  sourcemap: false,
  legalComments: "none",
  logLevel: "warning",
  plugins: [enginePlugin],
  banner: {
    js: "import { createRequire as __ofaiCreateRequire } from 'module'; const require = __ofaiCreateRequire(import.meta.url);",
  },
});
console.error(`bundled ${path.relative(process.cwd(), outFile)} in ${Date.now() - t0} ms`);
