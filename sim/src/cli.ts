/**
 * Bundle entry: `node engines/<commit>/sim.mjs <command> ...` (roadmap §3.3).
 * stdout carries one JSON object per line and nothing else (engine logs go to
 * stderr). Exit 0 ok, 2 usage error, 1 anything else.
 */
import fs from "fs";
import path from "path";
import { quietEngineLogs } from "./engine/Adapter";
import { readRecord } from "./record/Record";
import { replayRecord } from "./record/Replayer";
import { makeSyntheticRecord } from "./record/SyntheticRecord";
import { LAYOUT_VERSION, SIM_VERSION } from "./version";

class UsageError extends Error {}

const USAGE = "usage: sim.mjs <selfcheck|replay> [record] --engine-root <dir>";

function out(obj: unknown): void {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function parseArgs(argv: string[]): { positional: string[]; flags: Record<string, string> } {
  const positional: string[] = [];
  const flags: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const v = argv[i + 1];
      if (v === undefined || v.startsWith("--")) throw new UsageError(`missing value for ${a}`);
      flags[a.slice(2)] = v;
      i++;
    } else positional.push(a);
  }
  return { positional, flags };
}

function engineRoot(flags: Record<string, string>): string {
  const root = flags["engine-root"];
  if (!root) throw new UsageError(`--engine-root is required. ${USAGE}`);
  return path.resolve(root);
}

/** Smallest map this engine commit has; selfcheck plays a short game on it. */
function selfcheckMap(root: string): string {
  const maps = path.join(root, "resources", "maps");
  for (const [dir, name] of [["onion", "Onion"], ["pangaea", "Pangaea"], ["world", "World"]]) {
    if (fs.existsSync(path.join(maps, dir, "map.bin"))) return name;
  }
  throw new Error(`no known small map under ${maps}`);
}

async function main(argv: string[]): Promise<number> {
  const { positional, flags } = parseArgs(argv);
  const [command, ...rest] = positional;
  switch (command) {
    case "selfcheck": {
      const root = engineRoot(flags);
      const rec = await makeSyntheticRecord({ map: selfcheckMap(root), seats: 2, turns: 200, seed: "selfcheck", engineRoot: root });
      const r = await replayRecord(rec, root);
      if (r.error !== null || r.mismatchAt !== null || r.hashChecks === 0) {
        out({ ok: false, error: r.error ?? `selfcheck replay mismatched at ${r.mismatchAt}`, layoutVersion: LAYOUT_VERSION });
        return 1;
      }
      out({ ok: true, layoutVersion: LAYOUT_VERSION, simVersion: SIM_VERSION, ticksPerSec: r.ticksPerSec });
      return 0;
    }
    case "replay": {
      const root = engineRoot(flags);
      if (rest.length === 0) throw new UsageError(USAGE);
      let failed = false;
      for (const file of rest) {
        const r = await replayRecord(readRecord(file), root);
        out({ ...r, path: file });
        if (r.error !== null && r.error.startsWith("setup failed")) failed = true;
      }
      return failed ? 1 : 0;
    }
    default:
      throw new UsageError(USAGE);
  }
}

quietEngineLogs();
main(process.argv.slice(2)).then(
  (code) => process.exit(code),
  (e: unknown) => {
    if (e instanceof UsageError) {
      out({ error: e.message });
      process.exit(2);
    }
    out({ error: e instanceof Error ? e.message : String(e) });
    process.exit(1);
  },
);
