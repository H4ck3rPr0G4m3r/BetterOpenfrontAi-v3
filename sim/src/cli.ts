/**
 * Bundle entry: `node engines/<commit>/sim.mjs <command> ...` (roadmap §3.3).
 * stdout carries one JSON object per line and nothing else (engine logs go to
 * stderr). Exit 0 ok, 2 usage error, 1 anything else.
 */
import fs from "fs";
import path from "path";
import { quietEngineLogs } from "./engine/Adapter";
import { readRecord, writeRecord } from "./record/Record";
import { replayRecord } from "./record/Replayer";
import { makeSyntheticRecord } from "./record/SyntheticRecord";
import { ReportBuilder, reportRecord } from "./translate/Report";
import { layoutJson } from "./obs/Layout";
import { LAYOUT_VERSION, SIM_VERSION } from "./version";

class UsageError extends Error {}

const USAGE = "usage: sim.mjs <selfcheck|replay|synth|layout> [args] --engine-root <dir>";

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
    case "layout":
      out(layoutJson());
      return 0;
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
    case "translate-report": {
      // translate-report <records...> [--limit n]: round-trip every human intent (spec §10.4 gate).
      const root = engineRoot(flags);
      const files = rest.slice(0, flags.limit ? Number(flags.limit) : undefined);
      if (files.length === 0) throw new UsageError("usage: translate-report <records...> [--limit n] --engine-root <dir>");
      const builder = new ReportBuilder();
      for (const file of files) await reportRecord(readRecord(file), root, builder);
      out(builder.build());
      return 0;
    }
    case "synth": {
      // synth <out.json.gz> --map World --seats 4 --turns 3000 --seed s [--bots 400] [--nations default] [--commit <sha>]
      const root = engineRoot(flags);
      const file = rest[0];
      if (!file || !flags.map) throw new UsageError("usage: synth <out.json.gz> --map <name> [--seats n] [--turns n] [--seed s] [--bots n] [--nations default|disabled|n]");
      const nations = flags.nations === undefined ? "disabled" : /^\d+$/.test(flags.nations) ? Number(flags.nations) : (flags.nations as "default" | "disabled");
      const t0 = performance.now();
      const rec = await makeSyntheticRecord({
        map: flags.map,
        seats: Number(flags.seats ?? 4),
        turns: Number(flags.turns ?? 3000),
        seed: flags.seed ?? "synth",
        bots: Number(flags.bots ?? 0),
        nations,
        engineRoot: root,
      });
      if (flags.commit) rec.gitCommit = flags.commit;
      fs.mkdirSync(path.dirname(path.resolve(file)), { recursive: true });
      writeRecord(file, rec);
      out({ path: file, gameId: rec.info.gameID, turns: rec.info.num_turns, ms: Math.round(performance.now() - t0) });
      return 0;
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
