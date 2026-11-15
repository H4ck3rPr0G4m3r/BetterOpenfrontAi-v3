import { execFileSync, spawnSync } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { writeRecord } from "../src/record/Record";
import { makeSyntheticRecord } from "../src/record/SyntheticRecord";
import { ENGINE_ROOT, SMALL_MAP } from "./helpers";

const SIM = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let dir: string;
let bundle: string;

beforeAll(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "ofai-cli-"));
  bundle = path.join(dir, "sim.mjs");
  execFileSync(process.execPath, [path.join(SIM, "scripts", "bundle.mjs"), "--engine-root", ENGINE_ROOT, "--out", bundle], {
    stdio: "pipe",
  });
});

afterAll(() => fs.rmSync(dir, { recursive: true, force: true }));

function run(...args: string[]) {
  const r = spawnSync(process.execPath, [bundle, ...args], { encoding: "utf8" });
  const lines = r.stdout.trim().split("\n").filter(Boolean).map((l) => JSON.parse(l));
  return { code: r.status, lines, stderr: r.stderr };
}

test("cli replay prints one ReplayResult line", async () => {
  const rec = await makeSyntheticRecord({ map: SMALL_MAP, seats: 2, turns: 300, seed: "cli", engineRoot: ENGINE_ROOT });
  const file = path.join(dir, "rec.json.gz");
  writeRecord(file, rec);
  const { code, lines } = run("replay", file, "--engine-root", ENGINE_ROOT);
  expect(code).toBe(0);
  expect(lines).toHaveLength(1);
  expect(lines[0]).toMatchObject({ gameId: rec.info.gameID, mismatchAt: null, error: null, hashChecks: 30, turns: 300 });
});

test("cli selfcheck is ok", () => {
  const { code, lines } = run("selfcheck", "--engine-root", ENGINE_ROOT);
  expect(code).toBe(0);
  expect(lines[0]).toMatchObject({ ok: true });
  expect(typeof lines[0].layoutVersion).toBe("number");
});

test("cli usage error exits 2", () => {
  const { code, lines } = run("nonsense");
  expect(code).toBe(2);
  expect(lines[0].error).toMatch(/usage/i);
});

test("cli missing file exits 1 with an error line", () => {
  const { code, lines } = run("replay", path.join(dir, "missing.json.gz"), "--engine-root", ENGINE_ROOT);
  expect(code).toBe(1);
  expect(lines[0].error).toBeTruthy();
});
