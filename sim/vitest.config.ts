import path from "path";
import { fileURLToPath } from "url";
import { defineConfig } from "vitest/config";
import { engineAliases } from "./scripts/aliases.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
// Tests run against the pinned submodule unless OFAI_ENGINE_ROOT says otherwise.
const engineRoot = path.resolve(here, process.env.OFAI_ENGINE_ROOT ?? "../engine");

export default defineConfig({
  resolve: { alias: engineAliases(engineRoot) },
  test: {
    globals: true,
    environment: "node",
    testTimeout: 120_000,
    hookTimeout: 120_000,
    pool: "forks",
  },
});
