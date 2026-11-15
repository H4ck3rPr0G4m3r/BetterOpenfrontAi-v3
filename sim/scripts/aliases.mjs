// One alias table for vitest, esbuild and anything else that resolves the
// engine: workspace packages map to <engineRoot>/packages/<pkg>/src, and the
// engine's bare "resources/*" imports map to <engineRoot>/resources.
import path from "path";

export const ENGINE_PACKAGES = ["engine", "engine-api", "engine-lib", "shared", "zbin"];

/** Vite/vitest alias list (order matters: longer package names first). */
export function engineAliases(engineRoot) {
  const pkgs = [...ENGINE_PACKAGES].sort((a, b) => b.length - a.length);
  return [
    { find: /^@openfront\/zbin$/, replacement: path.join(engineRoot, "packages/zbin/src/index.ts") },
    ...pkgs.map((p) => ({
      find: new RegExp(`^@openfront/${p}/`),
      replacement: path.join(engineRoot, "packages", p, "src") + "/",
    })),
    { find: /^resources\//, replacement: path.join(engineRoot, "resources") + "/" },
  ];
}
