# The engine pin

`engine/` is a git submodule of [openfrontio/OpenFrontIO](https://github.com/openfrontio/OpenFrontIO), pinned to the
release live on openfront.io (spec §7).

| Date | Commit | Why |
|---|---|---|
| 2026-11-15 | `8b5d071eed95f07c701b32e4c64e95b03f8577c1` (tag `v0.36.2`) | The newest release tag, and the commit of the one public record fetched while planning (2026-11-14). The plan picks the pin as the most common `gitCommit` in freshly fetched records; that wasn't possible because the record API is behind a Cloudflare challenge for our client (`docs/measurements.md`). |

## How the engine is reached

Since 2026-10-05 (commit `e82286146`) the engine is split into workspace packages: `@openfront/engine`,
`@openfront/engine-api`, `@openfront/engine-lib`, `@openfront/shared`, `@openfront/zbin`. Records older than that
can't be replayed by our adapter; they are outside the 21-day version window anyway.

`sim/` resolves those packages by alias to `<engineRoot>/packages/<pkg>/src` (`sim/scripts/aliases.mjs`, used by
vitest and the esbuild bundler). It never goes through `engine/node_modules/@openfront/*`: those are workspace links
to absolute paths inside the pinned checkout, so a bundle for another commit that resolved them would silently run
the pinned engine instead of its own. Third-party packages (zod, jose, nanoid) come from the commit's
`node_modules` (roadmap §3.2).

Only `sim/src/engine/` imports `@openfront/*` (the adapter).

## Installing

```bash
git submodule update --init engine
cd engine && npm ci --ignore-scripts --engine-strict=false
```

`--engine-strict=false` because the engine's `.npmrc` asks for npm ≥ 12.1 and Node 24.16 ships npm 12.0.1; nothing
else depends on the difference.

## Bumping

M1 adds the "Update engine" action (fetch, check out, `npm ci`, sim tests, bundle, replay fresh records, roll back on
failure). By hand: `git -C engine fetch origin && git -C engine checkout <commit>`, reinstall, `cd sim && npm test`,
then commit the submodule pointer with a line in the table above.
