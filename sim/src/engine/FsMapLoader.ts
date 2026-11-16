/**
 * Reads production maps from `<engineRoot>/resources/maps/<dir>/`, where
 * `<dir>` is the lowercased GameMapType key: the rule the engine's own
 * loaders use (tests/perf/fullgame/NodeGameMapLoader.ts at the pin).
 *
 * Every call reads fresh bytes: `loadTerrainMap` takes the files over and
 * mutates them (compact scaling, the live GameMap's terrain), so files must
 * never be shared between games.
 */
import { GameMapType } from "@openfront/engine-api/game/GameTypes";
import type { MapManifest } from "@openfront/engine-api/game/MapFiles";
import type { GameMapLoader, MapData } from "@openfront/shared/GameMapLoader";
import fs from "fs";
import path from "path";

export function mapDirName(map: GameMapType): string {
  const key = Object.keys(GameMapType).find((k) => GameMapType[k as keyof typeof GameMapType] === map);
  if (key === undefined) throw new Error(`unknown map: ${map}`);
  return key.toLowerCase();
}

export class FsMapLoader implements GameMapLoader {
  constructor(private readonly mapsDir: string) {}

  dir(map: GameMapType): string {
    return path.join(this.mapsDir, mapDirName(map));
  }

  getMapData(map: GameMapType): MapData {
    const dir = this.dir(map);
    const readBin = (name: string) => async () => new Uint8Array(fs.readFileSync(path.join(dir, name)));
    return {
      mapBin: readBin("map.bin"),
      map4xBin: readBin("map4x.bin"),
      map16xBin: readBin("map16x.bin"),
      manifest: async () => JSON.parse(fs.readFileSync(path.join(dir, "manifest.json"), "utf8")) as MapManifest,
      webpPath: path.join(dir, "thumbnail.webp"),
      layerImage: async () => {
        throw new Error("layer images are not available headless");
      },
    };
  }
}
