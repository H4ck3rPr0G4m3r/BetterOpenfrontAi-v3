import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));

/** The engine the tests run against (the pinned submodule unless overridden). */
export const ENGINE_ROOT = path.resolve(here, "..", process.env.OFAI_ENGINE_ROOT ?? "../engine");

/** Smallest FFA-style map by map.bin size at the pin (onion, 512x512). */
export const SMALL_MAP = "Onion";
