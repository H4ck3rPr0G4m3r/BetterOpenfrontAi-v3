/**
 * Browser types the engine's shared package names in signatures we never
 * call headless (GameMapLoader.layerImage). Declared opaque instead of
 * pulling the whole DOM lib into a Node program.
 */
interface ImageBitmap {
  readonly width: number;
  readonly height: number;
  close(): void;
}
