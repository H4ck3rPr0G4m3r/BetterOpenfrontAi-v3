import { fieldBytes, layoutJson, obsFieldSpecs, PLANES, rowBytes, TYPE_ARGS, ACTION_TYPES, ACTION_FIELDS } from "../../src/obs/Layout";

test("field specs sum to the expected byte sizes", () => {
  const { game, seat, total } = rowBytes();
  console.log(`row bytes: game ${game}, seat ${seat}, total ${total}`);
  expect(seat).toBeLessThan(64 * 1024);
  expect(game + seat).toBe(obsFieldSpecs().reduce((s, f) => s + fieldBytes(f), 0));
  expect(total).toBe(game + seat);
});

test("plane list has 31 unique names ending with valid", () => {
  expect(PLANES).toHaveLength(31);
  expect(new Set(PLANES.map((p) => p.name)).size).toBe(31);
  expect(PLANES[PLANES.length - 1].name).toBe("valid");
});

test("every action type names only known fields", () => {
  for (const t of ACTION_TYPES) for (const f of TYPE_ARGS[t]) expect(ACTION_FIELDS).toContain(f);
});

test("layout json snapshot", () => {
  expect(layoutJson()).toMatchSnapshot();
});
