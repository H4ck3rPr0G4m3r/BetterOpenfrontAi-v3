import { simpleHash } from "@openfront/engine-lib/Util";

test("engine imports through the alias", () => {
  expect(typeof simpleHash("abc")).toBe("number");
});
