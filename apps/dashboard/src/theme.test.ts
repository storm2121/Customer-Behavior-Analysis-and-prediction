import { applyTheme, nextTheme, storedTheme } from "./theme";

describe("themes", () => {
  afterEach(() => applyTheme("system"));

  it("cycles from the system theme to light, dark and back", () => {
    expect(nextTheme("system")).toBe("light");
    expect(nextTheme("light")).toBe("dark");
    expect(nextTheme("dark")).toBe("system");
  });

  it("applies a theme to the document and remembers it", () => {
    applyTheme("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(storedTheme()).toBe("dark");
    applyTheme("system");
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
    expect(storedTheme()).toBe("system");
  });
});
