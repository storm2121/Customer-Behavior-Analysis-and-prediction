// Light, dark or the system's choice, remembered per browser. The tokens in styles/tokens.css do
// the rest; this module only stamps the choice on the document and tells charts when to re-read.

import { useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";

const KEY = "strata-theme";

export function storedTheme(): Theme {
  try {
    const value = window.localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
  try {
    if (theme === "system") window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, theme);
  } catch {
    // storage may be unavailable; the theme still applies for this visit
  }
}

export function nextTheme(theme: Theme): Theme {
  return theme === "system" ? "light" : theme === "light" ? "dark" : "system";
}

/** The current value of a design token on `element` (the document by default). */
export function token(name: string, element: Element = document.documentElement): string {
  return getComputedStyle(element).getPropertyValue(`--${name}`).trim();
}

/** The theme in effect: the chosen one, or the system's when none is chosen. */
export function effectiveTheme(): "light" | "dark" {
  const chosen = document.documentElement.getAttribute("data-theme");
  if (chosen === "light" || chosen === "dark") return chosen;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

const PREFERENCES = ["(prefers-color-scheme: dark)", "(prefers-contrast: more)"];

function themeKey(): string {
  const contrast = window.matchMedia?.("(prefers-contrast: more)").matches ? "more" : "normal";
  return `${effectiveTheme()}|${contrast}`;
}

/**
 * Changes whenever the token values in effect change (the chosen theme, the system's colour
 * scheme, the contrast preference), so code that reads tokens into a canvas can re-read them.
 */
export function useThemeKey(): string {
  const [key, setKey] = useState(themeKey);
  useEffect(() => {
    const update = () => setKey(themeKey());
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const queries = PREFERENCES.map((query) => window.matchMedia?.(query)).filter(
      (query): query is MediaQueryList => Boolean(query),
    );
    for (const query of queries) query.addEventListener("change", update);
    return () => {
      observer.disconnect();
      for (const query of queries) query.removeEventListener("change", update);
    };
  }, []);
  return key;
}
