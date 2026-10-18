// Light, dark or the system's choice, remembered per browser. The tokens in styles.css do the rest.

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

/** The current value of a design token, for libraries (charts) that need concrete colours. */
export function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
}

/** The theme in effect: the chosen one, or the system's when none is chosen. */
export function effectiveTheme(): "light" | "dark" {
  const chosen = document.documentElement.getAttribute("data-theme");
  if (chosen === "light" || chosen === "dark") return chosen;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Re-renders when the theme in effect changes, for code that reads tokens (chart colours). */
export function useThemeKey(): "light" | "dark" {
  const [theme, setTheme] = useState(effectiveTheme);
  useEffect(() => {
    const update = () => setTheme(effectiveTheme());
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const media = window.matchMedia?.("(prefers-color-scheme: dark)");
    media?.addEventListener("change", update);
    return () => {
      observer.disconnect();
      media?.removeEventListener("change", update);
    };
  }, []);
  return theme;
}
