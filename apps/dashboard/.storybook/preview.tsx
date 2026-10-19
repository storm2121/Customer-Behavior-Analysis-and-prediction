import "../src/styles/index.css";

import type { Decorator, Preview } from "@storybook/react";
import { MotionConfig } from "motion/react";

// Every story renders in both themes side by side by default: each half stamps its own
// data-theme, and the tokens (and the charts, which read them on their own element) follow it.
// The toolbar narrows the view to one theme.
const withThemes: Decorator = (Story, context) => {
  const chosen = context.globals.theme;
  const themes = chosen === "light" || chosen === "dark" ? [chosen] : ["light", "dark"];
  return (
    <MotionConfig reducedMotion="user">
      <div className={themes.length === 2 ? "grid min-h-dvh grid-cols-1 xl:grid-cols-2" : "min-h-dvh"}>
        {themes.map((theme) => (
          <div key={theme} data-theme={theme} className="flex min-w-0 flex-col gap-3 bg-canvas p-6 text-ink">
            <p className="text-label font-label text-ink-secondary">{theme === "light" ? "Light theme" : "Dark theme"}</p>
            <Story />
          </div>
        ))}
      </div>
    </MotionConfig>
  );
};

const preview: Preview = {
  decorators: [withThemes],
  globalTypes: {
    theme: {
      description: "Both themes side by side, or one",
      toolbar: { title: "Theme", icon: "mirror", items: ["both", "light", "dark"], dynamicTitle: true },
    },
  },
  initialGlobals: { theme: "both" },
  parameters: {
    layout: "fullscreen",
    a11y: { element: "#storybook-root" },
    controls: { expanded: true },
  },
};

export default preview;
