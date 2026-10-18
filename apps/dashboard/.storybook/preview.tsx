import "../src/styles.css";

import type { Decorator, Preview } from "@storybook/react";

const withTheme: Decorator = (Story, context) => {
  const theme = context.globals.theme === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", theme);
  return (
    <div className="bg-canvas p-4 text-ink">
      <Story />
    </div>
  );
};

const preview: Preview = {
  decorators: [withTheme],
  globalTypes: {
    theme: {
      description: "Light or dark tokens",
      toolbar: { title: "Theme", icon: "mirror", items: ["light", "dark"], dynamicTitle: true },
    },
  },
  initialGlobals: { theme: "light" },
  parameters: {
    layout: "fullscreen",
    a11y: { element: "#storybook-root" },
    controls: { expanded: true },
  },
};

export default preview;
