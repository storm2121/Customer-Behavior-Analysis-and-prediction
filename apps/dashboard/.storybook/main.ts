import type { StorybookConfig } from "@storybook/react-vite";

// The components with illustrative fixtures (src/test-fixtures.ts), in both themes, with the
// accessibility panel. Built by `pnpm storybook:build` in CI.
const config: StorybookConfig = {
  stories: ["../src/**/*.stories.@(ts|tsx)"],
  addons: ["@storybook/addon-essentials", "@storybook/addon-a11y"],
  framework: { name: "@storybook/react-vite", options: {} },
  core: { disableTelemetry: true },
};

export default config;
