import "./styles/index.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { applyTheme, storedTheme } from "./theme";

applyTheme(storedTheme());

const root = document.getElementById("root");
if (!root) throw new Error("the page has no #root element");
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
