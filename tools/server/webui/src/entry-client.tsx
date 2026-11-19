import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import App from "./App";
import { TooltipProvider } from "./app/components/ui/HintTooltip";
import { normalizeAppPath } from "./app/routing";
import "./app/styles/global.css";
import "./app/styles/scroll-bar.css";
import "./app/styles/sticky-panel.css";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Missing #root element");
}

const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
const normalizedUrl = normalizeAppPath(currentUrl, import.meta.env.BASE_URL ?? "/");
if (normalizedUrl !== currentUrl) {
  window.history.replaceState(window.history.state, "", normalizedUrl);
}

createRoot(rootElement).render(
  <React.StrictMode>
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <TooltipProvider>
        <App />
      </TooltipProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
