import React from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import App from "./App";
import { TooltipProvider } from "./app/components/ui/HintTooltip";
import { normalizeAppPath } from "./app/routing";
import "./siteDataGlobals";
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

const initialData = window.__SITE_DATA__ ?? null;

const app = (
  <React.StrictMode>
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <TooltipProvider>
        <App initialData={initialData} />
      </TooltipProvider>
    </BrowserRouter>
  </React.StrictMode>
);

if (rootElement.childElementCount > 0) {
  hydrateRoot(rootElement, app);
} else {
  createRoot(rootElement).render(app);
}
