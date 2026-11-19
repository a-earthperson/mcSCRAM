import { renderToReadableStream } from "react-dom/server";
import { StaticRouter } from "react-router";
import App from "./App";
import { normalizeAppPath } from "./app/routing";
import { TooltipProvider } from "./app/components/ui/HintTooltip";
import { loadSiteDataFromDisk } from "./app/lib/loadSiteData.server";

function normalizeBasename(baseUrl: string): string {
  const trimmed = baseUrl.trim();
  if (!trimmed || trimmed === "/") return "";
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

const basename = normalizeBasename(import.meta.env.BASE_URL ?? "/");

export async function render(url: string) {
  const siteData = await loadSiteDataFromDisk();
  const location = normalizeAppPath(url, basename);
  const stream = await renderToReadableStream(
    <StaticRouter location={location} basename={basename}>
      <TooltipProvider>
        <App initialData={siteData} />
      </TooltipProvider>
    </StaticRouter>,
  );
  await stream.allReady;
  const html = await new Response(stream).text();
  return { html, siteData };
}
