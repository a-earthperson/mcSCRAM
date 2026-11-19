import type { SiteData } from "./app/types";

declare global {
  interface Window {
    __SITE_DATA__?: SiteData;
  }
}

export {};
