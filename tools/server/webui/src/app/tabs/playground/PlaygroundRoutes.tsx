import type { ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router";
import { ClientOnly } from "../../components/ui/ClientOnly";
import { PlaygroundTab } from "../lazyTabs";
import { playgroundPath } from "./routes";

type PlaygroundRoutesProps = {
  fallback: ReactNode;
};

export function PlaygroundRouter({ fallback }: PlaygroundRoutesProps) {
  return (
    <Routes>
      <Route
        index
        element={(
          <ClientOnly fallback={fallback}>
            <PlaygroundTab />
          </ClientOnly>
        )}
      />
      <Route path="*" element={<Navigate to={playgroundPath} replace />} />
    </Routes>
  );
}
