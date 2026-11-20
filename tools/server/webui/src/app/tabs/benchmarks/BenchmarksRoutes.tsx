import type { ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router";
import { ClientOnly } from "../../components/ui/ClientOnly";
import { BenchmarksTab } from "../lazyTabs";
import { benchmarksPath } from "./routes";

type BenchmarksRoutesProps = {
  fallback: ReactNode;
};

export function BenchmarksRouter({ fallback }: BenchmarksRoutesProps) {
  return (
    <Routes>
      <Route
        index
        element={(
          <ClientOnly fallback={fallback}>
            <BenchmarksTab />
          </ClientOnly>
        )}
      />
      <Route path="*" element={<Navigate to={benchmarksPath} replace />} />
    </Routes>
  );
}
