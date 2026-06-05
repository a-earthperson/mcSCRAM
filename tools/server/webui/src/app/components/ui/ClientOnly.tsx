import { type ReactNode, useEffect, useState } from "react";

export function ClientOnly({ fallback, children }: { fallback: ReactNode; children: ReactNode }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return fallback;
  return children;
}
