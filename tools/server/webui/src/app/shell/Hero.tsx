import { useLocation } from "react-router";
import { benchmarksPath } from "../tabs/benchmarks/routes";

export function Hero() {
  const { pathname } = useLocation();
  const benchmarks = pathname === benchmarksPath || pathname.startsWith(`${benchmarksPath}/`);

  if (benchmarks) {
    return (
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">fuzzybits</p>
          <h1>Benchmarks</h1>
          <p className="lede">
            Frozen dataset v1, measured on V100S, RTX 3070, and the CPU bench
            runner. Each tagged release is compared with current develop. Charts
            are GPU kernel time, CPU sample time, and accuracy z. Numbers only
            compare on the same runner and recipe.
          </p>
          <div className="hero-actions">
            <a href="https://github.com/fuzzybit-org/fuzzybits">fuzzybits</a>
            <a href="https://github.com/fuzzybit-org/fuzzybits/blob/develop/docs/benchmarks.md">docs</a>
          </div>
        </div>
      </header>
    );
  }

  return (
    <header className="hero">
      <div className="hero-copy">
        <p className="eyebrow">fuzzybits</p>
        <h1>Playground</h1>
        <p className="lede">
          Catalog of OpenPSA/MEF XML. Views: first fault tree in the file, source DAG,
          compiled PDAG, kernel DAG, execution plan. Sampling estimates the compiled
          top-event probability with a confidence interval over <code>/api/v1/ws</code>.
          Event-tree sequences are not compiled or sampled.
        </p>
        <div className="hero-actions">
          <a href="https://github.com/fuzzybit-org/fuzzybits">fuzzybits</a>
        </div>
      </div>
    </header>
  );
}
