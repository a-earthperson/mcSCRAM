export function Hero() {
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
