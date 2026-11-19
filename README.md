fuzzybits
=========

fuzzybits compiles a whole PRA model, its fault trees and event trees,
into one probabilistic directed acyclic graph (PDAG) and estimates
probabilities on that single graph by Monte Carlo sampling. Shared
basic events and gates are evaluated once, for every query at the same
time. Models are OpenPSA MEF XML files. It runs on the CPU, or on an
NVIDIA GPU if one is available.

Status: alpha (0.17.1). Every result comes with a 95% confidence
interval. In this release the compiled query is the top event of the
first fault tree in the model; event-tree sequences are next
(issue #92).


Install
-------

    pip install fuzzybits fuzzybits-server

Needs CPython 3.12 to 3.15 on Linux x86_64 (glibc 2.39 or newer, for
example Ubuntu 24.04), macOS 15+ on Intel (CPU only), or macOS 26+ on
Apple Silicon. To use an NVIDIA GPU, install the NVIDIA driver; nothing
else is needed.


Command line
------------

Compile the model once, then sample it:

    fzb compile model.xml -o model.pcfb
    fzb run model.pcfb --trials 1e8

The result is one line per query: name, estimate, 95% confidence
interval, number of samples and number of hits. For example:

    r1  1.17652615e-03  [1.16982950e-03, 1.18322280e-03]  n=100663296  hits=118433

Add --backend cuda to run on an NVIDIA GPU, or --json for
machine-readable output. fzb --help lists all options.


Web UI
------

Put your .xml models in a folder and start the server:

    fuzzybits-server --data-dir ./models

Then open http://localhost:8000, pick a model and start sampling.
The Benchmarks tab (http://localhost:8000/benchmarks) plots published
release-vs-develop history; see docs/benchmarks.md.

Or run the same server with Docker, no Python needed:

    docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/models:/data:ro" \
        ghcr.io/fuzzybit-org/fuzzybits-server:v0.17.1

For an NVIDIA GPU, add --gpus all -e ACPP_VISIBILITY_MASK=cuda (this
needs the NVIDIA Container Toolkit).

To try the example models that come with this repository:

    git clone https://github.com/fuzzybit-org/fuzzybits.git
    cd fuzzybits
    docker compose up -d


More
----

- Building from source and developer notes: docs/
- Questions and bug reports: https://github.com/fuzzybit-org/fuzzybits/issues
- License: AGPL-3.0 (see LICENSE). fuzzybits is derived from SCRAM by
  Olzhas Rakhimov (https://github.com/rakhimov/scram).
