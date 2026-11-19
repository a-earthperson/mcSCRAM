<!--
SPDX-FileCopyrightText: 2026 fuzzybit-org
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# tools/server/webui

CSR Vite/React UI served by `mcscram-server` at `/`.

```bash
npm ci
npm run typecheck
npm run build     # dist/ — FastAPI serves this
npm run dev       # Vite on :5173, proxies /api/v1 to FUZZYBITS_SERVER_URL (default :8000)
```

What it does: list OpenPSA/MEF XML, draw the first `define-fault-tree` in the
file, compile that fault tree to a PDAG, sample the top event, plot p̂ and the
interval. Event-tree sequences and end states are not compiled or sampled ([#92](https://github.com/fuzzybit-org/fuzzybits/issues/92)).
