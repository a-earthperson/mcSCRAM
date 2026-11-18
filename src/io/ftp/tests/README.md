# FTP IO tests

Public fixtures live in `tests/testdata/` (hand-written snippets covering AND/OR,
complement, k-of-n, LIMIT, ENDTREE, and ignored PROCESS/IMPORT lines) and
`input/ftp/synthetic/` (generated trees). They are round-tripped by
`fuzzybits_io_ftp_tests`.

CAFTA / MHTGR / G-PWR / other vendor FTP models are **not** in this repository
(see fuzzybits#38). To run the corpus against a private copy of those files:

```
FUZZYBITS_PRIVATE_FIXTURES_DIR=/path/to/ftp ctest -R fuzzybits_io_ftp_tests
```

The directory is scanned recursively for `*.ftp`. If no public or private
fixtures are found, the corpus case `SKIP`s rather than failing.
