# Final Round 4: finite-source Moser audit

Four original artifacts are copied byte-for-byte. This includes all 1024 trajectories at five fixed times, original source/protocol hashes, measured CPU costs, and an independent CDF-inverse / tangent-half-map / quadrature review. No model or GPU sampling, training or image FID was performed.

The positive smooth population example validates exact target-compatible Moser flow; the finite Galerkin example disproves inheritance of the full density path from one projected weak equation. The separate empirical Dirac example demonstrates a null gradient Gram with nonzero source; it is not a counterexample to the smooth theorem.

See the [frozen protocol](../../RAEV2_MOSER_FINITE_SOURCE_PROTOCOL_20260906_ZH.md).
