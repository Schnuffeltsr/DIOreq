# Stage 2 - Diagnostic Propagation and DPS (Section 3.2)

This stage issues **no LLM calls**. It is a deterministic numerical
layer implemented in `dioreq.py`, so there are no prompt templates here.

Components:

- `DependencyGraphs.build()` - semantic `MultiDiGraph` plus the acyclic
  computational `DiGraph` (lowest-supported edge dropped on cycles,
  incoming edges above `max_parents` dropped).
- `DependencyGraphs.propagate()` - Equation (3), forward activation in
  topological order with optional source clamping.
- `DependencyGraphs.dps()` - Equation (4), the difference between the
  clamped-at-1 and clamped-at-0 runs.
- `deterministic_shortest_path()` / `maximum_geometric_path_support()`
  / `AllPathSupportIndex` - the Topology and Extraction Support ranking
  signals compared against DPS in RQ3.
