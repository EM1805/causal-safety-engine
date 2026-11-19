# Amantia Resource Genesis v0.1 — CPU and memory pilot

**Research only / review candidate.** This branch optimizes one reusable hot
path in \`tools/do_readiness.py\`, not all of Causal Safety Engine.

## Hypothesis and frozen original

The exact original is \`08b2e17bf5d00d165909390f21293a5c18515e7d\`:
\`tools/do_readiness.py\` blob \`ed560a783df87b7694312abad87e88157451191c\`.
Original code and \`main\` are unchanged. Comparisons load the original with
\`git show\` rather than copying a potentially stale function into tests.

In the original, checking whether each parent of X can reach Y while
excluding X requires repeating a BFS for every parent. The list-backed FIFO
also has expensive \`pop(0)\` operations on wide graph frontiers.

## Proposed operator variants

- **Original:** separate BFS per parent, list queue (frozen Git version).
- **Strong fixed:** separate BFS per parent, \`collections.deque\` queue.
- **Genesis candidate:** for 1–2 parents use deque BFS, otherwise traverse
  *backwards once* from Y (excluding X), intersecting with the parents of X.
  This preserves the reachability criterion even on graphs with cycles and
  does not claim causal identifiability.

All strategies were specified by the developer. The v0.1 protocol selects
among their outcomes, not a self-inventing compiler/optimizer. Graph size,
branch width and reachability pattern determine the speed trade-off.

## Independent correctness

\`tests/test_resource_genesis_do_readiness.py\` compares all returned
confounders, core identifiability summaries and path reachability with the
frozen original on seeded cyclic graphs, sparse graphs, missing nodes, self
queries, target edges and wide shared-ancestor cases. An independent CI
workflow runs these before resource measurement.

## Real CPU and memory measurement

\`resource_genesis/benchmark.py\` launches separate fresh Python subprocesses,
alternating algorithms across 3 repetitions of four deterministic *synthetic*
graphs. It records:

- \`time.process_time_ns\`: CPU time per graph query (actual CPU, not a proxy);
- \`resource.getrusage().ru_maxrss\`: peak Linux process RSS, **including
  Python startup, imported code and graph allocation**;
- \`tracemalloc\`: peak Python temporary allocations during one query,
  **not equivalent to total RAM**.

A separate frozen-original comparator and queue-only fixed control prevent
crediting the generated method for a trivial queue data-structure change.
Results include all raw receipts and median ratios by shape and policy.
The empirical gate considers CPU improvement across four shapes and RSS not
more than 10% above original, but this is only a research triage criterion:
**no automatic merge or universal optimization claim**.

Run from the repository root with Python 3.10+ and git:

    python -m unittest discover -s tests -p 'test_resource_genesis_do_readiness.py' -v
    python -m resource_genesis.benchmark --samples 3 --repeats 3 --output resource-results.json

## Limits

The benchmark is synthetic and micro-level. Improved traversal does not
demonstrate reduced CPU time or RAM for the entire PCB pipeline, other Python
projects, databases or arbitrary applications. Peak RSS is noisy and
hardware/platform dependent; compare paired CI runs and, before a merge,
replay on representative real graph inputs and conduct an end-to-end
PCB pipeline regression. A future version can learn new search heuristics,
but must keep the verifier and original baseline independent.
