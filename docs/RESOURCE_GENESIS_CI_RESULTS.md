# Amantia Resource Genesis v0.1 — GitHub Actions CPU/RAM evidence

**Execution:** [PR workflow 44535270796](https://github.com/EM1805/causal-safety-engine/actions/runs/44535270796), November 19, 2026, Python 3.12 on GitHub Ubuntu runner. **Decision: REVIEW_CANDIDATE, not merged/promoted.**

- Frozen native baseline: `08b2e17bf5d00d165909390f21293a5c18515e7d`
- New algorithm: small-parent deque BFS; multi-parent reverse reachability
- Strong fixed control: deque-only, parent-wise BFS
- Correctness: 4/4 independent frozen-original tests passed, including random and cyclic graphs. All nine pre-existing PR workflows and the dedicated workflow passed.
- Sampling: three fresh subprocesses per algorithm and synthetic graph type, three calls per sample; below values are **medians**.
- Full machine-readable receipts: [GitHub workflow artifact](https://github.com/EM1805/causal-safety-engine/actions/runs/44535270796/artifacts/14535257896).
- JSON report SHA-256: `9b6800cace49ad8025fc1c75428050b5b3617f3cd47a43e7aa3dfb2deefc4c40`

| Workload (synthetic) | Original CPU/query | Candidate CPU/query | Original process peak RSS | Candidate RSS | Original Python peak alloc | Candidate alloc |
|---|---:|---:|---:|---:|---:|---:|
| Narrow reachable | 0.3766 ms | 0.3526 ms | 22,280 KiB | 21,640 KiB | 41,736 B | 43,096 B |
| Wide mixed | 23.8979 ms | 0.01372 ms | 22,312 KiB | 21,640 KiB | 43,784 B | 13,568 B |
| Wide reachable | 29.9821 ms | 0.2478 ms | 22,288 KiB | 21,640 KiB | 49,928 B | 51,968 B |
| Wide unreachable | 28.1961 ms | 0.00520 ms | 22,216 KiB | 21,640 KiB | 41,736 B | 9,032 B |

**Interpretation.** The CPU gain is large specifically for shared reachability queries with many parents; the narrow case improves only ~6%. Total process RSS was about 2.6–3.0% lower in this run, but much of that metric is interpreter and graph memory; the difference is small enough that OS startup effects may explain part of it. Temporary allocations **increase** by 3.3% in narrow and 4.1% in wide reachable, so it is false to claim uniform RAM reduction.

The original and candidate have the **same outputs on all tested cases**, including cycles, not a mathematical proof on every graph. The experiment covers a single function under deliberately chosen graph shapes, not end-to-end PCB processing or a random software sample.

**Before production promotion:** replay on representative real graph inputs, repeat on another machine, profile full pipeline including memory and latency, and confirm no regression in the original scientific/safety semantics.
