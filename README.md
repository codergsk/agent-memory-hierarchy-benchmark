# Agent Memory Hierarchy (AMH) Benchmark

Empirical benchmark for the concepts in *Production Agent Memory:
Architecture Patterns for Stateful AI Systems* — the Agent Memory
Hierarchy (AMH), the Memory Relevance Decay Function (MRDF), and the
Forgetting Architecture — comparing an AMH agent against three
generations of traditional memory approaches on Amazon Bedrock.

## Agents under test

| Agent | Generation | Memory approach |
|---|---|---|
| StatelessAgent | Gen1 | None — every request answered cold |
| BufferAgent | Gen2 | Sliding window over the current session only |
| NaiveVectorAgent | Gen3 | Append-only vector store, top-k cosine retrieval, never forgets |
| **AMHAgent** | This work | L1 working / L2 episodic / L3 semantic hierarchy, MRDF-scored retrieval, consolidation + temporal-decay forgetting |

## Scenarios

- **Customer Support** — cross-session continuity on an evolving support case
- **Personal Assistant** — preference memory and preference *evolution* (contradiction handling)
- **Multi-Topic Research** — retrieval precision under topic interference, long-form outputs

Probe turns in each scenario are scored three ways: deterministic keyword
hit rate, plus LLM-as-judge quality and personalization (0–1).

## Architecture

- `infrastructure/` — Bedrock client, MRDF scorer (4-factor, parameterized
  profiles), in-memory store with hierarchy levels
- `agents/` — the four agents
- `scenarios/` — the three multi-session scripted scenarios
- `benchmarks/` — harness, metric aggregation, LLM judge
- `reports/` — self-contained HTML report generator (Chart.js)

Models: `us.anthropic.claude-sonnet-4-20250514-v1:0` (agent + judge) and
`amazon.titan-embed-text-v2:0` (embeddings), configurable via environment.

## Run

```bash
pip install -r requirements.txt
# Credentials via your usual mechanism (profile/SSO). Then:
AWS_REGION=us-east-1 python3 run_benchmark.py
```

Outputs `benchmark_results_<ts>.json` (raw data) and
`benchmark_report_<ts>.html` (interactive report). Re-render a report
from saved data without re-running:

```bash
python3 generate_report_from_data.py benchmark_results_<ts>.json
```

## Cost & duration

A full run makes roughly 200–250 Claude Sonnet calls (~15–40 min,
~US$1.50–2.50 depending on output lengths). The MRDF clock runs in
compressed "turn units" so recency decay is exercised without real
multi-day waits.

## Notes on methodology

- Gains are reported against the Stateless baseline with the conventions:
  cost/latency/token deltas positive = leaner; quality deltas positive = better.
- Trajectory of memory lifecycle (created / consolidated / forgotten /
  retrieved-per-turn) is reported for the AMH agent per scenario.
- LLM-as-judge runs at temperature 0 with a JSON contract; malformed
  judgments score 0 rather than crashing the run.

## Reconstruction and comparison to the July 2026 run

This codebase is a faithful reconstruction of a benchmark whose original
source was lost (only `run_benchmark.py`, `bedrock_client.py`, and the
published HTML report survived — the original report is preserved in
`results/original-2026-07/`). The current results
(`results/benchmark_results_final.json`, `results/benchmark_report_final.html`)
differ from the July numbers for reasons that were reviewed deliberately:

1. **Model change (forced):** the original ran on Claude Sonnet 4, which
   is now provider-flagged Legacy and cannot be invoked. This run uses
   `us.anthropic.claude-sonnet-4-6` for agent, judge, and maintenance.
2. **Harder test conditions (intentional):** both persistent-memory
   agents are now preloaded with 24 identical aged "service history"
   memories (`scenarios/noise.py`), so retrieval quality is measured
   under accumulated noise rather than cold-start — closer to the
   whitepaper's scale-testing methodology than a 12-turn-only run.
3. **Directional findings hold, magnitudes differ:** AMH still beats
   Stateless and Buffer across the board and dominates on the
   cross-session Customer Support scenario (best quality,
   personalization, and keyword recall of all four agents). Under heavy
   noise, the naive vector store remains competitive on two scenarios in
   this run — an honest result worth studying, not tuned away.
4. **Known judge caveat:** quality/personalization are single-judge LLM
   scores at temperature 0; run-to-run variance of ±0.1 is normal.

Reconstruction lessons are captured in code comments: episodic encoding
needs a generous token budget plus a salvage parser (truncated JSON once
silently zeroed the memory store), consolidation thresholds must be tight
enough not to blend adjacent-but-different topics before evicting the
accurate originals, and retrieval needs a similarity floor with
context-enriched queries, calibrated to the embedding model
(Titan v2 short-text similarities are low; facts ~0.1–0.4, noise <0.1).
