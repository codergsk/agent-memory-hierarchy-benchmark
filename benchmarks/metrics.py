"""Metric aggregation and the BenchmarkResults container.

Per (scenario, agent) run we aggregate:
    latency  - average and p95 per-turn LLM latency (ms)
    tokens   - total input / output tokens across the scenario
    cost     - total USD (turn LLM calls + embeddings + maintenance calls)
             - plus cost normalized per 1,000 turns for comparability
    quality  - LLM-judge quality and personalization averages over probes
    keywords - deterministic keyword hit rate over probe turns
    lifecycle- memories created / consolidations / forgettings /
               average memories retrieved per turn

Gains vs. baseline follow the original report's conventions:
    cost/latency/tokens: (baseline - agent) / baseline * 100
        (positive = agent is cheaper/faster/leaner; negative = regression)
    quality/personalization/keywords: (agent - baseline) / baseline * 100
        (positive = agent scores higher)
"""
import json
import statistics
from typing import Dict, List, Optional


def p95(values: List[float]) -> float:
    """95th percentile with a small-sample-safe fallback."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    # Dashboard-style rank: 95% of samples fall at or below this index.
    idx = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return ordered[idx]


class RunAggregate:
    """Accumulates per-turn observations for one (scenario, agent) run."""

    def __init__(self):
        self.latencies_ms: List[float] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.turn_cost = 0.0
        self.quality_scores: List[float] = []
        self.personalization_scores: List[float] = []
        self.keyword_hits: List[float] = []
        self.judge_calls = 0
        # Lifecycle counters copied from the agent at run end.
        self.memories_created = 0
        self.consolidations = 0
        self.forgettings = 0
        self.avg_retrieved = 0.0
        self.extra_cost = 0.0
        self.turns = 0

    def record_turn(self, metrics: Dict) -> None:
        """Record one conversational turn's LLM metrics."""
        self.turns += 1
        self.latencies_ms.append(metrics["latency_ms"])
        self.input_tokens += metrics["input_tokens"]
        self.output_tokens += metrics["output_tokens"]
        self.turn_cost += metrics.get("cost", 0.0)

    def finalize(self, agent) -> Dict:
        """Copy agent lifecycle counters and emit the aggregate row dict."""
        self.memories_created = agent.memories_created
        self.consolidations = agent.consolidations
        self.forgettings = agent.forgettings
        self.extra_cost = agent.extra_cost
        counts = agent.retrieved_counts
        self.avg_retrieved = (sum(counts) / len(counts)) if counts else 0.0
        total_cost = self.turn_cost + self.extra_cost
        return {
            "turns": self.turns,
            "avg_latency_ms": (statistics.mean(self.latencies_ms)
                               if self.latencies_ms else 0.0),
            "p95_latency_ms": p95(self.latencies_ms),
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_cost": total_cost,
            "cost_per_1k_turns": (total_cost / self.turns * 1000
                                  if self.turns else 0.0),
            "avg_quality": (statistics.mean(self.quality_scores)
                            if self.quality_scores else 0.0),
            "avg_personalization": (statistics.mean(self.personalization_scores)
                                    if self.personalization_scores else 0.0),
            "keyword_hit_rate": (statistics.mean(self.keyword_hits)
                                 if self.keyword_hits else 0.0),
            "judge_calls": self.judge_calls,
            "memories_created": self.memories_created,
            "consolidations": self.consolidations,
            "forgettings": self.forgettings,
            "avg_retrieved_per_turn": self.avg_retrieved,
        }


class BenchmarkResults:
    """Results container: {scenario: {agent: aggregate-row-dict}}."""

    # Metrics where LOWER raw values are better (gain sign flips).
    LOWER_IS_BETTER = {
        "latency_delta_pct": "avg_latency_ms",
        "cost_delta_pct": "total_cost",
        "token_delta_pct": "total_input_tokens",
    }
    # Metrics where HIGHER raw values are better.
    HIGHER_IS_BETTER = {
        "quality_delta_pct": "avg_quality",
        "personalization_delta_pct": "avg_personalization",
        "keyword_delta_pct": "keyword_hit_rate",
    }

    def __init__(self, llm_model_id: str = "", embedding_model_id: str = ""):
        self.data: Dict[str, Dict[str, Dict]] = {}
        self.llm_model_id = llm_model_id
        self.embedding_model_id = embedding_model_id
        self.run_timestamp: Optional[str] = None

    def add(self, scenario: str, agent: str, row: Dict) -> None:
        """Store the aggregate row for one (scenario, agent) run."""
        self.data.setdefault(scenario, {})[agent] = row

    def total_judge_calls(self) -> int:
        """Total LLM-as-judge invocations across the whole run."""
        return sum(row.get("judge_calls", 0)
                   for agents in self.data.values()
                   for row in agents.values())

    def gains_vs_baseline(self, baseline: str) -> Dict[str, Dict[str, Dict]]:
        """Per-agent, per-scenario percentage deltas vs. the baseline agent.

        Returns {agent: {scenario: {metric_delta_pct: value}}} using the
        sign conventions documented in the module docstring.
        """
        out: Dict[str, Dict[str, Dict]] = {}
        for scenario, agents in self.data.items():
            base = agents.get(baseline)
            if not base:
                continue
            for agent, row in agents.items():
                if agent == baseline:
                    continue
                deltas: Dict[str, float] = {}
                for key, field in self.LOWER_IS_BETTER.items():
                    b, a = base[field], row[field]
                    deltas[key] = ((b - a) / b * 100.0) if b else 0.0
                for key, field in self.HIGHER_IS_BETTER.items():
                    b, a = base[field], row[field]
                    # Floor the denominator at 0.05: a stateless baseline can
                    # legitimately score 0.0 on personalization, and dividing
                    # by ~0 produces absurd percentages. The floor keeps the
                    # delta meaningful ("vs a 0.05-grade baseline") and is
                    # noted in the report.
                    b = max(b, 0.05)
                    deltas[key] = (a - b) / b * 100.0
                out.setdefault(agent, {})[scenario] = deltas
        return out

    # -------------------------------------------------------- serialization
    def to_dict(self) -> Dict:
        """Full results as a JSON-serializable dict."""
        return {
            "llm_model_id": self.llm_model_id,
            "embedding_model_id": self.embedding_model_id,
            "run_timestamp": self.run_timestamp,
            "results": self.data,
        }

    def save(self, path: str) -> str:
        """Write results JSON to path; returns the path."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "BenchmarkResults":
        """Rehydrate a BenchmarkResults from a saved JSON file."""
        with open(path) as f:
            payload = json.load(f)
        obj = cls(payload.get("llm_model_id", ""),
                  payload.get("embedding_model_id", ""))
        obj.run_timestamp = payload.get("run_timestamp")
        obj.data = payload["results"]
        return obj
