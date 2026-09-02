"""BenchmarkHarness: drives every agent through every scenario.

Execution model per (scenario, agent):
    - agent.reset(), simulated clock starts at 0 turn-units
    - for each session: start_session(); each turn advances the clock by 1;
      end_session() runs the agent's memory lifecycle; a session boundary
      advances the clock by SESSION_GAP units so MRDF recency decay operates
      on a compressed timescale
    - probe turns are scored: deterministic keyword hit rate, plus
      LLM-as-judge quality/personalization when evaluate_quality is True

Keyword hits are substring matches (case-insensitive) so that "2 business
days" and "400 Wh/kg" style phrases work as expected.
"""
import time
from datetime import datetime, timezone
from typing import Dict, List

from benchmarks.judge import LLMJudge
from benchmarks.metrics import BenchmarkResults, RunAggregate
from infrastructure.mrdf import SESSION_GAP
from scenarios.noise import NOISE_PRELOAD


class BenchmarkHarness:
    """Runs agents x scenarios and aggregates results."""

    def __init__(self, bedrock_client, judge_model_id: str,
                 evaluate_quality: bool = True, turn_pause_s: float = 0.2):
        self.bedrock = bedrock_client
        self.judge = LLMJudge(bedrock_client, judge_model_id)
        self.evaluate_quality = evaluate_quality
        # Small pause between calls to stay clear of throttling.
        self.turn_pause_s = turn_pause_s
        self.results = BenchmarkResults()

    @staticmethod
    def _keyword_hit_rate(response: str, keywords: List[str]) -> float:
        """Fraction of expected keywords present (case-insensitive)."""
        if not keywords:
            return 0.0
        low = response.lower()
        hits = sum(1 for k in keywords if k.lower() in low)
        return hits / len(keywords)

    def run_one(self, agent, scenario: Dict, verbose: bool = False) -> Dict:
        """Run a single agent through a single scenario; returns row dict."""
        agent.reset()
        agent.max_tokens = scenario.get("max_tokens", 512)
        # Persistent-memory agents start with identical aged service
        # history (see scenarios/noise.py), so retrieval quality under
        # accumulated noise is measured, not just cold-start behavior.
        if hasattr(agent, "preload_memories"):
            agent.preload_memories(NOISE_PRELOAD, created_at=-200.0)
        agg = RunAggregate()
        now = 0.0
        for s_idx, session in enumerate(scenario["sessions"], 1):
            agent.start_session()
            for turn in session:
                user_msg = turn["user"]
                response, metrics = agent.respond(user_msg, now)
                agg.record_turn(metrics)
                if turn.get("probe"):
                    agg.keyword_hits.append(self._keyword_hit_rate(
                        response, turn.get("expected_keywords", [])))
                    if self.evaluate_quality:
                        scores = self.judge.evaluate(
                            user_msg, response,
                            turn.get("memory_facts", []))
                        agg.quality_scores.append(scores["quality"])
                        agg.personalization_scores.append(
                            scores["personalization"])
                        agg.judge_calls += 1
                now += 1.0
                if self.turn_pause_s:
                    time.sleep(self.turn_pause_s)
            agent.end_session(now)
            now += SESSION_GAP  # simulated gap between sessions
            if verbose:
                print(f"      session {s_idx}/{len(scenario['sessions'])} done")
        return agg.finalize(agent)

    def run_all(self, agents: Dict[str, object], scenarios: List[Dict],
                verbose: bool = False) -> BenchmarkResults:
        """Run every agent through every scenario; returns BenchmarkResults."""
        self.results = BenchmarkResults()
        self.results.run_timestamp = datetime.now(
            timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for scenario in scenarios:
            sname = scenario["name"]
            if verbose:
                print(f"\n  Scenario: {sname}")
            for aname, agent in agents.items():
                if verbose:
                    print(f"    Agent: {aname}")
                row = self.run_one(agent, scenario, verbose=verbose)
                self.results.add(sname, aname, row)
                if verbose:
                    print(f"      avg latency {row['avg_latency_ms']:.0f} ms, "
                          f"tokens {row['total_input_tokens']}, "
                          f"cost ${row['total_cost']:.4f}, "
                          f"quality {row['avg_quality']:.2f}")
        # Record judge/model metadata for the report header.
        self.results.llm_model_id = getattr(
            self.judge, "judge_model_id", "")
        return self.results

    def save_results(self, path: str) -> str:
        """Persist raw results JSON; returns the path written."""
        return self.results.save(path)
