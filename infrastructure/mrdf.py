"""Memory Relevance Decay Function (MRDF).

Implements Section 4 of 'Production Agent Memory: Architecture Patterns for
Stateful AI Systems':

    MRDF(m, c, t) = alpha * R(m, t) + beta * F(m) + gamma * I(m) + delta * S(m, c)

Where:
    R(m, t) = exp(-lambda * (t - t_last_access))      (recency, Ebbinghaus decay)
    F(m)    = log(1 + access_count) / log(1 + max_ac) (frequency, log-scaled)
    I(m)    = base_importance * outcome_modifier       (importance)
    S(m, c) = cosine_similarity(emb(m), emb(c))        (similarity)

Time is measured in abstract "turn units" during benchmarking: each
conversational turn advances the clock by 1.0 and each session boundary by
SESSION_GAP units, which lets the exponential decay operate on a compressed
timescale instead of wall-clock days.
"""
import math
from typing import Dict, List, Sequence

# Canonical parameter profiles (whitepaper Table 4.3). Weights sum to 1.
PROFILES: Dict[str, Dict[str, float]] = {
    "recency_dominant": {
        "alpha": 0.40, "beta": 0.10, "gamma": 0.15, "delta": 0.35,
        "lam": 0.05,  # decay per turn-unit
    },
    "importance_dominant": {
        "alpha": 0.15, "beta": 0.15, "gamma": 0.40, "delta": 0.30,
        "lam": 0.02,
    },
    "similarity_dominant": {
        "alpha": 0.10, "beta": 0.10, "gamma": 0.20, "delta": 0.60,
        "lam": 0.02,
    },
}

# Simulated clock advance at a session boundary (turn units).
SESSION_GAP = 24.0
# Eviction threshold (whitepaper Section 4.4). Memories scoring below this
# against the average context are candidates for temporal decay.
THETA_EVICT = 0.35


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors (0.0 on zero-norm input)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class MRDF:
    """Parameterized memory relevance scorer with eviction support."""

    def __init__(self, profile: str = "recency_dominant"):
        if profile not in PROFILES:
            raise ValueError(f"Unknown MRDF profile: {profile}")
        self.profile_name = profile
        p = PROFILES[profile]
        self.alpha = p["alpha"]
        self.beta = p["beta"]
        self.gamma = p["gamma"]
        self.delta = p["delta"]
        self.lam = p["lam"]

    # ---------------------------------------------------------- components
    def recency(self, memory, now: float) -> float:
        """R(m, t): exponential decay since last access."""
        dt = max(0.0, now - memory.last_access)
        return math.exp(-self.lam * dt)

    @staticmethod
    def frequency(memory, max_access_count: int) -> float:
        """F(m): log access count, normalized by pool maximum."""
        if max_access_count <= 0:
            return 0.0
        return math.log(1 + memory.access_count) / math.log(1 + max_access_count)

    @staticmethod
    def importance(memory) -> float:
        """I(m): encoding-time importance adjusted by outcomes."""
        return max(0.0, min(1.0, memory.importance * memory.outcome_modifier))

    def similarity(self, memory, context_embedding: Sequence[float]) -> float:
        """S(m, c): cosine similarity of memory and context embeddings."""
        if memory.embedding is None or context_embedding is None:
            return 0.0
        return max(0.0, cosine_similarity(memory.embedding, context_embedding))

    # -------------------------------------------------------------- scoring
    def score(self, memory, context_embedding: Sequence[float], now: float,
              max_access_count: int) -> float:
        """Full MRDF score for one memory against the current context."""
        return (self.alpha * self.recency(memory, now)
                + self.beta * self.frequency(memory, max_access_count)
                + self.gamma * self.importance(memory)
                + self.delta * self.similarity(memory, context_embedding))

    def rank(self, memories: List, context_embedding: Sequence[float],
             now: float) -> List[tuple]:
        """Return (score, memory) pairs sorted by descending relevance."""
        max_ac = max((m.access_count for m in memories), default=0)
        scored = [(self.score(m, context_embedding, now, max_ac), m)
                  for m in memories]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    # ------------------------------------------------------------- eviction
    def eviction_candidates(self, memories: List, avg_context: Sequence[float],
                            now: float, theta: float = THETA_EVICT) -> List:
        """Memories whose MRDF against the average context falls below theta.

        Implements the temporal-decay forgetting trigger (whitepaper 5.2).
        Consolidated source episodes are the primary candidates: their
        content lives on in the semantic memory that summarizes them.
        """
        max_ac = max((m.access_count for m in memories), default=0)
        out = []
        for m in memories:
            if m.pinned:
                continue
            s = self.score(m, avg_context, now, max_ac)
            if s < theta:
                out.append(m)
        return out
