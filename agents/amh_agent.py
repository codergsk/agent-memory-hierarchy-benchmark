"""AMHAgent: the Agent Memory Hierarchy implementation under benchmark.

Implements the whitepaper's L0-L3 lifecycle for a single-agent deployment:

    L0 immediate context - rebuilt per request from L1-L3 via MRDF selection
    L1 working memory    - sliding window over the current session
    L2 episodic memory   - salient facts encoded by the LLM at session end
    L3 semantic memory   - durable facts consolidated from related episodes

Plus two forgetting patterns exercised by the benchmark:
    consolidation   - >= CONSOLIDATE_MIN related episodic facts are
                      compressed into one semantic fact (Pattern 3)
    temporal decay  - consolidated source episodes whose MRDF score falls
                      below the eviction threshold are removed (Pattern 1)
"""
import json
from typing import Dict, List, Tuple

from agents.traditional_agent import BaseAgent
from infrastructure.bedrock_client import BedrockClient
from infrastructure.memory_store import MemoryItem, MemoryStore
from infrastructure.mrdf import MRDF, SESSION_GAP, cosine_similarity

# Retrieval breadth for L0 reconstruction (memories injected per turn).
RETRIEVE_K = 5
# Relevance floor: memories below this cosine similarity to the current
# context are never injected or access-touched. Without it, early turns
# (before any real facts exist) inject stale noise, and touch() then
# boosts that noise's recency/frequency - a rich-get-richer loop that
# lets junk outrank real facts. Calibrated empirically for Titan v2
# short-text embeddings: scenario facts score ~0.10-0.40 against
# context-enriched queries, stale noise ~0.03-0.10.
SIM_FLOOR = 0.10
# Minimum related episodes required to trigger consolidation.
CONSOLIDATE_MIN = 3
# Similarity threshold for "related" episodes during consolidation.
# Deliberately tight: loose thresholds cluster adjacent-but-different
# topics (e.g. "battery research" with "battery recycling" noise), and a
# blended semantic fact plus source eviction then DESTROYS accurate
# memories - the whitepaper's consolidation-quality risk (Section 10).
CONSOLIDATE_SIM = 0.62
# Working-memory window (turns) kept verbatim in the prompt.
WORKING_WINDOW = 6

ENCODE_PROMPT = (
    "Extract the durable, user-specific facts from this conversation that "
    "would help personalize future conversations (preferences, constraints, "
    "identity details, decisions, open issues). Return a JSON array of "
    "short strings, each one self-contained fact. Return ONLY the JSON "
    "array, no prose. If nothing is worth remembering, return [].\n\n"
    "Conversation:\n{transcript}"
)

CONSOLIDATE_PROMPT = (
    "Compress these related memory items into ONE concise fact that "
    "preserves the essential durable pattern. Return only the fact text.\n\n"
    "{items}"
)


class AMHAgent(BaseAgent):
    """Hierarchy-managed agent with MRDF retrieval and principled forgetting."""

    name = "AMHAgent"

    def __init__(self, bedrock_client: BedrockClient,
                 mrdf_profile: str = "recency_dominant",
                 embedding_model_id: str = "amazon.titan-embed-text-v2:0",
                 llm_model_id: str = "", max_tokens: int = 512):
        super().__init__(bedrock_client, llm_model_id, max_tokens)
        self.mrdf = MRDF(profile=mrdf_profile)
        self.embedding_model_id = embedding_model_id
        self.store = MemoryStore()
        self.working: List[str] = []          # L1: current-session turns
        self._session_transcript: List[str] = []

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Clear counters, the store, and session state."""
        super().reset()
        self.store = MemoryStore()
        self.working = []
        self._session_transcript = []

    def start_session(self) -> None:
        """Fresh L1 working memory for the new session."""
        self.working = []
        self._session_transcript = []

    def end_session(self, now: float) -> None:
        """Encode episodics from the session, then consolidate and forget."""
        if self._session_transcript:
            self._encode_episodics(now)
        self._consolidate(now)
        self._forget(now)

    # -------------------------------------------------------------- helpers
    def _embed(self, text: str) -> List[float]:
        """Embed text and account for its cost in extra_cost."""
        emb, _ = self.bedrock.embed_text(text, self.embedding_model_id)
        self.extra_cost += self._embed_cost(text)
        return emb

    def preload_memories(self, items: List[str], created_at: float) -> None:
        """Seed aged episodic memories (identical list to the naive agent).

        AMH stores them with old timestamps and default importance, so the
        MRDF recency term (exp(-lambda * age)) naturally down-weights them
        at retrieval time - the differentiation under test.
        """
        for content in items:
            self.store.add(MemoryItem(
                content=content, level="episodic",
                embedding=self._embed(content),
                created_at=created_at, last_access=created_at,
                importance=0.4))

    def _maintenance_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """LLM call for memory maintenance (encoding / consolidation).

        Costs are charged to extra_cost, not per-turn metrics, mirroring
        how maintenance runs out-of-band from the user interaction.
        """
        text, metrics = self.bedrock.generate_text(
            messages=[{"role": "user", "content": prompt}],
            model_id=self.llm_model_id, max_tokens=max_tokens,
            temperature=0.0)
        self.extra_cost += self.bedrock.calculate_cost(
            metrics["input_tokens"], metrics["output_tokens"],
            self.llm_model_id)
        return text

    # ------------------------------------------------------------ encoding
    def _encode_episodics(self, now: float) -> None:
        """L2 encoding: LLM extracts salient facts from the ended session."""
        transcript = "\n".join(self._session_transcript)
        # Generous budget: truncated JSON was silently dropping all facts.
        raw = self._maintenance_llm(
            ENCODE_PROMPT.format(transcript=transcript), max_tokens=1000)
        facts: List[str] = []
        try:
            # The model is instructed to return a bare JSON array; tolerate
            # code fences or leading prose by slicing to the brackets.
            start, end = raw.find("["), raw.rfind("]")
            if start >= 0 and end > start:
                facts = [str(f) for f in json.loads(raw[start:end + 1])]
        except (json.JSONDecodeError, ValueError):
            facts = []
        if not facts:
            # Fallback for truncated/malformed JSON: salvage quoted strings
            # so one bad generation cannot silently zero out the memory store.
            import re
            facts = re.findall(r'"([^"\n]{8,200})"', raw)
        for fact in facts[:8]:  # cap per-session encodings
            self.store.add(MemoryItem(
                content=fact, level="episodic",
                embedding=self._embed(fact),
                created_at=now, last_access=now, importance=0.6))
            self.memories_created += 1

    # ------------------------------------------------------- consolidation
    def _consolidate(self, now: float) -> None:
        """Pattern 3: compress clusters of related episodes into semantics."""
        episodes = [m for m in self.store.by_level("episodic")
                    if not m.consolidated and m.embedding]
        used: set = set()
        for anchor in episodes:
            if anchor.memory_id in used:
                continue
            cluster = [anchor]
            for other in episodes:
                if other.memory_id in used or other is anchor:
                    continue
                if cosine_similarity(anchor.embedding,
                                     other.embedding) >= CONSOLIDATE_SIM:
                    cluster.append(other)
            if len(cluster) >= CONSOLIDATE_MIN:
                items = "\n".join(f"- {m.content}" for m in cluster)
                fact = self._maintenance_llm(
                    CONSOLIDATE_PROMPT.format(items=items),
                    max_tokens=150).strip()
                self.store.add(MemoryItem(
                    content=fact, level="semantic",
                    embedding=self._embed(fact),
                    created_at=now, last_access=now,
                    importance=0.8, pinned=True,
                    source_ids=[m.memory_id for m in cluster]))
                self.memories_created += 1
                self.consolidations += 1
                for m in cluster:
                    m.consolidated = True  # now eligible for decay
                    used.add(m.memory_id)

    # ----------------------------------------------------------- forgetting
    def _forget(self, now: float) -> None:
        """Pattern 1: temporal decay of low-MRDF consolidated episodes."""
        avg = self.store.average_embedding()
        if avg is None:
            return
        # Grace period: only consolidated sources older than one session
        # gap are evictable, so a fact can never be consolidated and
        # destroyed within the same session it was learned.
        aged = [m for m in self.store.by_level("episodic")
                if m.consolidated and (now - m.created_at) > SESSION_GAP]
        candidates = self.mrdf.eviction_candidates(
            aged, avg_context=avg, now=now)
        if candidates:
            self.forgettings += self.store.remove(candidates)

    # -------------------------------------------------------------- respond
    def respond(self, user_msg: str, now: float) -> Tuple[str, Dict]:
        """Reconstruct L0 from L1 + MRDF-selected L2/L3, then answer."""
        # Context-enriched query: MRDF's S(m, c) is similarity to the
        # CONTEXT, not the bare question. "Suggest a restaurant for date
        # night" alone carries no dietary signal; with the last exchange
        # appended, the embedding points at the right memories.
        recent = " ".join(self.working[-2:])[-500:]
        q_emb = self._embed(f"{user_msg} {recent}".strip())
        ranked = self.mrdf.rank(self.store.retrievable(), q_emb, now)
        # Apply the relevance floor BEFORE selection: low-similarity
        # memories are neither injected nor strengthened by touch().
        top = []
        for _, m in ranked:
            if len(top) >= RETRIEVE_K:
                break
            if self.mrdf.similarity(m, q_emb) >= SIM_FLOOR:
                top.append(m)
        for m in top:
            m.touch(now)  # accessing a memory boosts recency + frequency
        self.retrieved_counts.append(len(top))

        blocks: List[str] = []
        if top:
            blocks.append("What you remember about this user "
                          "(most relevant first):\n"
                          + "\n".join(f"- {m.content}" for m in top))
        window = self.working[-WORKING_WINDOW:]
        if window:
            blocks.append("Current conversation:\n" + "\n".join(window))

        text, metrics = self._generate(blocks, user_msg)
        self.working.append(f"User: {user_msg}")
        self.working.append(f"Assistant: {text}")
        self._session_transcript.append(f"User: {user_msg}")
        self._session_transcript.append(f"Assistant: {text}")
        return text, metrics
