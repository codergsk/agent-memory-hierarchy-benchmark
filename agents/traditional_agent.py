"""Baseline agents representing three generations of memory approaches.

    Gen1 StatelessAgent   - no memory at all (the Groundhog Day baseline)
    Gen2 BufferAgent      - sliding window over the CURRENT session only;
                            loses all cross-session context
    Gen3 NaiveVectorAgent - append-only vector store over every turn ever
                            seen; retrieves top-k by cosine similarity;
                            never forgets, so retrieval noise grows

All agents share the BaseAgent lifecycle used by the benchmark harness:
    start_session() / respond(user_msg, now) / end_session(now)
and report per-turn metrics (latency, tokens, cost, memories retrieved).
"""
from typing import Dict, List, Tuple

from infrastructure.bedrock_client import BedrockClient
from infrastructure.memory_store import MemoryItem, MemoryStore
from infrastructure.mrdf import cosine_similarity

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user directly and concisely. "
    "If context about the user or prior conversations is provided, use it "
    "to personalize your answer."
)


class BaseAgent:
    """Common lifecycle and accounting for all benchmark agents."""

    name = "BaseAgent"

    def __init__(self, bedrock_client: BedrockClient,
                 llm_model_id: str, max_tokens: int = 512):
        self.bedrock = bedrock_client
        self.llm_model_id = llm_model_id
        self.max_tokens = max_tokens
        # Lifecycle counters surfaced in the report's AMH lifecycle table.
        self.consolidations = 0
        self.forgettings = 0
        self.memories_created = 0
        self.retrieved_counts: List[int] = []
        self.extra_cost = 0.0  # embeddings + memory-maintenance LLM calls

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Reset all state before a fresh scenario run."""
        self.consolidations = 0
        self.forgettings = 0
        self.memories_created = 0
        self.retrieved_counts = []
        self.extra_cost = 0.0

    def start_session(self) -> None:
        """Called at the start of each conversation session."""

    def end_session(self, now: float) -> None:
        """Called at the end of each conversation session."""

    # ------------------------------------------------------------- helpers
    def _generate(self, context_blocks: List[str], user_msg: str,
                  temperature: float = 0.3) -> Tuple[str, Dict]:
        """Build the prompt from context blocks and call the LLM."""
        parts = []
        for block in context_blocks:
            if block:
                parts.append(block)
        parts.append(f"User: {user_msg}")
        messages = [{"role": "user", "content": "\n\n".join(parts)}]
        text, metrics = self.bedrock.generate_text(
            messages=messages, model_id=self.llm_model_id,
            max_tokens=self.max_tokens, temperature=temperature,
            system_prompt=SYSTEM_PROMPT)
        metrics["cost"] = self.bedrock.calculate_cost(
            metrics["input_tokens"], metrics["output_tokens"],
            self.llm_model_id)
        return text, metrics

    def _embed_cost(self, text: str) -> float:
        """Approximate embedding cost from a 4-chars-per-token estimate."""
        est_tokens = max(1, len(text) // 4)
        return self.bedrock.calculate_cost(
            est_tokens, 0, "amazon.titan-embed-text-v2:0")

    def respond(self, user_msg: str, now: float) -> Tuple[str, Dict]:
        """Produce a reply and per-turn metrics. Subclasses override."""
        raise NotImplementedError


class StatelessAgent(BaseAgent):
    """Gen1: every request answered cold, with zero memory."""

    name = "StatelessAgent"

    def respond(self, user_msg: str, now: float) -> Tuple[str, Dict]:
        """Answer with no context at all."""
        self.retrieved_counts.append(0)
        return self._generate([], user_msg)


class BufferAgent(BaseAgent):
    """Gen2: sliding window over the current session; wiped between sessions."""

    name = "BufferAgent"

    def __init__(self, bedrock_client: BedrockClient, buffer_size: int = 10,
                 llm_model_id: str = "", max_tokens: int = 512):
        super().__init__(bedrock_client, llm_model_id, max_tokens)
        self.buffer_size = buffer_size
        self.buffer: List[str] = []

    def reset(self) -> None:
        """Clear counters and the conversation buffer."""
        super().reset()
        self.buffer = []

    def start_session(self) -> None:
        """New session starts with an empty buffer: no cross-session memory."""
        self.buffer = []

    def respond(self, user_msg: str, now: float) -> Tuple[str, Dict]:
        """Answer with the last N turns of the current session as context."""
        window = self.buffer[-self.buffer_size:]
        ctx = ("Conversation so far:\n" + "\n".join(window)) if window else ""
        self.retrieved_counts.append(len(window))
        text, metrics = self._generate([ctx], user_msg)
        self.buffer.append(f"User: {user_msg}")
        self.buffer.append(f"Assistant: {text}")
        return text, metrics


class NaiveVectorAgent(BaseAgent):
    """Gen3: append-only vector store over every turn; top-k retrieval.

    Persists across sessions but never scores for recency/importance and
    never forgets, so old and irrelevant turns keep polluting retrieval.
    """

    name = "NaiveVectorAgent"

    def __init__(self, bedrock_client: BedrockClient, top_k: int = 5,
                 embedding_model_id: str = "amazon.titan-embed-text-v2:0",
                 llm_model_id: str = "", max_tokens: int = 512):
        super().__init__(bedrock_client, llm_model_id, max_tokens)
        self.top_k = top_k
        self.embedding_model_id = embedding_model_id
        self.store = MemoryStore()

    def reset(self) -> None:
        """Clear counters and the vector store."""
        super().reset()
        self.store = MemoryStore()

    def _embed(self, text: str) -> List[float]:
        """Embed text and account for its (tiny) cost."""
        emb, _ = self.bedrock.embed_text(text, self.embedding_model_id)
        self.extra_cost += self._embed_cost(text)
        return emb

    def preload_memories(self, items: List[str], created_at: float) -> None:
        """Seed the store with aged memories (simulated service history).

        The naive store treats them exactly like any other memory: pure
        similarity retrieval with no recency or importance weighting -
        which is precisely the weakness under test.
        """
        for content in items:
            self.store.add(MemoryItem(
                content=content, level="episodic",
                embedding=self._embed(content),
                created_at=created_at, last_access=created_at))

    def respond(self, user_msg: str, now: float) -> Tuple[str, Dict]:
        """Retrieve top-k similar past turns, answer, store the exchange."""
        q_emb = self._embed(user_msg)
        scored = [(cosine_similarity(q_emb, m.embedding), m)
                  for m in self.store.items if m.embedding]
        scored.sort(key=lambda t: t[0], reverse=True)
        top = [m for _, m in scored[: self.top_k]]
        self.retrieved_counts.append(len(top))
        ctx = ""
        if top:
            ctx = ("Potentially relevant past exchanges (unfiltered):\n"
                   + "\n".join(f"- {m.content}" for m in top))
        text, metrics = self._generate([ctx], user_msg)
        # Store the full exchange verbatim - naive, unbounded accumulation.
        content = f"User said: {user_msg} | Assistant replied: {text[:300]}"
        self.store.add(MemoryItem(content=content, level="episodic",
                                  embedding=self._embed(content),
                                  created_at=now, last_access=now))
        self.memories_created += 1
        return text, metrics
