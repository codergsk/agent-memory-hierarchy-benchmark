"""In-memory vector store and memory item model for the AMH benchmark.

Models the storage substrate for hierarchy levels L1-L3:
    L1 working   - current-session turns (buffer semantics)
    L2 episodic  - per-exchange facts encoded at session end
    L3 semantic  - consolidated durable facts (promoted from L2)

A single MemoryStore holds items tagged with their level; retrieval and
eviction operate through MRDF scoring (infrastructure/mrdf.py).
"""
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

_id_counter = itertools.count(1)


@dataclass
class MemoryItem:
    """One memory: content plus the metadata MRDF scoring needs."""
    content: str
    level: str                       # "working" | "episodic" | "semantic"
    embedding: Optional[List[float]] = None
    created_at: float = 0.0          # turn-unit timestamp
    last_access: float = 0.0
    access_count: int = 0
    importance: float = 0.5          # base importance assigned at encoding
    outcome_modifier: float = 1.0    # adjusted by downstream outcomes
    pinned: bool = False             # exempt from eviction (semantic facts)
    consolidated: bool = False       # True once compressed into a semantic fact
    source_ids: List[int] = field(default_factory=list)
    memory_id: int = field(default_factory=lambda: next(_id_counter))

    def touch(self, now: float) -> None:
        """Record an access (updates recency and frequency signals)."""
        self.last_access = now
        self.access_count += 1


class MemoryStore:
    """Flat store with level tags; retrieval/eviction via MRDF outside."""

    def __init__(self):
        self.items: List[MemoryItem] = []
        self.created_total = 0
        self.forgotten_total = 0

    # -------------------------------------------------------------- writes
    def add(self, item: MemoryItem) -> MemoryItem:
        """Insert a memory and count it toward lifecycle stats."""
        self.items.append(item)
        self.created_total += 1
        return item

    def remove(self, items: Sequence[MemoryItem]) -> int:
        """Evict memories (forgetting); returns count removed."""
        ids = {m.memory_id for m in items}
        before = len(self.items)
        self.items = [m for m in self.items if m.memory_id not in ids]
        removed = before - len(self.items)
        self.forgotten_total += removed
        return removed

    # --------------------------------------------------------------- reads
    def by_level(self, level: str) -> List[MemoryItem]:
        """All memories at one hierarchy level."""
        return [m for m in self.items if m.level == level]

    def retrievable(self) -> List[MemoryItem]:
        """Memories eligible for MRDF retrieval (episodic + semantic)."""
        return [m for m in self.items if m.level in ("episodic", "semantic")]

    def average_embedding(self) -> Optional[List[float]]:
        """Mean embedding across retrievable memories (c_avg for eviction)."""
        embs = [m.embedding for m in self.retrievable() if m.embedding]
        if not embs:
            return None
        dim = len(embs[0])
        return [sum(e[i] for e in embs) / len(embs) for i in range(dim)]
