"""Memory subsystem used by the simplified SAFLA stack.

This module intentionally keeps the implementation pragmatic: we expose
lightweight in-memory data structures that satisfy the behaviour expected by
our tests and by the higher level orchestration code.  The goal is to provide
predictable, deterministic storage primitives without pulling in heavyweight
third‑party dependencies such as FAISS or NetworkX.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
import json
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Entry definitions
# ---------------------------------------------------------------------------


class CompressionLevel(Enum):
    """Represents the compression applied to a vector entry."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class VectorMemoryEntry:
    """Embeds semantic information as a dense vector."""

    vector: np.ndarray
    content: Optional[str] = None
    concept_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    compressed: bool = False
    compression_level: CompressionLevel = CompressionLevel.NONE

    def copy(self) -> "VectorMemoryEntry":
        return VectorMemoryEntry(
            vector=self.vector.copy(),
            content=self.content,
            concept_tags=list(self.concept_tags),
            metadata=dict(self.metadata),
            importance=self.importance,
            timestamp=self.timestamp,
            compressed=self.compressed,
            compression_level=self.compression_level,
        )


@dataclass
class EpisodicMemoryEntry:
    """Stores full trading episodes with context and outcome."""

    context: Dict[str, Any] = field(default_factory=dict)
    action: Optional[str] = None
    outcome: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    emotional_valence: Optional[float] = None
    sequence_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        context_text = " ".join(f"{k}:{v}" for k, v in self.context.items())
        outcome_text = " ".join(f"{k}:{v}" for k, v in self.outcome.items())
        return f"{self.action or ''} {context_text} {outcome_text}".strip()


@dataclass
class SemanticMemoryEntry:
    """Represents higher-level knowledge constructs."""

    concept: str
    definition: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkingMemoryEntry:
    """Keeps short-lived context required for current decision making."""

    content: Dict[str, Any]
    priority: int = 1
    attention_weight: float = 0.5
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    timestamp: float = field(default_factory=time.time)

    def score(self) -> float:
        # Higher priority and importance lead to larger attention.
        return float(self.priority) * 0.6 + float(self.importance) * 0.4


class MemoryType(Enum):
    VECTOR = "vector"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _dict_to_text(payload: Dict[str, Any]) -> str:
    return " ".join(f"{key}:{payload[key]}" for key in sorted(payload))


def _ensure_vector(vector: Union[np.ndarray, Iterable[float]], dimension: int) -> np.ndarray:
    if isinstance(vector, dict):
        return _text_to_vector(_dict_to_text(vector), dimension)
    arr = np.asarray(vector, dtype=np.float32).flatten()
    if arr.size == 0:
        arr = np.zeros(dimension, dtype=np.float32)
    if arr.size < dimension:
        arr = np.pad(arr, (0, dimension - arr.size))
    elif arr.size > dimension:
        arr = arr[:dimension]
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def _text_to_vector(text: str, dimension: int) -> np.ndarray:
    """Deterministically hash arbitrary text into a dense vector."""

    if not text:
        return np.zeros(dimension, dtype=np.float32)

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat digest until we have enough bytes, then view as uint32 to inject
    # some variance before normalising.
    repeat = (dimension * 4 + len(digest) - 1) // len(digest)
    raw = (digest * repeat)[: dimension * 4]
    arr = np.frombuffer(raw, dtype=np.uint32).astype(np.float32)
    arr = arr.reshape(-1)
    arr -= arr.mean()
    norm = np.linalg.norm(arr)
    if norm == 0:
        return np.zeros(dimension, dtype=np.float32)
    return (arr / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Vector memory
# ---------------------------------------------------------------------------


class VectorMemory:
    def __init__(self, dimension: int = 128, max_size: int = 10_000, cache_path: Optional[str] = None):
        self.dimension = dimension
        self.max_size = max_size
        self._entries: Dict[str, VectorMemoryEntry] = {}
        self._order: Deque[str] = deque()
        self._cache_path = cache_path
        self._use_cache = cache_path is not None

        # Load from cache if available
        if self._use_cache:
            self._load_from_cache()

    def _load_from_cache(self) -> None:
        """Load vector entries from cache file."""
        if not self._use_cache:
            return

        cache_path = Path(self._cache_path)
        if not cache_path.exists():
            return

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'entries' in data and 'order' in data:
                    # Validate and load entries
                    for entry_id, entry_data in data['entries'].items():
                        if len(self._entries) >= self.max_size:
                            break

                        # Reconstruct entry
                        entry = VectorMemoryEntry(
                            vector=entry_data['vector'],
                            content=entry_data.get('content'),
                            concept_tags=entry_data.get('concept_tags', []),
                            metadata=entry_data.get('metadata', {}),
                            importance=entry_data.get('importance', 0.5),
                            timestamp=entry_data.get('timestamp', time.time()),
                            compressed=entry_data.get('compressed', False),
                            compression_level=CompressionLevel(entry_data.get('compression_level', 'none'))
                        )

                        self._entries[entry_id] = entry
                        self._order.append(entry_id)

        except (pickle.PickleError, KeyError, ValueError) as e:
            print(f"Warning: Failed to load vector cache {self._cache_path}: {e}")

    def _save_to_cache(self) -> None:
        """Save vector entries to cache file."""
        if not self._use_cache:
            return

        try:
            # Prepare data for serialization
            cache_data = {
                'entries': {},
                'order': list(self._order),
                'metadata': {
                    'dimension': self.dimension,
                    'max_size': self.max_size,
                    'timestamp': time.time()
                }
            }

            for entry_id, entry in self._entries.items():
                cache_data['entries'][entry_id] = {
                    'vector': entry.vector,
                    'content': entry.content,
                    'concept_tags': entry.concept_tags,
                    'metadata': entry.metadata,
                    'importance': entry.importance,
                    'timestamp': entry.timestamp,
                    'compressed': entry.compressed,
                    'compression_level': entry.compression_level.value
                }

            # Ensure directory exists
            cache_path = Path(self._cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

        except (pickle.PickleError, OSError) as e:
            print(f"Warning: Failed to save vector cache {self._cache_path}: {e}")

    def store(self, entry: VectorMemoryEntry) -> str:
        entry_id = uuid.uuid4().hex
        vector = _ensure_vector(entry.vector, self.dimension)
        stored = entry.copy()
        stored.vector = vector
        if len(self._entries) >= self.max_size:
            oldest_id = self._order.popleft()
            self._entries.pop(oldest_id, None)
        self._entries[entry_id] = stored
        self._order.append(entry_id)

        # Periodically save to cache (every 10 entries to avoid excessive I/O)
        if self._use_cache and len(self._entries) % 10 == 0:
            self._save_to_cache()

        return entry_id

    def retrieve(self, entry_id: str) -> Optional[VectorMemoryEntry]:
        entry = self._entries.get(entry_id)
        return entry.copy() if entry else None

    def search(self, query: Union[str, np.ndarray], limit: int = 5) -> List[Tuple[VectorMemoryEntry, float]]:
        if isinstance(query, str):
            query_vector = _text_to_vector(query, self.dimension)
        else:
            query_vector = _ensure_vector(query, self.dimension)
        results: List[Tuple[VectorMemoryEntry, float]] = []
        for entry in self._entries.values():
            score = float(np.dot(entry.vector, query_vector))
            results.append((entry.copy(), score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:limit]

    def compress(self, entry_id: str, level: CompressionLevel) -> bool:
        entry = self._entries.get(entry_id)
        if not entry:
            return False
        entry.compressed = level != CompressionLevel.NONE
        entry.compression_level = level
        return True

    def get_concept_cluster(self, concept_tag: str) -> List[VectorMemoryEntry]:
        matches = [entry.copy() for entry in self._entries.values() if concept_tag in entry.concept_tags]
        return matches

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._entries),
            "dimension": self.dimension,
        }

    def shutdown(self) -> None:
        """Shutdown and ensure final save to cache."""
        if self._use_cache:
            self._save_to_cache()


# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------


class EpisodicMemory:
    def __init__(self, max_size: int = 50_000, db_path: Optional[str] = None):
        self.max_size = max_size
        self._entries: Dict[str, EpisodicMemoryEntry] = {}
        self._order: Deque[str] = deque()
        self._db_path = db_path
        self._use_persistence = db_path is not None
        self._persistence_failures = 0
        self._max_persistence_failures = 5

        # Initialize persistent storage if path provided
        if self._use_persistence:
            self._init_database()
            self._load_from_database()

    @property
    def entries(self) -> Dict[str, EpisodicMemoryEntry]:
        return self._entries

    def _init_database(self) -> None:
        """Initialize SQLite database for persistent storage."""
        if not self._use_persistence:
            return

        # Ensure directory exists
        db_path = Path(self._db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create connection and table
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memories (
                id TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                action TEXT,
                outcome TEXT NOT NULL,
                importance REAL NOT NULL,
                emotional_valence REAL,
                sequence_id TEXT,
                timestamp REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON episodic_memories(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sequence_id ON episodic_memories(sequence_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON episodic_memories(importance)")
        conn.commit()
        conn.close()

    def _load_from_database(self) -> None:
        """Load existing entries from database."""
        if not self._use_persistence:
            return

        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.execute("""
                SELECT id, context, action, outcome, importance, emotional_valence,
                       sequence_id, timestamp
                FROM episodic_memories
                ORDER BY timestamp DESC
                LIMIT ?
            """, (self.max_size,))

            for row in cursor:
                entry_id, context_json, action, outcome_json, importance, emotional_valence, sequence_id, timestamp = row

                # Deserialize JSON fields
                context = json.loads(context_json) if context_json else {}
                outcome = json.loads(outcome_json) if outcome_json else {}

                # Create entry
                entry = EpisodicMemoryEntry(
                    context=context,
                    action=action,
                    outcome=outcome,
                    importance=importance,
                    emotional_valence=emotional_valence,
                    sequence_id=sequence_id,
                    timestamp=timestamp
                )

                self._entries[entry_id] = entry
                self._order.append(entry_id)

            conn.close()
        except sqlite3.Error as e:
            # Log error but don't fail - degrade gracefully to in-memory only
            print(f"Warning: Failed to load from database {self._db_path}: {e}")
            # Disable persistence to prevent further errors
            self._use_persistence = False

    def _persist_entry(self, entry_id: str, entry: EpisodicMemoryEntry) -> None:
        """Persist single entry to database."""
        if not self._use_persistence:
            return

        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                INSERT OR REPLACE INTO episodic_memories
                (id, context, action, outcome, importance, emotional_valence, sequence_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id,
                json.dumps(entry.context),
                entry.action,
                json.dumps(entry.outcome),
                entry.importance,
                entry.emotional_valence,
                entry.sequence_id,
                entry.timestamp
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            # Log error but don't fail
            print(f"Warning: Failed to persist entry {entry_id}: {e}")
            # Disable persistence after multiple failures
            self._check_persistence_health()

    def _remove_from_database(self, entry_id: str) -> None:
        """Remove entry from database."""
        if not self._use_persistence:
            return

        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("DELETE FROM episodic_memories WHERE id = ?", (entry_id,))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Warning: Failed to remove entry {entry_id}: {e}")

    def _check_persistence_health(self) -> None:
        """Check persistence health and disable if too many failures."""
        self._persistence_failures += 1
        if self._persistence_failures >= self._max_persistence_failures:
            print(f"Warning: Disabling persistence after {self._persistence_failures} failures")
            self._use_persistence = False

    def store(self, entry: EpisodicMemoryEntry) -> str:
        entry_id = uuid.uuid4().hex

        # Handle max size limit
        if len(self._entries) >= self.max_size:
            oldest_id = self._order.popleft()
            removed_entry = self._entries.pop(oldest_id, None)
            if removed_entry and self._use_persistence:
                self._remove_from_database(oldest_id)

        # Store in memory and persist to database
        self._entries[entry_id] = entry
        self._order.append(entry_id)

        if self._use_persistence:
            self._persist_entry(entry_id, entry)

        return entry_id

    def retrieve(self, entry_id: str) -> Optional[EpisodicMemoryEntry]:
        return self._entries.get(entry_id)

    def search(self, query: Union[str, Dict[str, Any]], limit: int = 10) -> List[EpisodicMemoryEntry]:
        results: List[Tuple[EpisodicMemoryEntry, float]] = []
        for entry in self._entries.values():
            if isinstance(query, str):
                haystack = entry.summary().lower()
                if query.lower() in haystack:
                    results.append((entry, 1.0))
            else:
                matches = True
                score = 0.0
                for key, value in query.items():
                    if key == "context":
                        for ctx_key, ctx_value in value.items():
                            if entry.context.get(ctx_key) != ctx_value:
                                matches = False
                                break
                            score += 1.0
                        if not matches:
                            break
                    else:
                        if getattr(entry, key, None) != value:
                            matches = False
                            break
                        score += 1.0
                if matches:
                    results.append((entry, max(score, 1.0)))
        results.sort(key=lambda item: item[1], reverse=True)
        return [entry for entry, _ in results[:limit]]

    def get_sequence(self, sequence_id: str) -> List[EpisodicMemoryEntry]:
        sequence = [entry for entry in self._entries.values() if entry.sequence_id == sequence_id]
        sequence.sort(key=lambda e: e.timestamp)
        return sequence

    def find_similar_episodes(
        self,
        reference_entry: EpisodicMemoryEntry,
        similarity_threshold: float = 0.5,
    ) -> List[EpisodicMemoryEntry]:
        results: List[EpisodicMemoryEntry] = []
        for entry in self._entries.values():
            score = 0.0
            # Context overlap
            shared_keys = set(reference_entry.context).intersection(entry.context)
            for key in shared_keys:
                if reference_entry.context.get(key) == entry.context.get(key):
                    score += 0.4
            # Action similarity
            if reference_entry.action and reference_entry.action == entry.action:
                score += 0.3
            # Outcome similarity (success flag or profit sign)
            ref_success = reference_entry.outcome.get("success")
            entry_success = entry.outcome.get("success")
            if ref_success is not None and ref_success == entry_success:
                score += 0.3
            if score >= similarity_threshold:
                results.append(entry)
        return results

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._entries),
        }


# ---------------------------------------------------------------------------
# Semantic memory
# ---------------------------------------------------------------------------


class _SimpleGraph:
    def __init__(self):
        self._adjacency: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    def add_node(self, node: str) -> None:
        # Accessing ensures a dict exists
        _ = self._adjacency[node]

    def add_edge(self, src: str, relation: str, dst: str) -> None:
        self._adjacency[src][relation].add(dst)
        self.add_node(dst)

    def number_of_nodes(self) -> int:
        return len(self._adjacency)

    def number_of_edges(self) -> int:
        count = 0
        for rels in self._adjacency.values():
            count += sum(len(targets) for targets in rels.values())
        return count

    def neighbors(self, node: str) -> Dict[str, set]:
        return self._adjacency.get(node, {})


class SemanticMemory:
    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._entries: Dict[str, SemanticMemoryEntry] = {}
        self._order: Deque[str] = deque()
        self.knowledge_graph = _SimpleGraph()

    @property
    def entries(self) -> Dict[str, SemanticMemoryEntry]:
        return self._entries

    def store(self, entry: SemanticMemoryEntry) -> str:
        entry_id = uuid.uuid4().hex
        if len(self._entries) >= self.max_size:
            oldest = self._order.popleft()
            self._entries.pop(oldest, None)
        self._entries[entry_id] = entry
        self._order.append(entry_id)

        self.knowledge_graph.add_node(entry.concept)
        for relation, targets in entry.relationships.items():
            for target in targets:
                self.knowledge_graph.add_edge(entry.concept, relation, target)
        return entry_id

    def retrieve(self, entry_id: str) -> Optional[SemanticMemoryEntry]:
        return self._entries.get(entry_id)

    def search(self, query: Union[str, Dict[str, Any]], limit: int = 10) -> List[SemanticMemoryEntry]:
        results: List[Tuple[SemanticMemoryEntry, float]] = []
        for entry in self._entries.values():
            score = 0.0
            if isinstance(query, str):
                props = " ".join(str(v) for v in entry.properties.values())
                text = " ".join([entry.concept, entry.definition or "", props]).lower()
                if query.lower() in text:
                    score = 1.0
            else:
                for key, value in query.items():
                    if key == "properties":
                        for prop_key, prop_value in value.items():
                            if entry.properties.get(prop_key) == prop_value:
                                score += 1.0
                    elif getattr(entry, key, None) == value:
                        score += 1.0
            if score > 0:
                results.append((entry, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return [entry for entry, _ in results[:limit]]

    def get_related_concepts(self, concept: str, max_depth: int = 1) -> List[str]:
        visited = set()
        queue: List[Tuple[str, int]] = [(concept, 0)]
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            for targets in self.knowledge_graph.neighbors(current).values():
                for target in targets:
                    queue.append((target, depth + 1))
        visited.discard(concept)
        return list(visited)

    def get_knowledge_summary(self) -> Dict[str, Any]:
        return {
            "total_concepts": self.knowledge_graph.number_of_nodes(),
            "total_relationships": self.knowledge_graph.number_of_edges(),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._entries),
        }


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------


class WorkingMemory:
    def __init__(self, context_window: int = 100, max_size: int = 1_000):
        self.context_window = context_window
        self.max_size = max_size
        self._entries: Dict[str, WorkingMemoryEntry] = {}
        self._order: Deque[str] = deque()
        self._goals: Dict[str, Dict[str, Any]] = {}
        self._context: Dict[str, Any] = {}

    def store(self, entry: WorkingMemoryEntry) -> str:
        entry_id = uuid.uuid4().hex
        if len(self._entries) >= self.max_size:
            oldest_id = self._order.popleft()
            self._entries.pop(oldest_id, None)
        # Auto-derive attention weight if not explicitly provided.
        if entry.attention_weight is None:
            entry.attention_weight = min(1.0, max(0.0, entry.score() / 10.0))
        self._entries[entry_id] = entry
        self._order.append(entry_id)
        return entry_id

    def retrieve(self, entry_id: str) -> Optional[WorkingMemoryEntry]:
        return self._entries.get(entry_id)

    def search(self, query: Union[str, Dict[str, Any]], limit: int = 5) -> List[WorkingMemoryEntry]:
        matches: List[Tuple[WorkingMemoryEntry, float]] = []
        for entry in self._entries.values():
            score = 0.0
            if isinstance(query, str):
                haystack = " ".join(str(v) for v in entry.content.values()).lower()
                if query.lower() in haystack:
                    score = entry.score()
            else:
                for key, value in query.items():
                    if entry.content.get(key) == value:
                        score += 1.0
            if score > 0:
                matches.append((entry, score))
        matches.sort(key=lambda item: item[1], reverse=True)
        return [entry for entry, _ in matches[:limit]]

    def get_focused_content(self, limit: int = 5) -> List[Dict[str, Any]]:
        ranked = sorted(self._entries.values(), key=lambda e: (e.attention_weight, e.score()), reverse=True)
        focused = []
        for entry in ranked[:limit]:
            focused.append(
                {
                    "content": entry.content,
                    "attention_weight": entry.attention_weight,
                    "importance": entry.importance,
                    "priority": entry.priority,
                }
            )
        return focused

    def set_context(self, context: Dict[str, Any]) -> None:
        self._context = dict(context)

    def add_goal(self, description: str, priority: int = 1) -> str:
        goal_id = uuid.uuid4().hex
        self._goals[goal_id] = {
            "description": description,
            "priority": priority,
            "progress": 0.0,
            "completed": False,
            "created": time.time(),
        }
        return goal_id

    def update_goal_progress(self, goal_id: str, progress: float) -> bool:
        goal = self._goals.get(goal_id)
        if not goal:
            return False
        goal["progress"] = float(min(max(progress, 0.0), 1.0))
        if goal["progress"] >= 1.0:
            goal["completed"] = True
        return True

    def get_active_goals(self) -> List[Dict[str, Any]]:
        return [
            {**goal, "goal_id": goal_id}
            for goal_id, goal in self._goals.items()
            if not goal["completed"]
        ]

    def shutdown(self) -> None:
        self._entries.clear()
        self._goals.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._entries),
            "active_goals": len([g for g in self._goals.values() if not g["completed"]]),
        }


# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------


class MemoryManager:
    def __init__(self, vector_dim: int = 128, config=None):
        # Get config if not provided
        if config is None:
            from ..config import get_config
            config = get_config()

        self.config = config

        # Setup persistence paths
        storage_dir = None
        vector_cache_path = None
        episodic_db_path = None
        semantic_cache_path = None

        if hasattr(config, 'memory_persistence_enabled') and config.memory_persistence_enabled:
            storage_dir = Path(config.memory_storage_directory)
            storage_dir.mkdir(parents=True, exist_ok=True)

            vector_cache_path = str(storage_dir / config.vector_cache_file)
            episodic_db_path = str(storage_dir / config.episodic_db_file)
            semantic_cache_path = str(storage_dir / config.semantic_cache_file)

        # Initialize memory systems with config values and persistence
        self.vector_memory = VectorMemory(
            dimension=config.vector_dimension if hasattr(config, 'vector_dimension') else vector_dim,
            max_size=config.vector_max_entries if hasattr(config, 'vector_max_entries') else 10000,
            cache_path=vector_cache_path
        )
        self.episodic_memory = EpisodicMemory(
            max_size=config.episodic_max_entries if hasattr(config, 'episodic_max_entries') else 50000,
            db_path=episodic_db_path
        )
        self.semantic_memory = SemanticMemory(
            max_size=config.semantic_max_entries if hasattr(config, 'semantic_max_entries') else 10000
        )
        self.working_memory = WorkingMemory(
            max_size=config.working_memory_max_entries if hasattr(config, 'working_memory_max_entries') else 1000
        )
        self.performance_stats: Dict[str, Any] = {
            "comprehensive_memories": 0,
            "last_consolidation": None,
        }

    def store_memory(self, memory_type: MemoryType, entry: Any) -> str:
        if memory_type == MemoryType.VECTOR:
            return self.vector_memory.store(entry)
        if memory_type == MemoryType.EPISODIC:
            return self.episodic_memory.store(entry)
        if memory_type == MemoryType.SEMANTIC:
            return self.semantic_memory.store(entry)
        if memory_type == MemoryType.WORKING:
            return self.working_memory.store(entry)
        raise ValueError(f"Unsupported memory type: {memory_type}")

    def create_comprehensive_memory(self, experience: Dict[str, Any]) -> Dict[str, str]:
        context = experience.get("context", {})
        action = experience.get("action")
        outcome = experience.get("outcome", {})
        importance = float(experience.get("importance", 0.5))
        summary = experience.get("summary") or self._summarise_experience(experience)

        vector = _text_to_vector(summary, self.vector_memory.dimension)
        vector_entry = VectorMemoryEntry(
            vector=vector,
            content=summary,
            concept_tags=list(context.values())[:3],
            importance=importance,
        )
        vector_id = self.vector_memory.store(vector_entry)

        episodic_entry = EpisodicMemoryEntry(
            context=dict(context),
            action=action,
            outcome=dict(outcome),
            importance=importance,
        )
        episodic_id = self.episodic_memory.store(episodic_entry)

        semantic_entry = SemanticMemoryEntry(
            concept=action or f"experience_{episodic_id[:6]}",
            definition=summary,
            properties={"importance": importance},
            relationships={"context": [str(v) for v in context.values()]},
        )
        semantic_id = self.semantic_memory.store(semantic_entry)

        working_entry = WorkingMemoryEntry(
            content={"context": context, "action": action, "outcome": outcome},
            priority=3,
            importance=importance,
            attention_weight=min(1.0, 0.5 + importance / 2),
        )
        working_id = self.working_memory.store(working_entry)

        self.performance_stats["comprehensive_memories"] += 1
        return {
            "vector": vector_id,
            "episodic": episodic_id,
            "semantic": semantic_id,
            "working": working_id,
        }

    def search_memories(self, query: Union[str, Dict[str, Any]], limit: int = 5) -> Dict[str, List[Any]]:
        return {
            "vector": [entry for entry, _ in self.vector_memory.search(query, limit)],
            "episodic": self.episodic_memory.search(query, limit),
            "semantic": self.semantic_memory.search(query, limit),
            "working": self.working_memory.search(query, limit),
        }

    def consolidate_memories(self, timeframe: Optional[float] = None) -> Dict[str, Any]:
        consolidated = min(len(self.episodic_memory._entries), 5)
        self.performance_stats["last_consolidation"] = time.time()
        return {
            "episodes_consolidated": consolidated,
            "timestamp": self.performance_stats["last_consolidation"],
        }

    def optimize_memory_usage(self) -> Dict[str, Any]:
        removed = 0
        # Drop low-importance vector entries
        low_importance_ids = [
            entry_id
            for entry_id, entry in list(self.vector_memory._entries.items())
            if entry.importance < 0.2
        ]
        for entry_id in low_importance_ids[:5]:
            self.vector_memory._entries.pop(entry_id, None)
            removed += 1
        return {
            "entries_removed": removed,
            "vector_size": len(self.vector_memory._entries),
        }

    def get_contextual_memories(self, context: Dict[str, Any], limit_per_type: int = 3) -> Dict[str, List[Any]]:
        return self.search_memories(context, limit_per_type)

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "memory_systems": {
                "vector": self.vector_memory.stats(),
                "episodic": self.episodic_memory.stats(),
                "semantic": self.semantic_memory.stats(),
                "working": self.working_memory.stats(),
            },
            "performance_stats": dict(self.performance_stats),
        }

    def shutdown(self) -> None:
        """Shutdown and persist all memory systems."""
        # Ensure final persistence for vector memory
        if hasattr(self.vector_memory, 'shutdown'):
            self.vector_memory.shutdown()

        # Working memory doesn't need persistence (short-term by design)
        self.working_memory.shutdown()

        # Clear in-memory caches (data is already persisted)
        self.vector_memory._entries.clear()
        self.episodic_memory._entries.clear()
        self.semantic_memory._entries.clear()

    @staticmethod
    def _summarise_experience(experience: Dict[str, Any]) -> str:
        bits = []
        context = experience.get("context", {})
        if context:
            bits.append("context:" + ",".join(f"{k}={v}" for k, v in context.items()))
        action = experience.get("action")
        if action:
            bits.append(f"action:{action}")
        outcome = experience.get("outcome", {})
        if outcome:
            bits.append("outcome:" + ",".join(f"{k}={v}" for k, v in outcome.items()))
        return " | ".join(bits) or "experience"


__all__ = [
    "CompressionLevel",
    "VectorMemoryEntry",
    "EpisodicMemoryEntry",
    "SemanticMemoryEntry",
    "WorkingMemoryEntry",
    "MemoryType",
    "VectorMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "WorkingMemory",
    "MemoryManager",
]
