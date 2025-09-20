"""
WORKING MEMORY SYSTEM
Attention-based active context management for real-time trading decisions
"""

import numpy as np
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging


class AttentionMechanism:
    """Attention mechanism for focus management"""

    def __init__(self, dimension: int = 64):
        """Initialize attention mechanism

        Args:
            dimension: Attention dimension
        """
        self.dimension = dimension

        # Simple attention weights (in production, use neural attention)
        self.weights = np.ones(dimension) / dimension

    def compute_attention(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Compute attention weights

        Args:
            query: Query vector
            keys: Key vectors

        Returns:
            Attention weights
        """
        if len(keys) == 0:
            return np.array([])

        # Simple dot-product attention
        scores = np.dot(keys, query)

        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / np.sum(exp_scores)

        return attention_weights

    def update_weights(self, feedback: float, learning_rate: float = 0.01):
        """Update attention weights based on feedback

        Args:
            feedback: Performance feedback (-1 to 1)
            learning_rate: Learning rate
        """
        # Simple weight update (in production, use proper gradients)
        adjustment = feedback * learning_rate
        self.weights += adjustment
        self.weights = np.clip(self.weights, 0.01, 2.0)
        self.weights /= np.sum(self.weights)


class WorkingMemoryItem:
    """Item in working memory"""

    def __init__(self, item_id: str, content: Any, importance: float = 1.0,
                 attention_weight: float = 1.0, decay_rate: float = 0.1):
        """Initialize working memory item

        Args:
            item_id: Unique identifier
            content: Item content
            importance: Static importance score
            attention_weight: Current attention weight
            decay_rate: How fast importance decays
        """
        self.item_id = item_id
        self.content = content
        self.importance = importance
        self.attention_weight = attention_weight
        self.decay_rate = decay_rate
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0

    def access(self) -> Any:
        """Access the item, updating statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        return self.content

    def get_current_importance(self) -> float:
        """Get current importance (with decay)"""
        time_since_creation = (datetime.now() - self.created_at).total_seconds() / 3600
        time_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600

        # Decay based on time
        decay_factor = np.exp(-self.decay_rate * time_since_access)

        # Boost based on access frequency
        frequency_boost = 1 + 0.1 * self.access_count

        return self.importance * decay_factor * frequency_boost

    def update_attention(self, weight: float):
        """Update attention weight"""
        self.attention_weight = max(0.0, min(2.0, weight))


class WorkingMemory:
    """Attention-based working memory for active trading context"""

    def __init__(self, capacity: int = 50, attention_dimension: int = 64):
        """Initialize working memory

        Args:
            capacity: Maximum number of items to hold
            attention_dimension: Dimension for attention mechanism
        """
        self.capacity = capacity
        self.attention = AttentionMechanism(attention_dimension)

        # Thread safety
        self._lock = threading.RLock()

        # Memory storage
        self.items: Dict[str, WorkingMemoryItem] = {}
        self.access_history: deque = deque(maxlen=1000)

        # Focus management
        self.current_focus: List[str] = []  # Currently focused item IDs
        self.focus_threshold = 0.5  # Minimum attention weight for focus

    def add_item(self, item_id: str, content: Any, importance: float = 1.0,
                item_type: str = 'general', metadata: Optional[Dict] = None) -> bool:
        """Add item to working memory

        Args:
            item_id: Unique identifier
            content: Item content
            importance: Static importance score
            item_type: Type of item
            metadata: Additional metadata

        Returns:
            True if added successfully
        """
        with self._lock:
            # Check capacity
            if len(self.items) >= self.capacity and item_id not in self.items:
                self._evict_least_important()

            # Create item
            item = WorkingMemoryItem(item_id, {
                'content': content,
                'type': item_type,
                'metadata': metadata or {}
            }, importance)

            self.items[item_id] = item

            # Update attention
            self._update_attention_weights()

            return True

    def get_item(self, item_id: str) -> Optional[Any]:
        """Get item by ID

        Args:
            item_id: Item identifier

        Returns:
            Item content or None
        """
        with self._lock:
            if item_id not in self.items:
                return None

            item = self.items[item_id]
            content = item.access()

            # Record access
            self.access_history.append({
                'item_id': item_id,
                'timestamp': datetime.now(),
                'action': 'access'
            })

            # Update attention
            self._update_attention_weights()

            return content

    def get_focused_items(self, max_items: int = 10) -> List[Tuple[str, Any, float]]:
        """Get currently focused items

        Args:
            max_items: Maximum items to return

        Returns:
            List of (item_id, content, attention_weight) tuples
        """
        with self._lock:
            focused = []

            for item_id, item in self.items.items():
                if item.attention_weight >= self.focus_threshold:
                    focused.append((item_id, item.content, item.attention_weight))

            # Sort by attention weight
            focused.sort(key=lambda x: x[2], reverse=True)
            return focused[:max_items]

    def search_items(self, query: str, item_type: Optional[str] = None,
                    max_results: int = 5) -> List[Tuple[str, Any, float]]:
        """Search items by content

        Args:
            query: Search query
            item_type: Filter by item type
            max_results: Maximum results

        Returns:
            List of (item_id, content, relevance_score) tuples
        """
        with self._lock:
            results = []

            for item_id, item in self.items.items():
                content = item.content

                # Type filter
                if item_type and content.get('type') != item_type:
                    continue

                # Simple relevance scoring
                relevance = self._calculate_relevance(query, content)
                if relevance > 0.1:
                    results.append((item_id, content, relevance))

            # Sort by relevance
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:max_results]

    def update_item_importance(self, item_id: str, new_importance: float) -> bool:
        """Update item importance

        Args:
            item_id: Item identifier
            new_importance: New importance score

        Returns:
            True if updated
        """
        with self._lock:
            if item_id not in self.items:
                return False

            self.items[item_id].importance = max(0.0, new_importance)
            self._update_attention_weights()
            return True

    def provide_feedback(self, item_id: str, feedback: float):
        """Provide feedback on item utility

        Args:
            item_id: Item identifier
            feedback: Feedback score (-1 to 1)
        """
        with self._lock:
            if item_id not in self.items:
                return

            # Update item importance based on feedback
            current_importance = self.items[item_id].importance
            adjustment = feedback * 0.1
            new_importance = max(0.1, current_importance + adjustment)
            self.items[item_id].importance = new_importance

            # Update attention mechanism
            self.attention.update_weights(feedback)
            self._update_attention_weights()

    def clear_focus(self):
        """Clear current focus"""
        with self._lock:
            self.current_focus.clear()
            for item in self.items.values():
                item.attention_weight = 0.5

    def consolidate_memory(self, min_importance: float = 0.1) -> List[str]:
        """Remove items below importance threshold

        Args:
            min_importance: Minimum importance to keep

        Returns:
            List of removed item IDs
        """
        with self._lock:
            to_remove = []

            for item_id, item in self.items.items():
                current_importance = item.get_current_importance()
                if current_importance < min_importance:
                    to_remove.append(item_id)

            for item_id in to_remove:
                del self.items[item_id]

            logging.info(f"Consolidated working memory: removed {len(to_remove)} items")
            return to_remove

    def _evict_least_important(self):
        """Evict least important item to make space"""
        if not self.items:
            return

        # Find least important item
        least_important_id = min(self.items.keys(),
                               key=lambda x: self.items[x].get_current_importance())

        del self.items[least_important_id]

    def _update_attention_weights(self):
        """Update attention weights for all items"""
        if not self.items:
            return

        # Create importance vector
        importances = np.array([item.get_current_importance()
                              for item in self.items.values()])

        if len(importances) == 0:
            return

        # Normalize and apply attention
        normalized_importance = importances / (np.sum(importances) + 1e-8)

        # Update item attention weights
        for i, item_id in enumerate(self.items.keys()):
            self.items[item_id].update_attention(normalized_importance[i])

        # Update current focus
        self.current_focus = [item_id for item_id, item in self.items.items()
                            if item.attention_weight >= self.focus_threshold]

    def _calculate_relevance(self, query: str, content: Dict[str, Any]) -> float:
        """Calculate relevance score between query and content

        Args:
            query: Search query
            content: Item content

        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Simple keyword matching (in production, use better methods)
        query_words = set(query.lower().split())

        content_text = str(content.get('content', ''))
        content_words = set(content_text.lower().split())

        if not query_words:
            return 0.0

        # Jaccard similarity
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))

        return intersection / union if union > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self._lock:
            if not self.items:
                return {
                    'total_items': 0,
                    'focused_items': 0,
                    'avg_importance': 0.0,
                    'capacity_usage': 0.0
                }

            importances = [item.get_current_importance() for item in self.items.values()]
            focused_count = len(self.current_focus)

            return {
                'total_items': len(self.items),
                'focused_items': focused_count,
                'avg_importance': np.mean(importances),
                'max_importance': np.max(importances),
                'min_importance': np.min(importances),
                'capacity_usage': len(self.items) / self.capacity
            }

    def clear(self):
        """Clear all items"""
        with self._lock:
            self.items.clear()
            self.current_focus.clear()
            self.access_history.clear()