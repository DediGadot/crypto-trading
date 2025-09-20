"""Tests for SAFLA memory systems."""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
import tempfile
import os

from safla_trading.memory import (
    MemoryManager, VectorMemory, EpisodicMemory, SemanticMemory, WorkingMemory,
    VectorMemoryEntry, EpisodicMemoryEntry, SemanticMemoryEntry, WorkingMemoryEntry,
    MemoryType, CompressionLevel
)


class TestVectorMemory:
    """Test vector memory system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.vector_memory = VectorMemory(dimension=128, max_size=1000)

    def test_vector_storage_and_retrieval(self):
        """Test storing and retrieving vectors."""
        vector = np.random.randn(128).astype(np.float32)
        entry = VectorMemoryEntry(
            vector=vector,
            content="test vector",
            concept_tags=["test", "vector"],
            importance=0.8
        )

        # Store vector
        entry_id = self.vector_memory.store(entry)
        assert entry_id is not None

        # Retrieve vector
        retrieved = self.vector_memory.retrieve(entry_id)
        assert retrieved is not None
        assert np.allclose(retrieved.vector, vector / np.linalg.norm(vector))  # Normalized
        assert retrieved.content == "test vector"
        assert "test" in retrieved.concept_tags

    def test_vector_search(self):
        """Test vector similarity search."""
        # Store some vectors
        vectors = []
        for i in range(10):
            vector = np.random.randn(128).astype(np.float32)
            vectors.append(vector)
            entry = VectorMemoryEntry(
                vector=vector,
                content=f"vector_{i}",
                concept_tags=[f"tag_{i}"]
            )
            self.vector_memory.store(entry)

        # Search with a query vector
        query_vector = vectors[0] + np.random.randn(128) * 0.1  # Similar to first vector
        results = self.vector_memory.search(query_vector, limit=5)

        assert len(results) > 0
        assert len(results) <= 5

        # Results should be sorted by similarity
        similarities = [score for _, score in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_vector_compression(self):
        """Test vector compression."""
        vector = np.random.randn(128).astype(np.float32)
        entry = VectorMemoryEntry(vector=vector, content="test")
        entry_id = self.vector_memory.store(entry)

        # Compress the vector
        success = self.vector_memory.compress(entry_id, CompressionLevel.MEDIUM)
        assert success

        # Retrieve and check compression
        retrieved = self.vector_memory.retrieve(entry_id)
        assert retrieved.compressed
        assert retrieved.compression_level == CompressionLevel.MEDIUM

    def test_concept_clustering(self):
        """Test concept clustering functionality."""
        # Store vectors with same concept tags
        for i in range(5):
            vector = np.random.randn(128).astype(np.float32)
            entry = VectorMemoryEntry(
                vector=vector,
                concept_tags=["trading", "momentum"]
            )
            self.vector_memory.store(entry)

        # Check concept cluster
        cluster_entries = self.vector_memory.get_concept_cluster("trading")
        assert len(cluster_entries) == 5


class TestEpisodicMemory:
    """Test episodic memory system."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.episodic_memory = EpisodicMemory(db_path=self.temp_db.name, max_size=1000)

    def teardown_method(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_db.name)

    def test_episode_storage_and_retrieval(self):
        """Test storing and retrieving episodes."""
        entry = EpisodicMemoryEntry(
            context={"market": "bull", "strategy": "momentum"},
            action="buy_btc",
            outcome={"profit": 0.05, "success": True},
            emotional_valence=0.8,
            importance=0.9
        )

        # Store episode
        entry_id = self.episodic_memory.store(entry)
        assert entry_id is not None

        # Retrieve episode
        retrieved = self.episodic_memory.retrieve(entry_id)
        assert retrieved is not None
        assert retrieved.action == "buy_btc"
        assert retrieved.context["market"] == "bull"
        assert retrieved.outcome["profit"] == 0.05

    def test_episode_search(self):
        """Test episode search functionality."""
        # Store multiple episodes
        episodes = [
            {"action": "buy_btc", "market": "bull", "profit": 0.05},
            {"action": "sell_eth", "market": "bear", "profit": -0.02},
            {"action": "buy_btc", "market": "sideways", "profit": 0.01},
        ]

        for i, ep in enumerate(episodes):
            entry = EpisodicMemoryEntry(
                context={"market": ep["market"], "index": i},
                action=ep["action"],
                outcome={"profit": ep["profit"]},
                importance=0.5 + i * 0.1
            )
            self.episodic_memory.store(entry)

        # Search by text
        results = self.episodic_memory.search("buy_btc", limit=5)
        assert len(results) == 2  # Two buy_btc episodes

        # Search by filters
        filter_query = {
            "action": "buy_btc",
            "context": {"market": "bull"}
        }
        results = self.episodic_memory.search(filter_query, limit=5)
        assert len(results) == 1

    def test_episode_sequences(self):
        """Test episode sequence functionality."""
        sequence_id = "test_sequence"

        # Store episodes in sequence
        for i in range(3):
            entry = EpisodicMemoryEntry(
                action=f"action_{i}",
                sequence_id=sequence_id,
                importance=0.5
            )
            self.episodic_memory.store(entry)

        # Retrieve sequence
        sequence = self.episodic_memory.get_sequence(sequence_id)
        assert len(sequence) == 3
        assert sequence[0].action == "action_0"

    def test_similar_episodes(self):
        """Test finding similar episodes."""
        reference_entry = EpisodicMemoryEntry(
            context={"market": "bull", "volatility": "high"},
            action="buy",
            outcome={"success": True},
            emotional_valence=0.8
        )

        # Store reference and similar episodes
        ref_id = self.episodic_memory.store(reference_entry)

        similar_entry = EpisodicMemoryEntry(
            context={"market": "bull", "volatility": "medium"},
            action="buy",
            outcome={"success": True},
            emotional_valence=0.7
        )
        self.episodic_memory.store(similar_entry)

        different_entry = EpisodicMemoryEntry(
            context={"market": "bear", "volatility": "low"},
            action="sell",
            outcome={"success": False},
            emotional_valence=-0.5
        )
        self.episodic_memory.store(different_entry)

        # Find similar episodes
        similar = self.episodic_memory.find_similar_episodes(reference_entry, similarity_threshold=0.5)
        assert len(similar) >= 1  # Should find at least the similar episode


class TestSemanticMemory:
    """Test semantic memory system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.semantic_memory = SemanticMemory(max_size=1000)

    def test_concept_storage_and_retrieval(self):
        """Test storing and retrieving concepts."""
        entry = SemanticMemoryEntry(
            concept="momentum_indicator",
            definition="Technical indicator measuring price momentum",
            properties={"type": "indicator", "timeframe": "short"},
            relationships={"is_a": ["technical_indicator"], "used_for": ["trend_analysis"]},
            confidence=0.9
        )

        # Store concept
        entry_id = self.semantic_memory.store(entry)
        assert entry_id is not None

        # Retrieve concept
        retrieved = self.semantic_memory.retrieve(entry_id)
        assert retrieved is not None
        assert retrieved.concept == "momentum_indicator"
        assert retrieved.properties["type"] == "indicator"
        assert "technical_indicator" in retrieved.relationships["is_a"]

    def test_concept_search(self):
        """Test concept search functionality."""
        # Store concepts
        concepts = [
            {"concept": "rsi", "definition": "Relative Strength Index", "type": "oscillator"},
            {"concept": "macd", "definition": "Moving Average Convergence Divergence", "type": "trend"},
            {"concept": "bollinger_bands", "definition": "Volatility indicator", "type": "volatility"},
        ]

        for concept_data in concepts:
            entry = SemanticMemoryEntry(
                concept=concept_data["concept"],
                definition=concept_data["definition"],
                properties={"type": concept_data["type"]}
            )
            self.semantic_memory.store(entry)

        # Search by concept name
        results = self.semantic_memory.search("rsi", limit=5)
        assert len(results) >= 1

        # Search by structured query
        query = {"properties": {"type": "oscillator"}}
        results = self.semantic_memory.search(query, limit=5)
        assert len(results) >= 1

    def test_concept_relationships(self):
        """Test concept relationship functionality."""
        # Store related concepts
        parent_entry = SemanticMemoryEntry(
            concept="technical_indicator",
            definition="Mathematical calculation based on price/volume",
            properties={"category": "analysis_tool"}
        )
        parent_id = self.semantic_memory.store(parent_entry)

        child_entry = SemanticMemoryEntry(
            concept="momentum_indicator",
            definition="Indicator measuring momentum",
            relationships={"is_a": ["technical_indicator"]}
        )
        child_id = self.semantic_memory.store(child_entry)

        # Get related concepts
        related = self.semantic_memory.get_related_concepts("momentum_indicator", max_depth=2)
        assert "technical_indicator" in related

    def test_knowledge_graph(self):
        """Test knowledge graph functionality."""
        # Store concepts with relationships
        concepts = [
            {"concept": "price", "relationships": {"has": ["open", "high", "low", "close"]}},
            {"concept": "volume", "relationships": {"indicates": ["activity"]}},
            {"concept": "trend", "relationships": {"determined_by": ["price", "volume"]}},
        ]

        for concept_data in concepts:
            entry = SemanticMemoryEntry(
                concept=concept_data["concept"],
                relationships=concept_data["relationships"]
            )
            self.semantic_memory.store(entry)

        # Check that knowledge graph has nodes
        assert self.semantic_memory.knowledge_graph.number_of_nodes() >= 3

        # Get knowledge summary
        summary = self.semantic_memory.get_knowledge_summary()
        assert summary["total_concepts"] >= 3
        assert summary["total_relationships"] >= 0


class TestWorkingMemory:
    """Test working memory system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.working_memory = WorkingMemory(context_window=100, max_size=500)

    def teardown_method(self):
        """Clean up test fixtures."""
        self.working_memory.shutdown()

    def test_working_memory_storage(self):
        """Test working memory storage and attention."""
        entry = WorkingMemoryEntry(
            content={"task": "analyze_market", "priority": "high"},
            priority=3,
            attention_weight=0.9,
            importance=0.8,
            active=True
        )

        # Store entry
        entry_id = self.working_memory.store(entry)
        assert entry_id is not None

        # Retrieve entry
        retrieved = self.working_memory.retrieve(entry_id)
        assert retrieved is not None
        assert retrieved.content["task"] == "analyze_market"
        assert retrieved.active

    def test_context_management(self):
        """Test context setting and relevance."""
        # Set context
        context = {"market_state": "bullish", "volatility": "high"}
        self.working_memory.set_context(context)

        # Store relevant entry
        entry = WorkingMemoryEntry(
            content={"analysis": "market_bullish"},
            metadata={"market_state": "bullish"},
            importance=0.7
        )
        entry_id = self.working_memory.store(entry)

        # Search by context
        results = self.working_memory.search("bullish", limit=5)
        assert len(results) >= 1

    def test_attention_mechanism(self):
        """Test attention mechanism."""
        # Store entries with different importance
        high_importance = WorkingMemoryEntry(
            content={"alert": "critical_signal"},
            importance=0.9,
            priority=5
        )
        low_importance = WorkingMemoryEntry(
            content={"info": "routine_update"},
            importance=0.3,
            priority=1
        )

        self.working_memory.store(high_importance)
        self.working_memory.store(low_importance)

        # Get focused content
        focused = self.working_memory.get_focused_content(limit=5)
        assert len(focused) >= 1

        # Higher importance should have higher attention
        if len(focused) >= 2:
            assert focused[0]['attention_weight'] >= focused[1]['attention_weight']

    def test_goal_tracking(self):
        """Test goal tracking functionality."""
        # Add goals
        goal_id = self.working_memory.add_goal("maximize_profit", priority=3)
        assert goal_id is not None

        # Check active goals
        active_goals = self.working_memory.get_active_goals()
        assert len(active_goals) == 1
        assert active_goals[0]['description'] == "maximize_profit"

        # Update goal progress
        success = self.working_memory.update_goal_progress(goal_id, 0.5)
        assert success

        # Complete goal
        success = self.working_memory.update_goal_progress(goal_id, 1.0)
        assert success

        # Check goal completion
        active_goals = self.working_memory.get_active_goals()
        assert len(active_goals) == 0  # Goal should be completed


class TestMemoryManager:
    """Test memory manager integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.memory_manager = MemoryManager()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.memory_manager.shutdown()

    def test_cross_memory_storage(self):
        """Test storing across different memory types."""
        # Create comprehensive memory from experience
        experience = {
            "context": {"market": "bull", "signal": "buy"},
            "action": "buy_btc",
            "outcome": {"profit": 0.05, "success": True},
            "importance": 0.8,
            "summary": "Successful BTC purchase in bull market"
        }

        memory_ids = self.memory_manager.create_comprehensive_memory(experience)

        # Should create entries in multiple memory types
        assert "episodic" in memory_ids
        assert "working" in memory_ids

    def test_memory_search_integration(self):
        """Test integrated memory search."""
        # Store memories
        experience = {
            "context": {"strategy": "momentum"},
            "action": "buy",
            "outcome": {"success": True}
        }
        self.memory_manager.create_comprehensive_memory(experience)

        # Search across memory types
        results = self.memory_manager.search_memories("momentum", limit=5)
        assert isinstance(results, dict)

        # Should have results from at least one memory type
        total_results = sum(len(result_list) for result_list in results.values())
        assert total_results > 0

    def test_memory_consolidation(self):
        """Test memory consolidation process."""
        # Create multiple experiences
        for i in range(5):
            experience = {
                "context": {"trade": f"trade_{i}"},
                "action": f"action_{i}",
                "outcome": {"profit": 0.01 * i}
            }
            self.memory_manager.create_comprehensive_memory(experience)

        # Perform consolidation
        stats = self.memory_manager.consolidate_memories(timeframe=timedelta(hours=1))
        assert isinstance(stats, dict)
        assert "episodes_consolidated" in stats

    def test_memory_optimization(self):
        """Test memory optimization."""
        # Fill memory with entries
        for i in range(50):
            experience = {
                "context": {"test": f"test_{i}"},
                "action": "test_action",
                "importance": 0.1 if i % 10 == 0 else 0.05  # Some low importance
            }
            self.memory_manager.create_comprehensive_memory(experience)

        # Perform optimization
        results = self.memory_manager.optimize_memory_usage()
        assert isinstance(results, dict)

    def test_contextual_memory_retrieval(self):
        """Test contextual memory retrieval."""
        # Store context-specific memories
        context = {"market_state": "volatile", "strategy": "conservative"}

        experience = {
            "context": context,
            "action": "reduce_position",
            "outcome": {"success": True}
        }
        self.memory_manager.create_comprehensive_memory(experience)

        # Retrieve contextual memories
        contextual = self.memory_manager.get_contextual_memories(context, limit_per_type=3)
        assert isinstance(contextual, dict)

    def test_system_status(self):
        """Test system status reporting."""
        status = self.memory_manager.get_system_status()
        assert isinstance(status, dict)
        assert "performance_stats" in status
        assert "memory_systems" in status


if __name__ == "__main__":
    pytest.main([__file__])