"""
SEMANTIC MEMORY SYSTEM
NetworkX-based knowledge graphs with concept relationships
"""

import networkx as nx
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import logging


class SemanticMemory:
    """Knowledge graph-based semantic memory for trading concepts"""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize semantic memory

        Args:
            storage_path: Path for graph storage
        """
        self.storage_path = storage_path or Path('data/memory/semantic_graph.pkl')
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        # Knowledge graph
        self.graph = nx.DiGraph()

        # Concept categories
        self.concept_types = {
            'market_regime', 'trading_strategy', 'risk_factor',
            'technical_indicator', 'fundamental_factor', 'correlation',
            'pattern', 'anomaly', 'event', 'outcome'
        }

        # Load existing graph
        self._load_graph()

    def add_concept(self, concept_id: str, concept_type: str,
                   attributes: Dict[str, Any]) -> bool:
        """Add a concept to semantic memory

        Args:
            concept_id: Unique concept identifier
            concept_type: Type of concept
            attributes: Concept attributes

        Returns:
            True if added successfully
        """
        with self._lock:
            if concept_type not in self.concept_types:
                logging.warning(f"Unknown concept type: {concept_type}")

            # Add node with attributes
            self.graph.add_node(concept_id, **{
                'type': concept_type,
                'created_at': datetime.now().isoformat(),
                **attributes
            })

            return True

    def add_relationship(self, source_concept: str, target_concept: str,
                        relationship_type: str, strength: float = 1.0,
                        attributes: Optional[Dict[str, Any]] = None) -> bool:
        """Add relationship between concepts

        Args:
            source_concept: Source concept ID
            target_concept: Target concept ID
            relationship_type: Type of relationship
            strength: Relationship strength (0.0 to 1.0)
            attributes: Additional relationship attributes

        Returns:
            True if added successfully
        """
        with self._lock:
            # Ensure both concepts exist
            if source_concept not in self.graph:
                self.add_concept(source_concept, 'unknown', {})
            if target_concept not in self.graph:
                self.add_concept(target_concept, 'unknown', {})

            # Add edge with attributes
            self.graph.add_edge(source_concept, target_concept, **{
                'type': relationship_type,
                'strength': max(0.0, min(1.0, strength)),
                'created_at': datetime.now().isoformat(),
                **(attributes or {})
            })

            return True

    def get_related_concepts(self, concept_id: str, relationship_types: Optional[List[str]] = None,
                           min_strength: float = 0.1, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get concepts related to a given concept

        Args:
            concept_id: Source concept ID
            relationship_types: Filter by relationship types
            min_strength: Minimum relationship strength
            max_depth: Maximum traversal depth

        Returns:
            List of related concepts with relationship info
        """
        with self._lock:
            if concept_id not in self.graph:
                return []

            related = []
            visited = set()

            def traverse(current_id: str, depth: int):
                if depth > max_depth or current_id in visited:
                    return

                visited.add(current_id)

                # Get outgoing edges
                for neighbor in self.graph.successors(current_id):
                    edge_data = self.graph[current_id][neighbor]

                    # Filter by relationship type
                    if relationship_types and edge_data.get('type') not in relationship_types:
                        continue

                    # Filter by strength
                    if edge_data.get('strength', 0.0) < min_strength:
                        continue

                    # Add to results
                    related.append({
                        'concept_id': neighbor,
                        'concept_type': self.graph.nodes[neighbor].get('type', 'unknown'),
                        'relationship_type': edge_data.get('type'),
                        'strength': edge_data.get('strength', 0.0),
                        'depth': depth + 1,
                        'attributes': dict(self.graph.nodes[neighbor])
                    })

                    # Recursive traversal
                    if depth < max_depth:
                        traverse(neighbor, depth + 1)

            traverse(concept_id, 0)
            return related

    def find_patterns(self, concept_type: str, min_connections: int = 3) -> List[Dict[str, Any]]:
        """Find concepts with many connections (hubs)

        Args:
            concept_type: Filter by concept type
            min_connections: Minimum number of connections

        Returns:
            List of hub concepts
        """
        with self._lock:
            hubs = []

            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]

                if concept_type and node_data.get('type') != concept_type:
                    continue

                # Count connections
                in_degree = self.graph.in_degree(node)
                out_degree = self.graph.out_degree(node)
                total_degree = in_degree + out_degree

                if total_degree >= min_connections:
                    hubs.append({
                        'concept_id': node,
                        'concept_type': node_data.get('type'),
                        'total_connections': total_degree,
                        'incoming': in_degree,
                        'outgoing': out_degree,
                        'attributes': dict(node_data)
                    })

            # Sort by total connections
            hubs.sort(key=lambda x: x['total_connections'], reverse=True)
            return hubs

    def get_concept_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """Find clusters of strongly connected concepts

        Args:
            min_cluster_size: Minimum cluster size

        Returns:
            List of concept clusters
        """
        with self._lock:
            # Convert to undirected for clustering
            undirected = self.graph.to_undirected()

            # Find connected components
            components = list(nx.connected_components(undirected))

            # Filter by size
            clusters = [list(component) for component in components
                       if len(component) >= min_cluster_size]

            return clusters

    def strengthen_relationship(self, source_concept: str, target_concept: str,
                              delta: float = 0.1) -> bool:
        """Strengthen a relationship based on evidence

        Args:
            source_concept: Source concept ID
            target_concept: Target concept ID
            delta: Strength increase

        Returns:
            True if strengthened
        """
        with self._lock:
            if not self.graph.has_edge(source_concept, target_concept):
                return False

            edge_data = self.graph[source_concept][target_concept]
            current_strength = edge_data.get('strength', 0.0)
            new_strength = min(1.0, current_strength + delta)

            self.graph[source_concept][target_concept]['strength'] = new_strength
            self.graph[source_concept][target_concept]['last_updated'] = datetime.now().isoformat()

            return True

    def weaken_relationship(self, source_concept: str, target_concept: str,
                          delta: float = 0.1) -> bool:
        """Weaken a relationship based on contradictory evidence

        Args:
            source_concept: Source concept ID
            target_concept: Target concept ID
            delta: Strength decrease

        Returns:
            True if weakened
        """
        with self._lock:
            if not self.graph.has_edge(source_concept, target_concept):
                return False

            edge_data = self.graph[source_concept][target_concept]
            current_strength = edge_data.get('strength', 0.0)
            new_strength = max(0.0, current_strength - delta)

            if new_strength < 0.05:  # Remove very weak relationships
                self.graph.remove_edge(source_concept, target_concept)
            else:
                self.graph[source_concept][target_concept]['strength'] = new_strength
                self.graph[source_concept][target_concept]['last_updated'] = datetime.now().isoformat()

            return True

    def get_concept_path(self, source_concept: str, target_concept: str) -> Optional[List[str]]:
        """Find shortest path between concepts

        Args:
            source_concept: Source concept ID
            target_concept: Target concept ID

        Returns:
            List of concepts in path or None
        """
        with self._lock:
            try:
                return nx.shortest_path(self.graph, source_concept, target_concept)
            except nx.NetworkXNoPath:
                return None

    def analyze_concept_importance(self, concept_id: str) -> Dict[str, float]:
        """Analyze the importance of a concept in the knowledge graph

        Args:
            concept_id: Concept ID

        Returns:
            Importance metrics
        """
        with self._lock:
            if concept_id not in self.graph:
                return {}

            # Calculate various centrality measures
            try:
                betweenness = nx.betweenness_centrality(self.graph).get(concept_id, 0.0)
                closeness = nx.closeness_centrality(self.graph).get(concept_id, 0.0)
                pagerank = nx.pagerank(self.graph).get(concept_id, 0.0)

                in_degree = self.graph.in_degree(concept_id)
                out_degree = self.graph.out_degree(concept_id)

                return {
                    'betweenness_centrality': betweenness,
                    'closeness_centrality': closeness,
                    'pagerank': pagerank,
                    'in_degree': in_degree,
                    'out_degree': out_degree,
                    'total_degree': in_degree + out_degree
                }

            except Exception as e:
                logging.error(f"Error calculating centrality for {concept_id}: {e}")
                return {}

    def prune_weak_relationships(self, min_strength: float = 0.1) -> int:
        """Remove weak relationships

        Args:
            min_strength: Minimum strength threshold

        Returns:
            Number of relationships removed
        """
        with self._lock:
            edges_to_remove = []

            for source, target, data in self.graph.edges(data=True):
                if data.get('strength', 0.0) < min_strength:
                    edges_to_remove.append((source, target))

            for edge in edges_to_remove:
                self.graph.remove_edge(*edge)

            logging.info(f"Pruned {len(edges_to_remove)} weak relationships")
            return len(edges_to_remove)

    def save_graph(self):
        """Save graph to disk"""
        with self._lock:
            try:
                with open(self.storage_path, 'wb') as f:
                    pickle.dump(self.graph, f)
                logging.info(f"Semantic memory saved to {self.storage_path}")
                return True
            except Exception as e:
                logging.error(f"Failed to save semantic memory: {e}")
                return False

    def _load_graph(self):
        """Load graph from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'rb') as f:
                    self.graph = pickle.load(f)
                logging.info(f"Semantic memory loaded from {self.storage_path}")
        except Exception as e:
            logging.error(f"Failed to load semantic memory: {e}")
            self.graph = nx.DiGraph()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self._lock:
            stats = {
                'total_concepts': self.graph.number_of_nodes(),
                'total_relationships': self.graph.number_of_edges(),
                'concept_types': {}
            }

            # Count by type
            for node in self.graph.nodes():
                concept_type = self.graph.nodes[node].get('type', 'unknown')
                stats['concept_types'][concept_type] = stats['concept_types'].get(concept_type, 0) + 1

            return stats

    def clear(self):
        """Clear all concepts and relationships"""
        with self._lock:
            self.graph.clear()