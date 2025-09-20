"""
VECTOR MEMORY SYSTEM
High-performance semantic memory using FAISS for similarity search
"""

import numpy as np
import faiss
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging


class VectorMemory:
    """High-performance vector memory with FAISS similarity search"""

    def __init__(self, dimension: int = 128, index_type: str = 'IVF',
                 storage_path: Optional[Path] = None):
        """Initialize vector memory

        Args:
            dimension: Vector dimension
            index_type: FAISS index type ('IVF' or 'Flat')
            storage_path: Path for persistence
        """
        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = storage_path or Path('data/memory/vectors')
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        # Initialize FAISS index
        if index_type == 'IVF':
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.index.nprobe = 10  # Number of clusters to search
        else:
            # Flat index for smaller datasets
            self.index = faiss.IndexFlatL2(dimension)

        # Metadata storage
        self.metadata = {}  # id -> metadata dict
        self.id_counter = 0

        # Load existing data
        self._load_index()

    def add_vector(self, vector: np.ndarray, metadata: Dict[str, Any]) -> int:
        """Add vector with metadata

        Args:
            vector: Vector to add
            metadata: Associated metadata

        Returns:
            Vector ID
        """
        with self._lock:
            # Normalize vector
            vector = vector.astype(np.float32).reshape(1, -1)
            if vector.shape[1] != self.dimension:
                raise ValueError(f"Vector dimension {vector.shape[1]} != {self.dimension}")

            # Add to index
            if self.index_type == 'IVF' and not self.index.is_trained:
                # Train IVF index if needed
                if self.index.ntotal == 0:
                    # Add some random vectors for training
                    training_data = np.random.random((1000, self.dimension)).astype(np.float32)
                    self.index.train(training_data)

            self.index.add(vector)

            # Store metadata
            vector_id = self.id_counter
            self.metadata[vector_id] = {
                'timestamp': datetime.now().isoformat(),
                **metadata
            }
            self.id_counter += 1

            return vector_id

    def search_similar(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Search for similar vectors

        Args:
            query_vector: Query vector
            k: Number of results

        Returns:
            List of (id, distance, metadata) tuples
        """
        with self._lock:
            if self.index.ntotal == 0:
                return []

            # Normalize query
            query_vector = query_vector.astype(np.float32).reshape(1, -1)

            # Search
            distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))

            # Return results with metadata
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0:  # Valid index
                    results.append((idx, float(dist), self.metadata.get(idx, {})))

            return results

    def get_vector_by_id(self, vector_id: int) -> Optional[Tuple[np.ndarray, Dict]]:
        """Get vector and metadata by ID

        Args:
            vector_id: Vector ID

        Returns:
            (vector, metadata) or None
        """
        with self._lock:
            if vector_id not in self.metadata:
                return None

            # FAISS doesn't support direct vector retrieval by ID
            # This is a limitation - in production you'd store vectors separately
            return None, self.metadata[vector_id]

    def remove_vector(self, vector_id: int) -> bool:
        """Remove vector by ID

        Args:
            vector_id: Vector ID

        Returns:
            True if removed
        """
        with self._lock:
            if vector_id not in self.metadata:
                return False

            # FAISS doesn't support removal - would need reconstruction
            # For now, just remove metadata
            del self.metadata[vector_id]
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self._lock:
            return {
                'total_vectors': self.index.ntotal,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'metadata_count': len(self.metadata),
                'is_trained': getattr(self.index, 'is_trained', True)
            }

    def save_index(self):
        """Save index to disk"""
        with self._lock:
            try:
                # Save FAISS index
                index_path = self.storage_path / 'faiss_index.bin'
                faiss.write_index(self.index, str(index_path))

                # Save metadata
                metadata_path = self.storage_path / 'metadata.pkl'
                with open(metadata_path, 'wb') as f:
                    pickle.dump({
                        'metadata': self.metadata,
                        'id_counter': self.id_counter,
                        'dimension': self.dimension,
                        'index_type': self.index_type
                    }, f)

                logging.info(f"Vector memory saved to {self.storage_path}")
                return True

            except Exception as e:
                logging.error(f"Failed to save vector memory: {e}")
                return False

    def _load_index(self):
        """Load index from disk"""
        try:
            index_path = self.storage_path / 'faiss_index.bin'
            metadata_path = self.storage_path / 'metadata.pkl'

            if index_path.exists() and metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))

                # Load metadata
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data['metadata']
                    self.id_counter = data['id_counter']

                logging.info(f"Vector memory loaded from {self.storage_path}")

        except Exception as e:
            logging.error(f"Failed to load vector memory: {e}")
            # Continue with empty index

    def compress_memory(self, compression_ratio: float = 0.6) -> Dict[str, Any]:
        """Compress memory by removing less important vectors

        Args:
            compression_ratio: Target compression ratio

        Returns:
            Compression statistics
        """
        with self._lock:
            if self.index.ntotal == 0:
                return {'compressed': 0, 'remaining': 0}

            # Simple compression: remove oldest vectors
            # In production, use more sophisticated methods
            target_size = int(self.index.ntotal * compression_ratio)
            vectors_to_remove = self.index.ntotal - target_size

            if vectors_to_remove <= 0:
                return {'compressed': 0, 'remaining': self.index.ntotal}

            # Remove oldest entries from metadata
            sorted_ids = sorted(self.metadata.keys(),
                              key=lambda x: self.metadata[x].get('timestamp', ''))

            removed_count = 0
            for vector_id in sorted_ids[:vectors_to_remove]:
                if self.remove_vector(vector_id):
                    removed_count += 1

            return {
                'compressed': removed_count,
                'remaining': len(self.metadata),
                'compression_ratio': removed_count / self.index.ntotal if self.index.ntotal > 0 else 0
            }

    def clear(self):
        """Clear all vectors and metadata"""
        with self._lock:
            # Reset index
            if self.index_type == 'IVF':
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                self.index.nprobe = 10
            else:
                self.index = faiss.IndexFlatL2(self.dimension)

            # Clear metadata
            self.metadata.clear()
            self.id_counter = 0