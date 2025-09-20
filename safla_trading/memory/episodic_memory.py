"""
EPISODIC MEMORY SYSTEM
SQLite-based experience storage with temporal indexing
"""

import sqlite3
import threading
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging


class EpisodicMemory:
    """SQLite-based episodic memory for trading experiences"""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize episodic memory

        Args:
            storage_path: Path for SQLite database
        """
        self.storage_path = storage_path or Path('data/memory/episodic.db')
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database schema"""
        with self._lock:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            # Episodes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    episode_type TEXT NOT NULL,
                    context TEXT,
                    action TEXT,
                    outcome TEXT,
                    reward REAL,
                    confidence REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Temporal index for fast time-based queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_episodes_timestamp
                ON episodes (timestamp)
            ''')

            # Type index
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_episodes_type
                ON episodes (episode_type)
            ''')

            # Reward index for performance queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_episodes_reward
                ON episodes (reward)
            ''')

            conn.commit()
            conn.close()

    def add_episode(self, episode_type: str, context: Dict[str, Any],
                   action: Dict[str, Any], outcome: Dict[str, Any],
                   reward: float, confidence: float = 1.0,
                   metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add an episode to memory

        Args:
            episode_type: Type of episode (e.g., 'trade', 'prediction', 'market_event')
            context: Context/state when episode occurred
            action: Action taken
            outcome: Result of the action
            reward: Reward/performance score
            confidence: Confidence in the episode
            metadata: Additional metadata

        Returns:
            Episode ID
        """
        with self._lock:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            timestamp = datetime.now().timestamp()

            cursor.execute('''
                INSERT INTO episodes (timestamp, episode_type, context, action,
                                    outcome, reward, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                episode_type,
                json.dumps(context),
                json.dumps(action),
                json.dumps(outcome),
                reward,
                confidence,
                json.dumps(metadata or {})
            ))

            episode_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return episode_id

    def get_recent_episodes(self, episode_type: Optional[str] = None,
                           hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent episodes

        Args:
            episode_type: Filter by episode type
            hours: Hours to look back
            limit: Maximum episodes to return

        Returns:
            List of episodes
        """
        with self._lock:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            since_timestamp = (datetime.now() - timedelta(hours=hours)).timestamp()

            if episode_type:
                cursor.execute('''
                    SELECT * FROM episodes
                    WHERE timestamp >= ? AND episode_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (since_timestamp, episode_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM episodes
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (since_timestamp, limit))

            episodes = []
            for row in cursor.fetchall():
                episodes.append(self._row_to_episode(row))

            conn.close()
            return episodes

    def get_episodes_by_performance(self, min_reward: float = 0.0,
                                   episode_type: Optional[str] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get episodes by performance threshold

        Args:
            min_reward: Minimum reward threshold
            episode_type: Filter by episode type
            limit: Maximum episodes to return

        Returns:
            List of high-performing episodes
        """
        with self._lock:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            if episode_type:
                cursor.execute('''
                    SELECT * FROM episodes
                    WHERE reward >= ? AND episode_type = ?
                    ORDER BY reward DESC
                    LIMIT ?
                ''', (min_reward, episode_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM episodes
                    WHERE reward >= ?
                    ORDER BY reward DESC
                    LIMIT ?
                ''', (min_reward, limit))

            episodes = []
            for row in cursor.fetchall():
                episodes.append(self._row_to_episode(row))

            conn.close()
            return episodes

    def get_similar_contexts(self, context: Dict[str, Any],
                           episode_type: Optional[str] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Find episodes with similar contexts

        Args:
            context: Context to match against
            episode_type: Filter by episode type
            limit: Maximum episodes to return

        Returns:
            List of similar episodes
        """
        # Simple similarity based on common keys
        # In production, use vector similarity
        with self._lock:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            if episode_type:
                cursor.execute('''
                    SELECT * FROM episodes
                    WHERE episode_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (episode_type, limit * 5))  # Get more to filter
            else:
                cursor.execute('''
                    SELECT * FROM episodes
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit * 5,))

            episodes = []
            context_keys = set(context.keys())

            for row in cursor.fetchall():
                episode = self._row_to_episode(row)
                episode_context = episode['context']

                # Calculate simple similarity
                episode_keys = set(episode_context.keys())
                similarity = len(context_keys.intersection(episode_keys)) / len(context_keys.union(episode_keys))

                if similarity > 0.3:  # Threshold for similarity
                    episode['similarity'] = similarity
                    episodes.append(episode)

            # Sort by similarity and limit
            episodes.sort(key=lambda x: x['similarity'], reverse=True)
            conn.close()

            return episodes[:limit]

    def get_performance_stats(self, episode_type: Optional[str] = None,
                            hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics

        Args:
            episode_type: Filter by episode type
            hours: Hours to analyze

        Returns:
            Performance statistics
        """
        with self._lock:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            since_timestamp = (datetime.now() - timedelta(hours=hours)).timestamp()

            if episode_type:
                cursor.execute('''
                    SELECT COUNT(*), AVG(reward), MIN(reward), MAX(reward),
                           AVG(confidence)
                    FROM episodes
                    WHERE timestamp >= ? AND episode_type = ?
                ''', (since_timestamp, episode_type))
            else:
                cursor.execute('''
                    SELECT COUNT(*), AVG(reward), MIN(reward), MAX(reward),
                           AVG(confidence)
                    FROM episodes
                    WHERE timestamp >= ?
                ''', (since_timestamp,))

            row = cursor.fetchone()
            conn.close()

            if row and row[0] > 0:
                return {
                    'count': row[0],
                    'avg_reward': row[1] or 0.0,
                    'min_reward': row[2] or 0.0,
                    'max_reward': row[3] or 0.0,
                    'avg_confidence': row[4] or 0.0
                }
            else:
                return {
                    'count': 0,
                    'avg_reward': 0.0,
                    'min_reward': 0.0,
                    'max_reward': 0.0,
                    'avg_confidence': 0.0
                }

    def cleanup_old_episodes(self, days: int = 30) -> int:
        """Remove old episodes

        Args:
            days: Days to keep

        Returns:
            Number of episodes removed
        """
        with self._lock:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            cutoff_timestamp = (datetime.now() - timedelta(days=days)).timestamp()

            cursor.execute('''
                DELETE FROM episodes WHERE timestamp < ?
            ''', (cutoff_timestamp,))

            removed_count = cursor.rowcount
            conn.commit()
            conn.close()

            logging.info(f"Removed {removed_count} old episodes")
            return removed_count

    def _row_to_episode(self, row: Tuple) -> Dict[str, Any]:
        """Convert database row to episode dictionary"""
        return {
            'id': row[0],
            'timestamp': row[1],
            'episode_type': row[2],
            'context': json.loads(row[3]),
            'action': json.loads(row[4]),
            'outcome': json.loads(row[5]),
            'reward': row[6],
            'confidence': row[7],
            'metadata': json.loads(row[8]),
            'created_at': row[9]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self._lock:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM episodes')
            total_episodes = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT episode_type) FROM episodes')
            unique_types = cursor.fetchone()[0]

            cursor.execute('''
                SELECT episode_type, COUNT(*)
                FROM episodes
                GROUP BY episode_type
                ORDER BY COUNT(*) DESC
            ''')
            type_counts = dict(cursor.fetchall())

            conn.close()

            return {
                'total_episodes': total_episodes,
                'unique_types': unique_types,
                'type_distribution': type_counts
            }

    def clear(self):
        """Clear all episodes"""
        with self._lock:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()
            cursor.execute('DELETE FROM episodes')
            conn.commit()
            conn.close()