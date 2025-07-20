"""Storage backend for metrics persistence"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import threading

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetricsStorage:
    """SQLite-based storage for historical metrics"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Path to SQLite database file. If None, uses in-memory database.
        """
        self.db_path = db_path or ":memory:"
        self._lock = threading.Lock()
        self._init_db()
        
        logger.info(f"Initialized MetricsStorage (db={self.db_path})")
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            # Analysis metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_metrics (
                    analysis_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    methods TEXT NOT NULL,  -- JSON array
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_ms REAL,
                    input_length INTEGER,
                    attention_entropy REAL,
                    anomaly_score REAL,
                    failure_probability REAL,
                    error TEXT,
                    metadata TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Aggregated metrics table (for time series)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aggregated_metrics (
                    timestamp REAL PRIMARY KEY,
                    interval_minutes INTEGER NOT NULL,
                    total_analyses INTEGER,
                    total_errors INTEGER,
                    avg_duration_ms REAL,
                    avg_entropy REAL,
                    anomaly_count INTEGER,
                    high_risk_count INTEGER,
                    metadata TEXT  -- JSON
                )
            """)
            
            # Alert history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT,
                    metrics TEXT,  -- JSON
                    metadata TEXT  -- JSON
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_time ON analysis_metrics(start_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_model ON analysis_metrics(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alert_time ON alert_history(timestamp)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper locking"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def store_analysis_metrics(self, metrics: Dict[str, Any]):
        """Store analysis metrics to database"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO analysis_metrics (
                        analysis_id, model_name, methods, start_time, end_time,
                        duration_ms, input_length, attention_entropy, anomaly_score,
                        failure_probability, error, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.get("analysis_id"),
                    metrics.get("model_name"),
                    json.dumps(metrics.get("methods", [])),
                    metrics.get("start_time"),
                    metrics.get("end_time"),
                    metrics.get("duration_ms"),
                    metrics.get("input_length", 0),
                    metrics.get("attention_entropy"),
                    metrics.get("anomaly_score"),
                    metrics.get("failure_probability"),
                    metrics.get("error"),
                    json.dumps(metrics.get("metadata", {}))
                ))
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to store analysis metrics: {e}")
    
    def store_alert(self, alert: Dict[str, Any]):
        """Store alert to history"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO alert_history (
                        timestamp, rule_name, severity, message, metrics, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert.get("timestamp", time.time()),
                    alert.get("rule_name"),
                    alert.get("severity"),
                    alert.get("message"),
                    json.dumps(alert.get("metrics", {})),
                    json.dumps(alert.get("metadata", {}))
                ))
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to store alert: {e}")
    
    def aggregate_metrics(self, interval_minutes: int = 5):
        """Aggregate metrics over time intervals"""
        current_time = time.time()
        interval_start = current_time - (current_time % (interval_minutes * 60))
        
        with self._get_connection() as conn:
            # Get metrics for current interval
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_analyses,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as total_errors,
                    AVG(duration_ms) as avg_duration_ms,
                    AVG(attention_entropy) as avg_entropy,
                    SUM(CASE WHEN anomaly_score > 0.5 THEN 1 ELSE 0 END) as anomaly_count,
                    SUM(CASE WHEN failure_probability > 0.7 THEN 1 ELSE 0 END) as high_risk_count
                FROM analysis_metrics
                WHERE start_time >= ? AND start_time < ?
            """, (interval_start - (interval_minutes * 60), interval_start))
            
            row = cursor.fetchone()
            if row and row["total_analyses"] > 0:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO aggregated_metrics (
                            timestamp, interval_minutes, total_analyses, total_errors,
                            avg_duration_ms, avg_entropy, anomaly_count, high_risk_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        interval_start,
                        interval_minutes,
                        row["total_analyses"],
                        row["total_errors"] or 0,
                        row["avg_duration_ms"],
                        row["avg_entropy"],
                        row["anomaly_count"] or 0,
                        row["high_risk_count"] or 0
                    ))
                    conn.commit()
                except Exception as e:
                    logger.error(f"Failed to store aggregated metrics: {e}")
    
    def get_time_series(self, 
                       metric: str,
                       hours: int = 24,
                       interval_minutes: int = 5) -> List[Dict[str, Any]]:
        """Get time series data for a metric"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT timestamp, {} as value
                FROM aggregated_metrics
                WHERE timestamp >= ? AND interval_minutes = ?
                ORDER BY timestamp
            """.format(metric), (cutoff_time, interval_minutes))
            
            return [
                {"timestamp": row["timestamp"], "value": row["value"]}
                for row in cursor
            ]
    
    def get_model_stats(self, hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """Get statistics by model"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    model_name,
                    COUNT(*) as count,
                    AVG(duration_ms) as avg_duration,
                    AVG(attention_entropy) as avg_entropy,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
                FROM analysis_metrics
                WHERE start_time >= ?
                GROUP BY model_name
            """, (cutoff_time,))
            
            return {
                row["model_name"]: {
                    "count": row["count"],
                    "avg_duration": row["avg_duration"],
                    "avg_entropy": row["avg_entropy"],
                    "errors": row["errors"]
                }
                for row in cursor
            }
    
    def get_alert_history(self, 
                         hours: int = 24,
                         severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alert history"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._get_connection() as conn:
            if severity:
                cursor = conn.execute("""
                    SELECT * FROM alert_history
                    WHERE timestamp >= ? AND severity = ?
                    ORDER BY timestamp DESC
                """, (cutoff_time, severity))
            else:
                cursor = conn.execute("""
                    SELECT * FROM alert_history
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff_time,))
            
            alerts = []
            for row in cursor:
                alert = dict(row)
                alert["metrics"] = json.loads(alert["metrics"]) if alert["metrics"] else {}
                alert["metadata"] = json.loads(alert["metadata"]) if alert["metadata"] else {}
                alerts.append(alert)
            
            return alerts
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from the database"""
        cutoff_time = time.time() - (days_to_keep * 86400)
        
        with self._get_connection() as conn:
            # Clean up old analysis metrics
            conn.execute("DELETE FROM analysis_metrics WHERE start_time < ?", (cutoff_time,))
            
            # Clean up old aggregated metrics
            conn.execute("DELETE FROM aggregated_metrics WHERE timestamp < ?", (cutoff_time,))
            
            # Clean up old alerts
            conn.execute("DELETE FROM alert_history WHERE timestamp < ?", (cutoff_time,))
            
            conn.commit()
            
            # Vacuum to reclaim space
            conn.execute("VACUUM")
            
        logger.info(f"Cleaned up data older than {days_to_keep} days")