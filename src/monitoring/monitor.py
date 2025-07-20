"""Real-time analysis monitoring and metrics collection"""

import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import threading
import json

from ..utils.logger import get_logger
from .failure_patterns import FailurePatternDetector

logger = get_logger(__name__)


@dataclass
class AnalysisMetrics:
    """Metrics for a single analysis run"""
    analysis_id: str
    model_name: str
    methods: List[str]
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    input_length: int = 0
    attention_entropy: Optional[float] = None
    anomaly_score: Optional[float] = None
    failure_probability: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self):
        """Mark analysis as complete and calculate duration"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000


class MetricsCollector:
    """Collects and aggregates metrics across analyses"""
    
    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of recent analyses to keep in memory
        """
        self.window_size = window_size
        self.metrics_window = deque(maxlen=window_size)
        self.model_metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.method_metrics = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()
        
        # Aggregated metrics
        self.total_analyses = 0
        self.total_errors = 0
        self.anomalies_detected = 0
        self.high_risk_predictions = 0
        
    def add_metrics(self, metrics: AnalysisMetrics):
        """Add new metrics to the collector"""
        with self._lock:
            self.metrics_window.append(metrics)
            self.model_metrics[metrics.model_name].append(metrics)
            
            for method in metrics.methods:
                self.method_metrics[method].append(metrics)
            
            # Update counters
            self.total_analyses += 1
            if metrics.error:
                self.total_errors += 1
            if metrics.anomaly_score and metrics.anomaly_score > 0.5:
                self.anomalies_detected += 1
            if metrics.failure_probability and metrics.failure_probability > 0.7:
                self.high_risk_predictions += 1
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for monitoring"""
        with self._lock:
            if not self.metrics_window:
                return {}
            
            recent_durations = [m.duration_ms for m in self.metrics_window 
                               if m.duration_ms is not None]
            recent_entropies = [m.attention_entropy for m in self.metrics_window 
                               if m.attention_entropy is not None]
            
            return {
                "total_analyses": self.total_analyses,
                "total_errors": self.total_errors,
                "error_rate": self.total_errors / max(1, self.total_analyses),
                "anomalies_detected": self.anomalies_detected,
                "high_risk_predictions": self.high_risk_predictions,
                "recent_stats": {
                    "count": len(self.metrics_window),
                    "avg_duration_ms": sum(recent_durations) / len(recent_durations) if recent_durations else 0,
                    "max_duration_ms": max(recent_durations) if recent_durations else 0,
                    "avg_entropy": sum(recent_entropies) / len(recent_entropies) if recent_entropies else 0,
                },
                "models": {
                    model: len(metrics) for model, metrics in self.model_metrics.items()
                },
                "methods": {
                    method: len(metrics) for method, metrics in self.method_metrics.items()
                }
            }
    
    def get_time_series(self, metric: str, window_minutes: int = 5) -> List[Dict[str, Any]]:
        """Get time series data for a specific metric"""
        with self._lock:
            cutoff_time = time.time() - (window_minutes * 60)
            series = []
            
            for m in self.metrics_window:
                if m.start_time > cutoff_time:
                    value = getattr(m, metric, None)
                    if value is not None:
                        series.append({
                            "timestamp": m.start_time,
                            "value": value,
                            "model": m.model_name
                        })
            
            return series


class AnalysisMonitor:
    """Main monitoring interface for the analysis system"""
    
    def __init__(self, 
                 enable_alerts: bool = True,
                 metrics_window_size: int = 1000,
                 enable_failure_detection: bool = True):
        """
        Args:
            enable_alerts: Whether to enable anomaly alerts
            metrics_window_size: Number of recent analyses to track
            enable_failure_detection: Whether to enable failure pattern detection
        """
        self.collector = MetricsCollector(window_size=metrics_window_size)
        self.enable_alerts = enable_alerts
        self.enable_failure_detection = enable_failure_detection
        self.alert_callbacks = []
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Initialize failure pattern detector
        self.failure_detector = FailurePatternDetector() if enable_failure_detection else None
        
        logger.info(f"Initialized AnalysisMonitor (alerts={'enabled' if enable_alerts else 'disabled'}, "
                   f"failure_detection={'enabled' if enable_failure_detection else 'disabled'})")
    
    def start_analysis(self, 
                      analysis_id: str,
                      model_name: str,
                      methods: List[str],
                      input_text: str) -> AnalysisMetrics:
        """Start tracking a new analysis"""
        metrics = AnalysisMetrics(
            analysis_id=analysis_id,
            model_name=model_name,
            methods=methods,
            start_time=time.time(),
            input_length=len(input_text)
        )
        
        logger.debug(f"Started monitoring analysis {analysis_id}")
        return metrics
    
    def complete_analysis(self, 
                         metrics: AnalysisMetrics,
                         results: Optional[Dict[str, Any]] = None,
                         error: Optional[str] = None,
                         input_text: Optional[str] = None):
        """Complete analysis tracking and extract metrics from results"""
        metrics.complete()
        metrics.error = error
        
        if results and not error:
            # Extract key metrics from results
            if "attention" in results:
                entropy = results["attention"].get("entropy", {})
                if "mean_entropy_per_layer" in entropy:
                    entropies = entropy["mean_entropy_per_layer"]
                    if entropies is not None and len(entropies) > 0:
                        # Convert tensors to floats if needed
                        if hasattr(entropies[0], 'item'):
                            entropies = [e.item() for e in entropies]
                        metrics.attention_entropy = sum(entropies) / len(entropies)
                    else:
                        metrics.attention_entropy = 0
            
            if "anomaly_score" in results:
                metrics.anomaly_score = results["anomaly_score"]
            
            if "failure_probability" in results:
                metrics.failure_probability = results["failure_probability"]
            
            # Run failure pattern detection
            if self.failure_detector and input_text:
                failure_results = self.failure_detector.detect_failures(input_text, results)
                metrics.metadata["failure_patterns"] = failure_results
                
                # Add to metrics if significant failure detected
                if failure_results["is_failure"]:
                    metrics.metadata["detected_failures"] = [
                        p["name"] for p in failure_results["detected_patterns"]
                    ]
                    logger.warning(f"Failure patterns detected in {metrics.analysis_id}: "
                                 f"{metrics.metadata['detected_failures']}")
        
        self.collector.add_metrics(metrics)
        
        # Check for alerts
        if self.enable_alerts:
            self._check_alerts(metrics)
        
        logger.debug(f"Completed monitoring analysis {metrics.analysis_id} "
                    f"(duration={metrics.duration_ms:.1f}ms)")
    
    def add_alert_callback(self, callback):
        """Add a callback for when alerts are triggered"""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self, metrics: AnalysisMetrics):
        """Check if metrics trigger any alerts"""
        alerts = []
        
        # High failure probability alert
        if metrics.failure_probability and metrics.failure_probability > 0.8:
            alerts.append({
                "type": "high_failure_risk",
                "severity": "warning",
                "message": f"High failure probability detected: {metrics.failure_probability:.1%}",
                "analysis_id": metrics.analysis_id,
                "value": metrics.failure_probability
            })
        
        # Anomaly detection alert
        if metrics.anomaly_score and metrics.anomaly_score > 0.7:
            alerts.append({
                "type": "anomaly_detected", 
                "severity": "warning",
                "message": f"Anomaly detected with score: {metrics.anomaly_score:.2f}",
                "analysis_id": metrics.analysis_id,
                "value": metrics.anomaly_score
            })
        
        # Performance degradation alert
        if metrics.duration_ms and metrics.duration_ms > 5000:
            alerts.append({
                "type": "slow_analysis",
                "severity": "info",
                "message": f"Analysis took {metrics.duration_ms:.0f}ms",
                "analysis_id": metrics.analysis_id,
                "value": metrics.duration_ms
            })
        
        # Trigger callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for monitoring dashboards"""
        summary = self.collector.get_summary_stats()
        
        return {
            "summary": summary,
            "time_series": {
                "duration": self.collector.get_time_series("duration_ms", window_minutes=5),
                "entropy": self.collector.get_time_series("attention_entropy", window_minutes=5),
                "anomaly_score": self.collector.get_time_series("anomaly_score", window_minutes=5),
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        stats = self.collector.get_summary_stats()
        
        metrics = []
        
        # Counter metrics
        metrics.append(f'# HELP llm_insight_analyses_total Total number of analyses')
        metrics.append(f'# TYPE llm_insight_analyses_total counter')
        metrics.append(f'llm_insight_analyses_total {stats.get("total_analyses", 0)}')
        
        metrics.append(f'# HELP llm_insight_errors_total Total number of analysis errors')
        metrics.append(f'# TYPE llm_insight_errors_total counter')
        metrics.append(f'llm_insight_errors_total {stats.get("total_errors", 0)}')
        
        metrics.append(f'# HELP llm_insight_anomalies_total Total anomalies detected')
        metrics.append(f'# TYPE llm_insight_anomalies_total counter')
        metrics.append(f'llm_insight_anomalies_total {stats.get("anomalies_detected", 0)}')
        
        # Gauge metrics
        recent = stats.get("recent_stats", {})
        metrics.append(f'# HELP llm_insight_avg_duration_ms Average analysis duration in milliseconds')
        metrics.append(f'# TYPE llm_insight_avg_duration_ms gauge')
        metrics.append(f'llm_insight_avg_duration_ms {recent.get("avg_duration_ms", 0):.2f}')
        
        metrics.append(f'# HELP llm_insight_avg_entropy Average attention entropy')
        metrics.append(f'# TYPE llm_insight_avg_entropy gauge')
        metrics.append(f'llm_insight_avg_entropy {recent.get("avg_entropy", 0):.3f}')
        
        # Model-specific metrics
        for model, count in stats.get("models", {}).items():
            safe_model = model.replace("-", "_").replace("/", "_")
            metrics.append(f'llm_insight_model_analyses_total{{model="{model}"}} {count}')
        
        return '\n'.join(metrics)