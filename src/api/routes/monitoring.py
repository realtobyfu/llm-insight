"""Monitoring and metrics API routes"""

from typing import Dict, Optional
from fastapi import APIRouter, Depends, Query, Response
from datetime import datetime, timedelta

from ...monitoring import AnalysisMonitor, AlertManager, MetricsStorage
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# Global instances (will be set by API)
monitor = None
alert_manager = None
storage = None


def set_monitor_instances(analyzer_monitor, analyzer_storage=None):
    """Set monitor instances from the analyzer"""
    global monitor, alert_manager, storage
    monitor = analyzer_monitor
    if hasattr(monitor, 'alert_manager'):
        alert_manager = monitor.alert_manager
    else:
        alert_manager = AlertManager()
    storage = analyzer_storage or MetricsStorage()


def get_monitor() -> AnalysisMonitor:
    """Dependency to get monitor instance"""
    if monitor is None:
        # Create a default instance if not set
        return AnalysisMonitor()
    return monitor


def get_alert_manager() -> AlertManager:
    """Dependency to get alert manager instance"""
    if alert_manager is None:
        return AlertManager()
    return alert_manager


def get_storage() -> MetricsStorage:
    """Dependency to get storage instance"""
    if storage is None:
        return MetricsStorage()
    return storage


@router.get("/metrics")
async def get_metrics(
    monitor: AnalysisMonitor = Depends(get_monitor)
) -> Dict:
    """Get current monitoring metrics"""
    return monitor.get_metrics()


@router.get("/metrics/prometheus")
async def prometheus_metrics(
    monitor: AnalysisMonitor = Depends(get_monitor)
) -> Response:
    """Export metrics in Prometheus format"""
    metrics = monitor.export_prometheus_metrics()
    return Response(
        content=metrics,
        media_type="text/plain; version=0.0.4",
        headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"}
    )


@router.get("/time-series/{metric}")
async def get_time_series(
    metric: str,
    hours: int = Query(24, ge=1, le=168),  # Max 1 week
    interval: int = Query(5, ge=1, le=60),  # Minutes
    storage: MetricsStorage = Depends(get_storage)
) -> Dict:
    """Get time series data for a specific metric"""
    valid_metrics = [
        "total_analyses", "total_errors", "avg_duration_ms", 
        "avg_entropy", "anomaly_count", "high_risk_count"
    ]
    
    if metric not in valid_metrics:
        return {
            "error": f"Invalid metric. Valid metrics: {', '.join(valid_metrics)}"
        }
    
    series = storage.get_time_series(metric, hours, interval)
    
    return {
        "metric": metric,
        "hours": hours,
        "interval_minutes": interval,
        "data": series
    }


@router.get("/models/stats")
async def get_model_stats(
    hours: int = Query(24, ge=1, le=168),
    storage: MetricsStorage = Depends(get_storage)
) -> Dict:
    """Get statistics grouped by model"""
    return storage.get_model_stats(hours)


@router.get("/alerts")
async def get_alerts(
    hours: int = Query(24, ge=1, le=168),
    severity: Optional[str] = Query(None, regex="^(info|warning|error|critical)$"),
    storage: MetricsStorage = Depends(get_storage)
) -> Dict:
    """Get alert history"""
    alerts = storage.get_alert_history(hours, severity)
    
    return {
        "hours": hours,
        "severity": severity,
        "count": len(alerts),
        "alerts": alerts
    }


@router.get("/alerts/stats")
async def get_alert_stats(
    alert_manager: AlertManager = Depends(get_alert_manager)
) -> Dict:
    """Get alert statistics"""
    return alert_manager.get_alert_stats()


@router.post("/alerts/test")
async def test_alert(
    rule_name: str,
    alert_manager: AlertManager = Depends(get_alert_manager)
) -> Dict:
    """Test an alert rule"""
    # Find the rule
    rule = next((r for r in alert_manager.rules if r.name == rule_name), None)
    if not rule:
        return {"error": f"Rule '{rule_name}' not found"}
    
    # Create test metrics that trigger the rule
    test_metrics = {
        "error_rate": 0.5,
        "avg_duration_ms": 10000,
        "anomaly_rate": 0.5
    }
    
    # Force trigger
    rule.last_triggered = 0  # Reset cooldown
    if rule.should_trigger(test_metrics):
        alert = rule.trigger(test_metrics)
        alert_manager._send_alert(alert)
        return {
            "success": True,
            "alert": alert
        }
    else:
        return {
            "success": False,
            "message": "Rule did not trigger with test metrics"
        }


@router.get("/health")
async def health_check(
    monitor: AnalysisMonitor = Depends(get_monitor),
    storage: MetricsStorage = Depends(get_storage)
) -> Dict:
    """Health check endpoint with monitoring status"""
    try:
        # Get recent stats
        stats = monitor.get_metrics()
        summary = stats.get("summary", {})
        
        # Check if system is healthy
        error_rate = summary.get("error_rate", 0)
        recent = summary.get("recent_stats", {})
        avg_duration = recent.get("avg_duration_ms", 0)
        
        is_healthy = error_rate < 0.2 and avg_duration < 10000
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "error_rate": round(error_rate, 3),
                "avg_duration_ms": round(avg_duration, 1),
                "total_analyses": summary.get("total_analyses", 0)
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }