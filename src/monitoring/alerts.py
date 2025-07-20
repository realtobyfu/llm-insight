"""Alert management and notification system"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """Definition of an alert rule"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: int = 300  # Avoid alert spam
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.last_triggered = 0
        if self.metadata is None:
            self.metadata = {}
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        """Check if alert should trigger based on condition and cooldown"""
        if time.time() - self.last_triggered < self.cooldown_seconds:
            return False
        
        try:
            return self.condition(metrics)
        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.name}: {e}")
            return False
    
    def trigger(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger the alert and return alert data"""
        self.last_triggered = time.time()
        
        # Format message with metrics
        try:
            message = self.message_template.format(**metrics)
        except KeyError:
            message = self.message_template
        
        return {
            "rule_name": self.name,
            "severity": self.severity.value,
            "message": message,
            "timestamp": time.time(),
            "metrics": metrics,
            "metadata": self.metadata
        }


class AlertChannel(ABC):
    """Abstract base class for alert notification channels"""
    
    @abstractmethod
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert through this channel"""
        pass


class LogChannel(AlertChannel):
    """Simple logging channel for alerts"""
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Log alert to the system logger"""
        severity = alert.get("severity", "info")
        message = alert.get("message", "Alert triggered")
        
        if severity == "critical":
            logger.critical(f"ALERT: {message}")
        elif severity == "error":
            logger.error(f"ALERT: {message}")
        elif severity == "warning":
            logger.warning(f"ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")
        
        return True


class WebhookChannel(AlertChannel):
    """Send alerts to a webhook endpoint"""
    
    def __init__(self, webhook_url: str, timeout: int = 5):
        self.webhook_url = webhook_url
        self.timeout = timeout
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert to webhook"""
        try:
            response = requests.post(
                self.webhook_url,
                json=alert,
                timeout=self.timeout
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class EmailChannel(AlertChannel):
    """Send alerts via email"""
    
    def __init__(self, 
                 smtp_host: str,
                 smtp_port: int,
                 from_email: str,
                 to_emails: List[str],
                 smtp_user: Optional[str] = None,
                 smtp_password: Optional[str] = None,
                 use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_email = from_email
        self.to_emails = to_emails
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.use_tls = use_tls
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"LLM Insight Alert: {alert.get('rule_name', 'Unknown')}"
            
            body = f"""
Alert: {alert.get('message', 'Alert triggered')}
Severity: {alert.get('severity', 'info')}
Time: {time.ctime(alert.get('timestamp', time.time()))}

Metrics:
{json.dumps(alert.get('metrics', {}), indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class AlertManager:
    """Manages alert rules and notification channels"""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.channels: List[AlertChannel] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # Add default log channel
        self.add_channel(LogChannel())
        
        # Add default rules
        self._add_default_rules()
    
    def _add_default_rules(self):
        """Add default alert rules"""
        # High error rate
        self.add_rule(AlertRule(
            name="high_error_rate",
            condition=lambda m: m.get("error_rate", 0) > 0.1,
            severity=AlertSeverity.WARNING,
            message_template="High error rate detected: {error_rate:.1%}",
            cooldown_seconds=600
        ))
        
        # Slow analysis
        self.add_rule(AlertRule(
            name="slow_analysis",
            condition=lambda m: m.get("avg_duration_ms", 0) > 5000,
            severity=AlertSeverity.INFO,
            message_template="Average analysis duration high: {avg_duration_ms:.0f}ms",
            cooldown_seconds=300
        ))
        
        # Many anomalies
        self.add_rule(AlertRule(
            name="anomaly_spike",
            condition=lambda m: m.get("anomaly_rate", 0) > 0.3,
            severity=AlertSeverity.WARNING,
            message_template="High anomaly rate: {anomaly_rate:.1%}",
            cooldown_seconds=900
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule by name"""
        self.rules = [r for r in self.rules if r.name != rule_name]
    
    def add_channel(self, channel: AlertChannel):
        """Add a notification channel"""
        self.channels.append(channel)
        logger.info(f"Added alert channel: {channel.__class__.__name__}")
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all rules against current metrics"""
        # Add derived metrics
        summary = metrics.get("summary", {})
        if summary:
            total = summary.get("total_analyses", 1)
            metrics["error_rate"] = summary.get("total_errors", 0) / max(1, total)
            metrics["anomaly_rate"] = summary.get("anomalies_detected", 0) / max(1, total)
            
            recent = summary.get("recent_stats", {})
            metrics["avg_duration_ms"] = recent.get("avg_duration_ms", 0)
        
        # Check each rule
        for rule in self.rules:
            if rule.should_trigger(metrics):
                alert = rule.trigger(metrics)
                self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through all configured channels"""
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # Send through channels
        for channel in self.channels:
            try:
                channel.send_alert(alert)
            except Exception as e:
                logger.error(f"Error sending alert through {channel.__class__.__name__}: {e}")
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history"""
        return self.alert_history[-limit:]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get statistics about alerts"""
        if not self.alert_history:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "by_rule": {}
            }
        
        by_severity = {}
        by_rule = {}
        
        for alert in self.alert_history:
            severity = alert.get("severity", "unknown")
            rule = alert.get("rule_name", "unknown")
            
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_rule[rule] = by_rule.get(rule, 0) + 1
        
        return {
            "total_alerts": len(self.alert_history),
            "by_severity": by_severity,
            "by_rule": by_rule,
            "recent_alerts": self.alert_history[-10:]
        }