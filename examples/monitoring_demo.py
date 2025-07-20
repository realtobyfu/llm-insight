#!/usr/bin/env python3
"""Demo script showcasing the production monitoring features"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import InterpretabilityAnalyzer
from src.monitoring import AlertManager, AlertRule, AlertSeverity


def main():
    print("=== LLM Insight Production Monitoring Demo ===\n")
    
    # Initialize analyzer with monitoring enabled
    print("1. Initializing analyzer with monitoring...")
    analyzer = InterpretabilityAnalyzer(
        model_name="distilgpt2",
        enable_monitoring=True
    )
    
    # Add custom alert callback
    def alert_callback(alert):
        print(f"\n🚨 ALERT: {alert['severity'].upper()} - {alert['message']}")
    
    analyzer.monitor.add_alert_callback(alert_callback)
    
    # Test cases that trigger different patterns
    test_cases = [
        # Normal text
        ("The quick brown fox jumps over the lazy dog.", "Normal text"),
        
        # Repetition pattern
        ("the the the the the the the the", "Repetition failure"),
        
        # Character spam
        ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Character spam"),
        
        # Mixed patterns
        ("Hello hello hello HELLO hello!!!", "Mixed repetition"),
        
        # Normal longer text
        ("Machine learning models can exhibit various failure modes. "
         "Understanding these patterns helps improve model reliability.", "Normal analysis"),
    ]
    
    print("\n2. Running analyses with different text patterns...\n")
    
    for text, description in test_cases:
        print(f"\n--- Testing: {description} ---")
        print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # Analyze text
            results = analyzer.analyze(
                text,
                methods=["attention", "importance"],
                use_cache=False
            )
            
            # Show metrics
            if analyzer.monitor:
                metrics = analyzer.monitor.collector.get_summary_stats()
                print(f"Total analyses: {metrics['total_analyses']}")
                print(f"Error rate: {metrics['error_rate']:.1%}")
                
                # Check for detected patterns
                recent_metrics = list(analyzer.monitor.collector.metrics_window)
                if recent_metrics:
                    latest = recent_metrics[-1]
                    if "failure_patterns" in latest.metadata:
                        patterns = latest.metadata["failure_patterns"]
                        if patterns["detected_patterns"]:
                            print(f"⚠️  Detected patterns:")
                            for pattern in patterns["detected_patterns"]:
                                print(f"   - {pattern['name']} (score: {pattern['score']:.2f})")
                        else:
                            print("✅ No failure patterns detected")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
        
        time.sleep(0.5)  # Small delay between analyses
    
    # Show monitoring summary
    print("\n\n3. Monitoring Summary")
    print("=" * 50)
    
    if analyzer.monitor:
        summary = analyzer.monitor.get_metrics()
        stats = summary.get("summary", {})
        
        print(f"Total Analyses: {stats.get('total_analyses', 0)}")
        print(f"Total Errors: {stats.get('total_errors', 0)}")
        print(f"Anomalies Detected: {stats.get('anomalies_detected', 0)}")
        print(f"High Risk Predictions: {stats.get('high_risk_predictions', 0)}")
        
        recent = stats.get("recent_stats", {})
        if recent:
            print(f"\nRecent Performance:")
            print(f"  Average Duration: {recent['avg_duration_ms']:.1f}ms")
            print(f"  Average Entropy: {recent['avg_entropy']:.3f}")
    
    # Show Prometheus metrics
    print("\n\n4. Prometheus Metrics Export")
    print("=" * 50)
    if analyzer.monitor:
        prometheus_metrics = analyzer.monitor.export_prometheus_metrics()
        print(prometheus_metrics[:500] + "..." if len(prometheus_metrics) > 500 else prometheus_metrics)
    
    print("\n\n5. API Endpoints Available")
    print("=" * 50)
    print("When running the API server, these monitoring endpoints are available:")
    print("  GET  /monitoring/metrics           - Get current metrics")
    print("  GET  /monitoring/metrics/prometheus - Prometheus format export")
    print("  GET  /monitoring/time-series/{metric} - Time series data")
    print("  GET  /monitoring/models/stats      - Model statistics")
    print("  GET  /monitoring/alerts            - Alert history")
    print("  GET  /monitoring/health            - Health check with metrics")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()