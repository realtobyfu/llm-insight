#!/usr/bin/env python3
"""Test script to verify imports are working"""

import sys
print(f"Python path: {sys.path[0]}")

try:
    from src.core import InterpretabilityAnalyzer
    print("✓ Successfully imported InterpretabilityAnalyzer")
    
    from src.visualization import InteractiveVisualizer
    print("✓ Successfully imported InteractiveVisualizer")
    
    from src.core.anomaly_detection import AttentionPatternAnalyzer
    print("✓ Successfully imported AttentionPatternAnalyzer (no Union error)")
    
    print("\nAll imports successful! The Streamlit app should work now.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")