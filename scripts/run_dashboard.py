#!/usr/bin/env python
"""Script to run the interactive dashboard"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit.cli as stcli


def main():
    """Run the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent.parent / "src" / "dashboard" / "app.py"
    
    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--theme.primaryColor", "#FF4B4B",
        "--theme.backgroundColor", "#FFFFFF",
        "--theme.secondaryBackgroundColor", "#F0F2F6",
        "--theme.textColor", "#262730",
    ]
    
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()