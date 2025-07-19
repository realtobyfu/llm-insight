#!/bin/bash

# Activate virtual environment and run Streamlit dashboard

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Run Streamlit app
echo "Starting LLM Interpretability Dashboard..."
echo "The dashboard will be available at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

python -m streamlit run src/dashboard/app.py