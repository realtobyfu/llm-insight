"""Interactive dashboard for LLM Interpretability Toolkit"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.core import InterpretabilityAnalyzer
from src.visualization import InteractiveVisualizer
from src.monitoring import AnalysisMonitor


# Page configuration
st.set_page_config(
    page_title="LLM Interpretability Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Professional color scheme */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --success-color: #27AE60;
        --warning-color: #F39C12;
        --danger-color: #E74C3C;
        --neutral-bg: #F8F9FA;
        --card-bg: #FFFFFF;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: var(--card-bg);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Header styling */
    h1 {
        color: var(--primary-color);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #2C3E50;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #236590;
        transform: translateY(-1px);
    }
    
    /* Status indicators */
    .status-success {
        color: var(--success-color);
        font-weight: 600;
    }
    
    .status-warning {
        color: var(--warning-color);
        font-weight: 600;
    }
    
    .status-danger {
        color: var(--danger-color);
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--neutral-bg);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #EBF5FB;
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


class DashboardState:
    """Manage dashboard state"""
    
    def __init__(self):
        if "analyzer" not in st.session_state:
            st.session_state.analyzer = None
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}
        if "history" not in st.session_state:
            st.session_state.history = []


def round_nested_dict(obj, decimals=3):
    """Recursively round float values in nested structures"""
    if isinstance(obj, dict):
        return {k: round_nested_dict(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_nested_dict(item, decimals) for item in obj]
    elif isinstance(obj, float):
        return round(obj, decimals)
    else:
        return obj


def initialize_analyzer(model_name: str) -> InterpretabilityAnalyzer:
    """Initialize or get cached analyzer"""
    if (st.session_state.analyzer is None or 
        st.session_state.analyzer.model_name != model_name):
        with st.spinner(f"Loading model {model_name}..."):
            st.session_state.analyzer = InterpretabilityAnalyzer(model_name)
    return st.session_state.analyzer


def display_model_info():
    """Display supported models and formats"""
    with st.expander("Supported Models & Formats", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Supported Models
            - **GPT-2 Family**: gpt2, distilgpt2, gpt2-medium, gpt2-large, gpt2-xl
            - **BERT Family**: bert-base-uncased, bert-large-uncased, distilbert-base-uncased
            - **RoBERTa**: roberta-base, roberta-large
            - **T5**: t5-small, t5-base, t5-large
            - **Any HuggingFace Model**: Via model name/path
            
            ### Input Formats
            - **Text**: Plain text for analysis
            - **Batch**: Text file with one sample per line
            - **Token IDs**: For advanced users
            """)
        
        with col2:
            st.markdown("""
            ### Self-Hosting Considerations
            - **Memory Requirements**: 
                - Small models (distilgpt2): 2-4GB
                - Base models (gpt2, bert-base): 4-8GB
                - Large models: 8-16GB+
            - **GPU Recommended**: For faster inference
            - **Storage**: Models cached locally after first download
            - **API Alternative**: Consider API-based deployment for production
            
            ### Use Cases
            - Research & Development
            - Model debugging & testing
            - Educational purposes
            - Production monitoring (with proper infrastructure)
            """)


def main():
    """Main dashboard application"""
    state = DashboardState()
    
    # Header
    st.title("LLM Interpretability Dashboard")
    st.markdown("Toolkit for analyzing attention patterns, detecting anomalies, and predicting failures in language models")
    
    # Display model information
    display_model_info()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["distilgpt2", "gpt2", "bert-base-uncased", "distilbert-base-uncased"],
            help="Choose the language model to analyze"
        )
        
        # Initialize analyzer
        analyzer = initialize_analyzer(model_name)
        
        st.success(f"Model loaded: {model_name}")
        
        # Display model info
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Layers", analyzer.model_wrapper.get_num_layers())
            st.metric("Hidden Size", analyzer.model_wrapper.get_hidden_size())
        with col2:
            st.metric("Heads", analyzer.model_wrapper.get_num_attention_heads())
            st.metric("Vocab Size", analyzer.model_wrapper.get_vocab_size())
        
        # Analysis options
        st.subheader("Analysis Options")
        methods = st.multiselect(
            "Analysis Methods",
            ["attention", "importance", "sae", "head_patterns"],
            default=["attention", "importance"],
            help="Select which analysis methods to run"
        )
        
        use_cache = st.checkbox("Use Cache", value=True, help="Cache results for faster repeated analysis")
    
    # Main content
    tabs = st.tabs(["Text Analysis", "Anomaly Detection", "Failure Prediction", "Batch Analysis", "History", "Monitoring"])
    
    # Tab 1: Text Analysis
    with tabs[0]:
        st.header("Text Analysis")
        
        # Input section
        col1, col2 = st.columns([3, 1])
        with col1:
            input_text = st.text_area(
                "Enter text to analyze",
                value="The quick brown fox jumps over the lazy dog.",
                height=100,
                help="Enter the text you want to analyze"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")
            analyze_button = st.button("Analyze", type="primary", use_container_width=True)
        
        # Analysis results
        if analyze_button and input_text:
            with st.spinner("Analyzing text..."):
                try:
                    # Perform analysis
                    results = analyzer.analyze(
                        input_text,
                        methods=methods,
                        use_cache=use_cache
                    )
                    
                    # Store results
                    st.session_state.analysis_results = results
                    st.session_state.history.append({
                        "text": input_text,
                        "model": model_name,
                        "methods": methods,
                        "timestamp": pd.Timestamp.now()
                    })
                    
                    st.success("Analysis complete")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    return
        
        # Display results
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Metrics row
            st.markdown("### Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            if "attention" in results:
                with col1:
                    entropy = results["attention"]["entropy"]["mean_entropy_per_layer"]
                    mean_entropy = torch.tensor(entropy).mean().item()
                    st.metric("Mean Entropy", f"{mean_entropy:.3f}")
            
            if "importance" in results:
                with col2:
                    importance = results["importance"]["token_importance"]["importance_mean"]
                    max_importance = max(importance)
                    st.metric("Max Token Importance", f"{max_importance:.3f}")
            
            # Visualization tabs
            viz_tabs = st.tabs(["Attention Heatmap", "Token Importance", "Head Patterns", "Layer Analysis"])
            
            # Attention Heatmap
            with viz_tabs[0]:
                if "attention" in results:
                    st.subheader("Attention Patterns")
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        layer = st.slider("Layer", 0, results["metadata"]["num_layers"]-1, 0)
                        head = st.slider("Head", 0, results["metadata"]["num_heads"]-1, 0)
                    
                    with col2:
                        viz = InteractiveVisualizer()
                        attention_fig = viz.create_attention_heatmap(
                            torch.tensor(results["attention"]["patterns"]),
                            results["attention"]["tokens"][0],
                            layer=layer,
                            head=head
                        )
                        st.plotly_chart(attention_fig, use_container_width=True)
            
            # Token Importance
            with viz_tabs[1]:
                if "importance" in results:
                    st.subheader("Token Importance Analysis")
                    
                    viz = InteractiveVisualizer()
                    importance_fig = viz.create_token_importance_bar(
                        results["importance"]["token_importance"]["tokens"],
                        torch.tensor(results["importance"]["token_importance"]["importance_mean"])
                    )
                    st.plotly_chart(importance_fig, use_container_width=True)
            
            # Head Patterns
            with viz_tabs[2]:
                if "head_patterns" in results:
                    st.subheader("Attention Head Patterns")
                    
                    patterns = results["head_patterns"]["identified_patterns"]
                    pattern_df = pd.DataFrame([
                        {"Pattern": pattern, "Count": len(heads), "Heads": str(heads[:5])}
                        for pattern, heads in patterns.items()
                    ])
                    
                    st.dataframe(pattern_df, use_container_width=True)
            
            # Layer Analysis
            with viz_tabs[3]:
                if "attention" in results:
                    st.subheader("Layer-wise Metrics")
                    
                    entropy_data = results["attention"]["entropy"]
                    viz = InteractiveVisualizer()
                    
                    metrics_dict = {
                        "entropy": torch.tensor(entropy_data["mean_entropy_per_layer"]),
                    }
                    
                    layer_fig = viz.create_layer_wise_metrics(metrics_dict, ["Entropy"])
                    st.plotly_chart(layer_fig, use_container_width=True)
    
    # Tab 2: Anomaly Detection
    with tabs[1]:
        st.header("Anomaly Detection")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            anomaly_text = st.text_area(
                "Enter text to check for anomalies",
                value="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                height=100
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")
            detect_button = st.button("Detect Anomalies", type="primary", use_container_width=True)
        
        if detect_button and anomaly_text:
            with st.spinner("Detecting anomalies..."):
                try:
                    anomaly_results = analyzer.detect_anomalies(anomaly_text)
                    
                    if anomaly_results["is_anomaly"]:
                        st.error(f"Anomaly detected! Confidence: {anomaly_results['confidence']:.2%}")
                        st.write("**Anomaly Types:**")
                        for anomaly_type in anomaly_results["anomaly_types"]:
                            st.write(f"- {anomaly_type}")
                    else:
                        st.success("No anomalies detected")
                    
                    # Show detailed scores
                    with st.expander("Detailed Scores"):
                        rounded_scores = round_nested_dict(anomaly_results["scores"])
                        st.json(rounded_scores)
                
                except Exception as e:
                    st.error(f"Anomaly detection failed: {str(e)}")
    
    # Tab 3: Failure Prediction
    with tabs[2]:
        st.header("Failure Prediction")
        
        # Training section
        with st.expander("Train Failure Predictor", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                n_samples = st.number_input("Synthetic Samples", min_value=50, max_value=500, value=100)
            with col2:
                train_button = st.button("Train Model", use_container_width=True)
            
            if train_button:
                with st.spinner("Training failure predictor..."):
                    try:
                        train_results = analyzer.train_failure_predictor(
                            generate_synthetic=True,
                            n_synthetic_samples=n_samples
                        )
                        st.success(f"Model trained! Test accuracy: {train_results['test_accuracy']:.2%}")
                        
                        # Show feature importance
                        st.subheader("Top Important Features")
                        for feature, importance in train_results["feature_importance"][:5]:
                            st.write(f"- {feature}: {importance:.3f}")
                    
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
        
        # Prediction section
        col1, col2 = st.columns([3, 1])
        with col1:
            failure_text = st.text_area(
                "Enter text to predict failure",
                value="the the the the the the the the",
                height=100
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")
            predict_button = st.button("Predict Failure", type="primary", use_container_width=True)
        
        if predict_button and failure_text:
            with st.spinner("Predicting failure probability..."):
                try:
                    failure_results = analyzer.predict_failure_probability(failure_text)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Failure Probability", f"{failure_results['failure_probability']:.2%}")
                    with col2:
                        risk_level = failure_results["prediction"]
                        risk_color = "status-danger" if risk_level == "high_risk" else "status-success"
                        st.markdown(f'<p class="{risk_color}">Risk Level: {risk_level.replace("_", " ").title()}</p>', unsafe_allow_html=True)
                    with col3:
                        st.metric("Confidence", f"{failure_results['confidence']:.2%}")
                    
                    st.write("**Explanation:**", failure_results['explanation'])
                    
                    if failure_results['indicators']:
                        st.write("**Indicators:**")
                        for indicator in failure_results['indicators']:
                            st.write(f"- {indicator}")
                
                except Exception as e:
                    st.error(f"Failure prediction failed: {str(e)}")
    
    # Tab 4: Batch Analysis
    with tabs[3]:
        st.header("Batch Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload text file (one text per line)",
            type=["txt"],
            help="Upload a text file with one text sample per line"
        )
        
        if uploaded_file is not None:
            texts = uploaded_file.read().decode("utf-8").strip().split("\n")
            st.info(f"Loaded {len(texts)} texts")
            
            if st.button("Analyze Batch"):
                progress_bar = st.progress(0)
                results_container = st.container()
                
                batch_results = []
                for i, text in enumerate(texts):
                    progress_bar.progress((i + 1) / len(texts))
                    
                    try:
                        # Basic analysis
                        analysis = analyzer.analyze(text, methods=["attention"], use_cache=use_cache)
                        
                        # Failure prediction
                        failure = analyzer.predict_failure_probability(text)
                        
                        batch_results.append({
                            "index": i,
                            "text": text[:50] + "..." if len(text) > 50 else text,
                            "entropy": torch.tensor(analysis["attention"]["entropy"]["mean_entropy_per_layer"]).mean().item(),
                            "failure_prob": failure["failure_probability"],
                            "risk": failure["prediction"]
                        })
                    
                    except Exception as e:
                        batch_results.append({
                            "index": i,
                            "text": text[:50] + "...",
                            "error": str(e)
                        })
                
                # Display results
                results_df = pd.DataFrame(batch_results)
                # Round numeric columns for display
                numeric_cols = results_df.select_dtypes(include=['float64', 'float32']).columns
                results_df[numeric_cols] = results_df[numeric_cols].round(3)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Texts", len(texts))
                with col2:
                    high_risk = len(results_df[results_df["risk"] == "high_risk"]) if "risk" in results_df else 0
                    st.metric("High Risk", high_risk)
                with col3:
                    avg_entropy = results_df["entropy"].mean() if "entropy" in results_df else 0
                    st.metric("Avg Entropy", f"{avg_entropy:.3f}")
    
    # Tab 5: History
    with tabs[4]:
        st.header("Analysis History")
        
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)
            
            # Clear history button
            if st.button("Clear History"):
                st.session_state.history = []
                st.experimental_rerun()
        else:
            st.info("No analysis history yet")
    
    # Tab 6: Monitoring
    with tabs[5]:
        st.header("System Monitoring")
        
        # Get monitoring metrics if analyzer has monitoring enabled
        if hasattr(analyzer, 'monitor') and analyzer.monitor:
            monitor = analyzer.monitor
            metrics = monitor.get_metrics()
            
            # Summary metrics
            st.subheader("System Overview")
            summary = metrics.get("summary", {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyses", summary.get("total_analyses", 0))
            with col2:
                error_rate = summary.get("error_rate", 0)
                st.metric("Error Rate", f"{error_rate:.1%}", 
                         delta=None if error_rate < 0.05 else "⚠️")
            with col3:
                st.metric("Anomalies", summary.get("anomalies_detected", 0))
            with col4:
                st.metric("High Risk", summary.get("high_risk_predictions", 0))
            
            # Recent performance
            st.subheader("Recent Performance")
            recent = summary.get("recent_stats", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_duration = recent.get("avg_duration_ms", 0)
                st.metric("Avg Duration", f"{avg_duration:.0f}ms")
            with col2:
                max_duration = recent.get("max_duration_ms", 0)
                st.metric("Max Duration", f"{max_duration:.0f}ms")
            with col3:
                avg_entropy = recent.get("avg_entropy", 0)
                st.metric("Avg Entropy", f"{avg_entropy:.3f}")
            
            # Time series charts
            st.subheader("Performance Trends")
            
            # Duration trend
            duration_data = metrics.get("time_series", {}).get("duration", [])
            if duration_data:
                duration_df = pd.DataFrame(duration_data)
                duration_df['timestamp'] = pd.to_datetime(duration_df['timestamp'], unit='s')
                
                fig = px.line(duration_df, x='timestamp', y='value', 
                             color='model', title="Analysis Duration Over Time")
                fig.update_yaxis(title="Duration (ms)")
                st.plotly_chart(fig, use_container_width=True)
            
            # Entropy trend
            entropy_data = metrics.get("time_series", {}).get("entropy", [])
            if entropy_data:
                entropy_df = pd.DataFrame(entropy_data)
                entropy_df['timestamp'] = pd.to_datetime(entropy_df['timestamp'], unit='s')
                
                fig = px.line(entropy_df, x='timestamp', y='value',
                             color='model', title="Attention Entropy Over Time")
                fig.update_yaxis(title="Entropy")
                st.plotly_chart(fig, use_container_width=True)
            
            # Model breakdown
            st.subheader("Model Statistics")
            model_stats = summary.get("models", {})
            if model_stats:
                model_df = pd.DataFrame([
                    {"Model": model, "Analyses": count}
                    for model, count in model_stats.items()
                ])
                
                fig = px.bar(model_df, x='Model', y='Analyses',
                            title="Analyses by Model")
                st.plotly_chart(fig, use_container_width=True)
            
            # Method breakdown
            method_stats = summary.get("methods", {})
            if method_stats:
                method_df = pd.DataFrame([
                    {"Method": method, "Count": count}
                    for method, count in method_stats.items()
                ])
                
                fig = px.pie(method_df, values='Count', names='Method',
                            title="Analysis Methods Used")
                st.plotly_chart(fig, use_container_width=True)
            
            # Alerts section
            if hasattr(monitor, 'alert_manager'):
                st.subheader("Recent Alerts")
                alert_stats = monitor.alert_manager.get_alert_stats()
                
                if alert_stats["total_alerts"] > 0:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Total Alerts", alert_stats["total_alerts"])
                        
                        # Alert breakdown
                        for severity, count in alert_stats["by_severity"].items():
                            if severity == "critical":
                                st.error(f"{severity.upper()}: {count}")
                            elif severity == "error":
                                st.error(f"{severity.upper()}: {count}")
                            elif severity == "warning":
                                st.warning(f"{severity.upper()}: {count}")
                            else:
                                st.info(f"{severity.upper()}: {count}")
                    
                    with col2:
                        # Recent alerts list
                        recent_alerts = alert_stats.get("recent_alerts", [])[:5]
                        for alert in recent_alerts:
                            alert_time = pd.Timestamp(alert["timestamp"], unit='s')
                            st.write(f"**{alert_time.strftime('%H:%M:%S')}** - "
                                   f"{alert['severity'].upper()}: {alert['message']}")
                else:
                    st.success("No alerts triggered")
            
            # Refresh button
            if st.button("Refresh Metrics", type="secondary"):
                st.experimental_rerun()
                
        else:
            st.warning("Monitoring is not enabled for this analyzer instance")
            st.info("To enable monitoring, initialize the analyzer with `enable_monitoring=True`")


if __name__ == "__main__":
    main()