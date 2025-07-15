"""Interactive visualization components using Plotly"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch

from ..utils.logger import get_logger

logger = get_logger(__name__)


class InteractiveVisualizer:
    """Interactive visualizations using Plotly"""
    
    def __init__(self):
        """Initialize interactive visualizer"""
        self.default_layout = {
            "template": "plotly_white",
            "font": {"family": "Arial, sans-serif", "size": 12},
        }
    
    def create_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer: int = 0,
        head: int = 0,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create interactive attention heatmap
        
        Args:
            attention_weights: Attention tensor
            tokens: List of token strings
            layer: Layer index
            head: Head index
            title: Optional custom title
        
        Returns:
            Plotly figure
        """
        # Extract attention matrix
        if len(attention_weights.shape) == 5:
            att_matrix = attention_weights[layer, 0, head].cpu().numpy()
        else:
            att_matrix = attention_weights.cpu().numpy()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=att_matrix,
            x=tokens,
            y=tokens,
            colorscale='Blues',
            text=np.round(att_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='From: %{y}<br>To: %{x}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=title or f"Attention Pattern - Layer {layer}, Head {head}",
            xaxis_title="To Token",
            yaxis_title="From Token",
            width=800,
            height=600,
            **self.default_layout
        )
        
        return fig
    
    def create_multi_head_attention(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer: int = 0,
        max_heads: int = 8,
    ) -> go.Figure:
        """Create multi-head attention visualization
        
        Args:
            attention_weights: Full attention tensor
            tokens: List of token strings
            layer: Layer to visualize
            max_heads: Maximum number of heads to show
        
        Returns:
            Plotly figure with subplots
        """
        num_heads = min(attention_weights.shape[2], max_heads)
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Head {i}" for i in range(num_heads)],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Add heatmaps
        for head in range(num_heads):
            row = head // cols + 1
            col = head % cols + 1
            
            att_matrix = attention_weights[layer, 0, head].cpu().numpy()
            
            fig.add_trace(
                go.Heatmap(
                    z=att_matrix,
                    colorscale='Blues',
                    showscale=(head == 0),
                    hovertemplate='Weight: %{z:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Update axes
            fig.update_xaxis(
                ticktext=tokens,
                tickvals=list(range(len(tokens))),
                tickangle=45,
                row=row, col=col
            )
            fig.update_yaxis(
                ticktext=tokens,
                tickvals=list(range(len(tokens))),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title=f"Multi-Head Attention - Layer {layer}",
            height=200 * rows,
            width=200 * cols,
            **self.default_layout
        )
        
        return fig
    
    def create_token_importance_bar(
        self,
        tokens: List[str],
        importance_scores: torch.Tensor,
        title: str = "Token Importance",
    ) -> go.Figure:
        """Create interactive token importance bar chart
        
        Args:
            tokens: List of token strings
            importance_scores: Importance scores
            title: Chart title
        
        Returns:
            Plotly figure
        """
        scores = importance_scores.cpu().numpy()
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=tokens,
                y=scores,
                text=np.round(scores, 3),
                textposition='auto',
                marker_color=scores,
                marker_colorscale='Viridis',
                hovertemplate='Token: %{x}<br>Importance: %{y:.3f}<extra></extra>'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Token",
            yaxis_title="Importance Score",
            xaxis_tickangle=-45,
            **self.default_layout
        )
        
        return fig
    
    def create_feature_activation_3d(
        self,
        feature_activations: torch.Tensor,
        feature_names: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create 3D visualization of feature activations
        
        Args:
            feature_activations: Feature activation tensor
            feature_names: Optional names for features
        
        Returns:
            Plotly 3D scatter plot
        """
        activations = feature_activations.cpu().numpy()
        
        # Apply PCA for 3D visualization
        from sklearn.decomposition import PCA
        
        if activations.shape[1] > 3:
            pca = PCA(n_components=3)
            coords = pca.fit_transform(activations)
            variance_explained = pca.explained_variance_ratio_
        else:
            coords = activations
            variance_explained = [1.0, 0.0, 0.0]
        
        # Create 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2] if coords.shape[1] > 2 else np.zeros(coords.shape[0]),
            mode='markers',
            marker=dict(
                size=5,
                color=np.arange(coords.shape[0]),
                colorscale='Viridis',
                opacity=0.8
            ),
            text=feature_names if feature_names else [f"Sample {i}" for i in range(coords.shape[0])],
            hoverinfo='text'
        )])
        
        # Update layout
        fig.update_layout(
            title="Feature Activation Space (PCA)",
            scene=dict(
                xaxis_title=f"PC1 ({variance_explained[0]:.1%})",
                yaxis_title=f"PC2 ({variance_explained[1]:.1%})",
                zaxis_title=f"PC3 ({variance_explained[2]:.1%})",
            ),
            **self.default_layout
        )
        
        return fig
    
    def create_attention_flow_sankey(
        self,
        attention_rollout: torch.Tensor,
        tokens: List[str],
        threshold: float = 0.1,
    ) -> go.Figure:
        """Create Sankey diagram for attention flow
        
        Args:
            attention_rollout: Attention rollout matrix
            tokens: List of token strings
            threshold: Minimum attention weight to show
        
        Returns:
            Plotly Sankey diagram
        """
        rollout = attention_rollout.cpu().numpy()
        
        # Create source, target, and value lists
        sources = []
        targets = []
        values = []
        
        for i in range(len(tokens) - 1):
            for j in range(i + 1, len(tokens)):
                weight = rollout[i, j]
                if weight > threshold:
                    sources.append(i)
                    targets.append(j)
                    values.append(weight)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=tokens,
                color="lightblue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(0,0,255,0.2)"
            )
        )])
        
        # Update layout
        fig.update_layout(
            title="Attention Flow Between Tokens",
            font_size=10,
            **self.default_layout
        )
        
        return fig
    
    def create_layer_wise_metrics(
        self,
        metrics_by_layer: Dict[str, torch.Tensor],
        metric_names: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create layer-wise metrics visualization
        
        Args:
            metrics_by_layer: Dictionary of metrics per layer
            metric_names: Names for the metrics
        
        Returns:
            Plotly figure with multiple traces
        """
        fig = go.Figure()
        
        # Add traces for each metric
        for i, (name, values) in enumerate(metrics_by_layer.items()):
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines+markers',
                name=metric_names[i] if metric_names else name,
                hovertemplate='Layer: %{x}<br>Value: %{y:.3f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title="Layer-wise Metrics",
            xaxis_title="Layer",
            yaxis_title="Metric Value",
            hovermode='x unified',
            **self.default_layout
        )
        
        return fig
    
    def create_head_importance_heatmap(
        self,
        head_importance: torch.Tensor,
    ) -> go.Figure:
        """Create interactive head importance heatmap
        
        Args:
            head_importance: Tensor of shape (num_layers, num_heads)
        
        Returns:
            Plotly heatmap figure
        """
        importance = head_importance.cpu().numpy()
        num_layers, num_heads = importance.shape
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=importance,
            x=[f"Head {i}" for i in range(num_heads)],
            y=[f"Layer {i}" for i in range(num_layers)],
            colorscale='YlOrRd',
            text=np.round(importance, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='Layer: %{y}<br>Head: %{x}<br>Importance: %{z:.3f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title="Attention Head Importance Scores",
            xaxis_title="Attention Head",
            yaxis_title="Layer",
            **self.default_layout
        )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filepath: str, format: str = "html"):
        """Save interactive figure to file
        
        Args:
            fig: Plotly figure
            filepath: Path to save file
            format: File format ("html", "png", "svg", "pdf")
        """
        if format == "html":
            fig.write_html(filepath)
        else:
            fig.write_image(filepath, format=format)
        
        logger.info(f"Saved interactive figure to {filepath}")