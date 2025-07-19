"""Attention visualization components"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AttentionVisualizer:
    """Visualizer for attention patterns and related metrics"""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize visualizer with style settings"""
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default if style not found
            plt.style.use('default')
        self.cmap = LinearSegmentedColormap.from_list(
            "attention", ["white", "lightblue", "blue", "darkblue"]
        )
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer: int = 0,
        head: int = 0,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """Plot attention heatmap for a specific layer and head
        
        Args:
            attention_weights: Attention tensor (layers, batch, heads, seq, seq)
            tokens: List of token strings
            layer: Layer index to visualize
            head: Head index to visualize
            figsize: Figure size
            save_path: Path to save figure
            **kwargs: Additional plot parameters
        
        Returns:
            Matplotlib figure
        """
        # Extract specific attention matrix
        if len(attention_weights.shape) == 5:  # Full tensor
            att_matrix = attention_weights[layer, 0, head].cpu().numpy()
        elif len(attention_weights.shape) == 2:  # Already extracted
            att_matrix = attention_weights.cpu().numpy()
        else:
            raise ValueError(f"Unexpected attention shape: {attention_weights.shape}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            att_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap=self.cmap,
            cbar_kws={"label": "Attention Weight"},
            square=True,
            linewidths=0.5,
            ax=ax,
            **kwargs
        )
        
        # Formatting
        ax.set_title(f"Attention Pattern - Layer {layer}, Head {head}")
        ax.set_xlabel("To Token")
        ax.set_ylabel("From Token")
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved attention heatmap to {save_path}")
        
        return fig
    
    def plot_attention_rollout(
        self,
        rollout_matrix: torch.Tensor,
        tokens: List[str],
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        highlight_token: Optional[int] = None,
    ) -> plt.Figure:
        """Plot attention rollout visualization
        
        Args:
            rollout_matrix: Rolled out attention matrix
            tokens: List of token strings
            figsize: Figure size
            save_path: Path to save figure
            highlight_token: Token index to highlight
        
        Returns:
            Matplotlib figure
        """
        rollout = rollout_matrix.cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Main heatmap
        sns.heatmap(
            rollout,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="Blues",
            cbar_kws={"label": "Attention Flow"},
            square=True,
            ax=ax1
        )
        ax1.set_title("Attention Rollout - Information Flow Through Layers")
        ax1.set_xlabel("To Token")
        ax1.set_ylabel("From Token")
        
        # Attention to final token
        final_attention = rollout[-1, :]
        ax2.bar(range(len(tokens)), final_attention)
        ax2.set_xticks(range(len(tokens)))
        ax2.set_xticklabels(tokens, rotation=45, ha="right")
        ax2.set_ylabel("Attention")
        ax2.set_title("Attention to Final Token")
        
        if highlight_token is not None:
            ax2.axvline(highlight_token, color="red", linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved attention rollout to {save_path}")
        
        return fig
    
    def plot_head_importance_matrix(
        self,
        head_importance: torch.Tensor,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot head importance scores as a matrix
        
        Args:
            head_importance: Tensor of shape (num_layers, num_heads)
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        importance = head_importance.cpu().numpy()
        num_layers, num_heads = importance.shape
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(importance, cmap="YlOrRd", aspect="auto")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Importance Score")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(num_heads))
        ax.set_yticks(np.arange(num_layers))
        ax.set_xticklabels([f"Head {i}" for i in range(num_heads)])
        ax.set_yticklabels([f"Layer {i}" for i in range(num_layers)])
        
        # Add text annotations
        for i in range(num_layers):
            for j in range(num_heads):
                text = ax.text(j, i, f"{importance[i, j]:.2f}",
                             ha="center", va="center", color="black" if importance[i, j] < 0.5 else "white")
        
        ax.set_title("Attention Head Importance Scores")
        ax.set_xlabel("Attention Head")
        ax.set_ylabel("Layer")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved head importance matrix to {save_path}")
        
        return fig
    
    def plot_attention_patterns_grid(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot grid of attention patterns across layers and heads
        
        Args:
            attention_weights: Full attention tensor
            tokens: List of token strings
            layers: Specific layers to plot (None for first 4)
            heads: Specific heads to plot (None for first 4)
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        # Default to first 4 layers and heads
        if layers is None:
            layers = list(range(min(4, attention_weights.shape[0])))
        if heads is None:
            heads = list(range(min(4, attention_weights.shape[2])))
        
        n_rows = len(layers)
        n_cols = len(heads)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, layer in enumerate(layers):
            for j, head in enumerate(heads):
                ax = axes[i, j]
                
                # Get attention matrix
                att = attention_weights[layer, 0, head].cpu().numpy()
                
                # Plot
                im = ax.imshow(att, cmap="Blues", vmin=0, vmax=1)
                
                # Labels
                ax.set_title(f"L{layer} H{head}", fontsize=10)
                
                # Only show labels on edges
                if i == n_rows - 1:
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
                else:
                    ax.set_xticks([])
                
                if j == 0:
                    ax.set_yticks(range(len(tokens)))
                    ax.set_yticklabels(tokens, fontsize=8)
                else:
                    ax.set_yticks([])
        
        # Add colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Attention Weight")
        
        fig.suptitle("Attention Patterns Across Layers and Heads", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved attention grid to {save_path}")
        
        return fig
    
    def plot_token_importance(
        self,
        tokens: List[str],
        importance_scores: torch.Tensor,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> plt.Figure:
        """Plot token importance scores
        
        Args:
            tokens: List of token strings
            importance_scores: Importance scores for each token
            figsize: Figure size
            save_path: Path to save figure
            top_k: Only show top k tokens
        
        Returns:
            Matplotlib figure
        """
        scores = importance_scores.cpu().numpy()
        
        # Sort by importance if top_k specified
        if top_k and top_k < len(tokens):
            indices = np.argsort(scores)[-top_k:]
            tokens = [tokens[i] for i in indices]
            scores = scores[indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        bars = ax.bar(range(len(tokens)), scores)
        
        # Color bars by score
        norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())
        colors = plt.cm.viridis(norm(scores))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_ylabel("Importance Score")
        ax.set_title("Token Importance for Model Prediction")
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved token importance to {save_path}")
        
        return fig
    
    def plot_attention_entropy(
        self,
        entropy_by_layer: torch.Tensor,
        entropy_by_position: Optional[torch.Tensor] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot attention entropy statistics
        
        Args:
            entropy_by_layer: Mean entropy per layer
            entropy_by_position: Mean entropy per position (optional)
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        n_subplots = 2 if entropy_by_position is not None else 1
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
        
        if n_subplots == 1:
            axes = [axes]
        
        # Plot entropy by layer
        ax1 = axes[0]
        layer_entropy = entropy_by_layer.cpu().numpy()
        ax1.plot(range(len(layer_entropy)), layer_entropy, marker='o')
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Mean Entropy")
        ax1.set_title("Attention Entropy by Layer")
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line for reference
        ax1.axhline(y=layer_entropy.mean(), color='r', linestyle='--', 
                   label=f'Mean: {layer_entropy.mean():.2f}')
        ax1.legend()
        
        # Plot entropy by position if provided
        if entropy_by_position is not None:
            ax2 = axes[1]
            pos_entropy = entropy_by_position.cpu().numpy()
            ax2.plot(range(len(pos_entropy)), pos_entropy, marker='s', color='green')
            ax2.set_xlabel("Position")
            ax2.set_ylabel("Mean Entropy")
            ax2.set_title("Attention Entropy by Position")
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved entropy plot to {save_path}")
        
        return fig