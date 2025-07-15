"""Feature visualization for SAE and activation analysis"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureVisualizer:
    """Visualizer for SAE features and model activations"""
    
    def __init__(self, style: str = "seaborn"):
        """Initialize visualizer with style settings"""
        plt.style.use(style)
    
    def plot_feature_activations(
        self,
        feature_activations: torch.Tensor,
        feature_indices: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature activation patterns
        
        Args:
            feature_activations: Tensor of feature activations
            feature_indices: Specific features to highlight
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        activations = feature_activations.cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
        
        # Heatmap of feature activations
        im = ax1.imshow(activations.T, aspect='auto', cmap='viridis')
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Feature Index")
        ax1.set_title("SAE Feature Activations")
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label("Activation Strength")
        
        # Highlight specific features
        if feature_indices:
            for idx in feature_indices:
                ax1.axhline(y=idx, color='red', linestyle='--', alpha=0.7)
        
        # Feature activation frequency
        activation_freq = (activations > 0.1).mean(axis=0)
        ax2.bar(range(len(activation_freq)), activation_freq)
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Activation Frequency")
        ax2.set_title("Feature Activation Frequency")
        
        # Mark dead features
        dead_threshold = 0.01
        dead_features = np.where(activation_freq < dead_threshold)[0]
        if len(dead_features) > 0:
            ax2.scatter(dead_features, activation_freq[dead_features], 
                       color='red', s=50, marker='x', label=f'Dead features: {len(dead_features)}')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature activations to {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_importance: torch.Tensor,
        top_k: int = 20,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot top-k most important features
        
        Args:
            feature_importance: Importance scores for each feature
            top_k: Number of top features to show
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        importance = feature_importance.cpu().numpy()
        
        # Get top-k features
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_scores = importance[top_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_indices))
        bars = ax.barh(y_pos, top_scores)
        
        # Color by importance
        colors = plt.cm.plasma(top_scores / top_scores.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Feature {idx}" for idx in top_indices])
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top {top_k} Most Important SAE Features")
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center')
        
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature importance to {save_path}")
        
        return fig
    
    def plot_feature_correlation(
        self,
        correlation_matrix: torch.Tensor,
        feature_subset: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature correlation matrix
        
        Args:
            correlation_matrix: Feature correlation matrix
            feature_subset: Subset of features to plot
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        corr = correlation_matrix.cpu().numpy()
        
        # Subset if specified
        if feature_subset:
            corr = corr[np.ix_(feature_subset, feature_subset)]
            labels = [f"F{i}" for i in feature_subset]
        else:
            # If too many features, sample
            if corr.shape[0] > 50:
                indices = np.random.choice(corr.shape[0], 50, replace=False)
                corr = corr[np.ix_(indices, indices)]
                labels = [f"F{i}" for i in indices]
            else:
                labels = [f"F{i}" for i in range(corr.shape[0])]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        
        ax.set_title("SAE Feature Correlation Matrix")
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature correlation to {save_path}")
        
        return fig
    
    def plot_reconstruction_error(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        position_labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot reconstruction error analysis
        
        Args:
            original: Original activations
            reconstructed: Reconstructed activations
            position_labels: Labels for positions
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        orig = original.cpu().numpy()
        recon = reconstructed.cpu().numpy()
        
        # Reshape if needed
        if orig.ndim == 1:
            orig = orig.reshape(1, -1)
            recon = recon.reshape(1, -1)
        
        # Compute errors
        abs_error = np.abs(orig - recon)
        rel_error = abs_error / (np.abs(orig) + 1e-8)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Original vs Reconstructed
        ax1 = axes[0, 0]
        ax1.plot(orig[0], label='Original', alpha=0.7)
        ax1.plot(recon[0], label='Reconstructed', alpha=0.7)
        ax1.set_title("Original vs Reconstructed Activations")
        ax1.set_xlabel("Dimension")
        ax1.set_ylabel("Activation Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Absolute error
        ax2 = axes[0, 1]
        ax2.plot(abs_error[0])
        ax2.set_title("Absolute Reconstruction Error")
        ax2.set_xlabel("Dimension")
        ax2.set_ylabel("Absolute Error")
        ax2.grid(True, alpha=0.3)
        
        # Error heatmap across positions
        ax3 = axes[1, 0]
        if orig.shape[0] > 1:
            im = ax3.imshow(abs_error, aspect='auto', cmap='Reds')
            ax3.set_title("Reconstruction Error Heatmap")
            ax3.set_xlabel("Dimension")
            ax3.set_ylabel("Position")
            plt.colorbar(im, ax=ax3)
            
            if position_labels:
                ax3.set_yticks(range(len(position_labels)))
                ax3.set_yticklabels(position_labels)
        else:
            ax3.hist(abs_error[0], bins=30, edgecolor='black')
            ax3.set_title("Error Distribution")
            ax3.set_xlabel("Absolute Error")
            ax3.set_ylabel("Count")
        
        # Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""Reconstruction Statistics:
        
Mean Absolute Error: {abs_error.mean():.4f}
Max Absolute Error: {abs_error.max():.4f}
Mean Relative Error: {rel_error.mean():.4f}
R² Score: {1 - np.var(orig - recon) / np.var(orig):.4f}
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved reconstruction error to {save_path}")
        
        return fig
    
    def plot_activation_embeddings(
        self,
        activations: torch.Tensor,
        labels: Optional[List[int]] = None,
        method: str = "pca",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot 2D embeddings of activations using dimensionality reduction
        
        Args:
            activations: Activation tensor
            labels: Optional labels for coloring
            method: Dimensionality reduction method ("pca" or "tsne")
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        acts = activations.cpu().numpy()
        
        # Reshape if needed
        if acts.ndim > 2:
            acts = acts.reshape(acts.shape[0], -1)
        
        # Apply dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2)
            embeddings = reducer.fit_transform(acts)
            title = "PCA of Activations"
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
            embeddings = reducer.fit_transform(acts)
            title = "t-SNE of Activations"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot embeddings
        if labels is not None:
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                               c=labels, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label="Label")
        else:
            ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7)
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add variance explained for PCA
        if method == "pca":
            var_explained = reducer.explained_variance_ratio_
            ax.text(0.02, 0.98, f"Var explained: {var_explained[0]:.2%}, {var_explained[1]:.2%}",
                   transform=ax.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved embeddings to {save_path}")
        
        return fig