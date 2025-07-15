"""Sparse Autoencoder implementation for feature extraction"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from ..utils.logger import get_logger, TimedLogger

logger = get_logger(__name__)


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder"""
    
    n_input_features: int
    n_learned_features: int
    l1_coefficient: float = 1e-3
    learning_rate: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    normalize_decoder: bool = True
    tied_weights: bool = False
    activation: str = "relu"  # relu, gelu, or silu


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for learning interpretable features from activations
    
    Based on Anthropic's approach in "Scaling Monosemanticity" but optimized
    for practical use with smaller models.
    """
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        
        # Encoder: maps activations to sparse features
        self.encoder = nn.Linear(
            config.n_input_features,
            config.n_learned_features,
            bias=True
        )
        
        # Decoder: reconstructs activations from sparse features
        if config.tied_weights:
            # Use transposed encoder weights
            self.decoder = None
        else:
            self.decoder = nn.Linear(
                config.n_learned_features,
                config.n_input_features,
                bias=True
            )
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(config.device)
        
        # Activation function
        self.activation = self._get_activation(config.activation)
        
        logger.info(
            f"Initialized SAE with {config.n_input_features} input features "
            f"and {config.n_learned_features} learned features"
        )
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        if self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.ReLU())
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features"""
        features = self.encoder(x)
        features = self.activation(features)
        return features
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features to reconstructed activations"""
        if self.config.tied_weights:
            # Use transposed encoder weights
            reconstructed = F.linear(features, self.encoder.weight.t(), self.encoder.bias)
        else:
            reconstructed = self.decoder(features)
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder
        
        Args:
            x: Input activations
        
        Returns:
            Tuple of (reconstructed activations, sparse features)
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features
    
    def normalize_decoder_weights(self):
        """Normalize decoder weights to unit norm"""
        if self.config.normalize_decoder and self.decoder is not None:
            with torch.no_grad():
                self.decoder.weight.data = F.normalize(
                    self.decoder.weight.data,
                    p=2,
                    dim=1
                )
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute reconstruction loss with L1 sparsity penalty
        
        Args:
            x: Original activations
            reconstructed: Reconstructed activations
            features: Sparse features
        
        Returns:
            Tuple of (total loss, loss components dict)
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, x)
        
        # Sparsity loss (L1 penalty on features)
        sparsity_loss = features.abs().mean()
        
        # Total loss
        total_loss = reconstruction_loss + self.config.l1_coefficient * sparsity_loss
        
        # Return components for logging
        components = {
            "total": total_loss.item(),
            "reconstruction": reconstruction_loss.item(),
            "sparsity": sparsity_loss.item(),
            "l1_coefficient": self.config.l1_coefficient,
        }
        
        return total_loss, components
    
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations for input"""
        with torch.no_grad():
            features = self.encode(x)
        return features
    
    def get_feature_importance(self, features: torch.Tensor) -> torch.Tensor:
        """Compute importance scores for features based on activation frequency"""
        # Average activation across batch
        importance = features.mean(dim=0)
        return importance
    
    def get_most_active_features(
        self,
        x: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the most active features for given input
        
        Args:
            x: Input activations
            top_k: Number of top features to return
        
        Returns:
            Tuple of (feature indices, activation values)
        """
        features = self.get_feature_activations(x)
        
        # Get top-k features for each sample
        values, indices = torch.topk(features, k=top_k, dim=-1)
        
        return indices, values


class SAETrainer:
    """Trainer for Sparse Autoencoder"""
    
    def __init__(
        self,
        sae: SparseAutoencoder,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        self.sae = sae
        self.optimizer = optimizer or Adam(
            sae.parameters(),
            lr=sae.config.learning_rate
        )
        self.training_history = {
            "total_loss": [],
            "reconstruction_loss": [],
            "sparsity_loss": [],
        }
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.sae.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        reconstructed, features = self.sae(batch)
        
        # Compute loss
        loss, components = self.sae.compute_loss(batch, reconstructed, features)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Normalize decoder weights if configured
        self.sae.normalize_decoder_weights()
        
        return components
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        n_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """Train the SAE
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            n_epochs: Number of epochs (overrides config if provided)
        
        Returns:
            Training history dictionary
        """
        n_epochs = n_epochs or self.sae.config.n_epochs
        
        logger.info(f"Starting SAE training for {n_epochs} epochs")
        
        for epoch in range(n_epochs):
            epoch_losses = {
                "total": 0.0,
                "reconstruction": 0.0,
                "sparsity": 0.0,
            }
            n_batches = 0
            
            # Training loop
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}") as pbar:
                for batch in pbar:
                    # Move batch to device
                    if isinstance(batch, tuple):
                        batch = batch[0]
                    batch = batch.to(self.sae.config.device)
                    
                    # Training step
                    loss_components = self.train_step(batch)
                    
                    # Accumulate losses
                    for key in epoch_losses:
                        if key in loss_components:
                            epoch_losses[key] += loss_components[key]
                    n_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        "loss": f"{loss_components['total']:.4f}",
                        "recon": f"{loss_components['reconstruction']:.4f}",
                        "sparse": f"{loss_components['sparsity']:.4f}",
                    })
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= n_batches
                self.training_history.setdefault(f"{key}_loss", []).append(
                    epoch_losses[key]
                )
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.training_history.setdefault("val_loss", []).append(val_loss)
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train Loss: {epoch_losses['total']:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train Loss: {epoch_losses['total']:.4f}"
                )
        
        logger.info("SAE training completed")
        return self.training_history
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Validate the SAE"""
        self.sae.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, tuple):
                    batch = batch[0]
                batch = batch.to(self.sae.config.device)
                
                reconstructed, features = self.sae(batch)
                loss, _ = self.sae.compute_loss(batch, reconstructed, features)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches


class SAEAnalyzer:
    """Analyzer for interpreting SAE features"""
    
    def __init__(self, sae: SparseAutoencoder):
        self.sae = sae
    
    def analyze_features(
        self,
        activations: torch.Tensor,
        top_k: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Analyze which features are active for given activations"""
        # Get feature activations
        features = self.sae.get_feature_activations(activations)
        
        # Get most active features
        top_indices, top_values = self.sae.get_most_active_features(
            activations, top_k=top_k
        )
        
        # Compute feature statistics
        feature_stats = {
            "mean_activation": features.mean(dim=0),
            "max_activation": features.max(dim=0)[0],
            "activation_frequency": (features > 0).float().mean(dim=0),
            "top_features": top_indices,
            "top_values": top_values,
        }
        
        return feature_stats
    
    def compute_feature_correlation(
        self,
        activations: torch.Tensor
    ) -> torch.Tensor:
        """Compute correlation matrix between features"""
        features = self.sae.get_feature_activations(activations)
        
        # Normalize features
        features_normalized = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
        
        # Compute correlation matrix
        correlation = torch.matmul(features_normalized.t(), features_normalized) / features.shape[0]
        
        return correlation
    
    def identify_dead_features(
        self,
        activations: torch.Tensor,
        threshold: float = 0.01
    ) -> List[int]:
        """Identify features that rarely activate"""
        features = self.sae.get_feature_activations(activations)
        activation_frequency = (features > threshold).float().mean(dim=0)
        
        dead_features = torch.where(activation_frequency < 0.01)[0].tolist()
        
        return dead_features
    
    def decompose_activation(
        self,
        activation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Decompose an activation vector into SAE features"""
        # Get features
        features = self.sae.get_feature_activations(activation)
        
        # Reconstruct
        reconstructed = self.sae.decode(features)
        
        # Compute residual
        residual = activation - reconstructed
        
        return {
            "original": activation,
            "features": features,
            "reconstructed": reconstructed,
            "residual": residual,
            "reconstruction_error": F.mse_loss(reconstructed, activation),
        }