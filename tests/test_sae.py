"""Tests for Sparse Autoencoder functionality"""

import pytest
import torch

from src.core.sae import SparseAutoencoder, SAEConfig, SAETrainer, SAEAnalyzer


class TestSparseAutoencoder:
    """Test cases for SAE implementation"""
    
    @pytest.fixture
    def sae_config(self):
        """Create SAE configuration for testing"""
        return SAEConfig(
            n_input_features=64,
            n_learned_features=256,
            l1_coefficient=1e-3,
            learning_rate=1e-3,
            batch_size=16,
            n_epochs=2,
            device="cpu",
        )
    
    @pytest.fixture
    def sae(self, sae_config):
        """Create SAE instance"""
        return SparseAutoencoder(sae_config)
    
    @pytest.fixture
    def sample_activations(self):
        """Create sample activation data"""
        batch_size = 32
        hidden_size = 64
        return torch.randn(batch_size, hidden_size)
    
    def test_sae_initialization(self, sae, sae_config):
        """Test SAE initialization"""
        assert sae.config == sae_config
        assert sae.encoder is not None
        assert sae.decoder is not None
        
        # Check dimensions
        assert sae.encoder.in_features == sae_config.n_input_features
        assert sae.encoder.out_features == sae_config.n_learned_features
    
    def test_sae_forward_pass(self, sae, sample_activations):
        """Test SAE forward pass"""
        reconstructed, features = sae(sample_activations)
        
        # Check shapes
        assert reconstructed.shape == sample_activations.shape
        assert features.shape == (sample_activations.shape[0], sae.config.n_learned_features)
        
        # Check sparsity (features should be sparse due to ReLU)
        sparsity = (features == 0).float().mean()
        assert sparsity > 0.1  # At least some sparsity
    
    def test_sae_loss_computation(self, sae, sample_activations):
        """Test loss computation"""
        reconstructed, features = sae(sample_activations)
        loss, components = sae.compute_loss(sample_activations, reconstructed, features)
        
        assert "total" in components
        assert "reconstruction" in components
        assert "sparsity" in components
        assert components["reconstruction"] >= 0
        assert components["sparsity"] >= 0
    
    def test_sae_training(self, sae, sample_activations):
        """Test SAE training"""
        # Create dataset
        dataset = torch.utils.data.TensorDataset(sample_activations)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Train
        trainer = SAETrainer(sae)
        history = trainer.train(train_loader, n_epochs=2)
        
        assert "total_loss" in history
        assert len(history["total_loss"]) == 2
        
        # Loss should decrease
        assert history["total_loss"][-1] <= history["total_loss"][0]
    
    def test_sae_analyzer(self, sae, sample_activations):
        """Test SAE analyzer functionality"""
        analyzer = SAEAnalyzer(sae)
        
        # Analyze features
        feature_stats = analyzer.analyze_features(sample_activations, top_k=5)
        
        assert "mean_activation" in feature_stats
        assert "activation_frequency" in feature_stats
        assert "top_features" in feature_stats
        assert feature_stats["top_features"].shape == (sample_activations.shape[0], 5)
    
    def test_dead_feature_identification(self, sae, sample_activations):
        """Test dead feature identification"""
        analyzer = SAEAnalyzer(sae)
        
        # Create activations with some dead features
        features = sae.get_feature_activations(sample_activations)
        
        # Manually set some features to zero
        features[:, :10] = 0
        
        dead_features = analyzer.identify_dead_features(features)
        
        # Should identify at least some of the zeroed features as dead
        assert len(dead_features) >= 5
    
    def test_activation_decomposition(self, sae, sample_activations):
        """Test activation decomposition"""
        analyzer = SAEAnalyzer(sae)
        
        # Decompose single activation
        single_activation = sample_activations[0]
        decomp = analyzer.decompose_activation(single_activation)
        
        assert "original" in decomp
        assert "features" in decomp
        assert "reconstructed" in decomp
        assert "residual" in decomp
        assert "reconstruction_error" in decomp
        
        # Check reconstruction
        assert torch.allclose(
            decomp["reconstructed"] + decomp["residual"],
            decomp["original"],
            atol=1e-5
        )