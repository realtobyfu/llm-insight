"""Tests for the InterpretabilityAnalyzer"""

import pytest
import torch

from src.core import InterpretabilityAnalyzer


class TestInterpretabilityAnalyzer:
    """Test cases for InterpretabilityAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        # Use small model for faster tests
        return InterpretabilityAnalyzer(model_name="distilgpt2")
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.model_name == "distilgpt2"
        assert analyzer.model_wrapper is not None
        assert analyzer.attention_analyzer is not None
    
    def test_analyze_basic(self, analyzer):
        """Test basic analysis functionality"""
        text = "The cat sat on the mat"
        results = analyzer.analyze(text, methods=["attention"])
        
        assert "attention" in results
        assert "metadata" in results
        assert results["metadata"]["model_name"] == "distilgpt2"
    
    def test_analyze_multiple_methods(self, analyzer):
        """Test analysis with multiple methods"""
        text = "The quick brown fox jumps over the lazy dog"
        results = analyzer.analyze(
            text,
            methods=["attention", "importance", "head_patterns"]
        )
        
        assert "attention" in results
        assert "importance" in results
        assert "head_patterns" in results
        assert "metadata" in results
    
    def test_analyze_batch(self, analyzer):
        """Test analysis with batch of texts"""
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence"
        ]
        results = analyzer.analyze(texts, methods=["attention"])
        
        assert "attention" in results
        assert results["attention"]["shape"]["batch_size"] == 3
    
    def test_failure_prediction(self, analyzer):
        """Test failure prediction functionality"""
        text = "This is a test sentence for failure prediction"
        result = analyzer.predict_failure_probability(text)
        
        assert "failure_probability" in result
        assert "prediction" in result
        assert "indicators" in result
        assert "explanation" in result
        assert 0 <= result["failure_probability"] <= 1
    
    def test_token_importance(self, analyzer):
        """Test token importance analysis"""
        text = "The important word is highlighted"
        results = analyzer.analyze(text, methods=["importance"])
        
        assert "importance" in results
        assert "token_importance" in results["importance"]
        assert "head_importance" in results["importance"]
        
        token_imp = results["importance"]["token_importance"]
        assert "tokens" in token_imp
        assert "importance_mean" in token_imp
    
    def test_attention_patterns(self, analyzer):
        """Test attention pattern extraction"""
        text = "Attention patterns are being analyzed"
        results = analyzer.analyze(text, methods=["attention"])
        
        attention_data = results["attention"]
        assert "patterns" in attention_data
        assert "tokens" in attention_data
        assert "shape" in attention_data
        assert "entropy" in attention_data
        
        # Check shape information
        shape = attention_data["shape"]
        assert shape["num_layers"] > 0
        assert shape["num_heads"] > 0
        assert shape["seq_length"] > 0
    
    def test_head_patterns_detection(self, analyzer):
        """Test attention head pattern detection"""
        text = "The cat sat on the mat. The cat sat on the mat."
        results = analyzer.analyze(text, methods=["head_patterns"])
        
        patterns = results["head_patterns"]
        assert "identified_patterns" in patterns
        assert "pattern_summary" in patterns
        
        # Should identify at least some patterns
        assert len(patterns["identified_patterns"]) > 0
    
    def test_activations_extraction(self, analyzer):
        """Test activation extraction"""
        text = "Extract model activations"
        results = analyzer.analyze(text, methods=["activations"])
        
        assert "activations" in results
        assert "statistics" in results["activations"]
        assert "tokens" in results["activations"]
        
        stats = results["activations"]["statistics"]
        for layer_stats in stats.values():
            assert "mean" in layer_stats
            assert "std" in layer_stats
            assert "shape" in layer_stats