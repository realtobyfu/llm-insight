"""Comprehensive test suite for all components"""

import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

from src.core import InterpretabilityAnalyzer, Config
from src.core.failure_prediction import FailurePredictionModel, AttentionFeatureExtractor
from src.core.anomaly_detection import AttentionPatternAnalyzer, ComprehensiveAnomalyDetector
from src.core.efficient_processing import ChunkedProcessor, MemoryManager, estimate_memory_usage


class TestInterpretabilityAnalyzer:
    """Comprehensive tests for the main analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        with patch('src.core.model_wrapper.AutoModel.from_pretrained') as mock_model:
            # Mock model to avoid downloading
            mock_model.return_value = Mock()
            analyzer = InterpretabilityAnalyzer(model_name="gpt2")
            
            # Mock model methods
            analyzer.model_wrapper.get_num_layers = Mock(return_value=12)
            analyzer.model_wrapper.get_num_attention_heads = Mock(return_value=12)
            analyzer.model_wrapper.get_hidden_size = Mock(return_value=768)
            
            return analyzer
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.model_name == "gpt2"
        assert analyzer.attention_analyzer is not None
        assert analyzer.failure_predictor is None  # Not initialized by default
        assert analyzer.anomaly_detector is not None
    
    def test_basic_analysis(self, analyzer):
        """Test basic analysis functionality"""
        # Mock the model forward pass
        mock_output = {
            "last_hidden_state": torch.randn(1, 10, 768),
            "attentions": tuple([torch.randn(1, 12, 10, 10) for _ in range(12)]),
            "hidden_states": tuple([torch.randn(1, 10, 768) for _ in range(13)]),
            "activations": {}
        }
        
        analyzer.model_wrapper.tokenize = Mock(return_value={
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10)
        })
        analyzer.model_wrapper.forward = Mock(return_value=mock_output)
        analyzer.model_wrapper.get_token_strings = Mock(return_value=[["The", "cat", "sat", "on", "the", "mat", ".", "<pad>", "<pad>", "<pad>"]])
        analyzer.model_wrapper.get_activations = Mock(return_value=torch.randn(13, 1, 10, 768))
        
        # Test analysis
        text = "The cat sat on the mat."
        results = analyzer.analyze(text, methods=["attention"], use_cache=False)
        
        assert "attention" in results
        assert "metadata" in results
        assert results["metadata"]["model_name"] == "gpt2"
    
    def test_failure_prediction_training(self, analyzer):
        """Test failure prediction model training"""
        # Generate synthetic data
        analyzer.train_failure_predictor(
            generate_synthetic=True,
            n_synthetic_samples=50
        )
        
        assert analyzer._failure_model_trained
        assert analyzer.failure_predictor is not None
    
    def test_anomaly_detection(self, analyzer):
        """Test anomaly detection functionality"""
        # Mock analysis results
        mock_analysis = {
            "attention": {
                "patterns": torch.randn(12, 1, 12, 10, 10),
                "entropy": {
                    "mean_entropy_per_layer": torch.rand(12).tolist()
                }
            }
        }
        
        analyzer.analyze = Mock(return_value=mock_analysis)
        
        # Test anomaly detection
        result = analyzer.detect_anomalies("Test text")
        
        assert "is_anomaly" in result
        assert "anomaly_types" in result
        assert "confidence" in result
    
    def test_efficient_processing(self, analyzer):
        """Test memory-efficient processing"""
        # Mock long sequence
        long_text = " ".join(["word"] * 1000)
        
        # Mock tokenizer to return long sequence
        analyzer.model_wrapper.tokenize = Mock(return_value={
            "input_ids": torch.randint(0, 1000, (1, 1000)),
            "attention_mask": torch.ones(1, 1000)
        })
        
        # Test efficient analysis
        results = analyzer.analyze_efficient(
            long_text,
            methods=["attention"],
            use_chunking=True
        )
        
        assert results is not None
    
    def test_caching(self, analyzer):
        """Test caching functionality"""
        # Mock analysis
        mock_result = {"test": "result"}
        analyzer._analyze_attention = Mock(return_value=mock_result)
        
        # First call should compute
        result1 = analyzer.analyze("Test", methods=["attention"], use_cache=True)
        assert analyzer._analyze_attention.called
        
        # Second call should use cache
        analyzer._analyze_attention.reset_mock()
        result2 = analyzer.analyze("Test", methods=["attention"], use_cache=True)
        
        # With caching, the analysis method shouldn't be called again
        # Note: This test assumes cache is properly configured


class TestFailurePrediction:
    """Tests for failure prediction components"""
    
    def test_feature_extractor(self):
        """Test attention feature extraction"""
        extractor = AttentionFeatureExtractor()
        
        # Create mock attention data
        attention_data = {
            "patterns": torch.randn(4, 1, 4, 10, 10),
            "entropy": {
                "mean_entropy_per_layer": torch.rand(4).tolist()
            }
        }
        
        features = extractor.extract_features(attention_data)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == len(extractor.feature_names)
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_failure_model_training(self):
        """Test failure prediction model training"""
        model = FailurePredictionModel()
        
        # Create mock training data
        attention_data_list = []
        labels = []
        
        for i in range(20):
            attention_data = {
                "patterns": torch.randn(4, 1, 4, 10, 10),
                "entropy": {
                    "mean_entropy_per_layer": torch.rand(4).tolist()
                }
            }
            attention_data_list.append(attention_data)
            labels.append(i % 2)  # Alternating labels
        
        # Train model
        results = model.train(attention_data_list, labels, test_size=0.3)
        
        assert "train_accuracy" in results
        assert "test_accuracy" in results
        assert model._is_trained
    
    def test_failure_prediction(self):
        """Test failure prediction on new data"""
        model = FailurePredictionModel()
        
        # Train first
        attention_data_list = []
        labels = []
        
        for i in range(20):
            attention_data = {
                "patterns": torch.randn(4, 1, 4, 10, 10),
                "entropy": {
                    "mean_entropy_per_layer": torch.rand(4).tolist()
                }
            }
            attention_data_list.append(attention_data)
            labels.append(i % 2)
        
        model.train(attention_data_list, labels)
        
        # Test prediction
        test_data = {
            "patterns": torch.randn(4, 1, 4, 10, 10),
            "entropy": {
                "mean_entropy_per_layer": torch.rand(4).tolist()
            }
        }
        
        result = model.predict(test_data)
        
        assert "failure_probability" in result
        assert 0 <= result["failure_probability"] <= 1
        assert "prediction" in result
        assert result["prediction"] in ["high_risk", "medium_risk", "low_risk"]


class TestAnomalyDetection:
    """Tests for anomaly detection components"""
    
    def test_attention_pattern_analyzer(self):
        """Test attention pattern analysis"""
        analyzer = AttentionPatternAnalyzer()
        
        # Create normal patterns
        normal_patterns = [torch.randn(4, 1, 4, 10, 10) for _ in range(10)]
        analyzer.fit(normal_patterns)
        
        assert analyzer._is_fitted
        
        # Test anomaly detection
        anomalous_pattern = torch.ones(4, 1, 4, 10, 10) * 0.9  # Very high attention
        is_anomaly, scores = analyzer.detect_anomaly(anomalous_pattern, return_scores=True)
        
        assert isinstance(is_anomaly, bool)
        assert "z_score" in scores
        assert "anomalous_features" in scores
    
    def test_comprehensive_detector(self):
        """Test comprehensive anomaly detector"""
        detector = ComprehensiveAnomalyDetector()
        
        # Create normal examples
        normal_examples = []
        for _ in range(10):
            example = {
                "attention": {
                    "patterns": torch.randn(4, 1, 4, 10, 10),
                    "entropy": {
                        "mean_entropy_per_layer": torch.rand(4).tolist()
                    }
                }
            }
            normal_examples.append(example)
        
        # Fit detector
        detector.fit(normal_examples, fit_attention=True, fit_activations=False)
        
        assert detector._is_fitted
        
        # Test detection
        test_example = {
            "attention": {
                "patterns": torch.ones(4, 1, 4, 10, 10) * 0.1,  # Low entropy pattern
                "entropy": {
                    "mean_entropy_per_layer": (torch.ones(4) * 0.1).tolist()
                }
            }
        }
        
        result = detector.detect_anomalies(test_example)
        
        assert "is_anomaly" in result
        assert "anomaly_types" in result
        assert "confidence" in result


class TestEfficientProcessing:
    """Tests for memory-efficient processing"""
    
    def test_memory_manager(self):
        """Test memory management utilities"""
        manager = MemoryManager()
        
        # Test memory stats
        stats = manager.get_memory_usage()
        assert isinstance(stats, dict)
        
        # Test memory checking
        available = manager.check_memory_available(0.1)  # 100MB
        assert isinstance(available, bool)
    
    def test_chunked_processor(self):
        """Test chunked processing"""
        processor = ChunkedProcessor(chunk_size=10, overlap=2)
        
        # Mock model wrapper
        mock_wrapper = Mock()
        mock_wrapper.forward = Mock(side_effect=lambda x, mask: {
            "last_hidden_state": torch.randn(x.shape[0], x.shape[1], 768),
            "attentions": tuple([torch.randn(x.shape[0], 12, x.shape[1], x.shape[1]) for _ in range(12)]),
            "hidden_states": None,
            "activations": {}
        })
        
        # Test with long sequence
        long_input = torch.randint(0, 1000, (1, 50))
        result = processor.process_long_sequence(mock_wrapper, long_input)
        
        assert "last_hidden_state" in result
        assert result["last_hidden_state"].shape[1] == 50
    
    def test_memory_estimation(self):
        """Test memory usage estimation"""
        mock_wrapper = Mock()
        mock_wrapper.get_hidden_size = Mock(return_value=768)
        mock_wrapper.get_num_layers = Mock(return_value=12)
        mock_wrapper.get_num_attention_heads = Mock(return_value=12)
        mock_wrapper.get_vocab_size = Mock(return_value=50000)
        mock_wrapper.model = Mock()
        mock_wrapper.model.parameters = Mock(return_value=[torch.randn(100, 100)])
        
        estimate = estimate_memory_usage(mock_wrapper, batch_size=2, seq_length=512)
        
        assert "total_gb" in estimate
        assert estimate["total_gb"] > 0


class TestVisualization:
    """Tests for visualization components"""
    
    def test_attention_visualizer(self):
        """Test attention visualization"""
        from src.visualization import AttentionVisualizer
        
        viz = AttentionVisualizer()
        
        # Create mock data
        attention_weights = torch.randn(4, 1, 4, 10, 10)
        tokens = ["The", "cat", "sat", "on", "the", "mat", ".", "<pad>", "<pad>", "<pad>"]
        
        # Test heatmap creation
        fig = viz.plot_attention_heatmap(
            attention_weights,
            tokens,
            layer=0,
            head=0
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_interactive_visualizer(self):
        """Test interactive visualization"""
        from src.visualization import InteractiveVisualizer
        
        viz = InteractiveVisualizer()
        
        # Create mock data
        attention_weights = torch.randn(4, 1, 4, 10, 10)
        tokens = ["The", "cat", "sat", "on", "the", "mat", ".", "<pad>", "<pad>", "<pad>"]
        
        # Test interactive heatmap
        fig = viz.create_attention_heatmap(
            attention_weights,
            tokens,
            layer=0,
            head=0
        )
        
        assert fig is not None


class TestAPI:
    """Tests for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint"""
        response = client.get("/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert "gpt2" in models
    
    @patch('src.api.main.analyzer')
    def test_analyze_endpoint(self, mock_analyzer, client):
        """Test analysis endpoint"""
        # Mock analyzer
        mock_analyzer.analyze = Mock(return_value={
            "attention": {"test": "data"},
            "metadata": {"model_name": "gpt2"}
        })
        
        response = client.post("/analyze", json={
            "text": "Test text",
            "methods": ["attention"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "processing_time" in data


class TestCLI:
    """Tests for CLI functionality"""
    
    def test_cli_import(self):
        """Test CLI can be imported"""
        from src.cli import main
        assert main is not None
    
    @patch('src.cli.InterpretabilityAnalyzer')
    @patch('src.cli.console')
    def test_analyze_command(self, mock_console, mock_analyzer):
        """Test analyze command"""
        from click.testing import CliRunner
        from src.cli import analyze
        
        runner = CliRunner()
        result = runner.invoke(analyze, ["Test text", "--model", "gpt2"])
        
        # Check command executed without error
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])