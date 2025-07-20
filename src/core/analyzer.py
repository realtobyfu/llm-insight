"""Main interpretability analyzer combining all analysis methods"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import time

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import torch.utils.data

from .attention import AttentionAnalyzer
from .config import Config
from .model_wrapper import ModelWrapper
from .sae import SparseAutoencoder, SAEConfig, SAETrainer, SAEAnalyzer
from .failure_prediction import FailurePredictionModel, FailureDataCollector
from .anomaly_detection import ComprehensiveAnomalyDetector
from .efficient_processing import ChunkedProcessor, MemoryManager, StreamingAnalyzer, estimate_memory_usage
from ..utils.logger import get_logger, TimedLogger
from ..utils.cache import CacheManager, cached_analysis
from ..visualization import AttentionVisualizer, FeatureVisualizer, InteractiveVisualizer
from ..monitoring import AnalysisMonitor

logger = get_logger(__name__)


class InterpretabilityAnalyzer:
    """Main class for LLM interpretability analysis"""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        config: Optional[Config] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_monitoring: bool = True,
    ):
        """Initialize the interpretability analyzer
        
        Args:
            model_name: Name of the model to analyze
            config: Configuration object
            device: Device to run on (cuda/cpu)
            cache_dir: Directory for caching models
            enable_monitoring: Whether to enable analysis monitoring
        """
        self.config = config or Config.from_env()
        self.model_name = model_name
        self.device = device or self.config.model.device
        self.cache_dir = cache_dir or str(self.config.models_dir)
        self.enable_monitoring = enable_monitoring
        
        # Initialize model wrapper
        logger.info(f"Initializing InterpretabilityAnalyzer with model: {model_name}")
        self.model_wrapper = ModelWrapper.create(
            model_name=model_name,
            device=self.device,
            cache_dir=self.cache_dir,
        )
        
        # Initialize analyzers
        self.attention_analyzer = AttentionAnalyzer(self.model_wrapper)
        
        # Initialize SAE (will be trained on demand)
        self.sae = None
        self.sae_analyzer = None
        self._sae_trained = False
        
        # Initialize visualizers
        self.attention_visualizer = AttentionVisualizer()
        self.feature_visualizer = FeatureVisualizer()
        self.interactive_visualizer = InteractiveVisualizer()
        
        # Initialize failure prediction
        self.failure_predictor = None
        self.failure_collector = FailureDataCollector()
        self._failure_model_trained = False
        
        # Initialize anomaly detection
        self.anomaly_detector = ComprehensiveAnomalyDetector()
        self._anomaly_detector_fitted = False
        
        # Initialize efficient processing
        self.memory_manager = MemoryManager()
        self.chunked_processor = ChunkedProcessor()
        self.streaming_analyzer = StreamingAnalyzer(self)
        
        # Cache for analysis results
        self.cache_manager = CacheManager(self.config.cache)
        self._cache = {}
        
        # Initialize monitoring
        self.monitor = AnalysisMonitor(enable_alerts=True) if enable_monitoring else None
        
        logger.info("InterpretabilityAnalyzer initialized successfully")
    
    def analyze(
        self,
        text: Union[str, List[str]],
        methods: Optional[List[str]] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis on input text
        
        Args:
            text: Input text or list of texts
            methods: List of analysis methods to apply
                - "attention": Attention pattern analysis
                - "importance": Token importance scoring  
                - "activations": Hidden state analysis
                - "head_patterns": Attention head pattern detection
                - "sae": Sparse autoencoder feature analysis
            **kwargs: Additional arguments for specific methods
        
        Returns:
            Dictionary containing results from all requested analyses
        """
        if methods is None:
            methods = ["attention", "importance"]
        
        # Start monitoring if enabled
        analysis_id = f"{self.model_name}_{int(time.time() * 1000)}"
        metrics = None
        if self.monitor:
            input_text = text if isinstance(text, str) else " ".join(text)
            metrics = self.monitor.start_analysis(
                analysis_id=analysis_id,
                model_name=self.model_name,
                methods=methods,
                input_text=input_text
            )
        
        # Check cache if enabled
        if use_cache:
            cache_key = self.cache_manager._generate_key(
                "analysis",
                {
                    "text": text[:100] if isinstance(text, str) else str(text[:3]),  # Truncate for key
                    "methods": methods,
                    "model": self.model_name,
                    "kwargs": str(kwargs),
                }
            )
            
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.info("Returning cached analysis result")
                return cached_result
        
        results = {}
        error = None
        
        try:
            with TimedLogger(logger, f"Analyzing text with methods: {methods}"):
                
                if "attention" in methods:
                    with TimedLogger(logger, "Extracting attention patterns"):
                        results["attention"] = self._analyze_attention(text, **kwargs)
                
                if "importance" in methods:
                    with TimedLogger(logger, "Computing token importance"):
                        results["importance"] = self._analyze_importance(text, **kwargs)
                
                if "activations" in methods:
                    with TimedLogger(logger, "Extracting activations"):
                        results["activations"] = self._analyze_activations(text, **kwargs)
                
                if "head_patterns" in methods:
                    with TimedLogger(logger, "Detecting head patterns"):
                        results["head_patterns"] = self._analyze_head_patterns(text, **kwargs)
                
                if "sae" in methods:
                    with TimedLogger(logger, "Analyzing SAE features"):
                        results["sae"] = self._analyze_sae_features(text, **kwargs)
        except Exception as e:
            error = str(e)
            logger.error(f"Analysis failed: {error}")
            # Complete monitoring with error
            if self.monitor and metrics:
                input_text = text if isinstance(text, str) else " ".join(text)
                self.monitor.complete_analysis(metrics, None, error, input_text)
            raise
        
        # Add metadata
        results["metadata"] = {
            "model_name": self.model_name,
            "num_layers": self.model_wrapper.get_num_layers(),
            "num_heads": self.model_wrapper.get_num_attention_heads(),
            "hidden_size": self.model_wrapper.get_hidden_size(),
            "device": str(self.device),
        }
        
        # Cache results if enabled
        if use_cache and not error:
            self.cache_manager.set(cache_key, results)
            logger.debug("Cached analysis results")
        
        # Complete monitoring if enabled
        if self.monitor and metrics:
            input_text = text if isinstance(text, str) else " ".join(text)
            self.monitor.complete_analysis(metrics, results, error, input_text)
        
        return results
    
    def _analyze_attention(
        self,
        text: Union[str, List[str]],
        layer: Optional[int] = None,
        head: Optional[int] = None,
        compute_rollout: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze attention patterns"""
        # Extract attention patterns
        patterns = self.attention_analyzer.extract_attention_patterns(
            text, layer=layer, head=head
        )
        
        results = {
            "patterns": patterns["attention_weights"],
            "tokens": patterns["tokens"],
            "shape": patterns["shape"],
        }
        
        # Compute attention rollout if requested
        if compute_rollout:
            rollout = self.attention_analyzer.compute_attention_rollout(
                patterns["attention_weights"]
            )
            results["rollout"] = rollout
        
        # Compute entropy
        entropy_stats = self.attention_analyzer.compute_attention_entropy(
            patterns["attention_weights"]
        )
        results["entropy"] = entropy_stats
        
        return results
    
    def _analyze_importance(
        self,
        text: Union[str, List[str]],
        target_token_idx: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze token importance"""
        if isinstance(text, list):
            # Use first text for importance analysis
            text = text[0]
        
        # Analyze token importance
        importance = self.attention_analyzer.analyze_token_importance(
            text, target_token_idx=target_token_idx
        )
        
        # Compute head importance
        head_importance = self.attention_analyzer.compute_head_importance(
            text, metric=kwargs.get("importance_metric", "gradient")
        )
        
        return {
            "token_importance": importance,
            "head_importance": head_importance,
        }
    
    def _analyze_activations(
        self,
        text: Union[str, List[str]],
        layers: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze model activations"""
        # Tokenize and run forward pass
        inputs = self.model_wrapper.tokenize(text)
        outputs = self.model_wrapper.forward(**inputs)
        
        # Extract activations
        all_activations = self.model_wrapper.get_activations()
        
        if layers is not None:
            # Filter to specific layers
            activations = {}
            for layer in layers:
                if 0 <= layer < all_activations.shape[0]:
                    activations[f"layer_{layer}"] = all_activations[layer]
        else:
            activations = {"all_layers": all_activations}
        
        # Compute activation statistics
        stats = {}
        for name, acts in activations.items():
            stats[name] = {
                "mean": acts.mean().item(),
                "std": acts.std().item(),
                "min": acts.min().item(),
                "max": acts.max().item(),
                "shape": list(acts.shape),
            }
        
        return {
            "activations": activations,
            "statistics": stats,
            "tokens": self.model_wrapper.get_token_strings(inputs["input_ids"]),
        }
    
    def _analyze_head_patterns(
        self,
        text: Union[str, List[str]],
        pattern_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze attention head patterns"""
        if pattern_types is None:
            pattern_types = ["diagonal", "previous", "first", "last"]
        
        # Get attention patterns
        patterns = self.attention_analyzer.extract_attention_patterns(text)
        attention_weights = patterns["attention_weights"]
        
        # Identify patterns
        head_patterns = {}
        for pattern_type in pattern_types:
            identified = self.attention_analyzer.identify_attention_heads_by_pattern(
                attention_weights, pattern_type=pattern_type
            )
            head_patterns.update(identified)
        
        # Find induction heads if text has repetition
        if isinstance(text, str) and len(text.split()) > 10:
            induction_heads = self.attention_analyzer.find_induction_heads(
                text, threshold=kwargs.get("induction_threshold", 0.7)
            )
            if induction_heads:
                head_patterns["induction"] = induction_heads
        
        return {
            "identified_patterns": head_patterns,
            "pattern_summary": {
                pattern: len(heads) for pattern, heads in head_patterns.items()
            },
        }
    
    def predict_failure_probability(
        self,
        text: Union[str, List[str]],
        threshold: float = 0.5,
        collect_example: bool = False,
        is_failure: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Predict probability of model failure based on attention patterns
        
        Args:
            text: Input text to analyze
            threshold: Threshold for failure prediction
            collect_example: Whether to collect this example for training
            is_failure: Label for collected example (if collect_example=True)
        
        Returns:
            Dictionary with failure probability and indicators
        """
        # Analyze the text
        analysis = self.analyze(
            text,
            methods=["attention"],
            use_cache=False,  # Don't cache for failure prediction
        )
        
        # Use trained model if available
        if self._failure_model_trained and self.failure_predictor is not None:
            try:
                # Use trained model
                result = self.failure_predictor.predict(analysis["attention"])
                
                # Add explanation
                result["explanation"] = self._generate_failure_explanation(result["indicators"])
                
                # Collect example if requested
                if collect_example and is_failure is not None:
                    self.failure_collector.add_example(
                        text=text if isinstance(text, str) else text[0],
                        attention_data=analysis["attention"],
                        is_failure=is_failure
                    )
                
                return result
                
            except Exception as e:
                logger.warning(f"Trained model prediction failed, falling back to heuristic: {e}")
        
        # Fallback to heuristic-based prediction
        attention_data = analysis["attention"]
        
        # Extract indicators of potential failure
        indicators = []
        
        # Check for low attention entropy (over-focused attention)
        mean_entropy = torch.tensor(attention_data["entropy"]["mean_entropy_per_layer"]).mean()
        if mean_entropy < 0.5:
            indicators.append("low_entropy")
        
        # Check for attention collapse
        attention_weights = torch.tensor(attention_data["patterns"])
        
        # Simplified check: see if many heads have very peaked attention
        peaked_heads = 0
        total_heads = attention_weights.shape[0] * attention_weights.shape[2]
        
        for layer in range(attention_weights.shape[0]):
            for head in range(attention_weights.shape[2]):
                att = attention_weights[layer, 0, head]  # First item in batch
                max_attention = att.max(dim=-1)[0].mean()
                if max_attention > 0.9:  # Very peaked attention
                    peaked_heads += 1
        
        peaked_ratio = peaked_heads / total_heads
        if peaked_ratio > 0.5:
            indicators.append("attention_collapse")
        
        # Check for uniform attention (no clear focus)
        if mean_entropy > 2.0:
            indicators.append("uniform_attention")
        
        # Simple scoring based on indicators
        failure_score = len(indicators) * 0.3
        failure_probability = min(failure_score, 1.0)
        
        result = {
            "failure_probability": failure_probability,
            "prediction": "high_risk" if failure_probability > threshold else "low_risk",
            "indicators": indicators,
            "confidence": 0.5,  # Lower confidence for heuristic
            "explanation": self._generate_failure_explanation(indicators),
        }
        
        # Collect example if requested
        if collect_example and is_failure is not None:
            self.failure_collector.add_example(
                text=text if isinstance(text, str) else text[0],
                attention_data=attention_data,
                is_failure=is_failure
            )
        
        return result
    
    def _generate_failure_explanation(self, indicators: List[str]) -> str:
        """Generate human-readable explanation for failure prediction"""
        explanations = {
            "low_entropy": "Model attention is overly focused on specific tokens",
            "attention_collapse": "Many attention heads are collapsing to single positions",
            "uniform_attention": "Model attention is too dispersed without clear focus",
        }
        
        if not indicators:
            return "No significant failure indicators detected."
        
        explanation_parts = [explanations.get(ind, ind) for ind in indicators]
        return "Potential issues detected: " + "; ".join(explanation_parts)
    
    def visualize_attention(
        self,
        attention_data: Dict[str, Any],
        save_path: Optional[str] = None,
        interactive: bool = False,
        **kwargs
    ) -> Union[plt.Figure, go.Figure]:
        """Create attention visualization
        
        Args:
            attention_data: Attention analysis results
            save_path: Path to save visualization
            interactive: Whether to create interactive plot
            **kwargs: Additional visualization parameters
        
        Returns:
            Matplotlib or Plotly figure
        """
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        
        attention_weights = torch.tensor(attention_data["patterns"])
        tokens = attention_data["tokens"][0]  # First sequence
        
        if interactive:
            # Create interactive visualization
            fig = self.interactive_visualizer.create_attention_heatmap(
                attention_weights,
                tokens,
                layer=kwargs.get("layer", 0),
                head=kwargs.get("head", 0)
            )
            
            if save_path:
                self.interactive_visualizer.save_figure(fig, save_path)
            
            return fig
        else:
            # Create static visualization
            fig = self.attention_visualizer.plot_attention_heatmap(
                attention_weights,
                tokens,
                layer=kwargs.get("layer", 0),
                head=kwargs.get("head", 0),
                save_path=save_path,
                **kwargs
            )
            
            return fig
    
    def visualize_token_importance(
        self,
        importance_data: Dict[str, Any],
        save_path: Optional[str] = None,
        interactive: bool = False,
        **kwargs
    ) -> Union[plt.Figure, go.Figure]:
        """Create token importance visualization
        
        Args:
            importance_data: Token importance analysis results
            save_path: Path to save visualization
            interactive: Whether to create interactive plot
            **kwargs: Additional visualization parameters
        
        Returns:
            Matplotlib or Plotly figure
        """
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        
        tokens = importance_data["token_importance"]["tokens"]
        scores = torch.tensor(importance_data["token_importance"]["importance_mean"])
        
        if interactive:
            fig = self.interactive_visualizer.create_token_importance_bar(
                tokens, scores, title="Token Importance"
            )
            
            if save_path:
                self.interactive_visualizer.save_figure(fig, save_path)
            
            return fig
        else:
            fig = self.attention_visualizer.plot_token_importance(
                tokens, scores, save_path=save_path, **kwargs
            )
            
            return fig
    
    def visualize_sae_features(
        self,
        sae_data: Dict[str, Any],
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """Create SAE feature visualization
        
        Args:
            sae_data: SAE analysis results
            save_path: Path to save visualization
            **kwargs: Additional visualization parameters
        
        Returns:
            Matplotlib figure
        """
        # Convert feature stats to tensors
        mean_activation = torch.tensor(sae_data["feature_stats"]["mean_activation"])
        activation_freq = torch.tensor(sae_data["feature_stats"]["activation_frequency"])
        
        # Create visualization
        fig = self.feature_visualizer.plot_feature_importance(
            mean_activation,
            top_k=kwargs.get("top_k", 20),
            save_path=save_path
        )
        
        return fig
    
    def _initialize_sae(self, layer: int = -1) -> None:
        """Initialize SAE for specified layer"""
        hidden_size = self.model_wrapper.get_hidden_size()
        
        # Configure SAE
        sae_config = SAEConfig(
            n_input_features=hidden_size,
            n_learned_features=hidden_size * 8,  # 8x expansion factor
            l1_coefficient=3e-4,
            learning_rate=1e-3,
            device=self.device,
        )
        
        self.sae = SparseAutoencoder(sae_config)
        self.sae_analyzer = SAEAnalyzer(self.sae)
        self.sae_layer = layer
        
        logger.info(f"Initialized SAE for layer {layer} with {sae_config.n_learned_features} features")
    
    def train_sae(
        self,
        training_texts: List[str],
        layer: int = -1,
        n_epochs: int = 5,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """Train SAE on activations from specified layer
        
        Args:
            training_texts: List of texts to train on
            layer: Layer to extract activations from (-1 for last layer)
            n_epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training history
        """
        if self.sae is None or self.sae_layer != layer:
            self._initialize_sae(layer)
        
        logger.info(f"Training SAE on {len(training_texts)} texts")
        
        # Collect activations
        all_activations = []
        
        for i in range(0, len(training_texts), batch_size):
            batch_texts = training_texts[i:i + batch_size]
            inputs = self.model_wrapper.tokenize(batch_texts)
            
            # Forward pass to get activations
            with torch.no_grad():
                outputs = self.model_wrapper.forward(**inputs)
                activations = self.model_wrapper.get_activations(layer)
                
                # Flatten batch and sequence dimensions
                # Shape: (batch_size * seq_len, hidden_size)
                batch_size_actual = activations.shape[0]
                seq_len = activations.shape[1]
                hidden_size = activations.shape[2]
                
                flat_activations = activations.view(-1, hidden_size)
                all_activations.append(flat_activations)
        
        # Concatenate all activations
        all_activations = torch.cat(all_activations, dim=0)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(all_activations)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        
        # Train SAE
        trainer = SAETrainer(self.sae)
        history = trainer.train(train_loader, n_epochs=n_epochs)
        
        self._sae_trained = True
        logger.info("SAE training completed")
        
        return history
    
    def _analyze_sae_features(
        self,
        text: Union[str, List[str]],
        layer: int = -1,
        top_k: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze text using trained SAE features"""
        # Check if SAE is trained
        if not self._sae_trained:
            # Auto-train on a small dataset if not already trained
            logger.warning("SAE not trained. Training on sample data...")
            sample_texts = [
                "The cat sat on the mat.",
                "The dog ran in the park.",
                "Machine learning is fascinating.",
                "Natural language processing enables many applications.",
                "Artificial intelligence is transforming the world.",
            ] * 10  # Repeat for more training data
            
            self.train_sae(sample_texts, layer=layer, n_epochs=3)
        
        # Get activations for input text
        inputs = self.model_wrapper.tokenize(text)
        outputs = self.model_wrapper.forward(**inputs)
        activations = self.model_wrapper.get_activations(layer)
        
        # Flatten activations
        batch_size = activations.shape[0]
        seq_len = activations.shape[1]
        hidden_size = activations.shape[2]
        flat_activations = activations.view(-1, hidden_size)
        
        # Analyze with SAE
        feature_analysis = self.sae_analyzer.analyze_features(
            flat_activations, top_k=top_k
        )
        
        # Decompose specific positions
        decompositions = []
        for i in range(min(5, seq_len)):  # Analyze first 5 positions
            pos_activation = activations[0, i]  # First item in batch
            decomp = self.sae_analyzer.decompose_activation(pos_activation)
            decompositions.append({
                "position": i,
                "reconstruction_error": decomp["reconstruction_error"].item(),
                "top_features": decomp["features"].topk(5).indices.tolist(),
                "top_values": decomp["features"].topk(5).values.tolist(),
            })
        
        # Identify dead features
        dead_features = self.sae_analyzer.identify_dead_features(flat_activations)
        
        return {
            "feature_stats": {
                "mean_activation": feature_analysis["mean_activation"].tolist(),
                "activation_frequency": feature_analysis["activation_frequency"].tolist(),
                "top_features": feature_analysis["top_features"].tolist(),
                "top_values": feature_analysis["top_values"].tolist(),
            },
            "decompositions": decompositions,
            "dead_features": dead_features,
            "num_dead_features": len(dead_features),
            "sae_config": {
                "n_features": self.sae.config.n_learned_features,
                "l1_coefficient": self.sae.config.l1_coefficient,
                "layer": layer,
            },
        }
    
    def train_failure_predictor(
        self,
        generate_synthetic: bool = True,
        n_synthetic_samples: int = 100,
    ) -> Dict[str, Any]:
        """Train the failure prediction model
        
        Args:
            generate_synthetic: Whether to generate synthetic training data
            n_synthetic_samples: Number of synthetic samples to generate
        
        Returns:
            Training results and metrics
        """
        logger.info("Training failure prediction model")
        
        # Initialize model if not already done
        if self.failure_predictor is None:
            self.failure_predictor = FailurePredictionModel()
        
        # Generate synthetic data if requested
        if generate_synthetic:
            self.failure_collector.generate_synthetic_failures(
                self, n_samples=n_synthetic_samples
            )
        
        # Get training data
        attention_data_list, labels = self.failure_collector.get_training_data()
        
        if len(attention_data_list) < 20:
            logger.warning(f"Only {len(attention_data_list)} training examples available. Generating more...")
            self.failure_collector.generate_synthetic_failures(self, n_samples=50)
            attention_data_list, labels = self.failure_collector.get_training_data()
        
        # Train model
        results = self.failure_predictor.train(attention_data_list, labels)
        self._failure_model_trained = True
        
        logger.info(f"Failure prediction model trained with {results['train_samples']} samples")
        logger.info(f"Test accuracy: {results['test_accuracy']:.2%}")
        
        return results
    
    def save_failure_predictor(self, path: Union[str, Path]):
        """Save trained failure prediction model"""
        if not self._failure_model_trained:
            raise ValueError("No trained failure prediction model to save")
        
        self.failure_predictor.save(path)
    
    def load_failure_predictor(self, path: Union[str, Path]):
        """Load trained failure prediction model"""
        if self.failure_predictor is None:
            self.failure_predictor = FailurePredictionModel()
        
        self.failure_predictor.load(path)
        self._failure_model_trained = True
    
    def detect_anomalies(
        self,
        text: Union[str, List[str]],
        check_attention: bool = True,
        check_activations: bool = True,
    ) -> Dict[str, Any]:
        """Detect anomalies in model behavior for given text
        
        Args:
            text: Input text to analyze
            check_attention: Whether to check attention patterns
            check_activations: Whether to check activation distributions
        
        Returns:
            Anomaly detection results
        """
        # Perform analysis
        methods = []
        if check_attention:
            methods.append("attention")
        if check_activations:
            methods.append("activations")
        
        if not methods:
            raise ValueError("At least one check type must be enabled")
        
        analysis = self.analyze(text, methods=methods, use_cache=False)
        
        # Check if detector is fitted
        if not self._anomaly_detector_fitted:
            logger.warning("Anomaly detector not fitted. Training on synthetic data...")
            self._fit_anomaly_detector_synthetic()
        
        # Detect anomalies
        anomaly_results = self.anomaly_detector.detect_anomalies(analysis)
        
        # Add text and metadata
        anomaly_results["text"] = text if isinstance(text, str) else text[0]
        anomaly_results["model"] = self.model_name
        
        return anomaly_results
    
    def fit_anomaly_detector(
        self,
        normal_texts: List[str],
        fit_attention: bool = True,
        fit_activations: bool = True,
    ) -> None:
        """Fit anomaly detector on normal examples
        
        Args:
            normal_texts: List of normal text examples
            fit_attention: Whether to fit attention analyzer
            fit_activations: Whether to fit activation detector
        """
        logger.info(f"Fitting anomaly detector on {len(normal_texts)} normal examples")
        
        # Analyze all normal examples
        normal_examples = []
        methods = []
        if fit_attention:
            methods.append("attention")
        if fit_activations:
            methods.append("activations")
        
        for i, text in enumerate(normal_texts):
            if i % 10 == 0:
                logger.info(f"Processing example {i}/{len(normal_texts)}")
            
            analysis = self.analyze(text, methods=methods, use_cache=False)
            normal_examples.append(analysis)
        
        # Fit detector
        self.anomaly_detector.fit(
            normal_examples,
            fit_attention=fit_attention,
            fit_activations=fit_activations
        )
        self._anomaly_detector_fitted = True
        
        logger.info("Anomaly detector fitted successfully")
    
    def _fit_anomaly_detector_synthetic(self):
        """Fit anomaly detector on synthetic normal examples"""
        normal_texts = [
            "The weather today is quite pleasant with clear skies.",
            "Machine learning has revolutionized many industries.",
            "Scientific research requires careful methodology.",
            "The economy shows signs of gradual improvement.",
            "Education plays a crucial role in society.",
            "Technology continues to advance at a rapid pace.",
            "Climate change is a pressing global issue.",
            "Healthcare systems face numerous challenges.",
            "Artificial intelligence offers both opportunities and risks.",
            "International cooperation is essential for progress.",
        ]
        
        self.fit_anomaly_detector(normal_texts, fit_activations=False)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        return self.anomaly_detector.get_anomaly_summary()
    
    def analyze_efficient(
        self,
        text: Union[str, List[str]],
        methods: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        use_chunking: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Memory-efficient analysis for long texts
        
        Args:
            text: Input text or list of texts
            methods: Analysis methods to apply
            max_length: Maximum sequence length
            use_chunking: Whether to use chunked processing for long sequences
            **kwargs: Additional arguments
        
        Returns:
            Analysis results
        """
        if methods is None:
            methods = ["attention", "importance"]
        
        # Check memory before processing
        memory_stats = self.memory_manager.get_memory_usage()
        logger.info(f"Memory usage before analysis: {memory_stats}")
        
        # Tokenize to check length
        inputs = self.model_wrapper.tokenize(text, max_length=max_length)
        seq_length = inputs["input_ids"].shape[1]
        
        # Estimate memory usage
        batch_size = inputs["input_ids"].shape[0]
        memory_estimate = estimate_memory_usage(
            self.model_wrapper,
            batch_size,
            seq_length
        )
        
        logger.info(f"Estimated memory usage: {memory_estimate['total_gb']:.2f} GB")
        
        # Use chunked processing if sequence is long
        if use_chunking and seq_length > self.chunked_processor.chunk_size:
            logger.info(f"Using chunked processing for sequence length {seq_length}")
            
            with self.memory_manager.memory_efficient_mode():
                # Process in chunks
                outputs = self.chunked_processor.process_long_sequence(
                    self.model_wrapper,
                    inputs["input_ids"],
                    inputs.get("attention_mask")
                )
                
                # Run analysis on chunked outputs
                # Note: This is simplified - in practice you'd need to adapt
                # the analysis methods to work with pre-computed outputs
                results = {
                    "attention": {
                        "patterns": outputs["attentions"],
                        "tokens": self.model_wrapper.get_token_strings(inputs["input_ids"]),
                        "shape": {
                            "num_layers": len(outputs["attentions"]),
                            "batch_size": batch_size,
                            "num_heads": self.model_wrapper.get_num_attention_heads(),
                            "seq_length": seq_length,
                        }
                    }
                }
        else:
            # Use regular analysis
            results = self.analyze(text, methods=methods, **kwargs)
        
        # Check memory after processing
        memory_stats_after = self.memory_manager.get_memory_usage()
        logger.info(f"Memory usage after analysis: {memory_stats_after}")
        
        return results
    
    def analyze_document_streaming(
        self,
        document: str,
        methods: List[str] = ["attention"],
        window_size: int = 512,
        stride: int = 256,
        aggregate_results: bool = True,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Analyze long document using streaming windows
        
        Args:
            document: Long document text
            methods: Analysis methods to apply
            window_size: Size of analysis window
            stride: Stride between windows
            aggregate_results: Whether to aggregate results
        
        Returns:
            List of window results or aggregated results
        """
        # Update streaming analyzer settings
        self.streaming_analyzer.window_size = window_size
        self.streaming_analyzer.stride = stride
        
        # Collect results from all windows
        window_results = []
        
        for window_result in self.streaming_analyzer.analyze_document(document, methods):
            window_results.append(window_result)
            
            # Log progress
            window_info = window_result["window_info"]
            logger.info(
                f"Analyzed window {window_info['start_token']}-{window_info['end_token']}"
            )
        
        if not aggregate_results:
            return window_results
        
        # Aggregate results across windows
        aggregated = self._aggregate_window_results(window_results, methods)
        aggregated["num_windows"] = len(window_results)
        
        return aggregated
    
    def _aggregate_window_results(
        self,
        window_results: List[Dict[str, Any]],
        methods: List[str]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple analysis windows"""
        if not window_results:
            return {}
        
        aggregated = {"metadata": window_results[0]["metadata"]}
        
        # Aggregate by method
        if "attention" in methods:
            # Average entropy across windows
            all_entropies = []
            for result in window_results:
                if "attention" in result and "entropy" in result["attention"]:
                    entropy = result["attention"]["entropy"]["mean_entropy_per_layer"]
                    all_entropies.append(torch.tensor(entropy))
            
            if all_entropies:
                mean_entropy = torch.stack(all_entropies).mean(dim=0)
                aggregated["attention"] = {
                    "aggregated_entropy": mean_entropy.tolist(),
                    "window_count": len(all_entropies),
                }
        
        if "importance" in methods:
            # Track most important tokens across all windows
            token_importance_scores = {}
            
            for result in window_results:
                if "importance" in result:
                    tokens = result["importance"]["token_importance"]["tokens"]
                    scores = result["importance"]["token_importance"]["importance_mean"]
                    
                    for token, score in zip(tokens, scores):
                        if token not in token_importance_scores:
                            token_importance_scores[token] = []
                        token_importance_scores[token].append(score)
            
            # Average importance scores
            avg_importance = {
                token: sum(scores) / len(scores)
                for token, scores in token_importance_scores.items()
            }
            
            # Get top tokens
            top_tokens = sorted(
                avg_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            aggregated["importance"] = {
                "top_tokens": top_tokens,
                "unique_tokens": len(token_importance_scores),
            }
        
        return aggregated
    
    def estimate_processing_requirements(
        self,
        text: Union[str, List[str]],
        methods: List[str] = ["attention", "importance"],
    ) -> Dict[str, Any]:
        """Estimate memory and time requirements for analysis
        
        Args:
            text: Input text
            methods: Analysis methods to apply
        
        Returns:
            Dictionary with resource estimates
        """
        # Tokenize to get dimensions
        inputs = self.model_wrapper.tokenize(text)
        batch_size = inputs["input_ids"].shape[0]
        seq_length = inputs["input_ids"].shape[1]
        
        # Estimate memory
        memory_estimate = estimate_memory_usage(
            self.model_wrapper,
            batch_size,
            seq_length
        )
        
        # Estimate time (very rough)
        # Based on typical processing speeds
        time_per_token = 0.001  # 1ms per token (rough estimate)
        estimated_time = batch_size * seq_length * time_per_token * len(methods)
        
        # Check if chunking needed
        needs_chunking = seq_length > self.chunked_processor.chunk_size
        
        # Recommendations
        recommendations = []
        if memory_estimate["total_gb"] > 4.0:
            recommendations.append("Consider using memory-efficient mode")
        if seq_length > 1024:
            recommendations.append("Consider using chunked processing")
        if batch_size > 16:
            recommendations.append("Consider reducing batch size")
        
        return {
            "batch_size": batch_size,
            "sequence_length": seq_length,
            "memory_estimate_gb": memory_estimate,
            "estimated_time_seconds": estimated_time,
            "needs_chunking": needs_chunking,
            "recommendations": recommendations,
        }