"""Advanced failure prediction system using attention patterns"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AttentionFeatureExtractor:
    """Extract features from attention patterns for failure prediction"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, attention_data: Dict[str, Any]) -> np.ndarray:
        """Extract statistical features from attention patterns
        
        Args:
            attention_data: Attention analysis results
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        self.feature_names = []
        
        # Get attention weights and entropy
        attention_weights = torch.tensor(attention_data["patterns"])
        entropy_data = attention_data["entropy"]
        
        # 1. Entropy statistics
        mean_entropy = torch.tensor(entropy_data["mean_entropy_per_layer"])
        features.extend([
            mean_entropy.mean().item(),
            mean_entropy.std().item(),
            mean_entropy.min().item(),
            mean_entropy.max().item(),
        ])
        self.feature_names.extend([
            "entropy_mean", "entropy_std", "entropy_min", "entropy_max"
        ])
        
        # 2. Attention concentration metrics
        # Check for highly peaked attention (potential collapse)
        num_layers, batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Maximum attention per head
        max_attentions = attention_weights.max(dim=-1)[0]  # Shape: (L, B, H, seq_len)
        
        # Percentage of very peaked attention (>0.9)
        peaked_ratio = (max_attentions > 0.9).float().mean().item()
        features.append(peaked_ratio)
        self.feature_names.append("peaked_attention_ratio")
        
        # 3. Attention diversity
        # Compute attention entropy per head
        eps = 1e-8
        head_entropies = -torch.sum(
            attention_weights * torch.log(attention_weights + eps),
            dim=-1
        ).mean(dim=-1)  # Average over sequence positions
        
        # Diversity: standard deviation of head entropies within each layer
        diversity_per_layer = head_entropies.std(dim=-1)  # Std across heads
        features.extend([
            diversity_per_layer.mean().item(),
            diversity_per_layer.min().item(),
        ])
        self.feature_names.extend([
            "head_diversity_mean", "head_diversity_min"
        ])
        
        # 4. Attention to special positions
        # Average attention to first and last tokens
        attention_to_first = attention_weights[:, :, :, :, 0].mean().item()
        attention_to_last = attention_weights[:, :, :, :, -1].mean().item()
        
        features.extend([attention_to_first, attention_to_last])
        self.feature_names.extend(["attention_to_first", "attention_to_last"])
        
        # 5. Diagonal attention (self-attention)
        # Extract diagonal elements
        batch_idx = 0  # Use first batch item
        diagonal_scores = []
        for layer in range(num_layers):
            for head in range(num_heads):
                att_matrix = attention_weights[layer, batch_idx, head]
                diagonal = torch.diagonal(att_matrix).mean().item()
                diagonal_scores.append(diagonal)
        
        features.extend([
            np.mean(diagonal_scores),
            np.std(diagonal_scores),
        ])
        self.feature_names.extend([
            "diagonal_attention_mean", "diagonal_attention_std"
        ])
        
        # 6. Attention pattern consistency
        # Measure how similar attention patterns are across heads
        layer_similarities = []
        for layer in range(num_layers):
            layer_atts = attention_weights[layer, batch_idx]  # Shape: (H, seq, seq)
            
            # Compute pairwise cosine similarity between heads
            flat_atts = layer_atts.view(num_heads, -1)
            normalized = F.normalize(flat_atts, p=2, dim=1)
            similarity_matrix = torch.matmul(normalized, normalized.t())
            
            # Average off-diagonal similarities
            mask = ~torch.eye(num_heads, dtype=torch.bool)
            avg_similarity = similarity_matrix[mask].mean().item()
            layer_similarities.append(avg_similarity)
        
        features.extend([
            np.mean(layer_similarities),
            np.max(layer_similarities),
        ])
        self.feature_names.extend([
            "head_similarity_mean", "head_similarity_max"
        ])
        
        return np.array(features)


class FailurePredictionModel:
    """Model for predicting LLM failures based on attention patterns"""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.feature_extractor = AttentionFeatureExtractor()
        self.classifier = None
        self.threshold = 0.5
        self._is_trained = False
        
        # Initialize classifier
        if model_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_training_data(
        self,
        attention_data_list: List[Dict[str, Any]],
        labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from attention analysis results
        
        Args:
            attention_data_list: List of attention analysis results
            labels: Binary labels (0: normal, 1: failure)
        
        Returns:
            Feature matrix and label array
        """
        features = []
        
        for attention_data in attention_data_list:
            feat_vector = self.feature_extractor.extract_features(attention_data)
            features.append(feat_vector)
        
        X = np.array(features)
        y = np.array(labels)
        
        return X, y
    
    def train(
        self,
        attention_data_list: List[Dict[str, Any]],
        labels: List[int],
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Train the failure prediction model
        
        Args:
            attention_data_list: List of attention analysis results
            labels: Binary labels (0: normal, 1: failure)
            test_size: Fraction of data to use for testing
        
        Returns:
            Training results and metrics
        """
        logger.info(f"Training failure prediction model with {len(labels)} samples")
        
        # Prepare data
        X, y = self.prepare_training_data(attention_data_list, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.classifier.fit(X_train, y_train)
        self._is_trained = True
        
        # Evaluate
        train_pred = self.classifier.predict(X_train)
        test_pred = self.classifier.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_extractor.feature_names,
            self.classifier.feature_importances_
        ))
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        results = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": classification_report(y_test, test_pred),
            "feature_importance": sorted_features[:10],  # Top 10 features
            "train_samples": len(y_train),
            "test_samples": len(y_test),
        }
        
        logger.info(f"Training complete. Test accuracy: {test_accuracy:.2%}")
        
        return results
    
    def predict(self, attention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict failure probability for given attention patterns
        
        Args:
            attention_data: Attention analysis results
        
        Returns:
            Prediction results
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features = self.feature_extractor.extract_features(attention_data)
        features = features.reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.classifier.predict(features)[0]
        probability = self.classifier.predict_proba(features)[0, 1]  # Probability of failure
        
        # Determine risk level
        if probability > 0.8:
            risk_level = "high_risk"
        elif probability > 0.5:
            risk_level = "medium_risk"
        else:
            risk_level = "low_risk"
        
        # Get feature contributions
        feature_values = dict(zip(
            self.feature_extractor.feature_names,
            features[0]
        ))
        
        # Identify key indicators
        indicators = []
        if feature_values.get("entropy_mean", 1.0) < 0.5:
            indicators.append("low_attention_entropy")
        if feature_values.get("peaked_attention_ratio", 0) > 0.5:
            indicators.append("attention_collapse")
        if feature_values.get("head_similarity_max", 0) > 0.9:
            indicators.append("redundant_attention_heads")
        
        return {
            "failure_probability": float(probability),
            "prediction": risk_level,
            "indicators": indicators,
            "confidence": float(max(probability, 1 - probability)),
            "feature_values": feature_values,
        }
    
    def save(self, path: Union[str, Path]):
        """Save trained model to disk"""
        import joblib
        
        if not self._is_trained:
            raise ValueError("Model not trained")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "classifier": self.classifier,
            "feature_names": self.feature_extractor.feature_names,
            "model_type": self.model_type,
            "threshold": self.threshold,
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Saved failure prediction model to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load trained model from disk"""
        import joblib
        
        path = Path(path)
        model_data = joblib.load(path)
        
        self.classifier = model_data["classifier"]
        self.feature_extractor.feature_names = model_data["feature_names"]
        self.model_type = model_data["model_type"]
        self.threshold = model_data.get("threshold", 0.5)
        self._is_trained = True
        
        logger.info(f"Loaded failure prediction model from {path}")


class FailureDataCollector:
    """Collect and manage failure examples for training"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/failure_examples.json")
        self.examples = self._load_examples()
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load existing examples from storage"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return []
    
    def add_example(
        self,
        text: str,
        attention_data: Dict[str, Any],
        is_failure: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a new example to the collection
        
        Args:
            text: The input text
            attention_data: Attention analysis results
            is_failure: Whether this is a failure case
            metadata: Additional metadata
        """
        example = {
            "text": text,
            "attention_data": attention_data,
            "is_failure": is_failure,
            "metadata": metadata or {},
            "timestamp": str(np.datetime64('now')),
        }
        
        self.examples.append(example)
        self._save_examples()
    
    def _save_examples(self):
        """Save examples to storage"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to lists for JSON serialization
        serializable_examples = []
        for ex in self.examples:
            serializable_ex = ex.copy()
            # Deep conversion of attention_data would go here
            serializable_examples.append(serializable_ex)
        
        with open(self.storage_path, 'w') as f:
            json.dump(serializable_examples, f, indent=2)
    
    def get_training_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Get training data from collected examples
        
        Returns:
            Tuple of (attention_data_list, labels)
        """
        attention_data_list = []
        labels = []
        
        for example in self.examples:
            attention_data_list.append(example["attention_data"])
            labels.append(1 if example["is_failure"] else 0)
        
        return attention_data_list, labels
    
    def generate_synthetic_failures(self, analyzer, n_samples: int = 100):
        """Generate synthetic failure examples
        
        Args:
            analyzer: InterpretabilityAnalyzer instance
            n_samples: Number of synthetic examples to generate
        """
        logger.info(f"Generating {n_samples} synthetic failure examples")
        
        # Patterns that often indicate failures
        failure_patterns = [
            # Repetitive text
            lambda: " ".join(["word"] * 20),
            lambda: " ".join(["the"] * 15),
            lambda: "a" * 50,
            
            # Incoherent sequences
            lambda: " ".join(np.random.choice(["cat", "dog", "tree", "blue", "quickly"], 20)),
            
            # Truncated/incomplete
            lambda: "The quick brown fox jumps over the",
            lambda: "Once upon a time there was a",
        ]
        
        # Normal patterns
        normal_patterns = [
            "The weather today is quite pleasant with clear skies.",
            "Machine learning has many practical applications in industry.",
            "Scientific research requires careful methodology and analysis.",
            "The economy shows signs of gradual improvement this quarter.",
            "Education plays a crucial role in societal development.",
        ]
        
        # Generate failure examples
        for i in range(n_samples // 2):
            # Create failure text
            pattern_fn = np.random.choice(failure_patterns)
            text = pattern_fn()
            
            # Analyze
            results = analyzer.analyze(text, methods=["attention"])
            
            # Add to collection
            self.add_example(
                text=text,
                attention_data=results["attention"],
                is_failure=True,
                metadata={"synthetic": True, "pattern": "failure"}
            )
        
        # Generate normal examples
        for i in range(n_samples // 2):
            # Use normal text
            text = np.random.choice(normal_patterns)
            
            # Add slight variations
            if np.random.random() > 0.5:
                text = text.replace(".", "!").lower()
            
            # Analyze
            results = analyzer.analyze(text, methods=["attention"])
            
            # Add to collection
            self.add_example(
                text=text,
                attention_data=results["attention"],
                is_failure=False,
                metadata={"synthetic": True, "pattern": "normal"}
            )
        
        logger.info(f"Generated {n_samples} synthetic examples")