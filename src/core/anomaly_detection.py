"""Anomaly detection for attention patterns and model behaviors"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AttentionPatternAnalyzer:
    """Analyze and detect anomalous attention patterns"""
    
    def __init__(self):
        self.pattern_database = defaultdict(list)
        self.anomaly_threshold = 2.0  # Standard deviations
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def analyze_pattern(
        self,
        attention_weights: torch.Tensor,
        layer: Optional[int] = None,
        head: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze attention pattern characteristics
        
        Args:
            attention_weights: Attention tensor
            layer: Specific layer to analyze
            head: Specific head to analyze
        
        Returns:
            Pattern characteristics
        """
        # Extract specific layer/head if specified
        if layer is not None and head is not None:
            att = attention_weights[layer, 0, head]  # First batch item
        else:
            # Average across all layers and heads
            att = attention_weights.mean(dim=(0, 1, 2))
        
        att = att.cpu().numpy()
        seq_len = att.shape[0]
        
        # Compute pattern features
        features = {}
        
        # 1. Diagonal dominance (self-attention strength)
        diagonal_sum = np.trace(att)
        features["diagonal_dominance"] = diagonal_sum / seq_len
        
        # 2. Off-diagonal spread
        off_diagonal_mask = ~np.eye(seq_len, dtype=bool)
        off_diagonal_values = att[off_diagonal_mask]
        features["off_diagonal_mean"] = off_diagonal_values.mean()
        features["off_diagonal_std"] = off_diagonal_values.std()
        
        # 3. Attention to special positions
        features["attention_to_first"] = att[:, 0].mean()
        features["attention_to_last"] = att[:, -1].mean()
        features["attention_from_first"] = att[0, :].mean()
        features["attention_from_last"] = att[-1, :].mean()
        
        # 4. Attention sparsity
        features["sparsity"] = (att < 0.01).sum() / (seq_len * seq_len)
        
        # 5. Attention peakedness
        max_per_row = att.max(axis=1)
        features["mean_max_attention"] = max_per_row.mean()
        features["std_max_attention"] = max_per_row.std()
        
        # 6. Attention symmetry
        symmetry_diff = np.abs(att - att.T).mean()
        features["asymmetry"] = symmetry_diff
        
        # 7. Block patterns (local vs global attention)
        # Check for block-diagonal patterns
        block_size = max(1, seq_len // 4)
        block_scores = []
        for i in range(0, seq_len - block_size, block_size):
            block = att[i:i+block_size, i:i+block_size]
            block_scores.append(block.mean())
        
        features["block_diagonal_score"] = np.mean(block_scores) if block_scores else 0
        
        # 8. Distance-based attention decay
        # Check if attention decreases with distance
        distances = []
        decays = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    dist = abs(i - j)
                    distances.append(dist)
                    decays.append(att[i, j])
        
        if distances:
            # Correlation between distance and attention
            correlation = np.corrcoef(distances, decays)[0, 1]
            features["distance_decay_correlation"] = correlation
        else:
            features["distance_decay_correlation"] = 0
        
        return features
    
    def fit(self, attention_patterns: List[torch.Tensor]):
        """Fit the analyzer on normal attention patterns
        
        Args:
            attention_patterns: List of attention tensors from normal examples
        """
        logger.info(f"Fitting anomaly detector on {len(attention_patterns)} patterns")
        
        # Extract features from all patterns
        all_features = []
        for pattern in attention_patterns:
            features = self.analyze_pattern(pattern)
            feature_vector = list(features.values())
            all_features.append(feature_vector)
        
        # Fit scaler
        all_features = np.array(all_features)
        self.scaler.fit(all_features)
        
        # Store normalized features for comparison
        self.normal_features = self.scaler.transform(all_features)
        self.feature_names = list(features.keys())
        self._is_fitted = True
        
        logger.info("Anomaly detector fitted successfully")
    
    def detect_anomaly(
        self,
        attention_weights: torch.Tensor,
        return_scores: bool = False
    ) -> Union[bool, Tuple[bool, Dict[str, float]]]:
        """Detect if attention pattern is anomalous
        
        Args:
            attention_weights: Attention tensor to check
            return_scores: Whether to return detailed scores
        
        Returns:
            Boolean indicating anomaly, optionally with detailed scores
        """
        if not self._is_fitted:
            raise ValueError("Analyzer not fitted. Call fit() first.")
        
        # Extract features
        features = self.analyze_pattern(attention_weights)
        feature_vector = np.array([features[name] for name in self.feature_names])
        
        # Normalize
        normalized = self.scaler.transform(feature_vector.reshape(1, -1))[0]
        
        # Compute anomaly scores (distance from normal patterns)
        distances = np.linalg.norm(self.normal_features - normalized, axis=1)
        
        # Use robust statistics
        median_distance = np.median(distances)
        mad = np.median(np.abs(distances - median_distance))
        
        # Modified z-score
        z_score = (distances.min() - median_distance) / (1.4826 * mad + 1e-8)
        
        is_anomaly = abs(z_score) > self.anomaly_threshold
        
        if return_scores:
            scores = {
                "z_score": float(z_score),
                "min_distance": float(distances.min()),
                "median_distance": float(median_distance),
                "is_anomaly": bool(is_anomaly),
                "anomalous_features": self._identify_anomalous_features(features, normalized)
            }
            return is_anomaly, scores
        
        return is_anomaly
    
    def _identify_anomalous_features(
        self,
        features: Dict[str, float],
        normalized: np.ndarray
    ) -> List[str]:
        """Identify which features are most anomalous"""
        anomalous = []
        
        # Compare each feature to normal distribution
        for i, (name, value) in enumerate(features.items()):
            normal_values = self.normal_features[:, i]
            z_score = abs(normalized[i] - normal_values.mean()) / (normal_values.std() + 1e-8)
            
            if z_score > self.anomaly_threshold:
                anomalous.append(f"{name} (z={z_score:.2f})")
        
        return anomalous


class ActivationAnomalyDetector:
    """Detect anomalies in model activations using clustering"""
    
    def __init__(self, method: str = "dbscan"):
        self.method = method
        self.reducer = PCA(n_components=50)
        self.scaler = StandardScaler()
        self._is_fitted = False
        
        if method == "dbscan":
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, activations: List[torch.Tensor]):
        """Fit the detector on normal activations
        
        Args:
            activations: List of activation tensors
        """
        logger.info(f"Fitting activation anomaly detector with {len(activations)} samples")
        
        # Flatten and concatenate activations
        flat_activations = []
        for act in activations:
            if act.dim() > 2:
                act = act.view(act.size(0), -1)
            flat_activations.append(act.cpu().numpy())
        
        all_activations = np.vstack(flat_activations)
        
        # Scale and reduce dimensionality
        scaled = self.scaler.fit_transform(all_activations)
        
        if scaled.shape[1] > 50:
            reduced = self.reducer.fit_transform(scaled)
        else:
            reduced = scaled
            self.reducer = None
        
        # Fit clusterer
        self.cluster_labels = self.clusterer.fit_predict(reduced)
        self.normal_samples = reduced
        self._is_fitted = True
        
        # Identify outlier percentage
        outliers = (self.cluster_labels == -1).sum()
        logger.info(f"Fitted with {outliers}/{len(reduced)} outliers ({outliers/len(reduced)*100:.1f}%)")
    
    def detect_anomaly(
        self,
        activation: torch.Tensor
    ) -> Tuple[bool, Dict[str, Any]]:
        """Detect if activation is anomalous
        
        Args:
            activation: Activation tensor
        
        Returns:
            Tuple of (is_anomaly, details)
        """
        if not self._is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Prepare activation
        if activation.dim() > 2:
            activation = activation.view(activation.size(0), -1)
        
        flat = activation.cpu().numpy()
        if flat.ndim == 1:
            flat = flat.reshape(1, -1)
        
        # Transform
        scaled = self.scaler.transform(flat)
        if self.reducer is not None:
            reduced = self.reducer.transform(scaled)
        else:
            reduced = scaled
        
        # Predict cluster
        # For DBSCAN, we need to find nearest neighbors
        distances = np.linalg.norm(self.normal_samples - reduced, axis=1)
        min_distance = distances.min()
        
        # Check if it would be an outlier
        is_anomaly = min_distance > self.clusterer.eps
        
        details = {
            "min_distance": float(min_distance),
            "threshold": float(self.clusterer.eps),
            "nearest_cluster": int(self.cluster_labels[distances.argmin()]) if not is_anomaly else -1,
        }
        
        return is_anomaly, details


class ComprehensiveAnomalyDetector:
    """Comprehensive anomaly detection combining multiple signals"""
    
    def __init__(self):
        self.attention_analyzer = AttentionPatternAnalyzer()
        self.activation_detector = ActivationAnomalyDetector()
        self.anomaly_history = []
        self._is_fitted = False
    
    def fit(
        self,
        normal_examples: List[Dict[str, Any]],
        fit_attention: bool = True,
        fit_activations: bool = True
    ):
        """Fit detector on normal examples
        
        Args:
            normal_examples: List of analysis results from normal texts
            fit_attention: Whether to fit attention analyzer
            fit_activations: Whether to fit activation detector
        """
        logger.info(f"Fitting comprehensive anomaly detector on {len(normal_examples)} examples")
        
        if fit_attention:
            attention_patterns = []
            for example in normal_examples:
                if "attention" in example and "patterns" in example["attention"]:
                    patterns = torch.tensor(example["attention"]["patterns"])
                    attention_patterns.append(patterns)
            
            if attention_patterns:
                self.attention_analyzer.fit(attention_patterns)
        
        if fit_activations:
            activations = []
            for example in normal_examples:
                if "activations" in example and "all_layers" in example["activations"]:
                    acts = torch.tensor(example["activations"]["all_layers"])
                    activations.append(acts)
            
            if activations:
                self.activation_detector.fit(activations)
        
        self._is_fitted = True
    
    def detect_anomalies(
        self,
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect anomalies in analysis results
        
        Args:
            analysis_result: Complete analysis result from analyzer
        
        Returns:
            Dictionary with anomaly detection results
        """
        if not self._is_fitted:
            logger.warning("Detector not fitted, returning default results")
            return {"is_anomaly": False, "anomaly_types": [], "confidence": 0.0}
        
        anomalies = []
        scores = {}
        
        # Check attention patterns
        if "attention" in analysis_result and "patterns" in analysis_result["attention"]:
            patterns = torch.tensor(analysis_result["attention"]["patterns"])
            is_attention_anomaly, attention_scores = self.attention_analyzer.detect_anomaly(
                patterns, return_scores=True
            )
            
            if is_attention_anomaly:
                anomalies.append("attention_pattern")
            
            scores["attention"] = attention_scores
        
        # Check activations
        if "activations" in analysis_result and "all_layers" in analysis_result["activations"]:
            acts = torch.tensor(analysis_result["activations"]["all_layers"])
            is_activation_anomaly, activation_details = self.activation_detector.detect_anomaly(acts)
            
            if is_activation_anomaly:
                anomalies.append("activation_distribution")
            
            scores["activation"] = activation_details
        
        # Check for specific anomaly patterns
        if "attention" in analysis_result:
            # Check for attention collapse
            entropy = analysis_result["attention"]["entropy"]["mean_entropy_per_layer"]
            if isinstance(entropy, list):
                entropy = torch.tensor(entropy)
            
            if entropy.mean() < 0.3:  # Very low entropy
                anomalies.append("attention_collapse")
            elif entropy.mean() > 3.0:  # Very high entropy
                anomalies.append("attention_dispersion")
        
        # Calculate overall confidence
        confidence = len(anomalies) / 4.0  # Normalize by max possible anomalies
        
        result = {
            "is_anomaly": len(anomalies) > 0,
            "anomaly_types": anomalies,
            "confidence": min(confidence, 1.0),
            "scores": scores,
            "timestamp": str(np.datetime64('now')),
        }
        
        # Store in history
        self.anomaly_history.append(result)
        
        return result
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        if not self.anomaly_history:
            return {"total_checks": 0, "anomalies_found": 0, "anomaly_rate": 0.0}
        
        total = len(self.anomaly_history)
        anomalies = sum(1 for h in self.anomaly_history if h["is_anomaly"])
        
        # Count by type
        type_counts = defaultdict(int)
        for history in self.anomaly_history:
            for anomaly_type in history["anomaly_types"]:
                type_counts[anomaly_type] += 1
        
        return {
            "total_checks": total,
            "anomalies_found": anomalies,
            "anomaly_rate": anomalies / total,
            "anomaly_types": dict(type_counts),
            "recent_anomalies": self.anomaly_history[-10:],  # Last 10
        }