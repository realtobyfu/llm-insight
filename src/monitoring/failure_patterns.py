"""Pattern-based failure detection for monitoring"""

from typing import Any, Dict, List, Optional, Tuple
import re
from collections import Counter

import torch
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FailurePatternDetector:
    """Detects known failure patterns in model outputs and behavior"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize failure pattern definitions"""
        return {
            "repetition": {
                "name": "Excessive Repetition",
                "severity": "high",
                "detector": self._detect_repetition,
                "threshold": 0.3,
                "description": "Text contains excessive repetition of tokens or phrases"
            },
            "degeneration": {
                "name": "Text Degeneration", 
                "severity": "high",
                "detector": self._detect_degeneration,
                "threshold": 0.5,
                "description": "Text degenerates into nonsense or single character repetition"
            },
            "attention_collapse": {
                "name": "Attention Collapse",
                "severity": "medium",
                "detector": self._detect_attention_collapse,
                "threshold": 0.7,
                "description": "Attention weights collapse to single token"
            },
            "entropy_anomaly": {
                "name": "Entropy Anomaly",
                "severity": "medium", 
                "detector": self._detect_entropy_anomaly,
                "threshold": 0.8,
                "description": "Attention entropy is abnormally low or high"
            },
            "token_imbalance": {
                "name": "Token Imbalance",
                "severity": "low",
                "detector": self._detect_token_imbalance,
                "threshold": 0.5,
                "description": "Severe imbalance in token importance"
            },
            "context_loss": {
                "name": "Context Loss",
                "severity": "high",
                "detector": self._detect_context_loss,
                "threshold": 0.6,
                "description": "Model loses track of context"
            }
        }
    
    def detect_failures(self, 
                       text: str,
                       analysis_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detect failure patterns in text and analysis results"""
        detected_patterns = []
        pattern_scores = {}
        
        for pattern_id, pattern_config in self.patterns.items():
            try:
                detector = pattern_config["detector"]
                score = detector(text, analysis_results)
                pattern_scores[pattern_id] = score
                
                if score > pattern_config["threshold"]:
                    detected_patterns.append({
                        "pattern": pattern_id,
                        "name": pattern_config["name"],
                        "severity": pattern_config["severity"],
                        "score": score,
                        "description": pattern_config["description"]
                    })
                    logger.debug(f"Detected pattern: {pattern_id} (score={score:.3f})")
                    
            except Exception as e:
                logger.error(f"Error in pattern detector {pattern_id}: {e}")
                pattern_scores[pattern_id] = 0.0
        
        # Calculate overall failure score
        severity_weights = {"high": 1.0, "medium": 0.6, "low": 0.3}
        total_score = 0.0
        for pattern in detected_patterns:
            weight = severity_weights.get(pattern["severity"], 0.5)
            total_score += pattern["score"] * weight
        
        # Normalize by number of patterns
        overall_score = min(1.0, total_score / len(self.patterns))
        
        return {
            "detected_patterns": detected_patterns,
            "pattern_scores": pattern_scores,
            "overall_failure_score": overall_score,
            "is_failure": overall_score > 0.5,
            "failure_count": len(detected_patterns)
        }
    
    def _detect_repetition(self, text: str, analysis: Optional[Dict[str, Any]]) -> float:
        """Detect excessive repetition in text"""
        if not text:
            return 0.0
            
        # Tokenize into words
        words = text.lower().split()
        if len(words) < 3:
            return 0.0
        
        # Check for repeated words
        word_counts = Counter(words)
        max_count = max(word_counts.values())
        word_repetition = max_count / len(words)
        
        # Check for repeated n-grams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        max_bigram = max(bigram_counts.values()) if bigrams else 0
        bigram_repetition = max_bigram / max(1, len(bigrams))
        
        # Check for character-level repetition
        char_runs = re.findall(r'(.)\1{2,}', text)
        char_repetition = sum(len(run) for run in char_runs) / max(1, len(text))
        
        # Combine scores
        score = max(word_repetition, bigram_repetition * 1.5, char_repetition * 2)
        
        return min(1.0, score)
    
    def _detect_degeneration(self, text: str, analysis: Optional[Dict[str, Any]]) -> float:
        """Detect text degeneration patterns"""
        if not text:
            return 1.0
            
        # Check for single character spam
        unique_chars = len(set(text.replace(" ", "")))
        char_diversity = unique_chars / max(1, len(text.replace(" ", "")))
        
        # Check for very short "words"
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            short_word_score = 1.0 - min(1.0, avg_word_length / 4.0)
        else:
            short_word_score = 1.0
        
        # Check for non-alphabetic content
        alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
        non_alpha_score = 1.0 - alpha_ratio
        
        # Combine scores
        score = max(1.0 - char_diversity, short_word_score, non_alpha_score)
        
        return min(1.0, score)
    
    def _detect_attention_collapse(self, text: str, analysis: Optional[Dict[str, Any]]) -> float:
        """Detect if attention weights collapse to single token"""
        if not analysis or "attention" not in analysis:
            return 0.0
            
        attention_data = analysis["attention"]
        if "patterns" not in attention_data:
            return 0.0
            
        patterns = attention_data["patterns"]
        if isinstance(patterns, torch.Tensor):
            patterns = patterns.cpu().numpy()
        elif not isinstance(patterns, np.ndarray):
            patterns = np.array(patterns)
            
        # Check each head for collapsed attention
        collapse_scores = []
        
        # patterns shape: [layers, heads, seq_len, seq_len]
        for layer in range(patterns.shape[0]):
            for head in range(patterns.shape[1]):
                attention_matrix = patterns[layer, head]
                
                # Calculate max attention weight per position
                max_weights = np.max(attention_matrix, axis=-1)
                
                # High max weight indicates collapse
                collapse_score = np.mean(max_weights)
                collapse_scores.append(collapse_score)
        
        # Return average collapse score
        return np.mean(collapse_scores) if collapse_scores else 0.0
    
    def _detect_entropy_anomaly(self, text: str, analysis: Optional[Dict[str, Any]]) -> float:
        """Detect abnormal attention entropy patterns"""
        if not analysis or "attention" not in analysis:
            return 0.0
            
        attention_data = analysis["attention"]
        if "entropy" not in attention_data:
            return 0.0
            
        entropy_data = attention_data["entropy"]
        if "mean_entropy_per_layer" not in entropy_data:
            return 0.0
            
        entropies = entropy_data["mean_entropy_per_layer"]
        if entropies is None or len(entropies) == 0:
            return 0.0
            
        # Convert to numpy array, handling tensors
        if hasattr(entropies[0], 'cpu'):  # Check if it's a tensor
            entropies = np.array([e.cpu().numpy() if hasattr(e, 'cpu') else e for e in entropies])
        else:
            entropies = np.array(entropies)
        
        # Check for very low entropy (attention collapse)
        low_entropy_score = np.mean(entropies < 0.5)
        
        # Check for entropy variance (should have some variation)
        entropy_std = np.std(entropies)
        low_variance_score = 1.0 - min(1.0, entropy_std * 10)
        
        # Check for sudden drops
        if len(entropies) > 1:
            diffs = np.diff(entropies)
            sudden_drop_score = np.mean(diffs < -0.3)
        else:
            sudden_drop_score = 0.0
        
        # Combine scores
        score = max(low_entropy_score, low_variance_score * 0.5, sudden_drop_score)
        
        return min(1.0, score)
    
    def _detect_token_imbalance(self, text: str, analysis: Optional[Dict[str, Any]]) -> float:
        """Detect severe imbalance in token importance"""
        if not analysis or "importance" not in analysis:
            return 0.0
            
        importance_data = analysis["importance"]
        if "token_importance" not in importance_data:
            return 0.0
            
        token_imp = importance_data["token_importance"]
        if "importance_mean" not in token_imp:
            return 0.0
            
        importances = token_imp["importance_mean"]
        if importances is None or len(importances) < 2:
            return 0.0
            
        # Convert to numpy array, handling tensors
        if hasattr(importances[0], 'cpu'):  # Check if it's a tensor
            importances = np.array([i.cpu().numpy() if hasattr(i, 'cpu') else i for i in importances])
        else:
            importances = np.array(importances)
        
        # Normalize
        total_importance = np.sum(importances)
        if total_importance == 0:
            return 1.0
            
        norm_importances = importances / total_importance
        
        # Calculate Gini coefficient for inequality
        sorted_imp = np.sort(norm_importances)
        n = len(sorted_imp)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_imp)) / (n * np.sum(sorted_imp)) - (n + 1) / n
        
        return min(1.0, gini)
    
    def _detect_context_loss(self, text: str, analysis: Optional[Dict[str, Any]]) -> float:
        """Detect if model loses track of context"""
        if not analysis or "attention" not in analysis:
            return 0.0
            
        attention_data = analysis["attention"]
        if "patterns" not in attention_data:
            return 0.0
            
        patterns = attention_data["patterns"]
        if isinstance(patterns, torch.Tensor):
            patterns = patterns.cpu().numpy()
        elif not isinstance(patterns, np.ndarray):
            patterns = np.array(patterns)
            
        # Check if attention to early tokens decreases dramatically in later layers
        context_scores = []
        
        # Look at attention to first 20% of sequence
        seq_len = patterns.shape[-1]
        context_size = max(1, seq_len // 5)
        
        for layer in range(patterns.shape[0]):
            # Average attention to context tokens
            layer_attention = patterns[layer].mean(axis=0)  # Average over heads
            context_attention = layer_attention[:, :context_size].mean()
            context_scores.append(context_attention)
        
        if len(context_scores) > 1:
            # Check if context attention drops
            context_drop = (context_scores[0] - context_scores[-1]) / max(0.01, context_scores[0])
            return min(1.0, max(0.0, context_drop * 2))
        
        return 0.0
    
    def get_pattern_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all failure patterns"""
        return {
            pid: pconfig["description"] 
            for pid, pconfig in self.patterns.items()
        }