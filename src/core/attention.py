"""Attention analysis module for transformer models"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AttentionAnalyzer:
    """Analyzer for attention patterns in transformer models"""
    
    def __init__(self, model_wrapper):
        """Initialize attention analyzer
        
        Args:
            model_wrapper: Instance of ModelWrapper
        """
        self.model_wrapper = model_wrapper
        self.num_layers = model_wrapper.get_num_layers()
        self.num_heads = model_wrapper.get_num_attention_heads()
    
    def extract_attention_patterns(
        self,
        text: Union[str, List[str]],
        layer: Optional[int] = None,
        head: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from the model
        
        Args:
            text: Input text or list of texts
            layer: Specific layer to analyze (None for all)
            head: Specific head to analyze (None for all)
        
        Returns:
            Dictionary containing attention patterns and metadata
        """
        # Tokenize input
        inputs = self.model_wrapper.tokenize(text)
        
        # Forward pass
        outputs = self.model_wrapper.forward(**inputs)
        
        # Get attention weights
        attention_weights = self.model_wrapper.get_attention_weights()
        
        if attention_weights is None:
            raise ValueError("No attention weights available from model")
        
        # Extract specific layer/head if requested
        if layer is not None:
            attention_weights = attention_weights[layer:layer+1]
        if head is not None:
            attention_weights = attention_weights[:, :, head:head+1]
        
        # Get token strings
        token_strings = self.model_wrapper.get_token_strings(inputs["input_ids"])
        
        return {
            "attention_weights": attention_weights,
            "tokens": token_strings,
            "shape": {
                "num_layers": attention_weights.shape[0],
                "batch_size": attention_weights.shape[1],
                "num_heads": attention_weights.shape[2],
                "seq_length": attention_weights.shape[3],
            }
        }
    
    def compute_attention_rollout(
        self,
        attention_weights: torch.Tensor,
        start_layer: int = 0,
    ) -> torch.Tensor:
        """Compute attention rollout across layers
        
        Attention rollout aggregates attention across layers to show
        how information flows from input to output tokens.
        
        Args:
            attention_weights: Attention weights tensor
            start_layer: Layer to start rollout from
        
        Returns:
            Rolled out attention matrix
        """
        # Average attention weights across heads
        # Shape: (num_layers, batch_size, seq_len, seq_len)
        attention_avg = attention_weights.mean(dim=2)
        
        # Add residual connections (identity matrix)
        residual = torch.eye(
            attention_avg.shape[-1],
            device=attention_avg.device
        ).unsqueeze(0).unsqueeze(0)
        
        attention_with_residual = attention_avg + residual
        
        # Normalize
        attention_with_residual = attention_with_residual / attention_with_residual.sum(dim=-1, keepdim=True)
        
        # Compute rollout
        rollout = attention_with_residual[start_layer]
        for layer in range(start_layer + 1, attention_avg.shape[0]):
            rollout = torch.matmul(rollout, attention_with_residual[layer])
        
        return rollout
    
    def identify_attention_heads_by_pattern(
        self,
        attention_weights: torch.Tensor,
        pattern_type: str = "diagonal",
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Identify attention heads with specific patterns
        
        Args:
            attention_weights: Attention weights tensor
            pattern_type: Type of pattern to identify
                - "diagonal": Attending to current position
                - "previous": Attending to previous token
                - "first": Attending to first token
                - "last": Attending to last token
        
        Returns:
            Dictionary mapping pattern types to list of (layer, head) tuples
        """
        results = {}
        num_layers, batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        for layer in range(num_layers):
            for head in range(num_heads):
                att = attention_weights[layer, :, head].mean(dim=0)  # Average over batch
                
                if pattern_type == "diagonal":
                    # Check if attention is mostly on diagonal
                    diagonal_score = torch.diagonal(att).mean()
                    if diagonal_score > 0.5:
                        results.setdefault("diagonal", []).append((layer, head))
                
                elif pattern_type == "previous":
                    # Check if attention is mostly on previous token
                    prev_score = torch.diagonal(att, offset=-1).mean()
                    if prev_score > 0.3:
                        results.setdefault("previous", []).append((layer, head))
                
                elif pattern_type == "first":
                    # Check if attention is mostly on first token
                    first_score = att[:, 0].mean()
                    if first_score > 0.3:
                        results.setdefault("first", []).append((layer, head))
                
                elif pattern_type == "last":
                    # Check if attention is mostly on last token
                    last_score = att[:, -1].mean()
                    if last_score > 0.3:
                        results.setdefault("last", []).append((layer, head))
        
        return results
    
    def compute_head_importance(
        self,
        text: Union[str, List[str]],
        metric: str = "gradient",
    ) -> torch.Tensor:
        """Compute importance scores for each attention head
        
        Args:
            text: Input text for analysis
            metric: Importance metric to use
                - "gradient": Based on gradient magnitude
                - "mean": Based on mean attention weight
                - "max": Based on max attention weight
        
        Returns:
            Tensor of shape (num_layers, num_heads) with importance scores
        """
        # Get attention patterns
        patterns = self.extract_attention_patterns(text)
        attention_weights = patterns["attention_weights"]
        
        num_layers, batch_size, num_heads, seq_len, _ = attention_weights.shape
        importance_scores = torch.zeros(num_layers, num_heads)
        
        if metric == "mean":
            # Average attention weight per head
            for layer in range(num_layers):
                for head in range(num_heads):
                    importance_scores[layer, head] = attention_weights[layer, :, head].mean()
        
        elif metric == "max":
            # Maximum attention weight per head
            for layer in range(num_layers):
                for head in range(num_heads):
                    importance_scores[layer, head] = attention_weights[layer, :, head].max()
        
        elif metric == "gradient":
            # This would require gradient computation through the model
            # For now, we'll use a proxy based on attention entropy
            for layer in range(num_layers):
                for head in range(num_heads):
                    att = attention_weights[layer, :, head]
                    # Compute entropy as a measure of importance
                    entropy = -torch.sum(att * torch.log(att + 1e-9), dim=-1).mean()
                    importance_scores[layer, head] = entropy
        
        return importance_scores
    
    def find_induction_heads(
        self,
        repeated_sequence: str,
        threshold: float = 0.7,
    ) -> List[Tuple[int, int]]:
        """Find induction heads that copy repeated patterns
        
        Induction heads are attention heads that learn to copy
        repeated sequences in the input.
        
        Args:
            repeated_sequence: A sequence with repeated patterns
            threshold: Minimum attention score to consider
        
        Returns:
            List of (layer, head) tuples identifying induction heads
        """
        # Get attention patterns
        patterns = self.extract_attention_patterns(repeated_sequence)
        attention_weights = patterns["attention_weights"]
        
        induction_heads = []
        num_layers, batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Find repeated patterns in the sequence
        tokens = patterns["tokens"][0]  # Assuming single sequence
        pattern_positions = []
        
        for i in range(len(tokens) - 1):
            for j in range(i + 2, len(tokens)):
                if tokens[i] == tokens[j] and i + 1 < len(tokens) and j + 1 < len(tokens):
                    if tokens[i + 1] == tokens[j + 1]:
                        pattern_positions.append((i, j))
        
        # Check which heads attend from second occurrence to first
        for layer in range(num_layers):
            for head in range(num_heads):
                att = attention_weights[layer, 0, head]  # First item in batch
                
                for i, j in pattern_positions:
                    if j + 1 < seq_len and i + 1 < seq_len:
                        # Check if position j+1 attends to position i+1
                        if att[j + 1, i + 1] > threshold:
                            induction_heads.append((layer, head))
                            break
        
        return list(set(induction_heads))  # Remove duplicates
    
    def compute_attention_entropy(
        self,
        attention_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute entropy of attention distributions
        
        Lower entropy indicates more focused attention,
        higher entropy indicates more distributed attention.
        
        Args:
            attention_weights: Attention weights tensor
        
        Returns:
            Dictionary with entropy statistics
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-9
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + epsilon),
            dim=-1
        )
        
        # Aggregate statistics
        return {
            "entropy": entropy,
            "mean_entropy_per_layer": entropy.mean(dim=(1, 2, 3)),
            "mean_entropy_per_head": entropy.mean(dim=(1, 3)),
            "entropy_by_position": entropy.mean(dim=(0, 1, 2)),
        }
    
    def analyze_token_importance(
        self,
        text: str,
        target_token_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Analyze importance of tokens based on attention patterns
        
        Args:
            text: Input text
            target_token_idx: Index of target token (None for last token)
        
        Returns:
            Dictionary with token importance scores
        """
        # Get attention patterns
        patterns = self.extract_attention_patterns(text)
        attention_weights = patterns["attention_weights"]
        
        # Use last token as target if not specified
        if target_token_idx is None:
            target_token_idx = attention_weights.shape[-1] - 1
        
        # Extract attention from target token to all other tokens
        # Shape: (num_layers, num_heads, seq_len)
        target_attention = attention_weights[:, 0, :, target_token_idx, :]
        
        # Aggregate across layers and heads
        # Different aggregation strategies
        importance_mean = target_attention.mean(dim=(0, 1))
        importance_max = target_attention.max(dim=1)[0].max(dim=0)[0]
        importance_last_layer = target_attention[-1].mean(dim=0)
        
        return {
            "tokens": patterns["tokens"][0],
            "importance_mean": importance_mean,
            "importance_max": importance_max,
            "importance_last_layer": importance_last_layer,
            "target_token_idx": target_token_idx,
        }