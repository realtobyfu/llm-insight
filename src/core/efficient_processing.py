"""Memory-efficient processing utilities for large models"""

import gc
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Manage memory usage during model analysis"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "free_gb": self.max_memory_gb - allocated,
            }
        else:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "used_gb": memory_info.rss / 1024**3,
                "available_gb": psutil.virtual_memory().available / 1024**3,
            }
    
    def check_memory_available(self, required_gb: float) -> bool:
        """Check if enough memory is available"""
        usage = self.get_memory_usage()
        if torch.cuda.is_available():
            return usage["free_gb"] >= required_gb
        else:
            return usage["available_gb"] >= required_gb
    
    @contextmanager
    def memory_efficient_mode(self):
        """Context manager for memory-efficient operations"""
        # Clear cache before
        self.clear_memory()
        
        # Set PyTorch to use less memory
        old_cublas_workspace_config = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        
        try:
            yield
        finally:
            # Restore settings
            torch.backends.cuda.matmul.allow_tf32 = old_cublas_workspace_config
            
            # Clear cache after
            self.clear_memory()
    
    def clear_memory(self):
        """Clear GPU/CPU memory caches"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class ChunkedProcessor:
    """Process large inputs in chunks to save memory"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.memory_manager = MemoryManager()
    
    def process_long_sequence(
        self,
        model_wrapper,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Process long sequences in chunks
        
        Args:
            model_wrapper: Model wrapper instance
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Aggregated results from all chunks
        """
        seq_len = input_ids.shape[1]
        
        if seq_len <= self.chunk_size:
            # Process normally if within chunk size
            return model_wrapper.forward(input_ids, attention_mask)
        
        logger.info(f"Processing long sequence of length {seq_len} in chunks")
        
        # Process in chunks
        all_hidden_states = []
        all_attentions = []
        
        for start_idx in range(0, seq_len - self.overlap, self.chunk_size - self.overlap):
            end_idx = min(start_idx + self.chunk_size, seq_len)
            
            # Extract chunk
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            chunk_attention_mask = None
            if attention_mask is not None:
                chunk_attention_mask = attention_mask[:, start_idx:end_idx]
            
            # Process chunk
            with self.memory_manager.memory_efficient_mode():
                chunk_outputs = model_wrapper.forward(
                    chunk_input_ids,
                    chunk_attention_mask
                )
            
            # Store results
            if start_idx == 0:
                # First chunk - keep all
                all_hidden_states.append(chunk_outputs["last_hidden_state"])
                if chunk_outputs["attentions"] is not None:
                    all_attentions.append([att for att in chunk_outputs["attentions"]])
            else:
                # Subsequent chunks - skip overlap
                skip = self.overlap if start_idx > 0 else 0
                all_hidden_states.append(chunk_outputs["last_hidden_state"][:, skip:])
                
                if chunk_outputs["attentions"] is not None:
                    for i, att in enumerate(chunk_outputs["attentions"]):
                        if i < len(all_attentions):
                            # Append to existing layer, skipping overlap
                            all_attentions[i].append(att[:, :, skip:, skip:])
        
        # Concatenate results
        final_hidden_states = torch.cat(all_hidden_states, dim=1)
        
        final_attentions = None
        if all_attentions and all_attentions[0]:
            final_attentions = []
            for layer_chunks in all_attentions:
                # Concatenate attention matrices for each layer
                layer_attention = self._merge_attention_chunks(layer_chunks)
                final_attentions.append(layer_attention)
        
        return {
            "last_hidden_state": final_hidden_states,
            "attentions": final_attentions,
            "hidden_states": None,  # Not supported in chunked mode
            "activations": {},
        }
    
    def _merge_attention_chunks(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """Merge attention chunks into single attention matrix"""
        if len(chunks) == 1:
            return chunks[0]
        
        # This is a simplified merge - in practice, you'd need more sophisticated merging
        # that properly handles the overlap regions
        batch_size = chunks[0].shape[0]
        num_heads = chunks[0].shape[1]
        
        # Calculate total sequence length
        total_len = sum(chunk.shape[2] for chunk in chunks)
        
        # Create full attention matrix
        full_attention = torch.zeros(batch_size, num_heads, total_len, total_len)
        
        # Fill in the chunks (simplified - doesn't handle overlaps properly)
        pos = 0
        for chunk in chunks:
            chunk_len = chunk.shape[2]
            full_attention[:, :, pos:pos+chunk_len, pos:pos+chunk_len] = chunk
            pos += chunk_len
        
        return full_attention


class GradientCheckpointing:
    """Enable gradient checkpointing for memory efficiency during training"""
    
    @staticmethod
    def enable_for_model(model: nn.Module):
        """Enable gradient checkpointing for supported models"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")
    
    @staticmethod
    def checkpoint_forward(module: nn.Module, *args, **kwargs):
        """Wrapper for checkpoint-enabled forward pass"""
        return checkpoint(module, *args, **kwargs)


class StreamingAnalyzer:
    """Analyze text in a streaming fashion for very long documents"""
    
    def __init__(self, analyzer, window_size: int = 512, stride: int = 256):
        self.analyzer = analyzer
        self.window_size = window_size
        self.stride = stride
    
    def analyze_document(
        self,
        document: str,
        methods: List[str] = ["attention"],
    ) -> Iterator[Dict[str, Any]]:
        """Analyze document in streaming windows
        
        Args:
            document: Long document text
            methods: Analysis methods to apply
        
        Yields:
            Analysis results for each window
        """
        # Tokenize full document first to get proper tokens
        tokenizer = self.analyzer.model_wrapper.tokenizer
        tokens = tokenizer.tokenize(document)
        
        # Process in windows
        for start_idx in range(0, len(tokens) - self.window_size + 1, self.stride):
            end_idx = min(start_idx + self.window_size, len(tokens))
            
            # Get window text
            window_tokens = tokens[start_idx:end_idx]
            window_text = tokenizer.convert_tokens_to_string(window_tokens)
            
            # Analyze window
            result = self.analyzer.analyze(window_text, methods=methods, use_cache=False)
            
            # Add window metadata
            result["window_info"] = {
                "start_token": start_idx,
                "end_token": end_idx,
                "window_text": window_text[:100] + "..." if len(window_text) > 100 else window_text,
            }
            
            yield result


class ModelQuantization:
    """Utilities for model quantization to reduce memory usage"""
    
    @staticmethod
    def quantize_model(model: nn.Module, dtype: torch.dtype = torch.float16) -> nn.Module:
        """Quantize model to use less memory
        
        Args:
            model: Model to quantize
            dtype: Target data type (float16 or bfloat16)
        
        Returns:
            Quantized model
        """
        if dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(f"Unsupported dtype for quantization: {dtype}")
        
        # Check if GPU supports the dtype
        if torch.cuda.is_available():
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                logger.warning("BFloat16 not supported, falling back to Float16")
                dtype = torch.float16
        
        # Convert model
        model = model.to(dtype)
        
        # Update model config if available
        if hasattr(model, 'config'):
            model.config.torch_dtype = dtype
        
        logger.info(f"Model quantized to {dtype}")
        return model
    
    @staticmethod
    def dynamic_quantization(model: nn.Module) -> nn.Module:
        """Apply dynamic quantization for CPU inference"""
        if torch.cuda.is_available():
            logger.warning("Dynamic quantization is primarily for CPU inference")
        
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8
        )
        
        logger.info("Applied dynamic quantization")
        return quantized_model


def estimate_memory_usage(
    model_wrapper,
    batch_size: int,
    seq_length: int,
) -> Dict[str, float]:
    """Estimate memory usage for given input size
    
    Args:
        model_wrapper: Model wrapper instance
        batch_size: Batch size
        seq_length: Sequence length
    
    Returns:
        Dictionary with memory estimates
    """
    hidden_size = model_wrapper.get_hidden_size()
    num_layers = model_wrapper.get_num_layers()
    num_heads = model_wrapper.get_num_attention_heads()
    vocab_size = model_wrapper.get_vocab_size()
    
    # Estimate memory usage (simplified)
    # Embeddings
    embedding_memory = vocab_size * hidden_size * 4  # float32
    
    # Attention weights
    attention_memory = (
        batch_size * num_layers * num_heads * seq_length * seq_length * 4
    )
    
    # Hidden states
    hidden_memory = batch_size * num_layers * seq_length * hidden_size * 4
    
    # Model parameters (rough estimate)
    param_memory = sum(p.numel() * p.element_size() for p in model_wrapper.model.parameters())
    
    total_memory = (
        embedding_memory + attention_memory + hidden_memory + param_memory
    ) / (1024**3)  # Convert to GB
    
    return {
        "embedding_gb": embedding_memory / (1024**3),
        "attention_gb": attention_memory / (1024**3),
        "hidden_gb": hidden_memory / (1024**3),
        "parameters_gb": param_memory / (1024**3),
        "total_gb": total_memory,
    }