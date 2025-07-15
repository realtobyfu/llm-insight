"""Model wrapper for unified interface across different transformer implementations"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers"""
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Forward pass through the model"""
        pass
    
    @abstractmethod
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Extract attention weights from the last forward pass"""
        pass
    
    @abstractmethod
    def get_activations(self, layer: Optional[int] = None) -> torch.Tensor:
        """Extract activations from specified layer"""
        pass


class HuggingFaceModelWrapper(BaseModelWrapper):
    """Wrapper for HuggingFace transformer models"""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            output_attentions=True,
            output_hidden_states=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Storage for intermediate outputs
        self._last_outputs = None
        self._attention_weights = None
        self._hidden_states = None
        
        # Register hooks for activation extraction
        self._register_hooks()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations"""
        self.activation_cache = {}
        
        def get_activation_hook(name):
            def hook(module, input, output):
                self.activation_cache[name] = output.detach()
            return hook
        
        # Register hooks for each layer
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "mlp" in name.lower():
                module.register_forward_hook(get_activation_hook(name))
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input text"""
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass through the model"""
        # Clear previous cache
        self.activation_cache.clear()
        
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                **kwargs
            )
        
        # Store outputs
        self._last_outputs = outputs
        self._attention_weights = outputs.attentions if hasattr(outputs, 'attentions') else None
        self._hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attentions": self._attention_weights,
            "hidden_states": self._hidden_states,
            "activations": dict(self.activation_cache),
        }
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Extract attention weights from the last forward pass"""
        if self._attention_weights is None:
            logger.warning("No attention weights available. Run forward() first.")
            return None
        
        # Stack attention weights from all layers
        # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
        return torch.stack(self._attention_weights)
    
    def get_activations(self, layer: Optional[int] = None) -> torch.Tensor:
        """Extract activations from specified layer"""
        if self._hidden_states is None:
            logger.warning("No hidden states available. Run forward() first.")
            return None
        
        if layer is None:
            # Return all hidden states
            return torch.stack(self._hidden_states)
        else:
            # Return specific layer
            if 0 <= layer < len(self._hidden_states):
                return self._hidden_states[layer]
            else:
                raise ValueError(f"Layer {layer} out of range. Model has {len(self._hidden_states)} layers.")
    
    def get_layer_names(self) -> List[str]:
        """Get names of all layers in the model"""
        return [name for name, _ in self.model.named_modules()]
    
    def get_num_layers(self) -> int:
        """Get number of transformer layers"""
        if hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model.config, 'n_layer'):
            return self.model.config.n_layer
        else:
            # Count transformer blocks
            return len([name for name in self.get_layer_names() if 'layer.' in name and 'attention' in name])
    
    def get_num_attention_heads(self) -> int:
        """Get number of attention heads"""
        if hasattr(self.model.config, 'num_attention_heads'):
            return self.model.config.num_attention_heads
        elif hasattr(self.model.config, 'n_head'):
            return self.model.config.n_head
        else:
            return 12  # Default fallback
    
    def get_hidden_size(self) -> int:
        """Get hidden dimension size"""
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        elif hasattr(self.model.config, 'n_embd'):
            return self.model.config.n_embd
        else:
            return 768  # Default fallback
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.model.config.vocab_size
    
    def decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs to text"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
    
    def get_token_strings(self, token_ids: torch.Tensor) -> List[List[str]]:
        """Get string representation of tokens"""
        batch_size = token_ids.shape[0]
        token_strings = []
        
        for i in range(batch_size):
            tokens = []
            for token_id in token_ids[i]:
                token = self.tokenizer.decode([token_id])
                tokens.append(token)
            token_strings.append(tokens)
        
        return token_strings


class ModelWrapper:
    """Factory class for creating model wrappers"""
    
    @staticmethod
    def create(
        model_name: str,
        model_type: str = "huggingface",
        **kwargs
    ) -> BaseModelWrapper:
        """Create a model wrapper based on the specified type"""
        if model_type == "huggingface":
            return HuggingFaceModelWrapper(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def list_supported_models() -> List[str]:
        """List all supported model architectures"""
        return [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "bert-base-uncased",
            "bert-base-cased",
            "bert-large-uncased",
            "bert-large-cased",
            "roberta-base",
            "roberta-large",
            "distilbert-base-uncased",
            "distilgpt2",
            "facebook/opt-125m",
            "facebook/opt-350m",
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
        ]