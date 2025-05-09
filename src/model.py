import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# Assuming quantization.py and gain_cell_attention.py are accessible
from .quantization import BitLinear, BitActivationQuant
from .gain_cell_attention import GainCellAttention

# --- Layer Normalization --- 
# Standard LayerNorm is used here. BitNet papers sometimes mention specific RMSNorm or other variants.
# For simplicity, we start with standard LayerNorm. Quantizing LayerNorm is also possible but adds complexity.
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        # Apply Layer Normalization
        # Adding epsilon for numerical stability
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# --- Transformer Block --- 
class Block(nn.Module):
    """ Transformer block combining Gain-Cell Attention and BitLinear MLP."""

    def __init__(self, config):
        super().__init__()
        # Layer Normalization before attention
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # Gain-Cell Attention layer
        self.attn = GainCellAttention(config)
        # Layer Normalization before MLP
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # MLP using BitLinear layers
        self.mlp = nn.Sequential(
            BitLinear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            # Activation function - Needs to be compatible with 8-bit quantization if applied here
            # Common choices include GELU or SwiGLU. BitNet might specify a particular one.
            # Using nn.ReLU for simplicity, but replace if needed.
            # Apply activation quantization after the activation function
            nn.ReLU(), # Replace with GELU or other if specified by BitNet/GainCell papers
            BitActivationQuant(), # Quantize activation output
            BitLinear(4 * config.n_embd, config.n_embd, bias=config.bias),
            BitActivationQuant(), # Quantize output of second linear layer
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        """ Forward pass for the Transformer Block."""
        # Residual connection around Attention
        x = x + self.attn(self.ln_1(x))
        # Residual connection around MLP
        x = x + self.mlp(self.ln_2(x))
        return x

# --- Model Configuration --- 
@dataclass
class BitNetGainCellConfig:
    block_size: int = 1024 # Max sequence length
    vocab_size: int = 50304 # GPT-2 vocab size, adjust as needed
    n_layer: int = 12      # Number of transformer blocks
    n_head: int = 12       # Number of attention heads
    n_embd: int = 768      # Embedding dimension
    dropout: float = 0.1   # Dropout rate
    bias: bool = False     # Use bias in Linears and LayerNorms?

# --- Full Model --- 
class BitNetGainCellLLM(nn.Module):
    """ The full BitNet-based LLM with Gain-Cell Attention."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Token and Position Embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Dropout for embeddings
            drop = nn.Dropout(config.dropout),
            # Transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final Layer Normalization
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # Final linear layer (Language Model Head)
        # Note: The final layer might or might not be quantized depending on the specific BitNet variant.
        # Using standard nn.Linear for now, but could be replaced with BitLinear if needed.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: share weights between token embedding and final linear layer
        self.transformer.wte.weight = self.lm_head.weight

        # Optional: Activation quantization for embeddings/final output
        self.emb_act_quant = BitActivationQuant()
        self.head_act_quant = BitActivationQuant()

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("mlp.3.weight"): # Second linear layer in MLP
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params()/1e6:.2f} M")

    def get_num_params(self, non_embedding=True):
        """ Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Exclude embedding parameters if requested (often done for reporting core model size)
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """ Initialize weights."""
        if isinstance(module, (nn.Linear, BitLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the LLM.

        Args:
            idx (torch.Tensor): Input sequence of token indices (B, T).
            targets (torch.Tensor, optional): Target sequence of token indices (B, T). If provided, computes loss.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Logits and Loss (if targets provided).
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # Get token and position embeddings
        tok_emb = self.transformer.wte(idx) # shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # shape (1, t, n_embd)
        
        # Quantize embeddings if needed
        x = self.emb_act_quant(tok_emb + pos_emb)
        x = self.transformer.drop(x)
        
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final layer normalization
        x = self.transformer.ln_f(x)

        # Quantize output of final layer norm
        x = self.head_act_quant(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

# Example Usage (Conceptual)
if __name__ == '__main__':
    config = BitNetGainCellConfig(
        block_size=64, # Smaller block size for testing
        vocab_size=500, # Smaller vocab size for testing
        n_layer=2,     # Fewer layers for testing
        n_head=4,      # Fewer heads for testing
        n_embd=128,    # Smaller embedding dim for testing
        dropout=0.1,
        bias=False
    )
    model = BitNetGainCellLLM(config)
    print(model)

    # Example input
    batch_size = 4
    seq_len = config.block_size
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print("Input indices shape:", idx.shape)

    # Forward pass (training mode)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(idx, targets)
    print("Logits shape (training):", logits.shape)
    print("Loss:", loss.item() if loss is not None else "N/A")

    # Forward pass (inference mode)
    logits_inf, _ = model(idx)
    print("Logits shape (inference):", logits_inf.shape)

