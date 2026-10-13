"""
exp_24: LiT-inspired LOB Transformer for Regression

Architecture adapted from "LiT: Limit Order Book Transformer for High-Frequency Trading"

Key Components:
1. 2D Patch Embedding: LOB data arranged as (levels × time) then patched
2. Transformer Encoder: Self-attention on patch sequence with causal mask
3. LSTM/GRU: Final temporal modeling
4. Regression Head: Linear output for t0, t1 (not classification)

Input: (batch, seq_len, 32) raw LOB features
Output: (batch, seq_len, 2) predictions for t0, t1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LOBPatchEmbedding(nn.Module):
    """
    Reshape raw LOB features into 2D spatial arrangement, then create patches.
    
    Raw features layout (32 total):
    - p0-p5: bid prices (6)
    - p6-p11: ask prices (6)
    - v0-v5: bid volumes (6)
    - v6-v11: ask volumes (6)
    - dp0-dp3: delta prices (4)
    - dv0-dv3: delta volumes (4)
    
    Rearranged as 2D (levels=6, features_per_level=4):
    For each level i:
        [bid_price_i, ask_price_i, bid_vol_i, ask_vol_i]
    Plus delta features appended
    
    Alternative: Keep as 1D and use linear projection (simpler, works well)
    """
    
    def __init__(self, input_dim=32, embed_dim=64, patch_size=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # For single-step inference, we use 1D projection
        # patch_size=1 means each timestep is a patch
        self.projection = nn.Linear(input_dim * patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            patches: (batch, num_patches, embed_dim)
        """
        batch, seq_len, _ = x.shape
        
        # Pad sequence if not divisible by patch_size
        if seq_len % self.patch_size != 0:
            pad_len = self.patch_size - (seq_len % self.patch_size)
            x = F.pad(x, (0, 0, pad_len, 0))  # Pad at beginning (causal)
            seq_len = x.shape[1]
        
        # Reshape to patches: (batch, num_patches, patch_size * input_dim)
        num_patches = seq_len // self.patch_size
        x = x.view(batch, num_patches, self.patch_size * self.input_dim)
        
        # Project to embedding dimension
        x = self.projection(x)
        x = self.norm(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patches."""
    
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            x + positional encoding
        """
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        return x + self.pos_embedding(positions)


class CausalTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with causal (lower triangular) attention mask.
    Prevents future information leakage.
    """
    
    def __init__(self, embed_dim, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (batch, seq_len, embed_dim)
            attn_mask: (seq_len, seq_len) causal mask
        """
        # Self-attention with causal mask
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class CausalTransformerEncoder(nn.Module):
    """Stack of causal transformer encoder layers."""
    
    def __init__(self, embed_dim, num_layers=2, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            CausalTransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def generate_causal_mask(self, seq_len, device):
        """Generate causal attention mask (upper triangular = -inf)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        """
        seq_len = x.shape[1]
        attn_mask = self.generate_causal_mask(seq_len, x.device)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        return x


class LiTModel(nn.Module):
    """
    LiT-inspired LOB Transformer for Regression.
    
    Architecture:
    1. Patch Embedding: Linear projection of input features
    2. Positional Encoding: Learnable position embeddings
    3. Causal Transformer: Self-attention with causal mask
    4. GRU: Temporal modeling (replaces LSTM from original paper)
    5. Regression Head: Output t0, t1 predictions
    
    For single-step inference:
    - patch_size=1 (each timestep is a patch)
    - GRU maintains hidden state across timesteps
    """
    
    def __init__(
        self,
        input_dim=32,
        embed_dim=64,
        patch_size=1,  # 1 for single-step inference compatibility
        num_transformer_layers=2,
        num_heads=4,
        ff_dim=256,
        gru_hidden=128,
        gru_layers=2,
        output_dim=2,
        dropout=0.2,
        max_seq_len=100,
        vb=False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.vb = vb
        
        # 1. Patch Embedding
        self.patch_embed = LOBPatchEmbedding(input_dim, embed_dim, patch_size)
        
        # 2. Positional Encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # 3. Causal Transformer Encoder
        self.transformer = CausalTransformerEncoder(
            embed_dim, num_transformer_layers, num_heads, ff_dim, dropout
        )
        
        # 4. GRU for temporal modeling
        self.gru = nn.GRU(
            embed_dim, gru_hidden, gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0
        )
        
        # 5. Regression Head
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, 1, input_dim) for single-step
            hidden: GRU hidden state for stateful inference
        
        Returns:
            predictions: (batch, seq_len, 2) for t0, t1
            hidden: Updated GRU hidden state
        """
        # 1. Patch Embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        if self.vb:
            print(f"Patch embedding output: {x.shape}")

        # 2. Add Positional Encoding
        x = self.pos_encoding(x)
        if self.vb:
            print(f"After positional encoding: {x.shape}")

        # 3. Transformer Encoder
        x = self.transformer(x)  # (batch, num_patches, embed_dim)
        if self.vb:
            print(f"After transformer encoder: {x.shape}")

        # 4. GRU
        x, hidden = self.gru(x, hidden)  # (batch, num_patches, gru_hidden)
        if self.vb:
            print(f"After GRU: {x.shape}, Hidden: {hidden.shape}")

        # 5. Regression Head
        predictions = self.output_head(x)  # (batch, num_patches, 2)
        if self.vb:
            print(f"Predictions shape: {predictions.shape}")
            print(f"GRU hidden state shape: {hidden.shape}")
        return predictions, hidden


class LiTModelSingleStep(nn.Module):
    """
    LiT model optimized for single-step inference.
    
    Key differences from training:
    - Takes single timestep input
    - Maintains internal buffer for transformer context
    - GRU hidden state for temporal continuity
    """
    
    def __init__(
        self,
        input_dim=32,
        embed_dim=64,
        num_transformer_layers=2,
        num_heads=4,
        ff_dim=256,
        gru_hidden=128,
        gru_layers=2,
        output_dim=2,
        dropout=0.2,
        context_len=10,  # How many past steps for transformer context
        vb=False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.context_len = context_len
        self.vb = vb
        
        # Input projection (single step)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(embed_dim, context_len + 1)
        
        # Causal Transformer
        self.transformer = CausalTransformerEncoder(
            embed_dim, num_transformer_layers, num_heads, ff_dim, dropout
        )
        
        # GRU
        self.gru = nn.GRU(
            embed_dim, gru_hidden, gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, output_dim)
        )
    
    def forward(self, x, hidden=None, context_buffer=None):
        """
        Single-step forward pass with context buffer.
        
        Args:
            x: (batch, input_dim) single timestep features
            hidden: GRU hidden state
            context_buffer: (batch, context_len, embed_dim) past embeddings
        
        Returns:
            predictions: (batch, 2)
            hidden: Updated GRU state
            context_buffer: Updated context buffer
        """
        batch = x.shape[0]
        if self.vb:
            print(f"Input: {x.shape}, Hidden: {hidden.shape if hidden is not None else None}, Context: {context_buffer.shape if context_buffer is not None else None}")
        
        # Project input
        x_embed = self.input_proj(x.unsqueeze(1))  # (batch, 1, embed_dim)
        if self.vb:
            print(f"Projected input: {x_embed.shape}")
            
        # Build context for transformer
        if context_buffer is None:
            # Initialize with zeros
            context_buffer = torch.zeros(batch, self.context_len, self.embed_dim, device=x.device)
        if self.vb:
            print(f"Context buffer: {context_buffer.shape}")

        # Concatenate context and current embedding
        full_context = torch.cat([context_buffer, x_embed], dim=1)  # (batch, context_len+1, embed_dim)
        if self.vb:
            print(f"Full context for transformer: {full_context.shape}")

        # Add positional encoding
        full_context = self.pos_encoding(full_context)
        if self.vb:
            print(f"Full context with positional encoding: {full_context.shape}")

        # Transformer (causal attention)
        transformer_out = self.transformer(full_context)  # (batch, context_len+1, embed_dim)
        if self.vb:
            print(f"Transformer output: {transformer_out.shape}")

        # Take last position output
        current_out = transformer_out[:, -1:, :]  # (batch, 1, embed_dim)
        if self.vb:
            print(f"Current output for GRU: {current_out.shape}")

        # GRU step
        gru_out, hidden = self.gru(current_out, hidden)  # (batch, 1, gru_hidden)
        if self.vb:
            print(f"GRU output: {gru_out.shape}, Updated hidden: {hidden.shape}")

        # Predictions
        predictions = self.output_head(gru_out.squeeze(1))  # (batch, 2)
        if self.vb:
            print(f"Predictions: {predictions.shape}")
            print(f"Hidden state: {hidden.shape}")
            print(f"Context buffer: {context_buffer.shape}")

        # Update context buffer (FIFO)
        new_context = torch.cat([context_buffer[:, 1:, :], x_embed], dim=1)
        if self.vb:
            print(f"Updated context buffer: {new_context.shape}")

        return predictions, hidden, new_context


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test LiT model
    print("=" * 60)
    print("LiT Model (Training Mode)")
    print("=" * 60)
    
    model = LiTModel(
        input_dim=32,
        embed_dim=64,
        patch_size=1,
        num_transformer_layers=2,
        num_heads=4,
        ff_dim=256,
        gru_hidden=128,
        gru_layers=2,
        dropout=0.2, 
        vb=True
    )
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(4, 100, 32)  # batch=4, seq=100, features=32
    out, hidden = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Hidden shape: {hidden.shape}")
    
    print("\n" + "=" * 60)
    print("LiT Model (Single-Step Inference)")
    print("=" * 60)
    
    model_ss = LiTModelSingleStep(
        input_dim=32,
        embed_dim=64,
        num_transformer_layers=2,
        num_heads=4,
        ff_dim=256,
        gru_hidden=128,
        gru_layers=2,
        context_len=10,
        dropout=0.0  # No dropout at inference
    )
    
    print(f"Parameters: {count_parameters(model_ss):,}")
    
    # Test single-step inference
    x_step = torch.randn(1, 32)  # batch=1, features=32
    out, hidden, ctx = model_ss(x_step)
    print(f"Input shape: {x_step.shape}")
    print(f"Output shape: {out.shape}")
    
    # Simulate sequence inference
    print("\nSimulating sequence inference:")
    hidden = None
    ctx = None
    for i in range(10):
        x_step = torch.randn(1, 32)
        out, hidden, ctx = model_ss(x_step, hidden, ctx)
    print(f"After 10 steps - Output: {out.shape}, Hidden: {hidden.shape}, Context: {ctx.shape}")
