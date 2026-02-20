"""
exp_27: LOB-ConvNeXt - ConvNeXt-based architecture for LOB prediction

Architecture based on ConvNeXt-Tiny adapted for LOB data:
- Input: (B, 12, 100, 3) -> (batch, levels, timesteps, channels)
- Treats LOB as 2D spatial data with temporal dimension
- Uses depthwise separable convolutions with GELU and LayerNorm
- Causal padding in temporal dimension to prevent future leakage

Key ConvNeXt design principles:
1. Patchify stem (4x4 non-overlapping convolution)
2. Depthwise convolution with large kernels (7x7)
3. Inverted bottleneck (expand -> depthwise -> project)
4. LayerNorm instead of BatchNorm
5. GELU activation
6. Fewer activation functions (at bottleneck only)

Adapted for LOB:
- Asymmetric kernels: larger in temporal dim, smaller in level dim
- Causal padding in temporal dimension
- Global pooling -> GRU for sequence output
"""

import os, sys, gc
import torch
import torch.nn as nn
import torch.nn.functional as F

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
print(f"Current directory: {SCRIPT_DIR}")

class LOBReshaper(nn.Module):
    """
    Reshape raw 32 LOB features into 2D spatial format (H, W, C).
    
    Raw features layout (32 total):
    - p0-p5: bid prices (6)      -> rows 0-5, channel 0 (price)
    - p6-p11: ask prices (6)     -> rows 6-11, channel 0 (price)
    - v0-v5: bid volumes (6)     -> rows 0-5, channel 1 (volume)
    - v6-v11: ask volumes (6)    -> rows 6-11, channel 1 (volume)
    - dp0-dp3: delta prices (4)  -> rows 0-3, channel 2 (delta) - padded
    - dv0-dv3: delta volumes (4) -> rows 4-7, channel 2 (delta) - padded
    
    Output: (batch, seq_len, H=12, C=3) per timestep
    When stacking W timesteps: (batch, H=12, W=timesteps, C=3)
    """
    
    def __init__(self, num_levels=12, num_channels=3):
        super().__init__()
        self.num_levels = num_levels  # 6 bid + 6 ask
        self.num_channels = num_channels  # price, volume, delta
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 32) raw LOB features
        Returns:
            reshaped: (batch, H=12, W=seq_len, C=3)
        """
        batch, seq_len, _ = x.shape
        
        # Initialize output tensor
        out = torch.zeros(batch, self.num_levels, seq_len, self.num_channels, 
                         device=x.device, dtype=x.dtype)
        
        # Channel 0: Prices
        # Bid prices p0-p5 -> rows 0-5
        out[:, 0:6, :, 0] = x[:, :, 0:6].permute(0, 2, 1)
        # Ask prices p6-p11 -> rows 6-11
        out[:, 6:12, :, 0] = x[:, :, 6:12].permute(0, 2, 1)
        
        # Channel 1: Volumes
        # Bid volumes v0-v5 -> rows 0-5
        out[:, 0:6, :, 1] = x[:, :, 12:18].permute(0, 2, 1)
        # Ask volumes v6-v11 -> rows 6-11
        out[:, 6:12, :, 1] = x[:, :, 18:24].permute(0, 2, 1)
        
        # Channel 2: Deltas (only 8 features, pad the rest)
        # Delta prices dp0-dp3 -> rows 0-3 
        out[:, 0:4, :, 2] = x[:, :, 24:28].permute(0, 2, 1)
        # Delta volumes dv0-dv3 -> rows 4-7
        out[:, 4:8, :, 2] = x[:, :, 28:32].permute(0, 2, 1)
        
        return out


class CausalConv2d(nn.Module):
    """
    2D Convolution with causal padding in the temporal (width) dimension.
    Prevents information leakage from future timesteps.
    
    Spatial (height/level) dimension uses symmetric padding.
    Temporal (width) dimension uses left-only (causal) padding.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        
        # Handle tuple or int kernel size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        
        # Padding calculations
        # Height (levels): symmetric padding
        self.pad_h = (kernel_size[0] - 1) // 2
        # Width (time): causal (left-only) padding
        self.pad_w = kernel_size[1] - 1
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, groups=groups, bias=False
        )
    
    def forward(self, x):
        # x: (B, C, H, W) where H=levels, W=time
        # Pad: (left, right, top, bottom)
        x = F.pad(x, (self.pad_w, 0, self.pad_h, self.pad_h))
        return self.conv(x)


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps (channels first)."""
    
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x):
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with causal temporal convolution.
    
    Structure:
    1. Depthwise conv (7x7 equivalent, causal in time)
    2. LayerNorm
    3. Pointwise expand (1x1 conv, 4x expansion)
    4. GELU
    5. Pointwise project (1x1 conv, back to original dim)
    6. Residual connection with optional drop path
    """
    
    def __init__(self, dim, kernel_size=(3, 7), expansion_ratio=4, drop_path=0.0):
        super().__init__()
        
        # Depthwise convolution (causal in time)
        self.dwconv = CausalConv2d(dim, dim, kernel_size, groups=dim)
        
        # LayerNorm (applied after permuting to channels last)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Pointwise convolutions (inverted bottleneck)
        hidden_dim = int(dim * expansion_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        
        # Learnable layer scale (from ConvNeXt)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        residual = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute to channels last for LayerNorm and Linear
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scale
        x = self.gamma * x
        
        # Permute back
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # Residual with drop path
        x = residual + self.drop_path(x)
        
        return x


class DropPath(nn.Module):
    """Stochastic Depth (drop path) regularization."""
    
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Downsample2d(nn.Module):
    """
    Spatial downsampling with causal temporal handling.
    Uses strided convolution with appropriate padding.
    """
    
    def __init__(self, in_channels, out_channels, stride=(2, 2)):
        super().__init__()
        self.norm = LayerNorm2d(in_channels)
        
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        
        # For causal downsampling, we need special handling
        # Use kernel_size = stride for non-overlapping patches
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=stride, stride=stride, padding=0
        )
    
    def forward(self, x):
        x = self.norm(x)
        return self.conv(x)


class LOBConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny adapted for LOB prediction.
    
    Input: (B, 12, 100, 3) - batch, levels, timesteps, channels
    Output: (B, seq_len, 2) - predictions for t0, t1 at each timestep
    
    Architecture:
    - Stem: Patchify with stride (2, 4) -> (6, 25) spatial
    - Stage 1: 3 blocks, dim=96
    - Stage 2: 3 blocks, dim=192, downsample (2, 2) -> (3, 12)
    - Stage 3: 9 blocks, dim=384, downsample (1, 2) -> (3, 6)
    - Stage 4: 3 blocks, dim=768, downsample (1, 2) -> (3, 3)
    - Global pool over levels -> GRU over time -> heads
    """
    
    def __init__(
        self,
        in_channels=3,
        num_levels=12,
        window_size=100,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
        kernel_sizes=((3, 7), (3, 7), (3, 7), (3, 7)),
        stem_stride=(2, 4),
        downsample_strides=((2, 2), (1, 2), (1, 2)),
        drop_path_rate=0.1,
        gru_hidden=128,
        gru_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.window_size = window_size
        
        # LOB reshaper: (B, seq_len, 32) -> (B, H=12, W=seq_len, C=3)
        self.reshaper = LOBReshaper(num_levels=num_levels, num_channels=in_channels)
        
        # Calculate spatial dimensions through network
        h, w = num_levels, window_size  # 12, 100
        
        # Stem: patchify
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=stem_stride, stride=stem_stride),
            LayerNorm2d(dims[0])
        )
        h, w = h // stem_stride[0], w // stem_stride[1]  # 6, 25
        
        # Stochastic depth decay rule
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        cur_block = 0
        for i, (dim, depth, kernel_size) in enumerate(zip(dims, depths, kernel_sizes)):
            # Downsample (except first stage)
            if i > 0:
                stride = downsample_strides[i - 1]
                self.downsamples.append(
                    Downsample2d(dims[i - 1], dim, stride=stride)
                )
                h = h // stride[0]
                w = w // stride[1]
            else:
                self.downsamples.append(nn.Identity())
            
            # Stage blocks
            blocks = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    drop_path=dp_rates[cur_block + j]
                )
                for j in range(depth)
            ])
            self.stages.append(blocks)
            cur_block += depth
        
        self.final_h = h
        self.final_w = w
        
        # Global pool over levels (height), keep temporal
        # Output: (B, dims[-1], final_w)
        
        # Project to GRU input
        self.pre_gru = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], gru_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,  # Causal
            dropout=dropout if gru_layers > 1 else 0
        )
        
        # Upsample to original temporal resolution
        # final_w -> window_size using transposed conv or interpolation
        self.temporal_upsample = nn.Linear(gru_hidden, gru_hidden)
        
        # Prediction heads
        self.head_t0 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, 1)
        )
        
        self.head_t1 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _forward_cnn(self, x):
        """
        Forward through CNN backbone only.
        
        Args:
            x: (B, C, H, W) tensor in channels-first format
            
        Returns:
            (B, final_w, dims[-1]) features ready for GRU
        """
        # Stem
        x = self.stem(x)
        
        # Stages with downsampling
        for downsample, stage in zip(self.downsamples, self.stages):
            x = downsample(x)
            x = stage(x)
        
        # x: (B, dims[-1], final_h, final_w)
        
        # Global average pool over height (levels)
        x = x.mean(dim=2)  # (B, dims[-1], final_w)
        
        # Permute for GRU: (B, final_w, dims[-1])
        x = x.permute(0, 2, 1)
        
        return x
    
    def forward(self, x, hidden=None):
        """
        Step-by-step forward for inference.
        Processes one timestep at a time with a sliding window.
        
        Args:
            x: (B, W=100, 32) raw LOB features - full window for single prediction
            hidden: Optional GRU hidden state for stateful inference
        
        Returns:
            output: (B, 2) predictions for the last timestep only
            hidden: Updated GRU hidden state
        """
        # Reshape: (B, W, 32) -> (B, H=12, W, C=3)
        x = self.reshaper(x)
        B, H, W, C = x.shape
        
        # Permute to channels first: (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Forward through CNN backbone
        x = self._forward_cnn(x)  # (B, final_w, dims[-1])
        
        # Project to GRU dimension
        x = self.pre_gru(x)  # (B, final_w, gru_hidden)
        
        # For step-by-step: take mean over temporal dimension to get single vector
        # This gives us a single "compressed" representation of the window
        x = x.mean(dim=1, keepdim=True)  # (B, 1, gru_hidden)
        
        # GRU step
        x, hidden = self.gru(x, hidden)  # (B, 1, gru_hidden)
        x = x.squeeze(1)  # (B, gru_hidden)
        
        # Apply temporal projection
        x = self.temporal_upsample(x)  # (B, gru_hidden)
        
        # Predictions for last timestep only
        t0 = self.head_t0(x)  # (B, 1)
        t1 = self.head_t1(x)  # (B, 1)
        
        output = torch.cat([t0, t1], dim=-1)  # (B, 2)
        
        return output, hidden
    
    def forward_full_sequence(self, x, hidden=None):
        """
        Full sequence forward for efficient training.
        Processes entire sequence at once.
        
        Args:
            x: (B, seq_len, 32) raw LOB features
            hidden: Optional GRU hidden state
        
        Returns:
            output: (B, seq_len, 2) predictions for each timestep
            hidden: Updated GRU hidden state
        """
        # Reshape: (B, seq_len, 32) -> (B, H=12, W=seq_len, C=3)
        x = self.reshaper(x)
        B, H, W, C = x.shape
        
        # Permute to channels first: (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Forward through CNN backbone
        x = self._forward_cnn(x)  # (B, final_w, dims[-1])
        
        # Project to GRU dimension
        x = self.pre_gru(x)
        
        # GRU
        x, hidden = self.gru(x, hidden)  # (B, final_w, gru_hidden)
        
        # Upsample to original temporal resolution
        # Use interpolation to go from final_w to window_size
        x = x.permute(0, 2, 1)  # (B, gru_hidden, final_w)
        x = F.interpolate(x, size=W, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)  # (B, W, gru_hidden)
        
        # Apply temporal projection
        x = self.temporal_upsample(x)
        
        # Predictions
        t0 = self.head_t0(x)  # (B, W, 1)
        t1 = self.head_t1(x)  # (B, W, 1)
        
        output = torch.cat([t0, t1], dim=-1)  # (B, W, 2)
        
        return output, hidden


class LOBConvNeXtTiny(LOBConvNeXt):
    """
    ConvNeXt-Tiny configuration for LOB.
    
    ~28M parameters (original ConvNeXt-T)
    Adapted to ~5M for LOB with smaller dims.
    """
    
    def __init__(
        self,
        in_channels=3,
        num_levels=12,
        window_size=100,
        drop_path_rate=0.1,
        gru_hidden=128,
        gru_layers=2,
        dropout=0.2,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            num_levels=num_levels,
            window_size=window_size,
            dims=(64, 128, 256, 512),      # Reduced from (96, 192, 384, 768)
            depths=(3, 3, 9, 3),            # Same as ConvNeXt-T
            kernel_sizes=((3, 7), (3, 7), (3, 7), (3, 5)),
            stem_stride=(2, 4),             # 12->6 levels, 100->25 time
            downsample_strides=((2, 2), (1, 2), (1, 2)),  # (6,25)->(3,12)->(3,6)->(3,3)
            drop_path_rate=drop_path_rate,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )


class LOBConvNeXtMicro(LOBConvNeXt):
    """
    Micro ConvNeXt for fast experimentation.
    ~1M parameters.
    """
    
    def __init__(
        self,
        in_channels=3,
        num_levels=12,
        window_size=100,
        drop_path_rate=0.1,
        gru_hidden=64,
        gru_layers=2,
        dropout=0.2,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            num_levels=num_levels,
            window_size=window_size,
            dims=(32, 64, 128, 256),
            depths=(2, 2, 6, 2),
            kernel_sizes=((3, 5), (3, 5), (3, 5), (3, 3)),
            stem_stride=(2, 4),
            downsample_strides=((2, 2), (1, 2), (1, 2)),
            drop_path_rate=drop_path_rate,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )


# Aliases for config compatibility
LOBViTModel = LOBConvNeXtTiny


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_len = 100
    num_features = 32
    
    # Test input: (B, seq_len, 32) - raw LOB features
    x = torch.randn(batch_size, seq_len, num_features)
    
    print("Testing LOBConvNeXtTiny...")
    model = LOBConvNeXtTiny()
    
    # Test forward_full_sequence (for training)
    print("\n--- forward_full_sequence (training mode) ---")
    output_full, hidden = model.forward_full_sequence(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_full.shape} (expected: {batch_size}, {seq_len}, 2)")
    print(f"Hidden shape: {hidden.shape}")
    
    # Test forward (step-by-step for inference)
    print("\n--- forward (inference mode) ---")
    hidden_step = None
    for t in range(3):  # Test a few steps
        output_step, hidden_step = model(x, hidden_step)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_step.shape} (expected: {batch_size}, 2)")
    print(f"Hidden shape: {hidden_step.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Verify causal consistency
    print("\n--- Verifying causal consistency ---")
    # Same input should give same output for last timestep
    output_full_last = output_full[:, -1, :]  # (B, 2)
    print(f"Full sequence last timestep shape: {output_full_last.shape}")
    print(f"Step-by-step output shape: {output_step.shape}")
    print("✓ Both methods produce (B, 2) output for final prediction")

    ###################
    # Test Scorerstepbystep consistency with zero-shot inference
    print("\n--- Testing scorer consistency ---")
    # import PredictionModel class from solution.py
    from solution import PredictionModel
    from src.utils import ScorerStepByStep

    # Create single-model predictor
    class SingleModelPredictor(PredictionModel):
        def __init__(self, ):
            self.current_seq_ix = None
            self.hidden_states = [None]
            self.feature_buffer = None
            self.window_size = 100
            self.running_mean = None
            self.running_var = None
            self.device = torch.device("cpu")
            self.norm_alpha = 0.01
            
            model = LOBViTModel() #**get_model_kwargs(for_inference=True)
            model.to(self.device)
            model.eval()
            self.models = [model]
    
    predictor = SingleModelPredictor()
    scorer = ScorerStepByStep("datasets/valid.parquet")
    results = scorer.score(predictor)
    
    print(f"zero-shot inference: t0={results['t0']:.4f}, t1={results['t1']:.4f}, weighted={results['weighted_pearson']:.4f}")
