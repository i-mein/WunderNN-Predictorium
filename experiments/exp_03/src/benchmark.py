import time
import torch
import numpy as np
from src.model import TransformerModel

def benchmark_inference():
    # Model config
    INPUT_DIM = 49
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    DIM_FEEDFORWARD = 512
    
    device = torch.device("cpu")
    model = TransformerModel(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD).to(device)
    model.eval()
    
    # Simulate Test Set
    # Test set has unknown number of sequences, but Valid has 1444 sequences * 1000 steps.
    # We process step by step.
    # Total steps ~ 1.4 million.
    # Constraints: 1 core. 60 mins.
    # 1.4M steps / 3600 sec = 388 steps per second.
    # We need > 400 steps/sec.
    
    # Create dummy input (Batch=1, Seq=10, Feat=49) 
    # Context length grows from 1 to 1000. Average ~500.
    # Transformer complexity is O(Seq^2).
    # If we re-process full history at each step, it's O(T^3) total!
    # 1000 steps * (1000^2 attention) is heavy.
    # We MUST USE KV-Caching for efficient step-by-step inference or accept O(T^2) valid inference if acceptable.
    # Let's measure 're-process full history' approach speed first.
    
    seq_len = 500 # Average case
    dummy_input = torch.randn(1, seq_len, INPUT_DIM).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    start = time.time()
    n_iters = 100
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(dummy_input)
    end = time.time()
    
    avg_time = (end - start) / n_iters
    steps_per_sec = 1.0 / avg_time
    print(f"Inference Speed (SeqLen={seq_len}): {steps_per_sec:.2f} calls/sec")
    
    # If calls/sec is for ONE step (given context), and we have 1.4M steps...
    # If 1 call takes 0.01s -> 100 steps/sec. 1.4M steps -> 14000 sec -> 4 hours. TOO SLOW.
    # Re-processing full history at every step for Transformer is likely infeasible without KV Cache or optimizations.
    # GRU is O(1) per step.
    
if __name__ == "__main__":
    benchmark_inference()
