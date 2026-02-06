import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils.model import create_small_gpt, SimpleDataset
from utils.profiler import MemoryProfiler, Timer

def train_step(model, batch, optimizer, device):
    """Single training step"""
    model.train()
    
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()

def main():
    print("Starting basic training run...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type != 'cuda':
        print("WARNING: No GPU found. This won't show memory bottlenecks properly.")
        return
    
    profiler = MemoryProfiler()
    
    seq_len = 128
    model = create_small_gpt(seq_len=seq_len)
    model = model.to(device)
    
    dataset = SimpleDataset(num_samples=500, seq_len=seq_len)
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    profiler.reset_peak_memory()
    profiler.print_memory_summary()
    
    breakdown = profiler.get_memory_breakdown(model, optimizer)
    print("\nMemory Breakdown:")
    for component, memory in breakdown.items():
        if component.endswith('_mb'):
            print(f"  {component}: {memory:.2f} MB")
    
    print(f"\nTraining with batch_size={batch_size}, seq_len={seq_len}")
    
    timer = Timer()
    total_time = 0
    num_steps = 0
    
    for i, batch in enumerate(dataloader):
        if i >= 50:
            break
        timer.start()
        loss = train_step(model, batch, optimizer, device)
        step_time = timer.stop()
        total_time += step_time
        num_steps += 1
        
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
    
    avg_step_time = total_time / num_steps
    
    profiler.print_memory_summary(model, optimizer, avg_step_time)
    
    print(f"\nTraining Summary:")
    print(f"  Average loss: {loss:.4f}")
    print(f"  Average step time: {avg_step_time:.3f}s")
    print(f"  Throughput: {batch_size / avg_step_time:.2f} samples/sec")
    print(f"  Tokens/sec: {batch_size * seq_len / avg_step_time:.2f}")
    
    print("\nBasic training complete! Now run the experiments to see bottlenecks.")

if __name__ == "__main__":
    main()
