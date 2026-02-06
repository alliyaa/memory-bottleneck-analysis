import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.model import create_small_gpt, SimpleDataset
from utils.profiler import MemoryProfiler, Timer

def train_step_fp32(model, batch, optimizer, device):
    model.train()
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def train_step_fp16(model, batch, optimizer, scaler, device):
    model.train()
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    with autocast():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return loss.item()

def run_precision_experiment(batch_size=32, seq_len=128, num_steps=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("FP32 vs FP16 WITH BANDWIDTH TRACKING")
    print("="*60)
    
    profiler = MemoryProfiler()
    results = []
    
    for precision in ['fp32', 'fp16']:
        print(f"\nTesting {precision.upper()}")
        
        model = create_small_gpt(seq_len=seq_len).to(device)
        dataset = SimpleDataset(num_samples=200, seq_len=seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scaler = GradScaler() if precision == 'fp16' else None
        
        profiler.reset_peak_memory()
        torch.cuda.empty_cache()
        
        for i, batch in enumerate(dataloader):
            if i >= 5: break
            if precision == 'fp32':
                train_step_fp32(model, batch, optimizer, device)
            else:
                train_step_fp16(model, batch, optimizer, scaler, device)
        
        profiler.reset_peak_memory()
        
        timer = Timer()
        total_time = 0
        step_count = 0
        
        for i, batch in enumerate(dataloader):
            if step_count >= num_steps: break
            timer.start()
            if precision == 'fp32':
                train_step_fp32(model, batch, optimizer, device)
            else:
                train_step_fp16(model, batch, optimizer, scaler, device)
            total_time += timer.stop()
            step_count += 1
        
        mem_stats = profiler.get_memory_stats()
        avg_step_time = total_time / step_count
        bandwidth_stats = profiler.estimate_memory_bandwidth(model, optimizer, avg_step_time)
        
        result = {
            'precision': precision,
            'peak_memory_mb': mem_stats['peak'],
            'avg_step_time_s': avg_step_time,
            'throughput_samples_per_sec': batch_size / avg_step_time,
            'estimated_bandwidth_gb_per_sec': bandwidth_stats['estimated_bandwidth_gb_per_sec'],
        }
        results.append(result)
        print(f"✓ Memory: {mem_stats['peak']:.2f} MB, Bandwidth: {bandwidth_stats['estimated_bandwidth_gb_per_sec']:.2f} GB/s")
        
        del model, optimizer
        if scaler: del scaler
        torch.cuda.empty_cache()
    
    df = pd.DataFrame(results)
    df.to_csv('results/logs/precision_comparison.csv', index=False)
    print(f"\nSaved to results/logs/precision_comparison.csv")
    print(df.to_string(index=False))
    
    fp32_mem = results[0]['peak_memory_mb']
    fp16_mem = results[1]['peak_memory_mb']
    savings = (1 - fp16_mem / fp32_mem) * 100
    speedup = results[1]['throughput_samples_per_sec'] / results[0]['throughput_samples_per_sec']
    print(f"\nFP16 saves {savings:.1f}% memory, {speedup:.2f}x speedup")
    
    return df

if __name__ == "__main__":
    run_precision_experiment(batch_size=32, seq_len=128, num_steps=50)
