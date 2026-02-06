import torch
from torch.utils.data import DataLoader
import sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.model import create_small_gpt, SimpleDataset
from utils.profiler import MemoryProfiler, Timer
from train import train_step

def run_sequence_length_experiment(seq_lengths=[64, 128, 256, 512, 1024], batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("SEQUENCE LENGTH SCALING WITH BANDWIDTH TRACKING")
    print("="*60)
    
    profiler = MemoryProfiler()
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nTesting seq_len = {seq_len}")
        
        try:
            model = create_small_gpt(seq_len=seq_len).to(device)
            dataset = SimpleDataset(num_samples=200, seq_len=seq_len)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
            
            profiler.reset_peak_memory()
            torch.cuda.empty_cache()
            
            for i, batch in enumerate(dataloader):
                if i >= 3: break
                _ = train_step(model, batch, optimizer, device)
            
            profiler.reset_peak_memory()
            
            timer = Timer()
            total_time = 0
            num_steps = 0
            
            for i, batch in enumerate(dataloader):
                if i >= 20: break
                timer.start()
                _ = train_step(model, batch, optimizer, device)
                total_time += timer.stop()
                num_steps += 1
            
            mem_stats = profiler.get_memory_stats()
            breakdown = profiler.get_memory_breakdown(model, optimizer)
            avg_step_time = total_time / num_steps
            bandwidth_stats = profiler.estimate_memory_bandwidth(model, optimizer, avg_step_time)
            
            result = {
                'seq_len': seq_len,
                'peak_memory_mb': mem_stats['peak'],
                'avg_step_time_s': avg_step_time,
                'throughput_samples_per_sec': batch_size / avg_step_time,
                'tokens_per_sec': batch_size * seq_len / avg_step_time,
                'estimated_bandwidth_gb_per_sec': bandwidth_stats['estimated_bandwidth_gb_per_sec'],
                'activations_mb': breakdown['activations_misc_mb'],
            }
            results.append(result)
            print(f"✓ Peak Memory: {mem_stats['peak']:.2f} MB, Bandwidth: {bandwidth_stats['estimated_bandwidth_gb_per_sec']:.2f} GB/s")
            
            del model, optimizer, dataloader
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ OOM at seq_len={seq_len}")
                results.append({'seq_len': seq_len, 'peak_memory_mb': -1, 'avg_step_time_s': -1, 
                               'throughput_samples_per_sec': -1, 'tokens_per_sec': -1, 
                               'estimated_bandwidth_gb_per_sec': -1, 'activations_mb': -1})
                torch.cuda.empty_cache()
                break
    
    df = pd.DataFrame(results)
    df.to_csv('results/logs/sequence_length_scaling.csv', index=False)
    print(f"\nSaved to results/logs/sequence_length_scaling.csv")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    run_sequence_length_experiment(seq_lengths=[64, 128, 256, 512, 1024, 2048], batch_size=8)
