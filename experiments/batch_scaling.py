import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.model import create_small_gpt, SimpleDataset
from utils.profiler import MemoryProfiler, Timer
from train import train_step

def run_batch_size_experiment(batch_sizes=[4, 8, 16, 32, 64, 128], seq_len=128):
    """
    THE KEY EXPERIMENT - NOW WITH BANDWIDTH ESTIMATION
    
    Run training with different batch sizes and see when memory becomes the bottleneck.
    
    Should see:
    - Memory usage scales linearly with batch size
    - Bandwidth demand increases
    - Throughput stops improving at some point (bandwidth limit)
    - Eventually OOM crash (capacity limit)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        print("Need GPU for this experiment!")
        return
    
    print("="*60)
    print("BATCH SIZE SCALING EXPERIMENT (WITH BANDWIDTH TRACKING)")
    print("="*60)
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Sequence length: {seq_len}")
    print()
    
    profiler = MemoryProfiler()
    
    # Get theoretical GPU bandwidth for comparison
    theoretical_bw = profiler.get_gpu_theoretical_bandwidth()
    if theoretical_bw:
        print(f"Your GPU theoretical bandwidth: {theoretical_bw} GB/s")
        print("We'll compare our estimated bandwidth to this.\n")
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Testing batch_size = {batch_size}")
        print(f"{'='*60}")
        
        try:
            # Fresh model each time to avoid memory leaks
            model = create_small_gpt(seq_len=seq_len)
            model = model.to(device)
            
            # Dataset and loader
            dataset = SimpleDataset(num_samples=200, seq_len=seq_len)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
            
            # Reset memory tracking
            profiler.reset_peak_memory()
            torch.cuda.empty_cache()
            
            # Warmup - let things stabilize
            print("Warming up...")
            for i, batch in enumerate(dataloader):
                if i >= 3:
                    break
                _ = train_step(model, batch, optimizer, device)
            
            profiler.reset_peak_memory()
            
            # Timed run
            print("Running timed measurements...")
            timer = Timer()
            num_steps = 0
            total_time = 0
            
            for i, batch in enumerate(dataloader):
                if i >= 20:  # 20 steps should be enough
                    break
                
                timer.start()
                _ = train_step(model, batch, optimizer, device)
                step_time = timer.stop()
                
                total_time += step_time
                num_steps += 1
            
            # Get final memory stats
            mem_stats = profiler.get_memory_stats()
            breakdown = profiler.get_memory_breakdown(model, optimizer)
            
            # Calculate throughput
            avg_step_time = total_time / num_steps
            throughput = batch_size / avg_step_time
            tokens_per_sec = batch_size * seq_len / avg_step_time
            
            # 🔥 THE BIG UPGRADE - Bandwidth estimation
            bandwidth_stats = profiler.estimate_memory_bandwidth(model, optimizer, avg_step_time)
            
            # Memory efficiency
            mem_efficiency = throughput / mem_stats['peak'] if mem_stats['peak'] > 0 else 0
            
            # Bandwidth utilization
            bw_utilization = 0
            if theoretical_bw:
                bw_utilization = (bandwidth_stats['estimated_bandwidth_gb_per_sec'] / theoretical_bw) * 100
            
            result = {
                'batch_size': batch_size,
                'peak_memory_mb': mem_stats['peak'],
                'allocated_memory_mb': mem_stats['allocated'],
                'avg_step_time_s': avg_step_time,
                'throughput_samples_per_sec': throughput,
                'tokens_per_sec': tokens_per_sec,
                'memory_efficiency': mem_efficiency,
                'model_params_mb': breakdown['model_params_mb'],
                'activations_misc_mb': breakdown['activations_misc_mb'],
                # 🔥 NEW: Bandwidth metrics
                'estimated_bandwidth_gb_per_sec': bandwidth_stats['estimated_bandwidth_gb_per_sec'],
                'total_traffic_gb_per_step': bandwidth_stats['total_gb_per_step'],
                'bandwidth_utilization_percent': bw_utilization,
                'param_traffic_mb': bandwidth_stats['param_traffic_mb'],
                'activation_traffic_mb': bandwidth_stats['activation_traffic_mb'],
            }
            
            results.append(result)
            
            print(f"\n✓ SUCCESS")
            print(f"  Peak Memory: {mem_stats['peak']:.2f} MB")
            print(f"  Avg Step Time: {avg_step_time:.3f}s")
            print(f"  Throughput: {throughput:.2f} samples/sec")
            print(f"  Tokens/sec: {tokens_per_sec:.2f}")
            print(f"  Memory Efficiency: {mem_efficiency:.4f} samples/sec/MB")
            print(f"\n  🔥 BANDWIDTH ANALYSIS:")
            print(f"  Estimated Bandwidth: {bandwidth_stats['estimated_bandwidth_gb_per_sec']:.2f} GB/s")
            print(f"  Traffic per step: {bandwidth_stats['total_gb_per_step']:.3f} GB")
            if theoretical_bw:
                print(f"  Bandwidth utilization: {bw_utilization:.1f}% of theoretical peak")
            
            # Clean up
            del model
            del optimizer
            del dataloader
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n✗ OUT OF MEMORY at batch_size={batch_size}")
                print(f"  This is the capacity limit!")
                result = {
                    'batch_size': batch_size,
                    'peak_memory_mb': -1,
                    'allocated_memory_mb': -1,
                    'avg_step_time_s': -1,
                    'throughput_samples_per_sec': -1,
                    'tokens_per_sec': -1,
                    'memory_efficiency': -1,
                    'model_params_mb': -1,
                    'activations_misc_mb': -1,
                    'estimated_bandwidth_gb_per_sec': -1,
                    'total_traffic_gb_per_step': -1,
                    'bandwidth_utilization_percent': -1,
                    'param_traffic_mb': -1,
                    'activation_traffic_mb': -1,
                }
                results.append(result)
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    # Save results
    df = pd.DataFrame(results)
    output_file = 'results/logs/batch_size_scaling.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    
    print("\nSUMMARY:")
    print(df.to_string(index=False))
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")
    
    if len(results) > 1:
        valid_results = [r for r in results if r['peak_memory_mb'] > 0]
        
        if len(valid_results) > 1:
            first = valid_results[0]
            last = valid_results[-1]
            
            batch_ratio = last['batch_size'] / first['batch_size']
            bw_ratio = last['estimated_bandwidth_gb_per_sec'] / first['estimated_bandwidth_gb_per_sec']
            throughput_ratio = last['throughput_samples_per_sec'] / first['throughput_samples_per_sec']
            
            print(f"\nBatch size increased {batch_ratio:.1f}x")
            print(f"Bandwidth demand increased {bw_ratio:.1f}x")
            print(f"Throughput increased {throughput_ratio:.1f}x")
            
            if bw_ratio > throughput_ratio * 1.5:
                print(f"\n⚠ Bandwidth is growing faster than throughput!")
                print(f"  → We're becoming BANDWIDTH LIMITED")
                print(f"  → Memory can't keep up with compute demand")
            
            efficiencies = [r['memory_efficiency'] for r in valid_results]
            max_eff = max(efficiencies)
            for i, r in enumerate(valid_results):
                drop = (max_eff - r['memory_efficiency']) / max_eff * 100
                if drop > 20:
                    print(f"\n⚠ Memory efficiency dropped {drop:.1f}% at batch_size={r['batch_size']}")
                    print(f"  → This is where bandwidth becomes the bottleneck")
                    break
        
        if results[-1]['peak_memory_mb'] == -1:
            print(f"\n⚠ Hit OUT OF MEMORY at batch_size={results[-1]['batch_size']}")
            print(f"  → This is the CAPACITY limit")
            if len(valid_results) > 0:
                last_working = valid_results[-1]
                print(f"  → Max working batch size: {last_working['batch_size']}")
                print(f"  → Used {last_working['peak_memory_mb']:.2f} MB")
                print(f"  → Peak bandwidth: {last_working['estimated_bandwidth_gb_per_sec']:.2f} GB/s")
    
    return df


if __name__ == "__main__":
    df = run_batch_size_experiment(
        batch_sizes=[4, 8, 16, 32, 64, 128, 256],
        seq_len=128
    )
