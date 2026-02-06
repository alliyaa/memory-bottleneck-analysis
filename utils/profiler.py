import torch
import time
import csv
import os
from datetime import datetime

class MemoryProfiler:
    """
    Tracks GPU memory usage during training
    NOW WITH BANDWIDTH ESTIMATION - this is the upgrade
    """
    
    def __init__(self, log_dir='results/logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def reset_peak_memory(self):
        """Clear the peak memory stats"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def get_memory_stats(self):
        """Get current memory usage in MB"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'peak': 0}
        
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'reserved': torch.cuda.memory_reserved() / 1024**2,
            'peak': torch.cuda.max_memory_allocated() / 1024**2
        }
    
    def get_memory_breakdown(self, model, optimizer=None):
        """
        Figure out where all the memory is going
        Parameters vs activations vs optimizer state
        """
        breakdown = {}
        
        # Model parameters
        model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        breakdown['model_params_mb'] = model_memory
        breakdown['model_params_bytes'] = model_memory * 1024**2
        
        # Gradients (same size as params if they exist)
        grad_memory = sum(p.grad.numel() * p.grad.element_size() 
                         for p in model.parameters() if p.grad is not None) / 1024**2
        breakdown['gradients_mb'] = grad_memory
        breakdown['gradients_bytes'] = grad_memory * 1024**2
        
        # Optimizer state (approx - Adam keeps 2 copies of params)
        if optimizer is not None and hasattr(optimizer, 'state'):
            opt_memory = 0
            for param in model.parameters():
                if param in optimizer.state:
                    # Adam has exp_avg and exp_avg_sq
                    opt_memory += 2 * param.numel() * param.element_size()
            breakdown['optimizer_state_mb'] = opt_memory / 1024**2
            breakdown['optimizer_state_bytes'] = opt_memory
        else:
            breakdown['optimizer_state_mb'] = 0
            breakdown['optimizer_state_bytes'] = 0
        
        # Everything else is activations + misc
        total_allocated = torch.cuda.memory_allocated() / 1024**2
        activations_mb = total_allocated - (breakdown['model_params_mb'] + 
                                            breakdown['gradients_mb'] + 
                                            breakdown['optimizer_state_mb'])
        breakdown['activations_misc_mb'] = max(0, activations_mb)  # can't be negative
        breakdown['activations_misc_bytes'] = max(0, activations_mb * 1024**2)
        
        return breakdown
    
    def estimate_memory_bandwidth(self, model, optimizer, step_time_seconds):
        """
        🔥 THE BIG UPGRADE - Bandwidth Estimation
        
        Estimate how much memory traffic happens per training step
        This is what Micron cares about!
        
        Memory traffic per step:
        1. Read model parameters (forward pass)
        2. Write gradients (backward pass)  
        3. Read gradients (optimizer update)
        4. Write updated parameters (optimizer update)
        5. Read/write activations (forward + backward)
        6. Read/write optimizer state (momentum, variance for Adam)
        
        Simplified model:
        - Params: read once (forward), write once (update)
        - Gradients: write once (backward), read once (update)
        - Activations: read/write multiple times (forward + backward)
        - Optimizer state: read + write (2x params for Adam)
        """
        breakdown = self.get_memory_breakdown(model, optimizer)
        
        # Parameter traffic
        param_bytes = breakdown['model_params_bytes']
        param_traffic = param_bytes * 2  # read for forward, write for update
        
        # Gradient traffic  
        grad_bytes = breakdown['gradients_bytes']
        grad_traffic = grad_bytes * 2  # write in backward, read in optimizer
        
        # Activation traffic (forward + backward, multiple reads/writes)
        # This is approximate - actual is complex
        activation_bytes = breakdown['activations_misc_bytes']
        activation_traffic = activation_bytes * 2  # forward + backward
        
        # Optimizer state traffic (Adam reads/writes momentum and variance)
        opt_state_bytes = breakdown['optimizer_state_bytes']
        opt_state_traffic = opt_state_bytes * 2  # read + write
        
        # Total bytes moved per step
        total_bytes_per_step = (param_traffic + 
                                grad_traffic + 
                                activation_traffic + 
                                opt_state_traffic)
        
        # Bandwidth = bytes / time
        if step_time_seconds > 0:
            bandwidth_bytes_per_sec = total_bytes_per_step / step_time_seconds
            bandwidth_gb_per_sec = bandwidth_bytes_per_sec / (1024**3)
        else:
            bandwidth_gb_per_sec = 0
        
        return {
            'total_bytes_per_step': total_bytes_per_step,
            'total_gb_per_step': total_bytes_per_step / (1024**3),
            'estimated_bandwidth_gb_per_sec': bandwidth_gb_per_sec,
            'param_traffic_mb': param_traffic / 1024**2,
            'grad_traffic_mb': grad_traffic / 1024**2,
            'activation_traffic_mb': activation_traffic / 1024**2,
            'opt_state_traffic_mb': opt_state_traffic / 1024**2,
        }
    
    def get_gpu_theoretical_bandwidth(self):
        """
        Try to get GPU specs for comparison
        This would be the max possible bandwidth
        """
        # Common GPU specs (GB/s)
        # User should replace with their actual GPU
        gpu_bandwidth = {
            'RTX 3090': 936,
            'RTX 4090': 1008,
            'A100': 1935,
            'H100': 3350,
            'V100': 900,
        }
        
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                for name, bw in gpu_bandwidth.items():
                    if name in gpu_name:
                        return bw
        except:
            pass
        
        return None  # unknown GPU
    
    def measure_throughput(self, func, batch_size, num_iterations=10):
        """
        Measure how fast training actually runs
        Returns samples/sec
        """
        # Warmup
        for _ in range(3):
            func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(num_iterations):
            func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        
        total_samples = batch_size * num_iterations
        throughput = total_samples / (end - start)
        
        return throughput
    
    def log_to_csv(self, filename, data):
        """Save results to CSV"""
        filepath = os.path.join(self.log_dir, filename)
        file_exists = os.path.exists(filepath)
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    
    def print_memory_summary(self, model=None, optimizer=None, step_time=None):
        """
        Enhanced summary with bandwidth estimation
        """
        stats = self.get_memory_stats()
        print(f"\n{'='*60}")
        print(f"GPU Memory Summary")
        print(f"{'='*60}")
        print(f"Allocated: {stats['allocated']:.2f} MB")
        print(f"Reserved:  {stats['reserved']:.2f} MB")
        print(f"Peak:      {stats['peak']:.2f} MB")
        
        if model and step_time:
            bandwidth = self.estimate_memory_bandwidth(model, optimizer, step_time)
            print(f"\n{'='*60}")
            print(f"Estimated Memory Bandwidth")
            print(f"{'='*60}")
            print(f"Total traffic per step: {bandwidth['total_gb_per_step']:.3f} GB")
            print(f"Estimated bandwidth: {bandwidth['estimated_bandwidth_gb_per_sec']:.2f} GB/s")
            
            theoretical = self.get_gpu_theoretical_bandwidth()
            if theoretical:
                utilization = (bandwidth['estimated_bandwidth_gb_per_sec'] / theoretical) * 100
                print(f"Theoretical peak: {theoretical} GB/s")
                print(f"Bandwidth utilization: {utilization:.1f}%")
            
            print(f"\nTraffic breakdown:")
            print(f"  Parameters:  {bandwidth['param_traffic_mb']:.2f} MB")
            print(f"  Gradients:   {bandwidth['grad_traffic_mb']:.2f} MB")
            print(f"  Activations: {bandwidth['activation_traffic_mb']:.2f} MB")
            print(f"  Opt state:   {bandwidth['opt_state_traffic_mb']:.2f} MB")
        
        print(f"{'='*60}\n")


class Timer:
    """Simple timer for measuring step time"""
    
    def __init__(self):
        self.start_time = None
        
    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
    
    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time() - self.start_time
