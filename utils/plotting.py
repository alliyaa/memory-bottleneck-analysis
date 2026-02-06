import matplotlib.pyplot as plt
import pandas as pd
import os, numpy as np

plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

def plot_batch_size_results(csv_path='results/logs/batch_size_scaling.csv'):
    if not os.path.exists(csv_path):
        print(f"Run experiments/batch_scaling.py first!")
        return
    
    df = pd.read_csv(csv_path)
    valid_df = df[df['peak_memory_mb'] > 0]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Batch Size Scaling with Bandwidth Analysis', fontsize=16, fontweight='bold')
    
    # Memory vs Batch
    ax1.plot(valid_df['batch_size'], valid_df['peak_memory_mb'], marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Peak GPU Memory (MB)')
    ax1.set_title('Memory Usage Scaling')
    ax1.grid(True, alpha=0.3)
    
    # Throughput vs Batch
    ax2.plot(valid_df['batch_size'], valid_df['throughput_samples_per_sec'], marker='s', linewidth=2, markersize=8, color='#3498db')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('Training Throughput')
    ax2.grid(True, alpha=0.3)
    
    # 🔥 NEW: Bandwidth vs Batch
    ax3.plot(valid_df['batch_size'], valid_df['estimated_bandwidth_gb_per_sec'], marker='^', linewidth=2, markersize=8, color='#2ecc71')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Estimated Bandwidth (GB/s)')
    ax3.set_title('Memory Bandwidth Demand')
    ax3.grid(True, alpha=0.3)
    
    # Memory Efficiency
    ax4.plot(valid_df['batch_size'], valid_df['memory_efficiency'], marker='D', linewidth=2, markersize=8, color='#9b59b6')
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Efficiency (samples/sec per MB)')
    ax4.set_title('Memory Efficiency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/batch_size_scaling.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved batch_size_scaling.png")
    plt.close()

def plot_sequence_length_results(csv_path='results/logs/sequence_length_scaling.csv'):
    if not os.path.exists(csv_path):
        return
    
    df = pd.read_csv(csv_path)
    valid_df = df[df['peak_memory_mb'] > 0]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sequence Length Scaling with Bandwidth', fontsize=16, fontweight='bold')
    
    # Memory with O(n^2) reference
    ax1.plot(valid_df['seq_len'], valid_df['peak_memory_mb'], marker='o', linewidth=2, markersize=8, color='#e74c3c', label='Actual')
    if len(valid_df) >= 2:
        x = valid_df['seq_len'].values
        y0 = valid_df.iloc[0]['peak_memory_mb']
        x0 = valid_df.iloc[0]['seq_len']
        quadratic = y0 * (x / x0) ** 2
        ax1.plot(x, quadratic, '--', linewidth=2, alpha=0.5, color='gray', label='O(n²) reference')
        ax1.legend()
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Peak Memory (MB)')
    ax1.set_title('Memory Scaling (O(n²) attention)')
    ax1.grid(True, alpha=0.3)
    
    # Throughput
    ax2.plot(valid_df['seq_len'], valid_df['tokens_per_sec'], marker='s', linewidth=2, markersize=8, color='#3498db')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Tokens/sec')
    ax2.set_title('Token Processing Throughput')
    ax2.grid(True, alpha=0.3)
    
    # 🔥 NEW: Bandwidth
    ax3.plot(valid_df['seq_len'], valid_df['estimated_bandwidth_gb_per_sec'], marker='^', linewidth=2, markersize=8, color='#2ecc71')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Bandwidth (GB/s)')
    ax3.set_title('Memory Bandwidth Demand')
    ax3.grid(True, alpha=0.3)
    
    # Activations
    ax4.plot(valid_df['seq_len'], valid_df['activations_mb'], marker='D', linewidth=2, markersize=8, color='#f39c12')
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Activation Memory (MB)')
    ax4.set_title('Activation Memory Growth')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/sequence_length_scaling.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved sequence_length_scaling.png")
    plt.close()

def plot_precision_comparison(csv_path='results/logs/precision_comparison.csv'):
    if not os.path.exists(csv_path):
        return
    
    df = pd.read_csv(csv_path)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FP32 vs FP16 with Bandwidth', fontsize=16, fontweight='bold')
    
    precisions = df['precision'].values
    x_pos = np.arange(len(precisions))
    colors = ['#e74c3c', '#2ecc71']
    
    # Memory
    ax1.bar(x_pos, df['peak_memory_mb'], color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([p.upper() for p in precisions])
    ax1.set_ylabel('Peak Memory (MB)')
    ax1.set_title('Memory Usage')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Throughput
    ax2.bar(x_pos, df['throughput_samples_per_sec'], color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([p.upper() for p in precisions])
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('Training Speed')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 🔥 NEW: Bandwidth
    ax3.bar(x_pos, df['estimated_bandwidth_gb_per_sec'], color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([p.upper() for p in precisions])
    ax3.set_ylabel('Bandwidth (GB/s)')
    ax3.set_title('Memory Bandwidth')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Step time
    ax4.bar(x_pos, df['avg_step_time_s'], color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([p.upper() for p in precisions])
    ax4.set_ylabel('Step Time (seconds)')
    ax4.set_title('Time per Training Step')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/precision_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved precision_comparison.png")
    plt.close()

def generate_all_plots():
    print("\nGenerating plots...\n")
    plot_batch_size_results()
    plot_sequence_length_results()
    plot_precision_comparison()
    print("\n✓ All plots generated in results/plots/")

if __name__ == "__main__":
    generate_all_plots()
