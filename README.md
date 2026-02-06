# Memory Bottleneck Analysis with Bandwidth Estimation

Figuring out when GPU memory becomes the problem and how much bandwidth we're actually using.

## What This Is

I wanted real data on how memory limits AI training. Not just "we need more VRAM" but actual numbers showing:
1. When we hit capacity limits (out of memory)
2. When we hit bandwidth limits (memory can't keep up)
3. How much memory traffic happens per training step

This project trains a small GPT transformer and stresses it until memory breaks, while measuring everything including estimated memory bandwidth.

## The Big Addition: Bandwidth Estimation

Most projects just measure memory capacity (how much VRAM you use). But Micron also cares about **memory bandwidth** - how fast data moves between GPU compute and memory.

I added bandwidth estimation that calculates:
```
Memory traffic per step = 
    params * 2 (read for forward, write for update) +
    gradients * 2 (write in backward, read in optimizer) +
    activations * 2 (forward + backward passes) +
    optimizer state * 2 (Adam momentum/variance)

Bandwidth (GB/s) = total traffic / step time
```

This shows when you're **bandwidth limited** vs **capacity limited**.

## What I Found

### 1. Batch Size Hits Bandwidth Wall First

Memory keeps growing linearly but throughput plateaus around batch_size 32-64. The GPU can compute faster than memory can feed it data.

**Bandwidth grows with batch size** but eventually you're just waiting on memory.

### 2. Sequence Length Hits Capacity Limit

Attention is O(n²) so memory explodes. Hits OOM way before bandwidth saturates because activation memory grows quadratically.

### 3. FP16 Helps Both

Mixed precision (FP16) cuts memory ~40% AND reduces bandwidth demand because you're moving fewer bytes per step. Real production optimization.

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

Need a CUDA GPU.

### Quick Start
```bash
python train.py  # sanity check
```

### Run Experiments
```bash
python experiments/batch_scaling.py      # bandwidth bottleneck
python experiments/sequence_scaling.py   # capacity bottleneck
python experiments/precision_test.py     # FP16 optimization
```

### Generate Plots
```bash
python utils/plotting.py
```

Check `results/plots/*.png`

## Results

Data saved to `results/logs/*.csv`

Metrics tracked:
- Peak GPU memory (MB)
- Training throughput (samples/sec, tokens/sec)
- **🔥 Estimated memory bandwidth (GB/s)**
- **🔥 Bandwidth utilization (% of theoretical peak)**
- Memory breakdown (params, activations, optimizer)
- Traffic breakdown (where bandwidth goes)

## Key Insights for Memory Architecture

### 1. Bandwidth Often Bottlenecks Before Capacity
- Throughput stops scaling but memory usage keeps growing
- Efficiency (throughput/memory) drops hard
- You have VRAM left but can't use it effectively

### 2. Bandwidth Demand Grows with Workload
- Bigger batches → more activation traffic
- Longer sequences → quadratic activation growth
- This is why HBM bandwidth matters for AI

### 3. Different Workloads Stress Different Things
- Batch scaling: bandwidth limited
- Sequence scaling: capacity limited
- FP16 helps both

### 4. Why This Matters 
- AI needs both capacity AND bandwidth
- Can't just add more VRAM if bandwidth is stuck
- Memory architecture is becoming the critical path
- This is why everyone obsesses over HBM specs

## Limitations

- Only single GPU (no multi-GPU communication bandwidth)
- Bandwidth is estimated, not directly measured (would need nvprof)
- Didn't test gradient accumulation or checkpointing
- Model parallelism would add more complexity

Basically this shows the bottlenecks exist and estimates bandwidth, but doesn't fully characterize the memory system. That needs lower-level profiling tools.

## Files

```
project1_memory_bottleneck/
├── train.py                       # basic training
├── experiments/
│   ├── batch_scaling.py          # bandwidth bottleneck
│   ├── sequence_scaling.py       # capacity bottleneck
│   └── precision_test.py         # FP16 optimization
├── utils/
│   ├── model.py                  # GPT model
│   ├── profiler.py               # 🔥 WITH BANDWIDTH ESTIMATION
│   └── plotting.py               # bandwidth plots included
├── results/
│   ├── logs/                     # CSV data
│   └── plots/                    # graphs with bandwidth
└── requirements.txt
```

## What Makes This Better

### Standard approach:
- Measure memory usage
- See when it crashes

### This approach:
- Measure memory usage
- **Estimate bandwidth demand**
- **Calculate bandwidth utilization**
- See when throughput stops scaling (bandwidth limit)
- See when it crashes (capacity limit)
- Understand WHERE bandwidth goes (params vs activations)

The bandwidth estimation is the key addition that makes this architecture-focused instead of just ML-focused.

## Conclusion

Memory bottlenecks AI training in two ways:
1. **Capacity**: Run out of VRAM
2. **Bandwidth**: Can't move data fast enough

You often hit bandwidth limits before capacity limits. Bigger AI models need both more memory AND faster memory.

This is why:
- HBM bandwidth specs matter as much as capacity
- Memory architecture innovation is critical
- Can't just solve with more compute

Future AI systems need memory technology improvements as much as compute improvements.

---

If you want to run then Just `pip install -r requirements.txt` and `python experiments/batch_scaling.py`
