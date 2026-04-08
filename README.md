# GPU Memory Profiler

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Supported-red)
![License](https://img.shields.io/badge/license-MIT-green)

A lightweight, plug-and-play utility for tracking, analyzing, and visualizing GPU memory allocation in PyTorch. Identify memory leaks and optimize your batch sizes easily.

## Features
- Real-time memory tracking (allocated vs. reserved)
- Peak memory usage reporting per training step
- Identification of large tensor allocations
- Export profiling data to JSON/CSV

## Installation
```bash
pip install gpu-memory-profiler
```

## Usage
```python
import torch
from gpu_profiler import MemoryTracker

tracker = MemoryTracker(device="cuda:0")

# Start tracking before your training loop
tracker.start()

# ... your training code ...
model(inputs)
loss.backward()

# Print summary
tracker.summary()
```
