
import torch
import pynvml
import time
import os
import threading
from collections import defaultdict

class GPUMemoryProfiler:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.handle = None
        self.monitoring = False
        self.memory_snapshots = defaultdict(list)
        self.lock = threading.Lock()

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            print(f"Initialized NVML for GPU {self.device_id}: {pynvml.nvmlDeviceGetName(self.handle).decode()}")
        except pynvml.NVMLError as error:
            print(f"Failed to initialize NVML: {error}")
            self.handle = None

    def _get_gpu_memory_usage(self):
        if not self.handle:
            return None, None
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            total_memory_mb = info.total / (1024**2)
            used_memory_mb = info.used / (1024**2)
            return used_memory_mb, total_memory_mb
        except pynvml.NVMLError as error:
            print(f"Failed to get GPU memory info: {error}")
            return None, None

    def _monitor_thread(self, interval_sec, tag):
        while self.monitoring:
            used_mb, total_mb = self._get_gpu_memory_usage()
            if used_mb is not None:
                with self.lock:
                    self.memory_snapshots[tag].append((time.time(), used_mb))
            time.sleep(interval_sec)

    def start_monitoring(self, interval_sec=0.1, tag="default"):
        if self.handle is None:
            print("GPU monitoring not available. NVML not initialized.")
            return
        if self.monitoring:
            print("Monitoring already in progress.")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_thread, args=(interval_sec, tag))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Started GPU memory monitoring for tag: {tag}")

    def stop_monitoring(self):
        if not self.monitoring:
            print("Monitoring not active.")
            return
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()
        print("Stopped GPU memory monitoring.")

    def get_memory_usage_history(self, tag="default"):
        with self.lock:
            return self.memory_snapshots.get(tag, [])

    def get_peak_memory_usage(self, tag="default"):
        history = self.get_memory_usage_history(tag)
        if not history:
            return 0.0
        return max(snapshot[1] for snapshot in history)

    def clear_snapshots(self, tag=None):
        with self.lock:
            if tag:
                if tag in self.memory_snapshots:
                    del self.memory_snapshots[tag]
                    print(f"Cleared memory snapshots for tag: {tag}")
            else:
                self.memory_snapshots.clear()
                print("Cleared all memory snapshots.")

    def __del__(self):
        if self.monitoring:
            self.stop_monitoring()
        try:
            pynvml.nvmlShutdown()
            print("NVML shutdown.")
        except pynvml.NVMLError as error:
            print(f"Failed to shutdown NVML: {error}")

if __name__ == '__main__':
    profiler = GPUMemoryProfiler()

    if profiler.handle:
        profiler.start_monitoring(tag="model_training")

        # Simulate some GPU workload
        print("\nSimulating GPU workload...")
        try:
            # Allocate some tensors on GPU
            tensor1 = torch.randn(1000, 1000, 1000).cuda()
            print(f"Allocated tensor1: {tensor1.element_size() * tensor1.numel() / (1024**2):.2f} MB")
            time.sleep(1)

            tensor2 = torch.randn(2000, 2000).cuda()
            print(f"Allocated tensor2: {tensor2.element_size() * tensor2.numel() / (1024**2):.2f} MB")
            time.sleep(2)

            del tensor1
            torch.cuda.empty_cache()
            print("Deallocated tensor1.")
            time.sleep(1)

        except RuntimeError as e:
            print(f"Could not run CUDA operations. Is a GPU available? Error: {e}")
        except Exception as e:
            print(f"An error occurred during workload simulation: {e}")

        profiler.stop_monitoring()

        peak_memory = profiler.get_peak_memory_usage(tag="model_training")
        print(f"\nPeak GPU memory usage during model_training: {peak_memory:.2f} MB")

        history = profiler.get_memory_usage_history(tag="model_training")
        if history:
            print("First 5 memory snapshots (timestamp, MB_used):")
            for i, snapshot in enumerate(history[:5]):
                print(f"  {i+1}. {snapshot[0]:.2f}, {snapshot[1]:.2f}")

        profiler.clear_snapshots()
    else:
        print("Skipping GPU workload simulation as NVML is not initialized.")


# Commit timestamp: 2025-01-14 00:00:00 - 813
