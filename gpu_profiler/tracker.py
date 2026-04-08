import torch
import time
import gc
from collections import defaultdict

class MemoryTracker:
    """
    Tracks CUDA memory usage in PyTorch.
    """
    def __init__(self, device="cuda:0"):
        self.device = device
        self.history = defaultdict(list)
        self.start_time = None
        
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available. Tracker will not function.")

    def start(self):
        """Resets stats and starts tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        self.start_time = time.time()
        self.record("start")
        print(f"Started memory tracking on {self.device}")

    def record(self, tag: str):
        """Records current memory stats with a given tag."""
        if not torch.cuda.is_available(): return
        
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2) # MB
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2) # MB
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) # MB
        
        timestamp = time.time() - self.start_time if self.start_time else 0
        
        self.history['tag'].append(tag)
        self.history['timestamp'].append(timestamp)
        self.history['allocated_mb'].append(allocated)
        self.history['reserved_mb'].append(reserved)
        self.history['peak_mb'].append(peak)

    def summary(self):
        """Prints a summary of the recorded memory stats."""
        if not self.history['tag']:
            print("No memory stats recorded.")
            return
            
        print("\n--- GPU Memory Summary ---")
        print(f"{'Tag':<20} | {'Allocated (MB)':<15} | {'Reserved (MB)':<15} | {'Peak (MB)':<15}")
        print("-" * 70)
        
        for i in range(len(self.history['tag'])):
            tag = self.history['tag'][i]
            alloc = self.history['allocated_mb'][i]
            res = self.history['reserved_mb'][i]
            peak = self.history['peak_mb'][i]
            print(f"{tag:<20} | {alloc:<15.2f} | {res:<15.2f} | {peak:<15.2f}")
            
        max_peak = max(self.history['peak_mb'])
        print("-" * 70)
        print(f"Absolute Peak Memory: {max_peak:.2f} MB")

    def clear_cache(self):
        """Forces garbage collection and empties CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.record("after_clear_cache")

if __name__ == "__main__":
    # Mock test
    tracker = MemoryTracker(device="cpu") # Use CPU for mock test to avoid CUDA errors in sandbox
    tracker.start()
    tracker.record("init")
    
    # Simulate some allocation
    dummy_tensor = torch.randn(1000, 1000)
    tracker.record("after_alloc")
    
    del dummy_tensor
    tracker.clear_cache()
    
    tracker.summary()
