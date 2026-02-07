# GPU Acceleration

Nicole leverages PyTorch's device management to enable GPU acceleration for large-scale tensor network calculations. This guide covers how to use GPUs effectively with Nicole.

## Overview

Nicole supports three device types:

- **CPU**: Default device, supports all dtypes (float32, float64, complex64, complex128)
- **CUDA** (NVIDIA GPUs): Full support for all dtypes, best performance for deep learning
- **MPS** (Apple Silicon): Supports float32/complex64 natively; float64/complex128 are automatically downgraded

## Checking Device Availability

Before using GPU acceleration, check what devices are available:

```python exec="1" session="gpu" result=""
import torch
from nicole import Tensor, Index, Sector, Direction, U1Group

group = U1Group()
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
```

```python exec="1" session="gpu" result="console" idprefix="" source="material-block"
# Check NVIDIA GPU (CUDA)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Check Apple Silicon GPU (MPS)
print(f"MPS available: {torch.backends.mps.is_available()}")

# Current default device
print(f"Default device: {torch.get_default_device()}")
```

## Creating Tensors on GPU

### Specify Device at Creation

```python exec="1" session="gpu" result="console" idprefix="" source="material-block"
# Create directly on GPU
if torch.cuda.is_available():
    T_gpu = Tensor.random([idx, idx.flip()], itags=["i", "j"], device='cuda', seed=42)
elif torch.backends.mps.is_available():
    T_gpu = Tensor.random([idx, idx.flip()], itags=["i", "j"], device='mps', seed=42)
else:
    T_gpu = Tensor.random([idx, idx.flip()], itags=["i", "j"], device='cpu', seed=42)

print(f"Tensor device: {T_gpu.device}")
```

### Transfer Existing Tensor

```python exec="1" session="gpu" result="console" idprefix="" source="material-block"
# Create on CPU
T_cpu = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=10)

# Transfer to GPU
if torch.cuda.is_available():
    T_gpu_transfer = T_cpu.to('cuda')
elif torch.backends.mps.is_available():
    T_gpu_transfer = T_cpu.to('mps')
else:
    T_gpu_transfer = T_cpu  # Stay on CPU

# Transfer back to CPU
T_back = T_gpu_transfer.cpu()

print(f"Original: {T_cpu.device}")
print(f"On GPU: {T_gpu_transfer.device}")
print(f"Back to CPU: {T_back.device}")
```

### Convenience Methods

```python exec="1" session="gpu" result="console" idprefix="" source="material-block"
T_cpu2 = Tensor.random([idx, idx.flip()], itags=["k", "l"], seed=20)

# Use convenience methods
if torch.cuda.is_available():
    T_gpu_conv = T_cpu2.cuda()  # Equivalent to .to('cuda')
    print(f"After .cuda(): {T_gpu_conv.device}")
elif torch.backends.mps.is_available():
    T_gpu_conv = T_cpu2.to('mps')  # MPS doesn't have .mps() method
    print(f"After .to('mps'): {T_gpu_conv.device}")
else:
    T_gpu_conv = T_cpu2
    print(f"No GPU available, staying on: {T_gpu_conv.device}")

# Transfer back
T_back2 = T_gpu_conv.cpu()  # Equivalent to .to('cpu')
print(f"After .cpu(): {T_back2.device}")
```

## Device Consistency in Operations

All tensors in an operation must be on the same device:

```python
from nicole import contract

# Create tensors on the same device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
A = Tensor.random([idx, idx.flip()], itags=["i", "mid"], device=device)
B = Tensor.random([idx, idx.flip()], itags=["mid", "j"], device=device)

# Contract (both on same device)
C = contract(A, B)
print(f"Result device: {C.device}")

# This would fail:
# A_cpu = Tensor.random([idx, idx.flip()], itags=["i", "mid"], device='cpu')
# B_gpu = Tensor.random([idx, idx.flip()], itags=["mid", "j"], device='cuda')
# C = contract(A_cpu, B_gpu)  # Error: tensors on different devices
```

## Data Types and GPU Compatibility

### CUDA (NVIDIA)

```python
import torch

# CUDA supports all dtypes
T_f32 = Tensor.random([idx, idx.flip()], dtype=torch.float32, device='cuda')
T_f64 = Tensor.random([idx, idx.flip()], dtype=torch.float64, device='cuda')
T_c64 = Tensor.random([idx, idx.flip()], dtype=torch.complex64, device='cuda')
T_c128 = Tensor.random([idx, idx.flip()], dtype=torch.complex128, device='cuda')
```

### MPS (Apple Silicon)

MPS devices don't support `float64` or `complex128` dtypes. Nicole automatically handles this by converting to compatible dtypes:

- `float64` → `float32`
- `complex128` → `complex64`

This conversion happens automatically in two scenarios:

**1. Creating tensors directly on MPS:**

```python exec="1" session="gpu" result="console" idprefix="" source="material-block"
# Request float64 on MPS - automatically uses float32
if torch.backends.mps.is_available():
    T = Tensor.random([idx, idx.flip()], dtype=torch.float64, device='mps')
    print(f"Requested: float64, Actual: {T.dtype}")  # float32
    
    # Same for complex types
    T_complex = Tensor.random([idx, idx.flip()], dtype=torch.complex128, device='mps')
    print(f"Requested: complex128, Actual: {T_complex.dtype}")  # complex64
else:
    print("MPS not available")
```

**2. Moving tensors to MPS:**

```python exec="1" session="gpu" result="console" idprefix="" source="material-block"
# Create on CPU with float64
T_cpu = Tensor.random([idx, idx.flip()], dtype=torch.float64, device='cpu')
print(f"On CPU: {T_cpu.dtype}")

# Transfer to MPS - auto-converts to float32
if torch.backends.mps.is_available():
    T_mps = T_cpu.to('mps')
    print(f"On MPS: {T_mps.dtype}")  # float32
else:
    print("MPS not available")
```

!!! tip "Native float32 works directly"
    If you already use `float32` or `complex64`, no conversion is needed:
    
    ```python
    T = Tensor.random([idx, idx.flip()], dtype=torch.float32, device='mps')
    # Uses float32 directly, no conversion
    ```

## Performance Best Practices

### 1. Minimize Device Transfers

Device transfers are expensive. Do them once at the start:

```python
# BAD: Transfer every iteration
for i in range(100):
    T_cpu = Tensor.random([idx, idx.flip()], device='cpu')
    T_gpu = T_cpu.cuda()
    result = contract(T_gpu, B_gpu)
    result_cpu = result.cpu()  # Unnecessary transfer

# GOOD: Work entirely on GPU
B_gpu = Tensor.random([idx, idx.flip()], device='cuda')
results = []
for i in range(100):
    T_gpu = Tensor.random([idx, idx.flip()], device='cuda')
    result = contract(T_gpu, B_gpu)
    results.append(result)

# Transfer back once at the end if needed
results_cpu = [r.cpu() for r in results]
```

### 2. Batch Transfer Operations

```python
# Transfer multiple tensors together
tensors_cpu = [Tensor.random([idx, idx.flip()], itags=[f"a{i}", f"b{i}"]) 
               for i in range(10)]

# BAD: Transfer one by one
# tensors_gpu = [t.cuda() for t in tensors_cpu]  # Each transfer has overhead

# GOOD: Use list comprehension (still transfers one by one, but more pythonic)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensors_gpu = [t.to(device) for t in tensors_cpu]
```

### 3. Choose Data Type Wisely

```python
# For MPS, use float32
if torch.backends.mps.is_available():
    T = Tensor.random([idx, idx.flip()], dtype=torch.float32, device='mps')

# For CUDA, choose based on precision needs
if torch.cuda.is_available():
    # float32: faster, less memory, good for most applications
    T_fast = Tensor.random([idx, idx.flip()], dtype=torch.float32, device='cuda')
    
    # float64: slower, more memory, higher precision
    T_precise = Tensor.random([idx, idx.flip()], dtype=torch.float64, device='cuda')
```

## Monitoring GPU Memory

```python
if torch.cuda.is_available():
    # Check memory usage
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    # Reset peak stats
    torch.cuda.reset_peak_memory_stats()
    
    # Clear cache if needed
    torch.cuda.empty_cache()
```

## Benchmarking CPU vs GPU

```python
import time

# Create test tensors
idx_large = Index(Direction.OUT, group, 
                  sectors=tuple(Sector(i, 10) for i in range(10)))

# CPU benchmark
T_cpu = Tensor.random([idx_large, idx_large.flip()], itags=["i", "j"], device='cpu')
U_cpu = Tensor.random([idx_large, idx_large.flip()], itags=["j", "k"], device='cpu')

start = time.time()
result_cpu = contract(T_cpu, U_cpu)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f} seconds")

# GPU benchmark
if torch.cuda.is_available():
    T_gpu = T_cpu.cuda()
    U_gpu = U_cpu.cuda()
    
    # Warmup
    _ = contract(T_gpu, U_gpu)
    torch.cuda.synchronize()
    
    # Actual benchmark
    start = time.time()
    result_gpu = contract(T_gpu, U_gpu)
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
```

## Common Pitfalls

### 1. Profiler Synchronization 

```python
# BAD: Timing without synchronization
if torch.cuda.is_available():
    start = time.time()
    result = contract(A_gpu, B_gpu)
    elapsed = time.time() - start  # Wrong! GPU might still be running

# GOOD: Synchronize before timing
if torch.cuda.is_available():
    start = time.time()
    result = contract(A_gpu, B_gpu)
    torch.cuda.synchronize()
    elapsed = time.time() - start  # Correct
```

### 2. Mixed Devices in Operations

```python
# This will error
A_cpu = Tensor.random([idx, idx.flip()], device='cpu')
B_gpu = Tensor.random([idx, idx.flip()], device='cuda')
# result = contract(A_cpu, B_gpu)  # Error!

# Solution: Move to same device first
B_cpu = B_gpu.cpu()
result = contract(A_cpu, B_cpu)  # OK
```

## When to Use GPU

### GPU is Beneficial When:
- Tensor blocks are large (> 1000 elements per block)
- Many contractions with the same tensors
- Decompositions (SVD, QR) on large tensors
- Iterative algorithms (DMRG, TEBD)

### CPU is Better When:
- Tensor blocks are small (< 100 elements per block)
- Few operations (transfer overhead dominates)
- Memory-constrained problems where CPU RAM is abundant
- Prototyping and debugging

## See Also

- [Performance Tips](performance.md): General optimization strategies
- [Autograd Guide](autograd.md): Using gradients with GPU
- [API: Tensor.device](../../api/core/tensor.md): Device property documentation
