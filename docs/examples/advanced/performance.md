# Performance Tips

Nicole's symmetry-aware architecture provides automatic efficiency gains, but understanding a few optimization strategies can dramatically improve performance for large-scale tensor network calculations.

**Key strategies:**

- **Memory efficiency**: Use appropriate symmetries and aggressive truncation
- **Computational speed**: Leverage block sparsity and optimize contraction order
- **GPU acceleration**: Use CUDA or MPS for large tensors
- **Device management**: Minimize CPU↔GPU transfers
- **Profiling**: Identify bottlenecks in your specific application

While symmetry already eliminates most unnecessary computation, careful attention to tensor structure, contraction patterns, device placement, and truncation thresholds can achieve orders of magnitude speedup for demanding algorithms like DMRG, XTRG, or PEPS calculations.

## Memory Efficiency

### Use Appropriate Symmetries

```python
# More symmetries = fewer blocks = less memory
from nicole import ProductGroup, U1Group, Z2Group

# Without symmetry: full dense tensor
# With U(1): ~N blocks
# With U(1) × Z(2): ~N/2 blocks

# Choose symmetries that match your problem
```

### Truncate Aggressively

```python
from nicole import decomp

# Truncate small singular values
U, S, Vh = decomp(T, axes=0, mode="SVD", trunc=("thresh", 1e-12))

# Or keep fixed number
U, S, Vh = decomp(T, axes=0, mode="SVD", trunc=("nkeep", 50))
```

## Computation Efficiency

### Contraction Order Matters

```python
from nicole import contract

# BAD: Contract large tensors first
# A: (100×10), B: (10×100), C: (100×5)
# result = contract(contract(A, B), C)  # Creates 100×100 intermediate

# GOOD: Contract to reduce size early
# result = contract(A, contract(B, C))  # Creates 10×5 intermediate
```

### Reuse Index Objects

```python
# GOOD: Create once, reuse
idx = Index(Direction.OUT, group, sectors=sectors)
tensors = [Tensor.random([idx, idx.flip()], itags=[f"a{i}", f"b{i}"]) 
           for i in range(10)]

# BAD: Create new index each time
tensors = [Tensor.random([Index(...), Index(...)], ...) 
           for i in range(10)]  # Wastes time and memory
```

### Use Appropriate Data Types

```python
import torch

# Use float32 if precision allows (required for MPS on Apple Silicon)
T_32 = Tensor.random([idx, idx.flip()], dtype=torch.float32)  # 4 bytes per element

# Use float64 when needed (default for CPU/CUDA)
T_64 = Tensor.random([idx, idx.flip()], dtype=torch.float64)  # 8 bytes per element

# Complex types available
T_complex = Tensor.random([idx, idx.flip()], dtype=torch.complex128)

# Memory difference can be significant for large tensors
```

## Device Management

### GPU Acceleration

Nicole supports GPU acceleration through PyTorch:

```python
import torch
from nicole import Tensor

# Check device availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Create tensor on GPU (if available)
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Create tensor directly on device
T_gpu = Tensor.random([idx, idx.flip()], itags=["i", "j"], device=device)

# Or transfer existing tensor
T_cpu = Tensor.random([idx, idx.flip()], itags=["i", "j"])
T_gpu = T_cpu.to(device)
```

### Minimize Device Transfers

```python
# BAD: Frequent transfers
for i in range(100):
    T = Tensor.random([idx, idx.flip()], device='cpu')
    T_gpu = T.cuda()  # Transfer each iteration
    result = contract(T_gpu, B_gpu)

# GOOD: Transfer once, work on GPU
tensors_gpu = [T.cuda() for T in tensors_cpu]  # Transfer once
for T_gpu in tensors_gpu:
    result = contract(T_gpu, B_gpu)  # All work on GPU
```

## Profiling

### Check Memory Usage

```python
import torch

# Get memory per block
for key, block in tensor.data.items():
    mem_mb = block.element_size() * block.numel() / (1024 ** 2)
    print(f"Block {key}: {mem_mb:.2f} MB on {block.device}")

# Total memory
total_mb = sum(b.element_size() * b.numel() for b in tensor.data.values()) / (1024 ** 2)
print(f"Total: {total_mb:.2f} MB")

# GPU memory usage (if using CUDA)
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
```

### Time Operations

```python
import time
import torch

# For CPU timing
start = time.time()
result = contract(A, B)
elapsed = time.time() - start
print(f"Contraction took {elapsed:.3f} seconds")

# For GPU timing (need synchronization)
if torch.cuda.is_available():
    torch.cuda.synchronize()  # Ensure all GPU ops are done
    start = time.time()
    result = contract(A_gpu, B_gpu)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"GPU contraction took {elapsed:.3f} seconds")
```

## Common Pitfalls

### Don't Create Unnecessary Copies

```python
# BAD: Creates copies
for i in range(100):
    T_copy = tensor.copy()  # Expensive!
    # ... use T_copy

# GOOD: Use original or copy once
T_working = tensor.copy()
for i in range(100):
    # ... modify T_working in place
```

### Avoid Block Iteration When Possible

```python
import torch

# BAD: Manual iteration
result = 0
for block in tensor.data.values():
    result += torch.sum(block ** 2)

# GOOD: Use built-in methods
norm_squared = tensor.norm() ** 2
```

## Performance Recommendations

### General Recommendation: Default to CPU

**In tensor network algorithms, tensors often grow very large and can exceed GPU memory.**

- **CPU is the default and most reliable choice** for tensor network computations
- Use `torch.float64` (default) for numerical precision in iterative algorithms
- Focus on contraction order optimization and aggressive truncation
- CPU memory is typically much larger (32-1024 GB) than GPU memory (8-24 GB)

### GPU Acceleration: Use Selectively When Memory Permits

- **Only delegate to GPU when tensor sizes fit comfortably in GPU memory**
- Monitor GPU memory usage carefully (use `torch.cuda.memory_allocated()`)
- Consider `torch.float32` to reduce memory footprint and improve GPU performance
- Best for:
    - Small to medium tensors (< 4-8 GB depending on GPU)
    - Specific bottleneck operations (large contractions, SVD)
    - Batched operations where you can control memory usage
- Transfer tensors back to CPU after GPU operations to free GPU memory

### For Apple Silicon (MPS)
- Must use `torch.float32` (float64 not fully supported)
- Good for moderate-sized problems
- Automatic conversion happens via `.to('mps')`

### For NVIDIA GPUs (CUDA)
- Best performance for large-scale calculations
- Supports both float32 and float64
- Use mixed precision when appropriate

## See Also

- [GPU Acceleration Guide](gpu-acceleration.md): Detailed GPU usage
- API Reference: [decomp](../../api/decomposition/decomp.md)
- API Reference: [contract](../../api/contraction/contract.md)
- Previous: [Load Space](load-space-examples.md)
