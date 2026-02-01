# Performance Tips

Nicole's symmetry-aware architecture provides automatic efficiency gains, but understanding a few optimization strategies can dramatically improve performance for large-scale tensor network calculations.

**Key strategies:**

- **Memory efficiency**: Use appropriate symmetries and aggressive truncation
- **Computational speed**: Leverage block sparsity and optimize contraction order
- **Profiling**: Identify bottlenecks in your specific application

While symmetry already eliminates most unnecessary computation, careful attention to tensor structure, contraction patterns, and truncation thresholds can achieve orders of magnitude speedup for demanding algorithms like DMRG, XTRG, or PEPS calculations.

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
import numpy as np

# Use float32 if precision allows
T_32 = Tensor.random([idx, idx.flip()], dtype=np.float32)  # 4 bytes per element

# Use float64 when needed
T_64 = Tensor.random([idx, idx.flip()], dtype=np.float64)  # 8 bytes per element

# Memory difference can be significant for large tensors
```

## Profiling

### Check Memory Usage

```python
# Get memory per block
for key, block in tensor.data.items():
    mem_mb = block.nbytes / (1024 ** 2)
    print(f"Block {key}: {mem_mb:.2f} MB")

# Total memory
total_mb = sum(b.nbytes for b in tensor.data.values()) / (1024 ** 2)
print(f"Total: {total_mb:.2f} MB")
```

### Time Operations

```python
import time

start = time.time()
result = contract(A, B)
elapsed = time.time() - start
print(f"Contraction took {elapsed:.3f} seconds")
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
# BAD: Manual iteration
result = 0
for block in tensor.data.values():
    result += np.sum(block ** 2)

# GOOD: Use built-in methods
norm_squared = tensor.norm() ** 2
```

## See Also

- API Reference: [decomp](../../api/decomposition/decomp.md)
- API Reference: [contract](../../api/contraction/contract.md)
- Previous: [Load Space](load-space-examples.md)
