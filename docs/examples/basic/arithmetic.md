# Arithmetic Operations

Symmetry-aware tensors support standard arithmetic operations that automatically respect the block structure and charge conservation.

**Supported operations:**

- **Addition/Subtraction**: Combine tensors with matching index structures
- **Scalar Multiplication**: Scale all tensor blocks uniformly
- **Norms**: Compute Frobenius norm across all blocks

All operations are **block-wise**: they operate independently on each charge sector, preserving the symmetry structure. This means you can only add or subtract tensors that have compatible indices and identical block structures.

```python exec="1" session="arithmetic" result=""
from nicole import Tensor, Index, Sector, Direction, U1Group
import torch
```

## Addition and Subtraction

```python exec="1" session="arithmetic" result="console" idprefix="" source="material-block"
group = U1Group()
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))

# Create two tensors with same structure
A = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=1)
B = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=2)

# Addition (block-wise)
C = A + B
print(f"Norm of A: {A.norm():.4f}")
print(f"Norm of B: {B.norm():.4f}\n")
print(f"Norm of A+B: {C.norm():.4f}")

# Subtraction
D = A - B
print(f"Norm of A-B: {D.norm():.4f}\n")

# Verify: A - A = 0
zero = A - A
print(f"A - A norm: {zero.norm():.10f}")  # Should be ~0
```

## Scalar Operations

```python exec="1" session="arithmetic" result="console" idprefix="" source="material-block"
# Scalar multiplication
E = 2.5 * A
print(f"Norm scales: {E.norm():.4f} ≈ {2.5 * A.norm():.4f}")

# Right multiplication also works
F = A * 2.5
print(f"Right multiplication: {F.norm():.4f}")
```

## Requirements for Operations

Tensors must have:

- Same number of indices
- Same index structure (charges and dimensions)
- Same index tags

```python exec="1" session="arithmetic" result="console" idprefix="" source="material-block"
# This works: same structure
A_test = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=1)
B_test = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=2)
C_test = A_test + B_test  # OK
print(f"Compatible tensors added successfully, norm: {C_test.norm():.4f}")

# This fails: different structure
idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))  # Different dim
D_test = Tensor.random([idx2, idx2.flip()], itags=["i", "j"], seed=3)
try:
    E_test = A_test + D_test  # Error: incompatible structure
except ValueError as e:
    print(f"Expected error: {e}")
```

## Norm Computation

```python exec="1" session="arithmetic" result="console" idprefix="" source="material-block"
# Frobenius norm (sqrt of sum of squared elements)
A_norm = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=42)

norm = A_norm.norm()
print(f"Tensor norm: {norm:.4f}")

# Verify norm computation manually
manual_norm = torch.sqrt(sum(
    torch.sum(block ** 2) for block in A_norm.data.values()
))
print(f"Manual norm: {manual_norm:.4f}")
print(f"Match: {abs(norm - manual_norm.item()) < 1e-10}")
```

## Combining Operations

```python exec="1" session="arithmetic" result="console" idprefix="" source="material-block"
# Linear combinations
alpha, beta = 2.0, 1.5
result = alpha * A + beta * B

# Verify: compute manually
manual_result = (alpha * A) + (beta * B)

print(f"Combined result norm: {result.norm():.4f}")
print(f"Manual result norm: {manual_result.norm():.4f}")
print(f"Results match: {abs(result.norm() - manual_result.norm()) < 1e-10}")
```

## In-Place Operations

```python exec="1" session="arithmetic" result="console" idprefix="" source="material-block"
# Copy for in-place modification
X = A.copy()
original_norm = X.norm()

# In-place addition
for key in X.data:
    X.data[key] += B.data[key]

# Equivalent to X = X + B
expected = A + B
print(f"Original X norm: {original_norm:.4f}")
print(f"After in-place addition: {X.norm():.4f}")
print(f"Expected (A + B): {expected.norm():.4f}")
print(f"Results match: {abs(X.norm() - expected.norm()) < 1e-10}")
```

## See Also

- API Reference: [Arithmetic Operations](../../api/arithmetic/addition.md)
- API Reference: [Tensor](../../api/core/tensor.md)
- Next: [Direct Sum (oplus)](../../api/arithmetic/oplus.md)
- Previous: [First Tensor](first-tensor.md)
