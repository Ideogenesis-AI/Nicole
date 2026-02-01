# Manipulation Examples

Tensor manipulation operations rearrange indices, change directions, or modify data while preserving the underlying symmetry structure. These operations are essential for preparing tensors before contractions or implementing specific tensor network algorithms.

**Key operations:**

- **Conjugation** (`conj`): Complex conjugate data and flip all index directions
- **Permutation** (`permute`): Reorder indices to a specific layout
- **Transpose** (`transpose`): Swap two indices
- **In-place vs functional**: Many operations support both styles

All manipulation operations maintain the block structure and charge labels — only the organization or representation changes. This ensures that symmetry properties remain intact throughout your computation.

```python exec="1" session="manipulation" result=""
from nicole import conj, permute, transpose, Tensor, Index, Sector, Direction, U1Group
import numpy as np
```

## Conjugation

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
group = U1Group()
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))

# Complex tensor
T = Tensor.random([idx, idx.flip()], itags=["i", "j"], dtype=np.complex128, seed=42)

# Conjugate (flips directions + conjugates data)
T_conj = conj(T)

print(f"Original directions: {[i.direction for i in T.indices]}")
print(f"Conjugated directions: {[i.direction for i in T_conj.indices]}")
```

## Permutation

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
# 3-index tensor
T3 = Tensor.random([idx, idx.flip(), idx], itags=["i", "j", "k"], seed=7)
print(f"Original tags: {T3.itags}")

# Permute to (k, i, j)
T_perm = permute(T3, [2, 0, 1])
print(f"Permuted tags: {T_perm.itags}")

# Original unchanged (functional operation)
print(f"Original still: {T3.itags}")
```

## Transpose

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
# Default: reverse all axes
T_trans = transpose(T3)
print(f"Transposed tags: {T_trans.itags}")

# Custom order
T_trans2 = transpose(T3, 1, 0, 2)
print(f"Custom transpose: {T_trans2.itags}")
```

## In-Place vs Functional

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
# In-place modification
T_inplace = T3.copy()
T_inplace.permute([2, 0, 1])
print(f"After in-place permute: {T_inplace.itags}\n")

# Functional (returns new tensor)
T_functional = permute(T3, [2, 0, 1])
print(f"Original unchanged: {T3.itags}")
print(f"New tensor: {T_functional.itags}")
```

## Hermitian Conjugate

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
# For matrices: A† = (A*)ᵀ
A = Tensor.random([idx, idx.flip()], itags=["i", "j"], dtype=np.complex128, seed=11)

# Method 1: conj then transpose
A_dag1 = transpose(conj(A))

# Method 2: transpose then conj
A_dag2 = conj(transpose(A))

# Both give same result
error = (A_dag1 - A_dag2).norm()
print(f"Methods match: {error < 1e-10}")
```

## Cyclic Permutation

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
# Move first axis to last
n = len(T3.indices)
T_cycle = permute(T3, list(range(1, n)) + [0])
print(f"Original: {T3.itags}")
print(f"Cycled: {T_cycle.itags}")
```

## See Also

- API Reference: [conj](../../api/manipulation/conjugate.md)
- API Reference: [permute](../../api/manipulation/permute.md)
- API Reference: [transpose](../../api/manipulation/transpose.md)
