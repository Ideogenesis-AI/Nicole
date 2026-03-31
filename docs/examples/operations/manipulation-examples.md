# Manipulation Examples

Tensor manipulation operations rearrange indices, change directions, or modify data while preserving the underlying symmetry structure. These operations are essential for preparing tensors before contractions or implementing specific tensor network algorithms.

**Key operations:**

- **Conjugation** (`T.conj()`): Complex conjugate data and flip all index directions
- **Permutation** (`T.permute()`): Reorder indices to a specific layout
- **Transpose** (`T.transpose()`): Reorder indices, defaulting to full reversal

All operations are available as **tensor methods** and support **chaining**. Each method can be called in three ways:

- **`T.conj()`** (default, `in_place=False`): returns a new tensor that **shares** the underlying data storage with `T`. Memory-efficient and supports chaining.
- **`T.conj(in_place=True)`**: modifies `T` in place and returns `T` itself. Useful when you do not need the original.
- **Standalone functions** `conj(T)`, `permute(T, ...)`, `transpose(T, ...)`: return a fully independent tensor by **cloning all data blocks**, guaranteeing isolation from the original at the cost of extra memory. Prefer the method form unless a deep copy is explicitly needed.

All manipulation operations maintain the block structure and charge labels — only the ordering or direction of indices changes.

```python exec="1" session="manipulation" result=""
from nicole import Tensor, Index, Sector, Direction, U1Group
import torch
```

## Conjugation

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
group = U1Group()
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))

# Complex tensor
T = Tensor.random([idx, idx.flip()], itags=["i", "j"], dtype=torch.complex128, seed=42)

# .conj() flips all index directions and conjugates data
T_conj = T.conj()

print(f"Original directions:   {[i.direction for i in T.indices]}")
print(f"Conjugated directions: {[i.direction for i in T_conj.indices]}")
```

## Permutation

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
# 3-index tensor
T3 = Tensor.random([idx, idx.flip(), idx], itags=["i", "j", "k"], seed=7)
print(f"Original: {T3.itags}")

# .permute() returns a new tensor — original is unchanged
T_perm = T3.permute([2, 0, 1])
print(f"Permuted (k, i, j): {T_perm.itags}")
print(f"Original unchanged:  {T3.itags}")
```

## Transpose

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
# Default: reverse all indices
T_trans = T3.transpose()
print(f"Reversed: {T_trans.itags}")

# Custom order
T_trans2 = T3.transpose(1, 0, 2)
print(f"Custom (j, i, k): {T_trans2.itags}")
```

## Chaining

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
A = Tensor.random([idx, idx.flip()], itags=["i", "j"], dtype=torch.complex128, seed=11)

# Hermitian conjugate: conjugate then transpose — written as a chain
A_dag = A.conj().transpose()

print(f"Original: {A.itags}  dirs={[d.direction for d in A.indices]}")
print(f"A†:       {A_dag.itags}  dirs={[d.direction for d in A_dag.indices]}")
```

## Cyclic Permutation

```python exec="1" session="manipulation" result="console" idprefix="" source="material-block"
# Move first index to last via chained permute
n = len(T3.indices)
T_cycle = T3.permute(list(range(1, n)) + [0])
print(f"Original: {T3.itags}")
print(f"Cycled:   {T_cycle.itags}")

# Further chain: cycle twice
T_cycle2 = T3.permute(list(range(1, n)) + [0]).permute(list(range(1, n)) + [0])
print(f"Cycled twice: {T_cycle2.itags}")
```

## See Also

- API Reference: [conj](../../api/manipulation/conjugate.md)
- API Reference: [permute](../../api/manipulation/permute.md)
- API Reference: [transpose](../../api/manipulation/transpose.md)
