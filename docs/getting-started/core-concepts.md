# Core Concepts

Before diving into code, let's understand the key concepts in Nicole and the fundamental idea behind symmetry-aware tensors.

## What are Symmetry-Aware Tensors?

In quantum many-body physics, many systems exhibit symmetries—conserved quantum numbers like particle number, spin, or parity. Traditional dense tensors store all possible matrix elements, including many zeros mandated by symmetry. **Symmetry-aware tensors** exploit these conservation laws to:

- **Save memory**: Only store non-zero blocks that respect symmetry
- **Accelerate computations**: Skip operations on zeros
- **Enforce physics**: Automatically respect selection rules

Nicole implements this through a **block-sparse** representation where each block corresponds to a specific combination of conserved quantum numbers (charges).

## Symmetry Groups

Nicole supports Abelian symmetry groups that represent conserved quantum numbers:

- **U(1)**: Continuous symmetry with integer charges (e.g., particle number, magnetization)
- **Z(2)**: Binary symmetry with charges 0 or 1 (e.g., parity, Z₂ topological order)
- **ProductGroup**: Combines multiple symmetries (e.g., U(1) × Z(2) for particle number and parity)

### Group Operations

Each symmetry group defines:
- **Fusion**: How charges combine (addition for U(1), XOR for Z(2))
- **Inverse**: The opposite charge (negation for U(1), identity for Z(2))
- **Neutral element**: The identity charge (0 for both)

```python
from nicole import U1Group, Z2Group

# U(1) example
u1 = U1Group()
print(u1.fuse(2, 3))      # 5 (addition)
print(u1.dual(5))         # -5 (dual representation)

# Z(2) example
z2 = Z2Group()
print(z2.fuse(1, 1))      # 0 (XOR: 1⊕1=0)
print(z2.dual(1))         # 1 (self-dual)
```

## Charges

**Charges** are quantum numbers that label sectors of a tensor:

- For **U(1)**: integers like -2, -1, 0, 1, 2
- For **Z(2)**: 0 (even) or 1 (odd)
- For **ProductGroup**: tuples like (1, 0) or (2, 1)

Each charge represents a quantum number value. For example, in a system conserving particle number (U(1)), charge 2 means "2 particles".

## Sectors

A **Sector** pairs a charge with a dimension, representing a subspace of the tensor:

```python
from nicole import Sector

# A sector with charge 1 and dimension 3
# Means: 3 orthogonal states all having charge +1
sector = Sector(charge=1, dim=3)
```

Multiple sectors with different charges combine to form the full structure of an index.

## Indices

An **Index** defines a tensor leg (axis) with:

- **Direction**: `Direction.OUT` or `Direction.IN`
- **Symmetry Group**: The group governing the charge structure
- **Sectors**: Available charge sectors and their dimensions

```python
from nicole import Index, Sector, Direction, U1Group

group = U1Group()
index = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=2),   # 2 states with charge 0
        Sector(charge=1, dim=1),   # 1 state with charge +1
        Sector(charge=-1, dim=1),  # 1 state with charge -1
    )
)
```

### Index Direction

The direction (OUT/IN) is crucial for charge conservation:

- **OUT**: Contributes positive charge
- **IN**: Contributes negative charge (like a charge conjugate)

Use `index.flip()` to reverse the direction.

## Charge Conservation

Nicole automatically enforces charge conservation for all tensor blocks:

```
∑ (OUT charges) - ∑ (IN charges) = neutral element
```

For U(1), this means: `∑ OUT_charges - ∑ IN_charges = 0`

### Example

Consider a 2-index tensor with indices (i, j):

```python
# i: OUT with charges [0, 1, -1]
# j: IN  with charges [0, 1, -1]

# Valid blocks (charge conserved):
# - (0, 0): 0 - 0 = 0 ✓
# - (1, 1): 1 - 1 = 0 ✓
# - (-1, -1): -1 - (-1) = 0 ✓

# Invalid blocks (charge not conserved):
# - (0, 1): 0 - 1 = -1 ✗
# - (1, 0): 1 - 0 = 1 ✗
```

Only valid blocks are created and stored, automatically enforcing physical selection rules.

## Block-Sparse Storage

Nicole represents tensors as dictionaries of dense PyTorch tensors:

```python
tensor.data = {
    (0, 0): torch.tensor([[1.2, 0.3], [0.5, 0.8]]),      # 2×2 block
    (1, 1): torch.tensor([[0.7]]),                       # 1×1 block
    (-1, -1): torch.tensor([[0.4]]),                     # 1×1 block
}
```

Each key is a tuple of charges, one per index. The corresponding value is the dense PyTorch tensor for that block.

## Index Tags

Indices can have **tags** (string labels) to make operations more intuitive:

```python
tensor = Tensor.random([idx1, idx2, idx3], itags=["i", "j", "k"])

# Tags help identify indices
print(tensor.itags)  # ('i', 'j', 'k')

# Useful for automatic contraction
result = contract(A, B)  # Automatically contracts on matching tags
```

## Next Steps

Now that you understand the core concepts, continue to:

- **[Quick Examples](quick-examples.md)**: See these concepts in action
- **[API Reference](../api/index.md)**: Detailed documentation of all classes and functions
- **[Examples](../examples/index.md)**: More complex use cases and patterns
