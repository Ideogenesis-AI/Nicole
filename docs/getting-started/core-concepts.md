# Core Concepts

Before diving into code, let's understand the key concepts in Nicole and the fundamental idea behind symmetry-aware tensors.

## What are Symmetry-Aware Tensors?

In quantum many-body physics, many systems exhibit symmetries—conserved quantum numbers like particle number, spin, or parity. Traditional dense tensors store all possible matrix elements, including many zeros mandated by symmetry. **Symmetry-aware tensors** exploit these conservation laws to:

- **Save memory**: Only store non-zero blocks that respect symmetry
- **Accelerate computations**: Skip operations on zeros
- **Enforce physics**: Automatically respect selection rules

Nicole implements this through a **block-sparse** representation where each block corresponds to a specific combination of conserved quantum numbers (charges).

## Symmetry Groups

Nicole supports both Abelian (U(1) or Z(2)) and non-Abelian (SU(2)) symmetry groups:

- **U(1)**: Continuous symmetry with integer charges (e.g., particle number, magnetization)
- **Z(2)**: Binary symmetry with charges 0 or 1 (e.g., parity, Z₂ topological order)
- **SU(2)**: Non-Abelian rotational symmetry with spin-based charges (e.g., total angular momentum)
- **ProductGroup**: Combines multiple symmetries (e.g., U(1) × SU(2) for particle number and spin)

### Group Operations

Each symmetry group defines:
- **Fusion**: How charges combine (addition for U(1), XOR for Z(2), triangular constraint for SU(2))
- **Dual**: The conjugate representation (negation for U(1), self for Z(2) and SU(2))
- **Neutral element**: The identity charge (0 for all groups)
- **`is_abelian`**: Whether the group is Abelian (`False` for SU(2))
- **`irrep_dim`**: Dimension of the irreducible representation for a charge (always 1 for Abelian groups; `2j+1` for SU(2))

```python
from nicole import U1Group, Z2Group, SU2Group

# U(1) example
u1 = U1Group()
print(u1.fuse_unique(2, 3))       # 5 (addition)
print(u1.dual(5))          # -5 (dual representation)
print(u1.irrep_dim(5))     # 1 (Abelian: always 1)

# Z(2) example
z2 = Z2Group()
print(z2.fuse_unique(1, 1))       # 0 (XOR: 1⊕1=0)
print(z2.dual(1))          # 1 (self-dual)

# SU(2) example
su2 = SU2Group()
print(su2.fuse_channels(1, 1))  # (0, 2) — spin-1/2 ⊗ spin-1/2 → singlet or triplet
print(su2.dual(2))              # 2 (self-dual)
print(su2.irrep_dim(2))         # 3 (spin-1 triplet: 2j+1 = 3)
```

### Non-Abelian Groups and Irrep Dimension

For Abelian groups, each charge labels a one-dimensional sector. For non-Abelian groups like SU(2), each charge `2j` labels an entire **multiplet** of `2j+1` magnetic substates (irreducible representation). This is captured by `irrep_dim`:

| Group  | `irrep_dim(q)` | Example             |
|--------|----------------|---------------------|
| U(1)   | 1              | Any charge          |
| Z(2)   | 1              | 0 or 1              |
| SU(2)  | 2j + 1         | spin-1: `irrep_dim(2)` = 3 |

Nicole works with **reduced tensor elements** for SU(2): each data block stores reduced elements per multiplet combination (not `2j+1` entries per index), and full elements are reconstructed via Clebsch–Gordan coefficients stored in the tensor's intertwiners. This is the key source of exponential compression in SU(2) tensor networks.

SU(2) also differs in fusion: two charges can fuse into **multiple** channels (all spins from `|j1−j2|` to `j1+j2`), whereas Abelian fusion always gives a unique result.

For the full mathematical structure of how Nicole stores SU(2) tensors internally — including the Wigner–Eckart decomposition and the R-W-C representation — see the [Yuzuha Protocol](yuzuha-protocol.md) page.

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
# For U(1): 3 orthogonal states all having charge +1
sector = Sector(charge=1, dim=3)
```

Multiple sectors with different charges combine to form the full structure of an index.

For **SU(2)**, `dim` counts the number of independent **multiplets** (not individual states). A sector `Sector(charge=2j, dim=n)` represents `n` independent spin-j multiplets, each containing `2j+1` physical states. The total number of physical states in that sector is `n × (2j+1)`.

## Indices

An **Index** defines a tensor index with:

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

For SU(2) indices, `index.dim` counts the total number of multiplets across all sectors, while `index.num_states` counts the total physical states (summing `irrep_dim × dim` per sector). For Abelian groups, `dim` and `num_states` are always equal since `irrep_dim = 1`.

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
