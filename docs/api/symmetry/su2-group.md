# SU2Group

SU(2) non-Abelian symmetry group with spin-based quantum numbers.

::: nicole.SU2Group
    options:
      show_source: false
      heading_level: 2
      members:
        - name
        - neutral
        - is_abelian
        - irrep_dim
        - dual
        - fuse_channels
        - equal
        - validate_charge

## Description

Represents the SU(2) group of rotational symmetry. Unlike Abelian groups, SU(2) representations have an internal dimension (irrep dimension) and fuse into multiple channels simultaneously via Clebsch–Gordan decomposition.

### Charge Convention: 2j Integers

Charges are non-negative integers representing **twice the physical spin** (the 2j convention):

| Charge (2j) | Physical spin j | Irrep dim (2j+1) |
|:-----------:|:---------------:|:----------------:|
| 0           | 0               | 1 (singlet)      |
| 1           | 1&#47;2         | 2 (doublet)      |
| 2           | 1               | 3 (triplet)      |
| 3           | 3&#47;2         | 4 (quartet)      |
| 4           | 2               | 5 (quintet)      |

This convention keeps all quantum numbers as plain integers, avoiding fractions for half-integer spins.

### Irrep Dimension

Every SU(2) charge carries an **irrep dimension** equal to `2j + 1`. This is the number of magnetic quantum number states (`m = -j, ..., +j`) within each multiplet. Nicole works in the **reduced tensor** formalism: each block in a tensor stores the **reduced tensor element**, and the full tensor elements are reconstructed by contracting with Clebsch–Gordan coefficients stored in `Bridge` intertwiners.

### Fusion Channels

SU(2) spins fuse via the triangular inequality — two spins `j1` and `j2` can combine into any total spin from `|j1 − j2|` to `j1 + j2` in integer steps. In 2j notation:

```python
from nicole import SU2Group

group = SU2Group()

# Spin-1/2 ⊗ spin-1/2 → spin-0 or spin-1
group.fuse_channels(1, 1)   # (0, 2)

# Spin-1 ⊗ spin-1 → spin-0, spin-1, or spin-2
group.fuse_channels(2, 2)   # (0, 2, 4)

# Three spin-1/2 → spin-1/2 or spin-3/2
group.fuse_channels(1, 1, 1)  # (1, 3)
```

### Self-Duality

All SU(2) representations are self-dual — the contragredient of a spin-j representation is again spin-j:

```python
group.dual(0)  # 0  (singlet is its own dual)
group.dual(1)  # 1  (doublet is its own dual)
group.dual(2)  # 2  (triplet is its own dual)
```

## Physical Applications

- **Spin-S systems**: Arbitrary spin magnitude, e.g. spin-1/2 chains, spin-1 Haldane chains
- **Angular momentum conservation**: Total angular momentum of a multi-particle system
- **Wigner–Eckart theorem**: Operator matrix elements factorize into CG coefficients and a single reduced matrix element

## Using with Product Group

`SU2Group` can be combined with Abelian groups via `ProductGroup`. The SU(2) factor must always be the **last** component:

```python
from nicole import ProductGroup, U1Group, Z2Group, SU2Group

# U(1) × SU(2): particle number + full spin rotation (e.g. Hubbard model)
group = ProductGroup([U1Group(), SU2Group()])
print(group.name)      # "U1×SU2"
print(group.neutral)   # (0, 0)
print(group.is_abelian)  # False

# Charges are tuples (n, 2j)
# e.g. (2, 1) means 2 particles above, spin-1/2 multiplet
```

## Index and Sector Semantics

For SU(2) tensors, `Sector(charge=2j, dim=n)` means `n` independent spin-j **multiplets**. The total number of physical states in that sector is `n × (2j + 1)`, accessible via `Index.num_states`:

```python
from nicole import Index, Sector, Direction, SU2Group

group = SU2Group()
idx = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=1),  # 1 singlet  → 1 physical state
        Sector(charge=1, dim=2),  # 2 doublets → 4 physical states
        Sector(charge=2, dim=1),  # 1 triplet  → 3 physical states
    )
)

print(idx.dim)         # 4  (total multiplets: 1 + 2 + 1)
print(idx.num_states)  # 8  (total states: 1×1 + 2×2 + 1×3)
```

## See Also

- [Overview](overview.md): Symmetry system introduction
- [U1Group](u1-group.md): Abelian U(1) symmetry
- [Z2Group](z2-group.md): Abelian Z(2) symmetry
- [ProductGroup](product-group.md): Combine SU(2) with Abelian groups
- [Examples: SU(2)](../../examples/symmetries/su2-examples.md)

## Notes

- SU(2) tensors carry **intertwiners** (yuzuha `Bridge` objects) attached to each data block; these encode the Clebsch–Gordan structure in the R-W-C decomposition and are managed automatically by Nicole. See the [Yuzuha Protocol](../../getting-started/yuzuha-protocol.md) for a full explanation.
- All standard operations (`contract`, `trace`, `decomp`, `permute`, `conj`, etc.) handle SU(2) transparently.
- `is_abelian` returns `False` for `SU2Group` and for any `ProductGroup` containing it.
