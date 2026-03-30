# ProductGroup

Combine multiple symmetry groups into a single product group, enforcing all constituent conservation laws simultaneously. Charges are tuples of component charges.

::: nicole.ProductGroup
    options:
      show_source: false
      heading_level: 2
      members:
        - __init__
        - name
        - neutral
        - is_abelian
        - irrep_dim
        - dual
        - fuse_unique
        - fuse_channels
        - equal
        - validate_charge

## Description

Combines multiple independent symmetry groups. Charges are tuples with one component per group.

### Abelian and Non-Abelian Components

`ProductGroup` supports both Abelian and non-Abelian component groups. When an `SU2Group` is included, it must be the **last** component:

```python
from nicole import ProductGroup, U1Group, Z2Group, SU2Group

# Abelian only
group = ProductGroup([U1Group(), Z2Group()])
print(group.is_abelian)   # True

# With SU(2) as last component — non-Abelian
group = ProductGroup([U1Group(), SU2Group()])
print(group.is_abelian)  # False
print(group.name)        # "U1×SU2"
print(group.neutral)     # (0, 0)
```

### Charge Operations

For Abelian-only products, all operations are component-wise:

- **Fusion**: `(q1_a, q1_b) ⊕ (q2_a, q2_b) = (q1_a ⊕ q2_a, q1_b ⊕ q2_b)`
- **Dual**: Component-wise dual
- **Identity**: Tuple of component identities

For products containing `SU2Group`, use `fuse_channels` instead of `fuse_unique` — SU(2) fusion is multi-channel:

```python
# Abelian: fuse_unique gives a single result
group = ProductGroup([U1Group(), Z2Group()])
group.fuse_unique((2, 1), (1, 1))   # (3, 0)  — U1: 2+1=3, Z2: 1⊕1=0

# Non-Abelian: fuse_channels gives multiple result channels
group = ProductGroup([U1Group(), SU2Group()])
# (1, 1) ⊗ (1, 1): charge 1+1=2 for U1, 1⊗1=(0,2) for SU2
group.fuse_channels((1, 1), (1, 1))   # ((2, 0), (2, 2))
```

### `irrep_dim`

`irrep_dim` is the product of the component `irrep_dim` values — always 1 for Abelian factors, `2j+1` for SU(2):

```python
# Abelian: always 1
group = ProductGroup([U1Group(), Z2Group()])
group.irrep_dim((3, 1))   # 1 × 1 = 1

# Non-Abelian: SU(2) factor contributes 2j+1
group = ProductGroup([U1Group(), SU2Group()])
group.irrep_dim((2, 1))   # 1 × 2 = 2  (U1 contributes 1, SU2 spin-1/2 contributes 2)
group.irrep_dim((0, 2))   # 1 × 3 = 3  (triplet)
```

## Physical Applications

- **U(1) × U(1)**: Particle number and spin magnetization (two separate U(1) charges)
- **U(1) × Z(2)**: Particle number and fermion parity
- **U(1)ᴺ**: Multiple particle species
- **U(1) × SU(2)**: Particle number + full spin-rotation symmetry (Hubbard/band models)
- **Z(2) × SU(2)**: Parity + full spin-rotation symmetry

## See Also

- [U1Group](u1-group.md): Abelian component group
- [Z2Group](z2-group.md): Abelian component group
- [SU2Group](su2-group.md): Non-Abelian component group
- [Overview](overview.md): Symmetry system introduction
- [Examples: Product Groups](../../examples/symmetries/product-examples.md)

## Notes

- `SU2Group`, if present, must be the **last** component of `ProductGroup`.
- When `SU2Group` is a component, `is_abelian` returns `False` and tensors carry intertwiner (`Bridge`) data.
- Charges are tuples matching the number of component groups.
