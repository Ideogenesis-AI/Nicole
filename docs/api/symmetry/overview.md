# Symmetry Groups Overview

Nicole's symmetry system enables efficient block-sparse tensor operations by exploiting conserved quantum numbers.

## Available Groups

### Abelian Groups

- **[U1Group](u1-group.md)**: Continuous U(1) symmetry with integer charges
- **[Z2Group](z2-group.md)**: Binary Z(2) symmetry with charges 0 or 1
- **[ProductGroup](product-group.md)**: Combination of Abelian groups (e.g. U(1) × Z(2))

### Non-Abelian Groups

- **[SU2Group](su2-group.md)**: SU(2) rotational symmetry with spin-based quantum numbers
- **[ProductGroup](product-group.md)**: Combination of Abelian groups with `SU2Group` as the last component (e.g. U(1) × SU(2))

## Core Concepts

### Charges

Charges are quantum numbers that label symmetry sectors:

- **U(1)**: Integers (..., -2, -1, 0, 1, 2, ...)
- **Z(2)**: Binary (0, 1)
- **SU(2)**: Non-negative integers using the **2j convention** (0, 1, 2, 3, ...) for physical spins 0, 1/2, 1, 3/2, ...
- **ProductGroup**: Tuples of component charges, e.g. `(n, 2j)` for U(1) × SU(2)

### Charge Operations

All symmetry groups support:

- **`neutral`**: Identity element (0 for U1, Z2, and SU2)
- **`dual(q)`**: Dual/contragredient representation; self-dual for SU(2)
- **`equal(a, b)`**: Test equality
- **`is_abelian`**: `True` for U1/Z2/Abelian ProductGroup; `False` for SU2Group and ProductGroup containing SU2
- **`irrep_dim(q)`**: Dimension of irrep for charge `q`; always 1 for Abelian groups; `2j+1` for SU(2)

### Charge Fusion

Charge fusion depends on whether the group is Abelian:

- **Abelian groups** (U1, Z2, Abelian ProductGroup): `fuse_unique(*qs)` combines any number of charges into a single result (addition for U1, XOR for Z2).
- **Non-Abelian groups** (SU2, ProductGroup with SU2): two charges can fuse into **multiple** result channels. Use `fuse_channels` instead:

```python
from nicole import U1Group, SU2Group

# Abelian: always a unique result
u1 = U1Group()
u1.fuse_unique(2, 3)          # 5

# Non-Abelian: multiple channels
su2 = SU2Group()
su2.fuse_channels(1, 1)       # (0, 2)  — spin-0 or spin-1
su2.fuse_channels(2, 2)       # (0, 2, 4)  — spin-0, spin-1, or spin-2
```

### Charge Conservation

Tensor blocks must satisfy:

```
∑(OUT charges) - ∑(IN charges) = neutral element
```

This is enforced automatically throughout Nicole.

## Physical Examples

### U(1) Applications

- Particle number conservation
- Magnetization (Sz) conservation
- Electric charge

### Z(2) Applications

- Fermion parity (even/odd)
- Spatial inversion symmetry
- Time-reversal (in some contexts)

### SU(2) Applications

- Isotropic spin chains (Heisenberg model with full spin-rotation invariance)
- Angular momentum conservation in multi-particle systems
- Hubbard model with spin SU(2) symmetry (`U1 × SU2`)

SU(2) tensors use the Wigner–Eckart decomposition internally; see the [Yuzuha Protocol](../../getting-started/yuzuha-protocol.md) for the R-W-C structure and compression rationale.

### ProductGroup Applications

- Particle number + spin (U1 × U1)
- Particle number + parity (U1 × Z2)
- Particle number + full spin rotation (U1 × SU2)
- Parity + full spin rotation (Z2 × SU2)

## See Also

- [U1Group](u1-group.md): Integer charge symmetry
- [Z2Group](z2-group.md): Binary symmetry
- [SU2Group](su2-group.md): Non-Abelian spin symmetry
- [ProductGroup](product-group.md): Multiple symmetries
- [Examples: U(1)](../../examples/symmetries/u1-examples.md)
- [Examples: SU(2)](../../examples/symmetries/su2-examples.md)

## Notes

Symmetry groups are immutable and lightweight. Groups are typically shared across many indices and tensors.
