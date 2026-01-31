# Symmetry Groups Overview

Nicole's symmetry system enables efficient block-sparse tensor operations by exploiting conserved quantum numbers.

## Available Groups

### Abelian Groups

- **[U1Group](u1-group.md)**: Continuous U(1) symmetry with integer charges
- **[Z2Group](z2-group.md)**: Binary Z(2) symmetry with charges 0 or 1
- **[ProductGroup](product-group.md)**: Multiple simultaneous symmetries

## Core Concepts

### Charges

Charges are quantum numbers that label symmetry sectors:

- **U(1)**: Integers (..., -2, -1, 0, 1, 2, ...)
- **Z(2)**: Binary (0, 1)
- **ProductGroup**: Tuples of component charges

### Charge Operations

All symmetry groups support:

- **`neutral`**: Identity element (0 for U1 and Z2)
- **`inverse(q)`**: Inverse element
- **`fuse(*qs)`**: Combine charges (addition for U1, XOR for Z2)
- **`equal(a, b)`**: Test equality
- **`dual(q)`**: Dual/contragredient charge

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

### ProductGroup Applications

- Particle number + spin (U1 × U1)
- Particle number + parity (U1 × Z2)
- Multiple particle species (U1ᴺ)

## See Also

- [U1Group](u1-group.md): Integer charge symmetry
- [Z2Group](z2-group.md): Binary symmetry
- [ProductGroup](product-group.md): Multiple symmetries
- [Examples: Symmetries](../../examples/symmetries/u1-examples.md)

## Notes

Symmetry groups are immutable and lightweight. Groups are typically shared across many indices and tensors.
