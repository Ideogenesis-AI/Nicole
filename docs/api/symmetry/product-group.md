# ProductGroup

Multiple simultaneous symmetries.

::: nicole.ProductGroup
    options:
      show_source: false
      heading_level: 2
      members:
        - __init__
        - name
        - neutral
        - inverse
        - fuse
        - equal
        - validate_charge

## Description

Combines multiple independent symmetry groups. Charges are tuples with one component per group.

### Charge Operations

All operations are performed component-wise:

- **Fusion**: `(q1_a, q1_b) ⊕ (q2_a, q2_b) = (q1_a ⊕ q2_a, q1_b ⊕ q2_b)`
- **Inverse**: Component-wise inverse
- **Identity**: Tuple of component identities

## Physical Applications

- **U(1) × U(1)**: Particle number and spin
- **U(1) × Z(2)**: Particle number and parity
- **U(1)ᴺ**: Multiple particle species

## See Also

- [U1Group](u1-group.md): Component group
- [Z2Group](z2-group.md): Component group
- [Overview](overview.md): Symmetry system introduction
- [Examples: Product Groups](../../examples/symmetries/product-examples.md)

## Notes

Currently supports only Abelian component groups. Charges are tuples matching the number of component groups.
