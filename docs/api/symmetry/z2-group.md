# Z2Group

Z(2) symmetry group with binary charges.

::: nicole.Z2Group
    options:
      show_source: false
      heading_level: 2
      members:
        - name
        - neutral
        - inverse
        - fuse
        - equal
        - validate_charge

## Description

Represents discrete Z(2) symmetry. Charges are 0 or 1, representing binary quantum numbers.

### Charge Operations

- **Fusion**: XOR (`q1 ⊕ q2`)
- **Inverse**: Self-inverse (0→0, 1→1)
- **Identity**: 0

## Physical Applications

- **Fermion parity** (even/odd number)
- **Spatial inversion** symmetry
- **Time-reversal** (in some contexts)

## See Also

- [U1Group](u1-group.md): Integer charge symmetry
- [ProductGroup](product-group.md): Multiple symmetries
- [Overview](overview.md): Symmetry system introduction
- [Examples: Z2](../../examples/symmetries/z2-examples.md)

## Notes

Only charges 0 and 1 are valid. Other values raise `ValueError`.
