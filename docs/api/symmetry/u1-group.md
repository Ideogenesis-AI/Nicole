# U1Group

U(1) symmetry group with integer charges.

::: nicole.U1Group
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

Represents continuous U(1) symmetry. Charges are integers representing conserved quantum numbers.

### Charge Operations

- **Fusion**: Addition (`q1 + q2`)
- **Inverse**: Negation (`-q`)
- **Identity**: 0

## Physical Applications

- **Particle number** conservation
- **Magnetization** (Sz) conservation
- **Electric charge** conservation

## See Also

- [Z2Group](z2-group.md): Binary symmetry
- [ProductGroup](product-group.md): Multiple symmetries
- [Overview](overview.md): Symmetry system introduction
- [Examples: U1](../../examples/symmetries/u1-examples.md)

## Notes

All integers are valid charges. No bounds or restrictions on charge values.
