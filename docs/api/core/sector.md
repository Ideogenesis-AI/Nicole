# Sector

Charge sector descriptor pairing a charge with a dimension.

::: nicole.Sector
    options:
      show_source: false
      heading_level: 2
      members:
        - charge
        - dim

## Description

A `Sector` represents a symmetry sector within an index, containing:

- **charge**: The conserved quantum number (type depends on symmetry group)
- **dim**: Number of states in this sector (positive integer)

Sectors are immutable frozen dataclasses.

## Charge Types by Group

| Group | Charge Type | Examples |
|-------|-------------|----------|
| `U1Group` | `int` | `-2, -1, 0, 1, 2` |
| `Z2Group` | `int` (0 or 1) | `0, 1` |
| `ProductGroup` | `tuple` | `(2, 1), (0, 0)` |

## Validation

Sectors validate at construction:
- Dimension must be positive
- Charge must be valid for the group
- No duplicate charges allowed within an index

## See Also

- [Index](index-class.md): Uses sectors to define structure
- [Direction](direction.md): Index orientation
- [U1Group](../symmetry/u1-group.md): Integer charges
- [Z2Group](../symmetry/z2-group.md): Binary charges
- [Examples: First Tensor](../../examples/basic/first-tensor.md)
