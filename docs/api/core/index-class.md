# Index

Symmetry-aware tensor index defining direction, group, and charge sectors.

::: nicole.Index
    options:
      show_source: false
      heading_level: 2
      members:
        - __init__
        - dim
        - flip
        - dual
        - charges
        - sector_dim_map

## Description

An `Index` represents a single leg of a tensor, specifying:

- **Direction**: `Direction.OUT` or `Direction.IN`
- **Symmetry Group**: The group governing charge conservation
- **Sectors**: Available charge sectors with their dimensions

Indices are immutable (frozen dataclasses) to ensure consistency across tensor operations.

## Direction Rules

- **`Direction.OUT`**: Charge contributes with positive sign (+)
- **`Direction.IN`**: Charge contributes with negative sign (-)

For charge conservation: `∑(OUT) - ∑(IN) = neutral`

## Methods

### flip()

Returns a copy with direction reversed but charges unchanged.

### dual()

Returns a copy with direction reversed AND charges conjugated.

## See Also

- [Sector](sector.md): Charge-dimension pairs
- [Direction](direction.md): Index direction enum
- [Tensor](tensor.md): Main tensor class
- [Examples: First Tensor](../../examples/basic/first-tensor.md)

## Notes

For contraction, indices must have:
1. Opposite directions (OUT ↔ IN)
2. Same symmetry group
3. Compatible charge sectors
