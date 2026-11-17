# oplus

Form the direct sum of two tensors, placing their blocks block-diagonally along the axes you choose. Where a matrix direct sum sets A and B into opposite corners of a larger matrix, `oplus` does that per charge sector, and the `axes` argument decides which axes grow by stacking and which stay shared. It is the operation behind widening a bond or assembling a block-structured operator out of parts.

::: nicole.oplus
    options:
      show_source: false
      heading_level: 2

## Description

The two tensors must agree on their frame:

- The same number of indices
- Matching directions for every corresponding index
- The same symmetry group on every corresponding index

The `axes` parameter controls which indices are *stacked* (block-diagonal) and which are *shared*. It accepts an axis position, an itag, a sequence of either, or `None` to stack every axis. For shared axes, charge sectors that appear in **both** tensors must agree in dimension. Sectors exclusive to one tensor are included in the output via the union of sectors and contribute their blocks independently.

When charges coincide on stacked axes, blocks are placed on the block diagonal (not summed).

## See Also

- [Arithmetic Operations](addition.md): Basic operations
- [Tensor](../core/tensor.md): Main tensor class
- [Examples: Arithmetic](../../examples/basic/arithmetic.md)

## Notes

- The output index for each stacked axis is the union of sectors from both tensors, with dimensions added (`dim_A + dim_B` per shared charge, or `dim_A` / `dim_B` for exclusive charges).
- The output index for each shared axis is also the union of sectors, but dimensions of common charges must be equal (they are not summed).
- For non-Abelian groups, linearly dependent multiplicity components are automatically compressed away afterwards.
