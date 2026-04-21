# oplus

Direct sum (block diagonal) of tensors.

::: nicole.oplus
    options:
      show_source: false
      heading_level: 2

## Description

Creates a direct sum (block diagonal concatenation) of two tensors. The tensors must have:

- The same number of indices
- Matching directions for every corresponding index
- The same symmetry group on every corresponding index

The `axes` parameter controls which indices are *merged* (block-diagonal stacking) and which are *non-merged* (shared). For non-merged axes, only charge sectors that appear in **both** tensors must agree in dimension. Sectors exclusive to one tensor are included in the output via the union of sectors and contribute their blocks independently.

When charges coincide on merged axes, blocks are placed on the block diagonal (not summed).

## See Also

- [Arithmetic Operations](addition.md): Basic operations
- [Tensor](../core/tensor.md): Main tensor class
- [Examples: Arithmetic](../../examples/basic/arithmetic.md)

## Notes

- The output index for each merged axis is the union of sectors from both tensors, with dimensions added (`dim_A + dim_B` per shared charge, or `dim_A` / `dim_B` for exclusive charges).
- The output index for each non-merged axis is also the union of sectors, but dimensions of shared charges must be equal (they are not summed).
- For non-Abelian groups, linearly dependent multiplicity components are automatically compressed away after the merge.
