# isometry

Create a 3-leg fusion tensor for combining indices.

::: nicole.isometry
    options:
      show_source: false
      heading_level: 2

## Description

Returns a tensor that fuses two indices into a combined index. The fusion follows the symmetry group's charge combination rules (e.g., addition for U(1), XOR for Z(2)).

## Structure

- **Indices 1 & 2**: Input indices to be fused
- **Index 3**: Fused index with combined sectors

Charges are fused: `q_fused = group.fuse(q₁, q₂)`

## See Also

- [identity](identity.md): Create identity tensor
- [Index](../core/index-class.md): Index structure
- [ProductGroup](../symmetry/product-group.md): Multiple symmetries
- [Examples: Build Operators](../../examples/advanced/build-operators.md)

## Notes

Both input indices must share the same symmetry group. The fused index dimension is the product of input dimensions for matching charge combinations.
