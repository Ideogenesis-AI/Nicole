# identity

Create a 2-index identity tensor.

::: nicole.identity
    options:
      show_source: false
      heading_level: 2

## Description

Returns a tensor with two indices (original and flipped) containing identity matrices for each charge sector. Represents the Kronecker delta δᵢⱼ with symmetry structure.

## Block Structure

For each sector with charge `q` and dimension `d`:
- Block `(q, q)` contains the `d × d` identity matrix
- All other blocks are empty (not stored)

## See Also

- [isometry](isometry.md): Create fusion tensor
- [Tensor](../core/tensor.md): Main tensor class
- [Examples: Build Operators](../../examples/advanced/build-operators.md)

## Notes

The second index is automatically flipped (opposite direction) to enable contraction. Use for resolution of identity: ∑ᵢ |i⟩⟨i| = I.
