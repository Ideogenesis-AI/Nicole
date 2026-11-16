# identity

Build the 2nd-order identity tensor on an index and its flipped partner. Each charge sector contributes one diagonal block holding the identity matrix of that sector's dimension, making the tensor the symmetry-aware form of the Kronecker delta \(\delta_{ij}\). It serves as the neutral element of a contraction, as a resolution of the identity on a physical space, and as the starting point for building a diagonal operator.

::: nicole.identity.identity
    options:
      show_source: false
      heading_level: 2

## Description

The returned tensor carries the indices `(index, index.flip())`. Flipping reverses direction while leaving the charges untouched, so the identity contracts cleanly against either side of another tensor defined on that space.

For a generic symmetry group the diagonal blocks additionally carry a `Bridge` whose weights are set to `√(irrep_dim(q))`. The reduced block stays an identity on the multiplet space, and the normalization makes the full R-W-C product the identity on the complete physical space, where a single multiplet stands for `irrep_dim(q)` states.

## Block Structure

For each sector with charge `q` and dimension `d`:

- Block `(q, q)` contains the `d × d` identity matrix.
- All other blocks are absent. A key with two different charges cannot conserve charge, so it is never allocated, and the off-diagonal entries are implied as zeros.

## See Also

- [isometry](isometry.md): Create fusion tensor
- [diag](../arithmetic/diag.md): Build a diagonal tensor from per-sector values
- [Tensor](../core/tensor.md): Main tensor class
- [Yuzuha Protocol](../../getting-started/yuzuha-protocol.md): R-W-C decomposition behind the generic-group weights
- [Examples: Build Operators](../../examples/advanced/build-operators.md)

## Notes

- The default `dtype` is `torch.float64`, and `device` defaults to `torch.get_default_device()`. On MPS, `float64` and `complex128` are downgraded automatically.
- Both itags default to `"_init_"`. Pass `itags` as a 2-tuple to label the two indices for automatic contraction.
- The identity is defined by the sectors of `index` alone, so two indices with the same sectors but different directions yield the same blocks with opposite orientation.
