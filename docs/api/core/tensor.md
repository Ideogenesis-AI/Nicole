# Tensor

Block-sparse tensor with symmetry-aware indices.

::: nicole.Tensor
    options:
      show_source: false
      heading_level: 2
      members:
        - zeros
        - random
        - from_scalar
        - is_scalar
        - item
        - norm
        - clone
        - rand_fill
        - insert_index
        - trim_zero_blocks
        - normalize_sectors
        - regularize
        - sorted_keys
        - key
        - block
        - group
        - show
        - conj
        - permute
        - transpose
        - retag
        - invert
        - device
        - to
        - cpu
        - cuda
        - requires_grad
        - backward

## Description

The `Tensor` class is the core data structure in Nicole, representing block-sparse tensors backed by symmetry-aware indices. Each tensor stores a collection of dense PyTorch tensor blocks, where each block corresponds to a specific combination of charges that satisfies charge conservation rules.

### Key Features

- **Block-sparse storage**: Only admissible blocks are stored
- **Automatic charge conservation**: Selection rules enforced by structure
- **PyTorch-backed blocks**: Dense operations within each symmetry sector
- **Device management**: CPU and GPU (CUDA/MPS) support
- **Autograd control**: Optional gradient tracking
- **Immutable indices**: Index structure fixed at creation

## See Also

- [Index](index-class.md): Define tensor index structure
- [zeros](../creation/zeros.md): Create zero tensor
- [random](../creation/random.md): Create random tensor
- [Examples: Creating Tensors](../../examples/basic/first-tensor.md)
- [Examples: Arithmetic](../../examples/basic/arithmetic.md)

## Notes

- Tensors are mutable objects. Use `clone()` when independence is needed. For functional (non-mutating) operations, see the [operators](../manipulation/conjugate.md) module.
- `Tensor` supports `==` and `!=` for exact structural and element-wise equality. For numerical equality with tolerances, use `allclose(A, B)` instead — see [allclose](../arithmetic/allclose.md).
