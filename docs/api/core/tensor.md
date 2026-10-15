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
        - squeeze
        - trim_zero_blocks
        - normalize_sectors
        - regularize
        - sorted_keys
        - key
        - block
        - group
        - show
        - conj
        - real
        - imag
        - check_dtype
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
- **Coherent dtype**: Blocks and intertwiner weights always agree
- **Autograd control**: Optional gradient tracking
- **Immutable indices**: Index structure fixed at creation

### Data Types for Tensors

A tensor's `dtype` covers the whole object: every dense block and, for
non-Abelian tensors, every intertwiner weight matrix carries it. The default is
`float64`.

Casting is explicit, through `to(dtype=...)`, which converts blocks and weights
together. A complex-to-real cast is refused, since discarding an imaginary part
should never be implicit — use `real` or `imag` to say which part you want.
Note that `imag` returns zeros for a real tensor, following NumPy, whereas
`torch.Tensor.imag` rejects a real input.

Operations on two tensors promote them to a common dtype, so a real tensor can
be combined with a complex one directly. This covers `+` and `-` as well as
[contract](../contraction/contract.md), [einsum](../contraction/einsum.md) and
[oplus](../arithmetic/oplus.md). Two things to watch:

- Scalar operations preserve precision rather than widening, so a `float32`
  tensor times `1j` gives `complex64`, not `complex128`.
- `==` compares dtype as well as values, so the same values at two dtypes are
  not equal. Use [allclose](../arithmetic/allclose.md) when only the values
  matter.

## See Also

- [Index](index-class.md): Define tensor index structure
- [zeros](../creation/zeros.md): Create zero tensor
- [random](../creation/random.md): Create random tensor
- [Examples: Creating Tensors](../../examples/basic/first-tensor.md)
- [Examples: Arithmetic](../../examples/basic/arithmetic.md)

## Notes

- Tensors are mutable objects. Use `clone()` when independence is needed. For functional (non-mutating) operations, see the [operators](../manipulation/conjugate.md) module.
- `Tensor` supports `==` and `!=` for exact structural and element-wise equality. For numerical equality with tolerances, use `allclose(A, B)` instead — see [allclose](../arithmetic/allclose.md).
