# Tensor.random

Allocate a symmetry-aware tensor whose every admissible block is filled with random values. The structure is determined exactly as in [zeros](zeros.md) — the indices decide which charge combinations are allowed and what shape each block takes — and only the contents differ. A random tensor is the usual starting point for an iterative algorithm and the standard fixture for checking that an operation handles symmetry correctly.

::: nicole.Tensor.random
    options:
      show_source: false
      heading_level: 2

## Description

Admissibility, block shapes, sector pruning, and intertwiner handling all follow [zeros](zeros.md). Each block is drawn from `torch.randn`, so its entries are independent samples from a standard normal distribution with mean 0 and standard deviation 1. For a complex `dtype`, the real and imaginary parts are drawn separately and combined, giving each part unit variance.

Pass `seed` when reproducibility matters. The seed initializes a `torch.Generator` on the target device that then supplies every block, so a seeded call reproduces exactly without disturbing the global random stream. Left unseeded, the blocks are drawn from the global generator and differ between runs.

## See Also

- [zeros](zeros.md): Same structure, filled with zeros
- [Tensor](../core/tensor.md): Main tensor class, including `rand_fill` for refilling in place
- [Index](../core/index-class.md): Index structure that determines the blocks
- [Examples: First Tensor](../../examples/basic/first-tensor.md)

## Notes

- The default `dtype` is `torch.float64`, and it covers every block as well as the intertwiner weights of a generic group. On MPS, `float64` and `complex128` are downgraded automatically to `float32` and `complex64`.
- The result is not normalized. Its `norm()` grows with the number and size of the admissible blocks, so divide by `norm()` when a unit-norm tensor is wanted.
- Gradient tracking is off by default. Pass `requires_grad=True` to enable it, as described in [Autograd](../../examples/advanced/autograd.md).
- Seeding is per-call. Two calls sharing one seed produce the same values only when their indices and dtype agree, since the generator is consumed block by block.
