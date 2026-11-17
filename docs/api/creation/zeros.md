# Tensor.zeros

Allocate a symmetry-aware tensor whose every admissible block is filled with zeros. The indices you pass fix the entire structure: `zeros` enumerates the charge combinations they allow, keeps the ones that conserve charge, and allocates a dense block of the implied shape for each. Reach for it when a tensor's symmetry structure is known ahead of its contents, such as an accumulator to be filled block by block.

::: nicole.Tensor.zeros
    options:
      show_source: false
      heading_level: 2

## Description

A block is admissible when the charges drawn from its indices satisfy the group's conservation rule. For an Abelian group this means the signed sum `∑(OUT) − ∑(IN)` equals the neutral charge. Blocks that fail the test are never created, so the block count follows from the index sectors and the symmetry group. Each block's shape comes from the sector dimensions its charge key selects.

Sectors that end up in no admissible block are pruned from the returned indices, so the tensor handed back can carry narrower indices than the ones supplied. For a generic symmetry group every block also receives a `Bridge` intertwiner holding the default Clebsch–Gordan specification, and the stored block gains a trailing multiplicity dimension.

## See Also

- [random](random.md): Same structure, filled with random values
- [identity](identity.md): 2-index identity tensor
- [Tensor](../core/tensor.md): Main tensor class
- [Index](../core/index-class.md): Index structure that determines the blocks
- [Examples: First Tensor](../../examples/basic/first-tensor.md)

## Notes

- The default `dtype` is `torch.float64`, and it covers every block as well as the intertwiner weights of a generic group. On MPS, `float64` and `complex128` are downgraded automatically to `float32` and `complex64`.
- `device` defaults to `torch.get_default_device()`, so a tensor follows whatever default the surrounding code has set.
- Gradient tracking is off by default. Pass `requires_grad=True` to enable it, as described in [Autograd](../../examples/advanced/autograd.md).
- Every itag defaults to `"_init_"`. Set `itags` at creation when the tensor will take part in automatic contraction, which pairs indices by matching tag.
