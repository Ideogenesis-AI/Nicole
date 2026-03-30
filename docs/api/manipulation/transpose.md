# transpose

Return tensor with transposed axes.

::: nicole.transpose
    options:
      show_source: false
      heading_level: 2

## Description

Returns a fully independent tensor with axes transposed. By default, reverses all axes. Optionally specify a custom permutation order. All data blocks are deep-copied, so the result is completely isolated from the original.

Note: `Tensor.transpose()` defaults to `in_place=True` (it modifies the tensor in place and returns `self`). To obtain a new tensor without modifying the original, call `Tensor.transpose(in_place=False)` or use this standalone function.

## See Also

- [Tensor.transpose](../core/tensor.md): Method form (in-place by default)
- [permute](permute.md): General axis permutation
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

Without arguments, reverses axis order. With arguments, equivalent to `permute()` with specified order.
