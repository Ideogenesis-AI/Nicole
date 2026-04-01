# transpose

Return tensor with transposed axes.

::: nicole.transpose
    options:
      show_source: false
      heading_level: 2

## Description

Returns a fully independent tensor with all axes reversed. All data blocks are deep-copied, so the result is completely isolated from the original.

Note: `Tensor.transpose()` defaults to `in_place=False` — it returns a new tensor and leaves the original unchanged. To modify the tensor in place instead, call `Tensor.transpose(in_place=True)`. For arbitrary axis reordering, use [`permute`](permute.md).

## See Also

- [Tensor.transpose](../core/tensor.md): Method form (out-of-place by default)
- [permute](permute.md): General axis permutation
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

Always reverses axis order. For arbitrary reordering, use [`permute`](permute.md).
