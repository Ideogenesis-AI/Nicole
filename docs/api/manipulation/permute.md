# permute

Return tensor with permuted axes.

::: nicole.permute
    options:
      show_source: false
      heading_level: 2

## Description

Returns a fully independent tensor with axes reordered according to the specified permutation. All data blocks are deep-copied, so the result is completely isolated from the original. For a memory-efficient version that shares storage with the original, use `Tensor.permute()` (default `in_place=False`). To modify the tensor in place, use `Tensor.permute(order, in_place=True)`.

## See Also

- [Tensor.permute](../core/tensor.md): Method form (shared-storage default, or in-place)
- [transpose](transpose.md): Reverse axis order
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

The `order` parameter must be a valid permutation of `range(len(tensor.indices))`. Each block is transposed accordingly.
