# permute

Return tensor with permuted axes.

::: nicole.permute
    options:
      show_source: false
      heading_level: 2

## Description

Returns a new tensor with axes reordered according to the specified permutation. This is the functional (non-mutating) version. For in-place permutation, use `Tensor.permute()`.

## See Also

- [Tensor.permute](../core/tensor.md): In-place version
- [transpose](transpose.md): Reverse axis order
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

The `order` parameter must be a valid permutation of `range(len(tensor.indices))`. Each block is transposed accordingly.
