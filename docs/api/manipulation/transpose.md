# transpose

Return tensor with transposed axes.

::: nicole.transpose
    options:
      show_source: false
      heading_level: 2

## Description

Returns a new tensor with axes transposed. By default, reverses all axes. Optionally specify a custom permutation order.

This is the functional (non-mutating) version. For in-place transpose, use `Tensor.transpose()`.

## See Also

- [Tensor.transpose](../core/tensor.md): In-place version
- [permute](permute.md): General axis permutation
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

Without arguments, reverses axis order. With arguments, equivalent to `permute()` with specified order.
