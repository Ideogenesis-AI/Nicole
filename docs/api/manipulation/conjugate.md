# conj

Return conjugated tensor with flipped index directions.

::: nicole.conj
    options:
      show_source: false
      heading_level: 2

## Description

Returns a new tensor with:
- Conjugated block data (for complex dtypes)
- All index directions flipped (OUT ↔ IN)

This is the functional (non-mutating) version. For in-place conjugation, use `Tensor.conj()`.

## See Also

- [Tensor.conj](../core/tensor.md): In-place version
- [transpose](transpose.md): Transpose axes
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

For real dtypes, only directions are flipped. For complex dtypes, data is conjugated and directions flipped.
