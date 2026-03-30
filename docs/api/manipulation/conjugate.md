# conj

Return conjugated tensor with flipped index directions.

::: nicole.conj
    options:
      show_source: false
      heading_level: 2

## Description

Returns a fully independent tensor with:
- Conjugated block data (for complex dtypes)
- All index directions flipped (OUT ↔ IN)

All data blocks are deep-copied, so the result is completely isolated from the original. For a memory-efficient version that shares storage with the original, use `Tensor.conj()` (default `in_place=False`). To modify the tensor in place, use `Tensor.conj(in_place=True)`.

## See Also

- [Tensor.conj](../core/tensor.md): Method form (shared-storage default, or in-place)
- [transpose](transpose.md): Transpose axes
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

For real dtypes, only directions are flipped. For complex dtypes, data is conjugated and directions flipped.
