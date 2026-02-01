# oplus

Direct sum (block diagonal) of tensors.

::: nicole.oplus
    options:
      show_source: false
      heading_level: 2

## Description

Creates a direct sum (block diagonal concatenation) of multiple tensors. All tensors must have:
- Same number of indices
- Matching directions for corresponding indices
- Same symmetry groups
- Matching tags

When charges collide, blocks are placed on the block diagonal.

## See Also

- [Arithmetic Operations](addition.md): Basic operations
- [Tensor](../core/tensor.md): Main tensor class
- [Examples: Arithmetic](../../examples/basic/arithmetic.md)

## Notes

Resulting tensor has combined sectors from all input tensors. For colliding charges, blocks are arranged diagonally (not summed).
