# Tensor.zeros

Create a tensor with all admissible blocks filled with zeros.

::: nicole.Tensor.zeros
    options:
      show_source: false
      heading_level: 2

## Description

Creates a new tensor with zero-filled blocks for all charge combinations that satisfy conservation rules. The tensor structure is defined by the provided indices.

## See Also

- [random](random.md): Create random tensor
- [Tensor](../core/tensor.md): Main tensor class
- [Examples: First Tensor](../../examples/basic/first-tensor.md)

## Notes

Only admissible blocks (satisfying charge conservation) are created. The number of blocks depends on the index sectors and symmetry group.
