# Tensor.random

Create a tensor with all admissible blocks filled with random values.

::: nicole.Tensor.random
    options:
      show_source: false
      heading_level: 2

## Description

Creates a new tensor with random values in all charge-conserving blocks. Random values are drawn from a standard normal distribution. Use the `seed` parameter for reproducibility.

## See Also

- [zeros](zeros.md): Create zero tensor
- [Tensor](../core/tensor.md): Main tensor class
- [Examples: First Tensor](../../examples/basic/first-tensor.md)

## Notes

- Random values from standard normal: mean=0, std=1
- Use `seed` parameter for reproducible results
- Only admissible blocks are created and filled
