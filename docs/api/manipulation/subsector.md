# subsector

Extract subset of tensor blocks and prune unused sectors.

::: nicole.subsector
    options:
      show_source: false
      heading_level: 2

## Description

Returns a new tensor containing only specified blocks, with unused sectors automatically removed from the indices. Block indices are 1-based to match display output from `print(tensor)`.

Useful for:
- Extracting specific charge sectors
- Filtering by quantum numbers
- Debugging tensor structure
- Creating tensors with reduced sector structure

## See Also

- [Tensor](../core/tensor.md): Main tensor class
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

Block indices are 1-based. Use `Tensor.key(i)` to get the charge key for block `i`.

The returned tensor will have indices with only the sectors that appear in the selected blocks. This ensures the tensor structure remains consistent and minimal.
