# Tensor.retag

Update tensor index tags.

::: nicole.Tensor.retag
    options:
      show_source: false
      heading_level: 2

## Description

Changes the tags (labels) of tensor indices. Can update specific tags by name or position, or replace all tags at once.

This is an in-place operation.

## Usage Modes

1. **By keyword**: `retag(old_tag="new_tag")`
2. **By sequence**: `retag(["tag1", "tag2", "tag3"])`

## See Also

- [Tensor](../core/tensor.md): Main tensor class
- [Index](../core/index-class.md): Index structure
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

Tags are used for automatic contraction matching. Changing tags doesn't affect tensor data, only metadata.
