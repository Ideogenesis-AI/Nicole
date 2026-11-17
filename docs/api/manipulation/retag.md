# Tensor.retag

Rename the itags of a tensor's indices in place. Tags are how automatic contraction and trace identify the axes to pair, so retagging is the step that wires a tensor into a network; the data is never touched. Tensors from `zeros` and `random` arrive with every itag set to `"_init_"`, which makes this the normal first operation.

::: nicole.Tensor.retag
    options:
      show_source: false
      heading_level: 2

## Description

The call accepts three forms. A mapping renames tags by name and leaves any tag absent from the mapping as it stands. A full sequence of strings replaces every tag at once and must match the tensor's order. A position, or a sequence of positions, paired with `new_tags` updates those axes alone.

```python
tensor.retag({"a": "left", "b": "right"})  # by name, selectively
tensor.retag(["left", "middle", "right"])  # all tags at once
tensor.retag([0, 2], ["left", "right"])    # by position
tensor.retag(0, "left")                    # a single position
```

Only the itags change. The data blocks, index directions, charge sectors, and block keys are all left as they were, and the method mutates the tensor and returns `None`, so there is nothing to assign from it.

## See Also

- [Tensor](../core/tensor.md): Main tensor class
- [Index](../core/index-class.md): Index structure the tags label
- [contract](../contraction/contract.md): Pairs axes by matching itag
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

- A position outside the tensor's order raises `IndexError`, and a position list whose length differs from `new_tags` raises `ValueError`.
- Positions are 0-based here, unlike the 1-based block numbering of the display and of [filter_blocks](filter_blocks.md).
- Automatic contraction pairs every axis whose tag matches, so give the axes that should stay free a tag distinct from anything on the other tensor if intended.
