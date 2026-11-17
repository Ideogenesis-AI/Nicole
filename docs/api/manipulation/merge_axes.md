# merge_axes

Fuse several axes of a tensor into one, and receive alongside the result the isometry conjugate that undoes it. The merge loses nothing: the fused axis carries every charge combination the originals admitted, so a contraction against the returned conjugate restores the tensor as it was. It is the standard way to reshape a tensor for an operation that expects fewer axes, such as a matrix decomposition across a chosen partition.

::: nicole.merge_axes
    options:
      show_source: false
      heading_level: 2

## Description

The work is done by [isometry_n](../creation/isometry_n.md): an n-to-1 isometry is built over the selected indices and contracted with the tensor, which performs the merge and leaves the fused axis in place of the originals. The merged axis comes first in the output, followed by the axes left alone in their original order, so the resultant's itags are not those of the input in a shorter tuple.

The selection accepts integer positions and itags interchangeably, and at least two axes are required. The merged axis takes `merged_tag` as its tag, defaulting to `"_merged_"`, and `direction` as its orientation, defaulting to `Direction.OUT`. Fewer than two axes, an axis absent from the tensor, or a specification that is neither `int` nor `str` raises an error.

## Usage

### Merge by tags

```python
from nicole import Tensor, merge_axes

# Create a 4-index tensor
tensor = Tensor.random([idx1, idx2, idx3, idx4], itags=['a', 'b', 'c', 'd'])

# Merge three axes
merged, iso_conj = merge_axes(tensor, ['a', 'b', 'c'], merged_tag='abc')
print(merged.itags)  # ('abc', 'd')
```

### Merge by positions

```python
# Merge using integer positions (0-indexed)
merged, iso_conj = merge_axes(tensor, [0, 1, 2], merged_tag='merged')
```

### Unfuse merged axis

```python
from nicole import contract

# Reverse the merging operation
unmerged = contract(merged, iso_conj)
# unmerged should approximately match the original tensor structure
```

## See Also

- [isometry](../creation/isometry.md): 2-to-1 fusion isometry
- [isometry_n](../creation/isometry_n.md): N-to-1 fusion isometry
- [contract](../contraction/contract.md): Tensor contraction
- [permute](permute.md): Reorder axes

## Notes

- Axes fuse in the order they occupy in the tensor, not the order given in `axes`. Listing them differently reorders nothing, so the fused index is the same whichever way the selection is written.
- The isometry conjugate is the only route back. Discarding it leaves the merged axis fused for good, since the charge combinations behind it cannot be recovered from the merged tensor alone.
- Integer positions are 0-based, matching [retag](retag.md) and [permute](permute.md).
