# merge_axes

Merge multiple tensor axes into a single axis using isometry fusion.

::: nicole.merge_axes
    options:
      show_source: false
      heading_level: 2

## Description

Creates an n-to-1 isometry that fuses the specified axes, contracts it with the input tensor to perform the merging, and returns both the merged tensor and the conjugate of the isometry (which can be used to unfuse the axis later).

This is useful for reducing tensor order while preserving all information, with the ability to reverse the operation.

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

- Must specify at least 2 axes to merge
- Axes can be specified as integer positions (0-indexed) or string tags
- The merged axis appears first in the output tensor
- The returned isometry conjugate can contract with the merged tensor to reverse the operation
- Axes are merged in the order they appear in the tensor, not the order specified
- The merged axis direction can be controlled via the `direction` parameter
