# isometry_n

Create an n-to-1 fusion isometry tensor.

::: nicole.isometry_n
    options:
      show_source: false
      heading_level: 2

## Description

Constructs an (n+1)-leg tensor that fuses n indices into a single fused leg. This generalizes the 2-to-1 `isometry` function to arbitrary n≥2.

The function works by sequentially applying 2-to-1 isometries, fusing indices in order of increasing dimension to minimize intermediate tensor sizes and computational complexity.

The resulting isometry has n indices with directions opposite to the input indices (to enable natural contraction), plus one fused index whose direction is specified by the `direction` parameter.

## Usage

### Basic fusion

```python
from nicole import Index, Sector, Direction, U1Group, isometry_n

group = U1Group()
idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))

# Create 3-to-1 isometry
iso = isometry_n([idx1, idx2, idx3], itags=("a", "b", "c", "fused"))

# iso has 4 indices: first 3 are flipped (IN), last is fused (OUT by default)
print(iso.itags)  # ('a', 'b', 'c', 'fused')
```

### Contract with tensor to merge axes

```python
from nicole import Tensor, contract

# Create a 3-index tensor
T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"])

# Create isometry and contract to merge all axes
iso = isometry_n([idx1, idx2, idx3], itags=("a", "b", "c", "merged"))
merged = contract(T, iso)  # Now has single index "merged"
```

### Specify fused direction

```python
# Create isometry with IN direction for fused index
iso = isometry_n(
    [idx1, idx2, idx3],
    direction=Direction.IN,
    itags=("a", "b", "c", "fused_in")
)
```

## See Also

- [isometry](isometry.md): 2-to-1 fusion isometry
- [merge_axes](../manipulation/merge_axes.md): High-level axis merging
- [contract](../contraction/contract.md): Tensor contraction
- [identity](identity.md): Identity tensor

## Notes

- Requires at least 2 indices to fuse
- All indices must share the same symmetry group
- Indices are fused in order of increasing dimension for efficiency
- The first n output indices have opposite directions to the input indices
- The last output index is the fused index with specified direction
- If `itags` is provided, length must be n+1 (n indices + 1 fused)
- Only Abelian and ProductGroup symmetries are currently supported
