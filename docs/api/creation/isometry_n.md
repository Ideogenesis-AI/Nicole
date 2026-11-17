# isometry_n

Build the tensor that fuses n indices into one, generalizing [isometry](isometry.md) beyond a single pair. The first n indices mirror the inputs with reversed directions and the last carries the fully fused sectors, so one contraction collapses n axes of a tensor into one and a contraction against the conjugate restores them. Fusion proceeds pairwise under the hood, ordered to keep the intermediates small.

::: nicole.isometry_n
    options:
      show_source: false
      heading_level: 2

## Description

The output is an (n+1)th-order tensor, defined for any n ≥ 2. Its first n indices take the directions opposite to the inputs, which lets the isometry contract against them directly, and the final fused index takes the direction given by the `direction` parameter.

Fusion is carried out by applying 2-to-1 isometries in sequence. The indices are consumed in order of increasing dimension, so the smallest pair fuses first and each intermediate tensor stays as small as the fusion tree allows, holding down both memory and the work of the subsequent contractions.

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
