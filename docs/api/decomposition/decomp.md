# decomp

High-level tensor decomposition with multiple output modes.

::: nicole.decomp
    options:
      show_source: false
      heading_level: 2

## Description

Performs symmetry-preserving SVD with three output modes:

### Modes

- **"SVD"**: Returns (U, S, Vh) with S as diagonal tensor
- **"UR"**: Returns (U, R) where R = S @ Vh (most efficient for reconstruction)
- **"LV"**: Returns (L, V) where L = U @ S

Each symmetry sector is decomposed independently.

## Parameters

### axes
Index or indices to separate from all others. Can be:
- Single integer position or string tag
- Sequence of integer positions or string tags (merges multiple axes first)

### mode
Decomposition mode: "SVD", "UR", or "LV" (default: "SVD")

### flow
Arrow direction control for bond indices (default: "><"):
- **"><"**: Both arrows incoming (default)
- **">>"**: Both arrows outgoing
- **"<<"**: Both arrows incoming

### itag
Index tag(s) for the bond dimension(s):
- `None`: Use default tags "_bond_L" and "_bond_R"
- `str`: Use same tag for both left and right bonds
- `tuple[str, str]`: Use (left_tag, right_tag) for left and right bonds

### trunc
Optional truncation with `trunc` parameter (dict):
- **"nkeep"**: Keep n largest singular values globally
- **"thresh"**: Keep singular values ≥ t per block
- Both can be specified together (thresh applied first, then nkeep)

## Usage Examples

```python
from nicole import decomp

# UR mode (most efficient)
U, R = decomp(T, axes=0, mode="UR")

# SVD mode with custom bond tags
U, S, Vh = decomp(T, axes=0, mode="SVD", itag=("left", "right"))

# LV mode with truncation
L, V = decomp(T, axes=0, mode="LV", trunc={"nkeep": 100})

# Decompose merging multiple axes
U, R = decomp(T, axes=[0, 1, 2], mode="UR")
```

## See Also

- [svd](svd.md): Low-level SVD function
- [merge_axes](../manipulation/merge_axes.md): Merge multiple axes
- [Examples: Decomposition](../../examples/operations/decomposition-examples.md)

## Notes

- UR and LV modes are more efficient than SVD mode
- Bond dimension after truncation may be smaller
- Charge sectors with zero singular values are automatically removed
- When multiple axes specified, they are merged first using n-to-1 isometry
