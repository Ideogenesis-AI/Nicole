# decomp

High-level tensor decomposition with multiple output modes.

::: nicole.decomp.decomp
    options:
      show_source: false
      heading_level: 2

## Description

Performs symmetry-preserving decomposition with four output modes:

### Modes

- **"SVD"**: Returns (U, S, Vh) with S as diagonal tensor
- **"UR"**: Returns (U, R) where R = S @ Vh (most efficient for reconstruction)
- **"LV"**: Returns (L, V) where L = U @ S
- **"QR"**: Returns (Q, R) where Q is orthogonal and R is upper triangular (no truncation)

Each symmetry sector is decomposed independently.

## Parameters

### axes
Index or indices to separate from all others. Can be:
- Single integer position or string tag
- Sequence of integer positions or string tags (merges multiple axes first)

### mode
Decomposition mode: "SVD", "UR", "LV", or "QR" (default: "SVD")

- **"SVD"**, **"UR"**, **"LV"**: SVD-based decomposition with optional truncation
- **"QR"**: QR decomposition (orthogonal factorization, no truncation)

### flow
Arrow direction control for bond indices (default: "><"):
- **"><"**: Both bond arrows incoming — arrows converge from U and Vh into S; S has `(IN, IN)` (default)
- **">>"**: Arrow chain flows left to right — S has `(IN, OUT)`
- **"<<"**: Arrow chain flows right to left — S has `(OUT, IN)`

### itag
Index tag(s) for the bond dimension(s):
- `None`: Use default tags "_bond_L" and "_bond_R"
- `str`: Use same tag for both left and right bonds
- `tuple[str, str]`: Use (left_tag, right_tag) for left and right bonds

### trunc
Optional truncation with `trunc` parameter (dict) - only for SVD-based modes:
- **"nkeep"**: Keep n largest singular values globally
- **"thresh"**: Keep singular values ≥ t per block
- Both can be specified together (thresh applied first, then nkeep)
- Not applicable for "QR" mode (QR decomposition doesn't support truncation)

## Usage Examples

```python
from nicole import decomp

# UR mode (most efficient)
U, R = decomp(T, axes=0, mode="UR")

# SVD mode with custom bond tags
U, S, Vh = decomp(T, axes=0, mode="SVD", itag=("left", "right"))

# LV mode with truncation
L, V = decomp(T, axes=0, mode="LV", trunc={"nkeep": 100})

# QR mode (orthogonal factorization, no truncation)
Q, R = decomp(T, axes=0, mode="QR")

# Decompose merging multiple axes
U, R = decomp(T, axes=[0, 1, 2], mode="UR")
```

## See Also

- [svd](svd.md): Low-level SVD function
- [qr](qr.md): Low-level QR function
- [eig](eig.md): Eigenvalue decomposition
- [merge_axes](../manipulation/merge_axes.md): Merge multiple axes
- [Examples: Decomposition](../../examples/operations/decomposition-examples.md)

## Notes

- UR and LV modes are more efficient than SVD mode for reconstruction
- QR mode provides orthogonal factorization without truncation
- Bond dimension after truncation may be smaller (SVD-based modes only)
- Charge sectors with zero singular values are automatically removed (SVD-based modes)
- When multiple axes specified, they are merged first using n-to-1 isometry
- QR mode is useful for obtaining canonical forms without compression
