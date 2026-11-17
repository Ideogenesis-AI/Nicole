# decomp

High-level tensor decomposition with multiple output modes. This is the front door to factorization: name the axes that go to the left factor and a mode, and the tensor is factorized along a fresh bond with every charge sector decomposed on its own. The modes differ in how the singular values are distributed over the factors returned, so an algorithm can ask for the grouping it will actually require without ever bothering to absorb one back in afterwards.

::: nicole.decomp.decomp
    options:
      show_source: false
      heading_level: 2

## Description

Each charge sector is decomposed on its own, so the bond carries one sector per charge that survives and both factors inherit the symmetry of the input with no further work.

### Bond Arrows

`flow` fixes the directions of the two bond indices, whichever way the separated axis points. In the SVD-based modes the choice is visible on the index pair of `S`:

| `flow` | `S` directions | Reading |
|:------:|:--------------:|:--------|
| `"><"` (default) | `(IN, IN)` | arrows converge from `U` and `Vh` into `S` |
| `">>"` | `(IN, OUT)` | an arrow chain flowing left to right |
| `"<<"` | `(OUT, IN)` | an arrow chain flowing right to left |

`"UR"` keeps one bond and so has nothing to converge on: it reads `"><"` as `">>"`, and `"LV"` reads it as `"<<"`.

### Choosing a Mode

`"UR"` and `"LV"` cost less than `"SVD"` whenever the singular values are going to be absorbed into a neighbor anyway, since the diagonal tensor is never built. `"RSVD"` pays off only when the rank kept is well below the full rank of a sector and the spectrum decays — [svd](svd.md) carries the measured crossover. `"QR"` truncates nothing, and a `trunc` dict handed to it is ignored.

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
