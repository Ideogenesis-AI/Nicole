# qr

QR decomposition returning orthogonal and upper triangular tensors. One axis is separated from all the others and the tensor is factored as `T = Q @ R`, with the isometric factor `Q` carrying the designated axis and `R` carrying the rest. No truncation is involved, so the factorization is exact and the bond it opens is as small as the tensor allows: this is the cheap way to put a tensor into canonical form when no compression is required.

::: nicole.decomp.qr
    options:
      show_source: false
      heading_level: 2

## Description

The bond index between the two factors carries exactly the charges of the separated axis, with its direction reversed, so `Q` comes out with `(left_index, bond_index)` and `R` with the flipped bond followed by the remaining indices. Each bond sector holds the rank of its charge block, meaning the smaller of the separated dimension and the combined dimension of everything else — a sector wide on the left but narrow on the right shrinks accordingly, and the bond never grows beyond what the factorization supports.

Both bond indices are tagged `"_bond"`, and the other itags are carried over from `T`. `qr` takes one axis, as an integer position or an itag, and raises `ValueError` if the itag is missing, if it occurs on more than one axis, or if an integer position is out of range. To separate several axes at once, or to choose the bond itag, go through [decomp](decomp.md) with `mode="QR"`, which merges the axes first before calling this function.

## Import

```python
from nicole.decomp import qr
```

(Not exported in public API - use `decomp(..., mode="QR")` for most cases)

## Usage Examples

```python
from nicole.decomp import qr

# Basic QR decomposition
Q, R = qr(T, axis=0)

# Using itag
Q, R = qr(T, axis="physical")

# Verify orthogonality: Q†Q = I
from nicole import contract
QdagQ = contract(Q.conj(), Q)
```

## See Also

- [decomp](decomp.md): High-level decomposition with QR mode
- [svd](svd.md): Low-level SVD function
- [eig](eig.md): Eigenvalue decomposition
- [Examples: Decomposition](../../examples/operations/decomposition-examples.md)

## Notes

- Use `decomp(..., mode="QR")` for most cases with additional options (flow, itag customization)
- Use `qr()` directly when you need simple, low-level QR decomposition
- No truncation is applied in QR decomposition
- QR is useful for obtaining canonical forms without compression
- Contracting `Q` with its conjugate over the separated axis gives the identity on the bond, one identity block per bond charge
