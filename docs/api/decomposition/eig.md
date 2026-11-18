# eig

Eigenvalue decomposition of square matrices. The input must be a 2nd-order tensor whose two indices have opposite directions and matching charge structure, and each charge block is diagonalized on its own, so the spectrum comes back labeled by sector. Eigenvalues are returned as plain arrays keyed by block, and they can be sorted from either end and truncated, making this the routine useful for e.g. spectrum analysis and dominant-mode searches.

::: nicole.decomp.eig
    options:
      show_source: false
      heading_level: 2

## Description

Charge conservation on a square tensor allows only diagonal blocks, so the two indices pair each charge with itself and every block is diagonalized in isolation. The bond index therefore carries the charges of the row index, each sector as wide as the number of eigenvalues kept in that charge, and `U` comes out with `(row_index, bond_index)`. Because the eigenvalues arrive sector by sector, a `"thresh"` criterion filters within a block while `"nkeep"` has to compare across blocks, thereby the two are applied in that order when both are given.

!!! tip
    Always set `is_hermitian=True` when diagonalising Hamiltonians or density matrices — it is both faster and numerically more reliable. The general eigensolver may return complex eigenvalues even for a real input, and its eigenvectors are only normalized column by column, so `U` is unitary just to the extent the decomposition happens to deliver. The Hermitian path guarantees real eigenvalues and exactly orthonormal eigenvectors, degeneracies included.

## Import

```python
from nicole.decomp import eig
```

(Not exported in public API)

## Usage Examples

```python
from nicole.decomp import eig

# No truncation, ascending order (default - smallest eigenvalues first)
U, D_blocks = eig(T)

# Descending order (largest eigenvalues first)
U, D_blocks = eig(T, order="descend")

# Keep 5 smallest (most negative) eigenvalues - useful for ground states
U, D_blocks = eig(T, order="ascend", trunc={"nkeep": 5})

# Keep 5 largest (most positive) eigenvalues
U, D_blocks = eig(T, order="descend", trunc={"nkeep": 5})

# Keep eigenvalues >= 0.1 (positive eigenvalues above threshold)
U, D_blocks = eig(T, order="descend", trunc={"thresh": 0.1})

# Keep eigenvalues <= -0.5 (negative eigenvalues below threshold)
U, D_blocks = eig(T, order="ascend", trunc={"thresh": -0.5})

# Apply both: keep eigenvalues >= 0.1, then keep top 5
U, D_blocks = eig(T, order="descend", trunc={"thresh": 0.1, "nkeep": 5})

# Hermitian eigensolver — real eigenvalues, orthonormal eigenvectors
U, D_blocks = eig(T, is_hermitian=True)

# Hermitian + keep 5 lowest eigenvalues (e.g. ground state and low-lying excitations)
U, D_blocks = eig(T, is_hermitian=True, order="ascend", trunc={"nkeep": 5})
```

## See Also

- [decomp](decomp.md): High-level decomposition
- [svd](svd.md): Low-level SVD function
- [qr](qr.md): Low-level QR function

## Notes

- An eigendecomposition needs the canonical intertwiner weights in place, so `eig` calls `T.regularize()` on its argument before it starts. The tensor this represents does not change — the call only shifts a factor between the reduced data and the intertwiner weights — but for a non-Abelian input the stored blocks are rewritten in passing. Abelian tensors are untouched.
- Truncation can empty a charge sector, and an emptied sector leaves the bond index altogether. Its key disappears from `D` as well, so a block missing from the returned mapping is one that kept nothing.
- `order` decides which end of the spectrum both truncation criteria work from, so `order="ascend"` with `trunc` is the ground-state request and `order="descend"` the dominant-mode one.
