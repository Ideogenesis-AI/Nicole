# eig

Eigenvalue decomposition of square matrix tensors.

::: nicole.decomp.eig
    options:
      show_source: false
      heading_level: 2

## Description

Performs block-wise eigenvalue decomposition of a square matrix tensor, returning:
- **U**: Tensor containing eigenvectors as columns
- **D**: Dictionary mapping block keys to 1D eigenvalue arrays

The decomposition satisfies: T @ U = U @ diag(D) for each block.

Eigenvalues can be sorted in ascending or descending order, and truncation can be applied to keep only the most important eigenvalues (useful for finding ground states or dominant modes).

## Parameters

### T
Square matrix tensor to be decomposed. Must have exactly 2 indices with matching charge structure (opposite directions).

### itag
Index tag for the bond dimension. If `None`, uses default tag `"_bond_eig"`.

### order
Sorting order for eigenvalues:
- `"ascend"`: Ascending order (smallest to largest) - default
  - For real eigenvalues: e.g., -5 < -3 < 1 < 2
  - For complex eigenvalues: sorts by real part
  - Use for finding ground states (most negative eigenvalues)
- `"descend"`: Descending order (largest to smallest)
  - Use for finding dominant modes (most positive eigenvalues)

### trunc
Optional truncation specification as a dictionary:
- `"nkeep"`: Keep at most n eigenvalues globally
  - With `order="descend"`: keeps the n largest (most positive)
  - With `order="ascend"`: keeps the n smallest (most negative)
- `"thresh"`: Keep eigenvalues relative to threshold per block
  - With `order="descend"`: keeps eigenvalues ≥ threshold
  - With `order="ascend"`: keeps eigenvalues ≤ threshold
- Both can be specified: thresh applied first, then nkeep

## Returns

`tuple[Tensor, MutableMapping[BlockKey, torch.Tensor]]`: Pair `(U, D)` where:
- U has indices `(row_index, bond_index)` containing eigenvectors as columns
- D is a dictionary mapping block keys to 1D arrays of eigenvalues

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
```

## See Also

- [decomp](decomp.md): High-level decomposition
- [svd](svd.md): Low-level SVD function
- [qr](qr.md): Low-level QR function

## Notes

- Eigenvalues are returned as 1D arrays for memory efficiency
- Eigenvalues can be complex even for real matrices
- Use `order="ascend"` to find ground states (smallest/most negative eigenvalues)
- Use `order="descend"` to find dominant modes (largest/most positive eigenvalues)
- For complex eigenvalues, sorting is by real part only
- When both thresh and nkeep specified: thresh is applied per-block first, then nkeep globally
- Eigenvectors are stored as columns of U, normalized to be unitary (or as close as possible)
