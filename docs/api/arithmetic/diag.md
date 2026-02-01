# diag

Convert diagonal blocks into a diagonal matrix tensor.

::: nicole.diag
    options:
      show_source: false
      heading_level: 2

## Description

Takes a dictionary of 1D arrays (such as singular values from SVD or eigenvalues from eigendecomposition) and creates a diagonal matrix tensor where each 1D array becomes a diagonal matrix block.

This is useful for converting the singular values dict from `svd()` or eigenvalues dict from `eig()` into full diagonal matrix form for explicit matrix operations like contraction.

## Usage

```python
from nicole import decomp, diag, contract

# Perform decomposition to get singular value dict
U, S_blocks, Vh = decomp(T, axes=0, mode="UR")

# Convert S_blocks (dict of 1D arrays) to diagonal matrix tensor
S_diag = diag(S_blocks, U.indices[1], itags=("left", "right"))

# Now can use S_diag in contractions
result = contract(U, S_diag)  # Equivalent to U @ S
```

## See Also

- [decomp](../decomposition/decomp.md): Tensor decomposition returning S as dict
- [inv](inv.md): Matrix inversion for diagonal tensors
- [oplus](oplus.md): Direct sum operation

## Notes

The output tensor will have:
- Two indices: (bond_index.flip(), bond_index)
- Block keys (q, q) for each charge q in input dict
- Diagonal matrices as data blocks
- Label "Diagonal"

All input arrays must be 1-dimensional (one value per charge sector).
