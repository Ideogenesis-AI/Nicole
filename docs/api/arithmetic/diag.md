# diag

Turn a dictionary of per-sector 1D arrays into a diagonal matrix tensor. It is the way back from a spectrum into the tensor algebra: [svd](../decomposition/svd.md) and [eig](../decomposition/eig.md) hand their singular values and eigenvalues back as a dict keyed by charge, and `diag` makes that a 2nd-order tensor that contractions can use.

::: nicole.diag
    options:
      show_source: false
      heading_level: 2

## Description

Each 1D array in `S_blocks` becomes the diagonal of one matrix block, under the charge key that it arrived with. The two output indices are built from `bond_index`, one flipped and one as given, so the tensor is ready to sit on the bond the decomposition opened. Every input array must be 1-dimensional; anything else raises `ValueError`.

The output is tagged `("_bond_L", "_bond_R")` unless `itags` says otherwise, and labeled `Diagonal`. That label is load-bearing: [inv](inv.md) uses it to skip the check that a tensor really is diagonal. For a generic symmetry group the blocks gain the trailing multiplicity dimension and an intertwiner whose weights are set to `√(irrep_dim(q))`, matching the convention [identity](../creation/identity.md) uses.

## Usage

```python
from nicole import decomp, diag, contract

# Perform decomposition to get singular value dict
U, S_blocks, Vh = decomp(T, axes=0, mode="SVD")

# Convert S_blocks (dict of 1D arrays) to diagonal matrix tensor
S_diag = diag(S_blocks, U.indices[1], itags=("left", "right"))

# Now can use S_diag in contractions
result = contract(U, S_diag)  # Equivalent to U @ S
```

## See Also

- [decomp](../decomposition/decomp.md): Tensor decomposition returning S as dict
- [inv](inv.md): Matrix inversion for diagonal tensors
- [identity](../creation/identity.md): The same block structure with unit entries
- [oplus](oplus.md): Direct sum operation

## Notes

- The result carries the indices `(bond_index.flip(), bond_index)` and one block per charge `q` in the input, keyed `(q, q)`. No off-diagonal key can conserve charge.
- `dtype` and `device` are inferred from the input arrays when not given, so a spectrum computed on the GPU stays there.
- The input dict is primarily the `S` output of `decomp` or `svd` and the `D` output of `eig`, all of which key their arrays by charge in exactly this form.
