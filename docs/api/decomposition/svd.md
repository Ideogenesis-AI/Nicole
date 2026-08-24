# svd

Low-level SVD returning singular values as dictionary.

::: nicole.decomp.svd
    options:
      show_source: false
      heading_level: 2

## Description

Performs block-wise SVD, returning:
- **U**: Left unitary tensor
- **S_dict**: Dictionary mapping block keys to 1D singular value arrays
- **Vh**: Right unitary tensor
- **info** *(optional)*: Dict of auxiliary data, returned only when `requires_info=True`. Currently contains:
  - `"discarded_weight"`: `float` in `[0, 1)`, the relative sum of squared singular values truncated away across all charge sectors, i.e. the fractional squared 2-norm loss `‖T − T_trunc‖²/‖T‖²`.

This low-level function provides direct access to singular values for each block before they're combined into a tensor.

## Import

```python
from nicole.decomp import svd
```

(Not exported in public API)

## See Also

- [decomp](decomp.md): High-level decomposition
- [qr](qr.md): Low-level QR function
- [eig](eig.md): Eigenvalue decomposition
- [Examples: Decomposition](../../examples/operations/decomposition-examples.md)

## Notes

- Use `decomp()` for most cases. Use `svd()` when you need per-block singular value access before tensor creation.
- Pass `requires_info=True` to obtain the `info` dict containing `"discarded_weight"`. This incurs a small extra computation and is intended for diagnostic use.
- `"discarded_weight"` is a relative squared weight: squared singular values normalized by the full spectrum. For a generic symmetry group each reduced singular value carries a multiplicity factor `irrep_dim(q)`, the number of degenerate Schmidt values it stands for. The factor weights the reported value only, not the `"thresh"` or `"nkeep"` truncation criteria.
- For Abelian tensors, `Vh` satisfies `Vh @ Vh† = I` directly from the standard block-wise matrix SVD.
- For SU(2) tensors, the intertwiner weight matrix is canonicalized to a scaled unitary form before the SVD, ensuring `Vh` is a physical isometry (`Vh @ Vh† = I`).
