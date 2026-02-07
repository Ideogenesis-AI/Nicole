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

Use `decomp()` for most cases. Use `svd()` when you need per-block singular value access before tensor creation.
