# qr

QR decomposition returning orthogonal and upper triangular tensors.

::: nicole.decomp.qr
    options:
      show_source: false
      heading_level: 2

## Description

Performs block-wise QR decomposition, factorizing a tensor into:
- **Q**: Orthogonal tensor (Q†Q = I)
- **R**: Upper triangular tensor

Such that T = Q @ R. Unlike SVD, no truncation is applied.

The decomposition is performed block-wise, preserving symmetry structure. For each charge sector, blocks with the same left charge are concatenated horizontally, QR decomposed together, and then R is split back into individual blocks.

## Parameters

### T
Tensor to be decomposed.

### axis
Axis to separate from all others. Can be:
- Integer position (e.g., `0`)
- String itag (e.g., `"physical"`)

This axis forms the left partition (Q), all others form the right partition (R).

## Returns

`tuple[Tensor, Tensor]`: Pair `(Q, R)` where:
- Q has indices `(left_index, bond_index)` and is orthogonal
- R has indices `(bond_index.flip(), *right_indices)` and is upper triangular

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
