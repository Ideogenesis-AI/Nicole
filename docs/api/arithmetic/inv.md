# inv

Invert a diagonal matrix tensor by taking the reciprocal of each diagonal element. The result comes back transposed, with its two indices swapped and flipped, which lets it contract against the tensor it came from and yield an identity. The usual subject is a singular-value or eigenvalue tensor from [diag](diag.md), where the inverse usually restores the environment weights.

::: nicole.inv
    options:
      show_source: false
      heading_level: 2

## Description

The input must be 2nd-order and its blocks square, with zeros off the diagonal. Both a tensor from [diag](diag.md) and a hand-built one qualify, but the check differs: a tensor labeled `Diagonal` is trusted and the off-diagonal scan is skipped, while any other label is verified block by block and raises `ValueError` if an off-diagonal entry exceeds machine epsilon.

Inversion is elementwise on the diagonal, so a vanishing element has no inverse. An entry below machine epsilon for `float64` raises `ZeroDivisionError` to avoid returning an infinity. For a generic symmetry group the diagonal is read out of the block's leading multiplicity slice and written back with the trailing dimension intact; the intertwiner weights encode the `√(irrep_dim)` normalization and are unaffected by inversion, so they carry over with the transposed directions.

## Usage

```python
from nicole import decomp, diag, inv, contract

# Create diagonal tensor from SVD
U, S_blocks, Vh = decomp(T, axes=0, mode="SVD")
S_diag = diag(S_blocks, U.indices[1])

# Invert the diagonal matrix
S_inv = inv(S_diag)

# Verify: S @ S_inv should give identity (approximately)
result = contract(S_diag, S_inv)
```

## Manual Diagonal Tensor

```python
import torch
from nicole import Tensor, Index, Sector, Direction, U1Group, inv

group = U1Group()
idx = Index(Direction.IN, group, sectors=(Sector(0, 2),))

# Create manual diagonal tensor
D = Tensor(
    indices=(idx.flip(), idx),
    itags=("i", "j"),
    data={(0, 0): torch.diag(torch.tensor([2.0, 4.0]))},
    label="Diagonal"
)

# Invert it
D_inv = inv(D)
# D_inv.data[(0, 0)] = [[0.5, 0], [0, 0.25]]
```

## See Also

- [diag](diag.md): Create diagonal tensor from 1D arrays
- [decomp](../decomposition/decomp.md): Tensor decomposition
- [identity](../creation/identity.md): Identity tensor

## Notes

- The output indices are `(indices[1].flip(), indices[0].flip())` and the itags are swapped to match, so `contract(S, inv(S))` pairs the bond correctly without further preparation.
- The `Diagonal` label is the only way to skip the off-diagonal scan. Hand-built tensors that are known to be diagonal are worth labeling as such when they are inverted in a loop.
- A singular spectrum needs truncation before inversion, since `inv` refuses a zero without regularizing it.
