# inv

Invert a diagonal matrix tensor.

::: nicole.inv
    options:
      show_source: false
      heading_level: 2

## Description

Takes a diagonal matrix tensor and returns its inverse by inverting each diagonal element. The function handles diagonal tensors created by `diag()` or manually constructed diagonal matrices.

For charge conservation, the output tensor has transposed index structure (swapped and flipped indices) compared to the input.

## Usage

```python
from nicole import decomp, diag, inv, contract

# Create diagonal tensor from SVD
U, S_blocks, Vh = decomp(T, axes=0, mode="UR")
S_diag = diag(S_blocks, U.indices[1])

# Invert the diagonal matrix
S_inv = inv(S_diag)

# Verify: S @ S_inv should give identity (approximately)
result = contract(S_diag, S_inv)
```

## Manual Diagonal Tensor

```python
import numpy as np
from nicole import Tensor, Index, Sector, Direction, U1Group, inv

group = U1Group()
idx = Index(Direction.IN, group, sectors=(Sector(0, 2),))

# Create manual diagonal tensor
D = Tensor(
    indices=(idx.flip(), idx),
    itags=("i", "j"),
    data={(0, 0): np.diag([2.0, 4.0])},
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

- Input tensor must have exactly 2 indices
- Each block must be a square diagonal matrix
- Diagonal elements must be non-zero (raises ZeroDivisionError otherwise)
- If tensor label is "Diagonal", diagonal structure check is skipped for efficiency
- Output tensor has transposed index structure for charge conservation
