# Bridge

Storage container for the SU(2) Clebsch–Gordan intertwiner data attached to each block of a non-Abelian tensor.

::: nicole.symmetry.delegate.Bridge
    options:
      show_source: false
      heading_level: 2
      members:
        - cgspec
        - weights
        - om_dimension
        - num_components
        - num_external
        - device
        - from_block
        - to
        - clone
        - conj
        - conj_phase

## Description

Every block of an SU(2) or non-Abelian `ProductGroup` tensor carries a `Bridge` alongside its reduced-tensor data. The `Bridge` encodes the **outer multiplicity (OM) structure** of the block's Clebsch–Gordan decomposition using the [yuzuha](https://github.com/ideogenesis-ai/Yuzuha) canonical basis:

- **`cgspec`** (`yuzuha.CGSpec`) — identifies the canonical OM basis. It records the external edges (spins and their in/out directions) and all valid left-associative fusion trees (internal-spin tuples α). The OM dimension is the number of such trees.
- **`weights`** (`torch.Tensor`, shape `(num_components, om_dimension)`) — coefficient matrix expanded in the canonical basis. Each row corresponds to one component of the outer multiplicity; the number of rows grows as blocks mix under operations such as contraction.

Nicole manages `Bridge` objects internally. Users typically inspect them when examining tensor blocks or implementing custom SU(2)-aware routines.

### Role in the R-W-C Decomposition

Each SU(2) tensor block is stored as a three-factor product **R · W · C**:

- **R** (reduced tensor, `Tensor.data[key]`): the dense array of reduced tensor elements, indexed over the multiplet space of each index (`Sector.dim`).
- **W** (weight matrix, `Bridge.weights`, shape `(dim(r), dim(α))`): expands the component in the canonical OM basis. Each row is a vector in the OM space.
- **C** (CG basis): the Clebsch–Gordan basis tensor with one axis per external edge and one OM axis (α), in a canonical orthonormal basis, computed on demand by the Yuzuha engine.

The connecting index between R and W is `r` (component); between W and C is `α` (outer multiplicity). See the [Yuzuha Protocol](../../getting-started/yuzuha-protocol.md) for the full derivation and diagrams.

### Accessing Bridge Data

```python
from nicole import Tensor, SU2Group, Index, Sector, Direction

group = SU2Group()
idx = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 1)))
T = Tensor.random([idx, idx.flip(), idx], itags=["i", "j", "k"], seed=42)

# Iterate over blocks
for key, block in T.data.items():
    bridge = T.intw[key]             # Bridge for this block
    print(key, bridge.om_dimension)  # OM dimension for this charge sector
    print(bridge.weights.shape)      # (num_components, om_dimension)
```

### Constructing a Bridge

The `from_block` factory is the standard way to create a `Bridge` from a charge key and direction list:

```python
from nicole import SU2Group, Direction
from nicole.symmetry.delegate import Bridge

group = SU2Group()

# Three-edge block: spin-1/2 ⊗ spin-1/2 → spin-1
key = (1, 1, 2)                            # charges in 2j convention
dirs = [Direction.IN, Direction.IN, Direction.OUT]
bridge = Bridge.from_block(group, key, dirs)

print(bridge.om_dimension)   # 1 — only one valid fusion tree for j=1/2 ⊗ j=1/2 → j=1
print(bridge.num_components) # 1 — single component by default
print(bridge.weights)        # tensor([[1.]])
```

For `ProductGroup` tensors the `key` contains tuples and the SU(2) charge is always the last element:

```python
from nicole import ProductGroup, U1Group, SU2Group, Direction
from nicole.symmetry.delegate import Bridge
import torch

group = ProductGroup([U1Group(), SU2Group()])
key = ((0, 1), (-1, 1), (-1, 2))          # three edges: (U1, SU2) charges
dirs = [Direction.IN, Direction.IN, Direction.OUT]

bridge = Bridge.from_block(group, key, dirs)
print(bridge.num_external)   # 3
```

### Device and dtype

A `Bridge`'s `weights` tensor lives on the same device as the corresponding data block. Use `.to()` to move it:

```python
bridge_gpu = bridge.to('cuda')
bridge_gpu = bridge.to('cuda', dtype=torch.float32)  # also cast dtype
```

### OM Dimension

`om_dimension` is the number of linearly independent ways to couple the given spins to the target spin via a left-associative binary fusion tree. 2nd- and 3rd-order blocks always have OM = 1. OM > 1 first appears at 4th order, where multiple intermediate spin paths may be valid, and generally grows with both the tensor order and the spin magnitudes.

```python
from nicole import SU2Group, Direction
from nicole.symmetry.delegate import Bridge

group = SU2Group()

# Four edges: j=1/2 ⊗ j=1/2 ⊗ j=1/2 → j=1/2
# Two intermediate couplings: intermediate spin can be 0 or 1
key = (1, 1, 1, 1)
dirs = [Direction.IN, Direction.IN, Direction.IN, Direction.OUT]
bridge = Bridge.from_block(group, key, dirs)
print(bridge.om_dimension)   # 2
```

## See Also

- [SU2Group](su2-group.md): Non-Abelian symmetry group that generates Bridge-carrying tensors
- [ProductGroup](product-group.md): Non-Abelian product groups
- [Yuzuha Protocol](../../getting-started/yuzuha-protocol.md): Full explanation of the R-W-C decomposition and outer multiplicity
- [Examples: SU(2)](../../examples/symmetries/su2-examples.md)

## Notes

- `Bridge` is a `@dataclass`; all fields are set at construction time and validated by `__post_init__`.
- `cgspec` is shared between `Bridge` instances produced by non-mutating operations such as `.clone()`. Only `weights` is copied.
