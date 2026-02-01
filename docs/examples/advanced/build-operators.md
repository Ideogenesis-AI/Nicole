# Build Operators

Physical operators in quantum many-body systems must respect symmetries and charge conservation. This page demonstrates how to construct common operators—identity, isometry, and ladder operators—as symmetry-aware tensors.

**Key operators:**

- **Identity**: Diagonal operators mapping each state to itself
- **Isometry**: Fusion tensors that combine multiple indices
- **Number operator**: Diagonal operator returning charge values
- **Ladder operators**: Creation and annihilation with auxiliary indices for charge conservation
- **Spin operators**: \(S_z\) (diagonal z-component)

Operators with **auxiliary indices** (like creation/annihilation or spin raising/lowering) need an extra index to ensure the total charge is conserved. These auxiliary indices have specific charges that balance the charge transfer between physical states.

```python exec="1" session="build-operators" result=""
from nicole import identity, isometry, Tensor, Index, Sector, Direction, U1Group
import numpy as np
```

## Identity Operator

```python exec="1" session="build-operators" result="console" idprefix="" source="material-block"
group = U1Group()
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))

# Create identity operator
I = identity(idx, itags=("i", "j"))

print(f"Identity has {len(I.data)} blocks\n{I}")
```

## Fusion Isometry

```python exec="1" session="build-operators" result="console" idprefix="" source="material-block"
# Fuse two indices into one
idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

# Create fusion tensor
iso = isometry(idx1, idx2, itags=("i", "j", "ij"))

print(f"Isometry indices: {iso.itags}")
print(f"Fused index dim: {iso.indices[2].dim}\n{iso}")
```

## Number Operator

```python exec="1" session="build-operators" result="console" idprefix="" source="material-block"
# Diagonal operator that returns the charge value
idx_num = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1), Sector(2, 1)))

# Manually construct number operator
N_data = {}
for sector in idx_num.sectors:
    charge = sector.charge
    dim = sector.dim
    # Diagonal matrix with charge values
    N_data[(charge, charge)] = np.eye(dim) * charge

N = Tensor(indices=(idx_num, idx_num.flip()), itags=("out", "in"), data=N_data)

print(f"Number operator:\n{N}")
```

## Ladder Operators

```python exec="1" session="build-operators" result="console" idprefix="" source="material-block"
# Creation and annihilation operators for bosons
# a†|n⟩ = √(n+1)|n+1⟩
# a|n⟩ = √n|n-1⟩
# Need auxiliary indices to conserve total charge

n_max = 3
idx_ladder = Index(Direction.OUT, group, sectors=tuple(Sector(n, 1) for n in range(n_max + 1)))

# Auxiliary index with charge +1 for creation
idx_aux_plus = Index(Direction.OUT, group, sectors=(Sector(1, 1),))

# Creation operator a†: (out, in, aux+1)
a_dag_data = {}
for n in range(n_max):
    # Connects |n⟩ to |n+1⟩, charge conserved: (n+1) + (-n) + (-1) = 0
    a_dag_data[(n + 1, n, 1)] = np.array([[[np.sqrt(n + 1)]]])

a_dag = Tensor(
    indices=(idx_ladder, idx_ladder.flip(), idx_aux_plus.flip()),
    itags=("out", "in", "aux"),
    data=a_dag_data
)

print(f"Creation operator has {len(a_dag.data)} blocks\n{a_dag}\n")

# Auxiliary index with charge -1 for annihilation
idx_aux_minus = Index(Direction.OUT, group, sectors=(Sector(-1, 1),))

# Annihilation operator a: (out, in, aux-1)
a_data = {}
for n in range(1, n_max + 1):
    # Connects |n⟩ to |n-1⟩, charge conserved: (n-1) + (-n) + (1) = 0
    a_data[(n - 1, n, -1)] = np.array([[[np.sqrt(n)]]])

a = Tensor(
    indices=(idx_ladder, idx_ladder.flip(), idx_aux_minus.flip()),
    itags=("out", "in", "aux"),
    data=a_data
)

print(f"Annihilation operator has {len(a.data)} blocks\n{a}")
```

## Spin Operators

```python exec="1" session="build-operators" result="console" idprefix="" source="material-block"
# Sz operator for spin-1/2
# |↓⟩ has Sz = -1/2, |↑⟩ has Sz = +1/2
# Using units where 2*Sz is an integer

idx_spin = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(1, 1)))

# Sz is diagonal
Sz_data = {
    (-1, -1): np.array([[-0.5]]),  # |↓⟩
    (1, 1): np.array([[0.5]]),     # |↑⟩
}

Sz = Tensor(indices=(idx_spin, idx_spin.flip()), itags=("out", "in"), data=Sz_data)

print(f"Sz operator:\n{Sz}")
```

## See Also

- API Reference: [identity](../../api/creation/identity.md)
- API Reference: [isometry](../../api/creation/isometry.md)
- Next: [Load Space](load-space-examples.md)
