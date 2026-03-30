# SU(2) Symmetry Examples

SU(2) symmetry represents **full spin-rotation invariance**. Unlike Abelian groups, each SU(2) charge carries an internal structure — a multiplet of `2j+1` magnetic states — so tensors with SU(2) symmetry work in the **reduced tensor formalism**: they store reduced tensor elements, and Clebsch–Gordan coefficients are handled automatically under the hood.

**Common applications:**

- **Heisenberg spin chains**: Full SU(2) conservation for isotropic models
- **Hubbard model with SU(2) symmetry**: Separate charge (U(1)) and spin (SU(2)) channels
- **Angular momentum coupling**: General multi-particle spin states

SU(2) tensors automatically enforce the Wigner–Eckart theorem: a block labeled by charges `(j1, j2, ...)` exists only if the spins can be coupled to the total spin required by the tensor structure. Internally Nicole stores tensors in the reduced R-W-C representation; see the [Yuzuha Protocol](../../getting-started/yuzuha-protocol.md) for details.

```python exec="1" session="su2-examples" result=""
from nicole import Tensor, Index, Sector, Direction, SU2Group, U1Group, ProductGroup
from nicole import identity, contract, decomp, load_space
```

## The SU(2) Group

```python exec="1" session="su2-examples" result="console" idprefix="" source="material-block"
group = SU2Group()

print(f"Group name:      {group.name}")
print(f"Neutral element: {group.neutral}   (spin-0 singlet)")
print(f"Is Abelian:      {group.is_abelian}\n")

# Irrep dimension: 2j+1 states per multiplet
for two_j in range(5):
    j = two_j / 2
    print(f"  2j={two_j}  →  j={j}  →  irrep_dim={group.irrep_dim(two_j)}")
```

## Fusion Channels

Unlike Abelian groups where two charges fuse to a unique result, SU(2) spins can combine into **multiple** total spin channels:

```python exec="1" session="su2-examples" result="console" idprefix="" source="material-block"
# Spin-1/2 ⊗ spin-1/2 → singlet (j=0) or triplet (j=1)
print(f"1/2 ⊗ 1/2 → {group.fuse_channels(1, 1)}   (2j values: 0, 2)")

# Spin-1 ⊗ spin-1 → j=0, 1, or 2
print(f"1 ⊗ 1 →   {group.fuse_channels(2, 2)}  (2j values: 0, 2, 4)")

# Three spin-1/2 → j=1/2 or j=3/2
print(f"1/2⊗1/2⊗1/2 → {group.fuse_channels(1, 1, 1)}  (2j values: 1, 3)")

# SU(2) is self-dual
print(f"\nDual of 2j=3: {group.dual(3)}  (self-dual)")
```

## Creating an SU(2) Index

Each `Sector` in an SU(2) index stores a `charge` (= 2j) and `dim` (= number of multiplets). The total number of physical states differs from `dim`:

```python exec="1" session="su2-examples" result="console" idprefix="" source="material-block"
# Index with one singlet (dim=1), two doublets (dim=2), and one triplet (dim=1)
idx = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=1),  # 1 singlet  → 1×1 = 1 state
        Sector(charge=1, dim=2),  # 2 doublets → 2×2 = 4 states
        Sector(charge=2, dim=1),  # 1 triplet  → 1×3 = 3 states
    )
)

print(idx)
print(f"\ndim (multiplets): {idx.dim}")
print(f"num_states (physical states): {idx.num_states}")
```

## Creating a Random SU(2) Tensor

```python exec="1" session="su2-examples" result="console" idprefix="" source="material-block"
# Spin-1/2 index: one doublet sector
spin_half = Index(
    Direction.OUT,
    group,
    sectors=(Sector(charge=1, dim=1),),  # spin-1/2 multiplet
)

# Random 2-index SU(2) tensor (e.g. an operator in spin space)
T = Tensor.random([spin_half, spin_half.flip()], itags=["i", "j"], seed=42)
print(T)
```

The summary shows `'SU2'` as the symmetry signature (rather than `'A'` for Abelian tensors). When a block's reduced weight matrix is a single scalar, a sign indicator `{+}` or `{-}` is appended to flag its sign at a glance, without printing the full matrix.

## Tensor Contraction

All standard operations work identically for SU(2) tensors. Contraction automatically handles the Clebsch–Gordan algebra:

```python exec="1" session="su2-examples" result="console" idprefix="" source="material-block"
# Spin-1 index with one triplet
spin_one = Index(
    Direction.OUT,
    group,
    sectors=(Sector(charge=2, dim=1),),
)

A = Tensor.random([spin_one, spin_one.flip()], itags=["i", "mid"], seed=1)
B = Tensor.random([spin_one, spin_one.flip()], itags=["mid", "j"], seed=2)

result = contract(A, B)
print(f"A:\n{A}\n")
print(f"A @ B:\n{result}")
```

## SVD Decomposition

```python exec="1" session="su2-examples" result="console" idprefix="" source="material-block"
# Multi-sector index with spin-0 and spin-1 sectors
idx = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=2),  # 2 singlets
        Sector(charge=2, dim=1),  # 1 triplet
    )
)

T = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=10)
U, S, Vh = decomp(T, axes=0, mode="SVD")

print(f"T:\n{T}\n")
print(f"U:\n{U}\n")
print(f"S:\n{S}")
```

## Loading a Spin SU(2) Space

```python exec="1" session="su2-examples" result="console" idprefix="" source="material-block"
# Spin-1/2 system with full SU(2) symmetry
Spc, Op = load_space("Spin", "SU2", {"J": 0.5})

print(f"Physical space dimension: {Spc.dim}")
print(f"Physical states: {Spc.num_states}")
print(f"Available operators: {list(Op.keys())}\n")

# The spin operator S is a rank-1 spherical tensor
# It stores the reduced tensor element (Wigner-Eckart theorem)
print(f"S operator (reduced tensor element):\n{Op['S']}")
```

With SU(2) symmetry, there is only one independent spin operator `S` (the spherical tensor), compared to three separate operators (`Sp`, `Sm`, `Sz`) needed for the explicit U(1) case. Nicole uses the Wigner–Eckart theorem to reconstruct full tensor elements on the fly.

## See Also

- API Reference: [SU2Group](../../api/symmetry/su2-group.md)
- [Product Group Examples](product-examples.md): U(1) × SU(2) tensors
- [Load Space Examples](../advanced/load-space-examples.md): Spin and Band SU(2) presets
- [U(1) Examples](u1-examples.md): Abelian comparison
