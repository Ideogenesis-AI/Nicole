# Tensor Networks (BUG)

The `nicole.bug` subpackage adds a tensor-train (MPS/MPO) layer and a two-site
**Basis-Update & Galerkin (BUG)** time integrator built on Nicole tensors. It
keeps explicit itag/direction conventions so that symmetry structure is
preserved through every contraction, and it exploits block sparsity for
U(1)-symmetric states such as Heisenberg domain walls.

The subpackage is additive and imported explicitly:

```python
import nicole.bug as bug
```

## States and Operators

| Symbol | Description |
|--------|-------------|
| [TensorTrain](tensor-train.md) | Matrix-product state (MPS) of Nicole cores |
| [TensorTrainOperator](tensor-train.md) | Matrix-product operator (MPO) |
| [product_tt](tensor-train.md) | Build a product-state MPS from a spin string |
| [random_tt](tensor-train.md) | Build a random MPS at a target bond dimension |
| [siteinds](tensor-train.md) | Construct site indices (dense or U(1)) |
| [mpo_from_opsum](tensor-train.md) | Assemble an MPO from an `OpSum` |

## Time Evolution

| Symbol | Description |
|--------|-------------|
| [bug_two_site](bug.md) | One Lie/Strang odd–even BUG step |
| [BUGInfo](bug.md) | Per-step bond-growth and truncation record |
| [bug_xx_bond_gates](bug.md) | Nearest-neighbour XX bond terms |
| [bug_heisenberg_bond_gates](bug.md) | Nearest-neighbour Heisenberg bond terms |
| [bug_xx_parity_mpos](bug.md) | Odd/even/full XX parity MPOs |

## Quick Example

```python
import nicole.bug as bug

# Heisenberg domain wall on 8 U(1) sites.
psi = bug.product_tt("uuuudddd", symmetry="u1")
gates = bug.bug_heisenberg_bond_gates(bug.siteinds(8, symmetry="u1"))

for _ in range(5):
    info = bug.bug_two_site(psi, gates, 0.1, order="strang", maxdim=32, augment=True)

print("max bond dimension:", max(info.bond_dims_after))
```
