# U(1) Symmetry Examples

U(1) symmetry represents continuous conservation laws such as particle number, total magnetization, or electric charge. In Nicole, U(1) charges are **integers** that must sum to zero (neutral) across all tensor indices.

**Common applications:**

- **Particle number conservation**: Charges label the number of particles in each state
- **Spin magnetization**: Charges represent \(2 \times S_z\) (doubled to keep integers)
- **Electric charge**: Charges quantify the total charge in units of elementary charge

U(1) tensors automatically enforce charge conservation: a tensor block connecting states with charges \((q_1, q_2, \ldots)\) exists only if the total charge \(\sum_i d_i \cdot q_i = 0\), where \(d_i\) is the direction sign (\(+1\) for OUT, \(-1\) for IN).

```python exec="1" session="u1-examples" result=""
from nicole import Tensor, Index, Sector, Direction, U1Group
from nicole.blocks import BlockSchema
```

## Basic U(1) Tensor

```python exec="1" session="u1-examples" result="console" idprefix="" source="material-block"
# U(1) group for particle number conservation
group = U1Group()

# Create index with various charges
idx = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=2),   # vacuum-like states
        Sector(charge=1, dim=1),   # one particle
        Sector(charge=2, dim=1),   # two particles
    )
)

T = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=42)
print(T)
```

## Charge Operations

```python exec="1" session="u1-examples" result="console" idprefix="" source="material-block"
# U(1) operations
print(f"Group name: {group.name}")
print(f"Neutral element: {group.neutral}\n")

# Fusion (addition for U1)
charge1, charge2 = 2, 3
fused = group.fuse_unique(charge1, charge2)
print(f"{charge1} + {charge2} = {fused}\n")

# Dual (negation)
dual_charge = group.dual(5)
print(f"Dual of 5: {dual_charge}\n")

# Multiple fusion
result = group.fuse_unique(1, 2, -1, 3)
print(f"1 + 2 + (-1) + 3 = {result}")
```

## Charge Conservation Verification

```python exec="1" session="u1-examples" result="console" idprefix="" source="material-block"
# Only blocks with total charge = 0 exist
T_check = Tensor.random([idx, idx.flip(), idx], itags=["a", "b", "c"], seed=7)

print("Checking charge conservation:")
for key in T_check.data.keys():
    # key = (charge_a, charge_b, charge_c)
    # a is OUT, b is IN, c is OUT
    total = key[0] - key[1] + key[2]  # OUT - IN + OUT
    conserved = (total == 0)
    print(f"Block {key}: total = {total}, conserved = {conserved}")
```

## Particle Number States

```python exec="1" session="u1-examples" result="console" idprefix="" source="material-block"
# System with 0, 1, or 2 particles
idx_fock = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=1),  # |0⟩ (vacuum)
        Sector(charge=1, dim=2),  # |1⟩ (two different single-particle states)
        Sector(charge=2, dim=1),  # |2⟩ (two particles)
    )
)

# State represented as density matrix
psi = Tensor.random([idx_fock, idx_fock.flip()], itags=["bra", "ket"], seed=11)
print(f"State has {len(psi.data)} charge sectors")

# Operator that conserves particle number
H = Tensor.random([idx_fock, idx_fock.flip()], itags=["out", "in"], seed=22)
print(f"Hamiltonian has {len(H.data)} blocks\n")
print(f"H = \n{H}")
```

## See Also

- API Reference: [U1Group](../../api/symmetry/u1-group.md)
- Next: [Z2 Examples](z2-examples.md)
- Next: [Product Groups](product-examples.md)
