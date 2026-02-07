# Z(2) Symmetry Examples

Z(2) symmetry represents discrete \(\mathbb{Z}_2\) conservation laws such as parity, fermion number modulo 2, or spatial inversion symmetry. In Nicole, Z(2) charges are **binary** (0 or 1) and combine using addition modulo 2.

**Common applications:**

- **Fermion parity**: Charge 0 for even fermion number, charge 1 for odd
- **Spatial inversion**: Charge 0 for even parity states, charge 1 for odd
- **Discrete symmetries**: Any two-valued conserved quantum number

Z(2) tensors enforce parity conservation: a tensor block exists only if the total parity \(\bigoplus_i q_i = 0\) (mod 2), where the sum is over all OUT indices minus all IN indices. This means all tensors must have **even total parity**.

```python exec="1" session="z2-examples" result=""
from nicole import Tensor, Index, Sector, Direction, Z2Group
```

## Basic Z(2) Tensor

```python exec="1" session="z2-examples" result="console" idprefix="" source="material-block"
# Z(2) group for parity
group = Z2Group()

# Create index with even/odd sectors
idx = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=3),  # even parity (3 states)
        Sector(charge=1, dim=2),  # odd parity (2 states)
    )
)

T = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=42)
print(T)
```

## Charge Operations

```python exec="1" session="z2-examples" result="console" idprefix="" source="material-block"
# Z(2) operations
print(f"Group name: {group.name}")
print(f"Neutral element: {group.neutral}\n")

# Fusion (XOR for Z2)
print(f"0 ⊕ 0 = {group.fuse(0, 0)}")  # 0 (even + even = even)
print(f"0 ⊕ 1 = {group.fuse(0, 1)}")  # 1 (even + odd = odd)
print(f"1 ⊕ 0 = {group.fuse(1, 0)}")  # 1 (odd + even = odd)
print(f"1 ⊕ 1 = {group.fuse(1, 1)}\n")  # 0 (odd + odd = even)

# Self-dual
print(f"Dual of 0: {group.dual(0)}")  # 0
print(f"Dual of 1: {group.dual(1)}")  # 1
```

## Fermion Parity

```python exec="1" session="z2-examples" result="console" idprefix="" source="material-block"
# Fermion parity: even = bosonic, odd = fermionic
idx_fermion = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=1),  # 0 or 2 fermions (even)
        Sector(charge=1, dim=2),  # 1 fermion (odd, 2 spin states)
    )
)

# All tensors must have even total parity (parity conservation)
F = Tensor.random([idx_fermion, idx_fermion.flip()], itags=["out", "in"], seed=99)

# Check blocks - all must conserve parity
print("Fermion operator blocks (all have even total parity):")
for key in F.data.keys():
    parity_out, parity_in = key
    total_parity = group.fuse(parity_out, parity_in)
    print(f"  {key}: {parity_out} ⊕ {parity_in} = {total_parity} (conserved)")
```

## Spatial Inversion

```python exec="1" session="z2-examples" result="console" idprefix="" source="material-block"
# States with definite parity under spatial inversion
idx_spatial = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=5),  # symmetric (even parity)
        Sector(charge=1, dim=5),  # antisymmetric (odd parity)
    )
)

# Parity operator: +I for even, -I for odd
import torch
P_data = {
    (0, 0): torch.eye(5),      # +I for even parity states
    (1, 1): -torch.eye(5),     # -I for odd parity states
}
P = Tensor(
    indices=[idx_spatial, idx_spatial.flip()],
    itags=["out", "in"],
    data=P_data,
    label="Parity"
)

print(P)
```

## Combining Parities

```python exec="1" session="z2-examples" result="console" idprefix="" source="material-block"
# Multiple fusion
parities = [1, 1, 1]  # Three odd objects
total_parity = group.fuse(*parities)
print(f"Three odd objects: {parities} → parity {total_parity}\n")

parities2 = [1, 1, 1, 1]  # Four odd objects
total_parity2 = group.fuse(*parities2)
print(f"Four odd objects: {parities2} → parity {total_parity2}")
```

## Validation

```python exec="1" session="z2-examples" result="console" idprefix="" source="material-block"
# Z2 only allows charges 0 or 1
try:
    bad_sector = Sector(charge=2, dim=1)  # Invalid!
    bad_index = Index(Direction.OUT, group, sectors=(bad_sector,))
except ValueError as e:
    print(f"Expected error: {e}")
```

## See Also

- API Reference: [Z2Group](../../api/symmetry/z2-group.md)
- Previous: [U1 Examples](u1-examples.md)
- Next: [Product Groups](product-examples.md)
