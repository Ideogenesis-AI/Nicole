# Product Group Examples

Product groups combine multiple independent symmetries that all must be conserved simultaneously. Nicole represents these as tuples of charges, with each component corresponding to one symmetry factor.

**Common combinations:**

- **U(1) × U(1)**: Particle number and spin (e.g., \(N_\uparrow\) and \(N_\downarrow\) separately)
- **U(1) × Z(2)**: Particle number and fermion parity
- **Multiple U(1)s**: Independent conservation laws (e.g., baryon and lepton number)

For a product group, charge conservation requires **all components** to be independently conserved. For example, with U(1) × Z(2), a block with charges \((q_1^{(1)}, q_1^{(2)}), (q_2^{(1)}, q_2^{(2)}), \ldots\) exists only if both \(\sum_i d_i \cdot q_i^{(1)} = 0\) and \(\bigoplus_i q_i^{(2)} = 0\).

```python exec="1" session="product-examples" result=""
from nicole import ProductGroup, U1Group, Z2Group, Index, Sector, Direction, Tensor
```

## Creating a Product Group

```python exec="1" session="product-examples" result="console" idprefix="" source="material-block"
# U(1) × Z(2) for particle number and parity
group = ProductGroup([U1Group(), Z2Group()])

print(f"Group name: {group.name}")  # "U1×Z2"
print(f"Neutral element: {group.neutral}")  # (0, 0)
```

## Composite Charges

```python exec="1" session="product-examples" result="console" idprefix="" source="material-block"
# Charges are tuples: (U1_charge, Z2_charge)
charge1 = (2, 1)   # 2 particles, odd parity
charge2 = (1, 0)   # 1 particle, even parity

# Fusion (component-wise)
fused = group.fuse_unique(charge1, charge2)
print(f"{charge1} ⊕ {charge2} = {fused}")  # (3, 1)
# Because: (2+1, 1⊕0) = (3, 1)
print()

# Dual (component-wise)
dual_charge = group.dual((5, 1))
print(f"Dual of (5, 1): {dual_charge}")  # (-5, 1)
```

## Creating Tensors with Product Group

```python exec="1" session="product-examples" result="console" idprefix="" source="material-block"
# Index with composite charge sectors
idx = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector((0, 0), 2),   # 0 particles, even parity
        Sector((1, 1), 1),   # 1 particle, odd parity
        Sector((2, 0), 1),   # 2 particles, even parity
    )
)

T = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=42)
print(T)
```

## Component-wise Conservation

```python exec="1" session="product-examples" result="console" idprefix="" source="material-block"
# Each component conserves independently
# For (OUT, IN) indices with charges ((n1, p1), (n2, p2)):
# Must have: n1 - n2 = 0 AND p1 - p2 = 0

group_check = ProductGroup([U1Group(), Z2Group()])

# Valid blocks for (OUT, IN) tensor:
valid_keys = [
    ((0, 0), (0, 0)),  # U1: 0-0=0 ✓, Z2: 0⊕0=0 ✓
    ((1, 1), (1, 1)),  # U1: 1-1=0 ✓, Z2: 1⊕1=0 ✓
    ((2, 0), (2, 0)),  # U1: 2-2=0 ✓, Z2: 0⊕0=0 ✓
]

print("Valid blocks (component-wise conservation):")
for key in valid_keys:
    print(f"  {key}")

print("\n# Invalid blocks:")
invalid_keys = [
    ((1, 0), (0, 0)),  # U1: 1-0=1 ✗
    ((0, 1), (0, 0)),  # Z2: 1⊕0=1 ✗
    ((1, 1), (1, 0)),  # Z2: 1⊕0=1 ✗
]
for key in invalid_keys:
    print(f"  {key}")
```

## Three Groups: U(1) × U(1) × Z(2)

```python exec="1" session="product-examples" result="console" idprefix="" source="material-block"
# Three independent conserved quantities
triple_group = ProductGroup([U1Group(), U1Group(), Z2Group()])

print(f"Group: {triple_group.name}")  # "U1×U1×Z2"
print(f"Neutral: {triple_group.neutral}\n")

# Charges are 3-tuples
charge_a = (1, -2, 1)
charge_b = (2, 1, 0)
fused = triple_group.fuse_unique(charge_a, charge_b)
print(f"{charge_a} ⊕ {charge_b} = {fused}")  # (3, -1, 1)
```

## U(1) × SU(2): Non-Abelian Product Group

Combining U(1) with SU(2) gives a non-Abelian product group, suitable for systems where both particle number and full spin-rotation symmetry are conserved (e.g. the Hubbard model).

!!! note "Restriction"
    Nicole supports **at most one `SU2Group`** in a `ProductGroup`, and it **must be the last component**. For example, `ProductGroup([U1Group(), SU2Group()])` is valid, but `ProductGroup([SU2Group(), U1Group()])` is not.

```python exec="1" session="product-examples" result=""
from nicole import SU2Group
```

```python exec="1" session="product-examples" result="console" idprefix="" source="material-block"
# U(1) × SU(2): particle number + full spin rotation
group = ProductGroup([U1Group(), SU2Group()])

print(f"Group: {group.name}")
print(f"Neutral: {group.neutral}")
print(f"Is Abelian: {group.is_abelian}\n")

# Charges are tuples (n, 2j) — particle number and twice-spin
# Dual: U1 negates, SU2 is self-dual
print(f"Dual of (2, 1): {group.dual((2, 1))}")  # (-2, 1)
```

```python exec="1" session="product-examples" result="console" idprefix="" source="material-block"
# Fusion is multi-channel for the SU(2) component
# (1, 1) ⊗ (1, 1): U1 gives 1+1=2; SU2 gives 1⊗1 → (0,2)
channels = group.fuse_channels((1, 1), (1, 1))
print(f"Fusion channels of (1,1) ⊗ (1,1): {channels}")
# → ((2, 0), (2, 2)): charge-2 singlet and charge-2 triplet
```

```python exec="1" session="product-examples" result="console" idprefix="" source="material-block"
# Index for a Band U1×SU2 system: vacuum, singly-occupied (spin-1/2), doubly-occupied
band_idx = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector((0, 0), 1),  # vacuum: 0 particles, singlet
        Sector((1, 1), 1),  # one electron: charge 1, spin-1/2 doublet
        Sector((2, 0), 1),  # two electrons: charge 2, singlet (paired)
    )
)

print(f"dim (multiplets): {band_idx.dim}")
print(f"num_states (physical): {band_idx.num_states}")
print(band_idx)
```

```python exec="1" session="product-examples" result="console" idprefix="" source="material-block"
T = Tensor.random([band_idx, band_idx.flip()], itags=["i", "j"], seed=42)
print(T)
```

## See Also

- API Reference: [ProductGroup](../../api/symmetry/product-group.md)
- API Reference: [U1Group](../../api/symmetry/u1-group.md)
- API Reference: [Z2Group](../../api/symmetry/z2-group.md)
- API Reference: [SU2Group](../../api/symmetry/su2-group.md)
- SU(2) group basics: [SU(2) Examples](su2-examples.md)
