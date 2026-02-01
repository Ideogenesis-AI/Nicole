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

## Creating a ProductGroup

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
fused = group.fuse(charge1, charge2)
print(f"{charge1} ⊕ {charge2} = {fused}")  # (3, 1)
# Because: (2+1, 1⊕0) = (3, 1)
print()

# Inverse (component-wise)
inv = group.inverse((5, 1))
print(f"Inverse of (5, 1): {inv}")  # (-5, 1)
```

## Creating Tensors with ProductGroup

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
fused = triple_group.fuse(charge_a, charge_b)
print(f"{charge_a} ⊕ {charge_b} = {fused}")  # (3, -1, 1)
```

## See Also

- API Reference: [ProductGroup](../../api/symmetry/product-group.md)
- API Reference: [U1Group](../../api/symmetry/u1-group.md)
- API Reference: [Z2Group](../../api/symmetry/z2-group.md)
- Previous: [Z2 Examples](z2-examples.md)
