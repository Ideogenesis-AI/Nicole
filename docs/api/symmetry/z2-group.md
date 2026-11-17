# Z2Group

Z(2) symmetry group with binary charges. It has exactly two irreducible representations, labeled 0 and 1, which fuse by *exclusive or*, so the group tracks a parity instead of a count. Like U(1), it is Abelian and one-dimensional, so a Z(2) tensor carries no accompanying intertwiners, but its finite charge set keeps the number of blocks small.

::: nicole.Z2Group
    options:
      show_source: false
      heading_level: 2
      members:
        - name
        - neutral
        - dual
        - fuse_unique
        - equal
        - validate_charge

## Description

`Z2Group` represents the discrete group of a single involution — an operation that returns the identity when applied twice, such as flipping a sign or reflecting an arrow. It has an even representation, on which the involution acts as \(+1\), and an odd representation, on which it acts as \(-1\). Charge 0 labels the even sector and charge 1 the odd sector, and fusing two charges amounts to multiplying those signs, which in charge labels is addition modulo two.

Z(2) is self-dual: the odd representation is its own mirror, so `dual` is the identity. Charge conservation therefore has no signs to keep track of and reduces to a single statement — the charges on a block must contain an even number of odd indices. Representations are one-dimensional, so `irrep_dim` returns 1 and `fuse_unique` is the only fusion method, `fuse_channels` being reserved for non-Abelian groups.

### Charge Operations

```python
from nicole import Z2Group

group = Z2Group()

group.neutral                 # 0
group.fuse_unique(1, 1)       # 0   — two odd sectors make an even one
group.fuse_unique(1, 1, 1)    # 1   — three stay odd
group.dual(1)                 # 1   — self-dual
group.equal(0, 1)             # False
```

## Physical Applications

- **Fermion parity** (even/odd number)
- **Spatial inversion** symmetry
- **Time-reversal** (in some contexts)

## See Also

- [U1Group](u1-group.md): Integer charge symmetry
- [ProductGroup](product-group.md): Multiple symmetries
- [Overview](overview.md): Symmetry system introduction
- [Examples: Z2](../../examples/symmetries/z2-examples.md)

## Notes

- Only charges 0 and 1 are valid. `validate_charge` raises `ValueError` for any other integer and `TypeError` for a non-integer, and `Index` applies it to every sector charge at construction.
- The charge arithmetic itself reads only the low bit, so `dual`, `fuse_unique` and `equal` behave modulo two on any integer handed to them. Validation at the `Index` boundary keeps stray values out in the first place.
- `Z2Group` is a frozen dataclass, hence hashable, and two separately constructed instances compare equal, so indices built in different parts of a program interoperate without sharing a group object.
