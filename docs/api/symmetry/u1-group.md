# U1Group

U(1) symmetry group with integer charges. Its irreducible representations are labeled by a single integer, over the conserved quantum number carried by each sector, and they fuse by plain addition. This is the lightest group Nicole ships: every irrep is one-dimensional, so a U(1) tensor needs no accompanying intertwiners and the blocks hold tensor elements directly.

::: nicole.U1Group
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

`U1Group` represents the continuous group of phase rotations. A charge \(q\) labels the representation on which a rotation by angle \(\theta\) acts as the phase \(\exp\{\mathrm{i} q \theta\}\), so the charges live on the integer lattice and composing two of them adds their exponents. Physically the charge counts something conserved — particles, spin along an axis, electric charge — and a tensor block survives only when those counts balance.

Being Abelian, U(1) fuses two charges into exactly one outcome, so `fuse_unique` is the only fusion method; `fuse_channels` exists on non-Abelian groups alone. All representations are one-dimensional, so `irrep_dim` returns 1 for every charge and `Index.dim` and `Index.num_states` agree.

### Charge Operations

```python
from nicole import U1Group

group = U1Group()

group.neutral               # 0
group.fuse_unique(2, 3)     # 5   — charges add
group.fuse_unique(2, -2)    # 0   — back to the neutral charge
group.dual(3)               # -3  — the dual negates
group.equal(3, 3)           # True
```

The dual makes an incoming index the mirror of an outgoing index. A block is admissible when the outgoing charges, together with the duals of the incoming charges, fuse to 0 — for U(1) that is the familiar statement that the charge flowing out equals the charge flowing in.

## Physical Applications

- **Particle number** conservation
- **Magnetization** (\(S^z\)) conservation
- **Electric charge** conservation

## See Also

- [Z2Group](z2-group.md): Binary symmetry
- [ProductGroup](product-group.md): Multiple symmetries
- [Overview](overview.md): Symmetry system introduction
- [Examples: U1](../../examples/symmetries/u1-examples.md)

## Notes

- All integers are valid charges, positive or negative, with no bound on their magnitude. `validate_charge` only rejects non-integers, raising `TypeError`; `Index` applies it to every sector charge at construction.
- `U1Group` is a frozen dataclass, hence hashable, and two separately constructed instances compare equal. Indices built in different parts of a program therefore interoperate without sharing a group object.
