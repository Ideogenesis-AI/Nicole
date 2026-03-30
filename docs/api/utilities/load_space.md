# load_space

Load preset physical space and operators for quantum many-body systems.

::: nicole.load_space
    options:
      show_source: false
      heading_level: 2

## Description

Provides pre-configured physical spaces and operator sets for common quantum many-body systems, including spin systems, spinless fermions, and spinful fermions. This simplifies the creation of tensor network models for quantum physics applications.

Returns both the physical space index (defining the local Hilbert space) and a dictionary of operators commonly used in that system.

## Supported Systems

### Spin Systems (`preset="Spin"`)

Bosonic spin systems with arbitrary spin quantum number.

```python
from nicole import load_space

# Spin-1/2 system with U(1) symmetry (S_z conservation)
Spc, Op = load_space("Spin", "U1", {"J": 0.5})
print(Spc.dim)  # 2 states: |↓⟩, |↑⟩
print(list(Op.keys()))  # ['Sp', 'Sm', 'Sz', 'vac']

# Spin-1 system
Spc, Op = load_space("Spin", "U1", {"J": 1.0})
print(Spc.dim)  # 3 states: |-1⟩, |0⟩, |+1⟩

# Spin-1/2 with full SU(2) symmetry (spin-rotation invariance)
Spc, Op = load_space("Spin", "SU2", {"J": 0.5})
print(Spc.dim)        # 1 multiplet (one doublet sector)
print(Spc.num_states) # 2 physical states
print(list(Op.keys()))  # ['S', 'vac']
```

**Operators for `"U1"` symmetry:**
- `Sp`: Spin raising operator (S+)
- `Sm`: Spin lowering operator (S-)
- `Sz`: Spin z-component operator
- `vac`: Vacuum index (trivial space)

**Operators for `"SU2"` symmetry:**
- `S`: Rank-1 spherical tensor (stores the reduced tensor element via the Wigner–Eckart theorem; encodes all three components Sz, S+, S− in one object)
- `vac`: Vacuum index (trivial space)

### Spinless Fermion Systems (`preset="Ferm"`)

Fermions without spin degrees of freedom.

```python
# Spinless fermions with U(1) symmetry (particle number conservation)
Spc, Op = load_space("Ferm", "U1")
print(Spc.dim)  # 2 states: |0⟩, |1⟩
print(list(Op.keys()))  # ['F', 'Z', 'vac']
```

**Operators returned:**
- `F`: Annihilation operator (f)
- `Z`: Jordan-Wigner string operator
- `vac`: Vacuum index

### Spinful Fermion Systems (`preset="Band"`)

Fermions with spin-up and spin-down degrees of freedom.

```python
# Spinful fermions with U(1)×U(1) symmetry (N_up and N_down conservation)
Spc, Op = load_space("Band", "U1,U1")
print(Spc.dim)  # 4 states: |0⟩, |↑⟩, |↓⟩, |↑↓⟩
print(list(Op.keys()))  # ['F_up', 'F_dn', 'Z', 'Sz', 'Sp', 'Sm', 'vac']

# With Z2×U(1) symmetry (parity and total spin conservation)
Spc, Op = load_space("Band", "Z2,U1")

# With U(1)×SU(2) symmetry (particle number + full spin rotation)
Spc, Op = load_space("Band", "U1,SU2")
print(list(Op.keys()))  # ['F', 'Z', 'S', 'vac']

# With Z2×SU(2) symmetry (parity + full spin rotation)
Spc, Op = load_space("Band", "Z2,SU2")
```

**Operators for `"U1,U1"` and `"Z2,U1"` symmetry:**
- `F_up`: Spin-up annihilation operator
- `F_dn`: Spin-down annihilation operator
- `Z`: Jordan-Wigner string operator
- `Sz`: Spin z-component
- `Sp`: Spin raising operator
- `Sm`: Spin lowering operator
- `vac`: Vacuum index

**Operators for `"U1,SU2"` and `"Z2,SU2"` symmetry:**
- `F`: Rank-1/2 fermionic annihilation operator (spherical tensor; reduced tensor element `⟨j'‖F‖j⟩ = √2` for all allowed transitions)
- `Z`: Jordan-Wigner string operator (scalar, rank-0)
- `S`: Rank-1 spin tensor (reduced tensor element; encodes full spin-rotation algebra)
- `vac`: Vacuum index

!!! note "State Convention"
    The Abelian (`"U1,U1"` / `"Z2,U1"`) and SU(2) (`"U1,SU2"` / `"Z2,SU2"`) presets use different sign conventions:

    | State | Abelian | SU(2) |
    |-------|---------|-------|
    | \(\vert\!\uparrow\rangle\) | \(c^\dagger_\uparrow \vert 0\rangle\) | \(c^\dagger_\uparrow \vert 0\rangle\) |
    | \(\vert\!\downarrow\rangle\) | \(c^\dagger_\downarrow \vert 0\rangle\) | \(-c^\dagger_\downarrow \vert 0\rangle\) |
    | \(\vert\!\uparrow\downarrow\rangle\) | \(c^\dagger_\uparrow c^\dagger_\downarrow \vert 0\rangle\) | \(-c^\dagger_\uparrow c^\dagger_\downarrow \vert 0\rangle\) |

    The minus signs in the SU(2) case follow the phase convention of the standard CG basis used by Yuzuha. This difference does not affect physical computations — all expectation values and correlators remain identical.

## Symmetry Options

| `preserv` | Applicable presets | Description |
|-----------|-------------------|-------------|
| `"U1"` | Spin, Ferm | U(1) charge/magnetization conservation |
| `"Z2"` | Ferm | Z2 parity conservation |
| `"SU2"` | Spin | Full SU(2) spin-rotation symmetry |
| `"U1,U1"` | Band | U(1) particle number + U(1) spin |
| `"Z2,U1"` | Band | Z2 parity + U(1) spin |
| `"U1,SU2"` | Band | U(1) particle number + full SU(2) spin |
| `"Z2,SU2"` | Band | Z2 parity + full SU(2) spin |

## See Also

- [Index](../core/index-class.md): Physical space structure
- [Tensor](../core/tensor.md): Operator representation
- [U1Group](../symmetry/u1-group.md): U(1) symmetry
- [SU2Group](../symmetry/su2-group.md): SU(2) symmetry
- [ProductGroup](../symmetry/product-group.md): Multiple symmetries
- [Examples: Load Space](../../examples/advanced/load-space-examples.md)

## Notes

- For spin systems, the `"J"` option (total spin) is required for all symmetry choices
- For U(1) spin systems, charges are `2×m_z` (doubled to keep integers)
- For SU(2) systems, charges are `2j` integers; `Spc.num_states` gives the total physical Hilbert space dimension
- SU(2) operators store **reduced tensor elements** (Wigner–Eckart theorem); some are 3rd order tensors with an auxiliary index carrying the spin-channel label
- The `vac` entry is an `Index` representing the trivial vacuum space (neutral charge, dim=1)
- All operators are returned as `Tensor` objects except `vac`, which is an `Index`
