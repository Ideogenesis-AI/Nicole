# load_space

Load physical space and operators for quantum many-body systems.

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
```

**Operators returned:**
- `Sp`: Spin raising operator (S+)
- `Sm`: Spin lowering operator (S-)
- `Sz`: Spin z-component operator
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
```

**Operators returned:**
- `F_up`: Spin-up annihilation operator
- `F_dn`: Spin-down annihilation operator
- `Z`: Jordan-Wigner string operator
- `Sz`: Spin z-component
- `Sp`: Spin raising operator
- `Sm`: Spin lowering operator
- `vac`: Vacuum index

## Symmetry Options

- `"U1"`: U(1) charge conservation
- `"Z2"`: Z2 parity
- `"Z2,U1"`: Z2 parity + U1 spin (Band only)
- `"U1,U1"`: U1 particle number + U1 spin (Band only)

## See Also

- [Index](../core/index-class.md): Physical space structure
- [Tensor](../core/tensor.md): Operator representation
- [U1Group](../symmetry/u1-group.md): U(1) symmetry
- [ProductGroup](../symmetry/product-group.md): Multiple symmetries

## Notes

- For spin systems, the `"J"` option (total spin) is required
- Charges are mapped to integers: for spin-J, charge = 2×m_z
- All operators respect the specified symmetry
- Operators are returned as Tensors with appropriate index structure
- The `vac` entry is an Index representing the trivial vacuum space
