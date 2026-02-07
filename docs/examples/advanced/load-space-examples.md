# Load Space

Building physical spaces and operators from scratch requires careful attention to charge assignments, matrix elements, and symmetry conventions. The `load_space` function provides a **high-level interface** to quickly construct commonly used quantum many-body systems with correct symmetry implementation.

**Supported systems:**

- **Spin systems** (`"Spin"`): Arbitrary spin-J system with spin operators following spherical Condon–Shortley phase convention
- **Spinless fermions** (`"Ferm"`): Fermionic creation/annihilation operators and Jordan-Wigner strings with U(1) or Z(2)
- **Spinful fermions** (`"Band"`): Full electronic systems with both charge and spin degrees of freedom, with both fermionic and spin operators

Instead of manually defining sectors, charges, and operator matrix elements, `load_space` generates everything automatically based on the system type and desired symmetry. This is especially valuable for ensuring consistency in operator conventions across different parts of your code.

The function returns:

- **Local space index (`Spc`)**: Local Hilbert space with appropriate symmetry sectors
- **Operator dictionary (`Op`)**: Contains physical operators (e.g., spin operators, creation/annihilation operators)
- **Vacuum index (`vac`)**: A trivial index with neutral charge and dimension 1

```python exec="1" session="load-space" result=""
from nicole import load_space, contract
```

## Spin Systems

Create spin-1/2 and spin-1 systems with U(1) symmetry.

!!! note "Spherical Tensor Convention"
    The spin operators follow the **spherical tensor convention**:
    
    \[
    S_{+1} = -\frac{1}{\sqrt{2}}(S_x + iS_y), \quad S_{-1} = \frac{1}{\sqrt{2}}(S_x - iS_y)
    \]
    
    This convention ensures proper commutation relations: \([S_{+1}, S_{-1}] = -S_z\).

```python exec="1" session="load-space" result="console" idprefix="" source="material-block"
# Spin-1/2 system
Spc_half, Op_half = load_space("Spin", "U1", {"J": 0.5})

print(f"Spin-1/2 space dimension: {Spc_half.dim}")
print(f"Available operators: {list(Op_half.keys())}\n")

# Display Sz operator
print(f"Sz operator:\n{Op_half['Sz']}\n")

# Display Sp operator
print(f"Sp operator:\n{Op_half['Sp']}\n")

# Display Sm operator
print(f"Sm operator:\n{Op_half['Sm']}")
```

```python exec="1" session="load-space" result="console" idprefix="" source="material-block"
# Spin-1 system
Spc_one, Op_one = load_space("Spin", "U1", {"J": 1.0})

print(f"Spin-1 space dimension: {Spc_one.dim}")
print(f"Sz operator:\n{Op_one['Sz']}")
```

## Spinless Fermions

Create spinless fermion systems with U(1) or Z2 symmetry.

```python exec="1" session="load-space" result="console" idprefix="" source="material-block"
# U(1) symmetry (particle number conservation)
Spc_ferm_u1, Op_ferm_u1 = load_space("Ferm", "U1")

print(f"Spinless fermion space (U1) dimension: {Spc_ferm_u1.dim}")
print(f"Available operators: {list(Op_ferm_u1.keys())}\n")

# Display F operator (annihilation)
print(f"F operator:\n{Op_ferm_u1['F']}\n")

# Display Z operator (Jordan-Wigner string)
print(f"Z operator:\n{Op_ferm_u1['Z']}")
```

```python exec="1" session="load-space" result="console" idprefix="" source="material-block"
# Z2 symmetry (parity conservation)
Spc_ferm_z2, Op_ferm_z2 = load_space("Ferm", "Z2")

print(f"Spinless fermion space (Z2) dimension: {Spc_ferm_z2.dim}")
print(f"F operator:\n{Op_ferm_z2['F']}")
```

## Spinful Fermions (Band)

Create spinful fermion systems with product group symmetries.

!!! note "Spin Operators"
    The spin operators (Sz, Sp, Sm) follow the same [spherical tensor convention](#spin-systems) as in spin systems.

```python exec="1" session="load-space" result="console" idprefix="" source="material-block"
# U(1) x U(1) symmetry (particle number, spin)
Spc_band_u1u1, Op_band_u1u1 = load_space("Band", "U1,U1")

print(f"Spinful fermion space (U1xU1) dimension: {Spc_band_u1u1.dim}")
print(f"Available operators: {list(Op_band_u1u1.keys())}\n")

# Display F_up operator
print(f"F_up operator:\n{Op_band_u1u1['F_up']}\n")

# Display F_dn operator
print(f"F_dn operator:\n{Op_band_u1u1['F_dn']}\n")

# Display Sz operator
print(f"Sz operator:\n{Op_band_u1u1['Sz']}\n")

# Display Sp operator
print(f"Sp operator:\n{Op_band_u1u1['Sp']}")
```

```python exec="1" session="load-space" result="console" idprefix="" source="material-block"
# Z2 x U(1) symmetry (parity, spin)
Spc_band_z2u1, Op_band_z2u1 = load_space("Band", "Z2,U1")

print(f"Spinful fermion space (Z2xU1) dimension: {Spc_band_z2u1.dim}")
print(f"F_up operator:\n{Op_band_z2u1['F_up']}")
```

## See Also

- API Reference: [load_space](../../api/utilities/load-space.md)
- Previous: [Build Operators](build-operators.md)
- Next: [Performance Tips](performance.md)
