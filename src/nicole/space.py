# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Nicole (TN) library.
#
# Nicole (TN) is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Nicole (TN) is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Nicole (TN). If not, see <https://www.gnu.org/licenses/>.


"""Physical space and operator construction for quantum many-body systems."""

from typing import Dict, Optional, Tuple, Any
import numpy as np

from .index import Index, Direction, Sector
from .tensor import Tensor
from .symmetry import U1Group, Z2Group
from .symmetry import ProductGroup


def load_space(
    preset: str,
    preserv: str,
    option: Optional[Dict[str, Any]] = None
) -> Tuple[Index, Dict[str, Tensor]]:
    """Load physical space and operators for quantum many-body systems.
    
    Parameters
    ----------
    preset : str
        System preset: "Spin" for bosonic spin systems, "Ferm" for spinless fermions,
        "Band" for spinful fermions in conduction band
    preserv : str
        Symmetry to preserve:
        - "U1" for U(1) charge conservation
        - "Z2" for Z2 parity
        - "Z2,U1" for Z2 parity + U1 spin (Band only)
        - "U1,U1" for U1 particle number + U1 spin (Band only)
    option : dict, optional
        Additional options specific to the system:
        - For "Spin": {"J": float} where J is the total spin (half-integer)
        - For "Ferm": No options required for spinless fermions
        - For "Band": No options required for spinful fermions
    
    Returns
    -------
    Spc : Index
        Physical space index containing all sectors of the local Hilbert space
    Op : dict[str, Tensor | Index]
        Dictionary of operators and indices.
        For spin systems: {"Sp", "Sm", "Sz", "vac"}
        - Sz: 2-index tensor (OUT, IN) - charge neutral
        - Sp: 3-index tensor (OUT, IN, auxiliary) - charge neutral with auxiliary index
        - Sm: 3-index tensor (OUT, IN, auxiliary) - charge neutral with auxiliary index
        - vac: Index representing trivial vacuum space with charge 0
        For spinless fermion systems: {"F", "Z", "vac"}
        - F: 3-index tensor (IN, OUT, auxiliary) - annihilation operator, charge neutral
        - Z: 2-index tensor (IN, OUT) - Jordan-Wigner string, charge neutral
        - vac: Index representing trivial vacuum space with charge 0
        For spinful fermion (Band) systems: {"F_up", "F_dn", "Z", "Sz", "Sp", "Sm", "vac"}
        - F_up: 3-index tensor (IN, OUT, auxiliary) - spin-up annihilation operator
        - F_dn: 3-index tensor (IN, OUT, auxiliary) - spin-down annihilation operator
        - Z: 2-index tensor (IN, OUT) - Jordan-Wigner string
        - Sz: 2-index tensor (IN, OUT) - spin z-component
        - Sp: 3-index tensor (IN, OUT, auxiliary) - spin raising operator
        - Sm: 3-index tensor (IN, OUT, auxiliary) - spin lowering operator
        - vac: Index representing trivial vacuum space
    
    Raises
    ------
    ValueError
        If stat or preserv are not supported, or if required options are missing
    
    Examples
    --------
    >>> # Create spin-1/2 system with U(1) symmetry
    >>> Spc, Op = load_space("Spin", "U1", {"J": 0.5})
    >>> Spc.dim  # 2 states: m_z = -1/2, +1/2
    2
    >>> list(Op.keys())
    ['Sp', 'Sm', 'Sz', 'vac']
    
    >>> # Create spin-1 system
    >>> Spc, Op = load_space("Spin", "U1", {"J": 1.0})
    >>> Spc.dim  # 3 states: m_z = -1, 0, +1
    3
    
    >>> # Create spinless fermion system with U(1) symmetry
    >>> Spc, Op = load_space("Ferm", "U1")
    >>> Spc.dim  # 2 states: |0⟩, |1⟩
    2
    >>> list(Op.keys())
    ['F', 'Z', 'vac']
    
    >>> # Create spinful fermion system with U(1)xU(1) symmetry
    >>> Spc, Op = load_space("Band", "U1,U1")
    >>> Spc.dim  # 4 states: |0⟩, |↑⟩, |↓⟩, |↑↓⟩
    4
    >>> list(Op.keys())
    ['F_up', 'F_dn', 'Z', 'Sz', 'Sp', 'Sm', 'vac']
    """
    if option is None:
        option = {}
    
    if preset == "Spin":
        return _load_spin_space(preserv, option)
    elif preset == "Ferm":
        return _load_ferm_space(preserv, option)
    elif preset == "Band":
        return _load_band_space(preserv, option)
    else:
        raise ValueError(f"Unsupported system preset '{preset}'. Supported types: 'Spin', 'Ferm', 'Band'.")


def _load_spin_space(preserv: str, option: Dict[str, Any]) -> Tuple[Index, Dict[str, Tensor]]:
    """Load spin space and operators.
    
    Parameters
    ----------
    preserv : str
        Symmetry to preserve
    option : dict
        Options including "J" (total spin)
    
    Returns
    -------
    Spc : Index
        Physical space index for spin
    Op : dict[str, Tensor | Index]
        Spin operators and indices {Sp, Sm, Sz, vac}
    """
    if preserv != "U1":
        raise ValueError(f"Unsupported symmetry '{preserv}' for Spin. Currently only 'U1' is implemented.")
    
    if "J" not in option:
        raise ValueError("Option 'J' (total spin) is required for Spin systems.")
    
    J = option["J"]
    
    # Validate J is a half-integer
    if not isinstance(J, (int, float)):
        raise ValueError(f"J must be a number, got {type(J)}")
    if J < 0:
        raise ValueError(f"J must be non-negative, got {J}")
    if not (2 * float(J)).is_integer():
        raise ValueError(f"J must be a half-integer (0, 0.5, 1, 1.5, ...), got {J}")
    
    # Create physical space index
    # For spin-J system: m_z ranges from -J to J in steps of 1
    # Each m_z value is a separate sector with dimension 1
    # Note: U1Group requires integer charges, so we use 2*m_z as the charge
    # This maps spin-1/2 (m_z = ±0.5) to charges ±1, spin-1 (m_z = -1,0,1) to charges -2,0,2, etc.
    group = U1Group()
    sectors = []
    
    # Generate all m_z values: -J, -J+1, ..., J-1, J
    m_z = -J
    while m_z <= J + 1e-10:  # Small tolerance for float comparison
        # For U1 symmetry, charge is 2*m_z (integer)
        charge = int(round(2 * m_z))
        # Each state has dimension 1
        sectors.append(Sector(charge=charge, dim=1))
        m_z += 1.0
    
    Spc = Index(
        direction=Direction.IN,
        group=group,
        sectors=tuple(sectors)
    )
    
    # Create operators
    Op = {}
    
    # Build S^z operator (diagonal)
    # S^z |m_z⟩ = m_z |m_z⟩
    Sz_data = {}
    for sector in sectors:
        charge = sector.charge
        m_z = charge / 2.0  # Convert back from 2*m_z to m_z
        # Block key: (charge_in, charge_out) = (charge, charge) for diagonal
        key = (charge, charge)
        # Value is m_z (1x1 matrix)
        Sz_data[key] = np.array([[m_z]], dtype=np.float64)
    
    Op["Sz"] = Tensor(
        indices=(Spc, Spc.flip()),
        itags=("_init_", "_init_"),
        data=Sz_data,
        dtype=np.float64
    )
    
    # Build S^+ operator (raising operator)
    # S^+ |m_z⟩ = sqrt(J(J+1) - m_z(m_z+1)) |m_z+1⟩
    # With directions (IN, OUT, OUT): first index is output, second is input
    # Block (q_out, q_in, q_aux) represents ⟨m_z_out| S^+ |m_z_in⟩
    # Charge conservation: -q_out + q_in + q_aux = 0
    # For S^+: input m_z (charge 2*m_z), output m_z+1 (charge 2*m_z+2)
    # So: -(2*m_z+2) + 2*m_z + q_aux = 0 → q_aux = +2
    aux_plus = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=2, dim=1),)
    )
    
    Sp_data = {}
    for i, sector in enumerate(sectors[:-1]):  # Exclude highest m_z
        charge = sector.charge  # input charge (2*m_z)
        charge_next = sectors[i + 1].charge  # output charge (2*m_z+2)
        m_z = charge / 2.0  # input m_z value
        
        # Matrix element: -⟨m_z+1| S^+ |m_z⟩ / sqrt(2)
        # Additional minus sign for spherical tensor component convention
        # Scaled by 1/sqrt(2) so that S^+S^- + S^-S^+ = S_x^2 + S_y^2
        coeff = -np.sqrt(J * (J + 1) - m_z * (m_z + 1)) / np.sqrt(2.0)
        
        # Block key: (charge_out, charge_in, charge_aux)
        key = (charge_next, charge, 2)
        Sp_data[key] = np.array([[[coeff]]], dtype=np.float64)
    
    Op["Sp"] = Tensor(
        indices=(Spc, Spc.flip(), aux_plus),
        itags=("_init_", "_init_", "_aux_"),
        data=Sp_data,
        dtype=np.float64
    )
    
    # Build S^- operator (lowering operator)
    # S^- |m_z⟩ = sqrt(J(J+1) - m_z(m_z-1)) |m_z-1⟩
    # With directions (IN, OUT, OUT): first index is output, second is input
    # Block (q_out, q_in, q_aux) represents ⟨m_z_out| S^- |m_z_in⟩
    # Charge conservation: -q_out + q_in + q_aux = 0
    # For S^-: input m_z (charge 2*m_z), output m_z-1 (charge 2*m_z-2)
    # So: -(2*m_z-2) + 2*m_z + q_aux = 0 → q_aux = -2
    aux_minus = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=-2, dim=1),)
    )
    
    Sm_data = {}
    for i, sector in enumerate(sectors[1:], start=1):  # Exclude lowest m_z
        charge = sector.charge  # input charge (2*m_z)
        charge_prev = sectors[i - 1].charge  # output charge (2*m_z-2)
        m_z = charge / 2.0  # input m_z value
        
        # Matrix element: ⟨m_z-1| S^- |m_z⟩ / sqrt(2)
        # Scaled by 1/sqrt(2) so that S^+S^- + S^-S^+ = S_x^2 + S_y^2
        coeff = np.sqrt(J * (J + 1) - m_z * (m_z - 1)) / np.sqrt(2.0)
        
        # Block key: (charge_out, charge_in, charge_aux)
        key = (charge_prev, charge, -2)
        Sm_data[key] = np.array([[[coeff]]], dtype=np.float64)
    
    Op["Sm"] = Tensor(
        indices=(Spc, Spc.flip(), aux_minus),
        itags=("_init_", "_init_", "_aux_"),
        data=Sm_data,
        dtype=np.float64
    )
    
    # Create vacuum index (trivial space with charge 0)
    vac_index = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=1),)
    )
    Op["vac"] = vac_index
    
    return Spc, Op


def _load_ferm_space(preserv: str, option: Dict[str, Any]) -> Tuple[Index, Dict[str, Tensor]]:
    """Load spinless fermion space and operators.
    
    Parameters
    ----------
    preserv : str
        Symmetry to preserve: "U1" or "Z2"
    option : dict
        Options (none required for spinless fermions)
    
    Returns
    -------
    Spc : Index
        Physical space index for spinless fermion (2 states: empty and occupied)
    Op : dict[str, Tensor | Index]
        Fermionic operators and indices {F, Z, vac}
        - F: annihilation operator (3-index tensor)
        - Z: Jordan-Wigner string operator (2-index tensor)
        - vac: vacuum index (trivial space)
    """
    if preserv == "U1":
        return _load_ferm_u1(option)
    elif preserv == "Z2":
        return _load_ferm_z2(option)
    else:
        raise ValueError(f"Unsupported symmetry '{preserv}' for Ferm. Supported: 'U1', 'Z2'.")


def _load_ferm_u1(option: Dict[str, Any]) -> Tuple[Index, Dict[str, Tensor]]:
    """Load spinless fermion space with U(1) symmetry.
    
    For spinless fermions (half-filling at charge 0):
    - |0⟩: empty state, charge = -1
    - |1⟩: occupied state, charge = 1
    
    Returns F (annihilation) and Z (Jordan-Wigner string) operators.
    """
    group = U1Group()
    
    # Create physical space: two sectors for |0⟩ and |1⟩
    sectors = [
        Sector(charge=-1, dim=1),  # |0⟩: empty state
        Sector(charge=1, dim=1),   # |1⟩: occupied state
    ]
    
    Spc = Index(
        direction=Direction.IN,
        group=group,
        sectors=tuple(sectors)
    )
    
    Op = {}
    
    # Build F operator (annihilation operator)
    # F|1⟩ = |0⟩, F|0⟩ = 0
    # This is a 3-index tensor with directions (IN, OUT, OUT)
    # Charge conservation: (+1)*q₀ + (-1)*q₁ + (-1)*q₂ = 0
    # For ⟨0|F|1⟩: q₀=-1 (out), q₁=1 (in), q₂=?
    # (+1)*(-1) + (-1)*(1) + (-1)*q₂ = 0 → q₂ = -2
    
    aux_F = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=-2, dim=1),)
    )
    
    F_data = {}
    # Matrix element: ⟨0|F|1⟩ = 1
    # Block key: (q_out, q_in, q_aux) = (-1, 1, -2)
    F_data[(-1, 1, -2)] = np.array([[[1.0]]], dtype=np.float64)
    
    Op["F"] = Tensor(
        indices=(Spc, Spc.flip(), aux_F),
        itags=("_init_", "_init_", "_aux_"),
        data=F_data,
        dtype=np.float64
    )
    
    # Build Z operator (Jordan-Wigner string / Z-string)
    # Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    # This is a diagonal 2-index tensor
    Z_data = {}
    Z_data[(-1, -1)] = np.array([[1.0]], dtype=np.float64)   # ⟨0|Z|0⟩ = 1
    Z_data[(1, 1)] = np.array([[-1.0]], dtype=np.float64)    # ⟨1|Z|1⟩ = -1
    
    Op["Z"] = Tensor(
        indices=(Spc, Spc.flip()),
        itags=("_init_", "_init_"),
        data=Z_data,
        dtype=np.float64
    )
    
    # Create vacuum index (trivial space with charge 0)
    vac_index = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=1),)
    )
    Op["vac"] = vac_index
    
    return Spc, Op


def _load_ferm_z2(option: Dict[str, Any]) -> Tuple[Index, Dict[str, Tensor]]:
    """Load spinless fermion space with Z2 symmetry.
    
    For spinless fermions:
    - |0⟩: empty state, parity = 0 (even)
    - |1⟩: occupied state, parity = 1 (odd)
    
    Returns F (annihilation) and Z (Jordan-Wigner string) operators.
    """
    group = Z2Group()
    
    # Create physical space: two sectors for |0⟩ and |1⟩
    sectors = [
        Sector(charge=0, dim=1),  # |0⟩: empty state, even parity
        Sector(charge=1, dim=1),  # |1⟩: occupied state, odd parity
    ]
    
    Spc = Index(
        direction=Direction.IN,
        group=group,
        sectors=tuple(sectors)
    )
    
    Op = {}
    
    # Build F operator (annihilation operator)
    # F|1⟩ = |0⟩, F|0⟩ = 0
    # This is a 3-index tensor (IN, OUT, auxiliary)
    # Charge conservation (mod 2): -q_out + q_in + q_aux = 0 (mod 2)
    # For F: input parity 1 → output parity 0
    # -0 + 1 + q_aux = 0 (mod 2) → q_aux = -1 (mod 2) = 1
    
    aux_F = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=1, dim=1),)
    )
    
    F_data = {}
    # Matrix element: ⟨0|F|1⟩ = 1
    # Block key: (parity_out, parity_in, parity_aux) = (0, 1, 1)
    F_data[(0, 1, 1)] = np.array([[[1.0]]], dtype=np.float64)
    
    Op["F"] = Tensor(
        indices=(Spc, Spc.flip(), aux_F),
        itags=("_init_", "_init_", "_aux_"),
        data=F_data,
        dtype=np.float64
    )
    
    # Build Z operator (Jordan-Wigner string / Z-string)
    # Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    # This is a diagonal 2-index tensor
    Z_data = {}
    Z_data[(0, 0)] = np.array([[1.0]], dtype=np.float64)   # ⟨0|Z|0⟩ = 1
    Z_data[(1, 1)] = np.array([[-1.0]], dtype=np.float64)  # ⟨1|Z|1⟩ = -1
    
    Op["Z"] = Tensor(
        indices=(Spc, Spc.flip()),
        itags=("_init_", "_init_"),
        data=Z_data,
        dtype=np.float64
    )
    
    # Create vacuum index (trivial space with parity 0)
    vac_index = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=1),)
    )
    Op["vac"] = vac_index
    
    return Spc, Op


def _load_band_space(preserv: str, option: Dict[str, Any]) -> Tuple[Index, Dict[str, Tensor]]:
    """Load spinful fermion (Band) space and operators.
    
    Parameters
    ----------
    preserv : str
        Symmetry to preserve: "U1,U1" or "Z2,U1"
    option : dict
        Options (none required for Band systems)
    
    Returns
    -------
    Spc : Index
        Physical space index for spinful fermion (4 states: empty, up, down, doubly occupied)
    Op : dict[str, Tensor | Index]
        Fermionic and spin operators {F_up, F_dn, Z, Sz, Sp, Sm, vac}
    """
    # Normalize: remove spaces for comparison
    preserv_normalized = preserv.replace(" ", "")
    
    if preserv_normalized == "U1,U1":
        return _load_band_u1u1(option)
    elif preserv_normalized == "Z2,U1":
        return _load_band_z2u1(option)
    else:
        raise ValueError(f"Unsupported symmetry '{preserv}' for Band. Supported: 'U1, U1', 'Z2, U1'.")


def _load_band_u1u1(option: Dict[str, Any]) -> Tuple[Index, Dict[str, Tensor]]:
    """Load spinful fermion space with U(1)xU(1) symmetry (particle number, spin).
    
    States (half-filling at charge 0):
    - |0⟩: empty, charge = (-1, 0)
    - |↑⟩: spin-up, charge = (0, 1)  [2*Sz = 1]
    - |↓⟩: spin-down, charge = (0, -1)  [2*Sz = -1]
    - |↑↓⟩: doubly occupied, charge = (1, 0)
    """
    group = ProductGroup([U1Group(), U1Group()])
    
    # Create physical space: four sectors
    # Order: |0⟩, |↓⟩, |↑⟩, |↑↓⟩ (sorted by charge for consistency)
    sectors = [
        Sector(charge=(-1, 0), dim=1),   # |0⟩: empty
        Sector(charge=(0, -1), dim=1),   # |↓⟩: spin-down
        Sector(charge=(0, 1), dim=1),    # |↑⟩: spin-up
        Sector(charge=(1, 0), dim=1),    # |↑↓⟩: doubly occupied
    ]
    
    Spc = Index(
        direction=Direction.IN,
        group=group,
        sectors=tuple(sectors)
    )
    
    Op = {}
    
    # Build F_up operator (annihilates spin-up electron)
    # F_up|↑⟩ = |0⟩, F_up|↑↓⟩ = |↓⟩
    # Directions (IN, OUT, OUT): (+1)*q₀ + (-1)*q₁ + (-1)*q₂ = 0
    # For |↑⟩ → |0⟩: q₀=(-1,0), q₁=(0,1), solve for q₂
    # (+1)*(-1,0) + (-1)*(0,1) + (-1)*q₂ = 0
    # (-1,0) + (0,-1) + (-1)*q₂ = 0 → q₂ = (-1,-1)
    aux_F_up = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=(-1, -1), dim=1),)
    )
    
    F_up_data = {}
    # |↑⟩ → |0⟩: (0,1) → (-1,0)
    F_up_data[((-1, 0), (0, 1), (-1, -1))] = np.array([[[1.0]]], dtype=np.float64)
    # |↑↓⟩ → |↓⟩: (1,0) → (0,-1)
    F_up_data[((0, -1), (1, 0), (-1, -1))] = np.array([[[1.0]]], dtype=np.float64)
    
    Op["F_up"] = Tensor(
        indices=(Spc, Spc.flip(), aux_F_up),
        itags=("_init_", "_init_", "_aux_"),
        data=F_up_data,
        dtype=np.float64
    )
    
    # Build F_dn operator (annihilates spin-down electron)
    # F_dn|↓⟩ = |0⟩, F_dn|↑↓⟩ = -|↑⟩
    # For |↓⟩ → |0⟩: q₀=(-1,0), q₁=(0,-1)
    # (+1)*(-1,0) + (-1)*(0,-1) + (-1)*q₂ = 0
    # (-1,0) + (0,1) + (-1)*q₂ = 0 → q₂ = (-1,1)
    aux_F_dn = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=(-1, 1), dim=1),)
    )
    
    F_dn_data = {}
    # |↓⟩ → |0⟩: (0,-1) → (-1,0)
    F_dn_data[((-1, 0), (0, -1), (-1, 1))] = np.array([[[1.0]]], dtype=np.float64)
    # |↑↓⟩ → |↑⟩: (1,0) → (0,1), with a minus sign from anticommutation
    F_dn_data[((0, 1), (1, 0), (-1, 1))] = np.array([[[-1.0]]], dtype=np.float64)
    
    Op["F_dn"] = Tensor(
        indices=(Spc, Spc.flip(), aux_F_dn),
        itags=("_init_", "_init_", "_aux_"),
        data=F_dn_data,
        dtype=np.float64
    )
    
    # Build Z operator (Jordan-Wigner string)
    # Z|0⟩ = |0⟩, Z|↑⟩ = -|↑⟩, Z|↓⟩ = -|↓⟩, Z|↑↓⟩ = |↑↓⟩
    Z_data = {}
    Z_data[((-1, 0), (-1, 0))] = np.array([[1.0]], dtype=np.float64)
    Z_data[((0, -1), (0, -1))] = np.array([[-1.0]], dtype=np.float64)
    Z_data[((0, 1), (0, 1))] = np.array([[-1.0]], dtype=np.float64)
    Z_data[((1, 0), (1, 0))] = np.array([[1.0]], dtype=np.float64)
    
    Op["Z"] = Tensor(
        indices=(Spc, Spc.flip()),
        itags=("_init_", "_init_"),
        data=Z_data,
        dtype=np.float64
    )
    
    # Build Sz operator (spin z-component)
    # Sz|0⟩ = 0, Sz|↑⟩ = +1/2|↑⟩, Sz|↓⟩ = -1/2|↓⟩, Sz|↑↓⟩ = 0
    Sz_data = {}
    Sz_data[((-1, 0), (-1, 0))] = np.array([[0.0]], dtype=np.float64)
    Sz_data[((0, -1), (0, -1))] = np.array([[-0.5]], dtype=np.float64)
    Sz_data[((0, 1), (0, 1))] = np.array([[0.5]], dtype=np.float64)
    Sz_data[((1, 0), (1, 0))] = np.array([[0.0]], dtype=np.float64)
    
    Op["Sz"] = Tensor(
        indices=(Spc, Spc.flip()),
        itags=("_init_", "_init_"),
        data=Sz_data,
        dtype=np.float64
    )
    Op["Sz"].trim_zero_sectors()
    
    # Build Sp operator (spin raising: |↓⟩ → |↑⟩)
    # (0,1) = (0,-1) + q_aux → q_aux = (0, 2)
    aux_Sp = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=(0, 2), dim=1),)
    )
    
    Sp_data = {}
    # |↓⟩ → |↑⟩ with spherical convention coefficient
    coeff_Sp = -1.0 / np.sqrt(2.0)
    Sp_data[((0, 1), (0, -1), (0, 2))] = np.array([[[coeff_Sp]]], dtype=np.float64)
    
    Op["Sp"] = Tensor(
        indices=(Spc, Spc.flip(), aux_Sp),
        itags=("_init_", "_init_", "_aux_"),
        data=Sp_data,
        dtype=np.float64
    )
    
    # Build Sm operator (spin lowering: |↑⟩ → |↓⟩)
    # (0,-1) = (0,1) + q_aux → q_aux = (0, -2)
    aux_Sm = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=(0, -2), dim=1),)
    )
    
    Sm_data = {}
    # |↑⟩ → |↓⟩ with spherical convention coefficient
    coeff_Sm = +1.0 / np.sqrt(2.0)
    Sm_data[((0, -1), (0, 1), (0, -2))] = np.array([[[coeff_Sm]]], dtype=np.float64)
    
    Op["Sm"] = Tensor(
        indices=(Spc, Spc.flip(), aux_Sm),
        itags=("_init_", "_init_", "_aux_"),
        data=Sm_data,
        dtype=np.float64
    )
    
    # Create vacuum index
    vac_index = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=(0, 0), dim=1),)
    )
    Op["vac"] = vac_index
    
    return Spc, Op


def _load_band_z2u1(option: Dict[str, Any]) -> Tuple[Index, Dict[str, Tensor]]:
    """Load spinful fermion space with Z2xU(1) symmetry (parity, spin).
    
    States:
    - |0⟩: empty, charge = (0, 0)
    - |↑⟩: spin-up, charge = (1, 1)  [odd parity, 2*Sz = 1]
    - |↓⟩: spin-down, charge = (1, -1)  [odd parity, 2*Sz = -1]
    - |↑↓⟩: doubly occupied, charge = (0, 0)  [even parity]
    """
    group = ProductGroup([Z2Group(), U1Group()])
    
    # Create physical space: three sectors
    # Note: (0,0) appears twice: for |0⟩ and |↑↓⟩, so it has dim=2
    # |0⟩ (index 0 in sector (0,0)) and |↑↓⟩ (index 1 in sector (0,0))
    # |↓⟩ (sector (1,-1)) and |↑⟩ (sector (1,1))
    sectors = [
        Sector(charge=(0, 0), dim=2),    # |0⟩ and |↑↓⟩
        Sector(charge=(1, -1), dim=1),   # |↓⟩
        Sector(charge=(1, 1), dim=1),    # |↑⟩
    ]
    
    Spc = Index(
        direction=Direction.IN,
        group=group,
        sectors=tuple(sectors)
    )
    
    Op = {}
    
    # Build F_up operator
    # F_up|↑⟩ = |0⟩, F_up|↑↓⟩ = |↓⟩
    # -(0,0) + (1,1) + q_aux = 0 (mod 2, exact) → q_aux = (-1, -1) = (1, -1) mod 2
    aux_F_up = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=(1, -1), dim=1),)
    )
    
    F_up_data = {}
    # |↑⟩ → |0⟩: from sector (1,1) to sector (0,0) index 0
    # Block key: (charge_out, charge_in, charge_aux)
    # Shape: (dim_out, dim_in, dim_aux) = (2, 1, 1)
    F_up_data[((0, 0), (1, 1), (1, -1))] = np.array([[[1.0]], [[0.0]]], dtype=np.float64)  # Output to |0⟩
    # |↑↓⟩ → |↓⟩: from sector (0,0) index 1 to sector (1,-1)
    # Shape: (dim_out, dim_in, dim_aux) = (1, 2, 1)
    F_up_data[((1, -1), (0, 0), (1, -1))] = np.array([[[0.0], [1.0]]], dtype=np.float64)  # Input from |↑↓⟩
    
    Op["F_up"] = Tensor(
        indices=(Spc, Spc.flip(), aux_F_up),
        itags=("_init_", "_init_", "_aux_"),
        data=F_up_data,
        dtype=np.float64
    )
    
    # Build F_dn operator
    # F_dn|↓⟩ = |0⟩, F_dn|↑↓⟩ = |↑⟩
    # -(0,0) + (1,-1) + q_aux = 0 (mod 2, exact) → q_aux = (1, 1)
    aux_F_dn = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=(1, 1), dim=1),)
    )
    
    F_dn_data = {}
    # |↓⟩ → |0⟩: Shape: (2, 1, 1)
    F_dn_data[((0, 0), (1, -1), (1, 1))] = np.array([[[1.0]], [[0.0]]], dtype=np.float64)
    # |↑↓⟩ → |↑⟩ with minus sign: Shape: (1, 2, 1)
    F_dn_data[((1, 1), (0, 0), (1, 1))] = np.array([[[0.0], [-1.0]]], dtype=np.float64)
    
    Op["F_dn"] = Tensor(
        indices=(Spc, Spc.flip(), aux_F_dn),
        itags=("_init_", "_init_", "_aux_"),
        data=F_dn_data,
        dtype=np.float64
    )
    
    # Build Z operator
    # Z|0⟩ = |0⟩, Z|↑⟩ = -|↑⟩, Z|↓⟩ = -|↓⟩, Z|↑↓⟩ = |↑↓⟩
    Z_data = {}
    # (0,0) sector: diagonal 2x2 with [1, 0; 0, 1] for |0⟩ and |↑↓⟩
    Z_data[((0, 0), (0, 0))] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    Z_data[((1, -1), (1, -1))] = np.array([[-1.0]], dtype=np.float64)
    Z_data[((1, 1), (1, 1))] = np.array([[-1.0]], dtype=np.float64)
    
    Op["Z"] = Tensor(
        indices=(Spc, Spc.flip()),
        itags=("_init_", "_init_"),
        data=Z_data,
        dtype=np.float64
    )
    
    # Build Sz operator
    Sz_data = {}
    # (0,0) sector: both |0⟩ and |↑↓⟩ have Sz = 0
    Sz_data[((0, 0), (0, 0))] = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    Sz_data[((1, -1), (1, -1))] = np.array([[-0.5]], dtype=np.float64)
    Sz_data[((1, 1), (1, 1))] = np.array([[0.5]], dtype=np.float64)
    
    Op["Sz"] = Tensor(
        indices=(Spc, Spc.flip()),
        itags=("_init_", "_init_"),
        data=Sz_data,
        dtype=np.float64
    )
    Op["Sz"].trim_zero_sectors()
    
    # Build Sp operator (|↓⟩ → |↑⟩)
    # Following spherical tensor convention (consistent with Spin preset)
    # For spin-1/2: coefficient = -1/sqrt(2)
    # -(1,1) + (1,-1) + q_aux = 0 → q_aux = (0, 2)
    aux_Sp = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=(0, 2), dim=1),)
    )
    
    Sp_data = {}
    coeff_Sp = -1.0 / np.sqrt(2.0)
    Sp_data[((1, 1), (1, -1), (0, 2))] = np.array([[[coeff_Sp]]], dtype=np.float64)
    
    Op["Sp"] = Tensor(
        indices=(Spc, Spc.flip(), aux_Sp),
        itags=("_init_", "_init_", "_aux_"),
        data=Sp_data,
        dtype=np.float64
    )
    
    # Build Sm operator (|↑⟩ → |↓⟩)
    # Following spherical tensor convention (consistent with Spin preset)
    # For spin-1/2: coefficient = +1/sqrt(2)
    # -(1,-1) + (1,1) + q_aux = 0 → q_aux = (0, -2)
    aux_Sm = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(Sector(charge=(0, -2), dim=1),)
    )
    
    Sm_data = {}
    coeff_Sm = +1.0 / np.sqrt(2.0)
    Sm_data[((1, -1), (1, 1), (0, -2))] = np.array([[[coeff_Sm]]], dtype=np.float64)
    
    Op["Sm"] = Tensor(
        indices=(Spc, Spc.flip(), aux_Sm),
        itags=("_init_", "_init_", "_aux_"),
        data=Sm_data,
        dtype=np.float64
    )
    
    # Create vacuum index
    vac_index = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=(0, 0), dim=1),)
    )
    Op["vac"] = vac_index
    
    return Spc, Op
