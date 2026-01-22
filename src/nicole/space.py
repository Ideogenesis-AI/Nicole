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
from .symmetry import U1Group


def load_space(
    stat: str,
    preserv: str,
    option: Optional[Dict[str, Any]] = None
) -> Tuple[Index, Dict[str, Tensor]]:
    """Load physical space and operators for quantum many-body systems.
    
    Parameters
    ----------
    stat : str
        Statistics type: "Spin" for bosonic spin systems, "Ferm" for fermionic systems
    preserv : str
        Symmetry to preserve: "U1" for U(1) charge conservation
    option : dict, optional
        Additional options specific to the system:
        - For "Spin": {"J": float} where J is the total spin (half-integer)
    
    Returns
    -------
    Spc : Index
        Physical space index containing all sectors of the local Hilbert space
    Op : dict[str, Tensor]
        Dictionary of operators as charge-conserving Tensors.
        For spin systems: {"Sp", "Sm", "Sz"}
        - Sz: 2-index tensor (OUT, IN) - charge neutral
        - Sp: 3-index tensor (OUT, IN, auxiliary) - charge neutral with auxiliary index
        - Sm: 3-index tensor (OUT, IN, auxiliary) - charge neutral with auxiliary index
    
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
    ['Sp', 'Sm', 'Sz']
    
    >>> # Create spin-1 system
    >>> Spc, Op = load_space("Spin", "U1", {"J": 1.0})
    >>> Spc.dim  # 3 states: m_z = -1, 0, +1
    3
    """
    if option is None:
        option = {}
    
    if stat == "Spin":
        return _load_spin_space(preserv, option)
    else:
        raise ValueError(f"Unsupported statistics type '{stat}'. Currently only 'Spin' is implemented.")


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
    Op : dict[str, Tensor]
        Spin operators {Sp, Sm, Sz}
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
    
    return Spc, Op
