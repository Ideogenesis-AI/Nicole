# Copyright (C) 2026 Changkai Zhang.
#
# This file is part of Nicole library.
#
# Nicole is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Nicole is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Nicole. If not, see <https://www.gnu.org/licenses/>.


"""Tensor network utilities for MPS manipulation and canonicalization."""

from typing import Optional

import numpy as np

from nicole import Tensor, contract, conj, decomp, trace
from nicole import Direction, identity, permute


def canonical(
    mps: list[Tensor],
    direction: str = "right",
    trunc: Optional[dict] = None
) -> list[Tensor]:
    """Convert MPS between left-canonical and right-canonical forms.
    
    Parameters
    ----------
    mps : list of Tensor
        Input MPS as a list of tensors. Each tensor should have shape
        (left, right, phys) with itags following the convention
        ['R{i-1:02d}', 'R{i:02d}', 's{i:02d}'] for site i.
    direction : str, optional
        Target canonical form. Either "right" for right-canonical or
        "left" for left-canonical (default: "right").
    trunc : dict, optional
        Truncation parameters passed to decomp. If None, no truncation
        is applied. Common usage: {'nkeep': max_bond_dim}.
    
    Returns
    -------
    list of Tensor
        MPS in the target canonical form. A new list is returned;
        the original MPS is not modified.
    
    Notes
    -----
    Left-canonical MPS: Each tensor A satisfies A†A = I (isometry on left index)
    Right-canonical MPS: Each tensor B satisfies BB† = I (isometry on right index)
    
    The conversion preserves the physical state represented by the MPS.
    
    Examples
    --------
    >>> # Convert left-canonical MPS to right-canonical
    >>> mps_right = canonical(mps_left, direction="right")
    
    >>> # Convert with truncation to control bond dimension
    >>> mps_right = canonical(mps_left, direction="right", trunc={'nkeep': 100})
    """
    if direction not in ["left", "right"]:
        raise ValueError(f"direction must be 'left' or 'right', got '{direction}'")
    
    # Create a copy to avoid modifying the original
    mps_new = [tensor.clone() for tensor in mps]
    
    if direction == "right":
        # Convert left-canonical to right-canonical
        # Sweep from right to left
        for i in range(len(mps_new) - 1, 0, -1):
            # Decompose: split left index from (left, right, phys)
            # LV mode with << flow: L (with S) goes left, V (isometry) stays here
            L, V = decomp(mps_new[i], axes=0, flow='<<', mode='LV', trunc=trunc)
            
            # Isometry stays at current position
            mps_new[i] = V
            mps_new[i].retag(0, mps[i].itags[0])
            
            # Contract L into left neighbor
            # AK[i-1](left, right, phys) * L(right, left')
            mps_new[i-1] = contract(mps_new[i-1], L, axes=(1, 0), perm=(0, 2, 1))
            mps_new[i-1].retag([0, 1], [mps[i-1].itags[0], mps[i-1].itags[1]])
    
    else:  # direction == "left"
        # Convert right-canonical to left-canonical
        # Sweep from left to right
        for i in range(len(mps_new) - 1):
            # Decompose: split right index from (left, right, phys)
            # UR mode with >> flow: U (isometry) stays here, R (with S) goes right
            # After decomp: U has (left, phys, right'), R has (right', right)
            U, R = decomp(mps_new[i], axes=[0, 2], flow='>>', mode='UR', trunc=trunc)
            
            # Permute U to restore (left, right, phys) ordering
            mps_new[i] = permute(U, [0, 2, 1])
            mps_new[i].retag(1, mps[i].itags[1])
            
            # Contract R into right neighbor
            # R(right', right) * MPS[i+1](right, next_right, phys)
            # Result is already in (left, right, phys) form: (right', next_right, phys)
            mps_new[i+1] = contract(R, mps_new[i+1], axes=(1, 0))
            mps_new[i+1].retag([0, 1], [mps[i+1].itags[0], mps[i+1].itags[1]])
    
    return mps_new


def norm(mps: list[Tensor]) -> float:
    """Calculate the norm of an MPS.
    
    Parameters
    ----------
    mps : list of Tensor
        Input MPS as a list of tensors. Each tensor should have shape
        (left, right, phys).
    
    Returns
    -------
    float
        The norm ||ψ|| of the MPS state.
    
    Notes
    -----
    The norm is computed by contracting the MPS with its conjugate:
        ||ψ||² = <ψ|ψ> = Tr(A[0]† A[0] * A[1]† A[1] * ... * A[N-1]† A[N-1])
    
    This is evaluated by sequentially contracting each site's contribution
    A[i]† A[i] along the chain, starting from the left boundary and ending
    with a contraction over the right boundary.
    
    Examples
    --------
    >>> # Calculate MPS norm
    >>> norm_value = norm(mps)
    
    >>> # Check if MPS is normalized
    >>> assert abs(norm(mps) - 1.0) < 1e-10
    """
    if len(mps) == 0:
        return 0.0
    
    # Start with first site: contract A[0] with conj(A[0]) over left and physical indices
    # A[0]: (left, right, phys), conj(A[0]): (left, right, phys)
    # Result: (right_conj, right)
    result = contract(conj(mps[0]), mps[0], axes=([0, 2], [0, 2]))
    
    # Process remaining sites
    for i in range(1, len(mps)):
        # Contract with A[i]
        # result: (right_conj_{i-1}, right_{i-1})
        # A[i]: (left_i, right_i, phys_i) where left_i connects to right_{i-1}
        temp = contract(result, mps[i], axes=(1, 0))
        # temp: (right_conj_{i-1}, right_i, phys_i)
        
        # Contract with conj(A[i])
        # conj(A[i]): (left_i, right_i, phys_i) where left_i connects to right_conj_{i-1}
        result = contract(conj(mps[i]), temp, axes=([0, 2], [0, 2]))
        # result: (right_conj_{i-1}, right_i)
    
    # Contract final bond indices (trace) - returns scalar tensor
    norm_squared = trace(result, axes=(0, 1))
    
    return np.sqrt(norm_squared.item())


def observe(mps: list[Tensor], mpo: list[Tensor]) -> float:
    """Calculate the expectation value <mps|mpo|mps>.
    
    Parameters
    ----------
    mps : list of Tensor
        Input MPS as a list of tensors. Each tensor should have shape
        (left, right, phys).
    mpo : list of Tensor
        Input MPO as a list of tensors. Each tensor should have shape
        (left, right, phys_out, phys_in) with directions (IN, OUT, IN, OUT).
    
    Returns
    -------
    float
        The expectation value <ψ|O|ψ> where |ψ> is the MPS and O is the MPO.
    
    Notes
    -----
    The expectation value is computed by contracting layer by layer:
        <ψ|O|ψ> = conj(MPS) - MPO - MPS
    
    The contraction proceeds from left to right, building up a "transfer tensor"
    that carries the contracted bonds to the next site.
    
    Examples
    --------
    >>> # Calculate expectation value of Hamiltonian
    >>> energy = observe(mps, hamiltonian_mpo)
    """
    if len(mps) == 0 or len(mpo) == 0:
        return 0.0
    
    if len(mps) != len(mpo):
        raise ValueError(f"MPS and MPO must have same length, got {len(mps)} and {len(mpo)}")
    
    # Contract from left to right, site by site
    # Build transfer tensor E with indices (mps_right, mpo_right, conj_mps_right)
    # Note: mps and conj_mps share the same itags on bonds
    
    # Initialize: Create identity for left boundary connecting mps[0] and conj(mps[0])
    # Then insert index for mpo[0]
    
    # Get the space for the left bond of mps[0]
    # For site 0, left bond is trivial (dim 1)
    # Create identity: (left_mps, left_mps)
    mps_left_space = mps[0].indices[0]  # Get space of left index
    E = identity(mps_left_space)
    E.retag([0, 1], [mps[0].itags[0], mps[0].itags[0]])
    # E: (left_mps, left_mps) with matching itags
    
    # Insert index for MPO left bond
    E.insert_index(1, direction=Direction.OUT, itag=mpo[0].itags[0])
    # E: (left_mps, left_mpo, left_mps)
    
    # Now contract site by site
    for i in range(len(mps)):
        # E: (left_mps, left_mpo, left_mps) at site i
        # mps[i]: (left_mps, right_mps, phys_ket)
        # mpo[i]: (left_mpo, right_mpo, phys_bra, phys_ket)
        # conj(mps[i]): (left_mps, right_mps, phys_bra)
        
        # Contract E with mps[i]: left_mps (index 2 of E) with left_mps (index 0 of mps)
        temp = contract(E, mps[i], axes=(2, 0))
        # temp: (left_mps_conj, left_mpo, right_mps, phys_ket)
        
        # Contract with mpo[i]: left_mpo (index 1) with left_mpo, phys_ket (index 3) with phys_ket
        temp = contract(temp, mpo[i], axes=([1, 3], [0, 3]))
        # temp: (left_mps_conj, right_mps, right_mpo, phys_bra)
        
        # Contract with conj(mps[i]): left_mps_conj (index 0) with left_mps, phys_bra (index 3) with phys_bra
        E = contract(temp, conj(mps[i]), axes=([0, 3], [0, 2]), perm=[2, 1, 0])
        # E: (right_mps_conj, right_mpo, right_mps)
        
        # Note: right_mps and right_mps_conj have the same itag
    
    # After all sites, E has shape (right_mps_conj, right_mpo, right_mps)
    # At the right boundary, all indices have dimension 1
    # Extract the scalar from the first (and only) block and divide by number of sites
    
    value = E.block(1).item()  # Extract single element from numpy array
    return value / len(mps)
