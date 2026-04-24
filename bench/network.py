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

from nicole import Direction, Tensor
from nicole import identity, einsum, decomp, trace


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
            
            # Absorb L into left neighbor: (left, right, phys) * L(right, left') → (left, left', phys)
            mps_new[i-1] = einsum('abr,bc->acr', mps_new[i-1], L)
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
            mps_new[i] = U.permute([0, 2, 1])
            mps_new[i].retag(1, mps[i].itags[1])
            
            # Absorb R into right neighbor: R(left', right) * MPS(right, next_right, phys)
            # Result is already in (left, right, phys) form: (left', next_right, phys)
            mps_new[i+1] = einsum('ab,bcr->acr', R, mps_new[i+1])
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

    This is evaluated by building a Gram matrix in right-bond space site by site.
    At site 0 the left boundary is trivial so bra and ket share the same left bond a:

        einsum('abr,acr->bc', conj_mps, mps)

    For subsequent sites the update reads:

        einsum('abr,ac,cdr->bd', conj_mps, gram, mps)

    where the index letters denote:

    - a, b — bra (conj MPS) left and right bonds
    - c, d — ket (MPS) left and right bonds
    - r — physical index (shared between bra and ket)

    ||ψ||² is recovered by tracing gram[b, d] over the right bond.
    
    Examples
    --------
    >>> # Calculate MPS norm
    >>> norm_value = norm(mps)
    
    >>> # Check if MPS is normalized
    >>> assert abs(norm(mps) - 1.0) < 1e-10
    """
    if len(mps) == 0:
        return 0.0
    
    # Site 0: bra and ket share the same trivial left bond 'a'; sum over a and r (physical).
    # mps[0].conj(): (a=bra_left, b=bra_right, r=phys)
    # mps[0]:        (a=ket_left, c=ket_right, r=phys)
    gram = einsum('abr,acr->bc', mps[0].conj(), mps[0])
    # gram: (b=bra_right, c=ket_right)
    
    # Process remaining sites
    for i in range(1, len(mps)):
        # gram:          (a=bra_left, c=ket_left)
        # mps[i].conj(): (a=bra_left, b=bra_right, r=phys)
        # mps[i]:        (c=ket_left, d=ket_right, r=phys)
        gram = einsum('abr,ac,cdr->bd', mps[i].conj(), gram, mps[i])
        # gram: (b=bra_right, d=ket_right)
    
    # Trace the Gram matrix over the right bond to obtain ‖ψ‖²
    norm_squared = trace(gram, axes=(0, 1))
    
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
    
    The contraction proceeds from left to right, building up a transfer tensor E
    with indices (bra_right, mpo_right, ket_right). At each site the update is:

        einsum('aoc,cds,oprs,abr->bpd', E, mps, mpo, conj_mps)

    where the index letters denote:

    - a, b — bra (conj MPS) left and right bonds
    - o, p — MPO left and right bonds
    - c, d — ket (MPS) left and right bonds
    - r — physical bra index (shared between bra axis 2 and MPO axis 2)
    - s — physical ket index (shared between MPO axis 3 and ket axis 2)

    a, o, c are contracted against E; b, p, d become the updated E.
    
    Examples
    --------
    >>> # Calculate expectation value of Hamiltonian
    >>> energy = observe(mps, hamiltonian_mpo)
    """
    if len(mps) == 0 or len(mpo) == 0:
        return 0.0
    
    if len(mps) != len(mpo):
        raise ValueError(f"MPS and MPO must have same length, got {len(mps)} and {len(mpo)}")
    
    # Contract from left to right, site by site.
    # Build transfer tensor E with indices (a, c, e) = (bra_left, mpo_left, ket_left).
    # Note: mps and conj_mps share the same itags on bonds.
    
    # Initialise E as a rank-2 identity on the ket left bond connecting mps[0] and
    # conj(mps[0]), then insert the MPO left bond. For site 0 the left boundary is
    # trivial (dimension 1).
    mps_left_space = mps[0].indices[0]
    E = identity(mps_left_space)
    E.retag([0, 1], [mps[0].itags[0], mps[0].itags[0]])
    # E: (a=bra_left, c=ket_left)
    E.insert_index(1, direction=Direction.OUT, itag=mpo[0].itags[0])
    # E: (a=bra_left, o=mpo_left, c=ket_left)
    
    # Now contract site by site
    for i in range(len(mps)):
        # E: (a=bra_left, o=mpo_left, c=ket_left)
        # mps[i]: (c=ket_left, d=ket_right, s=phys_ket)
        # mpo[i]: (o=mpo_left, p=mpo_right, r=phys_bra, s=phys_ket)
        # mps[i].conj(): (a=bra_left, b=bra_right, r=phys_bra)
        E = einsum('aoc,cds,oprs,abr->bpd', E, mps[i], mpo[i], mps[i].conj())
        # E: (b=bra_right, p=mpo_right, d=ket_right)
    
    # After all sites E has shape (b=bra_right, p=mpo_right, d=ket_right).
    # At the right boundary all indices have dimension 1.
    # Extract the scalar as data × Bridge weight. For Abelian symmetries intw is
    # None (implicit weight 1); for non-Abelian groups the Bridge encodes the
    # CG normalisation that must be included for the correct physical value.
    k, v = next(iter(E.data.items()))
    weight = 1.0 if E.intw is None else E.intw[k].weights[0, 0].item()
    value = v.item() * weight
    
    return value / len(mps)
