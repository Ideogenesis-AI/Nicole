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


"""Hamiltonian builders for standard quantum lattice models (MPO form)."""

import torch

from nicole import Direction, Tensor
from nicole import identity, conj, permute, oplus, capcup, contract
from nicole.index import Index
from nicole.space import load_space
from nicole.symmetry.delegate import Bridge


def _make_zero_mid(op_idx: Index, zero4: Tensor) -> Tensor:
    """Build a zero tensor whose bond axes carry an operator's charge sectors.

    In an MPO matrix, the off-diagonal zero blocks in an operator row must
    have bond indices that are *structurally compatible* with the non-zero
    operator block in the same row. When the operator has no charge-0 sector
    (e.g. SU(2) spin operator, or fermionic annihilation/creation), the naïve
    ``op * 0`` tensor carries the wrong bond sector set, causing ``oplus`` to
    fail. This helper builds the correct zero tensor for those positions.

    The block key ``(lc, lc, bc, bc)`` (diagonal in both the bond charge and
    the physical charge) satisfies charge conservation for any Abelian or
    non-Abelian symmetry group. For non-Abelian groups the corresponding
    ``Bridge`` intertwiner is computed via ``Bridge.from_block``.

    Parameters
    ----------
    op_idx : Index
        The *op* axis (axis 2) of the operator tensor, e.g. ``S.indices[2]``
        or ``G.indices[2]``. Its charge sectors determine the bond sectors of
        the output tensor.
    zero4 : Tensor
        A zero tensor with shape ``(left, right, bra, ket)`` built from the
        physical identity. Its ``bra`` (axis 2) and ``ket`` (axis 3) indices
        provide the physical charge sectors, and its ``intw`` field indicates
        whether the symmetry group is Abelian (``intw is None``) or not.

    Returns
    -------
    Tensor
        A zero tensor with indices
        ``(op_idx.flip(), op_idx, zero4.indices[2], zero4.indices[3])``.
    """
    _op_sdm  = op_idx.sector_dim_map()
    _bra_sdm = zero4.indices[2].sector_dim_map()
    _ket_sdm = zero4.indices[3].sector_dim_map()
    _zp_idx  = (op_idx.flip(), op_idx, zero4.indices[2], zero4.indices[3])
    _zp_data = {}
    _zp_intw = {} if zero4.intw is not None else None
    for lc, ld in _op_sdm.items():
        for bc, bd in _bra_sdm.items():
            kd = _ket_sdm.get(bc, 0)
            if kd == 0:
                continue
            key = (lc, lc, bc, bc)
            if _zp_intw is None:
                _zp_data[key] = torch.zeros(ld, ld, bd, kd, dtype=torch.float64)
            else:
                bridge = Bridge.from_block(
                    op_idx.group, key, list(_zp_idx[i].direction for i in range(4))
                )
                _zp_data[key] = torch.zeros(ld, ld, bd, kd, bridge.num_components, dtype=torch.float64)
                _zp_intw[key] = bridge
    zero_mid = zero4.clone()
    zero_mid.indices = _zp_idx
    zero_mid.data    = _zp_data
    zero_mid.intw    = _zp_intw
    return zero_mid


def build_heisenberg(
    N: int = 50,
    J: float = 1.0,
    spin: float = 0.5,
    symmetry: str = "U1",
) -> list[Tensor]:
    """Build MPO representation of the Heisenberg Hamiltonian.
    
    Parameters
    ----------
    N : int
        Chain length (default: 50)
    J : float, optional
        Spin-spin coupling constant (default: 1.0)
    spin : float, optional
        Total spin quantum number for each site (default: 0.5 for spin-1/2)
    symmetry : str, optional
        Symmetry to exploit: ``"U1"`` (Sz conservation) or ``"SU2"`` (full
        spin rotation). Default: ``"U1"``.

    Returns
    -------
    list of Tensor
        MPO tensors, each with shape (left, right, phys_out, phys_in)
        with directions (IN, OUT, IN, OUT) and itags ["W{i:02d}", "W{i:02d}", "s{i:02d}", "s{i:02d}"]
        
    Notes
    -----
    The Heisenberg Hamiltonian is:
        H = J * sum_i (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+ + S_i^z S_{i+1}^z)
          = J * sum_i S_i† · S_{i+1}
    
    The MPO uses a bond dimension of 3 with structure:
        First site: [0, S, I]
        Middle sites: [[I, 0, 0], [S†, 0, 0], [0, S, I]]
        Last site: [I, S†, 0]^T
    
    Examples
    --------
    >>> # Build MPO for N=10 spin chain
    >>> mpo = build_heisenberg(N=10, J=1.0, spin=0.5)
    """
    # Load spin operators
    Spc, Op = load_space("Spin", symmetry, {"J": spin})

    # Construct S with (bra, ket, op) axes
    if symmetry == "SU2":
        S = Op["S"]
    else:
        # U1: assemble from Sp, Sm, Sz (Sz needs op index inserted)
        Op["Sz"].insert_index(2, direction=Direction.OUT)
        S = Op["Sp"] + Op["Sm"] + Op["Sz"]
    
    # Conjugate for the other side: Sdag has (bra, ket, op)
    Sdag = permute(conj(S), [1, 0, 2]) * J
    
    # Identity operator: (bra, ket)
    I = identity(Spc)
    
    # Prepare base 4-index tensors (will copy and retag for each site)
    # Identity with both bonds: (left, right, bra, ket)
    I4 = I.clone()
    I4.insert_index(0, direction=Direction.IN, itag="L")
    I4.insert_index(1, direction=Direction.OUT, itag="R")
    
    # Zero with both bonds: (left, right, bra, ket)
    zero4 = (I * 0.0).clone()
    zero4.insert_index(0, direction=Direction.IN, itag="L")
    zero4.insert_index(1, direction=Direction.OUT, itag="R")
    
    # S with left bond: (bra, ket, op) -> (left, bra, ket, op) -> (left, op, bra, ket)
    S4 = S.clone()
    S4.insert_index(0, direction=Direction.IN, itag="L")
    S4 = permute(S4, [0, 3, 1, 2])
    
    # S† with right bond: (bra, ket, op) -> (bra, ket, op, right) -> (op, right, bra, ket)
    S4dag = Sdag.clone()
    S4dag.insert_index(3, direction=Direction.OUT, itag="R")
    S4dag = permute(S4dag, [2, 3, 0, 1])

    zero_mid = _make_zero_mid(S.indices[2], zero4)

    mpo = []
    
    for i in range(N):
        if i == 0:
            # First site: row vector [0, S, I]
            W = zero4.clone()
            W.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            
            S_copy = S4.clone()
            S_copy.retag([0, 2, 3], [f"W{i:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, S_copy, axes=[1])
            
            I_copy = I4.clone()
            I_copy.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, I_copy, axes=[1])
            
            mpo.append(W)
            
        elif i == N - 1:
            # Last site: column vector [I, S†, 0]^T
            W = I4.clone()
            W.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            
            Sdag_copy = S4dag.clone()
            Sdag_copy.retag([1, 2, 3], [f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, Sdag_copy, axes=[0])
            
            zero_copy = zero4.clone()
            zero_copy.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, zero_copy, axes=[0])
            
            mpo.append(W)
            
        else:
            # Middle sites: 3x3 matrix [[I, 0, 0], [S†, 0, 0], [0, S, I]]
            # Row 0: [I, 0, 0]
            row0_col0 = I4.clone()
            row0_col0.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            
            row0_col1 = S4.clone() * 0
            row0_col1.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            
            row0_col2 = zero4.clone()
            row0_col2.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            
            row0 = oplus(row0_col0, row0_col1, axes=[1])
            row0 = oplus(row0, row0_col2, axes=[1])
            
            # Row 1: [S†, 0, 0]
            row1_col0 = S4dag.clone()
            row1_col0.retag([1, 2, 3], [f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            
            row1_col1 = zero_mid.clone()
            row1_col1.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row1_col2 = S4dag.clone() * 0
            row1_col2.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            
            row1 = oplus(row1_col0, row1_col1, axes=[1])
            row1 = oplus(row1, row1_col2, axes=[1])
            
            # Row 2: [0, S, I]
            row2_col0 = zero4.clone()
            row2_col0.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            
            row2_col1 = S4.clone()
            row2_col1.retag([0, 2, 3], [f"W{i:02d}", f"s{i:02d}", f"s{i:02d}"])
            
            row2_col2 = I4.clone()
            row2_col2.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            
            row2 = oplus(row2_col0, row2_col1, axes=[1])
            row2 = oplus(row2, row2_col2, axes=[1])
            
            # Combine rows
            W = oplus(row0, row1, axes=[0])
            W = oplus(W, row2, axes=[0])
            
            mpo.append(W)
    
    return mpo


def build_freefermion(
    N: int = 50,
    t: float = 1.0,
    symmetry: str = "U1",
) -> list[Tensor]:
    """Build MPO representation of the free spinless tight-binding Hamiltonian.

    Parameters
    ----------
    N : int
        Chain length (default: 50).
    t : float, optional
        Nearest-neighbor hopping amplitude (default: 1.0).
    symmetry : str, optional
        Symmetry to exploit: ``"U1"`` (particle number) or ``"Z2"`` (fermion
        parity). Default: ``"U1"``.

    Returns
    -------
    list of Tensor
        MPO tensors, each with shape (left, right, phys_out, phys_in)
        with directions (IN, OUT, IN, OUT) and itags
        ``["W{i:02d}", "W{i:02d}", "s{i:02d}", "s{i:02d}"]``.

    Notes
    -----
    The Hamiltonian is:

        H = -t Σ_i (c†_i c_{i+1} + h.c.)

    The MPO uses bond dimension 3, structurally identical to the Heisenberg
    MPO. Define:

        F    — annihilation operator (op=OUT convention)
        C    — F† in the complementary "left" convention (op=IN after conj)
        Fd   — h.c. of F  (op=IN)
        Cd   — h.c. of C  (op=OUT, same data as F for real operators)

    ``capcup`` flips the op-axis direction of C (IN→OUT) and Cd (OUT→IN)
    without changing numerical contractions. All four operators are then
    normalised to the full physical index ``Spc`` so that ``oplus`` can
    merge the op axis regardless of which individual bra/ket sectors each
    operator occupies. After the flip and normalisation:

        G    = oplus(F, C, axes=2)      (op=OUT, dim-2 within charge-1 sector)
        Gdag = oplus(Fd, Cd, axes=2)*(-t)  (op=IN,  the scaled Hermitian conjugate)

    Using ``oplus`` on the op axis (rather than plain ``+``) keeps the
    annihilation channel (F, position 0) and creation channel (C, position 1)
    in separate op-index slots.  This prevents the MPO bond from generating
    spurious c†c† / cc cross-terms, which would otherwise appear under Z2
    symmetry because both operators share the same Z2 charge (= 1) and would
    collapse to the same dim-1 bond slot with plain addition.

    The MPO matrix is:

        First site:   [0, G, I]
        Middle sites: [[I, 0, 0], [Gdag, 0, 0], [0, G, I]]
        Last site:    [I, Gdag, 0]^T

    When contracted in the charge-neutral sector, Gdag_i G_{i+1} gives
    -t (c†_i c_{i+1} + c_i c†_{i+1}) summed over i.

    Examples
    --------
    >>> mpo = build_freefermion(N=10, t=1.0, symmetry="U1")
    >>> mpo = build_freefermion(N=10, t=1.0, symmetry="Z2")
    """
    # Load spinless fermion operators
    Spc, Op = load_space("Ferm", symmetry)

    # F  : annihilation, directions (bra=IN, ket=OUT, op=OUT)
    # C  : F† in the "left" convention — conj flips all directions,
    #      so C has (IN, OUT, op=IN)
    # Fd : h.c. of F, same data as C for real F but an independent object
    # Cd : h.c. of C, same data as F for real C but an independent object
    F  = Op["F"]
    C  = conj(permute(F, [1, 0, 2]))
    Fd = permute(conj(F), [1, 0, 2])
    Cd = permute(conj(C), [1, 0, 2])

    # capcup(C, 2, Cd, 2) inverts the op-axis direction of both tensors:
    #   C  : op IN  → OUT   (now matches F, so oplus along op is valid)
    #   Cd : op OUT → IN    (now matches Fd, so oplus along op is valid)
    # Numerical contractions are unchanged by this operation.
    capcup(C, 2, Cd, 2)

    # Normalise all four operators to the full physical index so that oplus
    # can merge the op axis (axis 2) regardless of which bra/ket sectors each
    # operator individually occupies.  (Same pattern as build_conductor.)
    for op in (F, C, Fd, Cd):
        op.indices = (Spc, Spc.flip()) + op.indices[2:]

    # G    = oplus(F, C, axes=2)  : op=OUT, dim-2 within the charge-1 sector.
    #   F occupies op-position 0 (annihilation channel).
    #   C occupies op-position 1 (creation channel).
    # Keeping them in separate op-positions prevents the MPO bond from
    # cross-pairing annihilation at site i with creation at site i+1 AND
    # creation at site i with creation at site i+1 (spurious c†c† / cc
    # terms that appear when both channels collapse to the same dim-1 slot,
    # as happens with plain F+C under Z2 symmetry).
    # Gdag = oplus(Fd, Cd, axes=2) * (-t) : op=IN, the scaled Hermitian conjugate.
    G    = oplus(F, C, axes=2)
    Gdag = oplus(Fd, Cd, axes=2) * (-t)

    # Identity operator: (bra, ket)
    I = identity(Spc)

    # ---- Prepare base 4-index tensors (retagged per site in the loop) --------

    # Identity with both bonds: (left, right, bra, ket)
    I4 = I.clone()
    I4.insert_index(0, direction=Direction.IN,  itag="L")
    I4.insert_index(1, direction=Direction.OUT, itag="R")

    # Zero with both bonds: (left, right, bra, ket)
    zero4 = (I * 0.0).clone()
    zero4.insert_index(0, direction=Direction.IN,  itag="L")
    zero4.insert_index(1, direction=Direction.OUT, itag="R")

    zero_mid = _make_zero_mid(G.indices[2], zero4)

    # G with left bond: (bra, ket, op) → (left, bra, ket, op) → (left, op, bra, ket)
    G4 = G.clone()
    G4.insert_index(0, direction=Direction.IN, itag="L")
    G4 = permute(G4, [0, 3, 1, 2])

    # Gdag with right bond: (bra, ket, op) → (bra, ket, op, right) → (op, right, bra, ket)
    G4dag = Gdag.clone()
    G4dag.insert_index(3, direction=Direction.OUT, itag="R")
    G4dag = permute(G4dag, [2, 3, 0, 1])

    mpo = []

    for i in range(N):
        if i == 0:
            # First site: row vector [0, G, I]
            W = zero4.clone()
            W.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            G_copy = G4.clone()
            G_copy.retag([0, 2, 3], [f"W{i:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, G_copy, axes=[1])

            I_copy = I4.clone()
            I_copy.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, I_copy, axes=[1])

            mpo.append(W)

        elif i == N - 1:
            # Last site: column vector [I, Gdag, 0]^T
            W = I4.clone()
            W.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            Gdag_copy = G4dag.clone()
            Gdag_copy.retag([1, 2, 3], [f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, Gdag_copy, axes=[0])

            zero_copy = zero4.clone()
            zero_copy.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, zero_copy, axes=[0])

            mpo.append(W)

        else:
            # Middle sites: 3×3 block matrix [[I, 0, 0], [Gdag, 0, 0], [0, G, I]]

            # Row 0: [I, 0, 0]
            row0_col0 = I4.clone()
            row0_col0.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row0_col1 = G4.clone() * 0
            row0_col1.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row0_col2 = zero4.clone()
            row0_col2.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row0 = oplus(row0_col0, row0_col1, axes=[1])
            row0 = oplus(row0, row0_col2, axes=[1])

            # Row 1: [Gdag, 0, 0]
            row1_col0 = G4dag.clone()
            row1_col0.retag([1, 2, 3], [f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row1_col1 = zero_mid.clone()
            row1_col1.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row1_col2 = G4dag.clone() * 0
            row1_col2.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row1 = oplus(row1_col0, row1_col1, axes=[1])
            row1 = oplus(row1, row1_col2, axes=[1])

            # Row 2: [0, G, I]
            row2_col0 = zero4.clone()
            row2_col0.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row2_col1 = G4.clone()
            row2_col1.retag([0, 2, 3], [f"W{i:02d}", f"s{i:02d}", f"s{i:02d}"])

            row2_col2 = I4.clone()
            row2_col2.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row2 = oplus(row2_col0, row2_col1, axes=[1])
            row2 = oplus(row2, row2_col2, axes=[1])

            # Combine rows
            W = oplus(row0, row1, axes=[0])
            W = oplus(W, row2, axes=[0])

            mpo.append(W)

    return mpo


def build_conductor(
    N: int = 50,
    t: float = 1.0,
    symmetry: str = "U1,SU2",
) -> list[Tensor]:
    """Build MPO representation of the free spinful tight-binding Hamiltonian.

    Parameters
    ----------
    N : int
        Chain length (default: 50).
    t : float, optional
        Nearest-neighbor hopping amplitude (default: 1.0).
    symmetry : str, optional
        Symmetry to exploit. Accepted values (Band preset):

        * ``"U1,U1"``  — spin-up and spin-down particle numbers separately
        * ``"Z2,U1"``  — fermion parity × spin-z particle number
        * ``"U1,SU2"`` — particle number × full spin-rotation (default)
        * ``"Z2,SU2"`` — fermion parity × full spin-rotation

    Returns
    -------
    list of Tensor
        MPO tensors, each with shape (left, right, phys_out, phys_in)
        with directions (IN, OUT, IN, OUT) and itags
        ``["W{i:02d}", "W{i:02d}", "s{i:02d}", "s{i:02d}"]``.

    Notes
    -----
    The Hamiltonian is:

        H = -t Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.)

    The MPO uses bond dimension 3, structurally identical to
    :func:`build_freefermion`. For Abelian symmetries the two spin-flavour
    annihilation operators are merged first:

        F = F_up + F_dn   (Abelian: "U1,U1" or "Z2,U1")
        F = Op["F"]       (non-Abelian: "U1,SU2" or "Z2,SU2")

    The hopping term with the correct Jordan-Wigner (JW) string is:

        G    = oplus(ZF, (ZF)†, axes=2)  — JW-dressed operators at the LEFT site
        Gdag = oplus(Fd, F, axes=2) * (-t)  — bare operators at the RIGHT site

    where ZF = Z × F, Z = (-1)^N is the total-parity operator.
    ``oplus`` along the op axis keeps the annihilator and creator channels
    block-diagonally separate even when they collide in the same op charge
    sector (which happens for Z2-based symmetries, since Z2 is self-dual).
    All four operators are first normalised to the full physical index ``Spc``
    so that ``oplus`` succeeds regardless of which individual bra/ket sectors
    each operator occupies.

    Examples
    --------
    >>> mpo = build_conductor(N=10, t=1.0, symmetry="U1,SU2")
    >>> mpo = build_conductor(N=10, t=1.0, symmetry="Z2,SU2")
    >>> mpo = build_conductor(N=10, t=1.0, symmetry="U1,U1")
    >>> mpo = build_conductor(N=10, t=1.0, symmetry="Z2,U1")
    """
    # Load spinful fermion operators
    Spc, Op = load_space("Band", symmetry)

    # For Abelian symmetries, spin flavours have distinct op charges and can be
    # combined by ordinary tensor addition. For non-Abelian (SU2), a single
    # spin-doublet operator F already encodes both flavours.
    is_abelian = "SU2" not in symmetry
    if is_abelian:
        F = Op["F_up"] + Op["F_dn"]
    else:
        F = Op["F"]

    # The Band preset treats each site as a 4-state system (|0>, |↑>, |↓>, |↑↓>).
    # For nearest-neighbor hopping between such sites, the Jordan-Wigner (JW)
    # string Z_i = (-1)^{N_i} belongs on the LEFT site of each bond:
    #
    #   c†_{i,σ} c_{i+1,σ}  =  (ZF)†_{i,σ} ⊗ F_{i+1,σ}
    #   c_{i,σ} c†_{i+1,σ}  =  (ZF)_{i,σ}  ⊗ F†_{i+1,σ}
    #
    # So the MPO operators are:
    #
    #   G    = ZF + (ZF)†    (JW-dressed annihilator + creator — LEFT site)
    #   Gdag = (Fd + F)*(-t) (bare creator + annihilator    — RIGHT site)
    #
    # This matches iter_diag_band, which accumulates ZF = Z×F from the left
    # block and pairs it with the bare F† at the new (right) site.
    Z    = Op["Z"]
    ZF   = contract(Z, F, axes=(1, 0))  # ZF = Z×F, annihilator with JW (op=OUT)
    C_ZF = conj(permute(ZF, [1, 0, 2]))  # (ZF)†, JW creator (op=IN → flipped below)

    # Gdag uses bare operators (no Z): creator Fd and annihilator F_copy
    Fd     = permute(conj(F), [1, 0, 2])  # bare creator  (op=IN)
    F_copy = F.clone()                    # bare annihilator (op=OUT → flipped below)

    # Normalise all four operators to the full physical index so that oplus can
    # merge the op axis (axis 2) regardless of which bra/ket sectors each
    # operator individually occupies.
    for op in (ZF, C_ZF, Fd, F_copy):
        op.indices = (Spc, Spc.flip()) + op.indices[2:]

    # capcup: C_ZF IN  → OUT  (to match ZF's op=OUT for G    = oplus(ZF, C_ZF))
    #         F_copy OUT → IN  (to match Fd's op=IN  for Gdag = oplus(Fd, F_copy))
    # This is needed to ensure that bond direction inversion works correctly.
    # CANNOT BE REPLACED BY TWO INDIVIDUAL INVERSIONS!
    capcup(C_ZF, 2, F_copy, 2)

    # oplus along the op axis keeps the annihilator and creator channels
    # block-diagonally separate even when they share the same op charge sector
    # (which happens for Z2-based symmetries).
    G    = oplus(ZF, C_ZF, axes=2)
    Gdag = oplus(Fd, F_copy, axes=2) * (-t)

    # Identity operator: (bra, ket)
    I = identity(Spc)

    # ---- Prepare base 4-index tensors (retagged per site in the loop) --------

    I4 = I.clone()
    I4.insert_index(0, direction=Direction.IN,  itag="L")
    I4.insert_index(1, direction=Direction.OUT, itag="R")

    zero4 = (I * 0.0).clone()
    zero4.insert_index(0, direction=Direction.IN,  itag="L")
    zero4.insert_index(1, direction=Direction.OUT, itag="R")

    zero_mid = _make_zero_mid(G.indices[2], zero4)

    # G with left bond:   (bra, ket, op) → (left, bra, ket, op) → (left, op, bra, ket)
    G4 = G.clone()
    G4.insert_index(0, direction=Direction.IN, itag="L")
    G4 = permute(G4, [0, 3, 1, 2])

    # Gdag with right bond: (bra, ket, op) → (bra, ket, op, right) → (op, right, bra, ket)
    G4dag = Gdag.clone()
    G4dag.insert_index(3, direction=Direction.OUT, itag="R")
    G4dag = permute(G4dag, [2, 3, 0, 1])

    mpo = []

    for i in range(N):
        if i == 0:
            # First site: row vector [0, G, I]
            W = zero4.clone()
            W.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            G_copy = G4.clone()
            G_copy.retag([0, 2, 3], [f"W{i:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, G_copy, axes=[1])

            I_copy = I4.clone()
            I_copy.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, I_copy, axes=[1])

            mpo.append(W)

        elif i == N - 1:
            # Last site: column vector [I, Gdag, 0]^T
            W = I4.clone()
            W.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            Gdag_copy = G4dag.clone()
            Gdag_copy.retag([1, 2, 3], [f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, Gdag_copy, axes=[0])

            zero_copy = zero4.clone()
            zero_copy.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])
            W = oplus(W, zero_copy, axes=[0])

            mpo.append(W)

        else:
            # Middle sites: 3×3 block matrix [[I, 0, 0], [Gdag, 0, 0], [0, G, I]]

            # Row 0: [I, 0, 0]
            row0_col0 = I4.clone()
            row0_col0.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row0_col1 = G4.clone() * 0
            row0_col1.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row0_col2 = zero4.clone()
            row0_col2.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row0 = oplus(row0_col0, row0_col1, axes=[1])
            row0 = oplus(row0, row0_col2, axes=[1])

            # Row 1: [Gdag, 0, 0]
            row1_col0 = G4dag.clone()
            row1_col0.retag([1, 2, 3], [f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row1_col1 = zero_mid.clone()
            row1_col1.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row1_col2 = G4dag.clone() * 0
            row1_col2.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row1 = oplus(row1_col0, row1_col1, axes=[1])
            row1 = oplus(row1, row1_col2, axes=[1])

            # Row 2: [0, G, I]
            row2_col0 = zero4.clone()
            row2_col0.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row2_col1 = G4.clone()
            row2_col1.retag([0, 2, 3], [f"W{i:02d}", f"s{i:02d}", f"s{i:02d}"])

            row2_col2 = I4.clone()
            row2_col2.retag([0, 1, 2, 3], [f"W{i:02d}", f"W{i+1:02d}", f"s{i:02d}", f"s{i:02d}"])

            row2 = oplus(row2_col0, row2_col1, axes=[1])
            row2 = oplus(row2, row2_col2, axes=[1])

            # Combine rows
            W = oplus(row0, row1, axes=[0])
            W = oplus(W, row2, axes=[0])

            mpo.append(W)

    return mpo
