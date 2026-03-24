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


"""Hamiltonian builders for standard quantum spin models (MPO form)."""

from nicole import Direction, Tensor
from nicole import identity, conj, permute, oplus
from nicole.space import load_space


def build_heisenberg(
    N: int = 50,
    J: float = 1.0,
    spin: float = 0.5
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
    Spc, Op = load_space("Spin", "U1", {"J": spin})
    
    # Construct S = S+ + S- + Sz (total spin operator)
    # Sp and Sm have (bra, ket, op), Sz needs op index inserted
    Op["Sz"].insert_index(2, direction=Direction.OUT)
    S = Op["Sp"] + Op["Sm"] + Op["Sz"]  # Now all have (bra, ket, op)
    
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
            
            row1_col1 = S4dag.clone() * 0 + S4.clone() * 0
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
