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


"""Iterative diagonalization for spin chain (Heisenberg model).

This module implements an iterative diagonalization algorithm to compute the ground
state energy and matrix product state (MPS) representation of a quantum spin chain
with Heisenberg interactions. The algorithm builds up the chain site by site,
diagonalizing the Hamiltonian at each step and truncating to keep only the
lowest-energy states.

The Heisenberg Hamiltonian is:
    H = J * sum_i (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+ + S_i^z S_{i+1}^z)

Key features:
- Supports arbitrary spin quantum numbers (spin-1/2, spin-1, etc.)
- Uses U(1) symmetry to reduce computational cost
- Truncates to a maximum bond dimension to control memory usage
- Computes ground state energy per site at each iteration
- Returns the full MPS representation of the ground state

For spin-1/2 chains, the exact ground state energy per site for infinite N is
E_exact = 1/4 - ln(2) ≈ -0.443147, which can be used to verify convergence.
"""

import time
from typing import Tuple

import numpy as np

from nicole import Direction, Tensor, load_space
from nicole import contract, identity, isometry, conj, permute, transpose, diag
from nicole.decomp import eig



def disptime(msg):
    """Display message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def iter_diag_spin(
    N: int = 50,
    Nkeep: int = 300,
    J: float = 1.0,
    spin: float = 0.5,
    symmetry: str = "U1",
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Run iterative diagonalization for spin chain (Heisenberg model).
    
    Parameters
    ----------
    N : int, optional
        Maximum chain length (default: 50)
    Nkeep : int, optional
        Maximum number of states to keep after truncation (default: 300).
        For symmetry="U1" this counts individual states; for symmetry="SU2"
        this counts SU(2) multiplets (each contributes 2*spin+1 physical states).
    J : float, optional
        Spin-spin coupling constant (default: 1.0)
    spin : float, optional
        Total spin quantum number for each site (default: 0.5 for spin-1/2)
        Must be a half-integer: 0.5, 1.0, 1.5, 2.0, etc.
    symmetry : str, optional
        Symmetry sector to exploit (default: "U1").
        - "U1": conserve total S^z (block-diagonal in m_z)
        - "SU2": exploit full SU(2) spin-rotation symmetry (block-diagonal in
          total spin J). The Hamiltonian is stored as reduced matrix elements
          (Wigner-Eckart theorem); each block covers an entire multiplet.
          Convergence is typically faster than "U1" for the same Nkeep.
    verbose : bool, optional
        Print progress messages (default: True)
    
    Returns
    -------
    Eg : np.ndarray
        Ground state energies at each iteration (length N)
    Egs : np.ndarray
        Ground state energy per site at each iteration (length N)
    mps : list of Tensor
        List of all isometry tensors AK generated at each iteration (length N)
    
    Notes
    -----
    The Heisenberg Hamiltonian is:
        H = J * sum_i (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+ + S_i^z S_{i+1}^z)
    
    For spin-1/2 chains, the exact ground state energy per site for infinite N is:
        E_exact = 1/4 - ln(2) ≈ -0.443147
    This value is printed during execution when verbose=True.
    
    Examples
    --------
    >>> # Run with default parameters (U(1) symmetry)
    >>> Eg, Egs, mps = iter_diag_spin()
    
    >>> # Longer chain with more states kept
    >>> Eg, Egs, mps = iter_diag_spin(N=100, Nkeep=500)
    
    >>> # Spin-1 chain
    >>> Eg, Egs, mps = iter_diag_spin(spin=1.0)
    
    >>> # Use full SU(2) symmetry for faster convergence
    >>> Eg, Egs, mps = iter_diag_spin(symmetry="SU2")
    """
    
    if symmetry not in ("U1", "SU2"):
        raise ValueError(f"Unsupported symmetry '{symmetry}'. Use 'U1' or 'SU2'.")

    tol = Nkeep * 100 * np.finfo(float).eps  # numerical tolerance for degeneracy
    
    # Get local spin space and operators
    Spc, Op = load_space("Spin", symmetry, {"J": spin})
    if symmetry == "U1":
        Op["Sz"].insert_index(2, direction=Direction.OUT)
        S = Op["Sp"] + Op["Sm"] + Op["Sz"]  # Sum of all spin operators
    else:  # SU2
        S = Op["S"]  # Single rank-1 spherical tensor
    I = identity(Spc)  # Identity operator
    
    # Set itags for spin operators
    S.retag(["s00", "s00", "op"])
    
    # Initialize
    H0 = I * 1e-30  # Hamiltonian for only the 1st site (zero operator)
    H0.retag(["s00", "s00"])
    
    # A0: isometry from vacuum ⊗ physical space
    # Output: (vacuum, physical, fused) → permute to (vacuum, fused, physical)
    A0 = permute(isometry(Op["vac"], Spc), [0, 2, 1])
    A0.retag(["L00", "R00", "s00"])
    
    # Lowest energies at each iteration
    Eg = np.zeros(N)
    
    # Track the bond index for subsequent iterations
    bond_index = None
    
    # Initialize Sprev (will be set in first iteration)
    Sprev = None
    
    # Store all AK (isometry) tensors
    mps = []
    
    for itN in range(1, N + 1):
        # Create spin operator for the current site with proper itags
        Snow = S.clone()
        Snow.retag([f"s{itN-1:02d}", f"s{itN-1:02d}", "op"])
        
        if itN == 1:
            # First iteration: sandwich H0 with A0
            # A0: (left, right, phys), H0: (bra, ket)
            Anow = A0
            Hnow = contract(H0, A0, axes=(1, 2))
            Hnow = contract(conj(Anow), Hnow, axes=([0, 2], [1, 0]))
            
        else:
            # Add new site: create isometry (left, phys, right) → permute to (left, right, phys)
            Anow = permute(isometry(bond_index.flip(), Spc), [0, 2, 1])
            Anow.retag([f"R{itN-2:02d}", f"R{itN-1:02d}", f"s{itN-1:02d}"])
            
            # Update Hamiltonian: sandwich Hprev with Anow
            # Anow: (left, right, phys), Hprev: (bra, ket)
            Hnow = contract(Hprev, Anow, axes=(1, 0))
            Hnow = contract(conj(Anow), Hnow, axes=([0, 2], [0, 2]))
            
            # Spin-spin interaction: Sprev-Snow interaction sandwiched by Anow
            # Snow: (bra, ket, op) → permute and conjugate
            Sn = conj(permute(Snow, [2, 1, 0]))
            
            # Contract Sn with Anow, then with Sprev, then sandwich with conj(Anow)
            Sn_Anow = contract(Sn, Anow, axes=(2, 2))
            Sprev_Sn_Anow = contract(Sprev, Sn_Anow, axes=([1, 2], [2, 0]))
            HSS = contract(conj(Anow), Sprev_Sn_Anow, axes=([0, 2], [0, 1]))
            HSS = HSS * J
            
            Hnow = Hnow + HSS
        
        # Symmetrize and diagonalize
        Hnow_sym = (Hnow + transpose(conj(Hnow))) * 0.5
        
        # Diagonalize
        if itN == 1:
            V, D = eig(Hnow_sym, is_hermitian=True)
        elif itN == N:
            # Last site: keep only the ground state
            V, D = eig(Hnow_sym, trunc={"nkeep": 1}, is_hermitian=True)
        else:
            V, D = eig(Hnow_sym, trunc={"nkeep": Nkeep}, is_hermitian=True)
        
        # Set itags for eigenvectors
        V.retag([f"R{itN-1:02d}", f"R{itN-1:02d}"])
        
        # Get minimum eigenvalue
        all_eigvals = np.concatenate([eigvals for eigvals in D.values()])
        Eg[itN - 1] = np.min(all_eigvals)
        
        # Contract Anow with V to get AK
        # Anow: (left, right, phys), V: (right_old, right_new) 
        #   → result: (left, phys, right_new) → permute to (left, right_new, phys)
        AK = contract(Anow, V, axes=(1, 0), perm=[0, 2, 1])
        
        # Store AK for this iteration
        mps.append(AK.clone())
        
        # Create diagonal Hprev from truncated eigenvalues
        bond_index = V.indices[1]  # Update bond index for next iteration
        Hprev = diag(D, bond_index, itags=(f"R{itN-1:02d}", f"R{itN-1:02d}"))
        
        # Spin operator at the current site: sandwich Snow with AK
        # AK: (left, right, phys), Snow: (bra, ket, op)
        # Result: (left_conj, op, right) → permute to (left_conj, right, op)
        Snow_AK = contract(Snow, AK, axes=(1, 2))
        Sprev = contract(conj(AK), Snow_AK, axes=([0, 2], [2, 0]), perm=[0, 2, 1])
        
        # Display progress
        if verbose:
            NK = AK.indices[0].dim  # Size of truncated space
            Hnow_dim = Hnow.indices[1].dim  # Size of Hilbert space before truncation
            disptime(f"#{itN:02d}/{N:02d} : NK={NK}/{Hnow_dim}")
    
    # Ground state energy per site
    Egs = Eg / np.arange(1, N + 1)
    
    if verbose:
        # Print exact result for spin-1/2
        if spin == 0.5:
            Eexact = 0.25 - np.log(2)  # exact GS energy for infinite N
            print(f"\nExact ground state energy per site: {Eexact:.6f}")
        print(f"Final iterative estimate: {Egs[-1]:.6f}")
    
    return Eg, Egs, mps


def main():
    """Command-line interface for iterative diagonalization."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Iterative diagonalization for spin chain (Heisenberg model)"
    )
    parser.add_argument(
        "-N", "--length", type=int, default=50,
        help="Maximum chain length (default: 50)"
    )
    parser.add_argument(
        "-K", "--nkeep", type=int, default=300,
        help="Maximum number of states to keep (default: 300)"
    )
    parser.add_argument(
        "-J", "--coupling", type=float, default=1.0,
        help="Spin-spin coupling constant (default: 1.0)"
    )
    parser.add_argument(
        "-S", "--spin", type=float, default=0.5,
        help="Total spin quantum number (default: 0.5)"
    )
    parser.add_argument(
        "-Y", "--symmetry", type=str, default="U1",
        choices=["U1", "SU2"],
        help="Symmetry to exploit: 'U1' (default) or 'SU2'"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    Eg, Egs, mps = iter_diag_spin(
        N=args.length,
        Nkeep=args.nkeep,
        J=args.coupling,
        spin=args.spin,
        symmetry=args.symmetry,
        verbose=not args.quiet
    )
    
    return Eg, Egs, mps


if __name__ == "__main__":
    main()
