"""Iterative diagonalization for spin chain (Heisenberg model).
"""

import numpy as np
from typing import Optional, Tuple, Dict
from nicole import load_space, contract, identity, isometry, conj, permute, transpose
from nicole.decomp import eig
from nicole.index import Direction
from nicole.tensor import Tensor
import time


def disptime(msg):
    """Display message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def iter_diag_spin(
    N: int = 50,
    Nkeep: int = 300,
    J: float = 1.0,
    spin: float = 0.5,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run iterative diagonalization for spin chain (Heisenberg model).
    
    Parameters
    ----------
    N : int, optional
        Maximum chain length (default: 50)
    Nkeep : int, optional
        Maximum number of states to keep after truncation (default: 300)
    J : float, optional
        Spin-spin coupling constant (default: 1.0)
    spin : float, optional
        Total spin quantum number for each site (default: 0.5 for spin-1/2)
        Must be a half-integer: 0.5, 1.0, 1.5, 2.0, etc.
    verbose : bool, optional
        Print progress messages (default: True)
    
    Returns
    -------
    E0 : np.ndarray
        Ground state energies at each iteration (length N)
    EG_iter : np.ndarray
        Ground state energy per site at each iteration (length N)
    Eexact : float
        Exact ground state energy per site for infinite chain (spin-1/2 only)
    AK_list : list of Tensor
        List of all isometry tensors AK generated at each iteration (length N)
    
    Notes
    -----
    The Heisenberg Hamiltonian is:
        H = J * sum_i (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+ + S_i^z S_{i+1}^z)
    
    For spin-1/2 chains, the exact ground state energy per site for infinite N is:
        E_exact = 1/4 - ln(2) ≈ -0.443147
    
    Examples
    --------
    >>> # Run with default parameters
    >>> E0, EG_per_site, E_exact, AK_list = iter_diag_spin()
    
    >>> # Longer chain with more states kept
    >>> E0, EG_per_site, E_exact, AK_list = iter_diag_spin(N=100, Nkeep=500)
    
    >>> # Spin-1 chain
    >>> E0, EG_per_site, E_exact, AK_list = iter_diag_spin(spin=1.0)
    """
    
    tol = Nkeep * 100 * np.finfo(float).eps  # numerical tolerance for degeneracy
    
    # Get local spin space and operators
    Spc, Op = load_space("Spin", "U1", {"J": spin})
    Op["Sz"].insert_index(2, direction=Direction.OUT)
    S = Op["Sp"] + Op["Sm"] + Op["Sz"]  # Sum of all spin operators
    I = identity(Spc)  # Identity operator
    
    # Set itags for spin operators
    S.retag(["s00", "s00", "op"])
    
    # Initialize
    H0 = I * 1e-30  # Hamiltonian for only the 1st site (zero operator)
    H0.retag(["s00", "s00"])
    
    # A0: isometry from vacuum ⊗ physical space, then permute to get (left, right, physical)
    # In MATLAB: getIdentity(getvac(H0),2,H0,2,[1 3 2])
    # Natural isometry gives (vacuum, physical, fused) = (0, 1, 2)
    # [1 3 2] permutes to (vacuum, fused, physical) = (0, 2, 1)
    # With itags: (L00, R00, s00)
    A0_temp = isometry(Op["vac"], Spc)
    # Permute from (0, 1, 2) to (0, 2, 1) to get (left-bond, right-bond, physical)
    A0 = permute(A0_temp, [0, 2, 1])
    A0.retag(["L00", "R00", "s00"])
    
    # Lowest energies at each iteration
    E0 = np.zeros(N)
    
    # Track the bond index for subsequent iterations
    bond_index = None
    
    # Initialize Sprev (will be set in first iteration)
    Sprev = None
    
    # Store all AK (isometry) tensors
    AK_list = []
    
    for itN in range(1, N + 1):
        # Create spin operator for the current site with proper itags
        Snow = Op["Sp"] + Op["Sm"] + Op["Sz"]
        Snow.retag([f"s{itN-1:02d}", f"s{itN-1:02d}", "op"])
        
        if itN == 1:
            # First iteration: Hnow = contract(A0,'!2*',{H0,'!1',A0})
            # A0 has legs: (left=0, right=1, phys=2)
            # H0 has legs: (out=0, in=1)
            # {H0,'!1',A0}: contract H0[in=1] with A0[left=0]
            #   Result has: (H0[out=0], A0[right=1], A0[phys=2])
            # contract(A0,'!2*', result): conj(A0) excludes index 1 (right)
            #   Contract conj(A0)[left=0] with H0[out=0]
            #   Contract conj(A0)[phys=2] with A0[phys=2]
            #   Result has: (conj(A0)[right=1], A0[right=1])
            Anow = A0
            temp = contract(H0, A0, axes=(1, 2))
            Hnow = contract(conj(Anow), temp, axes=([0, 2], [1, 0]))
            
        else:
            # Add new site: Anow = getIdentity(Aprev, 2, I.E, 2, [1 3 2])
            # Create isometry and permute to get (left-bond, right-bond, physical)
            Anow_temp = isometry(bond_index.flip(), Spc)
            Anow = permute(Anow_temp, [0, 2, 1])
            # Update itags (left-bond, right-bond, physical)
            Anow.retag([f"R{itN-2:02d}", f"R{itN-1:02d}", f"s{itN-1:02d}"])
            
            # Update the Hamiltonian: Hnow = contract(Anow,'!2*',{Hprev,'!1',Anow})
            # Anow has legs: (left=0, right=1, phys=2)
            # Hprev has legs: (out=0, in=1)
            # {Hprev,'!1',Anow}: contract Hprev[in=1] with Anow[left=0]
            # contract(Anow,'!2*', result): conj(Anow) excludes index 1 (right)
            temp = contract(Hprev, Anow, axes=(1, 0))
            Hnow = contract(conj(Anow), temp, axes=([0, 2], [0, 2]))
            
            # Spin-spin interaction: HSS = contract(Anow,'!2*',{Sprev,'!1',{Sn,'!1',Anow}})
            # Hermitian conjugate of the spin operator at the current site
            # Snow has legs: (out=0, in=1, op=2)
            # Permute to (op=0, in=1, out=2) then conjugate
            Sn = conj(permute(Snow, [2, 1, 0]))
            
            # Innermost: {Sn,'!1',Anow}
            # Sn after permute+conj has legs: (op=0, in=1, out=2)
            # Anow has legs: (left=0, right=1, phys=2)
            # Contract Sn[out=2] with Anow[left=0]? Or match by itags...
            # Actually, Sn should contract its physical legs with Anow's physical leg
            # So contract Sn[in=1] with Anow[phys=2]
            temp1 = contract(Sn, Anow, axes=(2, 2))
            
            # Middle: {Sprev,'!1',temp1}
            # Sprev contracts with temp1
            temp2 = contract(Sprev, temp1, axes=([1, 2], [2, 0]))
            
            # Outer: contract(Anow,'!2*',temp2)
            # conj(Anow) excludes index 1 (right), contracts indices 0 and 2
            HSS = contract(conj(Anow), temp2, axes=([0, 2], [0, 1]))
            HSS = HSS * J
            
            Hnow = Hnow + HSS
        
        # Symmetrize and diagonalize
        Hnow_sym = (Hnow + transpose(conj(Hnow))) * 0.5
        
        # Diagonalize
        if itN == 1:
            V, D = eig(Hnow_sym)
        else:
            V, D = eig(Hnow_sym, trunc={"nkeep": Nkeep})
        
        # Set itags for eigenvectors
        V.retag([f"R{itN-1:02d}", f"R{itN-1:02d}"])
        
        # Get minimum eigenvalue
        all_eigvals = np.concatenate([eigvals for eigvals in D.values()])
        E0[itN - 1] = np.min(all_eigvals)
        
        # Contract Anow with V to get AK
        # In MATLAB: Aprev = contract(Anow,'!1',Ieig.AK,[1 3 2])
        # Anow has legs: (left=0, right=1, phys=2)
        # V has legs: (in=0, out=1) where in corresponds to the Hilbert space before truncation
        # Exclude index 1 (right) of Anow, so contract: Anow[left=0] with V[in=0]
        # Wait, that doesn't make sense. Let me reconsider...
        # Actually, Anow[right=1] is the NEW Hilbert space that we just diagonalized
        # V maps from this space to the truncated space
        # So contract Anow[right=1] with V[in=0]
        # Result: (Anow[left=0], Anow[phys=2], V[out=1])
        # [1 3 2] permutes to: (Anow[left=0], V[out=1], Anow[phys=2])
        AK_temp = contract(Anow, V, axes=(1, 0))
        AK = permute(AK_temp, [0, 2, 1])
        
        # Store AK for this iteration
        AK_list.append(AK.copy())
        
        # Create diagonal Hprev from truncated eigenvalues
        bond_index = V.indices[1]  # Update bond index for next iteration
        Hprev_blocks = {}
        for key, eigvals in D.items():
            Hprev_blocks[key] = np.diag(eigvals)
        
        Hprev = Tensor(
            indices=(bond_index.flip(), bond_index),
            itags=(f"R{itN-1:02d}", f"R{itN-1:02d}"),
            data=Hprev_blocks,
            dtype=np.float64
        )
        
        # Spin operator at the current site
        # Sprev = contract(Aprev,'!2*',{Fnow(ito),'!1',Aprev},[1 3 2])
        # AK has legs: (left=0, right=1, phys=2) after permutation
        # Snow has legs: (out=0, in=1, op=2)
        # Physical interpretation: conj(AK)-Snow-AK sandwich, contracting physical and left legs
        #
        # {Snow,'!1',AK}: contract Snow with AK, excluding Snow's index 1 (MATLAB) = index 0 (Python)
        #   So contract Snow[in=1] with AK[phys=2]
        #   Result: (Snow[out=0], Snow[op=2], AK[left=0], AK[right=1])
        temp = contract(Snow, AK, axes=(1, 2))
        # contract(AK,'!2*',temp): conj(AK) excludes index 1 (right in Python)
        #   Contract conj(AK)[left=0] with AK[left=0] (from temp)
        #   Contract conj(AK)[phys=2] with Snow[out=0] (from temp)
        #   Remaining: conj(AK)[right=1], Snow[op=2], AK[right=1]
        # After first Snow-AK contraction, temp has: (Snow[out], Snow[op], AK[left], AK[right])
        # Indices are: (0, 1, 2, 3)
        Sprev_temp = contract(conj(AK), temp, axes=([0, 2], [2, 0]))
        # Result indices: (conj(AK)[right], Snow[op], AK[right])
        # [1 3 2] permutes to: (conj(AK)[right], AK[right], Snow[op])
        # In the result, indices are (0, 1, 2), permute to [0, 2, 1]
        Sprev = permute(Sprev_temp, [0, 2, 1])
        
        # Display progress
        if verbose:
            NK = AK.indices[0].dim  # Size of truncated space
            Hnow_dim = Hnow.indices[1].dim  # Size of Hilbert space before truncation
            disptime(f"#{itN:02d}/{N:02d} : NK={NK}/{Hnow_dim}")
    
    # Ground state energy per site
    EG_iter = E0 / np.arange(1, N + 1)
    
    # Exact result only available for spin-1/2
    if spin == 0.5:
        Eexact = 0.25 - np.log(2)  # exact GS energy for infinite N
    else:
        Eexact = np.nan  # No analytical solution for higher spins
    
    if verbose:
        if not np.isnan(Eexact):
            print(f"\nExact ground state energy per site: {Eexact:.6f}")
        print(f"Final iterative estimate: {EG_iter[-1]:.6f}")
    
    return E0, EG_iter, Eexact, AK_list


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
        "-q", "--quiet", action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    E0, EG_iter, Eexact, AK_list = iter_diag_spin(
        N=args.length,
        Nkeep=args.nkeep,
        J=args.coupling,
        spin=args.spin,
        verbose=not args.quiet
    )
    
    return E0, EG_iter, Eexact, AK_list


if __name__ == "__main__":
    main()
