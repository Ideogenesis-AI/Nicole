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


"""Iterative diagonalization for free fermion tight-binding chain.

This module implements an iterative diagonalization algorithm to compute the
ground state energy and MPS representation of a tight-binding chain:

    H = -t Σ_i (c†_i c_{i+1} + c†_{i+1} c_i)

The algorithm builds up the chain site by site, diagonalizing the Hamiltonian
at each step and truncating to keep only the lowest-energy states.

Key features:
- Supports U(1) (total particle number) and Z2 (fermion parity) symmetry
- No on-site energy (pure hopping Hamiltonian)
- Exact reference solution from single-particle diagonalization

For the infinite tight-binding chain at half-filling (open boundary conditions),
the exact ground state energy per site is:

    E_exact / N = -2t/π ≈ -0.6366 t

which can be used to verify convergence.
"""

import time
from typing import Tuple

import numpy as np

from nicole import identity, isometry, diag
from nicole import einsum, load_space
from nicole.decomp import eig


def disptime(msg):
    """Display message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def exact_halffilling_energy(N: int, t: float = 1.0) -> float:
    """Exact ground state energy of an N-site tight-binding chain at half-filling.

    Uses the analytic single-particle spectrum of the open-boundary chain:
        ε_k = -2t cos(kπ / (N+1)),  k = 1, ..., N

    The ground state fills the ⌊N/2⌋ lowest single-particle levels.

    Parameters
    ----------
    N : int
        Chain length.
    t : float, optional
        Hopping amplitude (default: 1.0).

    Returns
    -------
    float
        Ground state energy at half-filling (⌊N/2⌋ particles).
    """
    n_particles = N // 2
    if n_particles == 0:
        return 0.0
    k = np.arange(1, n_particles + 1)
    return float(-2 * t * np.sum(np.cos(k * np.pi / (N + 1))))


def iter_diag_ferm(
    N: int = 50,
    Nkeep: int = 300,
    t: float = 1.0,
    symmetry: str = "U1",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Run iterative diagonalization for the free fermion tight-binding chain.

    Parameters
    ----------
    N : int, optional
        Maximum chain length (default: 50).
    Nkeep : int, optional
        Maximum number of states to keep after truncation (default: 300).
    t : float, optional
        Nearest-neighbor hopping amplitude (default: 1.0).
    symmetry : str, optional
        Symmetry sector to exploit (default: "U1").
        - "U1": conserve total particle number (block-diagonal in N_particles).
        - "Z2": exploit fermion parity only (block-diagonal in N_particles mod 2).
          Fewer conserved sectors; each block is larger.
    verbose : bool, optional
        Print progress messages (default: True).

    Returns
    -------
    Eg : np.ndarray
        Ground state energies at each iteration (length N).
    Egs : np.ndarray
        Ground state energy per site at each iteration (length N).
    mps : list of Tensor
        List of all isometry tensors AK generated at each iteration (length N).

    Notes
    -----
    The Hamiltonian is:
        H = -t Σ_i (c†_i c_{i+1} + c†_{i+1} c_i)

    At each step the hopping term between the last site of the left block (i)
    and the new site (i+1) is:
        H_hop = -t (c†_i c_{i+1} + c†_{i+1} c_i)
               = -t (F†_prev ⊗ F_now + F_prev ⊗ F†_now)

    where Fprev is the annihilation operator on site i accumulated in the
    truncated left-block basis. The second term is the Hermitian conjugate of
    the first, so only one is computed explicitly and the other added via
    `.conj().transpose()`.

    For the infinite chain at half-filling (N → ∞):
        E_exact / N = -2t/π ≈ -0.6366 t

    Examples
    --------
    >>> # Run with default parameters (U(1) symmetry)
    >>> Eg, Egs, mps = iter_diag_ferm()

    >>> # Use Z2 parity symmetry
    >>> Eg, Egs, mps = iter_diag_ferm(symmetry="Z2")

    >>> # Longer chain with more states kept
    >>> Eg, Egs, mps = iter_diag_ferm(N=100, Nkeep=500)
    """
    if symmetry not in ("U1", "Z2"):
        raise ValueError(f"Unsupported symmetry '{symmetry}'. Use 'U1' or 'Z2'.")

    tol = Nkeep * 100 * np.finfo(float).eps

    # Get local fermionic space and operators
    Spc, Op = load_space("Ferm", symmetry)
    F = Op["F"]       # Annihilation operator: (bra, ket, aux)
    I = identity(Spc)

    # Set itags for the fermionic operator
    F.retag(["s00", "s00", "op"])

    # Initialize: zero local Hamiltonian (no on-site energy in tight-binding)
    H0 = I * 1e-30
    H0.retag(["s00", "s00"])

    # A0: isometry from vacuum ⊗ physical space → permute to (vacuum, fused, physical)
    A0 = isometry(Op["vac"], Spc).permute([0, 2, 1])
    A0.retag(["L00", "R00", "s00"])

    Eg = np.zeros(N)
    bond_index = None
    Fprev = None   # Accumulated annihilation operator on the left-block edge
    mps = []

    for itN in range(1, N + 1):
        # Annihilation operator for the current site
        Fnow = F.clone()
        Fnow.retag([f"s{itN-1:02d}", f"s{itN-1:02d}", "op"])

        if itN == 1:
            # First iteration: sandwich zero Hamiltonian with A0
            # Anow.conj()[a,b,r], H0[r,s], Anow[a,c,s] → Hnow[b,c]
            Anow = A0
            Hnow = einsum('abr,rs,acs->bc', Anow.conj(), H0, Anow)

        else:
            # Add new site: isometry (left, phys, right) → permute to (left, right, phys)
            Anow = isometry(bond_index.flip(), Spc).permute([0, 2, 1])
            Anow.retag([f"R{itN-2:02d}", f"R{itN-1:02d}", f"s{itN-1:02d}"])

            # Sandwich previous Hamiltonian with Anow
            # Hprev[a,c], Anow[c,d,r], Anow.conj()[a,b,r] → Hnow[b,d]
            Hnow = einsum('ac,cdr,abr->bd', Hprev, Anow, Anow.conj())

            # Hopping term: -t (F†_prev F_now + F†_now F_prev)
            # Hopping term: F†_now = Fnow.conj().permute([2,1,0]) → Fn_dag[o,s,r]
            #   encodes ⟨β|F†|β'⟩ (creation on new site); axes: (op, ket, bra)
            Fn_dag = Fnow.conj().permute([2, 1, 0])

            # Fn_dag[o,s,r], Anow[c,d,r], Fprev[a,c,o], Anow.conj()[a,b,s] → HFF[b,d]
            HFF = einsum('osr,cdr,aco,abs->bd', Fn_dag, Anow, Fprev, Anow.conj())

            # Add Hermitian conjugate (= F†_prev ⊗ F_now) and scale
            HFF = (HFF + HFF.conj().transpose()) * (-t)

            Hnow = Hnow + HFF

        # Symmetrize and diagonalize
        Hnow_sym = (Hnow + Hnow.conj().transpose()) * 0.5

        if itN == 1:
            V, D = eig(Hnow_sym, is_hermitian=True)
        elif itN == N:
            V, D = eig(Hnow_sym, trunc={"nkeep": 1}, is_hermitian=True)
        else:
            V, D = eig(Hnow_sym, trunc={"nkeep": Nkeep}, is_hermitian=True)

        V.retag([f"R{itN-1:02d}", f"R{itN-1:02d}"])

        all_eigvals = np.concatenate([eigvals for eigvals in D.values()])
        Eg[itN - 1] = np.min(all_eigvals)

        # Anow[a,b,r], V[b,c] → AK[a,c,r] = (left, right_new, phys)
        AK = einsum('abr,bc->acr', Anow, V)
        mps.append(AK.clone())

        bond_index = V.indices[1]
        Hprev = diag(D, bond_index, itags=(f"R{itN-1:02d}", f"R{itN-1:02d}"))

        # Accumulate annihilation operator on the new right edge for the next step
        # AK[a,c,s], Fnow[r,s,o], AK.conj()[a,b,r] → Fprev[b,c,o] = (right_conj, right, op)
        Fprev = einsum('acs,rso,abr->bco', AK, Fnow, AK.conj())

        if verbose:
            NK = AK.indices[0].dim
            Hnow_dim = Hnow.indices[1].dim
            disptime(f"#{itN:02d}/{N:02d} : NK={NK}/{Hnow_dim}")

    Egs = Eg / np.arange(1, N + 1)

    if verbose:
        E_inf = -2 * t / np.pi
        E_exact_N = exact_halffilling_energy(N, t)
        print(f"\nExact GS energy per site ({N} sites, half-filling): {E_exact_N / N:.6f}")
        print(f"Exact GS energy per site (N → ∞, half-filling):    {E_inf:.6f}")
        print(f"Final iterative estimate:                            {Egs[-1]:.6f}")

    return Eg, Egs, mps


def main():
    """Command-line interface for iterative diagonalization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Iterative diagonalization for free fermion tight-binding chain"
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
        "-t", "--hopping", type=float, default=1.0,
        help="Nearest-neighbor hopping amplitude (default: 1.0)"
    )
    parser.add_argument(
        "-Y", "--symmetry", type=str, default="U1",
        choices=["U1", "Z2"],
        help="Symmetry to exploit: 'U1' (default) or 'Z2'"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    Eg, Egs, mps = iter_diag_ferm(
        N=args.length,
        Nkeep=args.nkeep,
        t=args.hopping,
        symmetry=args.symmetry,
        verbose=not args.quiet,
    )

    return Eg, Egs, mps


if __name__ == "__main__":
    main()
