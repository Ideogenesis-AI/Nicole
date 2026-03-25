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


"""Iterative diagonalization for a free spinful tight-binding (band conductor) chain.

This module implements an iterative diagonalization algorithm to compute the
ground state energy and MPS representation of the spinful tight-binding model:

    H = -t Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.)

The physical site has four states: |0⟩, |↑⟩, |↓⟩, |↑↓⟩ ("Band" preset).

Key features:
- Supports all four Band symmetries: "U1,U1", "Z2,U1", "U1,SU2", "Z2,SU2"
- Exact reference from single-particle diagonalization (spin-degenerate levels)

Because the Hamiltonian is spin-diagonal it decouples into two identical spinless
chains. At half-filling (2 electrons per site) each spin species fills ⌊N/2⌋
levels of

    ε_k = -2t cos(k π / (N+1)),  k = 1, …, N

giving an exact ground state energy of 2 × E_spinless(N, t). For large N this
converges to

    E / N  →  -4t/π  ≈  -1.2732 t  per site.
"""

import time
from typing import Tuple

import numpy as np

from nicole import Tensor, load_space
from nicole import contract, identity, isometry, conj, permute, transpose, diag
from nicole.decomp import eig


_ABELIAN_SYMMETRIES    = ("U1,U1", "Z2,U1")
_NONABELIAN_SYMMETRIES = ("U1,SU2", "Z2,SU2")
_ALL_SYMMETRIES        = _ABELIAN_SYMMETRIES + _NONABELIAN_SYMMETRIES


def disptime(msg: str) -> None:
    """Display message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def exact_halffilling_energy_band(N: int, t: float = 1.0) -> float:
    """Exact ground state energy of an N-site spinful tight-binding chain at half-filling.

    The Hamiltonian is spin-diagonal, so each spin species (↑ and ↓) is an
    independent spinless chain. At half-filling (⌊N/2⌋ particles per spin) the
    total energy is twice that of a single spinless chain:

        E = 2 × Σ_{k=1}^{⌊N/2⌋} (-2t cos(k π / (N+1)))

    Uses the analytic single-particle spectrum of the open-boundary chain:
        ε_k = -2t cos(k π / (N+1)),  k = 1, …, N

    Parameters
    ----------
    N : int
        Chain length.
    t : float, optional
        Hopping amplitude (default: 1.0).

    Returns
    -------
    float
        Exact ground state energy at half-filling.
    """
    n_particles = N // 2
    if n_particles == 0:
        return 0.0
    k = np.arange(1, n_particles + 1)
    return float(-4 * t * np.sum(np.cos(k * np.pi / (N + 1))))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_hff(ZFprev: Tensor, Fnow: Tensor, Anow: Tensor) -> Tensor:
    """Compute the one-body hopping contribution for a single fermionic species.

    ``ZFprev`` must be the accumulated (Z × F) operator on the left-block edge,
    NOT the bare F operator. This encodes the Jordan-Wigner string

        c†_{σ,i} c_{σ,i+1}  =  F†_{σ,i}  Z_{total,i}  F_{σ,i+1}

    which arises because site i contributes a factor (-1)^{N_i} when commuting
    the creation operator past all modes within that site in the site-ordered MPS.

    Returns the un-scaled, un-symmetrized term
        (Z F)†_{i} F_{i+1}  =  F†_{i} Z_{i} F_{i+1}
    expressed in the current effective Hilbert space. The caller is responsible
    for adding the Hermitian conjugate and the scale -t.
    """
    # (ZF)†_now = conj(permute(Fnow, [2, 1, 0]))   → (op*, ket*, bra*)
    Fn_dag = conj(permute(Fnow, [2, 1, 0]))
    # (op*, ket*, L, R)
    Fn_dag_Anow = contract(Fn_dag, Anow, axes=(2, 2))
    # (R*, ket*, R)
    ZFprev_Fn_dag_Anow = contract(ZFprev, Fn_dag_Anow, axes=([1, 2], [2, 0]))
    # (R*, R)
    return contract(conj(Anow), ZFprev_Fn_dag_Anow, axes=([0, 2], [0, 1]))


def _zf_product(Z: Tensor, F: Tensor) -> Tensor:
    """Return the operator product Z × F on a single physical site.

    Z : (bra=IN, ket=OUT)  – Jordan-Wigner parity operator
    F : (bra=IN, ket=OUT, aux=OUT)  – fermionic annihilation operator

    ⟨bra|Z F|ket⟩ = Σ_s Z[bra,s] F[s,ket,aux]

    Contracts Z's ket (axis 1, OUT) with F's bra (axis 0, IN).
    Result has the same index structure as F.
    """
    return contract(Z, F, axes=(1, 0))


def _push_zf(Z: Tensor, F: Tensor, AK: Tensor) -> Tensor:
    """Accumulate the (Z × F) operator through the isometry AK.

    Given AK: (left, right, phys) and F: (bra, ket, op), returns the effective
    (Z × F) operator on the new right edge with axes (left_conj, right, op).
    This accumulated operator is used to implement the inter-site JW string.
    """
    ZF = _zf_product(Z, F)
    ZF_AK = contract(ZF, AK, axes=(1, 2))
    return contract(conj(AK), ZF_AK, axes=([0, 2], [2, 0]), perm=[0, 2, 1])


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def iter_diag_band(
    N: int = 50,
    Nkeep: int = 300,
    t: float = 1.0,
    symmetry: str = "U1,U1",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Run iterative diagonalization for the free spinful tight-binding (band) chain.

    Parameters
    ----------
    N : int, optional
        Chain length (default: 50).
    Nkeep : int, optional
        Maximum states to keep per truncation step (default: 300).
        For non-Abelian symmetries this counts SU(2) multiplets.
    t : float, optional
        Nearest-neighbor hopping amplitude (default: 1.0).
    symmetry : str, optional
        Band symmetry to exploit (default: "U1,U1").

        ============  ====================================================
        "U1,U1"       particle number × spin-z  (fully Abelian)
        "Z2,U1"       fermion parity × spin-z   (Abelian)
        "U1,SU2"      particle number × SU(2)   (non-Abelian spin)
        "Z2,SU2"      fermion parity × SU(2)    (non-Abelian spin)
        ============  ====================================================

        Spaces in the symmetry string are ignored ("U1, SU2" == "U1,SU2").
    verbose : bool, optional
        Print per-step progress (default: True).

    Returns
    -------
    Eg : np.ndarray
        Ground state energy at each iteration (length N).
    Egs : np.ndarray
        Ground state energy per site at each iteration (length N).
    mps : list of Tensor
        Isometry tensors AK generated at each iteration (length N).

    Notes
    -----
    The Hamiltonian is

        H = -t Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.)

    At each step the hopping term between sites i and i+1 is

        H_hop = -t Σ_σ (c†_{i,σ} c_{i+1,σ} + h.c.)

    For Abelian symmetries (U1,U1 and Z2,U1), F_up and F_dn are handled
    separately and their hopping contributions are summed to form HFF:

        HFF = HFF_up + HFF_dn

    For non-Abelian symmetries (U1,SU2 and Z2,SU2), the single rank-1/2
    tensor F carries both spin components through its SU(2) auxiliary index
    (j = 1/2), so the spin sum is implicit in the contraction.

    For the infinite chain at half-filling (N → ∞)

        E / N  →  -4t/π  ≈  -1.2732 t.

    Examples
    --------
    >>> Eg, Egs, mps = iter_diag_band(symmetry="U1,U1")
    >>> Eg, Egs, mps = iter_diag_band(symmetry="Z2,SU2")
    """
    sym_norm = symmetry.replace(" ", "")
    if sym_norm not in _ALL_SYMMETRIES:
        raise ValueError(
            f"Unsupported symmetry '{symmetry}'. "
            f"Use one of: {_ALL_SYMMETRIES}."
        )

    is_su2    = "SU2" in sym_norm
    is_abelian = sym_norm in _ABELIAN_SYMMETRIES

    # Load local Band space and operators
    Spc, Op = load_space("Band", symmetry)
    I  = identity(Spc)

    Z = Op["Z"]
    Z.retag(["s00", "s00"])

    # For Abelian symmetries, sum F_up and F_dn into a single F. Tensor addition
    # takes the union of charge sectors, so the result has blocks for both aux
    # charges (-1,−1) and (-1,+1) — a combined F that carries both spin
    # components, exactly like the non-Abelian F. The loop logic is then
    # identical for all symmetries.
    F = Op["F_up"] + Op["F_dn"] if is_abelian else Op["F"]
    F.retag(["s00", "s00", "op"])

    # Zero on-site Hamiltonian (pure hopping model)
    H0 = I * 1e-30
    H0.retag(["s00", "s00"])

    # First isometry: vacuum ⊗ physical → (vacuum, bond, phys)
    A0 = permute(isometry(Op["vac"], Spc), [0, 2, 1])
    A0.retag(["L00", "R00", "s00"])

    Eg         = np.zeros(N)
    bond_index = None
    mps        = []
    ZFprev     = None   # accumulated (Z × F) at the left-block edge

    for itN in range(1, N + 1):

        Z_now = Z.clone()
        Z_now.retag([f"s{itN-1:02d}", f"s{itN-1:02d}"])

        Fnow = F.clone()
        Fnow.retag([f"s{itN-1:02d}", f"s{itN-1:02d}", "op"])

        if itN == 1:
            Anow = A0
            Hnow = contract(H0, A0, axes=(1, 2))
            Hnow = contract(conj(Anow), Hnow, axes=([0, 2], [1, 0]))

        else:
            Anow = permute(isometry(bond_index.flip(), Spc), [0, 2, 1])
            Anow.retag([f"R{itN-2:02d}", f"R{itN-1:02d}", f"s{itN-1:02d}"])

            # Sandwich previous Hamiltonian
            Hnow = contract(Hprev, Anow, axes=(1, 0))
            Hnow = contract(conj(Anow), Hnow, axes=([0, 2], [0, 2]))

            # Hopping term with JW string: (Z×F)†_prev × F_now.
            # The aux-index contraction sums over all spin components automatically.
            HFF = _compute_hff(ZFprev, Fnow, Anow)

            HFF = (HFF + transpose(conj(HFF))) * (-t)
            Hnow = Hnow + HFF

        # Symmetrize and diagonalize
        Hnow_sym = (Hnow + transpose(conj(Hnow))) * 0.5

        if itN == 1:
            V, D = eig(Hnow_sym, is_hermitian=True)
        elif itN == N:
            V, D = eig(Hnow_sym, trunc={"nkeep": 1}, is_hermitian=True)
        else:
            V, D = eig(Hnow_sym, trunc={"nkeep": Nkeep}, is_hermitian=True)

        V.retag([f"R{itN-1:02d}", f"R{itN-1:02d}"])

        all_eigvals  = np.concatenate([v.numpy() for v in D.values()])
        Eg[itN - 1]  = float(np.min(all_eigvals))

        AK = contract(Anow, V, axes=(1, 0), perm=[0, 2, 1])
        mps.append(AK.clone())

        bond_index = V.indices[1]
        Hprev = diag(D, bond_index, itags=(f"R{itN-1:02d}", f"R{itN-1:02d}"))

        # Accumulate (Z × F) through AK for the next step's hopping JW string
        ZFprev = _push_zf(Z_now, Fnow, AK)

        if verbose:
            dim_fn = (lambda idx: idx.num_states) if is_su2 else (lambda idx: idx.dim)
            disptime(f"#{itN:02d}/{N:02d} : NK={dim_fn(AK.indices[0])}/{dim_fn(Hnow.indices[1])}")

    Egs = Eg / np.arange(1, N + 1)

    if verbose:
        E_inf      = -4.0 * t / np.pi
        E_exact_N  = exact_halffilling_energy_band(N, t)
        print(f"\nExact GS energy per site ({N} sites, half-filling): {E_exact_N / N:.6f}")
        print(f"Exact GS energy per site (N → ∞, half-filling):    {E_inf:.6f}")
        print(f"Final iterative estimate:                            {Egs[-1]:.6f}")

    return Eg, Egs, mps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for band-conductor iterative diagonalization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Iterative diagonalization for free spinful tight-binding chain"
    )
    parser.add_argument(
        "-N", "--length", type=int, default=50,
        help="Chain length (default: 50)"
    )
    parser.add_argument(
        "-K", "--nkeep", type=int, default=300,
        help="Maximum states to keep (default: 300)"
    )
    parser.add_argument(
        "-t", "--hopping", type=float, default=1.0,
        help="Hopping amplitude (default: 1.0)"
    )
    parser.add_argument(
        "-Y", "--symmetry", type=str, default="U1,U1",
        choices=list(_ALL_SYMMETRIES),
        help="Symmetry to exploit (default: U1,U1)"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    iter_diag_band(
        N=args.length,
        Nkeep=args.nkeep,
        t=args.hopping,
        symmetry=args.symmetry,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
