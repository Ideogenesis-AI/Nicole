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


"""Spectrum tests: verify that contraction and axis-merging yield the correct
eigenvalue structure for physically known models."""

import torch
import pytest

from nicole import Direction, contract, load_space, merge_axes


def _heisenberg_eigenvalue(j_site: float, j_total: int) -> float:
    """Analytic eigenvalue of S1·S2 for two sites of spin j_site coupled to total spin j_total.

    Derived from the Wigner-Eckart decomposition of S1·S2:

        S1·S2 = (1/2) [ J_total(J_total+1) - J1(J1+1) - J2(J2+1) ]
              = (1/2) [ J_total(J_total+1) - 2*j_site*(j_site+1) ]
    """
    return 0.5 * (j_total * (j_total + 1) - 2 * j_site * (j_site + 1))


# ---------------------------------------------------------------------------
# U(1) Heisenberg spectrum
# ---------------------------------------------------------------------------


def _build_heisenberg_u1(j_site: float):
    """Build the two-site Heisenberg Hamiltonian S₁·S₂ for U(1) symmetry.

    A 3-index "S vector" tensor is assembled from the three spin operators by
    packing them into a shared auxiliary index (same structure as the SU(2) S
    tensor):

        aux charge  0 → Sz component  (scalar)
        aux charge +2 → Sp = −S⁺/√2  (raises site charge by 2)
        aux charge −2 → Sm =  S⁻/√2  (lowers site charge by 2)

    Sz is cloned and given a trivial OUT(0) auxiliary via insert_index; its
    sectors are then unioned with the existing auxiliaries of Sp and Sm via
    tensor addition.  The contraction then follows the identical procedure as
    the SU(2) case, and yields S₁·S₂ = Sz₁Sz₂ − (Sp₁Sm₂ + Sm₁Sp₂).
    """
    _, Op = load_space("Spin", preserv="U1", option={"J": j_site})
    Sz, Sp, Sm = Op["Sz"], Op["Sp"], Op["Sm"]

    # Give Sz a trivial auxiliary (charge 0, dim 1) so it has the same
    # 3-index structure as Sp and Sm
    Sz3 = Sz.clone()
    Sz3.insert_index(2, Direction.OUT)

    # Union the three operators into one 3-index tensor.
    # _align_for_binary checks only group and direction, so the sector
    # sets on the auxiliary may differ; __add__ calls union_indices, which
    # expands the auxiliary to cover charges {0, +2, -2}.
    S = Sz3 + Sp + Sm

    # Identical contraction and fusion as in the SU(2) case
    SS = contract(S.conj().permute([1, 0, 2]), S, axes=(2, 2))
    SSdiag, _ = merge_axes(SS, (0, 2), merged_tag="ss", direction=Direction.IN)
    SSdiag, _ = merge_axes(SSdiag, (1, 2), merged_tag="ss")

    return SSdiag


def test_heisenberg_two_site_spectrum_u1():
    """Two-site U(1) Heisenberg spectrum: −3/4 (singlet) and +1/4 (triplet).

    H = Sz₁Sz₂ − (Sp₁Sm₂ + Sm₁Sp₂) is block-diagonal in total Sz.
    After merging physical indices the blocks are:

    - q_total = ±2  (M = ±1, triplet extremes): 1×1 matrix,  E = +1/4
    - q_total =  0  (M =  0, singlet + triplet): 2×2 matrix,
                    eigenvalues −3/4 (singlet) and +1/4 (triplet)
    """
    H = _build_heisenberg_u1(0.5)

    # Triplet extreme sectors (M = ±1): 1×1 blocks, E = 1/4
    for q_total in (2, -2):
        assert (q_total, q_total) in H.data, f"Block ({q_total}, {q_total}) missing"
        block = H.data[(q_total, q_total)]
        assert torch.isclose(
            block.reshape(1)[0],
            torch.tensor(1.0 / 4.0, dtype=block.dtype),
            atol=1e-6,
        ), f"q_total={q_total}: expected 0.25, got {block.item()}"

    # Mixed sector (M = 0): 2×2 block, eigenvalues {−3/4, +1/4}
    assert (0, 0) in H.data, "Block (0, 0) missing"
    block = H.data[(0, 0)]
    n = block.shape[0]
    evals = torch.linalg.eigvalsh(block.reshape(n, n))
    expected = torch.tensor([-3.0 / 4.0, 1.0 / 4.0], dtype=block.dtype)
    assert torch.allclose(evals, expected, atol=1e-6), (
        f"q_total=0 eigenvalues: expected {expected.tolist()}, got {evals.tolist()}"
    )


@pytest.mark.parametrize(
    "j_site",
    [1.0, 1.5, 2.0, 2.5, 3.0],
    ids=["spin-1", "spin-3/2", "spin-2", "spin-5/2", "spin-3"],
)
def test_heisenberg_two_site_spectrum_generic_u1(j_site: float):
    """Two-site U(1) Heisenberg spectrum for spin-1 through spin-3.

    For two sites each carrying spin j_site, the block at total-Sz charge
    q_total = 2*M has eigenvalues

        E(J_total) = (1/2) [ J_total(J_total+1) − 2*j_site*(j_site+1) ]

    for J_total = |M|, |M|+1, …, 2*j_site (integer steps).  The block at
    q_total is an n×n matrix with n = 2*j_site − |M| + 1 eigenvalues.

    In Nicole's 2*m_z charge convention the per-site charges are integers,
    q_total ranges from −2*(2J) to +2*(2J) in steps of 2, and M = q_total/2.
    """
    H = _build_heisenberg_u1(j_site)

    two_J = round(2 * j_site)   # = 2*j_site as a positive integer

    # q_total = 2*M, ranging over even integers from −2*(2J) to +2*(2J)
    for q_total in range(-2 * two_J, 2 * two_J + 1, 2):
        m_total = q_total // 2  # total M (integer)

        assert (q_total, q_total) in H.data, (
            f"Block ({q_total}, {q_total}) missing (j_site={j_site})"
        )

        block = H.data[(q_total, q_total)]
        n = block.shape[0]
        evals = torch.linalg.eigvalsh(block.reshape(n, n))

        # Analytic eigenvalues: J_total = |M|, |M|+1, …, 2*j_site
        expected = torch.tensor(
            sorted(
                _heisenberg_eigenvalue(j_site, j_total)
                for j_total in range(abs(m_total), two_J + 1)
            ),
            dtype=block.dtype,
        )

        assert torch.allclose(evals, expected, atol=1e-6), (
            f"j_site={j_site}, q_total={q_total}: "
            f"expected {expected.tolist()}, got {evals.tolist()}"
        )


# ---------------------------------------------------------------------------
# SU(2) Heisenberg spectrum
# ---------------------------------------------------------------------------


def _build_heisenberg_su2(j_site: float):
    """Build the two-site Heisenberg Hamiltonian S₁·S₂ for SU(2) symmetry.

    S is a 3-index rank-1 spherical tensor loaded from the built-in 'Spin'
    space.  Contracting S†(site 1) with S(site 2) on the auxiliary index
    and fusing both pairs of physical indices produces a 2-index SU(2) matrix
    block-diagonal in total spin J.  Bridge weights are then brought to the
    sqrt(irrep_dim) canonical form via regularize so that the reduced data
    value in each block equals the sector eigenvalue directly.
    """
    _, Op = load_space("Spin", preserv="SU2", option={"J": j_site})
    S = Op["S"]

    # Contract S†(site 1) with S(site 2) along the auxiliary index;
    # S.conj().permute([1, 0, 2]) swaps the two physical indices of the conjugate
    SS = contract(S.conj().permute([1, 0, 2]), S, axes=(2, 2))

    # Fuse the two incoming physical indices (axes 0 and 2) into a combined bra
    SSdiag, _ = merge_axes(SS, (0, 2), merged_tag="ss", direction=Direction.IN)
    # Fuse the two outgoing physical indices (axes 1 and 2) into a combined ket
    SSdiag, _ = merge_axes(SSdiag, (1, 2), merged_tag="ss")

    # Bring Bridge weights to sqrt(irrep_dim) so reduced data = sector eigenvalue
    SSdiag.regularize()

    return SSdiag


def test_heisenberg_two_site_spectrum_su2():
    """Two-site SU(2) Heisenberg spectrum: -3/4 (singlet) and +1/4 (triplet).

    After regularization the reduced data value in each block equals the
    eigenvalue of S1·S2 in that sector:

    - J = 0 singlet  (charge q = 0, irrep_dim = 1): E = -3/4
    - J = 1 triplet  (charge q = 2, irrep_dim = 3): E = +1/4
    """
    H = _build_heisenberg_su2(0.5)

    # Singlet sector: total spin J = 0, charge q = 0, irrep_dim = 1
    assert (0, 0) in H.data, "Singlet block (0, 0) is missing"
    singlet = H.data[(0, 0)].item()
    assert torch.isclose(
        torch.tensor(singlet),
        torch.tensor(-3.0 / 4.0),
        atol=1e-6,
    ), f"Singlet eigenvalue: expected -0.75, got {singlet}"

    # Triplet sector: total spin J = 1, charge q = 2, irrep_dim = 3
    assert (2, 2) in H.data, "Triplet block (2, 2) is missing"
    triplet = H.data[(2, 2)].item()
    assert torch.isclose(
        torch.tensor(triplet),
        torch.tensor(1.0 / 4.0),
        atol=1e-6,
    ), f"Triplet eigenvalue: expected 0.25, got {triplet}"


@pytest.mark.parametrize(
    "j_site",
    [1.0, 1.5, 2.0, 2.5, 3.0],
    ids=["spin-1", "spin-3/2", "spin-2", "spin-5/2", "spin-3"],
)
def test_heisenberg_two_site_spectrum_generic_su2(j_site: float):
    """Two-site SU(2) Heisenberg spectrum for spin-1 through spin-3.

    For two sites each carrying spin j_site, the total spin J_total ranges from
    0 to 2*j_site in integer steps.  The analytic eigenvalue of S1·S2 in each
    sector is:

        E(J_total) = (1/2) [ J_total(J_total+1) - 2*j_site*(j_site+1) ]

    In Nicole's 2j convention the SU(2) charge of the site is q_site = 2*j_site
    and the charge of the combined system is q_total = 2*J_total.  After
    regularization, the reduced data value in block (q_total, q_total) equals
    the sector eigenvalue directly.
    """
    H = _build_heisenberg_su2(j_site)

    # q_site = 2*j_site in the 2j integer convention
    q_site = round(2 * j_site)

    # J_total = 0, 1, ..., 2*j_site  →  q_total = 0, 2, 4, ..., 2*q_site
    for j_total in range(q_site + 1):
        q_total = 2 * j_total
        expected = _heisenberg_eigenvalue(j_site, j_total)

        assert (q_total, q_total) in H.data, (
            f"Block ({q_total}, {q_total}) missing "
            f"(j_site={j_site}, J_total={j_total})"
        )
        actual = H.data[(q_total, q_total)].item()
        assert torch.isclose(
            torch.tensor(actual),
            torch.tensor(expected),
            atol=1e-6,
        ), (
            f"j_site={j_site}, J_total={j_total}: "
            f"expected {expected:.6f}, got {actual:.6f}"
        )
