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

from nicole import Direction, Index, Sector, Tensor, contract, load_space, merge_axes


def _heisenberg_eigenvalue(j_site: float, j_total: int) -> float:
    """Analytic eigenvalue of S1·S2 for two sites of spin j_site coupled to total spin j_total.

    Derived from the Wigner-Eckart decomposition of S1·S2:

        S1·S2 = (1/2) [ J_total(J_total+1) - J1(J1+1) - J2(J2+1) ]
              = (1/2) [ J_total(J_total+1) - 2*j_site*(j_site+1) ]
    """
    return 0.5 * (j_total * (j_total + 1) - 2 * j_site * (j_site + 1))


def _hopping_eigenvalues() -> list:
    """Analytic eigenvalues of the two-site hopping Hamiltonian F†₁F₂ + h.c.

    The spectrum is {−1, 0, 0, +1}: the ±1 pair lives in the single-particle
    sector (one fermion, 2×2 block) and the two zeros correspond to the vacuum
    and doubly-occupied sectors (no hopping, blocks absent from the data dict).
    """
    return [-1.0, 1.0]


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


# ---------------------------------------------------------------------------
# U(1) hopping spectrum
# ---------------------------------------------------------------------------


def _build_hopping_u1():
    """Build the two-site hopping Hamiltonian F†₁F₂ + h.c. for U(1) symmetry.

    F is a 3-index tensor (bra, ket, aux) loaded from the 'Ferm' space.
    Contracting F†(site 1) with F(site 2) on the auxiliary index gives H₁₂;
    its Hermitian conjugate H₁₂† = H₁₂.conj().permute([1, 0, 3, 2]) swaps
    bra ↔ ket on each site independently and represents F†₂F₁.

    No Jordan-Wigner Z factor is inserted: for nearest-neighbor hopping on a
    two-site chain there are no sites strictly between sites 1 and 2, so the
    JW string is trivial.  Algebraically, σ⁺Z = σ⁺ and Zσ⁻ = σ⁻ cause the
    Z₁ factors in c†₁c₂ = F†₁(Z₁F₂) and c†₂c₁ = (F†₂Z₁)F₁ to drop out,
    leaving F†₁F₂ and F†₂F₁ respectively.

    After merging physical indices the Hamiltonian is block-diagonal in total
    U(1) charge (sum of single-site charges, with empty = −1 and occupied = +1):

    - q_total = −2  (|00⟩, vacuum):           block absent, E = 0
    - q_total =  0  (|01⟩ and |10⟩):          2×2 block, eigenvalues {−1, +1}
    - q_total = +2  (|11⟩, doubly occupied):   block absent, E = 0
    """
    _, Op = load_space("Ferm", preserv="U1")
    F = Op["F"]

    # F†₁F₂: contract creation at site 1 with annihilation at site 2
    H12 = contract(F.conj().permute([1, 0, 2]), F, axes=(2, 2))

    # h.c.: swap bra ↔ ket on each site (F†₂F₁)
    H = H12 + H12.conj().permute([1, 0, 3, 2])

    H, _ = merge_axes(H, (0, 2), merged_tag="ff", direction=Direction.IN)
    H, _ = merge_axes(H, (1, 2), merged_tag="ff")

    return H


def test_hopping_two_site_spectrum_u1():
    """Two-site U(1) hopping spectrum: single-particle block has eigenvalues {−1, +1}.

    The vacuum (q_total = −2) and doubly-occupied (q_total = +2) sectors carry
    no hopping amplitude and their blocks are absent from the data dictionary.
    """
    H = _build_hopping_u1()

    # Vacuum and doubly-occupied sectors: no hopping → blocks absent
    assert (-2, -2) not in H.data, "Vacuum block (-2, -2) should be absent"
    assert (2, 2) not in H.data, "Doubly-occupied block (2, 2) should be absent"

    # Single-particle sector: 2×2 off-diagonal matrix, eigenvalues ±1
    assert (0, 0) in H.data, "Single-particle block (0, 0) missing"
    block = H.data[(0, 0)]
    n = block.shape[0]
    evals = torch.linalg.eigvalsh(block.reshape(n, n))
    expected = torch.tensor(_hopping_eigenvalues(), dtype=block.dtype)
    assert torch.allclose(evals, expected, atol=1e-6), (
        f"Single-particle eigenvalues: expected {expected.tolist()}, got {evals.tolist()}"
    )


# ---------------------------------------------------------------------------
# Z2 hopping spectrum
# ---------------------------------------------------------------------------


def _build_hopping_z2():
    """Build the two-site hopping Hamiltonian F†₁F₂ + h.c. for Z2 symmetry.

    The construction mirrors the U(1) case exactly; only the symmetry group
    changes (Z2 parity instead of U(1) charge).  Parity charges: empty = 0,
    occupied = 1.

    After merging physical indices the Hamiltonian is block-diagonal in total
    Z2 parity (XOR of single-site parities):

    - p_total = 0  (|00⟩ and |11⟩, even parity):  block absent, E = 0
    - p_total = 1  (|01⟩ and |10⟩, odd parity):   2×2 block, eigenvalues {−1, +1}
    """
    _, Op = load_space("Ferm", preserv="Z2")
    F = Op["F"]

    H12 = contract(F.conj().permute([1, 0, 2]), F, axes=(2, 2))
    H = H12 + H12.conj().permute([1, 0, 3, 2])

    H, _ = merge_axes(H, (0, 2), merged_tag="ff", direction=Direction.IN)
    H, _ = merge_axes(H, (1, 2), merged_tag="ff")

    return H


def test_hopping_two_site_spectrum_z2():
    """Two-site Z2 hopping spectrum: odd-parity block has eigenvalues {−1, +1}.

    States |00⟩ and |11⟩ both have even total parity and are not connected by
    hopping, so the even-parity block (p_total = 0) is absent.  States |01⟩
    and |10⟩ each have odd total parity and are connected by the hopping term,
    giving eigenvalues ±1 in the odd-parity block (p_total = 1).
    """
    H = _build_hopping_z2()

    # Even-parity sector: no hopping → block absent
    assert (0, 0) not in H.data, "Even-parity block (0, 0) should be absent"

    # Odd-parity sector: 2×2 off-diagonal matrix, eigenvalues ±1
    assert (1, 1) in H.data, "Odd-parity block (1, 1) missing"
    block = H.data[(1, 1)]
    n = block.shape[0]
    evals = torch.linalg.eigvalsh(block.reshape(n, n))
    expected = torch.tensor(_hopping_eigenvalues(), dtype=block.dtype)
    assert torch.allclose(evals, expected, atol=1e-6), (
        f"Odd-parity eigenvalues: expected {expected.tolist()}, got {evals.tolist()}"
    )


# ---------------------------------------------------------------------------
# U(1)×U(1) band hopping spectrum
# ---------------------------------------------------------------------------


def _build_band_hopping(preserv: str):
    """Build the two-site spinful hopping Hamiltonian Σ_σ(F†_σ₁F_σ₂ + h.c.).

    For spinful fermions each site has four states: |0⟩, |↑⟩, |↓⟩, |↑↓⟩.
    The physical operator at site 2 carries a Jordan-Wigner string from all
    modes at site 1:

        c_{2σ} = Z_site1 ⊗ F_{σ,2}

    Unlike the spinless case (where σ⁺Z = σ⁺ causes the Z to drop out), here
    F_up†Z ≠ F_up† because F_up† maps the odd-parity state |↓⟩ → |↑↓⟩ and Z
    flips the sign on that path.  The site-1 tensor is therefore F†_σ Z rather
    than F†_σ alone.

    The composition F†_σ Z is built by contracting F†'s ket index (axis 1, OUT)
    with Z's bra index (axis 0, IN), then permuting back to (bra, ket, aux) order.
    After that the construction is identical to the spinless case.

    No Z factor is needed at site 2: F_dn already encodes the intra-site
    Jordan-Wigner string Z_up within the site (the −1 coefficient in
    F_dn|↑↓⟩ = −|↑⟩).
    """
    _, Op = load_space("Band", preserv=preserv)
    F_up, F_dn, Z = Op["F_up"], Op["F_dn"], Op["Z"]

    # Compose F†_σ with Z at site 1: F†'s ket (axis 1) ↔ Z's bra (axis 0)
    FdagZ_up = contract(F_up.conj().permute([1, 0, 2]), Z, axes=(1, 0)).permute([0, 2, 1])
    FdagZ_dn = contract(F_dn.conj().permute([1, 0, 2]), Z, axes=(1, 0)).permute([0, 2, 1])

    # Hop each spin channel: (F†_σ Z)_site1 ⊗ F_{σ,site2}, contracted on aux
    H12_up = contract(FdagZ_up, F_up, axes=(2, 2))
    H12_dn = contract(FdagZ_dn, F_dn, axes=(2, 2))

    # Sum both channels then add h.c. via conj + bra↔ket swap per site
    H12 = H12_up + H12_dn
    H = H12 + H12.conj().permute([1, 0, 3, 2])

    H, _ = merge_axes(H, (0, 2), merged_tag="band", direction=Direction.IN)
    H, _ = merge_axes(H, (1, 2), merged_tag="band")

    return H


def _eigvalsh_block(H, key):
    """Return sorted eigenvalues of a square block from H.data."""
    block = H.data[key]
    n = block.shape[0]
    return torch.linalg.eigvalsh(block.reshape(n, n))


def test_hopping_two_site_spectrum_band_u1u1():
    """Two-site U(1)×U(1) spinful hopping spectrum.

    Block keys are charge tuples (q_charge, q_spin) where q_charge = N_site − 1
    offset (empty = −1, doubly occupied = +1) and q_spin = 2·Sz per site.
    Product-group charges are represented as tuples, so each block key has the
    form ((q_bra_charge, q_bra_spin), (q_ket_charge, q_ket_spin)).

    Absent blocks (no hopping amplitude):
      ((−2, 0), (−2, 0))  vacuum, N=0
      (( 0, ±2), (0, ±2)) both sites same-spin, N=2: Pauli blocks hopping
      (( 2, 0), ( 2, 0))  both sites doubly occupied, N=4

    Non-trivial blocks and eigenvalues (independent spin channels):
      ((−1, ±1), (−1, ±1))  N=1, one spin flavour:              {−1, +1}
      (( 0,  0), ( 0,  0))  N=2 Sz=0, both channels active:    {−2, 0, 0, +2}
      (( 1, ±1), ( 1, ±1))  N=3, one hole hops:                {−1, +1}
    """
    H = _build_band_hopping("U1,U1")

    # --- absent blocks ---
    for key, label in [
        (((-2, 0), (-2, 0)), "vacuum"),
        (((0,  2), (0,  2)), "N=2 spin-up"),
        (((0, -2), (0, -2)), "N=2 spin-down"),
        (((2,  0), (2,  0)), "doubly-occupied"),
    ]:
        assert key not in H.data, f"{label} block {key} should be absent"

    # --- single-particle / single-hole blocks: {−1, +1} ---
    for key in [((-1, 1), (-1, 1)), ((-1, -1), (-1, -1)),
                ((1, 1), (1, 1)), ((1, -1), (1, -1))]:
        assert key in H.data, f"Block {key} missing"
        evals = _eigvalsh_block(H, key)
        expected = torch.tensor([-1.0, 1.0], dtype=evals.dtype)
        assert torch.allclose(evals, expected, atol=1e-6), (
            f"Block {key}: expected {expected.tolist()}, got {evals.tolist()}"
        )

    # --- half-filling Sz=0: two channels each contribute ±1 → {−2, 0, 0, +2} ---
    key = ((0, 0), (0, 0))
    assert key in H.data, f"Half-filling block {key} missing"
    evals = _eigvalsh_block(H, key)
    expected = torch.tensor([-2.0, 0.0, 0.0, 2.0], dtype=evals.dtype)
    assert torch.allclose(evals, expected, atol=1e-6), (
        f"Block {key}: expected {expected.tolist()}, got {evals.tolist()}"
    )


# ---------------------------------------------------------------------------
# Z2×U(1) band hopping spectrum
# ---------------------------------------------------------------------------


def test_hopping_two_site_spectrum_band_z2u1():
    """Two-site Z2×U(1) spinful hopping spectrum.

    In Z2×U(1), |0⟩ and |↑↓⟩ share charge (0, 0) with dim=2, so after merging
    two sites the ((0,0),(0,0)) bra sector has dimension 6, combining N=0,
    N=2 Sz=0, and N=4 configurations.  Hopping conserves particle number so
    those sub-sectors are disconnected; the 6×6 block has eigenvalues
    {−2, 0, 0, 0, 0, +2}.

    The odd-parity sectors ((1,±1),(1,±1)) each contain four states spanning
    two independent hopping pairs: one N=1 hop and one N=3 hop (a hole hop
    with a spectator spin in the other channel).  Both pairs contribute ±1,
    giving eigenvalues {−1, −1, +1, +1}.

    Absent: ((0,±2),(0,±2)) — both sites carry the same spin, no hopping.
    """
    H = _build_band_hopping("Z2,U1")

    # --- absent blocks ---
    for key, label in [
        (((0,  2), (0,  2)), "N=2 spin-up"),
        (((0, -2), (0, -2)), "N=2 spin-down"),
    ]:
        assert key not in H.data, f"{label} block {key} should be absent"

    # --- mixed-N sector: 6×6, eigenvalues {−2, 0, 0, 0, 0, +2} ---
    key = ((0, 0), (0, 0))
    assert key in H.data, f"Block {key} missing"
    evals = _eigvalsh_block(H, key)
    expected = torch.tensor([-2.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=evals.dtype)
    assert torch.allclose(evals, expected, atol=1e-6), (
        f"Block {key}: expected {expected.tolist()}, got {evals.tolist()}"
    )

    # --- odd-parity sectors: 4×4, two independent hops → {−1, −1, +1, +1} ---
    for key in [((1, 1), (1, 1)), ((1, -1), (1, -1))]:
        assert key in H.data, f"Block {key} missing"
        evals = _eigvalsh_block(H, key)
        expected = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=evals.dtype)
        assert torch.allclose(evals, expected, atol=1e-6), (
            f"Block {key}: expected {expected.tolist()}, got {evals.tolist()}"
        )


# ---------------------------------------------------------------------------
# U(1)×SU(2) band hopping spectrum
# ---------------------------------------------------------------------------


def _build_band_hopping_su2(preserv: str):
    """Build the two-site spinful hopping Hamiltonian Σ_σ(F†_σ₁F_σ₂ + h.c.) for
    a symmetry group with SU(2) spin symmetry.

    SU(2) symmetry unifies |↑⟩ and |↓⟩ into a single doublet multiplet, so a
    single rank-½ tensor F (not separate F_up/F_dn) represents the fermionic
    annihilation operator.  The same Jordan-Wigner Z insertion applies: the
    Z factor does not drop out for spinful fermions, so site 1 uses (F†Z)₁
    composed via:

        FdagZ = contract(F†, Z, axes=(ket_of_F†=1, bra_of_Z=0)).permute([0, 2, 1])

    After merging physical indices, Bridge weights must be brought to canonical
    form via regularize() so that the reduced data values equal the physical
    eigenvalues directly (Wigner-Eckart theorem).

    The N=2 J=1 triplet at E=0 would yield a 1×1 block with zero reduced matrix
    element; Nicole trims such zero blocks, so it does not appear in H.data.
    """
    _, Op = load_space("Band", preserv=preserv)
    F, Z = Op["F"], Op["Z"]

    FdagZ = contract(F.conj().permute([1, 0, 2]), Z, axes=(1, 0)).permute([0, 2, 1])
    H12   = contract(FdagZ, F, axes=(2, 2))
    H     = H12 + H12.conj().permute([1, 0, 3, 2])

    H, _ = merge_axes(H, (0, 2), merged_tag="band", direction=Direction.IN)
    H, _ = merge_axes(H, (1, 2), merged_tag="band")
    H.regularize()

    return H


def test_hopping_two_site_spectrum_band_u1su2():
    """Two-site U(1)×SU(2) spinful hopping spectrum.

    SU(2) symmetry combines |↑⟩ and |↓⟩ into a single doublet sector.  Block
    keys are ((total_U1, 2*J_total), same) where total_U1 = N − 2 offset.
    After regularize() the reduced data values equal physical eigenvalues.

    Absent blocks:
      ((-2, 0), (-2, 0))  vacuum N=0, J=0
      ((2,  0), (2,  0))  N=4 doubly occupied, J=0
      ((0,  2), (0,  2))  N=2 triplet J=1 at E=0 (zero block, trimmed)

    Non-trivial blocks (eigenvalues from independent spin channels):
      ((-1, 1), (-1, 1))  N=1, J=½: bonding & antibonding doublets  → {−1, +1}
      (( 0, 0), ( 0, 0))  N=2, J=0: three singlets at E=−2, 0, +2   → {−2, 0, +2}
      (( 1, 1), ( 1, 1))  N=3, J=½: particle-hole of N=1            → {−1, +1}
    """
    H = _build_band_hopping_su2("U1,SU2")

    # --- absent blocks (vacuum, doubly occupied, zero triplet) ---
    for key, label in [
        (((-2, 0), (-2, 0)), "vacuum"),
        (((2,  0), (2,  0)), "doubly-occupied"),
        (((0,  2), (0,  2)), "N=2 triplet (zero block)"),
    ]:
        assert key not in H.data, f"{label} block {key} should be absent"

    # --- single-particle and single-hole doublet blocks: {−1, +1} ---
    for key in [((-1, 1), (-1, 1)), ((1, 1), (1, 1))]:
        assert key in H.data, f"Block {key} missing"
        evals = _eigvalsh_block(H, key)
        expected = torch.tensor([-1.0, 1.0], dtype=evals.dtype)
        assert torch.allclose(evals, expected, atol=1e-6), (
            f"Block {key}: expected {expected.tolist()}, got {evals.tolist()}"
        )

    # --- N=2 singlet sector: three J=0 multiplets → {−2, 0, +2} ---
    key = ((0, 0), (0, 0))
    assert key in H.data, f"Half-filling singlet block {key} missing"
    evals = _eigvalsh_block(H, key)
    expected = torch.tensor([-2.0, 0.0, 2.0], dtype=evals.dtype)
    assert torch.allclose(evals, expected, atol=1e-6), (
        f"Block {key}: expected {expected.tolist()}, got {evals.tolist()}"
    )


# ---------------------------------------------------------------------------
# Z2×SU(2) band hopping spectrum
# ---------------------------------------------------------------------------


def test_hopping_two_site_spectrum_band_z2su2():
    """Two-site Z2×SU(2) spinful hopping spectrum.

    In Z2×SU(2), |0⟩ and |↑↓⟩ share charge (0, 0) with dim=2 (like Z2×U1),
    while {|↑⟩, |↓⟩} form a doublet of charge (1, 1).  After merging two sites
    the blocks are:

    ((0, 0), (0, 0)) — 5×5 reduced matrix in the J=0 sector, combining:
        • 4 singlet multiplets from (0,0)⊗(0,0): the pair
          {|0₁0₂⟩, |0₁↑↓₂⟩, |↑↓₁0₂⟩, |↑↓₁↑↓₂⟩}
        • 1 singlet multiplet from (1,1)⊗(1,1)→(0,0): the N=2 singlet
          (|↑₁↓₂⟩ − |↓₁↑₂⟩)/√2
        Hopping connects |0₁,↑↓₂⟩ ↔ N=2 singlet ↔ |↑↓₁,0₂⟩ at eigenvalues
        {−2, 0, +2}; states |0₁0₂⟩ and |↑↓₁↑↓₂⟩ are isolated (two extra zeros).

    ((0, 2), (0, 2)) — N=2 triplet J=1 at E=0; zero block, trimmed by Nicole.

    ((1, 1), (1, 1)) — 4×4 reduced matrix: two N=1 doublets (bonding/antibonding)
        and two N=3 doublets, each channel contributing ±1 → {−1, −1, +1, +1}.
    """
    H = _build_band_hopping_su2("Z2,SU2")

    # --- absent: zero triplet block ---
    key = ((0, 2), (0, 2))
    assert key not in H.data, f"N=2 triplet block {key} should be absent (zero)"

    # --- mixed-N singlet sector: 5×5, eigenvalues {−2, 0, 0, 0, +2} ---
    key = ((0, 0), (0, 0))
    assert key in H.data, f"Block {key} missing"
    evals = _eigvalsh_block(H, key)
    expected = torch.tensor([-2.0, 0.0, 0.0, 0.0, 2.0], dtype=evals.dtype)
    assert torch.allclose(evals, expected, atol=1e-6), (
        f"Block {key}: expected {expected.tolist()}, got {evals.tolist()}"
    )

    # --- odd-parity doublet sector: 4×4, two independent hops → {−1, −1, +1, +1} ---
    key = ((1, 1), (1, 1))
    assert key in H.data, f"Block {key} missing"
    evals = _eigvalsh_block(H, key)
    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=evals.dtype)
    assert torch.allclose(evals, expected, atol=1e-6), (
        f"Block {key}: expected {expected.tolist()}, got {evals.tolist()}"
    )


# ---------------------------------------------------------------------------
# Band spin-spin operator: S₁·S₂ via Op["S"] and Op["F"] consistency
# ---------------------------------------------------------------------------


def _build_ss_band_from_S(preserv: str):
    """Build S₁·S₂ for a two-site Band system using the pre-built spin operator.

    The Band Op["S"] is the on-site rank-1 (spin-1) tensor whose reduced matrix
    element encodes the Wigner-Eckart content of the F†F bilinear projected onto
    the spin-1 sector.  The construction follows the same aux-contraction pattern
    as _build_heisenberg_su2: contract S† at site 1 with S at site 2 on the
    spin-1 auxiliary index, then merge the physical indices into a 2-index operator.
    """
    _, Op = load_space("Band", preserv=preserv)
    S = Op["S"]

    SS = contract(S.conj().permute([1, 0, 2]), S, axes=(2, 2))
    SS, _ = merge_axes(SS, (0, 2), merged_tag="ss", direction=Direction.IN)
    SS, _ = merge_axes(SS, (1, 2), merged_tag="ss")
    SS.regularize()

    return SS


def _build_ss_band_from_F(preserv: str):
    """Build S₁·S₂ for a two-site Band system using fermionic operators only.

    The on-site spin-1 operator S_from_F is extracted from the F†F bilinear by:

    1. Computing the on-site bilinear T = F†F with both spin-½ auxiliaries external.
    2. Permuting to (bra, ket, aux_F†, aux_F) and merging the two spin-½ aux into
       a combined auxiliary index with sectors (0,0) [scalar/N] and (0,2) [vector/S].
    3. Isolating the spin-1 sector (0,2) to obtain S_from_F, then rescaling by
       −1/√2 to match the RME convention of Op["S"].
    4. Contracting S†_from_F₁ with S_from_F₂ on the spin-1 aux (S₁·S₂ form).

    The resulting two-site operator is S₁·S₂, with eigenvalues matching those
    produced by _build_ss_band_from_S, which serves as the consistency check.
    """
    Spc, Op = load_space("Band", preserv=preserv)
    F = Op["F"]
    Fd = F.conj().permute([1, 0, 2])

    # On-site bilinear with both spin-½ aux external, then merge aux to get
    # a single combined index with sectors (0,0) [N-like] and (0,2) [S-like].
    T = contract(Fd, F, axes=(1, 0))
    T_perm = T.permute([0, 2, 1, 3])           # (bra, ket, aux_F†, aux_F)
    T_merged, _ = merge_axes(T_perm, (2, 3), merged_tag="_aux_")
    # T_merged: (merged_aux/OUT, ket/IN, bra/OUT)
    # Index 0 of T_merged is the merged aux with sectors (0,0) and (0,2).

    # Isolate the spin-1 (0,2) sector to obtain S_from_F ∝ Op["S"]
    aux_idx = T_merged.indices[0]
    aux_spin1 = Index(
        direction=aux_idx.direction,
        group=aux_idx.group,
        sectors=(Sector(charge=(0, 2), dim=1),),
    )
    S_data = {k: v for k, v in T_merged.data.items() if k[0] == (0, 2)}
    S_intw = {k: v for k, v in T_merged.intw.items() if k[0] == (0, 2)}
    S_from_F = Tensor(
        indices=(aux_spin1, T_merged.indices[1], T_merged.indices[2]),
        itags=T_merged.itags,
        data=S_data,
        intw=S_intw,
        dtype=T_merged.dtype,
    )

    # The merging of the two spin-½ aux indices via Clebsch-Gordan essentially
    # applies a −1/√2 · σ factor relative to Op["S"]'s RME convention.
    # Dividing by −√2 corrects this so S_from_F carries exactly the same RME
    # as Op["S"], making the two-site contraction give S₁·S₂ (not 2·S₁·S₂).
    S_from_F = S_from_F * (-1.0 / 2**0.5)

    # Contract S†_from_F at site 1 with S_from_F at site 2 on the spin-1 aux.
    # conj().permute([1,2,0]): (ket*/IN, bra*/OUT, aux*/OUT)
    Sconj = S_from_F.conj().permute([1, 2, 0])
    SS = contract(Sconj, S_from_F, axes=(2, 0))  # (ket1*/IN, bra1*/OUT, ket2/OUT, bra2/IN)
    SS = SS.permute([0, 1, 3, 2])                # → standard (ket1*/IN, bra1*/OUT, bra2/IN, ket2/OUT)
    SS, _ = merge_axes(SS, (0, 2), merged_tag="ss", direction=Direction.IN)
    SS, _ = merge_axes(SS, (1, 2), merged_tag="ss")
    SS.regularize()

    return SS   # = S₁·S₂


def _build_ss_band_from_fierz(preserv: str):
    """Build S₁·S₂ for a two-site Band system via the four-fermion Fierz identity.

    Starting from the Fierz decomposition of Pauli matrices,

        Σᵢ σⁱ_αβ σⁱ_γδ = 2 δ_αδ δ_βγ − δ_αβ δ_γδ,

    the spin–spin interaction can be written purely in terms of F and F†:

        S₁·S₂ = (1/4)(2·exchange − N₁N₂)

    where

        exchange = Σ_αβ F†_{1α} F_{1β} F†_{2β} F_{2α}

    and N = Σ_σ F†_σ F_σ is the on-site particle number.

    Construction:
    1. Build the on-site bilinear T_perm = F†F with both spin-½ aux indices
       external: (bra/IN, ket/OUT, aux_F†/IN, aux_F/OUT).
    2. Exchange term: cross-contract T1_perm and T2_perm on the aux pair with
       the σ indices swapped across sites — axes=([2,3],[3,2]).
    3. Number operator N: contract Fd and F on both the physical intermediate
       site and the spin-½ aux simultaneously.
    4. N₁N₂ outer product: extend N by inserting a dummy OUT/IN index pair, then
       contract those trivial indices to tensor-product the two on-site operators.
    5. Combine: SS = (1/4)(2·exchange − N₁N₂); merge and regularize.
    6. compress() collapses the num_components > 1 that arise from the sum of
       two tensors back to 1, making _eigvalsh_block applicable.
    """
    Spc, Op = load_space("Band", preserv=preserv)
    F = Op["F"]
    Fd = F.conj().permute([1, 0, 2])

    # On-site bilinear with both spin-½ aux indices external.
    # Fd: (ket*/IN, bra*/OUT, aux_Fd*/IN)  ;  F: (bra/IN, ket/OUT, aux_F/OUT)
    # contract on axis 1 of Fd (bra*/OUT) and axis 0 of F (bra/IN).
    # Result T: (ket*/IN, aux_Fd*/IN, ket/OUT, aux_F/OUT)
    T = contract(Fd, F, axes=(1, 0))
    T_perm = T.permute([0, 2, 1, 3])   # (bra/IN, ket/OUT, aux_Fd/IN, aux_F/OUT)

    # Exchange term: contract T1 and T2 with aux indices crossed.
    #   T1p indices 2 (aux_Fd/IN)  ↔  T2p indices 3 (aux_F/OUT)   [σ channel]
    #   T1p indices 3 (aux_F/OUT)  ↔  T2p indices 2 (aux_Fd/IN)   [σ′ channel]
    # Result: (bra1/IN, ket1/OUT, bra2/IN, ket2/OUT)
    exchange = contract(T_perm, T_perm.clone(), axes=([2, 3], [3, 2]))

    # Number operator N = Σ_σ F†_σ F_σ.
    # Contract both the physical site (Fd axis 1 ↔ F axis 0) and
    # the spin-½ aux (Fd axis 2 ↔ F axis 2) simultaneously.
    # Result N: (bra/IN, ket/OUT)
    N = contract(Fd, F, axes=([1, 2], [0, 2]))

    # N₁N₂ outer product via trivial-index trick.
    #   N1_ext: (bra1/IN, ket1/OUT, dummy/OUT)
    #   N2_ext: (dummy/IN, bra2/IN, ket2/OUT)
    # Contracting on the dummy pair gives the outer product.
    N1_ext = N.clone()
    N1_ext.insert_index(2, Direction.OUT, itag="_nn_")
    N2_ext = N.clone()
    N2_ext.insert_index(0, Direction.IN, itag="_nn_")
    N1N2 = contract(N1_ext, N2_ext, axes=(2, 0))
    # Result: (bra1/IN, ket1/OUT, bra2/IN, ket2/OUT)

    # Fierz combination: S₁·S₂ = (1/4)(2·exchange − N₁N₂)
    SS_fierz = (exchange * 2 - N1N2) * 0.25

    # Merge physical indices and normalise.
    SS_fierz, _ = merge_axes(SS_fierz, (0, 2), merged_tag="ss", direction=Direction.IN)
    SS_fierz, _ = merge_axes(SS_fierz, (1, 2), merged_tag="ss")
    # The sum 2·exchange − N₁N₂ inflates num_components (each summand contributes
    # its own weight row).  compress() performs SVD on the Bridge weight matrix
    # and retains only the linearly independent components.  It must run BEFORE
    # regularize() because regularize() for 2-index tensors reads bridge.weights[0,0]
    # alone; with num_components > 1 that first element may differ from the true
    # effective weight, leading to incorrect data scaling.
    SS_fierz.compress()
    SS_fierz.regularize()
    # exchange and N₁N₂ each populate sectors where spin is zero (e.g. empty or
    # doubly-occupied sites); after cancellation those blocks are numerically tiny
    # relative to the non-zero blocks. The residuals are a few ULPs of the
    # physical scale (~3×eps after the subtract), so we pass a slightly relaxed
    # eps to trim_zero_blocks rather than relying on the bare float64 default.
    SS_fierz.trim_zero_blocks(eps=10 * torch.finfo(torch.float64).eps)

    return SS_fierz


def test_spin_spin_band_u1su2():
    """S₁·S₂ for a two-site U(1)×SU(2) Band system: three independent routes.

    All three constructions must give the same S₁·S₂ operator:
      • Op["S"] route: direct contraction of the pre-built spin-1 tensor.
      • T_merged route: spin-1 sector extracted from the F†F bilinear and
        rescaled by −1/√2 to match the RME convention of Op["S"].
      • Fierz route: four-fermion identity S₁·S₂ = (1/4)(2·exchange − N₁N₂)
        followed by compress() + regularize().

    Block keys are ((total_U1, 2·J_total), same):

    Both Op["S"] and F are pruned to the singly-occupied sector (0,1), so SS
    only covers (0,1)⊗(0,1).

    ((0, 0), (0, 0)) — 1×1, singlet (|↑₁↓₂⟩ − |↓₁↑₂⟩)/√2: S₁·S₂ = −¾
    ((0, 2), (0, 2)) — 1×1, triplet: S₁·S₂ = +¼
    """
    SS  = _build_ss_band_from_S("U1,SU2")
    SS2 = _build_ss_band_from_F("U1,SU2")
    SS3 = _build_ss_band_from_fierz("U1,SU2")

    # --- all three produce the same non-trivial block keys ---
    assert SS.data.keys() == SS2.data.keys(), (
        f"T_merged key mismatch: SS={set(SS.data.keys())}, SS2={set(SS2.data.keys())}"
    )
    assert SS.data.keys() == SS3.data.keys(), (
        f"Fierz key mismatch: SS={set(SS.data.keys())}, SS3={set(SS3.data.keys())}"
    )

    # --- only singly-occupied sector survives after pruning ---
    for key, label in [
        (((-2, 0), (-2, 0)), "vacuum ⊗ vacuum"),
        (((-1, 1), (-1, 1)), "vacuum ⊗ singly-occ"),
        ((( 1, 1), ( 1, 1)), "singly-occ ⊗ doubly-occ"),
        ((( 2, 0), ( 2, 0)), "doubly-occ ⊗ doubly-occ"),
    ]:
        assert key not in SS.data, f"block {key} ({label}) should be absent"

    # --- analytic eigenvalues ---
    key = ((0, 0), (0, 0))
    expected_singlet = torch.tensor([-0.75], dtype=torch.float64)
    assert torch.allclose(_eigvalsh_block(SS,  key), expected_singlet, atol=1e-6), (
        f"Op['S'] block {key}: got {_eigvalsh_block(SS, key).tolist()}"
    )

    key = ((0, 2), (0, 2))
    expected_triplet = torch.tensor([0.25], dtype=torch.float64)
    assert torch.allclose(_eigvalsh_block(SS,  key), expected_triplet, atol=1e-6), (
        f"Op['S'] block {key}: got {_eigvalsh_block(SS, key).tolist()}"
    )

    # --- all three routes agree on every block ---
    for key in SS.data:
        ev = _eigvalsh_block(SS, key)
        for name, other in [("T_merged", SS2), ("Fierz", SS3)]:
            ev_other = _eigvalsh_block(other, key)
            assert torch.allclose(ev_other, ev, atol=1e-6), (
                f"{name} mismatch at {key}: got {ev_other.tolist()}, expected {ev.tolist()}"
            )


def test_spin_spin_band_z2su2():
    """S₁·S₂ for a two-site Z2×SU(2) Band system: three independent routes.

    Z2 combines the empty |0⟩ and doubly-occupied |↑↓⟩ states in a single dim=2
    sector (0,0), while the singly-occupied doublet {|↑⟩,|↓⟩} forms sector (1,1).

    Op["S"] is pruned to the singly-occupied sector (1,1), so SS covers only
    (1,1)⊗(1,1).  The F and Fierz operators cannot be restricted to this sector
    because F's blocks involve the dim=2 (0,0) sector (connecting |0⟩↔{↑,↓} and
    {↑,↓}↔|↑↓⟩), so SS2/SS3 span the full physical space and the singlet block
    carries extra zero-eigenvalue states.

    SS  (Op["S"]):         ((0,0),(0,0)) 1×1 → {−¾};   ((0,2),(0,2)) 1×1 → {+¼}
    SS2 (T_merged) / SS3 (Fierz): ((0,0),(0,0)) 5×5 → {−¾, 0, 0, 0, 0};
                                   ((0,2),(0,2)) 1×1 → {+¼}
    """
    SS  = _build_ss_band_from_S("Z2,SU2")
    SS2 = _build_ss_band_from_F("Z2,SU2")
    SS3 = _build_ss_band_from_fierz("Z2,SU2")

    # --- F and Fierz agree on the full space ---
    assert SS2.data.keys() == SS3.data.keys(), (
        f"Fierz key mismatch: SS2={set(SS2.data.keys())}, SS3={set(SS3.data.keys())}"
    )
    for key in SS2.data:
        ev2 = _eigvalsh_block(SS2, key)
        ev3 = _eigvalsh_block(SS3, key)
        assert torch.allclose(ev2, ev3, atol=1e-6), (
            f"T_merged vs Fierz mismatch at {key}: got {ev3.tolist()}, expected {ev2.tolist()}"
        )

    # --- Op["S"] covers the singly-occupied sector only ---
    key = ((1, 1), (1, 1))
    assert key not in SS.data, f"block {key} should be absent from Op['S']-derived SS"

    key = ((0, 0), (0, 0))
    expected_singlet = torch.tensor([-0.75], dtype=torch.float64)
    assert torch.allclose(_eigvalsh_block(SS, key), expected_singlet, atol=1e-6), (
        f"Op['S'] block {key}: got {_eigvalsh_block(SS, key).tolist()}"
    )

    key = ((0, 2), (0, 2))
    expected_triplet = torch.tensor([0.25], dtype=torch.float64)
    assert torch.allclose(_eigvalsh_block(SS, key), expected_triplet, atol=1e-6), (
        f"Op['S'] block {key}: got {_eigvalsh_block(SS, key).tolist()}"
    )

    # --- non-zero eigenvalues in SS2/SS3 match SS on the singly-occupied sector ---
    for key in SS.data:
        ev_ss = _eigvalsh_block(SS, key)
        for name, other in [("T_merged", SS2), ("Fierz", SS3)]:
            ev_other = _eigvalsh_block(other, key)
            ev_nz = ev_other[ev_other.abs() > 1e-8]
            assert torch.allclose(ev_nz, ev_ss, atol=1e-6), (
                f"{name} non-zero eigenvalues mismatch at {key}: "
                f"got {ev_nz.tolist()}, expected {ev_ss.tolist()}"
            )
