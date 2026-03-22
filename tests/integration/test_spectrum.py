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
