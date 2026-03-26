# Copyright (C) 2025-2026 Changkai Zhang.
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


"""Unitarity tests for U(1), SU(2), and U(1)×SU(2) tensor operations.

Tests verify unitarity properties across symmetry groups:
- Isometry orthonormality: V†V = I
- Isometry fusion-unfusion roundtrips
- Isometry linearity
- Conjugate norm consistency

Each section is ordered: U(1), SU(2), U(1)×SU(2).
"""

import math
import torch

from nicole import Direction, Tensor, Index, Sector
from nicole import contract, isometry, isometry_n, conj, identity, permute
from nicole import SU2Group, U1Group, ProductGroup
from ..utils import (
    assert_charge_neutral,
    assert_blocks_equal,
    populate_random_weights,
    assert_data_weights_equal,
    assert_physical_tensors_equal,
)


# ── Isometry orthonormality tests (V†V = I) ───────────────────────────────────
#
# isometry (2-index)

def test_contract_u1_isometry_orthonormality():
    """Test V†V = I for isometry (U(1))."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    for dir_a in [Direction.OUT, Direction.IN]:
        for dir_b in [Direction.OUT, Direction.IN]:
            idx_a = Index(dir_a, group, sectors=sectors)
            idx_b = Index(dir_b, group, sectors=sectors)

            V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

            result = contract(V.conj(), V, axes=([0, 1], [0, 1]))

            assert len(result.indices) == 2, f"Failed for directions ({dir_a}, {dir_b})"
            assert_charge_neutral(result)

            I_expected = identity(V.indices[2], itags=["f'", "f"])
            assert_blocks_equal(result, I_expected)


def test_contract_su2_isometry_orthonormality():
    """Test V†V = I for isometry (SU(2))."""
    group = SU2Group()

    for dir_a in [Direction.OUT, Direction.IN]:
        for dir_b in [Direction.OUT, Direction.IN]:
            idx_a = Index(dir_a, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
            idx_b = Index(dir_b, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))

            V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

            result = contract(V.conj(), V, axes=([0, 1], [0, 1]))

            assert len(result.indices) == 2, f"Failed for directions ({dir_a}, {dir_b})"
            assert result.intw is not None, f"Failed for directions ({dir_a}, {dir_b})"
            assert_charge_neutral(result)

            I_expected = identity(V.indices[2], itags=["f'", "f"])
            # Weights may differ by row-normalization gauge; compare physical tensors.
            assert_physical_tensors_equal(result, I_expected,
                                          msg=f"isometry orthonormality for directions ({dir_a}, {dir_b})")


def test_contract_u1su2_isometry_orthonormality():
    """Test V†V = I for isometry (U(1)×SU(2))."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    for dir_a in [Direction.OUT, Direction.IN]:
        for dir_b in [Direction.OUT, Direction.IN]:
            idx_a = Index(dir_a, group, sectors=sectors)
            idx_b = Index(dir_b, group, sectors=sectors)

            V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

            result = contract(V.conj(), V, axes=([0, 1], [0, 1]))

            assert len(result.indices) == 2, f"Failed for directions ({dir_a}, {dir_b})"
            assert result.intw is not None, f"Failed for directions ({dir_a}, {dir_b})"
            assert_charge_neutral(result)

            I_expected = identity(V.indices[2], itags=["f'", "f"])
            # Weights may differ by row-normalization gauge; compare physical tensors.
            assert_physical_tensors_equal(result, I_expected,
                                          msg=f"isometry orthonormality for directions ({dir_a}, {dir_b})")


# isometry_n

def test_contract_u1_isometry_n_orthonormality_3rd_order():
    """Test V†V = I for isometry_n with 3 input indices (U(1))."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    for dir_a in [Direction.OUT, Direction.IN]:
        for dir_b in [Direction.OUT, Direction.IN]:
            for dir_c in [Direction.OUT, Direction.IN]:
                idx_a = Index(dir_a, group, sectors=sectors)
                idx_b = Index(dir_b, group, sectors=sectors)
                idx_c = Index(dir_c, group, sectors=sectors)

                V = isometry_n([idx_a, idx_b, idx_c], itags=["a", "b", "c", "f"])

                result = contract(V.conj(), V, axes=([0, 1, 2], [0, 1, 2]))

                dirs = (dir_a, dir_b, dir_c)
                assert len(result.indices) == 2, f"Failed for directions {dirs}"
                assert_charge_neutral(result)

                I_expected = identity(V.indices[3], itags=["f'", "f"])
                assert_blocks_equal(result, I_expected)


def test_contract_su2_isometry_n_orthonormality_3rd_order():
    """Test V†V = I for isometry_n with 3 input indices (SU(2))."""
    group = SU2Group()
    sectors = (Sector(0, 1), Sector(1, 2), Sector(2, 3))

    for dir_a in [Direction.OUT, Direction.IN]:
        for dir_b in [Direction.OUT, Direction.IN]:
            for dir_c in [Direction.OUT, Direction.IN]:
                idx_a = Index(dir_a, group, sectors=sectors)
                idx_b = Index(dir_b, group, sectors=sectors)
                idx_c = Index(dir_c, group, sectors=sectors)

                V = isometry_n([idx_a, idx_b, idx_c], itags=["a", "b", "c", "f"])

                result = contract(V.conj(), V, axes=([0, 1, 2], [0, 1, 2]))

                dirs = (dir_a, dir_b, dir_c)
                assert len(result.indices) == 2, f"Failed for directions {dirs}"
                assert result.intw is not None, f"Failed for directions {dirs}"
                assert_charge_neutral(result)

                I_expected = identity(V.indices[3], itags=["f'", "f"])
                assert_physical_tensors_equal(result, I_expected,
                                             msg=f"isometry_n orthonormality for directions {dirs}")


def test_contract_u1su2_isometry_n_orthonormality_3rd_order():
    """Test V†V = I for isometry_n with 3 input indices (U(1)×SU(2))."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    for dir_a in [Direction.OUT, Direction.IN]:
        for dir_b in [Direction.OUT, Direction.IN]:
            for dir_c in [Direction.OUT, Direction.IN]:
                idx_a = Index(dir_a, group, sectors=sectors)
                idx_b = Index(dir_b, group, sectors=sectors)
                idx_c = Index(dir_c, group, sectors=sectors)

                V = isometry_n([idx_a, idx_b, idx_c], itags=["a", "b", "c", "f"])

                result = contract(V.conj(), V, axes=([0, 1, 2], [0, 1, 2]))

                dirs = (dir_a, dir_b, dir_c)
                assert len(result.indices) == 2, f"Failed for directions {dirs}"
                assert result.intw is not None, f"Failed for directions {dirs}"
                assert_charge_neutral(result)

                I_expected = identity(V.indices[3], itags=["f'", "f"])
                assert_physical_tensors_equal(result, I_expected,
                                             msg=f"isometry_n orthonormality for directions {dirs}")


def test_contract_u1_isometry_n_orthonormality_4th_order():
    """Test V†V = I for isometry_n with 4 input indices (U(1))."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    dir_combos = [
        (Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT),
        (Direction.OUT, Direction.IN,  Direction.OUT, Direction.IN),
        (Direction.IN,  Direction.OUT, Direction.IN,  Direction.OUT),
        (Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN),
    ]

    for dir_a, dir_b, dir_c, dir_d in dir_combos:
        idx_a = Index(dir_a, group, sectors=sectors)
        idx_b = Index(dir_b, group, sectors=sectors)
        idx_c = Index(dir_c, group, sectors=sectors)
        idx_d = Index(dir_d, group, sectors=sectors)

        V = isometry_n([idx_a, idx_b, idx_c, idx_d], itags=["a", "b", "c", "d", "f"])

        result = contract(V.conj(), V, axes=([0, 1, 2, 3], [0, 1, 2, 3]))

        dirs = (dir_a, dir_b, dir_c, dir_d)
        assert len(result.indices) == 2, f"Failed for directions {dirs}"
        assert_charge_neutral(result)

        I_expected = identity(V.indices[4], itags=["f'", "f"])
        assert_blocks_equal(result, I_expected)


def test_contract_su2_isometry_n_orthonormality_4th_order():
    """Test V†V = I for isometry_n with 4 input indices (SU(2))."""
    group = SU2Group()
    sectors = (Sector(0, 1), Sector(1, 2))  # 2 sectors only; 3 sectors is too slow at 4th order

    dir_combos = [
        (Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT),
        (Direction.OUT, Direction.IN,  Direction.OUT, Direction.IN),
        (Direction.IN,  Direction.OUT, Direction.IN,  Direction.OUT),
        (Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN),
    ]

    for dir_a, dir_b, dir_c, dir_d in dir_combos:
        idx_a = Index(dir_a, group, sectors=sectors)
        idx_b = Index(dir_b, group, sectors=sectors)
        idx_c = Index(dir_c, group, sectors=sectors)
        idx_d = Index(dir_d, group, sectors=sectors)

        V = isometry_n([idx_a, idx_b, idx_c, idx_d], itags=["a", "b", "c", "d", "f"])

        result = contract(V.conj(), V, axes=([0, 1, 2, 3], [0, 1, 2, 3]))

        dirs = (dir_a, dir_b, dir_c, dir_d)
        assert len(result.indices) == 2, f"Failed for directions {dirs}"
        assert result.intw is not None, f"Failed for directions {dirs}"
        assert_charge_neutral(result)

        I_expected = identity(V.indices[4], itags=["f'", "f"])
        assert_physical_tensors_equal(result, I_expected,
                                     msg=f"isometry_n orthonormality for directions {dirs}")


def test_contract_u1su2_isometry_n_orthonormality_4th_order():
    """Test V†V = I for isometry_n with 4 input indices (U(1)×SU(2))."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1))  # 2 sectors only; 3 sectors is too slow at 4th order

    dir_combos = [
        (Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT),
        (Direction.OUT, Direction.IN,  Direction.OUT, Direction.IN),
        (Direction.IN,  Direction.OUT, Direction.IN,  Direction.OUT),
        (Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN),
    ]

    for dir_a, dir_b, dir_c, dir_d in dir_combos:
        idx_a = Index(dir_a, group, sectors=sectors)
        idx_b = Index(dir_b, group, sectors=sectors)
        idx_c = Index(dir_c, group, sectors=sectors)
        idx_d = Index(dir_d, group, sectors=sectors)

        V = isometry_n([idx_a, idx_b, idx_c, idx_d], itags=["a", "b", "c", "d", "f"])

        result = contract(V.conj(), V, axes=([0, 1, 2, 3], [0, 1, 2, 3]))

        dirs = (dir_a, dir_b, dir_c, dir_d)
        assert len(result.indices) == 2, f"Failed for directions {dirs}"
        assert result.intw is not None, f"Failed for directions {dirs}"
        assert_charge_neutral(result)

        I_expected = identity(V.indices[4], itags=["f'", "f"])
        assert_physical_tensors_equal(result, I_expected,
                                     msg=f"isometry_n orthonormality for directions {dirs}")


def test_contract_u1_isometry_n_orthonormality_5th_order():
    """Test V†V = I for isometry_n with 5 input indices (U(1))."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    dir_combos = [
        (Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT),
        (Direction.OUT, Direction.IN,  Direction.OUT, Direction.IN,  Direction.OUT),
        (Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN),
    ]

    for dirs in dir_combos:
        indices = [Index(d, group, sectors=sectors) for d in dirs]

        V = isometry_n(indices, itags=["a", "b", "c", "d", "e", "f"])

        result = contract(V.conj(), V, axes=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))

        assert len(result.indices) == 2, f"Failed for directions {dirs}"
        assert_charge_neutral(result)

        I_expected = identity(V.indices[5], itags=["f'", "f"])
        assert_blocks_equal(result, I_expected)


def test_contract_su2_isometry_n_orthonormality_5th_order():
    """Test V†V = I for isometry_n with 5 input indices (SU(2))."""
    group = SU2Group()
    sectors = (Sector(0, 1), Sector(1, 2))  # 2 sectors only; 3 sectors is too slow at 5th order

    dir_combos = [
        (Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT),
        (Direction.OUT, Direction.IN,  Direction.OUT, Direction.IN,  Direction.OUT),
        (Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN),
    ]

    for dirs in dir_combos:
        indices = [Index(d, group, sectors=sectors) for d in dirs]

        V = isometry_n(indices, itags=["a", "b", "c", "d", "e", "f"])

        result = contract(V.conj(), V, axes=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))

        assert len(result.indices) == 2, f"Failed for directions {dirs}"
        assert result.intw is not None, f"Failed for directions {dirs}"
        assert_charge_neutral(result)

        I_expected = identity(V.indices[5], itags=["f'", "f"])
        assert_physical_tensors_equal(result, I_expected,
                                     msg=f"isometry_n orthonormality for directions {dirs}")


def test_contract_u1su2_isometry_n_orthonormality_5th_order():
    """Test V†V = I for isometry_n with 5 input indices (U(1)×SU(2))."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1))  # 2 sectors only; 3 sectors is too slow at 5th order

    dir_combos = [
        (Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT),
        (Direction.OUT, Direction.IN,  Direction.OUT, Direction.IN,  Direction.OUT),
        (Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN),
    ]

    for dirs in dir_combos:
        indices = [Index(d, group, sectors=sectors) for d in dirs]

        V = isometry_n(indices, itags=["a", "b", "c", "d", "e", "f"])

        result = contract(V.conj(), V, axes=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))

        assert len(result.indices) == 2, f"Failed for directions {dirs}"
        assert result.intw is not None, f"Failed for directions {dirs}"
        assert_charge_neutral(result)

        I_expected = identity(V.indices[5], itags=["f'", "f"])
        assert_physical_tensors_equal(result, I_expected,
                                     msg=f"isometry_n orthonormality for directions {dirs}")


# ── Isometry fusion-unfusion tests ────────────────────────────────────────────
#
# isometry (2-index)

def test_contract_u1_isometry_fusion_unfusion_roundtrip_basic():
    """Test fusion with isometry then unfusion preserves tensor (U(1), 3rd order)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c], seed=6000, itags=["a", "b", "c"])

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

    fused = contract(V.conj(), A, axes=([0, 1], [0, 1]))

    assert len(fused.indices) == 2
    assert_charge_neutral(fused)

    unfused = contract(V, fused, axes=(2, 0))

    assert len(unfused.indices) == 3
    assert_charge_neutral(unfused)

    assert_blocks_equal(A, unfused)


def test_contract_su2_isometry_fusion_unfusion_roundtrip_basic():
    """Test fusion with isometry then unfusion preserves tensor (SU(2), 3rd order)."""
    group = SU2Group()

    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))

    A = Tensor.random([idx_a, idx_b, idx_c], seed=3000, itags=["a", "b", "c"])
    populate_random_weights(A, seed=3001)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

    V_conj = conj(V)
    fused = contract(V_conj, A, axes=([0, 1], [0, 1]))

    assert len(fused.indices) == 2
    assert fused.intw is not None
    assert_charge_neutral(fused)

    unfused = contract(V, fused, axes=(2, 0))

    assert len(unfused.indices) == 3
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="fusion-unfusion roundtrip")


def test_contract_u1su2_isometry_fusion_unfusion_roundtrip_basic():
    """Test fusion with isometry then unfusion preserves tensor (U(1)×SU(2), 3rd order)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c], seed=7000, itags=["a", "b", "c"])
    populate_random_weights(A, seed=7001)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

    fused = contract(V.conj(), A, axes=([0, 1], [0, 1]))

    assert len(fused.indices) == 2
    assert fused.intw is not None
    assert_charge_neutral(fused)

    unfused = contract(V, fused, axes=(2, 0))

    assert len(unfused.indices) == 3
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="fusion-unfusion roundtrip")


def test_contract_u1_isometry_fusion_unfusion_roundtrip_4th_order():
    """Test fusion with isometry then unfusion preserves tensor (U(1), 4th order)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=6010, itags=["a", "b", "c", "d"])

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

    fused = contract(V.conj(), A, axes=([0, 1], [0, 1]))

    assert len(fused.indices) == 3
    assert_charge_neutral(fused)

    unfused = contract(V, fused, axes=(2, 0))

    assert len(unfused.indices) == 4
    assert_charge_neutral(unfused)

    assert_blocks_equal(A, unfused)


def test_contract_su2_isometry_fusion_unfusion_roundtrip_4th_order():
    """Test fusion with isometry then unfusion preserves tensor (SU(2), 4th order)."""
    group = SU2Group()

    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=3000, itags=["a", "b", "c", "d"])
    populate_random_weights(A, seed=3001)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

    V_conj = conj(V)
    fused = contract(V_conj, A, axes=([0, 1], [0, 1]))

    assert len(fused.indices) == 3
    assert fused.intw is not None
    assert_charge_neutral(fused)

    unfused = contract(V, fused, axes=(2, 0))

    assert len(unfused.indices) == 4
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="fusion-unfusion roundtrip")


def test_contract_u1su2_isometry_fusion_unfusion_roundtrip_4th_order():
    """Test fusion with isometry then unfusion preserves tensor (U(1)×SU(2), 4th order)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=7010, itags=["a", "b", "c", "d"])
    populate_random_weights(A, seed=7011)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

    fused = contract(V.conj(), A, axes=([0, 1], [0, 1]))

    assert len(fused.indices) == 3
    assert fused.intw is not None
    assert_charge_neutral(fused)

    unfused = contract(V, fused, axes=(2, 0))

    assert len(unfused.indices) == 4
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="fusion-unfusion roundtrip")


def test_contract_u1_isometry_fusion_unfusion_roundtrip_5th_order():
    """Test fusion with isometry then unfusion preserves tensor (U(1), 5th order)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)
    idx_e = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=6020,
                      itags=["a", "b", "c", "d", "e"])

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

    fused = contract(V.conj(), A, axes=([0, 1], [0, 1]))

    assert len(fused.indices) == 4
    assert_charge_neutral(fused)

    unfused = contract(V, fused, axes=(2, 0))

    assert len(unfused.indices) == 5
    assert_charge_neutral(unfused)

    assert_blocks_equal(A, unfused)


def test_contract_su2_isometry_fusion_unfusion_roundtrip_5th_order():
    """Test fusion with isometry then unfusion preserves tensor (SU(2), 5th order)."""
    group = SU2Group()

    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_e = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=3010,
                      itags=["a", "b", "c", "d", "e"])
    populate_random_weights(A, seed=3011)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

    V_conj = conj(V)
    fused = contract(V_conj, A, axes=([0, 1], [0, 1]))

    assert len(fused.indices) == 4
    assert fused.intw is not None
    assert_charge_neutral(fused)

    unfused = contract(V, fused, axes=(2, 0))

    assert len(unfused.indices) == 5
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="fusion-unfusion roundtrip")


def test_contract_u1su2_isometry_fusion_unfusion_roundtrip_5th_order():
    """Test fusion with isometry then unfusion preserves tensor (U(1)×SU(2), 5th order)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)
    idx_e = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=7020,
                      itags=["a", "b", "c", "d", "e"])
    populate_random_weights(A, seed=7021)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])

    fused = contract(V.conj(), A, axes=([0, 1], [0, 1]))

    assert len(fused.indices) == 4
    assert fused.intw is not None
    assert_charge_neutral(fused)

    unfused = contract(V, fused, axes=(2, 0))

    assert len(unfused.indices) == 5
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="fusion-unfusion roundtrip")


# isometry_n
#
# Note: for isometry_n, V's unfused indices are pre-flipped relative to the inputs,
# so V fuses (contract V with A) and V† unfuses (contract V.conj() with fused).

def test_contract_u1_isometry_n_fusion_unfusion_roundtrip_3rd_order():
    """Test fusion with isometry_n then unfusion preserves tensor (U(1), 3 fused indices)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.OUT, group, sectors=sectors)
    idx_d = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=6100, itags=["a", "b", "c", "d"])

    V = isometry_n([idx_a, idx_b, idx_c], itags=["a", "b", "c", "f"])

    fused = contract(V, A, axes=([0, 1, 2], [0, 1, 2]))

    assert len(fused.indices) == 2
    assert_charge_neutral(fused)

    unfused = contract(V.conj(), fused, axes=(3, 0))

    assert len(unfused.indices) == 4
    assert_charge_neutral(unfused)

    assert_blocks_equal(A, unfused)


def test_contract_su2_isometry_n_fusion_unfusion_roundtrip_3rd_order():
    """Test fusion with isometry_n then unfusion preserves tensor (SU(2), 3 fused indices)."""
    group = SU2Group()
    sectors = (Sector(0, 1), Sector(1, 2), Sector(2, 3))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.OUT, group, sectors=sectors)
    idx_d = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=5000, itags=["a", "b", "c", "d"])
    populate_random_weights(A, seed=5001)

    V = isometry_n([idx_a, idx_b, idx_c], itags=["a", "b", "c", "f"])

    fused = contract(V, A, axes=([0, 1, 2], [0, 1, 2]))

    assert len(fused.indices) == 2
    assert fused.intw is not None
    assert_charge_neutral(fused)

    unfused = contract(V.conj(), fused, axes=(3, 0))

    assert len(unfused.indices) == 4
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="isometry_n fusion-unfusion roundtrip")


def test_contract_u1su2_isometry_n_fusion_unfusion_roundtrip_3rd_order():
    """Test fusion with isometry_n then unfusion preserves tensor (U(1)×SU(2), 3 fused indices)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.OUT, group, sectors=sectors)
    idx_d = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=7100, itags=["a", "b", "c", "d"])
    populate_random_weights(A, seed=7101)

    V = isometry_n([idx_a, idx_b, idx_c], itags=["a", "b", "c", "f"])

    fused = contract(V, A, axes=([0, 1, 2], [0, 1, 2]))

    assert len(fused.indices) == 2
    assert fused.intw is not None
    assert_charge_neutral(fused)

    unfused = contract(V.conj(), fused, axes=(3, 0))

    assert len(unfused.indices) == 4
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="isometry_n fusion-unfusion roundtrip")


def test_contract_u1_isometry_n_fusion_unfusion_roundtrip_4th_order():
    """Test fusion with isometry_n then unfusion preserves tensor (U(1), 4 fused indices)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.OUT, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)
    idx_e = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=6110,
                      itags=["a", "b", "c", "d", "e"])

    V = isometry_n([idx_a, idx_b, idx_c, idx_d], itags=["a", "b", "c", "d", "f"])

    fused = contract(V, A, axes=([0, 1, 2, 3], [0, 1, 2, 3]))

    assert len(fused.indices) == 2
    assert_charge_neutral(fused)

    unfused = contract(V.conj(), fused, axes=(4, 0))

    assert len(unfused.indices) == 5
    assert_charge_neutral(unfused)

    assert_blocks_equal(A, unfused)


def test_contract_su2_isometry_n_fusion_unfusion_roundtrip_4th_order():
    """Test fusion with isometry_n then unfusion preserves tensor (SU(2), 4 fused indices)."""
    group = SU2Group()
    sectors = (Sector(0, 1), Sector(1, 2), Sector(2, 3))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.OUT, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)
    idx_e = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=5010,
                      itags=["a", "b", "c", "d", "e"])
    populate_random_weights(A, seed=5011)

    V = isometry_n([idx_a, idx_b, idx_c, idx_d], itags=["a", "b", "c", "d", "f"])

    fused = contract(V, A, axes=([0, 1, 2, 3], [0, 1, 2, 3]))

    assert len(fused.indices) == 2
    assert fused.intw is not None
    assert_charge_neutral(fused)

    unfused = contract(V.conj(), fused, axes=(4, 0))

    assert len(unfused.indices) == 5
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="isometry_n fusion-unfusion roundtrip")


def test_contract_u1su2_isometry_n_fusion_unfusion_roundtrip_4th_order():
    """Test fusion with isometry_n then unfusion preserves tensor (U(1)×SU(2), 4 fused indices)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.OUT, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)
    idx_e = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=7110,
                      itags=["a", "b", "c", "d", "e"])
    populate_random_weights(A, seed=7111)

    V = isometry_n([idx_a, idx_b, idx_c, idx_d], itags=["a", "b", "c", "d", "f"])

    fused = contract(V, A, axes=([0, 1, 2, 3], [0, 1, 2, 3]))

    assert len(fused.indices) == 2
    assert fused.intw is not None
    assert_charge_neutral(fused)

    unfused = contract(V.conj(), fused, axes=(4, 0))

    assert len(unfused.indices) == 5
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="isometry_n fusion-unfusion roundtrip")


# ── Isometry linearity tests ──────────────────────────────────────────────────

def test_contract_u1_isometry_linearity_basic():
    """Test isometry linearity: V⊗(A+B) = V⊗A + V⊗B (U(1), 3rd order)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c], seed=6200, itags=["f", "c"])
    B = Tensor.random([f_index.flip(), idx_c], seed=6201, itags=["f", "c"])

    result_combined = contract(V, A + B, axes=(2, 0))

    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB

    assert_blocks_equal(result_combined, result_separate)


def test_contract_su2_isometry_linearity_basic():
    """Test isometry linearity: V⊗(A+B) = V⊗A + V⊗B (SU(2), 3rd order)."""
    group = SU2Group()

    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c], seed=3100, itags=["f", "c"])
    B = Tensor.random([f_index.flip(), idx_c], seed=3101, itags=["f", "c"])
    populate_random_weights(A, seed=3102)
    populate_random_weights(B, seed=3103)

    result_combined = contract(V, A + B, axes=(2, 0))

    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB

    assert_data_weights_equal(result_combined, result_separate, msg="isometry linearity")


def test_contract_u1su2_isometry_linearity_basic():
    """Test isometry linearity: V⊗(A+B) = V⊗A + V⊗B (U(1)×SU(2), 3rd order)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c], seed=7200, itags=["f", "c"])
    B = Tensor.random([f_index.flip(), idx_c], seed=7201, itags=["f", "c"])
    populate_random_weights(A, seed=7202)
    populate_random_weights(B, seed=7203)

    result_combined = contract(V, A + B, axes=(2, 0))

    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB

    assert_data_weights_equal(result_combined, result_separate, msg="isometry linearity")


def test_contract_u1_isometry_linearity_4th_order():
    """Test isometry linearity: V⊗(A+B) = V⊗A + V⊗B (U(1), 4th order)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d], seed=6210, itags=["f", "c", "d"])
    B = Tensor.random([f_index.flip(), idx_c, idx_d], seed=6211, itags=["f", "c", "d"])

    result_combined = contract(V, A + B, axes=(2, 0))

    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB

    assert_blocks_equal(result_combined, result_separate)


def test_contract_su2_isometry_linearity_4th_order():
    """Test isometry linearity: V⊗(A+B) = V⊗A + V⊗B (SU(2), 4th order)."""
    group = SU2Group()

    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d], seed=3110, itags=["f", "c", "d"])
    B = Tensor.random([f_index.flip(), idx_c, idx_d], seed=3111, itags=["f", "c", "d"])
    populate_random_weights(A, seed=3112)
    populate_random_weights(B, seed=3113)

    result_combined = contract(V, A + B, axes=(2, 0))

    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB

    assert_physical_tensors_equal(result_combined, result_separate, msg="isometry linearity")


def test_contract_u1su2_isometry_linearity_4th_order():
    """Test isometry linearity: V⊗(A+B) = V⊗A + V⊗B (U(1)×SU(2), 4th order)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d], seed=7210, itags=["f", "c", "d"])
    B = Tensor.random([f_index.flip(), idx_c, idx_d], seed=7211, itags=["f", "c", "d"])
    populate_random_weights(A, seed=7212)
    populate_random_weights(B, seed=7213)

    result_combined = contract(V, A + B, axes=(2, 0))

    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB

    assert_physical_tensors_equal(result_combined, result_separate, msg="isometry linearity")


def test_contract_u1_isometry_linearity_5th_order():
    """Test isometry linearity: V⊗(A+B) = V⊗A + V⊗B (U(1), 5th order)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)
    idx_e = Index(Direction.IN, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=6220,
                      itags=["f", "c", "d", "e"])
    B = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=6221,
                      itags=["f", "c", "d", "e"])

    result_combined = contract(V, A + B, axes=(2, 0))

    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB

    assert_blocks_equal(result_combined, result_separate)


def test_contract_su2_isometry_linearity_5th_order():
    """Test isometry linearity: V⊗(A+B) = V⊗A + V⊗B (SU(2), 5th order)."""
    group = SU2Group()

    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_e = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=3120,
                      itags=["f", "c", "d", "e"])
    B = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=3121,
                      itags=["f", "c", "d", "e"])
    populate_random_weights(A, seed=3122)
    populate_random_weights(B, seed=3123)

    result_combined = contract(V, A + B, axes=(2, 0))

    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB

    assert_physical_tensors_equal(result_combined, result_separate, msg="isometry linearity")


def test_contract_u1su2_isometry_linearity_5th_order():
    """Test isometry linearity: V⊗(A+B) = V⊗A + V⊗B (U(1)×SU(2), 5th order)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)
    idx_e = Index(Direction.IN, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=7220,
                      itags=["f", "c", "d", "e"])
    B = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=7221,
                      itags=["f", "c", "d", "e"])
    populate_random_weights(A, seed=7222)
    populate_random_weights(B, seed=7223)

    result_combined = contract(V, A + B, axes=(2, 0))

    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB

    assert_physical_tensors_equal(result_combined, result_separate, msg="isometry linearity")


# ── Isometry scalar multiplication tests ─────────────────────────────────────

def test_contract_u1_isometry_scalar_multiplication_basic():
    """Test isometry contraction commutes with scalar multiplication (U(1), 3rd order)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c], seed=6300, itags=["f", "c"])

    alpha = 2.5
    result1 = contract(V, alpha * A, axes=(2, 0))
    result2 = alpha * contract(V, A, axes=(2, 0))

    assert_blocks_equal(result1, result2)


def test_contract_su2_isometry_scalar_multiplication_basic():
    """Test isometry contraction commutes with scalar multiplication (SU(2), 3rd order)."""
    group = SU2Group()

    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c], seed=3200, itags=["f", "c"])
    populate_random_weights(A, seed=3201)

    alpha = 2.5
    result1 = contract(V, alpha * A, axes=(2, 0))
    result2 = alpha * contract(V, A, axes=(2, 0))

    assert_data_weights_equal(result1, result2, msg="isometry scalar multiplication")


def test_contract_u1su2_isometry_scalar_multiplication_basic():
    """Test isometry contraction commutes with scalar multiplication (U(1)×SU(2), 3rd order)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c], seed=7300, itags=["f", "c"])
    populate_random_weights(A, seed=7301)

    alpha = 2.5
    result1 = contract(V, alpha * A, axes=(2, 0))
    result2 = alpha * contract(V, A, axes=(2, 0))

    assert_data_weights_equal(result1, result2, msg="isometry scalar multiplication")


def test_contract_u1_isometry_scalar_multiplication_4th_order():
    """Test isometry contraction commutes with scalar multiplication (U(1), 4th order)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d], seed=6310, itags=["f", "c", "d"])

    alpha = 2.5
    result1 = contract(V, alpha * A, axes=(2, 0))
    result2 = alpha * contract(V, A, axes=(2, 0))

    assert_blocks_equal(result1, result2)


def test_contract_su2_isometry_scalar_multiplication_4th_order():
    """Test isometry contraction commutes with scalar multiplication (SU(2), 4th order)."""
    group = SU2Group()

    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d], seed=3210, itags=["f", "c", "d"])
    populate_random_weights(A, seed=3211)

    alpha = 2.5
    result1 = contract(V, alpha * A, axes=(2, 0))
    result2 = alpha * contract(V, A, axes=(2, 0))

    assert_data_weights_equal(result1, result2, msg="isometry scalar multiplication")


def test_contract_u1su2_isometry_scalar_multiplication_4th_order():
    """Test isometry contraction commutes with scalar multiplication (U(1)×SU(2), 4th order)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d], seed=7310, itags=["f", "c", "d"])
    populate_random_weights(A, seed=7311)

    alpha = 2.5
    result1 = contract(V, alpha * A, axes=(2, 0))
    result2 = alpha * contract(V, A, axes=(2, 0))

    assert_data_weights_equal(result1, result2, msg="isometry scalar multiplication")


def test_contract_u1_isometry_scalar_multiplication_5th_order():
    """Test isometry contraction commutes with scalar multiplication (U(1), 5th order)."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 2), Sector(-1, 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)
    idx_e = Index(Direction.IN, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=6320,
                      itags=["f", "c", "d", "e"])

    alpha = 2.5
    result1 = contract(V, alpha * A, axes=(2, 0))
    result2 = alpha * contract(V, A, axes=(2, 0))

    assert_blocks_equal(result1, result2)


def test_contract_su2_isometry_scalar_multiplication_5th_order():
    """Test isometry contraction commutes with scalar multiplication (SU(2), 5th order)."""
    group = SU2Group()

    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_e = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=3220,
                      itags=["f", "c", "d", "e"])
    populate_random_weights(A, seed=3221)

    alpha = 2.5
    result1 = contract(V, alpha * A, axes=(2, 0))
    result2 = alpha * contract(V, A, axes=(2, 0))

    assert_data_weights_equal(result1, result2, msg="isometry scalar multiplication")


def test_contract_u1su2_isometry_scalar_multiplication_5th_order():
    """Test isometry contraction commutes with scalar multiplication (U(1)×SU(2), 5th order)."""
    group = ProductGroup([U1Group(), SU2Group()])
    sectors = (Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.IN, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)
    idx_e = Index(Direction.IN, group, sectors=sectors)

    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]

    A = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=7320,
                      itags=["f", "c", "d", "e"])
    populate_random_weights(A, seed=7321)

    alpha = 2.5
    result1 = contract(V, alpha * A, axes=(2, 0))
    result2 = alpha * contract(V, A, axes=(2, 0))

    assert_data_weights_equal(result1, result2, msg="isometry scalar multiplication")


# ── Conjugate norm tests ──────────────────────────────────────────────────────

def test_contract_u1_conjugate_norm_matrix():
    """Test ⟨A|A⟩ = ‖A‖² (U(1), 2nd order)."""
    group = U1Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))

    A = Tensor.random([idx1, idx2], seed=6400, dtype=torch.float64, itags=["i", "j"])

    scalar_AA = contract(conj(A), A, axes=([0, 1], [0, 1]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 2nd order tensor: {value_AA} vs {norm_sq}"


def test_contract_su2_conjugate_norm_matrix():
    """Test ⟨A|A⟩ = ‖A‖² (SU(2), 2nd order)."""
    group = SU2Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))

    A = Tensor.random([idx1, idx2], seed=4000, dtype=torch.float64, itags=["i", "j"])
    populate_random_weights(A, seed=4001)

    A_conj = conj(A)
    scalar_AA = contract(A_conj, A, axes=([0, 1], [0, 1]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 2nd order tensor: {value_AA} vs {norm_sq}"


def test_contract_u1su2_conjugate_norm_matrix():
    """Test ⟨A|A⟩ = ‖A‖² (U(1)×SU(2), 2nd order)."""
    group = ProductGroup([U1Group(), SU2Group()])

    idx1 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx2 = Index(Direction.IN, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))

    A = Tensor.random([idx1, idx2], seed=7400, dtype=torch.float64, itags=["i", "j"])
    populate_random_weights(A, seed=7401)

    scalar_AA = contract(conj(A), A, axes=([0, 1], [0, 1]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 2nd order tensor: {value_AA} vs {norm_sq}"


def test_contract_u1_conjugate_norm_basic():
    """Test ⟨A|A⟩ = ‖A‖² (U(1), 3rd order)."""
    group = U1Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx1, idx2, idx3], seed=6410, dtype=torch.float64, itags=["i", "j", "k"])

    scalar_AA = contract(conj(A), A, axes=([0, 1, 2], [0, 1, 2]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 3rd order tensor: {value_AA} vs {norm_sq}"


def test_contract_su2_conjugate_norm_basic():
    """Test ⟨A|A⟩ = ‖A‖² (SU(2), 3rd order)."""
    group = SU2Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx1, idx2, idx3], seed=4010, dtype=torch.float64, itags=["i", "j", "k"])
    populate_random_weights(A, seed=4011)

    A_conj = conj(A)
    scalar_AA = contract(A_conj, A, axes=([0, 1, 2], [0, 1, 2]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 3rd order tensor: {value_AA} vs {norm_sq}"


def test_contract_u1su2_conjugate_norm_basic():
    """Test ⟨A|A⟩ = ‖A‖² (U(1)×SU(2), 3rd order)."""
    group = ProductGroup([U1Group(), SU2Group()])

    idx1 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx2 = Index(Direction.IN, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx3 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 1), Sector((1, 2), 1)))

    A = Tensor.random([idx1, idx2, idx3], seed=7410, dtype=torch.float64, itags=["i", "j", "k"])
    populate_random_weights(A, seed=7411)

    scalar_AA = contract(conj(A), A, axes=([0, 1, 2], [0, 1, 2]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 3rd order tensor: {value_AA} vs {norm_sq}"


def test_contract_u1_conjugate_norm_4th_order():
    """Test ⟨A|A⟩ = ‖A‖² (U(1), 4th order)."""
    group = U1Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx1, idx2, idx3, idx4], seed=6420, dtype=torch.float64,
                      itags=["i", "j", "k", "l"])

    scalar_AA = contract(conj(A), A, axes=([0, 1, 2, 3], [0, 1, 2, 3]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 4th order tensor: {value_AA} vs {norm_sq}"


def test_contract_su2_conjugate_norm_4th_order():
    """Test ⟨A|A⟩ = ‖A‖² (SU(2), 4th order)."""
    group = SU2Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx1, idx2, idx3, idx4], seed=4020, dtype=torch.float64,
                      itags=["i", "j", "k", "l"])
    populate_random_weights(A, seed=4021)

    A_conj = conj(A)
    scalar_AA = contract(A_conj, A, axes=([0, 1, 2, 3], [0, 1, 2, 3]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 4th order tensor: {value_AA} vs {norm_sq}"


def test_contract_u1su2_conjugate_norm_4th_order():
    """Test ⟨A|A⟩ = ‖A‖² (U(1)×SU(2), 4th order)."""
    group = ProductGroup([U1Group(), SU2Group()])

    idx1 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx2 = Index(Direction.IN, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx3 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 1), Sector((1, 2), 1)))
    idx4 = Index(Direction.IN, group,
                 sectors=(Sector((0, 0), 1), Sector((1, 2), 1)))

    A = Tensor.random([idx1, idx2, idx3, idx4], seed=7420, dtype=torch.float64,
                      itags=["i", "j", "k", "l"])
    populate_random_weights(A, seed=7421)

    scalar_AA = contract(conj(A), A, axes=([0, 1, 2, 3], [0, 1, 2, 3]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 4th order tensor: {value_AA} vs {norm_sq}"


def test_contract_u1_conjugate_norm_5th_order():
    """Test ⟨A|A⟩ = ‖A‖² (U(1), 5th order)."""
    group = U1Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(-1, 1)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx5 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))

    A = Tensor.random([idx1, idx2, idx3, idx4, idx5], seed=6430, dtype=torch.float64,
                      itags=["i", "j", "k", "l", "m"])

    scalar_AA = contract(conj(A), A, axes=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 5th order tensor: {value_AA} vs {norm_sq}"


def test_contract_su2_conjugate_norm_5th_order():
    """Test ⟨A|A⟩ = ‖A‖² (SU(2), 5th order)."""
    group = SU2Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx5 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 4)))

    A = Tensor.random([idx1, idx2, idx3, idx4, idx5], seed=4030, dtype=torch.float64,
                      itags=["i", "j", "k", "l", "m"])
    populate_random_weights(A, seed=4031)

    A_conj = conj(A)
    scalar_AA = contract(A_conj, A, axes=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 5th order tensor: {value_AA} vs {norm_sq}"


def test_contract_u1su2_conjugate_norm_5th_order():
    """Test ⟨A|A⟩ = ‖A‖² (U(1)×SU(2), 5th order)."""
    group = ProductGroup([U1Group(), SU2Group()])

    idx1 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx2 = Index(Direction.IN, group,
                 sectors=(Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx3 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1)))
    idx4 = Index(Direction.IN, group,
                 sectors=(Sector((0, 0), 1), Sector((1, 2), 1)))
    idx5 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 3), Sector((1, 2), 1)))

    A = Tensor.random([idx1, idx2, idx3, idx4, idx5], seed=7430, dtype=torch.float64,
                      itags=["i", "j", "k", "l", "m"])
    populate_random_weights(A, seed=7431)

    scalar_AA = contract(conj(A), A, axes=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2

    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 5th order tensor: {value_AA} vs {norm_sq}"


# ── Conjugate norm with permuted contraction axes ─────────────────────────────
#
# Verify ⟨perm(A†)|A⟩ = ‖A‖² for non-trivial permutations of conj(A).
# After permuting conj(A) by `perm`, the full contraction uses
# axes=(range(n), perm), so axesA ≠ axesB — this exercises the R-symbol
# correction in the scalar branch of `contract`.
#
# Permutations used:
#   3rd order : perm=[2,0,1]      → axes=([0,1,2], [2,0,1])
#   4th order : perm=[2,3,0,1]    → axes=([0,1,2,3], [2,3,0,1])
#   5th order : perm=[2,4,0,3,1]  → axes=([0,1,2,3,4], [2,4,0,3,1])

def test_contract_u1_conj_permuted_norm_3rd_order():
    """Test ⟨perm(A†)|A⟩ = ‖A‖² with non-canonical axes (U(1), 3rd order)."""
    group = U1Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx1, idx2, idx3], seed=6510, dtype=torch.float64,
                      itags=["i", "j", "k"])

    perm = [2, 0, 1]
    scalar = contract(permute(conj(A), perm), A, axes=([0, 1, 2], perm))
    assert math.isclose(scalar.data[()].item(), A.norm() ** 2,
                        rel_tol=1e-10, abs_tol=1e-12)


def test_contract_u1_conj_permuted_norm_4th_order():
    """Test ⟨perm(A†)|A⟩ = ‖A‖² with non-canonical axes (U(1), 4th order)."""
    group = U1Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx4 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx1, idx2, idx3, idx4], seed=6520, dtype=torch.float64,
                      itags=["i", "j", "k", "l"])

    perm = [2, 3, 0, 1]
    scalar = contract(permute(conj(A), perm), A, axes=([0, 1, 2, 3], perm))
    assert math.isclose(scalar.data[()].item(), A.norm() ** 2,
                        rel_tol=1e-10, abs_tol=1e-12)


def test_contract_u1_conj_permuted_norm_5th_order():
    """Test ⟨perm(A†)|A⟩ = ‖A‖² with non-canonical axes (U(1), 5th order)."""
    group = U1Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2), Sector(-1, 1)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx4 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx5 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))

    A = Tensor.random([idx1, idx2, idx3, idx4, idx5], seed=6530, dtype=torch.float64,
                      itags=["i", "j", "k", "l", "m"])

    perm = [2, 4, 0, 3, 1]
    scalar = contract(permute(conj(A), perm), A, axes=([0, 1, 2, 3, 4], perm))
    assert math.isclose(scalar.data[()].item(), A.norm() ** 2,
                        rel_tol=1e-10, abs_tol=1e-12)


def test_contract_su2_conj_permuted_norm_3rd_order():
    """Test ⟨perm(A†)|A⟩ = ‖A‖² with non-canonical axes (SU(2), 3rd order)."""
    group = SU2Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx1, idx2, idx3], seed=4510, dtype=torch.float64,
                      itags=["i", "j", "k"])
    populate_random_weights(A, seed=4511)

    perm = [2, 0, 1]
    scalar = contract(permute(conj(A), perm), A, axes=([0, 1, 2], perm))
    assert math.isclose(scalar.data[()].item(), A.norm() ** 2,
                        rel_tol=1e-10, abs_tol=1e-12)


def test_contract_su2_conj_permuted_norm_4th_order():
    """Test ⟨perm(A†)|A⟩ = ‖A‖² with non-canonical axes (SU(2), 4th order)."""
    group = SU2Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx4 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx1, idx2, idx3, idx4], seed=4520, dtype=torch.float64,
                      itags=["i", "j", "k", "l"])
    populate_random_weights(A, seed=4521)

    perm = [2, 3, 0, 1]
    scalar = contract(permute(conj(A), perm), A, axes=([0, 1, 2, 3], perm))
    assert math.isclose(scalar.data[()].item(), A.norm() ** 2,
                        rel_tol=1e-10, abs_tol=1e-12)


def test_contract_su2_conj_permuted_norm_5th_order():
    """Test ⟨perm(A†)|A⟩ = ‖A‖² with non-canonical axes (SU(2), 5th order)."""
    group = SU2Group()

    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx4 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx5 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 4)))

    A = Tensor.random([idx1, idx2, idx3, idx4, idx5], seed=4530, dtype=torch.float64,
                      itags=["i", "j", "k", "l", "m"])
    populate_random_weights(A, seed=4531)

    perm = [2, 4, 0, 3, 1]
    scalar = contract(permute(conj(A), perm), A, axes=([0, 1, 2, 3, 4], perm))
    assert math.isclose(scalar.data[()].item(), A.norm() ** 2,
                        rel_tol=1e-10, abs_tol=1e-12)


def test_contract_u1su2_conj_permuted_norm_3rd_order():
    """Test ⟨perm(A†)|A⟩ = ‖A‖² with non-canonical axes (U(1)×SU(2), 3rd order)."""
    group = ProductGroup([U1Group(), SU2Group()])

    idx1 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx2 = Index(Direction.IN,  group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx3 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 1), Sector((1, 2), 1)))

    A = Tensor.random([idx1, idx2, idx3], seed=7510, dtype=torch.float64,
                      itags=["i", "j", "k"])
    populate_random_weights(A, seed=7511)

    perm = [2, 0, 1]
    scalar = contract(permute(conj(A), perm), A, axes=([0, 1, 2], perm))
    assert math.isclose(scalar.data[()].item(), A.norm() ** 2,
                        rel_tol=1e-10, abs_tol=1e-12)


def test_contract_u1su2_conj_permuted_norm_4th_order():
    """Test ⟨perm(A†)|A⟩ = ‖A‖² with non-canonical axes (U(1)×SU(2), 4th order)."""
    group = ProductGroup([U1Group(), SU2Group()])

    idx1 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx2 = Index(Direction.IN,  group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx3 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 1), Sector((1, 2), 1)))
    idx4 = Index(Direction.IN,  group,
                 sectors=(Sector((0, 0), 1), Sector((1, 2), 1)))

    A = Tensor.random([idx1, idx2, idx3, idx4], seed=7520, dtype=torch.float64,
                      itags=["i", "j", "k", "l"])
    populate_random_weights(A, seed=7521)

    perm = [2, 3, 0, 1]
    scalar = contract(permute(conj(A), perm), A, axes=([0, 1, 2, 3], perm))
    assert math.isclose(scalar.data[()].item(), A.norm() ** 2,
                        rel_tol=1e-10, abs_tol=1e-12)


def test_contract_u1su2_conj_permuted_norm_5th_order():
    """Test ⟨perm(A†)|A⟩ = ‖A‖² with non-canonical axes (U(1)×SU(2), 5th order)."""
    group = ProductGroup([U1Group(), SU2Group()])

    idx1 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx2 = Index(Direction.IN,  group,
                 sectors=(Sector((0, 0), 1), Sector((1, 2), 1), Sector((-1, 2), 1)))
    idx3 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 2), Sector((1, 2), 1)))
    idx4 = Index(Direction.IN,  group,
                 sectors=(Sector((0, 0), 1), Sector((1, 2), 1)))
    idx5 = Index(Direction.OUT, group,
                 sectors=(Sector((0, 0), 3), Sector((1, 2), 1)))

    A = Tensor.random([idx1, idx2, idx3, idx4, idx5], seed=7530, dtype=torch.float64,
                      itags=["i", "j", "k", "l", "m"])
    populate_random_weights(A, seed=7531)

    perm = [2, 4, 0, 3, 1]
    scalar = contract(permute(conj(A), perm), A, axes=([0, 1, 2, 3, 4], perm))
    assert math.isclose(scalar.data[()].item(), A.norm() ** 2,
                        rel_tol=1e-10, abs_tol=1e-12)
