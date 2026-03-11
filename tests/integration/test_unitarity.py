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


"""Unitarity tests for SU(2) tensor operations.

Tests verify unitarity properties of SU(2) operations:
- Isometry unitarity: V† V = I
- Isometry fusion-unfusion roundtrips
- Isometry linearity
- Conjugate norm consistency
"""

import math
import torch

from nicole import Direction, Tensor, Index, Sector
from nicole import contract, isometry, isometry_n, conj, identity
from nicole import SU2Group
from ..utils import (
    assert_charge_neutral,
    populate_random_weights,
    assert_data_weights_equal,
    assert_physical_tensors_equal,
)


# Isometry unitarity tests (U† U = I)

def test_contract_su2_isometry_orthonormality():
    """Test that contracting isometry with its conjugate gives identity: V†(a, b, f'*) ⊗ V(a*, b*, f) = I(f'*, f)."""
    group = SU2Group()
    
    # Test all 4 combinations of input directions
    for dir_a in [Direction.OUT, Direction.IN]:
        for dir_b in [Direction.OUT, Direction.IN]:
            # Create isometry V that fuses a⊗b → f
            idx_a = Index(dir_a, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
            idx_b = Index(dir_b, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
            
            V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
            
            # Contract: V†(a, b, f'*) ⊗ V(a*, b*, f) → result(f'*, f)
            # This should give identity on f
            result = contract(V.conj(), V, axes=([0, 1], [0, 1]))
            
            assert len(result.indices) == 2, f"Failed for directions ({dir_a}, {dir_b})"
            assert result.intw is not None, f"Failed for directions ({dir_a}, {dir_b})"
            assert_charge_neutral(result)
            
            # Create expected identity on the fused index
            # V.conj() has fused index flipped, so result is I(f'*, f)
            I_expected = identity(V.indices[2], itags=["f'", "f"])
            
            # Result should match identity (exactly for data and weights)
            assert_data_weights_equal(result, I_expected, 
                                     msg=f"isometry conjugate gives identity for directions ({dir_a}, {dir_b})")


# isometry_n unitarity tests (V† V = I)

def test_contract_isometry_n_orthonormality_3rd_order():
    """Test that contracting isometry_n with its conjugate gives identity: V†(a,b,c,f'*) ⊗ V(a*,b*,c*,f) = I(f'*,f) for 3 input indices."""
    group = SU2Group()
    sectors = (Sector(0, 1), Sector(1, 2), Sector(2, 3))

    for dir_a in [Direction.OUT, Direction.IN]:
        for dir_b in [Direction.OUT, Direction.IN]:
            for dir_c in [Direction.OUT, Direction.IN]:
                idx_a = Index(dir_a, group, sectors=sectors)
                idx_b = Index(dir_b, group, sectors=sectors)
                idx_c = Index(dir_c, group, sectors=sectors)

                V = isometry_n([idx_a, idx_b, idx_c], itags=["a", "b", "c", "f"])

                # Contract: V†(a,b,c,f'*) ⊗ V(a*,b*,c*,f) → result(f'*, f)
                result = contract(V.conj(), V, axes=([0, 1, 2], [0, 1, 2]))

                dirs = (dir_a, dir_b, dir_c)
                assert len(result.indices) == 2, f"Failed for directions {dirs}"
                assert result.intw is not None, f"Failed for directions {dirs}"
                assert_charge_neutral(result)

                I_expected = identity(V.indices[3], itags=["f'", "f"])
                assert_physical_tensors_equal(result, I_expected,
                                         msg=f"isometry_n conjugate gives identity for directions {dirs}")


def test_contract_isometry_n_orthonormality_4th_order():
    """Test that contracting isometry_n with its conjugate gives identity: V†V = I for 4 input indices."""
    group = SU2Group()
    sectors = (Sector(0, 1), Sector(1, 2))

    # Test a representative selection of direction combinations
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

        # Contract: V†(a,b,c,d,f'*) ⊗ V(a*,b*,c*,d*,f) → result(f'*, f)
        result = contract(V.conj(), V, axes=([0, 1, 2, 3], [0, 1, 2, 3]))

        dirs = (dir_a, dir_b, dir_c, dir_d)
        assert len(result.indices) == 2, f"Failed for directions {dirs}"
        assert result.intw is not None, f"Failed for directions {dirs}"
        assert_charge_neutral(result)

        I_expected = identity(V.indices[4], itags=["f'", "f"])
        assert_physical_tensors_equal(result, I_expected,
                                  msg=f"isometry_n conjugate gives identity for directions {dirs}")


def test_contract_isometry_n_orthonormality_5th_order():
    """Test that contracting isometry_n with its conjugate gives identity: V†V = I for 5 input indices."""
    group = SU2Group()
    sectors = (Sector(0, 1), Sector(1, 2))

    # Test a representative selection of direction combinations
    dir_combos = [
        (Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT, Direction.OUT),
        (Direction.OUT, Direction.IN,  Direction.OUT, Direction.IN,  Direction.OUT),
        (Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN,  Direction.IN),
    ]

    for dirs in dir_combos:
        indices = [Index(d, group, sectors=sectors) for d in dirs]

        V = isometry_n(indices, itags=["a", "b", "c", "d", "e", "f"])

        # Contract: V†(a,b,c,d,e,f'*) ⊗ V(a*,b*,c*,d*,e*,f) → result(f'*, f)
        result = contract(V.conj(), V, axes=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))

        assert len(result.indices) == 2, f"Failed for directions {dirs}"
        assert result.intw is not None, f"Failed for directions {dirs}"
        assert_charge_neutral(result)

        I_expected = identity(V.indices[5], itags=["f'", "f"])
        assert_physical_tensors_equal(result, I_expected,
                                  msg=f"isometry_n conjugate gives identity for directions {dirs}")


# Isometry fusion-unfusion tests

def test_contract_su2_isometry_fusion_unfusion_roundtrip_basic():
    """Test that fusion with isometry then unfusion preserves tensor properties (3rd order)."""
    group = SU2Group()

    # Use multiple sectors including spin-0, spin-1/2, and spin-1
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=3000, itags=["a", "b", "c"])
    populate_random_weights(A, seed=3001)
    
    # Create isometry V(a, b, f) that fuses a⊗b → f
    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    
    # Fuse: V†(a*, b*, f*) ⊗ A(a*, b*, c) → fused(f*, c)
    V_conj = conj(V)
    fused = contract(V_conj, A, axes=([0, 1], [0, 1]))
    
    assert len(fused.indices) == 2
    assert fused.intw is not None
    assert_charge_neutral(fused)
    
    # Unfuse: V(a, b, f) ⊗ fused(f*, c) → unfused(a, b, c)
    unfused = contract(V, fused, axes=(2, 0))
    
    assert len(unfused.indices) == 3
    assert_charge_neutral(unfused)
    
    # Verify physical tensor R@W equality
    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11, 
                                  msg="fusion-unfusion roundtrip")


def test_contract_su2_isometry_fusion_unfusion_roundtrip_4th_order():
    """Test that fusion with isometry then unfusion preserves tensor properties (4th order)."""
    group = SU2Group()

    # Use multiple sectors including spin-0, spin-1/2, and spin-1
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=3000, itags=["a", "b", "c", "d"])
    populate_random_weights(A, seed=3001)
    
    # Create isometry V(a, b, f) that fuses a⊗b → f
    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    
    # Fuse: V†(a*, b*, f*) ⊗ A(a*, b*, c, d*) → fused(f*, c, d*)
    V_conj = conj(V)
    fused = contract(V_conj, A, axes=([0, 1], [0, 1]))
    
    assert len(fused.indices) == 3
    assert fused.intw is not None
    assert_charge_neutral(fused)
    
    # Unfuse: V(a, b, f) ⊗ fused(f*, c, d*) → unfused(a, b, c, d*)
    unfused = contract(V, fused, axes=(2, 0))
    
    assert len(unfused.indices) == 4
    assert_charge_neutral(unfused)
    
    # Verify physical tensor R@W equality
    # (Fusion-unfusion changes reduced space dimensions, so compare physical content)
    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11, 
                                  msg="fusion-unfusion roundtrip")


def test_contract_su2_isometry_fusion_unfusion_roundtrip_5th_order():
    """Test that fusion with isometry then unfusion preserves tensor properties (5th order)."""
    group = SU2Group()

    # Use multiple sectors including spin-0, spin-1/2, and spin-1
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_e = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=3010, itags=["a", "b", "c", "d", "e"])
    populate_random_weights(A, seed=3011)
    
    # Create isometry V(a, b, f) that fuses a⊗b → f
    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    
    # Fuse: V†(a*, b*, f*) ⊗ A(a*, b*, c, d*, e) → fused(f*, c, d*, e)
    V_conj = conj(V)
    fused = contract(V_conj, A, axes=([0, 1], [0, 1]))
    
    assert len(fused.indices) == 4
    assert fused.intw is not None
    assert_charge_neutral(fused)
    
    # Unfuse: V(a, b, f) ⊗ fused(f*, c, d*, e) → unfused(a, b, c, d*, e)
    unfused = contract(V, fused, axes=(2, 0))
    
    assert len(unfused.indices) == 5
    assert_charge_neutral(unfused)
    
    # Verify physical tensor R@W equality
    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11, 
                                  msg="fusion-unfusion roundtrip")


# isometry_n fusion-unfusion tests

def test_contract_su2_isometry_n_fusion_unfusion_roundtrip_3rd_order():
    """Test that fusion with isometry_n then unfusion preserves tensor properties (3 fused indices)."""
    group = SU2Group()
    sectors = (Sector(0, 1), Sector(1, 2), Sector(2, 3))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.OUT, group, sectors=sectors)
    idx_d = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=5000, itags=["a", "b", "c", "d"])
    populate_random_weights(A, seed=5001)

    # Create isometry_n V(a*, b*, c*, f) that fuses a⊗b⊗c → f
    # V's unfused legs have opposite directions to inputs, so they contract naturally with A
    V = isometry_n([idx_a, idx_b, idx_c], itags=["a", "b", "c", "f"])

    # Fuse: V(a*, b*, c*, f) ⊗ A(a, b, c, d) → fused(f, d)
    fused = contract(V, A, axes=([0, 1, 2], [0, 1, 2]))

    assert len(fused.indices) == 2
    assert fused.intw is not None
    assert_charge_neutral(fused)

    # Unfuse: V†(a, b, c, f*) ⊗ fused(f, d) → unfused(a, b, c, d)
    unfused = contract(V.conj(), fused, axes=(3, 0))

    assert len(unfused.indices) == 4
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="isometry_n fusion-unfusion roundtrip")


def test_contract_su2_isometry_n_fusion_unfusion_roundtrip_4th_order():
    """Test that fusion with isometry_n then unfusion preserves tensor properties (4 fused indices)."""
    group = SU2Group()
    sectors = (Sector(0, 1), Sector(1, 2), Sector(2, 3))

    idx_a = Index(Direction.OUT, group, sectors=sectors)
    idx_b = Index(Direction.OUT, group, sectors=sectors)
    idx_c = Index(Direction.OUT, group, sectors=sectors)
    idx_d = Index(Direction.OUT, group, sectors=sectors)
    idx_e = Index(Direction.IN, group, sectors=sectors)

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=5010, itags=["a", "b", "c", "d", "e"])
    populate_random_weights(A, seed=5011)

    # Create isometry_n V(a*, b*, c*, d*, f) that fuses a⊗b⊗c⊗d → f
    V = isometry_n([idx_a, idx_b, idx_c, idx_d], itags=["a", "b", "c", "d", "f"])

    # Fuse: V(a*, b*, c*, d*, f) ⊗ A(a, b, c, d, e) → fused(f, e)
    fused = contract(V, A, axes=([0, 1, 2, 3], [0, 1, 2, 3]))

    assert len(fused.indices) == 2
    assert fused.intw is not None
    assert_charge_neutral(fused)

    # Unfuse: V†(a, b, c, d, f*) ⊗ fused(f, e) → unfused(a, b, c, d, e)
    unfused = contract(V.conj(), fused, axes=(4, 0))

    assert len(unfused.indices) == 5
    assert_charge_neutral(unfused)

    assert_physical_tensors_equal(A, unfused, rtol=1e-9, atol=1e-11,
                                  msg="isometry_n fusion-unfusion roundtrip")


# Isometry linearity tests

def test_contract_su2_isometry_linearity_basic():
    """Test that isometry contraction is linear: V ⊗ (A + B) = (V ⊗ A) + (V ⊗ B) (3rd order)."""
    group = SU2Group()
    
    # Use multiple sectors including spin-0, spin-1/2, and spin-1
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    
    # Create isometry
    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]
    
    # Create two tensors with matching fused index
    A = Tensor.random([f_index.flip(), idx_c], seed=3100, itags=["f", "c"])
    B = Tensor.random([f_index.flip(), idx_c], seed=3101, itags=["f", "c"])
    populate_random_weights(A, seed=3102)
    populate_random_weights(B, seed=3103)
    
    # Compute V ⊗ (A + B)
    A_plus_B = A + B
    result_combined = contract(V, A_plus_B, axes=(2, 0))
    
    # Compute (V ⊗ A) + (V ⊗ B)
    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB
    
    # Verify data and weights equality
    assert_data_weights_equal(result_combined, result_separate, msg="isometry linearity")


def test_contract_su2_isometry_linearity_4th_order():
    """Test that isometry contraction is linear: V ⊗ (A + B) = (V ⊗ A) + (V ⊗ B) (4th order)."""
    group = SU2Group()
    
    # Use multiple sectors including spin-0, spin-1/2, and spin-1
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    
    # Create isometry
    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]
    
    # Create two tensors with matching fused index
    A = Tensor.random([f_index.flip(), idx_c, idx_d], seed=3110, itags=["f", "c", "d"])
    B = Tensor.random([f_index.flip(), idx_c, idx_d], seed=3111, itags=["f", "c", "d"])
    populate_random_weights(A, seed=3112)
    populate_random_weights(B, seed=3113)
    
    # Compute V ⊗ (A + B)
    A_plus_B = A + B
    result_combined = contract(V, A_plus_B, axes=(2, 0))
    
    # Compute (V ⊗ A) + (V ⊗ B)
    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB
    
    # Verify physical tensor R@W equality (weights may have gauge freedom)
    assert_physical_tensors_equal(result_combined, result_separate, msg="isometry linearity")


def test_contract_su2_isometry_linearity_5th_order():
    """Test that isometry contraction is linear: V ⊗ (A + B) = (V ⊗ A) + (V ⊗ B) (5th order)."""
    group = SU2Group()
    
    # Use multiple sectors including spin-0, spin-1/2, and spin-1
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_e = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    
    # Create isometry
    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]
    
    # Create two tensors with matching fused index
    A = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=3120, itags=["f", "c", "d", "e"])
    B = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=3121, itags=["f", "c", "d", "e"])
    populate_random_weights(A, seed=3122)
    populate_random_weights(B, seed=3123)
    
    # Compute V ⊗ (A + B)
    A_plus_B = A + B
    result_combined = contract(V, A_plus_B, axes=(2, 0))
    
    # Compute (V ⊗ A) + (V ⊗ B)
    VA = contract(V, A, axes=(2, 0))
    VB = contract(V, B, axes=(2, 0))
    result_separate = VA + VB
    
    # Verify physical tensor R@W equality (weights may have gauge freedom)
    assert_physical_tensors_equal(result_combined, result_separate, msg="isometry linearity")


# Isometry scalar multiplication tests

def test_contract_su2_isometry_scalar_multiplication_basic():
    """Test that isometry contraction commutes with scalar multiplication (3rd order)."""
    group = SU2Group()
    
    # Use multiple sectors including spin-0, spin-1/2, and spin-1
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    
    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]
    
    A = Tensor.random([f_index.flip(), idx_c], seed=3200, itags=["f", "c"])
    populate_random_weights(A, seed=3201)
    
    alpha = 2.5
    
    # Compute V ⊗ (αA)
    result1 = contract(V, alpha * A, axes=(2, 0))
    
    # Compute α(V ⊗ A)
    result2 = alpha * contract(V, A, axes=(2, 0))
    
    # Should be equivalent - check data and weights equality
    assert_data_weights_equal(result1, result2, msg="isometry scalar multiplication")


def test_contract_su2_isometry_scalar_multiplication_4th_order():
    """Test that isometry contraction commutes with scalar multiplication (4th order)."""
    group = SU2Group()
    
    # Use multiple sectors including spin-0, spin-1/2, and spin-1
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    
    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]
    
    A = Tensor.random([f_index.flip(), idx_c, idx_d], seed=3210, itags=["f", "c", "d"])
    populate_random_weights(A, seed=3211)
    
    alpha = 2.5
    
    # Compute V ⊗ (αA)
    result1 = contract(V, alpha * A, axes=(2, 0))
    
    # Compute α(V ⊗ A)
    result2 = alpha * contract(V, A, axes=(2, 0))
    
    # Should be equivalent - check data and weights equality
    assert_data_weights_equal(result1, result2, msg="isometry scalar multiplication")


def test_contract_su2_isometry_scalar_multiplication_5th_order():
    """Test that isometry contraction commutes with scalar multiplication (5th order)."""
    group = SU2Group()
    
    # Use multiple sectors including spin-0, spin-1/2, and spin-1
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_e = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    
    V = isometry(idx_a, idx_b, itags=["a", "b", "f"])
    f_index = V.indices[2]
    
    A = Tensor.random([f_index.flip(), idx_c, idx_d, idx_e], seed=3220, itags=["f", "c", "d", "e"])
    populate_random_weights(A, seed=3221)
    
    alpha = 2.5
    
    # Compute V ⊗ (αA)
    result1 = contract(V, alpha * A, axes=(2, 0))
    
    # Compute α(V ⊗ A)
    result2 = alpha * contract(V, A, axes=(2, 0))
    
    # Should be equivalent - check data and weights equality
    assert_data_weights_equal(result1, result2, msg="isometry scalar multiplication")


# Conjugation interaction tests

def test_contract_su2_conjugate_norm_matrix():
    """Test that <A|A> = ||A||² for real SU(2) tensors (2nd order)."""
    group = SU2Group()
    
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    A = Tensor.random([idx1, idx2], seed=4000, dtype=torch.float64, itags=["i", "j"])
    populate_random_weights(A, seed=4001)
    
    # Compute <A|A> using conjugate - contract all indices
    A_conj = conj(A)
    
    scalar_AA = contract(A_conj, A, axes=([0, 1], [0, 1]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2
    
    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 2nd order tensor: {value_AA} vs {norm_sq}"


def test_contract_su2_conjugate_norm_basic():
    """Test that <A|A> = ||A||² for real SU(2) tensors (3rd order)."""
    group = SU2Group()
    
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx1, idx2, idx3], seed=4010, dtype=torch.float64, itags=["i", "j", "k"])
    populate_random_weights(A, seed=4011)
    
    # Compute <A|A> using conjugate - contract all indices
    A_conj = conj(A)
    
    scalar_AA = contract(A_conj, A, axes=([0, 1, 2], [0, 1, 2]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2
    
    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 3rd order tensor: {value_AA} vs {norm_sq}"


def test_contract_su2_conjugate_norm_4th_order():
    """Test that <A|A> = ||A||² for real SU(2) tensors (4th order)."""
    group = SU2Group()
    
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx1, idx2, idx3, idx4], seed=4020, dtype=torch.float64, itags=["i", "j", "k", "l"])
    populate_random_weights(A, seed=4021)
    
    # Compute <A|A> using conjugate - contract all indices
    A_conj = conj(A)
    
    scalar_AA = contract(A_conj, A, axes=([0, 1, 2, 3], [0, 1, 2, 3]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2
    
    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 4th order tensor: {value_AA} vs {norm_sq}"


def test_contract_su2_conjugate_norm_5th_order():
    """Test that <A|A> = ||A||² for real SU(2) tensors (5th order)."""
    group = SU2Group()
    
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx5 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 4)))
    
    A = Tensor.random([idx1, idx2, idx3, idx4, idx5], seed=4030, dtype=torch.float64, 
                     itags=["i", "j", "k", "l", "m"])
    populate_random_weights(A, seed=4031)
    
    # Compute <A|A> using conjugate - contract all indices
    A_conj = conj(A)
    
    scalar_AA = contract(A_conj, A, axes=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]))
    value_AA = scalar_AA.data[()].item()
    norm_sq = A.norm() ** 2
    
    assert math.isclose(value_AA, norm_sq, rel_tol=1e-10, abs_tol=1e-12), \
        f"<A|A> should equal ||A||² for 5th order tensor: {value_AA} vs {norm_sq}"
