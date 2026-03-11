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


"""Consistency tests for SU(2) tensor operations.

Tests verify that contraction interacts correctly with other operations:
identity, addition, scalar multiplication, and permutation.
"""

import math

from nicole import Direction, Tensor, Index, Sector
from nicole import contract, identity, permute, conj
from nicole import SU2Group
from .utils import (
    assert_charge_neutral, 
    populate_random_weights,
    assert_data_weights_equal,
    assert_physical_tensors_equal,
)


# Identity contraction tests

def test_contract_su2_identity_value_preservation_matrix():
    """Test that contracting 2nd order tensor (matrix) with identity preserves exact values."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    A = Tensor.random([idx_a, idx_b], seed=1000, itags=["a", "b"])
    populate_random_weights(A, seed=1001)
    
    # Contract last index with identity
    # A(a*, b) ⊗ I(b, b2*) → result(a*, b2*)
    I = identity(idx_b.flip(), itags=["b", "b2"])
    result = contract(A, I, axes=(1, 0))
    
    assert len(result.indices) == 2
    assert list(result.itags) == ["a", "b2"]
    assert result.intw is not None
    assert_charge_neutral(result)
    
    # Verify norm is preserved
    assert math.isclose(result.norm(), A.norm(), rel_tol=1e-10, abs_tol=1e-12)
    
    # Identity only has diagonal blocks, so result has same block keys as A
    # Verify each block and its weights match exactly
    assert_data_weights_equal(A, result)


def test_contract_su2_identity_value_preservation_basic():
    """Test that contracting 3rd order tensor with identity preserves exact values."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=1010, itags=["a", "b", "c"])
    populate_random_weights(A, seed=1011)
    
    # Contract last index with identity to preserve first two indices
    # A(a*, b, c*) ⊗ I(c, c2*) → result(a*, b, c2*)
    I = identity(idx_c.flip(), itags=["c", "c2"])
    result = contract(A, I, axes=(2, 0))
    
    assert len(result.indices) == 3
    assert list(result.itags) == ["a", "b", "c2"]
    assert result.intw is not None
    assert_charge_neutral(result)
    
    # Verify norm is preserved
    assert math.isclose(result.norm(), A.norm(), rel_tol=1e-10, abs_tol=1e-12)
    
    # Identity only has diagonal blocks, so result has same block keys as A
    # Verify each block and its weights match exactly
    assert_data_weights_equal(A, result)


def test_contract_su2_identity_value_preservation_high_order():
    """Test that contracting 5th order tensor with identity preserves exact values."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 4)))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=1020, itags=["a", "b", "c", "d", "e"])
    populate_random_weights(A, seed=1021)
    
    # Contract last index with identity
    # A(a*, b, c*, d, e*) ⊗ I(e, e2*) → result(a*, b, c*, d, e2*)
    I = identity(idx_e.flip(), itags=["e", "e2"])
    result = contract(A, I, axes=(4, 0))
    
    assert len(result.indices) == 5
    assert list(result.itags) == ["a", "b", "c", "d", "e2"]
    assert result.intw is not None
    assert_charge_neutral(result)
    
    # Verify norm is preserved
    assert math.isclose(result.norm(), A.norm(), rel_tol=1e-10, abs_tol=1e-12)
    
    # Identity only has diagonal blocks, so result has same block keys as A
    # Verify each block and its weights match exactly
    assert_data_weights_equal(A, result)


def test_contract_su2_double_identity_contraction():
    """Test that contracting with multiple identities preserves tensor norm."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=1100, itags=["a", "b", "c"])
    populate_random_weights(A, seed=1101)
    
    # Apply first identity contraction on idx_a
    # I1(a, a2*) ⊗ A(a*, b, c*) → temp(a2*, b, c*)
    I1 = identity(idx_a.flip(), itags=["a", "a2"])
    temp = contract(I1, A, axes=(0, 0))
    
    # temp has indices at positions 0,1,2 with itags ["a2", "b", "c"]
    # temp.indices[1] has itag "b" with direction IN
    # Create identity for this index
    I2 = identity(temp.indices[1].flip(), itags=["b", "b2"])
    result = contract(I2, temp, axes=(0, 1))
    
    # Result should still have same structure
    assert len(result.indices) == 3
    assert result.intw is not None
    assert_charge_neutral(result)
    
    # Norm should be preserved through both contractions
    assert math.isclose(result.norm(), A.norm(), rel_tol=1e-10, abs_tol=1e-12)


def test_contract_su2_identity_permutation_equivalence_matrix():
    """Test that contracting identity and permuting back restores original for 2nd order tensor."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    
    A = Tensor.random([idx_a, idx_b], seed=1200, itags=["a", "b"])
    populate_random_weights(A, seed=1201)
    
    # Test contracting identity on each index
    n = len(A.indices)
    for i in range(n):
        # Contract identity on index i
        I = identity(A.indices[i].flip(), itags=[A.itags[i], f"{A.itags[i]}2"])
        result = contract(A, I, axes=(i, 0))
        
        # Permute back to original ordering
        # Result has indices: [...indices before i..., ...indices after i..., new_index]
        # We need to move the last index back to position i
        perm = list(range(i)) + [n - 1] + list(range(i, n - 1))
        result_restored = permute(result, perm)
        
        # Retag to match original (last index has different tag)
        result_restored.retag(A.itags)
        
        # Should match original tensor exactly
        assert_data_weights_equal(A, result_restored, 
                                 msg=f"identity contraction on index {i}")


def test_contract_su2_identity_permutation_equivalence_basic():
    """Test that contracting identity and permuting back restores original for 3rd order tensor."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=1210, itags=["a", "b", "c"])
    populate_random_weights(A, seed=1211)
    
    # Test contracting identity on each index
    n = len(A.indices)
    for i in range(n):
        # Contract identity on index i
        I = identity(A.indices[i].flip(), itags=[A.itags[i], f"{A.itags[i]}2"])
        result = contract(A, I, axes=(i, 0))
        
        # Permute back to original ordering
        # Result has indices: [...indices before i..., ...indices after i..., new_index]
        # We need to move the last index back to position i
        perm = list(range(i)) + [n - 1] + list(range(i, n - 1))
        result_restored = permute(result, perm)
        
        # Retag to match original (last index has different tag)
        result_restored.retag(A.itags)
        
        # Should match original tensor exactly
        assert_data_weights_equal(A, result_restored, 
                                 msg=f"identity contraction on index {i}")


def test_contract_su2_identity_permutation_equivalence_high_order():
    """Test that contracting identity and permuting back restores original for 5th order tensor."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 4)))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=1220, itags=["a", "b", "c", "d", "e"])
    populate_random_weights(A, seed=1221)
    
    # Test contracting identity on each index
    n = len(A.indices)
    for i in range(n):
        # Contract identity on index i
        I = identity(A.indices[i].flip(), itags=[A.itags[i], f"{A.itags[i]}2"])
        result = contract(A, I, axes=(i, 0))
        
        # Permute back to original ordering
        # Result has indices: [...indices before i..., ...indices after i..., new_index]
        # We need to move the last index back to position i
        perm = list(range(i)) + [n - 1] + list(range(i, n - 1))
        result_restored = permute(result, perm)
        
        # Retag to match original (last index has different tag)
        result_restored.retag(A.itags)
        
        # Should match original tensor exactly
        assert_data_weights_equal(A, result_restored, 
                                 msg=f"identity contraction on index {i}")


# Distributivity tests

def test_contract_su2_distributivity_right_matrix():
    """Test right distributivity: A ⊗ (B + C) = (A ⊗ B) + (A ⊗ C) with 2nd order tensors."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b], seed=2000, itags=["a", "b"])
    B = Tensor.random([idx_b.flip(), idx_c], seed=2001, itags=["b", "c"])
    C = Tensor.random([idx_b.flip(), idx_c], seed=2002, itags=["b", "c"])
    populate_random_weights(A, seed=2003)
    populate_random_weights(B, seed=2004)
    populate_random_weights(C, seed=2005)
    
    # Compute A ⊗ (B + C)
    B_plus_C = B + C
    result_combined = contract(A, B_plus_C, axes=(1, 0))
    
    # Compute (A ⊗ B) + (A ⊗ C)
    AB = contract(A, B, axes=(1, 0))
    AC = contract(A, C, axes=(1, 0))
    result_separate = AB + AC
    
    # For SU(2) tensors, the physical blocks R*W should match
    # even if R and W individually differ due to gauge freedom
    assert_physical_tensors_equal(result_combined, result_separate)
    
    # Overall norm should also match
    assert math.isclose(result_combined.norm(), result_separate.norm(), 
                       rel_tol=1e-10, abs_tol=1e-12)


def test_contract_su2_distributivity_right_basic():
    """Test right distributivity: A ⊗ (B + C) = (A ⊗ B) + (A ⊗ C) with 3rd order tensors."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=2010, itags=["a", "b", "c"])
    B = Tensor.random([idx_b.flip(), idx_c.flip(), idx_d], seed=2011, itags=["b", "c", "d"])
    C = Tensor.random([idx_b.flip(), idx_c.flip(), idx_d], seed=2012, itags=["b", "c", "d"])
    populate_random_weights(A, seed=2013)
    populate_random_weights(B, seed=2014)
    populate_random_weights(C, seed=2015)
    
    # Compute A ⊗ (B + C)
    B_plus_C = B + C
    result_combined = contract(A, B_plus_C, axes=([1, 2], [0, 1]))
    
    # Compute (A ⊗ B) + (A ⊗ C)
    AB = contract(A, B, axes=([1, 2], [0, 1]))
    AC = contract(A, C, axes=([1, 2], [0, 1]))
    result_separate = AB + AC
    
    # Verify physical tensor R@W equality (weights may differ due to gauge freedom)
    assert_physical_tensors_equal(result_combined, result_separate)


def test_contract_su2_distributivity_right_high_order():
    """Test right distributivity: A ⊗ (B + C) = (A ⊗ B) + (A ⊗ C) with 5th order tensors."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 4)))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=2020, itags=["a", "b", "c", "d", "e"])
    B = Tensor.random([idx_b.flip(), idx_d.flip()], seed=2021, itags=["b", "d"])
    C = Tensor.random([idx_b.flip(), idx_d.flip()], seed=2022, itags=["b", "d"])
    populate_random_weights(A, seed=2023)
    populate_random_weights(B, seed=2024)
    populate_random_weights(C, seed=2025)
    
    # Compute A ⊗ (B + C)
    B_plus_C = B + C
    result_combined = contract(A, B_plus_C, axes=([1, 3], [0, 1]))
    
    # Compute (A ⊗ B) + (A ⊗ C)
    AB = contract(A, B, axes=([1, 3], [0, 1]))
    AC = contract(A, C, axes=([1, 3], [0, 1]))
    result_separate = AB + AC
    
    # Verify physical tensor R@W equality (weights may differ due to gauge freedom)
    assert_physical_tensors_equal(result_combined, result_separate)


def test_contract_su2_distributivity_left_matrix():
    """Test left distributivity: (A + B) ⊗ C = (A ⊗ C) + (B ⊗ C) with 2nd order tensors."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b], seed=2100, itags=["a", "b"])
    B = Tensor.random([idx_a, idx_b], seed=2101, itags=["a", "b"])
    C = Tensor.random([idx_b.flip(), idx_c], seed=2102, itags=["b", "c"])
    populate_random_weights(A, seed=2103)
    populate_random_weights(B, seed=2104)
    populate_random_weights(C, seed=2105)
    
    # Compute (A + B) ⊗ C
    A_plus_B = A + B
    result_combined = contract(A_plus_B, C, axes=(1, 0))
    
    # Compute (A ⊗ C) + (B ⊗ C)
    AC = contract(A, C, axes=(1, 0))
    BC = contract(B, C, axes=(1, 0))
    result_separate = AC + BC
    
    # Verify data and weights equality
    assert_data_weights_equal(result_combined, result_separate)


def test_contract_su2_distributivity_left_basic():
    """Test left distributivity: (A + B) ⊗ C = (A ⊗ C) + (B ⊗ C) with 3rd order tensors."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=2110, itags=["a", "b", "c"])
    B = Tensor.random([idx_a, idx_b, idx_c], seed=2111, itags=["a", "b", "c"])
    C = Tensor.random([idx_b.flip(), idx_c.flip(), idx_d], seed=2112, itags=["b", "c", "d"])
    populate_random_weights(A, seed=2113)
    populate_random_weights(B, seed=2114)
    populate_random_weights(C, seed=2115)
    
    # Compute (A + B) ⊗ C
    A_plus_B = A + B
    result_combined = contract(A_plus_B, C, axes=([1, 2], [0, 1]))
    
    # Compute (A ⊗ C) + (B ⊗ C)
    AC = contract(A, C, axes=([1, 2], [0, 1]))
    BC = contract(B, C, axes=([1, 2], [0, 1]))
    result_separate = AC + BC
    
    # Verify physical tensor R@W equality (weights may differ due to gauge freedom)
    assert_physical_tensors_equal(result_combined, result_separate)


def test_contract_su2_distributivity_left_high_order():
    """Test left distributivity: (A + B) ⊗ C = (A ⊗ C) + (B ⊗ C) with 5th order tensors."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 4)))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=2120, itags=["a", "b", "c", "d", "e"])
    B = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=2121, itags=["a", "b", "c", "d", "e"])
    C = Tensor.random([idx_b.flip(), idx_d.flip()], seed=2122, itags=["b", "d"])
    populate_random_weights(A, seed=2123)
    populate_random_weights(B, seed=2124)
    populate_random_weights(C, seed=2125)
    
    # Compute (A + B) ⊗ C
    A_plus_B = A + B
    result_combined = contract(A_plus_B, C, axes=([1, 3], [0, 1]))
    
    # Compute (A ⊗ C) + (B ⊗ C)
    AC = contract(A, C, axes=([1, 3], [0, 1]))
    BC = contract(B, C, axes=([1, 3], [0, 1]))
    result_separate = AC + BC
    
    # Verify physical tensor R@W equality (weights may differ due to gauge freedom)
    assert_physical_tensors_equal(result_combined, result_separate)


def test_contract_su2_scalar_multiplication_linearity_matrix():
    """Test scalar multiplication linearity: (αA) ⊗ B = α(A ⊗ B) with 2nd order tensors."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b], seed=2200, itags=["a", "b"])
    B = Tensor.random([idx_b.flip(), idx_c], seed=2201, itags=["b", "c"])
    populate_random_weights(A, seed=2202)
    populate_random_weights(B, seed=2203)
    
    alpha = 2.5
    
    # Compute (αA) ⊗ B
    scaled_A = alpha * A
    result1 = contract(scaled_A, B, axes=(1, 0))
    
    # Compute α(A ⊗ B)
    AB = contract(A, B, axes=(1, 0))
    result2 = alpha * AB
    
    # Results should match
    assert_data_weights_equal(result1, result2)
    
    # Also test from the right: A ⊗ (αB) = α(A ⊗ B)
    scaled_B = alpha * B
    result3 = contract(A, scaled_B, axes=(1, 0))
    
    assert_data_weights_equal(result3, result2)


def test_contract_su2_scalar_multiplication_linearity_basic():
    """Test scalar multiplication linearity: (αA) ⊗ B = α(A ⊗ B) with 3rd order tensors."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=2210, itags=["a", "b", "c"])
    B = Tensor.random([idx_b.flip(), idx_c.flip(), idx_d], seed=2211, itags=["b", "c", "d"])
    populate_random_weights(A, seed=2212)
    populate_random_weights(B, seed=2213)
    
    alpha = 2.5
    
    # Compute (αA) ⊗ B
    scaled_A = alpha * A
    result1 = contract(scaled_A, B, axes=([1, 2], [0, 1]))
    
    # Compute α(A ⊗ B)
    AB = contract(A, B, axes=([1, 2], [0, 1]))
    result2 = alpha * AB
    
    # Results should match
    assert_data_weights_equal(result1, result2)
    
    # Also test from the right: A ⊗ (αB) = α(A ⊗ B)
    scaled_B = alpha * B
    result3 = contract(A, scaled_B, axes=([1, 2], [0, 1]))
    
    assert_data_weights_equal(result3, result2)


def test_contract_su2_scalar_multiplication_linearity_high_order():
    """Test scalar multiplication linearity: (αA) ⊗ B = α(A ⊗ B) with 5th order tensors."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 4)))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=2220, itags=["a", "b", "c", "d", "e"])
    B = Tensor.random([idx_b.flip(), idx_d.flip()], seed=2221, itags=["b", "d"])
    populate_random_weights(A, seed=2222)
    populate_random_weights(B, seed=2223)
    
    alpha = 2.5
    
    # Compute (αA) ⊗ B
    scaled_A = alpha * A
    result1 = contract(scaled_A, B, axes=([1, 3], [0, 1]))
    
    # Compute α(A ⊗ B)
    AB = contract(A, B, axes=([1, 3], [0, 1]))
    result2 = alpha * AB
    
    # Results should match
    assert_data_weights_equal(result1, result2)
    
    # Also test from the right: A ⊗ (αB) = α(A ⊗ B)
    scaled_B = alpha * B
    result3 = contract(A, scaled_B, axes=([1, 3], [0, 1]))
    
    assert_data_weights_equal(result3, result2)


# Linearity tests

def test_contract_su2_bilinearity():
    """Test bilinearity: <αA + βB | C> = α<A|C> + β<B|C>."""
    group = SU2Group()
    
    # Create 3rd order tensors for full contraction
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    indices = [idx1, idx2, idx3]
    
    A = Tensor.random(indices, seed=2300, itags=["i", "j", "k"])
    B = Tensor.random(indices, seed=2301, itags=["i", "j", "k"])
    C = Tensor.random(indices, seed=2302, itags=["i", "j", "k"])
    populate_random_weights(A, seed=2303)
    populate_random_weights(B, seed=2304)
    populate_random_weights(C, seed=2305)
    
    alpha = 1.5
    beta = -0.8
    
    # Compute <αA + βB | C>
    linear_combo = alpha * A + beta * B
    linear_combo_conj = conj(linear_combo)
    result_combined = contract(linear_combo_conj, C)
    
    # Compute α<A|C> + β<B|C>
    A_conj = conj(A)
    B_conj = conj(B)
    
    scalar_AC = contract(A_conj, C)
    scalar_BC = contract(B_conj, C)
    result_separate = alpha * scalar_AC.data[()] + beta * scalar_BC.data[()]
    
    # Scalar results should match
    assert result_combined.is_scalar()
    assert math.isclose(result_combined.data[()].item(), result_separate.item(), 
                       rel_tol=1e-10, abs_tol=1e-12), \
        "Bilinearity should hold for SU(2) scalar products"


# Permutation interaction tests

def test_contract_su2_permutation_commutativity_basic():
    """Test that permutation and contraction commute: permute then contract vs contract then permute (3rd order)."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=5000, itags=["a", "b", "c"])
    B = Tensor.random([idx_b.flip(), idx_d], seed=5001, itags=["b", "d"])
    populate_random_weights(A, seed=5002)
    populate_random_weights(B, seed=5003)
    
    # Path 1: Contract then permute
    # A(a*, b, c*) ⊗ B(b*, d) → C(a*, c*, d)
    C1 = contract(A, B, axes=(1, 0))
    # Permute to (c*, a*, d)
    C1_perm = permute(C1, [1, 0, 2])
    
    # Path 2: Permute A then contract
    # Permute A to (c*, a*, b)
    A_perm = permute(A, [2, 0, 1])  # itags: ["c", "a", "b"]
    # A_perm(c*, a*, b) ⊗ B(b*, d) → C2(c*, a*, d)
    C2 = contract(A_perm, B, axes=(2, 0))
    
    # Results should match
    assert list(C1_perm.itags) == list(C2.itags) == ["c", "a", "d"]
    
    # Verify data and weights match exactly
    assert_data_weights_equal(C1_perm, C2, msg="permutation commutativity")
    
    # Norms should definitely match
    assert math.isclose(C1_perm.norm(), C2.norm(), rel_tol=1e-10, abs_tol=1e-12)


def test_contract_su2_permutation_commutativity_4th_order():
    """Test that permutation and contraction commute: permute then contract vs contract then permute (4th order)."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=5010, itags=["a", "b", "c", "d"])
    B = Tensor.random([idx_b.flip(), idx_d.flip(), idx_e], seed=5011, itags=["b", "d", "e"])
    populate_random_weights(A, seed=5012)
    populate_random_weights(B, seed=5013)
    
    # Path 1: Contract then permute
    # A(a*, b, c*, d) ⊗ B(b*, d*, e*) → C(a*, c*, e*)
    C1 = contract(A, B, axes=([1, 3], [0, 1]))
    # Permute to (c*, e*, a*)
    C1_perm = permute(C1, [1, 2, 0])
    
    # Path 2: Permute A then contract
    # Permute A to (c*, a*, b, d)
    A_perm = permute(A, [2, 0, 1, 3])  # itags: ["c", "a", "b", "d"]
    # A_perm(c*, a*, b, d) ⊗ B(b*, d*, e*) → C2(c*, a*, e*)
    C2 = contract(A_perm, B, axes=([2, 3], [0, 1]))
    # Permute C2 to (c*, e*, a*)
    C2_perm = permute(C2, [0, 2, 1])
    
    # Results should match
    assert list(C1_perm.itags) == list(C2_perm.itags) == ["c", "e", "a"]
    
    # Verify data and weights match exactly
    assert_data_weights_equal(C1_perm, C2_perm, msg="permutation commutativity")
    
    # Norms should definitely match
    assert math.isclose(C1_perm.norm(), C2_perm.norm(), rel_tol=1e-10, abs_tol=1e-12)


def test_contract_su2_permutation_commutativity_5th_order():
    """Test that permutation and contraction commute: permute then contract vs contract then permute (5th order)."""
    group = SU2Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 4)))
    idx_f = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=5020, itags=["a", "b", "c", "d", "e"])
    B = Tensor.random([idx_b.flip(), idx_d.flip(), idx_f], seed=5021, itags=["b", "d", "f"])
    populate_random_weights(A, seed=5022)
    populate_random_weights(B, seed=5023)
    
    # Path 1: Contract then permute
    # A(a*, b, c*, d, e*) ⊗ B(b*, d*, f) → C(a*, c*, e*, f)
    C1 = contract(A, B, axes=([1, 3], [0, 1]))
    # Permute to (e*, a*, f, c*)
    C1_perm = permute(C1, [2, 0, 3, 1])
    
    # Path 2: Permute A then contract
    # Permute A to (e*, a*, b, d, c*)
    A_perm = permute(A, [4, 0, 1, 3, 2])  # itags: ["e", "a", "b", "d", "c"]
    # A_perm(e*, a*, b, d, c*) ⊗ B(b*, d*, f) → C2(e*, a*, c*, f)
    C2 = contract(A_perm, B, axes=([2, 3], [0, 1]))
    # Permute C2 to (e*, a*, f, c*)
    C2_perm = permute(C2, [0, 1, 3, 2])
    
    # Results should match
    assert list(C1_perm.itags) == list(C2_perm.itags) == ["e", "a", "f", "c"]
    
    # Verify data and weights match exactly
    assert_data_weights_equal(C1_perm, C2_perm, msg="permutation commutativity")
    
    # Norms should definitely match
    assert math.isclose(C1_perm.norm(), C2_perm.norm(), rel_tol=1e-10, abs_tol=1e-12)
