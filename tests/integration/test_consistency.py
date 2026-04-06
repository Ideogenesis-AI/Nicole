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


"""Consistency tests for SU(2) and U(1)×SU(2) tensor operations.

Tests verify that contraction interacts correctly with other operations:
identity, addition, scalar multiplication, permutation, and bilinearity.

Organization:
- SU(2) tests: Verify consistency for pure SU(2) symmetric tensors
- U(1)×SU(2) tests: Verify consistency for product group tensors combining
  Abelian U(1) symmetry with non-Abelian SU(2) symmetry
"""

import math

from nicole import Direction, Tensor, Index, Sector
from nicole import contract, identity, permute, conj
from nicole import SU2Group, U1Group, ProductGroup
from ..utils import (
    assert_charge_neutral,
    assert_blocks_equal,
    populate_random_weights,
    assert_data_weights_equal,
    assert_physical_tensors_equal,
)


# ============================================================================
#  Pure SU(2) Consistency Tests
# ============================================================================

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
    # Weights may differ by row-normalization gauge; compare physical tensors.
    assert_physical_tensors_equal(A, result)


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
    # Weights may differ by row-normalization gauge; compare physical tensors.
    assert_physical_tensors_equal(A, result)


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
    # Weights may differ by row-normalization gauge; compare physical tensors.
    assert_physical_tensors_equal(A, result)


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
        
        # Weights may differ by row-normalization gauge; compare physical tensors.
        assert_physical_tensors_equal(A, result_restored, 
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
        
        # Weights may differ by row-normalization gauge; compare physical tensors.
        assert_physical_tensors_equal(A, result_restored, 
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
        
        # Weights may differ by row-normalization gauge; compare physical tensors.
        assert_physical_tensors_equal(A, result_restored, 
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
    
    # Distributivity preserves the physical tensor; internal k may differ
    # between (A+B)⊗C and (A⊗C)+(B⊗C) due to component concatenation.
    assert_physical_tensors_equal(result_combined, result_separate)


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
    
    # compress() may produce different (but equivalent) orthonormal bases for the
    # weight subspace across computation paths, so compare physical tensors R@W.
    assert_physical_tensors_equal(C1_perm, C2_perm, msg="permutation commutativity")
    
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
    
    # compress() may produce different (but equivalent) orthonormal bases for the
    # weight subspace across computation paths, so compare physical tensors R@W.
    assert_physical_tensors_equal(C1_perm, C2_perm, msg="permutation commutativity")
    
    # Norms should definitely match
    assert math.isclose(C1_perm.norm(), C2_perm.norm(), rel_tol=1e-10, abs_tol=1e-12)


# ============================================================================
#  U(1)×SU(2) Product Group Consistency Tests
# ============================================================================

# Identity contraction tests

def test_contract_u1su2_identity_value_preservation_matrix():
    """Test that contracting with identity preserves exact tensor values for 2nd order U(1)×SU(2) tensors."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    # Sectors with (u1_charge, su2_spin) tuples
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 0), 1), Sector((0, 1), 3), Sector((1, 1), 2)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 0), 1), Sector((0, 1), 3), Sector((1, 1), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b], seed=6000, itags=["a", "b"])
    populate_random_weights(A, seed=6001)
    
    # Create identity on idx_b.flip()
    # A(a*, b) ⊗ I(b, b'*) → result(a*, b'*)
    I = identity(idx_b.flip(), itags=["b", "b'"])
    
    result = contract(A, I, axes=(1, 0))
    
    # Weights may differ by row-normalization gauge; compare physical tensors.
    assert_physical_tensors_equal(result, A, msg="identity contraction")


def test_contract_u1su2_identity_value_preservation_basic():
    """Test that contracting with identity preserves exact tensor values for 3rd order U(1)×SU(2) tensors."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 0), 1), Sector((0, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 0), 1), Sector((0, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=6010, itags=["a", "b", "c"])
    populate_random_weights(A, seed=6011)
    
    # A(a*, b, c*) ⊗ I(b, b'*) → result(a*, c*, b'*)
    I = identity(idx_b.flip(), itags=["b", "b'"])
    
    result = contract(A, I, axes=(1, 0))
    # Result has indices in order: [a*, c*, b'*], permute to [a*, b'*, c*]
    result_perm = permute(result, [0, 2, 1])
    
    # Weights may differ by row-normalization gauge; compare physical tensors.
    assert_physical_tensors_equal(result_perm, A, msg="identity contraction")


def test_contract_u1su2_identity_value_preservation_high_order():
    """Test that contracting with identity preserves exact tensor values for 5th order U(1)×SU(2) tensors."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    idx_e = Index(Direction.OUT, group, sectors=(
        Sector((0, 1), 3), Sector((2, 0), 1)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=6020, itags=["a", "b", "c", "d", "e"])
    populate_random_weights(A, seed=6021)
    
    # A(a*, b, c*, d, e*) ⊗ I(b, b'*) → result(a*, c*, d, e*, b'*)
    I = identity(idx_b.flip(), itags=["b", "b'"])
    
    result = contract(A, I, axes=(1, 0))
    # Result has indices in order: [a*, c*, d, e*, b'*], permute to [a*, b'*, c*, d, e*]
    result_perm = permute(result, [0, 4, 1, 2, 3])
    
    # Weights may differ by row-normalization gauge; compare physical tensors.
    assert_physical_tensors_equal(result_perm, A, msg="identity contraction")


def test_contract_u1su2_double_identity_contraction():
    """Test that contracting with multiple identities preserves tensor norm for U(1)×SU(2)."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=6500, itags=["a", "b", "c"])
    populate_random_weights(A, seed=6501)
    
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


def test_contract_u1su2_identity_permutation_equivalence_matrix():
    """Test that contracting identity and permuting back restores original for 2nd order U(1)×SU(2) tensor."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 0), 1), Sector((0, 1), 3), Sector((1, 1), 2)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 0), 1), Sector((0, 1), 3), Sector((1, 1), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b], seed=6510, itags=["a", "b"])
    populate_random_weights(A, seed=6511)
    
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
        
        # Weights may differ by row-normalization gauge; compare physical tensors.
        assert_physical_tensors_equal(A, result_restored, 
                                      msg=f"identity contraction on index {i}")


def test_contract_u1su2_identity_permutation_equivalence_basic():
    """Test that contracting identity and permuting back restores original for 3rd order U(1)×SU(2) tensor."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 0), 1), Sector((0, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 0), 1), Sector((0, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=6520, itags=["a", "b", "c"])
    populate_random_weights(A, seed=6521)
    
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
        
        # Weights may differ by row-normalization gauge; compare physical tensors.
        assert_physical_tensors_equal(A, result_restored, 
                                      msg=f"identity contraction on index {i}")


def test_contract_u1su2_identity_permutation_equivalence_high_order():
    """Test that contracting identity and permuting back restores original for 5th order U(1)×SU(2) tensor."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    idx_e = Index(Direction.OUT, group, sectors=(
        Sector((0, 1), 3), Sector((2, 0), 1)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=6530, itags=["a", "b", "c", "d", "e"])
    populate_random_weights(A, seed=6531)
    
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
        
        # Weights may differ by row-normalization gauge; compare physical tensors.
        assert_physical_tensors_equal(A, result_restored, 
                                      msg=f"identity contraction on index {i}")


# Distributivity tests

def test_contract_u1su2_distributivity_right_matrix():
    """Test distributivity over addition on the right: A ⊗ (B + C) = (A ⊗ B) + (A ⊗ C) for 2nd order."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b], seed=6100, itags=["a", "b"])
    B = Tensor.random([idx_b.flip(), idx_c], seed=6101, itags=["b", "c"])
    C = Tensor.random([idx_b.flip(), idx_c], seed=6102, itags=["b", "c"])
    populate_random_weights(A, seed=6103)
    populate_random_weights(B, seed=6104)
    populate_random_weights(C, seed=6105)
    
    # Compute A ⊗ (B + C)
    B_plus_C = B + C
    result1 = contract(A, B_plus_C, axes=(1, 0))
    
    # Compute (A ⊗ B) + (A ⊗ C)
    AB = contract(A, B, axes=(1, 0))
    AC = contract(A, C, axes=(1, 0))
    result2 = AB + AC
    
    # Physical tensor R@W should match (gauge freedom due to SU(2) component)
    assert_physical_tensors_equal(result1, result2, msg="distributivity")


def test_contract_u1su2_distributivity_right_basic():
    """Test distributivity over addition on the right: A ⊗ (B + C) = (A ⊗ B) + (A ⊗ C) for 3rd order."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=6110, itags=["a", "b", "c"])
    B = Tensor.random([idx_b.flip(), idx_d], seed=6111, itags=["b", "d"])
    C = Tensor.random([idx_b.flip(), idx_d], seed=6112, itags=["b", "d"])
    populate_random_weights(A, seed=6113)
    populate_random_weights(B, seed=6114)
    populate_random_weights(C, seed=6115)
    
    B_plus_C = B + C
    result1 = contract(A, B_plus_C, axes=(1, 0))
    
    AB = contract(A, B, axes=(1, 0))
    AC = contract(A, C, axes=(1, 0))
    result2 = AB + AC
    
    # For multi-index contractions, physical tensor R@W should match
    assert_physical_tensors_equal(result1, result2, msg="distributivity")


def test_contract_u1su2_distributivity_right_high_order():
    """Test distributivity over addition on the right: A ⊗ (B + C) = (A ⊗ B) + (A ⊗ C) for 5th order."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    idx_e = Index(Direction.OUT, group, sectors=(
        Sector((0, 1), 3), Sector((2, 0), 1)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=6120, itags=["a", "b", "c", "d", "e"])
    B = Tensor.random([idx_b.flip(), idx_c.flip()], seed=6121, itags=["b", "c"])
    C = Tensor.random([idx_b.flip(), idx_c.flip()], seed=6122, itags=["b", "c"])
    populate_random_weights(A, seed=6123)
    populate_random_weights(B, seed=6124)
    populate_random_weights(C, seed=6125)
    
    B_plus_C = B + C
    result1 = contract(A, B_plus_C, axes=([1, 2], [0, 1]))
    
    AB = contract(A, B, axes=([1, 2], [0, 1]))
    AC = contract(A, C, axes=([1, 2], [0, 1]))
    result2 = AB + AC
    
    # For multi-index contractions, physical tensor R@W should match
    assert_physical_tensors_equal(result1, result2, msg="distributivity")


def test_contract_u1su2_distributivity_left_matrix():
    """Test distributivity over addition on the left: (A + B) ⊗ C = (A ⊗ C) + (B ⊗ C) for 2nd order."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b], seed=6150, itags=["a", "b"])
    B = Tensor.random([idx_a, idx_b], seed=6151, itags=["a", "b"])
    C = Tensor.random([idx_b.flip(), idx_c], seed=6152, itags=["b", "c"])
    populate_random_weights(A, seed=6153)
    populate_random_weights(B, seed=6154)
    populate_random_weights(C, seed=6155)
    
    # Compute (A + B) ⊗ C
    A_plus_B = A + B
    result1 = contract(A_plus_B, C, axes=(1, 0))
    
    # Compute (A ⊗ C) + (B ⊗ C)
    AC = contract(A, C, axes=(1, 0))
    BC = contract(B, C, axes=(1, 0))
    result2 = AC + BC
    
    # Physical tensor R@W should match (gauge freedom due to SU(2) component)
    assert_physical_tensors_equal(result1, result2, msg="distributivity")


def test_contract_u1su2_distributivity_left_basic():
    """Test distributivity over addition on the left: (A + B) ⊗ C = (A ⊗ C) + (B ⊗ C) for 3rd order."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=6160, itags=["a", "b", "c"])
    B = Tensor.random([idx_a, idx_b, idx_c], seed=6161, itags=["a", "b", "c"])
    C = Tensor.random([idx_b.flip(), idx_d], seed=6162, itags=["b", "d"])
    populate_random_weights(A, seed=6163)
    populate_random_weights(B, seed=6164)
    populate_random_weights(C, seed=6165)
    
    A_plus_B = A + B
    result1 = contract(A_plus_B, C, axes=(1, 0))
    
    AC = contract(A, C, axes=(1, 0))
    BC = contract(B, C, axes=(1, 0))
    result2 = AC + BC
    
    # For multi-index contractions, physical tensor R@W should match
    assert_physical_tensors_equal(result1, result2, msg="distributivity")


def test_contract_u1su2_distributivity_left_high_order():
    """Test distributivity over addition on the left: (A + B) ⊗ C = (A ⊗ C) + (B ⊗ C) for 5th order."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    idx_e = Index(Direction.OUT, group, sectors=(
        Sector((0, 1), 3), Sector((2, 0), 1)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=6170, itags=["a", "b", "c", "d", "e"])
    B = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=6171, itags=["a", "b", "c", "d", "e"])
    C = Tensor.random([idx_b.flip(), idx_c.flip()], seed=6172, itags=["b", "c"])
    populate_random_weights(A, seed=6173)
    populate_random_weights(B, seed=6174)
    populate_random_weights(C, seed=6175)
    
    A_plus_B = A + B
    result1 = contract(A_plus_B, C, axes=([1, 2], [0, 1]))
    
    AC = contract(A, C, axes=([1, 2], [0, 1]))
    BC = contract(B, C, axes=([1, 2], [0, 1]))
    result2 = AC + BC
    
    # For multi-index contractions, physical tensor R@W should match
    assert_physical_tensors_equal(result1, result2, msg="distributivity")


# Scalar multiplication linearity tests

def test_contract_u1su2_scalar_multiplication_linearity_matrix():
    """Test scalar multiplication linearity: α(A ⊗ B) = (αA) ⊗ B = A ⊗ (αB) for 2nd order."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b], seed=6200, itags=["a", "b"])
    B = Tensor.random([idx_b.flip(), idx_c], seed=6201, itags=["b", "c"])
    populate_random_weights(A, seed=6202)
    populate_random_weights(B, seed=6203)
    
    alpha = 2.5
    
    # Compute α(A ⊗ B)
    AB = contract(A, B, axes=(1, 0))
    result1 = alpha * AB
    
    # Compute (αA) ⊗ B
    result2 = contract(alpha * A, B, axes=(1, 0))
    
    # Compute A ⊗ (αB)
    result3 = contract(A, alpha * B, axes=(1, 0))
    
    # Physical tensors R@W should match (gauge freedom due to SU(2) component)
    assert_physical_tensors_equal(result1, result2, msg="scalar multiplication linearity (left)")
    assert_physical_tensors_equal(result1, result3, msg="scalar multiplication linearity (right)")


def test_contract_u1su2_scalar_multiplication_linearity_basic():
    """Test scalar multiplication linearity: α(A ⊗ B) = (αA) ⊗ B = A ⊗ (αB) for 3rd order."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=6210, itags=["a", "b", "c"])
    B = Tensor.random([idx_b.flip(), idx_d], seed=6211, itags=["b", "d"])
    populate_random_weights(A, seed=6212)
    populate_random_weights(B, seed=6213)
    
    alpha = 2.5
    
    AB = contract(A, B, axes=(1, 0))
    result1 = alpha * AB
    
    result2 = contract(alpha * A, B, axes=(1, 0))
    result3 = contract(A, alpha * B, axes=(1, 0))
    
    # For multi-index contractions, physical tensor R@W should match
    assert_physical_tensors_equal(result1, result2, msg="scalar multiplication linearity (left)")
    assert_physical_tensors_equal(result1, result3, msg="scalar multiplication linearity (right)")


def test_contract_u1su2_scalar_multiplication_linearity_high_order():
    """Test scalar multiplication linearity: α(A ⊗ B) = (αA) ⊗ B = A ⊗ (αB) for 5th order."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    idx_e = Index(Direction.OUT, group, sectors=(
        Sector((0, 1), 3), Sector((2, 0), 1)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=6220, itags=["a", "b", "c", "d", "e"])
    B = Tensor.random([idx_b.flip(), idx_c.flip()], seed=6221, itags=["b", "c"])
    populate_random_weights(A, seed=6222)
    populate_random_weights(B, seed=6223)
    
    alpha = 2.5
    
    AB = contract(A, B, axes=([1, 2], [0, 1]))
    result1 = alpha * AB
    
    result2 = contract(alpha * A, B, axes=([1, 2], [0, 1]))
    result3 = contract(A, alpha * B, axes=([1, 2], [0, 1]))
    
    # For multi-index contractions, physical tensor R@W should match
    assert_physical_tensors_equal(result1, result2, msg="scalar multiplication linearity (left)")
    assert_physical_tensors_equal(result1, result3, msg="scalar multiplication linearity (right)")


# Permutation commutativity tests

def test_contract_u1su2_permutation_commutativity_basic():
    """Test that permutation and contraction commute for U(1)×SU(2) tensors (3rd order)."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c], seed=6300, itags=["a", "b", "c"])
    B = Tensor.random([idx_b.flip(), idx_d], seed=6301, itags=["b", "d"])
    populate_random_weights(A, seed=6302)
    populate_random_weights(B, seed=6303)
    
    # Path 1: Contract then permute
    # A(a*, b, c*) ⊗ B(b*, d) → C(a*, c*, d)
    C1 = contract(A, B, axes=(1, 0))
    # Permute to (c*, d, a*)
    C1_perm = permute(C1, [1, 2, 0])
    
    # Path 2: Permute A then contract
    # Permute A to (c*, a*, b)
    A_perm = permute(A, [2, 0, 1])  # itags: ["c", "a", "b"]
    # A_perm(c*, a*, b) ⊗ B(b*, d) → C2(c*, a*, d)
    C2 = contract(A_perm, B, axes=(2, 0))
    # Permute to (c*, d, a*)
    C2_perm = permute(C2, [0, 2, 1])
    
    # Results should match
    assert list(C1_perm.itags) == list(C2_perm.itags) == ["c", "d", "a"]
    
    # Physical tensors R@W should match (gauge freedom due to SU(2) component)
    assert_physical_tensors_equal(C1_perm, C2_perm, msg="permutation commutativity")
    
    # Norms should match
    assert math.isclose(C1_perm.norm(), C2_perm.norm(), rel_tol=1e-10, abs_tol=1e-12)


def test_contract_u1su2_permutation_commutativity_4th_order():
    """Test that permutation and contraction commute for U(1)×SU(2) tensors (4th order)."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    idx_e = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=6310, itags=["a", "b", "c", "d"])
    B = Tensor.random([idx_b.flip(), idx_d.flip(), idx_e], seed=6311, itags=["b", "d", "e"])
    populate_random_weights(A, seed=6312)
    populate_random_weights(B, seed=6313)
    
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
    # Permute to (c*, e*, a*)
    C2_perm = permute(C2, [0, 2, 1])
    
    # Results should match
    assert list(C1_perm.itags) == list(C2_perm.itags) == ["c", "e", "a"]
    
    # Physical tensors R@W should match (gauge freedom due to SU(2) component)
    assert_physical_tensors_equal(C1_perm, C2_perm, msg="permutation commutativity")
    
    # Norms should match
    assert math.isclose(C1_perm.norm(), C2_perm.norm(), rel_tol=1e-10, abs_tol=1e-12)


def test_contract_u1su2_permutation_commutativity_5th_order():
    """Test that permutation and contraction commute for U(1)×SU(2) tensors (5th order)."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    idx_e = Index(Direction.OUT, group, sectors=(
        Sector((0, 1), 3), Sector((2, 0), 1)
    ))
    idx_f = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1), Sector((1, 1), 2)
    ))
    
    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e], seed=6320, itags=["a", "b", "c", "d", "e"])
    B = Tensor.random([idx_b.flip(), idx_d.flip(), idx_f], seed=6321, itags=["b", "d", "f"])
    populate_random_weights(A, seed=6322)
    populate_random_weights(B, seed=6323)
    
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
    
    # Physical tensors R@W should match (gauge freedom due to SU(2) component)
    assert_physical_tensors_equal(C1_perm, C2_perm, msg="permutation commutativity")
    
    # Norms should definitely match
    assert math.isclose(C1_perm.norm(), C2_perm.norm(), rel_tol=1e-10, abs_tol=1e-12)


# Bilinearity test

def test_contract_u1su2_bilinearity():
    """Test bilinearity: <αA + βB | C> = α<A|C> + β<B|C> for U(1)×SU(2) tensors."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    
    # Create 3rd order tensors for full contraction
    idx1 = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx2 = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx3 = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1), Sector((1, 0), 2)
    ))
    indices = [idx1, idx2, idx3]
    
    A = Tensor.random(indices, seed=6400, itags=["i", "j", "k"])
    B = Tensor.random(indices, seed=6401, itags=["i", "j", "k"])
    C = Tensor.random(indices, seed=6402, itags=["i", "j", "k"])
    populate_random_weights(A, seed=6403)
    populate_random_weights(B, seed=6404)
    populate_random_weights(C, seed=6405)
    
    alpha = 1.5
    beta = -0.8
    
    # Compute <αA + βB | C>
    linear_combo = alpha * A + beta * B
    linear_combo_conj = conj(linear_combo)
    result_combined = contract(linear_combo_conj, C, axes=([0, 1, 2], [0, 1, 2]))
    
    # Compute α<A|C> + β<B|C>
    A_conj = conj(A)
    B_conj = conj(B)
    scalar_AC = contract(A_conj, C, axes=([0, 1, 2], [0, 1, 2]))
    scalar_BC = contract(B_conj, C, axes=([0, 1, 2], [0, 1, 2]))
    result_separate = alpha * scalar_AC.data[()].item() + beta * scalar_BC.data[()].item()
    
    # Scalar results should match
    assert result_combined.is_scalar()
    assert math.isclose(result_combined.data[()].item(), result_separate, 
                       rel_tol=1e-10, abs_tol=1e-12), \
        "Bilinearity should hold for U(1)×SU(2) scalar products"


# ============================================================================
#  insert_index consistency: terminal insert vs leading insert + permute
# ============================================================================

def _make_insert_test_tensor(order: int, last_dir: Direction, seed: int) -> Tensor:
    """SU(2) tensor of given *order* whose last axis has direction *last_dir*."""
    group = SU2Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))
    last    = idx_out if last_dir == Direction.OUT else idx_in
    leading = [idx_out if i % 2 == 0 else idx_in for i in range(order - 1)]
    itags   = [chr(ord('a') + i) for i in range(order)]
    T = Tensor.random(leading + [last], seed=seed, itags=itags)
    populate_random_weights(T, seed=seed + 1)
    return T


def _assert_insert_terminal_equals_leading_permute(T: Tensor, inserted_dir: Direction) -> None:
    """Assert that inserting at the terminal position gives the same physical
    tensor as inserting at position 0 and then permuting the new axis to the end.

    Inserting at the terminal position demotes the previous terminal edge to a
    leading role, which can introduce a non-trivial recoupling phase. This test
    verifies that `Bridge.insert_edge` accounts for that phase via the R-symbol,
    making terminal and leading-then-permute insertion physically equivalent.
    """
    N = len(T.indices)

    # Path 1: insert directly at the terminal (last) position
    T1 = T.clone()
    T1.insert_index(N, inserted_dir, itag="x")

    # Path 2: insert at position 0 (leading), then permute "x" to the end
    T2 = T.clone()
    T2.insert_index(0, inserted_dir, itag="x")
    T2 = permute(T2, list(range(1, N + 1)) + [0])

    assert list(T1.itags) == list(T2.itags)
    assert_physical_tensors_equal(
        T1, T2,
        msg=f"insert terminal vs leading+permute, order={N}, inserted_dir={inserted_dir}",
    )


def test_insert_index_terminal_vs_permute_su2_2nd_order_last_out():
    """2nd-order SU(2) tensor, last axis OUT: terminal insert == leading insert + permute."""
    T = _make_insert_test_tensor(order=2, last_dir=Direction.OUT, seed=7000)
    for d in (Direction.OUT, Direction.IN):
        _assert_insert_terminal_equals_leading_permute(T, d)


def test_insert_index_terminal_vs_permute_su2_2nd_order_last_in():
    """2nd-order SU(2) tensor, last axis IN: terminal insert == leading insert + permute."""
    T = _make_insert_test_tensor(order=2, last_dir=Direction.IN, seed=7010)
    for d in (Direction.OUT, Direction.IN):
        _assert_insert_terminal_equals_leading_permute(T, d)


def test_insert_index_terminal_vs_permute_su2_3rd_order_last_out():
    """3rd-order SU(2) tensor, last axis OUT: terminal insert == leading insert + permute."""
    T = _make_insert_test_tensor(order=3, last_dir=Direction.OUT, seed=7020)
    for d in (Direction.OUT, Direction.IN):
        _assert_insert_terminal_equals_leading_permute(T, d)


def test_insert_index_terminal_vs_permute_su2_3rd_order_last_in():
    """3rd-order SU(2) tensor, last axis IN: terminal insert == leading insert + permute."""
    T = _make_insert_test_tensor(order=3, last_dir=Direction.IN, seed=7030)
    for d in (Direction.OUT, Direction.IN):
        _assert_insert_terminal_equals_leading_permute(T, d)


def test_insert_index_terminal_vs_permute_su2_4th_order_last_out():
    """4th-order SU(2) tensor, last axis OUT: terminal insert == leading insert + permute."""
    T = _make_insert_test_tensor(order=4, last_dir=Direction.OUT, seed=7040)
    for d in (Direction.OUT, Direction.IN):
        _assert_insert_terminal_equals_leading_permute(T, d)


def test_insert_index_terminal_vs_permute_su2_4th_order_last_in():
    """4th-order SU(2) tensor, last axis IN: terminal insert == leading insert + permute."""
    T = _make_insert_test_tensor(order=4, last_dir=Direction.IN, seed=7050)
    for d in (Direction.OUT, Direction.IN):
        _assert_insert_terminal_equals_leading_permute(T, d)


def test_insert_index_terminal_vs_permute_su2_6th_order_last_out():
    """6th-order SU(2) tensor, last axis OUT: terminal insert == leading insert + permute."""
    T = _make_insert_test_tensor(order=6, last_dir=Direction.OUT, seed=7060)
    for d in (Direction.OUT, Direction.IN):
        _assert_insert_terminal_equals_leading_permute(T, d)


def test_insert_index_terminal_vs_permute_su2_6th_order_last_in():
    """6th-order SU(2) tensor, last axis IN: terminal insert == leading insert + permute."""
    T = _make_insert_test_tensor(order=6, last_dir=Direction.IN, seed=7070)
    for d in (Direction.OUT, Direction.IN):
        _assert_insert_terminal_equals_leading_permute(T, d)


# ============================================================================
#  Outer product consistency: contract(axes=([],[])) vs insert_index
# ============================================================================

def _assert_outer_product_u1(A: Tensor, B: Tensor, msg: str = "") -> None:
    """Assert that outer product via empty axes agrees with insert_index + contract.

    Two paths to the outer product A⊗B:

    Path 1 – direct: ``contract(A, B, axes=([], []))``
        No indices are contracted; the result carries all indices of A followed
        by all indices of B.

    Path 2 – mediated: insert a trivial (neutral-charge, dim-1) OUT index "x"
        at the end of A, insert a trivial IN index "x" at the start of B, then
        contract the two trivial indices.  Because the trivial sector is the
        identity channel of U(1), this contraction squeezes the singleton
        dimension and reproduces the outer product exactly.
    """
    na = len(A.indices)

    # Path 1: outer product with no contractions
    result1 = contract(A, B, axes=([], []))

    # Path 2: insert trivial indices, then contract on them
    A_mod = A.clone()
    A_mod.insert_index(na, Direction.OUT, itag="x")
    B_mod = B.clone()
    B_mod.insert_index(0, Direction.IN, itag="x")
    result2 = contract(A_mod, B_mod, axes=(na, 0))

    assert list(result1.itags) == list(result2.itags), \
        f"{msg}: itags mismatch: {list(result1.itags)} vs {list(result2.itags)}"
    assert_charge_neutral(result1)
    assert_charge_neutral(result2)
    assert_blocks_equal(result1, result2)


def _assert_outer_product_su2(A: Tensor, B: Tensor, msg: str = "") -> None:
    """Assert that outer product via empty axes agrees with insert_index + contract for SU(2).

    Same two-path strategy as ``_assert_outer_product_u1``, but uses
    ``assert_physical_tensors_equal`` to tolerate the gauge freedom in the
    non-Abelian intertwiner weights.
    """
    na = len(A.indices)

    # Path 1: outer product with no contractions
    result1 = contract(A, B, axes=([], []))

    # Path 2: insert trivial j=0 indices, then contract on them
    A_mod = A.clone()
    A_mod.insert_index(na, Direction.OUT, itag="x")
    B_mod = B.clone()
    B_mod.insert_index(0, Direction.IN, itag="x")
    result2 = contract(A_mod, B_mod, axes=(na, 0))

    assert list(result1.itags) == list(result2.itags), \
        f"{msg}: itags mismatch: {list(result1.itags)} vs {list(result2.itags)}"
    assert_charge_neutral(result1)
    assert_charge_neutral(result2)
    assert_physical_tensors_equal(result1, result2, msg=msg)


# U(1) outer product tests (many sectors → many blocks)

def test_outer_product_consistency_u1_2nd_x_2nd():
    """U(1) outer product: 2nd-order × 2nd-order, 6 sectors per index (36 product blocks)."""
    u1 = U1Group()
    # Six sectors spanning charges −3…2; each factor tensor has one block per charge value.
    sectors = (
        Sector(-3, 2), Sector(-2, 2), Sector(-1, 3),
        Sector(0, 3), Sector(1, 2), Sector(2, 2),
    )
    idx_a = Index(Direction.OUT, u1, sectors=sectors)
    idx_b = Index(Direction.IN, u1, sectors=sectors)
    idx_c = Index(Direction.OUT, u1, sectors=sectors)
    idx_d = Index(Direction.IN, u1, sectors=sectors)

    A = Tensor.random([idx_a, idx_b], seed=8000, itags=["a", "b"])
    B = Tensor.random([idx_c, idx_d], seed=8001, itags=["c", "d"])

    _assert_outer_product_u1(A, B, msg="u1 2x2")


def test_outer_product_consistency_u1_2nd_x_3rd():
    """U(1) outer product: 2nd-order × 3rd-order, 5 sectors per index."""
    u1 = U1Group()
    sectors5 = (Sector(-2, 2), Sector(-1, 2), Sector(0, 3), Sector(1, 2), Sector(2, 2))

    idx_a = Index(Direction.OUT, u1, sectors=sectors5)
    idx_b = Index(Direction.IN, u1, sectors=sectors5)
    idx_c = Index(Direction.OUT, u1, sectors=sectors5)
    idx_d = Index(Direction.IN, u1, sectors=sectors5)
    idx_e = Index(Direction.OUT, u1, sectors=sectors5)

    A = Tensor.random([idx_a, idx_b], seed=8010, itags=["a", "b"])
    # 3rd-order tensor: blocks at (q_c, q_d, q_e) with q_c − q_d + q_e = 0 → ~19 blocks
    B = Tensor.random([idx_c, idx_d, idx_e], seed=8011, itags=["c", "d", "e"])

    _assert_outer_product_u1(A, B, msg="u1 2x3")


def test_outer_product_consistency_u1_3rd_x_2nd():
    """U(1) outer product: 3rd-order × 2nd-order, 5 sectors per index."""
    u1 = U1Group()
    sectors5 = (Sector(-2, 2), Sector(-1, 2), Sector(0, 3), Sector(1, 2), Sector(2, 2))

    idx_a = Index(Direction.OUT, u1, sectors=sectors5)
    idx_b = Index(Direction.IN, u1, sectors=sectors5)
    idx_c = Index(Direction.OUT, u1, sectors=sectors5)
    idx_d = Index(Direction.OUT, u1, sectors=sectors5)
    idx_e = Index(Direction.IN, u1, sectors=sectors5)

    # 3rd-order tensor: ~19 blocks
    A = Tensor.random([idx_a, idx_b, idx_c], seed=8020, itags=["a", "b", "c"])
    B = Tensor.random([idx_d, idx_e], seed=8021, itags=["d", "e"])

    _assert_outer_product_u1(A, B, msg="u1 3x2")


def test_outer_product_consistency_u1_3rd_x_3rd():
    """U(1) outer product: 3rd-order × 3rd-order, 5 sectors per index (~19×19 product blocks)."""
    u1 = U1Group()
    sectors5 = (Sector(-2, 2), Sector(-1, 2), Sector(0, 3), Sector(1, 2), Sector(2, 2))

    idx_a = Index(Direction.OUT, u1, sectors=sectors5)
    idx_b = Index(Direction.IN, u1, sectors=sectors5)
    idx_c = Index(Direction.OUT, u1, sectors=sectors5)
    idx_d = Index(Direction.OUT, u1, sectors=sectors5)
    idx_e = Index(Direction.IN, u1, sectors=sectors5)
    idx_f = Index(Direction.OUT, u1, sectors=sectors5)

    A = Tensor.random([idx_a, idx_b, idx_c], seed=8030, itags=["a", "b", "c"])
    B = Tensor.random([idx_d, idx_e, idx_f], seed=8031, itags=["d", "e", "f"])

    _assert_outer_product_u1(A, B, msg="u1 3x3")


# SU(2) outer product tests (moderate sector count)

def test_outer_product_consistency_su2_2nd_x_2nd():
    """SU(2) outer product: 2nd-order × 2nd-order, 3 spin sectors each."""
    group = SU2Group()
    # Three sectors: j = 0, 1, 2 with multiplicity 2, 3, 4.
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))

    A = Tensor.random([idx_a, idx_b], seed=8100, itags=["a", "b"])
    B = Tensor.random([idx_c, idx_d], seed=8101, itags=["c", "d"])
    populate_random_weights(A, seed=8102)
    populate_random_weights(B, seed=8103)

    _assert_outer_product_su2(A, B, msg="su2 2x2")


def test_outer_product_consistency_su2_2nd_x_3rd():
    """SU(2) outer product: 2nd-order × 3rd-order."""
    group = SU2Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx_a, idx_b], seed=8110, itags=["a", "b"])
    B = Tensor.random([idx_c, idx_d, idx_e], seed=8111, itags=["c", "d", "e"])
    populate_random_weights(A, seed=8112)
    populate_random_weights(B, seed=8113)

    _assert_outer_product_su2(A, B, msg="su2 2x3")


def test_outer_product_consistency_su2_3rd_x_2nd():
    """SU(2) outer product: 3rd-order × 2nd-order."""
    group = SU2Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_e = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))

    A = Tensor.random([idx_a, idx_b, idx_c], seed=8120, itags=["a", "b", "c"])
    B = Tensor.random([idx_d, idx_e], seed=8121, itags=["d", "e"])
    populate_random_weights(A, seed=8122)
    populate_random_weights(B, seed=8123)

    _assert_outer_product_su2(A, B, msg="su2 3x2")


def test_outer_product_consistency_su2_3rd_x_3rd():
    """SU(2) outer product: 3rd-order × 3rd-order."""
    group = SU2Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_e = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 4)))
    idx_f = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))

    A = Tensor.random([idx_a, idx_b, idx_c], seed=8130, itags=["a", "b", "c"])
    B = Tensor.random([idx_d, idx_e, idx_f], seed=8131, itags=["d", "e", "f"])
    populate_random_weights(A, seed=8132)
    populate_random_weights(B, seed=8133)

    _assert_outer_product_su2(A, B, msg="su2 3x3")
