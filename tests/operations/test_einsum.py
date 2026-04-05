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


"""Tests for the einsum function."""

import pytest
import torch

from nicole import (
    Direction, Index, Sector, Tensor,
    contract, einsum, permute, trace,
    U1Group, Z2Group, SU2Group, ProductGroup,
)
from ..utils import (
    assert_blocks_equal, assert_charge_neutral,
    assert_physical_tensors_equal, populate_random_weights,
)


# ===========================================================================
#  Basic Permutation
# ===========================================================================

def test_einsum_permutation_matches_permute():
    """einsum 'ij->ji' should match a direct permute call."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_j = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))

    A = Tensor.random([idx_i, idx_j], seed=10, itags=["i", "j"])

    result = einsum('ij->ji', A)
    expected = permute(A, [1, 0])

    assert list(result.itags) == ["j", "i"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_identity_permutation():
    """einsum 'ij->ij' should return an equivalent tensor unchanged."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_j = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))

    A = Tensor.random([idx_i, idx_j], seed=11, itags=["i", "j"])

    result = einsum('ij->ij', A)

    assert list(result.itags) == ["i", "j"]
    assert_blocks_equal(result, A)


def test_einsum_permutation_three_axes():
    """einsum 'ijk->kij' on an order-3 tensor matches a direct permute call."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_j = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 2)))

    A = Tensor.random([idx_i, idx_j, idx_k], seed=12, itags=["i", "j", "k"])

    result = einsum('ijk->kij', A)
    expected = permute(A, [2, 0, 1])

    assert list(result.itags) == ["k", "i", "j"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_permutation_z2():
    """einsum permutation works for a Z2 tensor."""
    group = Z2Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_j = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2)))

    A = Tensor.random([idx_i, idx_j], seed=20, itags=["i", "j"])

    result = einsum('ij->ji', A)
    expected = permute(A, [1, 0])

    assert list(result.itags) == ["j", "i"]
    assert_blocks_equal(result, expected)


def test_einsum_permutation_su2():
    """einsum permutation works for an SU(2) tensor."""
    group = SU2Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_j = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))

    A = Tensor.random([idx_i, idx_j], seed=30, itags=["i", "j"])
    populate_random_weights(A, seed=31)

    result = einsum('ij->ji', A)
    expected = permute(A, [1, 0])

    assert list(result.itags) == ["j", "i"]
    assert_physical_tensors_equal(result, expected)


# ===========================================================================
#  Basic Trace
# ===========================================================================

def test_einsum_trace_to_scalar_matches_trace():
    """einsum 'ii->' should match a direct trace call."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))

    A = Tensor.random([idx_out, idx_in], seed=40, itags=["x", "x"])

    result = einsum('ii->', A)
    expected = trace(A)

    assert result.is_scalar()
    assert torch.allclose(result.data[()], expected.data[()], rtol=1e-10, atol=1e-12)


def test_einsum_trace_with_free_axes_matches_trace():
    """einsum 'iijk->jk' on an order-4 tensor matches a direct trace call."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_j   = Index(Direction.IN,  group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_k   = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 2)))

    A = Tensor.random([idx_out, idx_in, idx_j, idx_k], seed=50, itags=["x", "x", "j", "k"])

    result = einsum('iijk->jk', A)
    expected = trace(A, axes=(0, 1))

    assert list(result.itags) == ["j", "k"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_trace_direction_mismatch_raises():
    """einsum should propagate an error when the trace pair has identical directions."""
    group = U1Group()
    # Both axes have the same direction — trace must reject this.
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_j   = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))

    A = Tensor.random([idx_out, idx_out, idx_j], seed=60, itags=["x", "x", "j"])

    with pytest.raises(ValueError):
        einsum('iij->', A)


def test_einsum_trace_triple_repeat_raises():
    """einsum should raise immediately if a letter appears three times."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_j   = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))

    A = Tensor.random([idx_out, idx_in, idx_j], seed=61, itags=["p", "q", "r"])

    with pytest.raises(ValueError, match="3"):
        einsum('iii->', A)


def test_einsum_trace_su2():
    """einsum trace-to-scalar works for an SU(2) tensor."""
    group = SU2Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))

    A = Tensor.random([idx_out, idx_in], seed=70, itags=["x", "x"])
    populate_random_weights(A, seed=71)

    result  = einsum('ii->', A)
    expected = trace(A)

    assert result.is_scalar()
    assert torch.allclose(result.data[()], expected.data[()], rtol=1e-10, atol=1e-12)


# ===========================================================================
#  Basic Two-tensor contraction
# ===========================================================================

def test_einsum_matmul_matches_contract():
    """einsum 'ij,jk->ik' should match a direct contract call."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_j_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_j_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_k = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(2, 2)))

    A = Tensor.random([idx_i, idx_j_out], seed=80, itags=["i", "j"])
    B = Tensor.random([idx_j_in, idx_k],  seed=81, itags=["j", "k"])

    result   = einsum('ij,jk->ik', A, B)
    expected = contract(A, B, axes=(1, 0))

    assert list(result.itags) == ["i", "k"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_matmul_with_output_permutation():
    """einsum 'ij,jk->ki' should contract then permute the output axes."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_j_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_j_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_k = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2)))

    A = Tensor.random([idx_i, idx_j_out], seed=82, itags=["i", "j"])
    B = Tensor.random([idx_j_in, idx_k],  seed=83, itags=["j", "k"])

    result   = einsum('ij,jk->ki', A, B)
    expected = permute(contract(A, B, axes=(1, 0)), [1, 0])

    assert list(result.itags) == ["k", "i"]
    assert_blocks_equal(result, expected)


def test_einsum_contract_to_scalar():
    """einsum 'ij,ji->' should fully contract two tensors to a scalar."""
    group = U1Group()
    idx_i_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_i_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_j_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_j_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))

    A = Tensor.random([idx_i_out, idx_j_out], seed=84, itags=["i", "j"])
    B = Tensor.random([idx_j_in, idx_i_in],   seed=85, itags=["j", "i"])

    result = einsum('ij,ji->', A, B)
    assert result.is_scalar()
    assert_charge_neutral(result)


def test_einsum_outer_product():
    """einsum 'ij,kl->ijkl' should produce an outer product with no contraction."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_j = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx_l = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(-1, 2)))

    A = Tensor.random([idx_i, idx_j], seed=90, itags=["i", "j"])
    B = Tensor.random([idx_k, idx_l], seed=91, itags=["k", "l"])

    result = einsum('ij,kl->ijkl', A, B)

    assert list(result.itags) == ["i", "j", "k", "l"]
    assert len(result.indices) == 4
    assert_charge_neutral(result)


def test_einsum_contraction_z2():
    """einsum two-tensor contraction works for a Z2 group."""
    group = Z2Group()
    idx_i    = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_j_out = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_j_in  = Index(Direction.IN,  group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_k    = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))

    A = Tensor.random([idx_i, idx_j_out], seed=100, itags=["i", "j"])
    B = Tensor.random([idx_j_in, idx_k],  seed=101, itags=["j", "k"])

    result   = einsum('ij,jk->ik', A, B)
    expected = contract(A, B, axes=(1, 0))

    assert list(result.itags) == ["i", "k"]
    assert_blocks_equal(result, expected)


def test_einsum_contraction_su2():
    """einsum two-tensor contraction works for an SU(2) group."""
    group = SU2Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_j = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx_i, idx_j],       seed=110, itags=["i", "j"])
    B = Tensor.random([idx_j.flip(), idx_k], seed=111, itags=["j", "k"])
    populate_random_weights(A, seed=112)
    populate_random_weights(B, seed=113)

    result   = einsum('ij,jk->ik', A, B)
    expected = contract(A, B, axes=(1, 0))

    assert list(result.itags) == ["i", "k"]
    assert_physical_tensors_equal(result, expected)
    assert_charge_neutral(result)


# ===========================================================================
#  Chain (sequential) contraction
# ===========================================================================

def test_einsum_chain_contraction_matches_sequential_contract():
    """einsum 'ij,jk,kl->il' should match two sequential contract calls."""
    group = U1Group()
    idx_i    = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_j_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_j_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_k_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_k_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_l    = Index(Direction.IN,  group, sectors=(Sector(0, 3), Sector(1, 2)))

    A = Tensor.random([idx_i, idx_j_out], seed=120, itags=["i", "j"])
    B = Tensor.random([idx_j_in, idx_k_out], seed=121, itags=["j", "k"])
    C = Tensor.random([idx_k_in, idx_l],  seed=122, itags=["k", "l"])

    result   = einsum('ij,jk,kl->il', A, B, C)
    expected = contract(contract(A, B, axes=(1, 0)), C, axes=(1, 0))

    assert list(result.itags) == ["i", "l"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_chain_contraction_su2():
    """einsum chain contraction works for an SU(2) group."""
    group = SU2Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_j = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_l = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx_i, idx_j],        seed=130, itags=["i", "j"])
    B = Tensor.random([idx_j.flip(), idx_k],  seed=131, itags=["j", "k"])
    C = Tensor.random([idx_k.flip(), idx_l],  seed=132, itags=["k", "l"])
    populate_random_weights(A, seed=133)
    populate_random_weights(B, seed=134)
    populate_random_weights(C, seed=135)

    result   = einsum('ij,jk,kl->il', A, B, C)
    expected = contract(contract(A, B, axes=(1, 0)), C, axes=(1, 0))

    assert list(result.itags) == ["i", "l"]
    assert_physical_tensors_equal(result, expected)
    assert_charge_neutral(result)


# ===========================================================================
#  Error handling
# ===========================================================================

def test_einsum_missing_arrow_raises():
    """einsum without '->' in the equation should raise ValueError."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_j = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))
    A = Tensor.random([idx_i, idx_j], seed=200, itags=["i", "j"])

    with pytest.raises(ValueError, match="->"):
        einsum('ij', A)


def test_einsum_subscript_count_mismatch_raises():
    """einsum with mismatched subscript count should raise ValueError."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_j = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))
    A = Tensor.random([idx_i, idx_j], seed=201, itags=["i", "j"])

    with pytest.raises(ValueError):
        einsum('ij,kl->ij', A)  # two subscripts but only one tensor


def test_einsum_subscript_length_mismatch_raises():
    """einsum with subscript length not matching tensor order should raise."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_j = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))
    A = Tensor.random([idx_i, idx_j], seed=202, itags=["i", "j"])

    with pytest.raises(ValueError):
        einsum('ijk->ijk', A)  # subscript length 3 but tensor has order 2


def test_einsum_contraction_itag_mismatch_raises():
    """einsum should propagate an error when contracted axes have mismatched itags."""
    group = U1Group()
    idx_i    = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_j_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_j_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_k    = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2)))

    # A has itag "j" on axis 1, B has itag "p" on axis 0 — mismatch!
    A = Tensor.random([idx_i, idx_j_out], seed=210, itags=["i", "j"])
    B = Tensor.random([idx_j_in, idx_k],  seed=211, itags=["p", "k"])

    with pytest.raises(ValueError):
        einsum('ij,jk->ik', A, B)


# ===========================================================================
#  Higher-order permutations
# ===========================================================================

def test_einsum_permutation_4th_order():
    """einsum 'abcd->dcba' on a 4th order U(1) tensor matches permute([3,2,1,0])."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))
    idx_b = Index(Direction.IN,  group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(-2,3), Sector(-1,2), Sector(0,4), Sector(1,2), Sector(2,3)))
    idx_d = Index(Direction.IN,  group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=300, itags=["a", "b", "c", "d"])

    result   = einsum('abcd->dcba', A)
    expected = permute(A, [3, 2, 1, 0])

    assert list(result.itags) == ["d", "c", "b", "a"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_permutation_6th_order():
    """einsum 'abcdef->fedcba' on a 6th order U(1) tensor matches permute([5,4,3,2,1,0])."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_b = Index(Direction.IN,  group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))
    idx_d = Index(Direction.IN,  group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3)))
    idx_f = Index(Direction.IN,  group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e, idx_f],
                      seed=301, itags=["a", "b", "c", "d", "e", "f"])

    result   = einsum('abcdef->fedcba', A)
    expected = permute(A, [5, 4, 3, 2, 1, 0])

    assert list(result.itags) == ["f", "e", "d", "c", "b", "a"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_permutation_4th_order_su2():
    """einsum 'abcd->cdab' on a 4th order SU(2) tensor matches permute([2,3,0,1])."""
    group = SU2Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_d = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=302, itags=["a", "b", "c", "d"])
    populate_random_weights(A, seed=303)

    result   = einsum('abcd->cdab', A)
    expected = permute(A, [2, 3, 0, 1])

    assert list(result.itags) == ["c", "d", "a", "b"]
    assert_physical_tensors_equal(result, expected)


# ===========================================================================
#  Higher-order partial traces
# ===========================================================================

def test_einsum_partial_trace_6th_order():
    """einsum 'aabcde->bcde' traces axes 0,1 of a 6th order U(1) tensor to 4th order."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))
    idx_b   = Index(Direction.IN,  group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_c   = Index(Direction.OUT, group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))
    idx_d   = Index(Direction.IN,  group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_e   = Index(Direction.OUT, group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3)))

    A = Tensor.random([idx_out, idx_in, idx_b, idx_c, idx_d, idx_e],
                      seed=310, itags=["x", "x", "b", "c", "d", "e"])

    result   = einsum('aabcde->bcde', A)
    expected = trace(A, axes=(0, 1))

    assert list(result.itags) == ["b", "c", "d", "e"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_middle_trace_6th_order():
    """einsum 'abiicd->abcd' traces axes 2,3 of a 6th order U(1) tensor to 4th order."""
    group = U1Group()
    idx_a   = Index(Direction.OUT, group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_b   = Index(Direction.IN,  group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3)))
    idx_out = Index(Direction.OUT, group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))
    idx_c   = Index(Direction.IN,  group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_d   = Index(Direction.OUT, group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3)))

    A = Tensor.random([idx_a, idx_b, idx_out, idx_in, idx_c, idx_d],
                      seed=311, itags=["a", "b", "x", "x", "c", "d"])

    result   = einsum('abiicd->abcd', A)
    expected = trace(A, axes=(2, 3))

    assert list(result.itags) == ["a", "b", "c", "d"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_partial_trace_4th_order_su2():
    """einsum 'iiab->ab' traces a 4th order SU(2) tensor to 2nd order."""
    group = SU2Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_a   = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_b   = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx_out, idx_in, idx_a, idx_b], seed=312, itags=["x", "x", "a", "b"])
    populate_random_weights(A, seed=313)

    result   = einsum('iiab->ab', A)
    expected = trace(A, axes=(0, 1))

    assert list(result.itags) == ["a", "b"]
    assert_physical_tensors_equal(result, expected)


# ===========================================================================
#  Multi-index contraction — non-trivial axis order
# ===========================================================================

def test_einsum_multi_index_contract_3rd_order():
    """einsum 'ijk,kjl->il' contracts with reversed shared-index order in B.

    j is at axis 1 of A and axis 1 of B; k is at axis 2 of A and axis 0 of B,
    so the effective contraction is axes_A=[1,2], axes_B=[1,0], which is
    non-trivial (NOT last-n of A contracted with first-n of B).
    """
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))
    idx_j = Index(Direction.OUT, group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))
    idx_l = Index(Direction.IN,  group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))

    A = Tensor.random([idx_i, idx_j, idx_k],          seed=320, itags=["i", "j", "k"])
    B = Tensor.random([idx_k.flip(), idx_j.flip(), idx_l], seed=321, itags=["k", "j", "l"])

    result   = einsum('ijk,kjl->il', A, B)
    expected = contract(A, B, axes=([1, 2], [1, 0]))

    assert list(result.itags) == ["i", "l"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_multi_index_contract_3rd_order_su2():
    """einsum 'ijk,kjl->il' with non-trivial axis order for SU(2) tensors."""
    group = SU2Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_j = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_l = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))

    A = Tensor.random([idx_i, idx_j, idx_k],          seed=322, itags=["i", "j", "k"])
    B = Tensor.random([idx_k.flip(), idx_j.flip(), idx_l], seed=323, itags=["k", "j", "l"])
    populate_random_weights(A, seed=324)
    populate_random_weights(B, seed=325)

    result   = einsum('ijk,kjl->il', A, B)
    expected = contract(A, B, axes=([1, 2], [1, 0]))

    assert list(result.itags) == ["i", "l"]
    assert_physical_tensors_equal(result, expected)
    assert_charge_neutral(result)


# ===========================================================================
#  Higher-order contraction — non-trivial axis order
# ===========================================================================

def test_einsum_contract_4th_order():
    """einsum 'ijkl,kjm->ilm' contracts 4th order × 3rd order with non-trivial axis order.

    j is at axis 1 of A and axis 1 of B; k is at axis 2 of A and axis 0 of B,
    giving axes_A=[1,2], axes_B=[1,0]. Result is 3rd order: [i, l, m].
    """
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))
    idx_j = Index(Direction.OUT, group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))
    idx_l = Index(Direction.IN,  group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))
    idx_m = Index(Direction.IN,  group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))

    A = Tensor.random([idx_i, idx_j, idx_k, idx_l],       seed=330, itags=["i", "j", "k", "l"])
    B = Tensor.random([idx_k.flip(), idx_j.flip(), idx_m], seed=331, itags=["k", "j", "m"])

    result   = einsum('ijkl,kjm->ilm', A, B)
    expected = contract(A, B, axes=([1, 2], [1, 0]))

    assert list(result.itags) == ["i", "l", "m"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_contract_4th_order_su2():
    """einsum 'ijkl,kjm->ilm' with non-trivial axis order for SU(2) tensors."""
    group = SU2Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_j = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_l = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_m = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx_i, idx_j, idx_k, idx_l],       seed=332, itags=["i", "j", "k", "l"])
    B = Tensor.random([idx_k.flip(), idx_j.flip(), idx_m], seed=333, itags=["k", "j", "m"])
    populate_random_weights(A, seed=334)
    populate_random_weights(B, seed=335)

    result   = einsum('ijkl,kjm->ilm', A, B)
    expected = contract(A, B, axes=([1, 2], [1, 0]))

    assert list(result.itags) == ["i", "l", "m"]
    assert_physical_tensors_equal(result, expected)
    assert_charge_neutral(result)


# ===========================================================================
#  4th order chain contraction
# ===========================================================================

def test_einsum_4th_order_chain():
    """einsum 'ijkl,jmkn,mnop->ilop' chains three 4th order U(1) tensors.

    Step 1: A('ijkl') x B('jmkn'): j at pos 1 of A and pos 0 of B;
    k at pos 2 of A and pos 2 of B — axes_A=[1,2], axes_B=[0,2] (non-trivial).
    Step 2: intermediate('ilmn') x C('mnop'): axes=[2,3],[0,1].
    """
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))
    idx_j = Index(Direction.OUT, group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))
    idx_l = Index(Direction.IN,  group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))
    idx_m = Index(Direction.OUT, group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2)))
    idx_n = Index(Direction.OUT, group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))
    idx_o = Index(Direction.IN,  group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2)))
    idx_p = Index(Direction.IN,  group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))

    A = Tensor.random([idx_i, idx_j, idx_k, idx_l],              seed=340, itags=["i", "j", "k", "l"])
    B = Tensor.random([idx_j.flip(), idx_m, idx_k.flip(), idx_n], seed=341, itags=["j", "m", "k", "n"])
    C = Tensor.random([idx_m.flip(), idx_n.flip(), idx_o, idx_p], seed=342, itags=["m", "n", "o", "p"])

    result   = einsum('ijkl,jmkn,mnop->ilop', A, B, C)
    expected = contract(contract(A, B, axes=([1, 2], [0, 2])), C, axes=([2, 3], [0, 1]))

    assert list(result.itags) == ["i", "l", "o", "p"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_4th_order_chain_su2():
    """einsum 'ijk,kjl,lmn->imn' chains three 3rd order SU(2) tensors.

    Step 1: A('ijk') x B('kjl'): j at pos 1 of A and pos 1 of B;
    k at pos 2 of A and pos 0 of B — axes_A=[1,2], axes_B=[1,0] (non-trivial).
    Step 2: intermediate('il') x C('lmn'): axes=[1],[0].
    """
    group = SU2Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_j = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_l = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_m = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_n = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))

    A = Tensor.random([idx_i, idx_j, idx_k],              seed=343, itags=["i", "j", "k"])
    B = Tensor.random([idx_k.flip(), idx_j.flip(), idx_l], seed=344, itags=["k", "j", "l"])
    C = Tensor.random([idx_l.flip(), idx_m, idx_n],        seed=345, itags=["l", "m", "n"])
    populate_random_weights(A, seed=346)
    populate_random_weights(B, seed=347)
    populate_random_weights(C, seed=348)

    result   = einsum('ijk,kjl,lmn->imn', A, B, C)
    expected = contract(contract(A, B, axes=([1, 2], [1, 0])), C, axes=(1, 0))

    assert list(result.itags) == ["i", "m", "n"]
    assert_physical_tensors_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_4tensor_chain_cross_u1xsu2():
    """einsum 'ijk,jlm,lkn,no->imo' chains four tensors with U(1)xSU(2) group.

    The 3rd tensor C contracts with indices originating from both A (k) and B (l).
    C is subscripted 'lkn', so k appears at pos 1 and l at pos 0 — after step 1
    produces intermediate subscript [i,k,l,m], step 2 contracts k (pos 1) and
    l (pos 2) against C with axes_B=[1,0], a non-trivial axis order.
    """
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])

    sec_4 = (Sector((0, 0), 2), Sector((1, 0), 2), Sector((0, 1), 3), Sector((1, 1), 2))
    sec_3 = (Sector((0, 0), 2), Sector((1, 0), 1), Sector((0, 1), 3))
    sec_2 = (Sector((0, 0), 1), Sector((1, 1), 2))

    idx_i = Index(Direction.OUT, group, sectors=sec_4)
    idx_j = Index(Direction.OUT, group, sectors=sec_3)
    idx_k = Index(Direction.OUT, group, sectors=sec_2)
    idx_l = Index(Direction.OUT, group, sectors=sec_3)
    idx_m = Index(Direction.IN,  group, sectors=sec_4)
    idx_n = Index(Direction.OUT, group, sectors=sec_2)
    idx_o = Index(Direction.IN,  group, sectors=sec_4)

    A = Tensor.random([idx_i, idx_j, idx_k],              seed=374, itags=["i", "j", "k"])
    B = Tensor.random([idx_j.flip(), idx_l, idx_m],        seed=375, itags=["j", "l", "m"])
    C = Tensor.random([idx_l.flip(), idx_k.flip(), idx_n], seed=376, itags=["l", "k", "n"])
    D = Tensor.random([idx_n.flip(), idx_o],               seed=377, itags=["n", "o"])
    populate_random_weights(A, seed=378)
    populate_random_weights(B, seed=379)
    populate_random_weights(C, seed=380)
    populate_random_weights(D, seed=381)

    result = einsum('ijk,jlm,lkn,no->imo', A, B, C, D)
    ab       = contract(A, B, axes=(1, 0))
    abc      = contract(ab, C, axes=([1, 2], [1, 0]))
    expected = contract(abc, D, axes=(2, 0))

    assert list(result.itags) == ["i", "m", "o"]
    assert_physical_tensors_equal(result, expected)
    assert_charge_neutral(result)


# ===========================================================================
#  Mixed within-tensor trace then contraction
# ===========================================================================

def test_einsum_trace_then_contract():
    """einsum 'iijk,kl->jl' traces axes 0,1 of A then contracts on k."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3), Sector(2,2)))
    idx_j   = Index(Direction.IN,  group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_k   = Index(Direction.OUT, group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))
    idx_l   = Index(Direction.IN,  group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))

    A = Tensor.random([idx_out, idx_in, idx_j, idx_k], seed=350, itags=["x", "x", "j", "k"])
    B = Tensor.random([idx_k.flip(), idx_l],            seed=351, itags=["k", "l"])

    result   = einsum('iijk,kl->jl', A, B)
    expected = contract(trace(A, axes=(0, 1)), B, axes=(1, 0))

    assert list(result.itags) == ["j", "l"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_trace_then_contract_su2():
    """einsum 'iiab,bc->ac' traces axes 0,1 of a 4th order SU(2) tensor then contracts on b."""
    group = SU2Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_a   = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_b   = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_c   = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx_out, idx_in, idx_a, idx_b], seed=352, itags=["x", "x", "a", "b"])
    B = Tensor.random([idx_b.flip(), idx_c],            seed=353, itags=["b", "c"])
    populate_random_weights(A, seed=354)
    populate_random_weights(B, seed=355)

    result   = einsum('iiab,bc->ac', A, B)
    expected = contract(trace(A, axes=(0, 1)), B, axes=(1, 0))

    assert list(result.itags) == ["a", "c"]
    assert_physical_tensors_equal(result, expected)
    assert_charge_neutral(result)


# ===========================================================================
#  Outer product with permuted output
# ===========================================================================

def test_einsum_outer_product_permuted_output():
    """einsum 'ij,kl->klij' produces a permuted outer product for U(1) tensors."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))
    idx_j = Index(Direction.IN,  group, sectors=(Sector(-2,2), Sector(-1,3), Sector(0,4), Sector(1,3)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(-1,3), Sector(0,2), Sector(1,3)))
    idx_l = Index(Direction.IN,  group, sectors=(Sector(-1,2), Sector(0,3), Sector(1,2), Sector(2,3)))

    A = Tensor.random([idx_i, idx_j], seed=360, itags=["i", "j"])
    B = Tensor.random([idx_k, idx_l], seed=361, itags=["k", "l"])

    result   = einsum('ij,kl->klij', A, B)
    expected = permute(contract(A, B, axes=([], [])), [2, 3, 0, 1])

    assert list(result.itags) == ["k", "l", "i", "j"]
    assert_blocks_equal(result, expected)
    assert_charge_neutral(result)


def test_einsum_outer_product_permuted_output_su2():
    """einsum 'ij,kl->klij' produces a permuted outer product for SU(2) tensors."""
    group = SU2Group()
    idx_i = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_j = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_k = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_l = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx_i, idx_j], seed=362, itags=["i", "j"])
    B = Tensor.random([idx_k, idx_l], seed=363, itags=["k", "l"])
    populate_random_weights(A, seed=364)
    populate_random_weights(B, seed=365)

    result   = einsum('ij,kl->klij', A, B)
    expected = permute(contract(A, B, axes=([], [])), [2, 3, 0, 1])

    assert list(result.itags) == ["k", "l", "i", "j"]
    assert_physical_tensors_equal(result, expected)
    assert_charge_neutral(result)
