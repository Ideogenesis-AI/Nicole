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
    U1Group, Z2Group, SU2Group,
)
from ..utils import (
    assert_blocks_equal, assert_charge_neutral,
    assert_physical_tensors_equal, populate_random_weights,
)


# ===========================================================================
# Permutation
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
# Trace
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
# Two-tensor contraction
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
# Chain (sequential) contraction
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
# Error cases
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
