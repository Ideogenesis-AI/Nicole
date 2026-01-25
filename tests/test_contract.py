# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Nicole (TN) library.
#
# Nicole (TN) is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Nicole (TN) is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Nicole (TN). If not, see <https://www.gnu.org/licenses/>.


"""Tests for tensor contraction operations: contract, trace, partial_trace."""

import numpy as np
import pytest

from nicole import Direction, Tensor, contract, identity, trace, U1Group, Z2Group, permute, Index, Sector
from nicole.symmetry.product import ProductGroup
from .utils import assert_charge_neutral


# Basic contraction tests

def test_contract_two_tensors_manual_pairs():
    """Test two-tensor contraction with manual pairs matches block-wise computation."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_b_left = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_b_right = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(2, 2)))

    A = Tensor.random([idx_a, idx_b_left], seed=20, itags=["a", "b"])
    B = Tensor.random([idx_b_right, idx_c], seed=21, itags=["b", "c"])

    result = contract(A, B, axes=(1, 0))
    assert_charge_neutral(result)

    # Manual blockwise contraction
    manual = {}
    for (qa, qb_left), block_a in A.data.items():
        for (qb_right, qc), block_b in B.data.items():
            if idx_b_left.group.equal(qb_left, qb_right):
                out_key = (qa, qc)
                contracted = np.tensordot(block_a, block_b, axes=(1, 0))
                manual[out_key] = manual.get(out_key, 0) + contracted

    assert set(result.data.keys()) == set(manual.keys())
    for key in manual:
        np.testing.assert_allclose(result.data[key], manual[key])


def test_contract_automatic_detection():
    """Test automatic contraction pair detection."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 1)))

    idx_c_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))

    A = Tensor.random([idx_a, idx_b_out, idx_c_out], seed=101, itags=["a", "b", "c"])
    B = Tensor.random([idx_c_in, idx_b_in, idx_d], seed=102, itags=["c", "b", "d"])

    # Automatic detection will find matching itags with opposite directions
    result = contract(A, B)
    assert list(result.itags) == ["a", "d"]
    assert_charge_neutral(result)

    manual = {}
    for (qa, qb, qc), block_a in A.data.items():
        for (qc2, qb2, qd), block_b in B.data.items():
            if group.equal(qb, qb2) and group.equal(qc, qc2):
                out_key = (qa, qd)
                contracted = np.tensordot(block_a, block_b, axes=([1, 2], [1, 0]))
                manual[out_key] = manual.get(out_key, 0) + contracted

    assert set(result.data.keys()) == set(manual)
    for key in manual:
        np.testing.assert_allclose(result.data[key], manual[key])


def test_contract_high_order_two_pairs():
    """Test high-order tensor contraction with two index pairs - manual verification."""
    group = U1Group()
    # A: 4 indices (a, b, c, d) - contract b and d with B
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_d_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    
    # B: 4 indices (d, e, b, f) - contract d and b with A
    idx_d_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_f = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b_out, idx_c, idx_d_out], seed=1001, itags=["a", "b", "c", "d"])
    B = Tensor.random([idx_d_in, idx_e, idx_b_in, idx_f], seed=1002, itags=["d", "e", "b", "f"])
    
    # Automatic detection should contract b and d
    result = contract(A, B)
    assert set(result.itags) == {"a", "c", "e", "f"}
    assert_charge_neutral(result)
    
    # Manual block-wise computation
    manual = {}
    for (qa, qb, qc, qd), block_a in A.data.items():
        for (qd2, qe, qb2, qf), block_b in B.data.items():
            # Check if charges match for contraction
            if group.equal(qb, qb2) and group.equal(qd, qd2):
                out_key = (qa, qc, qe, qf)
                # A: (a, b, c, d), B: (d, e, b, f)
                # Contract: b (axis 1 of A) with b (axis 2 of B)
                #           d (axis 3 of A) with d (axis 0 of B)
                # Result order: (a, c, e, f)
                contracted = np.einsum('abcd,debf->acef', block_a, block_b)
                
                if out_key not in manual:
                    manual[out_key] = contracted
                else:
                    manual[out_key] = manual[out_key] + contracted
    
    # Verify results match
    assert set(result.data.keys()) == set(manual.keys()), \
        f"Block keys mismatch: result has {set(result.data.keys())}, manual has {set(manual.keys())}"
    for key in manual:
        np.testing.assert_allclose(result.data[key], manual[key], rtol=1e-10, atol=1e-12)


def test_contract_high_order_three_pairs():
    """Test high-order tensor contraction with three index pairs - manual verification."""
    group = U1Group()
    # A: 5 indices (a, b, c, d, e) - contract b, c, e with B
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_c_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_e_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(2, 2)))
    
    # B: 5 indices (e, f, c, b, g) - contract e, c, b with A
    idx_e_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(2, 2)))
    idx_f = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_c_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_g = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b_out, idx_c_out, idx_d, idx_e_out], seed=2001, itags=["a", "b", "c", "d", "e"])
    B = Tensor.random([idx_e_in, idx_f, idx_c_in, idx_b_in, idx_g], seed=2002, itags=["e", "f", "c", "b", "g"])
    
    # Contract b, c, e
    result = contract(A, B)
    assert set(result.itags) == {"a", "d", "f", "g"}
    assert_charge_neutral(result)
    
    # Manual block-wise computation
    manual = {}
    for (qa, qb, qc, qd, qe), block_a in A.data.items():
        for (qe2, qf, qc2, qb2, qg), block_b in B.data.items():
            # Check if charges match for contraction
            if group.equal(qb, qb2) and group.equal(qc, qc2) and group.equal(qe, qe2):
                out_key = (qa, qd, qf, qg)
                # A: (a, b, c, d, e), B: (e, f, c, b, g)
                # Contract: b (axis 1 of A) with b (axis 3 of B)
                #           c (axis 2 of A) with c (axis 2 of B)
                #           e (axis 4 of A) with e (axis 0 of B)
                # Result order: (a, d, f, g)
                contracted = np.einsum('abcde,efcbg->adfg', block_a, block_b)
                
                if out_key not in manual:
                    manual[out_key] = contracted
                else:
                    manual[out_key] = manual[out_key] + contracted
    
    # Verify results match
    assert set(result.data.keys()) == set(manual.keys()), \
        f"Block keys mismatch: result has {set(result.data.keys())}, manual has {set(manual.keys())}"
    for key in manual:
        np.testing.assert_allclose(result.data[key], manual[key], rtol=1e-10, atol=1e-12)


def test_contract_high_order_asymmetric():
    """Test high-order asymmetric contraction (3-index x 6-index) - manual verification."""
    group = U1Group()
    # A: 3 indices (a, b, c) - contract a and c with B
    idx_a_out = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_c_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    
    # B: 6 indices (c, d, e, a, f, g) - contract c and a with A
    idx_c_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx_d = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_e = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_a_in = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_f = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_g = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    
    A = Tensor.random([idx_a_out, idx_b, idx_c_out], seed=3001, itags=["a", "b", "c"])
    B = Tensor.random([idx_c_in, idx_d, idx_e, idx_a_in, idx_f, idx_g], seed=3002, itags=["c", "d", "e", "a", "f", "g"])
    
    # Contract a and c
    result = contract(A, B)
    assert set(result.itags) == {"b", "d", "e", "f", "g"}
    assert_charge_neutral(result)
    
    # Manual block-wise computation
    manual = {}
    for (qa, qb, qc), block_a in A.data.items():
        for (qc2, qd, qe, qa2, qf, qg), block_b in B.data.items():
            # Check if charges match for contraction
            if group.equal(qa, qa2) and group.equal(qc, qc2):
                out_key = (qb, qd, qe, qf, qg)
                # A: (a, b, c), B: (c, d, e, a, f, g)
                # Contract: a (axis 0 of A) with a (axis 3 of B)
                #           c (axis 2 of A) with c (axis 0 of B)
                # Result order: (b, d, e, f, g)
                contracted = np.einsum('abc,cdeafg->bdefg', block_a, block_b)
                
                if out_key not in manual:
                    manual[out_key] = contracted
                else:
                    manual[out_key] = manual[out_key] + contracted
    
    # Verify results match
    assert set(result.data.keys()) == set(manual.keys()), \
        f"Block keys mismatch: result has {set(result.data.keys())}, manual has {set(manual.keys())}"
    for key in manual:
        np.testing.assert_allclose(result.data[key], manual[key], rtol=1e-10, atol=1e-12)


def test_contract_named_vs_positional():
    """Test that named axis automatic detection matches position-based pairs."""
    group = U1Group()
    idx_left = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_mid_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx_right_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-2, 2)))

    idx_mid_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx_end = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(2, 2)))

    A = Tensor.random([idx_left, idx_mid_out, idx_right_in], seed=201, itags=["L", "M", "R"])
    B = Tensor.random([idx_right_in.dual(), idx_mid_in, idx_end], seed=202, itags=["R", "M", "E"])

    # Automatic detection
    named = contract(A, B)
    positional = contract(A, B, axes=([2, 1], [0, 1]))

    assert list(named.itags) == list(positional.itags)
    for key in named.data:
        np.testing.assert_allclose(named.data[key], positional.data[key])


def test_contract_with_perm():
    """Test contraction with permutation parameter."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1),))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx_a, idx_b_out], seed=1, itags=["a", "b"])
    B = Tensor.random([idx_b_in, idx_c], seed=2, itags=["b", "c"])
    
    result = contract(A, B, axes=(1, 0), perm=[1, 0])
    
    assert list(result.itags) == ["c", "a"]


# Associativity and composition tests

def test_contract_three_tensor_associativity():
    """Test that tensor contraction is associative."""
    group = U1Group()
    idx_x = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_y_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_y_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_z_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx_z_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx_w = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(2, 2)))

    A = Tensor.random([idx_x, idx_y_out], seed=301, itags=["X", "Y"])
    B = Tensor.random([idx_y_in, idx_z_out], seed=302, itags=["Y", "Z"])
    C = Tensor.random([idx_z_in, idx_w], seed=303, itags=["Z", "W"])

    # Automatic detection
    left = contract(contract(A, B), C)
    right = contract(A, contract(B, C))

    assert list(left.itags) == list(right.itags)
    for key in left.data:
        np.testing.assert_allclose(left.data[key], right.data[key])


def test_contract_with_identity():
    """Test that contracting with identity matches direct contraction."""
    group = U1Group()
    idx_left = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_mid_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_mid_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_right = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(2, 2)))

    A = Tensor.random([idx_left, idx_mid_in], seed=401, itags=["L", "M"])
    B = Tensor.random([idx_mid_out, idx_right], seed=402, itags=["M", "R"])

    # Use matching itags for the identity
    I = identity(idx_mid_out, itags=("M", "M"))
    step = contract(A, I)
    bridge = contract(step, B)

    direct = contract(A, B)

    assert list(bridge.itags) == list(direct.itags)
    for key in bridge.data:
        np.testing.assert_allclose(bridge.data[key], direct.data[key])


def test_contract_after_permuting():
    """Test contraction after permuting axes."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 2),))

    A = Tensor.random([idx_a, idx_b], seed=801, itags=["a", "b"])
    B = Tensor.random([idx_c, idx_d], seed=802, itags=["b", "d"])  # Use "b" instead of "c"

    # permute B so matching axis moves
    permuted_B = permute(B, [1, 0])
    res1 = contract(A, B)  # Automatic detection
    res2 = contract(A, permuted_B)  # Automatic detection

    assert list(res1.itags) == list(res2.itags)
    for key in res1.data:
        np.testing.assert_allclose(res1.data[key], res2.data[key])


# Edge cases and error handling

def test_contract_empty_result_mismatched_dimensions():
    """Test contraction with mismatched dimensions gives empty result."""
    group = U1Group()
    idx_left = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_mid_out = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx_mid_in_bad = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    idx_right = Index(Direction.IN, group, sectors=(Sector(0, 2),))

    A = Tensor.random([idx_left, idx_mid_out], seed=501, itags=["L", "M"])
    B_bad = Tensor.random([idx_mid_in_bad, idx_right], seed=503, itags=["M", "R"])

    # Automatic detection for bad dimensions (will have empty result)
    result_bad = contract(A, B_bad)
    assert result_bad.data == {}


def test_contract_empty_result_no_matching_charge():
    """Test contraction with no matching charges gives empty result."""
    group = U1Group()
    idx_left = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_mid_out = Index(Direction.OUT, group, sectors=(Sector(1, 1),))
    idx_mid_in = Index(Direction.IN, group, sectors=(Sector(0, 1),))
    idx_right = Index(Direction.IN, group, sectors=(Sector(0, 2),))

    A = Tensor.random([idx_left, idx_mid_out], seed=701, itags=["L", "M"])
    B = Tensor.random([idx_mid_in, idx_right], seed=702, itags=["M", "R"])

    # Automatic detection
    result = contract(A, B)
    assert result.data == {}


def test_contract_no_pairs_raises():
    """Test that contract without valid pairs raises error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "c"])
    B = Tensor.random([idx, idx.flip()], seed=2, itags=["b", "d"])
    
    with pytest.raises(ValueError, match="No valid contraction pairs"):
        contract(A, B)


def test_contract_ambiguous_automatic_raises():
    """Test that ambiguous automatic contraction raises error."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    # Both indices have same itag and opposite directions - should be fine with 1:1
    A = Tensor.random([idx_out, idx_in], seed=1, itags=["x", "y"])
    B = Tensor.random([idx_out, idx_out.flip()], seed=2, itags=["y", "z"])
    
    # This should work fine (1 match per index)
    result = contract(A, B)
    assert result is not None


def test_contract_large_block_shapes():
    """Test contraction with large block shapes."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 4), Sector(1, 3), Sector(-1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 4), Sector(1, 3), Sector(-1, 3)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))

    A = Tensor.random([idx_a, idx_b], seed=901, itags=["a", "b"])
    B = Tensor.random([idx_b.dual(), idx_c, idx_d], seed=902, itags=["b", "c", "d"])

    # Automatic detection
    result = contract(A, B)
    assert result.itags[0] == "a"
    assert result.itags[1] == "c"
    assert result.itags[2] == "d"
    assert_charge_neutral(result)


# Tests for excl parameter

def test_contract_excl_exclude_from_A():
    """Test excl parameter excluding axes from tensor A only."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b_out, idx_c_out], seed=301, itags=["a", "b", "c"])
    B = Tensor.random([idx_b_in, idx_c_in, idx_d], seed=302, itags=["b", "c", "d"])
    
    # Exclude A's axis 2 ("c"), so only "b" should be contracted
    result = contract(A, B, excl=((2,), ()))
    
    # Result should have: a, c (from A), c, d (from B) - "c" appears twice
    assert list(result.itags) == ["a", "c", "c", "d"]
    assert_charge_neutral(result)
    
    # Verify equivalence with manual axes
    manual_result = contract(A, B, axes=(1, 0))
    assert result.itags == manual_result.itags
    for key in result.data:
        np.testing.assert_allclose(result.data[key], manual_result.data[key])


def test_contract_excl_exclude_from_B():
    """Test excl parameter excluding axes from tensor B only."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b_out, idx_c_out], seed=303, itags=["a", "b", "c"])
    B = Tensor.random([idx_b_in, idx_c_in, idx_d], seed=304, itags=["b", "c", "d"])
    
    # Exclude B's axis 1 ("c"), so only "b" should be contracted
    result = contract(A, B, excl=((), (1,)))
    
    # Result should have: a, c (from A), c, d (from B) - "c" appears twice
    assert list(result.itags) == ["a", "c", "c", "d"]
    assert_charge_neutral(result)
    
    # Verify equivalence with manual axes
    manual_result = contract(A, B, axes=(1, 0))
    assert result.itags == manual_result.itags
    for key in result.data:
        np.testing.assert_allclose(result.data[key], manual_result.data[key])


def test_contract_excl_exclude_from_both():
    """Test excl parameter excluding axes from both tensors."""
    group = U1Group()
    # Create tensors with 3 matching pairs: a, b, c
    idx_a_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    
    idx_a_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    
    A = Tensor.random([idx_a_out, idx_b_out, idx_c_out], seed=305, itags=["a", "b", "c"])
    B = Tensor.random([idx_a_in, idx_b_in, idx_c_in], seed=306, itags=["a", "b", "c"])
    
    # Exclude A's axis 0 ("a") and B's axis 2 ("c"), so only "b" will contract
    result = contract(A, B, excl=((0,), (2,)))
    
    # Result should have: a (from A), c (from A), a (from B), c (from B)
    # Order: ["a", "c", "a", "c"] - only b contracted
    assert list(result.itags) == ["a", "c", "a", "c"]
    assert_charge_neutral(result)
    
    # Verify equivalence with manual axes
    manual_result = contract(A, B, axes=(1, 1))
    assert result.itags == manual_result.itags


def test_contract_excl_empty_exclusion():
    """Test that empty exclusion is equivalent to automatic mode."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    
    A = Tensor.random([idx_a, idx_b_out], seed=307, itags=["a", "b"])
    B = Tensor.random([idx_b_in, idx_c], seed=308, itags=["b", "c"])
    
    # Empty exclusion should match automatic mode
    result_excl = contract(A, B, excl=((), ()))
    result_auto = contract(A, B)
    
    assert result_excl.itags == result_auto.itags
    for key in result_excl.data:
        np.testing.assert_allclose(result_excl.data[key], result_auto.data[key])


def test_contract_excl_single_contraction():
    """Test excl parameter leaving only one contraction pair."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(2, 1)))
    idx_c_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b_out, idx_c_out], seed=309, itags=["a", "b", "c"])
    B = Tensor.random([idx_b_in, idx_c_in, idx_d], seed=310, itags=["b", "c", "d"])
    
    # Exclude A's axis 2 and B's axis 1, leaving only b-b contraction
    result = contract(A, B, excl=((2,), (1,)))
    
    # Result should have: a, c (from A), c, d (from B)
    assert set(result.itags) == {"a", "c", "d"}
    # Note: "c" appears from A only since we excluded B's "c" axis
    
    # Verify equivalence with manual axes
    manual_result = contract(A, B, axes=(1, 0))
    assert sorted(result.itags) == sorted(manual_result.itags)


def test_contract_axes_single_pair_concise_syntax():
    """Test axes parameter with concise (int, int) syntax for single pair."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_b_left = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b_right = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2)))

    A = Tensor.random([idx_a, idx_b_left], seed=20, itags=["a", "b"])
    B = Tensor.random([idx_b_right, idx_c], seed=21, itags=["b", "c"])

    # Test concise syntax: axes=(1, 0)
    result_concise = contract(A, B, axes=(1, 0))
    assert list(result_concise.itags) == ["a", "c"]
    assert_charge_neutral(result_concise)
    
    # Test verbose syntax: axes=(1, 0)
    result_verbose = contract(A, B, axes=(1, 0))
    assert list(result_verbose.itags) == ["a", "c"]
    
    # Both should give same result
    assert result_concise.itags == result_verbose.itags
    for key in result_concise.data:
        np.testing.assert_allclose(result_concise.data[key], result_verbose.data[key])


def test_contract_axes_concise_with_permutation():
    """Test concise axes syntax with permutation."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1),))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx_a, idx_b_out], seed=1, itags=["a", "b"])
    B = Tensor.random([idx_b_in, idx_c], seed=2, itags=["b", "c"])
    
    result = contract(A, B, axes=(1, 0), perm=[1, 0])
    
    assert list(result.itags) == ["c", "a"]


def test_contract_axes_excl_mutually_exclusive():
    """Test that specifying both axes and excl raises an error."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1),))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx_a, idx_b_out], seed=311, itags=["a", "b"])
    B = Tensor.random([idx_b_in, idx_c], seed=312, itags=["b", "c"])
    
    # Should raise ValueError when both axes and excl are specified
    with pytest.raises(ValueError, match="Cannot specify both 'axes' and 'excl'"):
        contract(A, B, axes=(1, 0), excl=((0,), ()))


def test_contract_excl_no_valid_pairs():
    """Test that excluding all potential pairs raises an error."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1),))
    idx_c = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx_a, idx_b_out], seed=313, itags=["a", "b"])
    B = Tensor.random([idx_b_in, idx_c], seed=314, itags=["b", "c"])
    
    # Exclude the only matching pair
    with pytest.raises(ValueError, match="No valid contraction pairs found"):
        contract(A, B, excl=((1,), ()))


def test_contract_excl_with_permutation():
    """Test excl parameter with permutation."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx_c_out = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 1),))
    idx_c_in = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1),))
    
    A = Tensor.random([idx_a, idx_b_out, idx_c_out], seed=315, itags=["a", "b", "c"])
    B = Tensor.random([idx_b_in, idx_c_in, idx_d], seed=316, itags=["b", "c", "d"])
    
    # Exclude A's axis 1 ("b"), so only "c" gets contracted, then permute result
    result = contract(A, B, excl=((1,), ()), perm=[1, 0, 2, 3])
    
    # Result before perm: ["a", "b", "b", "d"], after perm (swap first two): ["b", "a", "b", "d"]
    assert list(result.itags) == ["b", "a", "b", "d"]
    assert_charge_neutral(result)


# Trace tests

def test_trace_automatic():
    """Test trace with automatic pairing and verify numeric correctness."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 1)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 1)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 1), Sector(2, 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector(0, 2), Sector(1, 1), Sector(2, 2)
    ))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["x", "x", "y", "y"])
    # Automatic mode: should trace both pairs
    traced = trace(tensor)
    assert len(traced.indices) == 0
    assert traced.is_scalar()
    
    # Verify by contracting with identity tensors (sequential)
    # For identical tags, automatic contraction works correctly
    id_x = identity(idx_a, itags=("x", "x"))
    contracted_x = contract(tensor, id_x)
    id_y = identity(contracted_x.indices[0], itags=("y", "y"))
    contracted_both = contract(contracted_x, id_y)
    
    # Automatic trace should match identity contraction
    np.testing.assert_allclose(traced.item(), contracted_both.item())


def test_trace_explicit_multi_pair():
    """Test explicit multi-pair trace and verify order independence."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 2), Sector(2, 1)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 2), Sector(2, 1)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 2), Sector(-1, 2), Sector(-2, 1)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector(0, 2), Sector(1, 2), Sector(-1, 2), Sector(-2, 1)
    ))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=601, itags=["a", "b", "c", "d"])
    
    # Multi-pair trace with explicit axes
    traced_multi = trace(tensor, axes=[(0, 1), (2, 3)])
    assert traced_multi.is_scalar()
    
    # Sequential trace in order: first (0,1), then (0,1) [indices shift after first trace]
    traced_seq1 = trace(trace(tensor, axes=(0, 1)), axes=(0, 1))
    assert traced_seq1.is_scalar()
    
    # Sequential trace in different order: first (2,3), then (0,1) [indices shift after first trace]
    traced_seq2 = trace(trace(tensor, axes=(2, 3)), axes=(0, 1))
    assert traced_seq2.is_scalar()
    
    # All three methods should give identical results
    np.testing.assert_allclose(traced_multi.item(), traced_seq1.item())
    np.testing.assert_allclose(traced_multi.item(), traced_seq2.item())


def test_trace_manual_single_pair():
    """Test trace with manually specified single pair."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 2), Sector(2, 1)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 2), Sector(2, 1)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 3), Sector(-1, 1), Sector(2, 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector(0, 2), Sector(1, 3), Sector(-1, 1), Sector(2, 2)
    ))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["a", "b", "c", "d"])
    # Manual mode: trace only axes (0, 1)
    traced = trace(tensor, axes=(0, 1))
    assert traced.indices == (idx_c, idx_d)

    manual = {}
    for (qa, qb, qc, qd), block in tensor.data.items():
        if qa == qb:
            diag = np.trace(block, axis1=0, axis2=1)
            key = (qc, qd)
            if key in manual:
                manual[key] += diag
            else:
                manual[key] = diag

    assert set(traced.data.keys()) == set(manual.keys())
    for key, expected in manual.items():
        np.testing.assert_allclose(traced.data[key], expected)


def test_trace_exclusion_by_index():
    """Test trace with exclusion by integer index."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["x", "x", "y", "y"])
    # Exclude first pair (0, 1), should trace only second pair (2, 3)
    traced = trace(tensor, excl=[0, 1])
    assert len(traced.indices) == 2
    assert traced.indices == (idx_a, idx_b)


def test_trace_manual_multiple_pairs():
    """Test trace with multiple manually specified pairs and identity verification."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 2), Sector(2, 1)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 2), Sector(2, 1)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 2), Sector(-1, 2), Sector(-2, 1)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector(0, 2), Sector(1, 2), Sector(-1, 2), Sector(-2, 1)
    ))

    # Use matching itags for identity verification
    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=601, itags=["x", "x", "y", "y"])
    traced = trace(tensor, axes=[(0, 1), (2, 3)])
    
    # Result should be scalar (all indices traced)
    assert len(traced.indices) == 0
    assert traced.is_scalar()
    
    # Verify with identity contraction (sequential)
    id_x = identity(idx_a, itags=("x", "x"))
    contracted_x = contract(tensor, id_x)
    id_y = identity(contracted_x.indices[0], itags=("y", "y"))
    contracted_all = contract(contracted_x, id_y)
    
    # Both methods should give the same result
    np.testing.assert_allclose(traced.item(), contracted_all.item())


def test_trace_exclusion_by_tag():
    """Test trace with exclusion by itag name and verify numeric correctness."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 1)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 1)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 1), Sector(2, 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector(0, 2), Sector(1, 1), Sector(2, 2)
    ))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["x", "x", "y", "y"])
    # Exclude "x" tags, should trace only "y" pair
    traced = trace(tensor, excl="x")
    assert len(traced.indices) == 2
    assert traced.indices == (idx_a, idx_b)
    
    # Verify numeric correctness: only y pair should be traced
    manual = {}
    for (qa, qb, qc, qd), block in tensor.data.items():
        if qc == qd:
            # Trace over y pair (axes 2, 3 -> axes 2, 3 in block)
            diag = np.trace(block, axis1=2, axis2=3)
            key = (qa, qb)
            if key in manual:
                manual[key] += diag
            else:
                manual[key] = diag
    
    assert set(traced.data.keys()) == set(manual.keys())
    for key, expected in manual.items():
        np.testing.assert_allclose(traced.data[key], expected)


def test_trace_exclusion_single_int():
    """Test trace with single integer exclusion."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["x", "x", "y", "y"])
    # Exclude axis 0, so pair (0,1) cannot form, only (2,3) should be traced
    traced = trace(tensor, excl=0)
    assert len(traced.indices) == 2
    assert traced.indices == (idx_a, idx_b)


def test_trace_exclusion_multiple():
    """Test trace with multiple exclusions."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx_e = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx_f = Index(Direction.IN, group, sectors=(Sector(0, 1),))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e, idx_f], 
                           seed=601, itags=["x", "x", "y", "y", "z", "z"])
    # Exclude pairs x and y, should trace only z pair
    traced = trace(tensor, excl=[0, 1, 2, 3])
    assert len(traced.indices) == 4
    assert traced.indices == (idx_a, idx_b, idx_c, idx_d)


def test_contract_trace_consistency_high_order():
    """Test consistency: direct 3-index contraction vs 2-index contraction + trace.
    
    Two 5-index tensors contracted on 3 indices can be computed in two ways:
    1. Direct 3-index contraction: contract all 3 pairs at once
    2. Sequential: contract 2 pairs first, then trace the remaining pair
    
    Both approaches should give identical results.
    """
    group = U1Group()
    
    # A: 5 indices (a, b, c, d, e) - will contract b, c, d with B
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_b_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_c_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_d_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(2, 2)))
    idx_e = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2)))
    
    # B: 5 indices (b, c, d, f, g) - will contract b, c, d with A
    idx_b_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_c_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_d_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(2, 2)))
    idx_f = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_g = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    
    A = Tensor.random([idx_a, idx_b_out, idx_c_out, idx_d_out, idx_e], 
                      seed=4001, itags=["a", "b", "c", "d", "e"])
    B = Tensor.random([idx_b_in, idx_c_in, idx_d_in, idx_f, idx_g], 
                      seed=4002, itags=["b", "c", "d", "f", "g"])
    
    # Method 1: Direct 3-index contraction
    # Contract b, c, d all at once using automatic detection
    direct_result = contract(A, B)
    
    assert set(direct_result.itags) == {"a", "e", "f", "g"}
    assert_charge_neutral(direct_result)
    
    # Method 2: Contract 2 indices first, then trace the third
    # First contract only b and c (using manual axes to avoid contracting d)
    # A indices: 0=a, 1=b, 2=c, 3=d, 4=e
    # B indices: 0=b, 1=c, 2=d, 3=f, 4=g
    partial_result = contract(A, B, axes=([1, 2], [0, 1]))  # Contract b and c only
    
    # After contracting b and c, we have:
    # - From A: a, d_out, e (d_out not contracted)
    # - From B: d_in, f, g (d_in not contracted)
    # Result should have: a, d_out, e, d_in, f, g
    # where d_out and d_in have matching tags "d" but weren't contracted
    
    # Now trace over the remaining d pair using automatic detection
    traced_result = trace(partial_result)
    
    assert set(traced_result.itags) == {"a", "e", "f", "g"}
    assert_charge_neutral(traced_result)
    
    # Verify both methods give identical results
    # Compare block keys
    assert set(direct_result.data.keys()) == set(traced_result.data.keys()), \
        f"Block keys mismatch: direct has {set(direct_result.data.keys())}, traced has {set(traced_result.data.keys())}"
    
    # Compare block values
    for key in direct_result.data.keys():
        np.testing.assert_allclose(
            direct_result.data[key], 
            traced_result.data[key], 
            rtol=1e-10, 
            atol=1e-12,
            err_msg=f"Block {key} values differ between direct and traced methods"
        )
    
    # Also verify norms match
    assert abs(direct_result.norm() - traced_result.norm()) < 1e-10


def test_trace_raises_with_both_axes_and_excl():
    """Test that trace raises error when both axes and excl are specified."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "b"])
    
    with pytest.raises(ValueError, match="Cannot specify both"):
        trace(tensor, axes=(0, 1), excl=0)


# ProductGroup integration tests for contraction

def test_contract_product_group():
    """Test contracting two tensors with ProductGroup."""
    group = ProductGroup([U1Group(), U1Group()])
    
    # A: OUT, OUT with charges
    left_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2),
        Sector((1, 0), 1),
    ))
    right_a = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1),
        Sector((0, 1), 2),
    ))
    
    # B: IN, OUT with charges (contract first index with A's second)
    left_b = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 1),
        Sector((0, 1), 2),
    ))
    right_b = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 3),
        Sector((1, 0), 1),
    ))
    
    A = Tensor.random([left_a, right_a], seed=42, itags=["a", "mid"])
    B = Tensor.random([left_b, right_b], seed=43, itags=["mid", "b"])
    
    # Contract automatically on matching "mid" tag
    C = contract(A, B)
    
    assert len(C.indices) == 2
    assert C.itags == ("a", "b")
    assert_charge_neutral(C)


def test_contract_product_group_manual_pairs():
    """Test manual contraction with ProductGroup."""
    group = ProductGroup([U1Group(), U1Group()])
    
    left = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2), Sector((1, -1), 1)))
    right = Index(Direction.IN, group, sectors=(Sector((0, 0), 2), Sector((1, -1), 1)))
    
    A = Tensor.random([left, left.dual()], seed=10, itags=["x", "y"])
    B = Tensor.random([right, right.dual()], seed=11, itags=["x", "z"])
    
    # Contract using manual axes
    C = contract(A, B, axes=(0, 0))
    
    assert len(C.indices) == 2  # y and z remain
    assert C.itags == ("y", "z")
    assert_charge_neutral(C)


def test_trace_product_group():
    """Test trace operation with ProductGroup and verify numeric correctness."""
    group = ProductGroup([U1Group(), Z2Group()])
    
    left = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 3),
        Sector((1, 1), 2),
        Sector((-1, 1), 1),
        Sector((2, 0), 2),
    ))
    right = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 3),
        Sector((1, 1), 2),
        Sector((-1, 1), 1),
        Sector((2, 0), 2),
    ))
    
    T = Tensor.random([left, right], seed=99, itags=["x", "x"])
    
    # Trace over both indices (automatic mode)
    result = trace(T)
    
    assert result.is_scalar()
    assert len(result.indices) == 0
    assert_charge_neutral(result)
    
    # Verify numeric correctness
    manual_scalar = 0.0
    for (ql, qr), block in T.data.items():
        if ql == qr:
            manual_scalar += np.trace(block)
    
    np.testing.assert_allclose(result.item(), manual_scalar)


# Scalar result tests

def test_trace_produces_scalar():
    """Test that tracing all indices produces a scalar (0D tensor) with numeric verification."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 1), Sector(2, 2)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 1), Sector(2, 2)
    ))
    
    tensor = Tensor.random([idx_a, idx_b], seed=30, itags=["x", "x"])
    
    # Trace all indices (automatic mode)
    scalar = trace(tensor)
    
    assert scalar.is_scalar()
    assert len(scalar.indices) == 0
    assert len(scalar.itags) == 0
    assert () in scalar.data
    
    # Verify it's a valid scalar value
    value = scalar.item()
    assert isinstance(value, (int, float, complex))
    
    # Verify numeric correctness
    manual_scalar = 0.0
    for (qa, qb), block in tensor.data.items():
        if qa == qb:
            manual_scalar += np.trace(block)
    
    np.testing.assert_allclose(value, manual_scalar)


def test_contract_produces_scalar():
    """Test that full contraction produces a scalar."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    A = Tensor.random([idx_out, idx_in], seed=1, itags=["a", "b"])
    B = Tensor.random([idx_in.flip(), idx_out.flip()], seed=2, itags=["b", "a"])
    
    # Contract all indices automatically (matching itags with opposite directions)
    scalar = contract(A, B)
    
    assert scalar.is_scalar()
    assert len(scalar.indices) == 0
    assert len(scalar.itags) == 0
    assert () in scalar.data


def test_trace_multiple_pairs_produces_scalar():
    """Test that tracing multiple pairs can produce a scalar with numeric verification."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 1)
    ))
    idx_b = Index(Direction.IN, group, sectors=(
        Sector(0, 3), Sector(1, 2), Sector(-1, 1)
    ))
    idx_c = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 1), Sector(2, 2)
    ))
    idx_d = Index(Direction.IN, group, sectors=(
        Sector(0, 2), Sector(1, 1), Sector(2, 2)
    ))
    
    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=601, itags=["x", "x", "y", "y"])
    
    # Trace all pairs (automatic mode)
    scalar = trace(tensor)
    
    assert scalar.is_scalar()
    assert len(scalar.indices) == 0
    
    # Verify by contracting with identity tensors (sequential)
    # For identical tags, automatic contraction works correctly
    id_x = identity(idx_a, itags=("x", "x"))
    contracted_x = contract(tensor, id_x)
    id_y = identity(contracted_x.indices[0], itags=("y", "y"))
    contracted_both = contract(contracted_x, id_y)
    
    # Also verify with multi-pair trace
    traced_multi = trace(tensor, axes=[(0, 1), (2, 3)])
    
    # All three methods should match: automatic trace, identity contraction, and multi-pair trace
    np.testing.assert_allclose(scalar.item(), traced_multi.item())
    np.testing.assert_allclose(scalar.item(), contracted_both.item())


def test_trace_ambiguous_raises():
    """Test that ambiguous automatic pairing raises an error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    # Create tensor with 3 indices with same tag "x": 2 OUT, 1 IN
    # This is ambiguous: which OUT should pair with the IN?
    tensor = Tensor.random([idx, idx, idx.flip()], seed=42, itags=["x", "x", "x"])
    
    with pytest.raises(ValueError, match="Ambiguous automatic trace"):
        trace(tensor)


def test_scalar_result_operations():
    """Test operations on scalar results from contractions."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx_out, idx_in], seed=10, itags=["a", "b"])
    B = Tensor.random([idx_in.flip(), idx_out.flip()], seed=20, itags=["b", "a"])
    
    # Get two scalars from contractions
    s1 = contract(A, B)
    s2 = contract(B, A)
    
    # Operations on scalar results
    s_sum = s1 + s2
    assert s_sum.is_scalar()
    
    s_diff = s1 - s2
    assert s_diff.is_scalar()
    
    s_scaled = s1 * 2.0
    assert s_scaled.is_scalar()


def test_trace_manual_pair_syntax():
    """Test trace with single pair using tuple syntax."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx_a, idx_b], seed=31, itags=["a", "b"])
    
    # Trace using single pair tuple syntax
    scalar = trace(tensor, axes=(0, 1))
    
    assert scalar.is_scalar()
    assert len(scalar.indices) == 0

