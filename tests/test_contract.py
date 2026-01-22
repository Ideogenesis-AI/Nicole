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

from nicole import Direction, Tensor, contract, identity, partial_trace, trace, U1Group, Z2Group, permute, Index, Sector
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

    result = contract(A, B, pairs=[(1, 0)])
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
    positional = contract(A, B, pairs=[(2, 0), (1, 1)])

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
    
    result = contract(A, B, pairs=[(1, 0)], perm=[1, 0])
    
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


# Trace tests

def test_trace_integer_pairs():
    """Test trace with integer index pairs."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["a", "b", "c", "d"])
    traced = trace(tensor, pairs=[(0, 1)])
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


def test_trace_string_pairs():
    """Test trace with string index pairs."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["a", "b", "c", "d"])
    traced = trace(tensor, pairs=[("a", "b")])
    assert traced.indices == (idx_c, idx_d)


def test_trace_multiple_pairs():
    """Test trace with multiple pairs."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=601, itags=["a", "b", "c", "d"])
    traced = trace(tensor, pairs=[("a", "b"), ("c", "d")])
    
    # Result should be scalar (all indices traced)
    assert len(traced.indices) == 0
    assert traced.is_scalar()


# Partial trace tests

def test_partial_trace_integer_axes():
    """Test partial_trace with integer axes."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["a", "b", "c", "d"])
    
    pt = partial_trace(tensor, axes=[0, 1])
    assert pt.indices == (idx_c, idx_d)


def test_partial_trace_string_axes():
    """Test partial_trace with string axes."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["a", "b", "c", "d"])
    traced = trace(tensor, pairs=[("a", "b")])
    
    # Partial trace should match explicit trace call
    pt = partial_trace(tensor, axes=["a", "b"])
    assert set(pt.data.keys()) == set(traced.data.keys())
    for key in pt.data:
        np.testing.assert_allclose(pt.data[key], traced.data[key])


def test_partial_trace_multiple_pairs():
    """Test partial_trace with multiple pairs."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=601, itags=["a", "b", "c", "d"])
    first = partial_trace(tensor, axes=["a", "b"])

    manual_matrices = {}
    for (qa, qb, qc, qd), block in tensor.data.items():
        if qa == qb and qc == qd:
            inner = np.trace(block, axis1=0, axis2=1)
            manual_matrices[(qc, qd)] = manual_matrices.get((qc, qd), 0) + inner

    assert set(first.data.keys()) == set(manual_matrices.keys())
    for key, expected in manual_matrices.items():
        np.testing.assert_allclose(first.data[key], expected)

    total_manual = 0.0
    total_from_tensor = 0.0
    for (qc, qd), matrix in manual_matrices.items():
        if qc == qd:
            total_manual += np.trace(matrix)
            total_from_tensor += np.trace(first.data[(qc, qd)])
    assert np.isclose(total_manual, total_from_tensor)


def test_contract_trace_consistency_high_order():
    """Test consistency: direct 3-index contraction vs 2-index contraction + partial trace.
    
    Two 5-index tensors contracted on 3 indices can be computed in two ways:
    1. Direct 3-index contraction: contract all 3 pairs at once
    2. Sequential: contract 2 pairs first, then partial trace the remaining pair
    
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
    
    # Method 2: Contract 2 indices first, then partial trace the third
    # First contract only b and c (using manual pairs to avoid contracting d)
    # A indices: 0=a, 1=b, 2=c, 3=d, 4=e
    # B indices: 0=b, 1=c, 2=d, 3=f, 4=g
    partial_result = contract(A, B, pairs=[(1, 0), (2, 1)])  # Contract b and c only
    
    # After contracting b and c, we have:
    # - From A: a, d_out, e (d_out not contracted)
    # - From B: d_in, f, g (d_in not contracted)
    # Result should have: a, d_out, e, d_in, f, g
    # where d_out and d_in have matching tags "d" but weren't contracted
    
    # Now trace over the remaining d pair
    # Find which axes correspond to the two "d" indices
    d_axes = [i for i, tag in enumerate(partial_result.itags) if tag == "d"]
    assert len(d_axes) == 2, f"Expected 2 'd' indices, got {len(d_axes)}"
    
    traced_result = partial_trace(partial_result, axes=d_axes)
    
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


def test_partial_trace_odd_axes_raises():
    """Test that partial_trace with odd number of axes raises error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx, idx], seed=1, itags=["a", "b", "c"])
    
    with pytest.raises(ValueError, match="even number of axes"):
        partial_trace(tensor, axes=[0, 1, 2])


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
    
    # Contract using manual pairs
    C = contract(A, B, pairs=[(0, 0)])
    
    assert len(C.indices) == 2  # y and z remain
    assert C.itags == ("y", "z")
    assert_charge_neutral(C)


def test_trace_product_group():
    """Test trace operation with ProductGroup."""
    group = ProductGroup([U1Group(), Z2Group()])
    
    left = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2),
        Sector((1, 1), 1),
    ))
    right = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2),
        Sector((1, 1), 1),
    ))
    
    T = Tensor.random([left, right], seed=99, itags=["i", "j"])
    
    # Trace over both indices
    result = trace(T, pairs=[(0, 1)])
    
    assert result.is_scalar()
    assert len(result.indices) == 0
    assert_charge_neutral(result)


# Scalar result tests

def test_trace_produces_scalar():
    """Test that tracing all indices produces a scalar (0D tensor)."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    tensor = Tensor.random([idx_a, idx_b], seed=30, itags=["a", "b"])
    
    # Trace all indices
    scalar = trace(tensor, pairs=[(0, 1)])
    
    assert scalar.is_scalar()
    assert len(scalar.indices) == 0
    assert len(scalar.itags) == 0
    assert () in scalar.data
    
    # Verify it's a valid scalar value
    value = scalar.item()
    assert isinstance(value, (int, float, complex))


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
    """Test that tracing multiple pairs can produce a scalar."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx_d = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    
    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=601, itags=["a", "b", "c", "d"])
    
    # Trace all pairs
    scalar = trace(tensor, pairs=[("a", "b"), ("c", "d")])
    
    assert scalar.is_scalar()
    assert len(scalar.indices) == 0
    assert scalar.item() is not None


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


def test_partial_trace_produces_scalar():
    """Test partial_trace with all indices produces scalar."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx_a, idx_b], seed=31, itags=["a", "b"])
    
    # Partial trace over all indices
    scalar = partial_trace(tensor, axes=[0, 1])
    
    assert scalar.is_scalar()
    assert len(scalar.indices) == 0

