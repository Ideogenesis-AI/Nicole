# Copyright (C) 2025 Changkai Zhang.
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
from .utils import make_u1_index, assert_charge_neutral


# Basic contraction tests

def test_contract_two_tensors_manual_pairs():
    """Test two-tensor contraction with manual pairs matches block-wise computation."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    idx_b_left = make_u1_index(Direction.IN, [(0, 1), (1, 1)], group)
    idx_b_right = make_u1_index(Direction.OUT, [(0, 1), (1, 1)], group)
    idx_c = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)

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
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    idx_b_out = make_u1_index(Direction.OUT, [(0, 1), (2, 1)], group)
    idx_c_out = make_u1_index(Direction.OUT, [(0, 2), (-1, 1)], group)

    idx_c_in = make_u1_index(Direction.IN, [(0, 2), (-1, 1)], group)
    idx_b_in = make_u1_index(Direction.IN, [(0, 1), (2, 1)], group)
    idx_d = make_u1_index(Direction.IN, [(0, 1), (1, 2)], group)

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


def test_contract_named_vs_positional():
    """Test that named axis automatic detection matches position-based pairs."""
    group = U1Group()
    idx_left = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    idx_mid_out = make_u1_index(Direction.OUT, [(0, 1), (2, 1)], group)
    idx_right_in = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)

    idx_mid_in = make_u1_index(Direction.IN, [(0, 1), (2, 1)], group)
    idx_end = make_u1_index(Direction.IN, [(0, 2)], group)

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
    idx_a = make_u1_index(Direction.OUT, [(0, 2)], group)
    idx_b_out = make_u1_index(Direction.OUT, [(0, 1)], group)
    idx_b_in = make_u1_index(Direction.IN, [(0, 1)], group)
    idx_c = make_u1_index(Direction.IN, [(0, 2)], group)
    
    A = Tensor.random([idx_a, idx_b_out], seed=1, itags=["a", "b"])
    B = Tensor.random([idx_b_in, idx_c], seed=2, itags=["b", "c"])
    
    result = contract(A, B, pairs=[(1, 0)], perm=[1, 0])
    
    assert list(result.itags) == ["c", "a"]


# Associativity and composition tests

def test_contract_three_tensor_associativity():
    """Test that tensor contraction is associative."""
    group = U1Group()
    idx_x = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    idx_y_out = make_u1_index(Direction.OUT, [(0, 1), (2, 1)], group)
    idx_y_in = make_u1_index(Direction.IN, [(0, 1), (2, 1)], group)
    idx_z_out = make_u1_index(Direction.OUT, [(0, 1), (-1, 1)], group)
    idx_z_in = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)
    idx_w = make_u1_index(Direction.IN, [(0, 2)], group)

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
    idx_left = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    idx_mid_in = make_u1_index(Direction.IN, [(0, 1), (2, 1)], group)
    idx_mid_out = make_u1_index(Direction.OUT, [(0, 1), (2, 1)], group)
    idx_right = make_u1_index(Direction.IN, [(0, 2), (-1, 1)], group)

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
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    idx_b = make_u1_index(Direction.OUT, [(0, 1), (2, 1)], group)
    idx_c = make_u1_index(Direction.IN, [(0, 1), (2, 1)], group)
    idx_d = make_u1_index(Direction.IN, [(0, 2)], group)

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
    idx_left = make_u1_index(Direction.OUT, [(0, 2)], group)
    idx_mid_out = make_u1_index(Direction.OUT, [(0, 3)], group)
    idx_mid_in_bad = make_u1_index(Direction.IN, [(0, 4)], group)
    idx_right = make_u1_index(Direction.IN, [(0, 2)], group)

    A = Tensor.random([idx_left, idx_mid_out], seed=501, itags=["L", "M"])
    B_bad = Tensor.random([idx_mid_in_bad, idx_right], seed=503, itags=["M", "R"])

    # Automatic detection for bad dimensions (will have empty result)
    result_bad = contract(A, B_bad)
    assert result_bad.data == {}


def test_contract_empty_result_no_matching_charge():
    """Test contraction with no matching charges gives empty result."""
    group = U1Group()
    idx_left = make_u1_index(Direction.OUT, [(0, 2)], group)
    idx_mid_out = make_u1_index(Direction.OUT, [(1, 1)], group)
    idx_mid_in = make_u1_index(Direction.IN, [(0, 1)], group)
    idx_right = make_u1_index(Direction.IN, [(0, 2)], group)

    A = Tensor.random([idx_left, idx_mid_out], seed=701, itags=["L", "M"])
    B = Tensor.random([idx_mid_in, idx_right], seed=702, itags=["M", "R"])

    # Automatic detection
    result = contract(A, B)
    assert result.data == {}


def test_contract_no_pairs_raises():
    """Test that contract without valid pairs raises error."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "c"])
    B = Tensor.random([idx, idx.flip()], seed=2, itags=["b", "d"])
    
    with pytest.raises(ValueError, match="No valid contraction pairs"):
        contract(A, B)


def test_contract_ambiguous_automatic_raises():
    """Test that ambiguous automatic contraction raises error."""
    group = U1Group()
    idx_out = make_u1_index(Direction.OUT, [(0, 2)], group)
    idx_in = make_u1_index(Direction.IN, [(0, 2)], group)
    
    # Both indices have same itag and opposite directions - should be fine with 1:1
    A = Tensor.random([idx_out, idx_in], seed=1, itags=["x", "y"])
    B = Tensor.random([idx_out, idx_out.flip()], seed=2, itags=["y", "z"])
    
    # This should work fine (1 match per index)
    result = contract(A, B)
    assert result is not None


def test_contract_large_block_shapes():
    """Test contraction with large block shapes."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 3), (1, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 3), (1, 2)], group)
    idx_c = make_u1_index(Direction.OUT, [(0, 2), (-1, 2)], group)
    idx_d = make_u1_index(Direction.IN, [(0, 2), (-1, 2)], group)

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
    idx_a = make_u1_index(Direction.OUT, [(0, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 2)], group)
    idx_c = make_u1_index(Direction.OUT, [(0, 1), (1, 1)], group)
    idx_d = make_u1_index(Direction.IN, [(0, 1)], group)

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
    idx_a = make_u1_index(Direction.OUT, [(0, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 2)], group)
    idx_c = make_u1_index(Direction.OUT, [(0, 1), (1, 1)], group)
    idx_d = make_u1_index(Direction.IN, [(0, 1)], group)

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["a", "b", "c", "d"])
    traced = trace(tensor, pairs=[("a", "b")])
    assert traced.indices == (idx_c, idx_d)


def test_trace_multiple_pairs():
    """Test trace with multiple pairs."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 2), (1, 1)], group)
    idx_c = make_u1_index(Direction.OUT, [(0, 1), (-1, 1)], group)
    idx_d = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)
    idx_e = make_u1_index(Direction.OUT, [(0, 1)], group)
    idx_f = make_u1_index(Direction.IN, [(0, 1)], group)

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d, idx_e, idx_f], seed=601, itags=["a", "b", "c", "d", "e", "f"])
    traced = trace(tensor, pairs=[("a", "b"), ("c", "d")])
    
    # Result should have 2 indices left (e and f)
    assert len(traced.indices) == 2
    assert traced.indices == (idx_e, idx_f)


# Partial trace tests

def test_partial_trace_integer_axes():
    """Test partial_trace with integer axes."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 2)], group)
    idx_c = make_u1_index(Direction.OUT, [(0, 1), (1, 1)], group)
    idx_d = make_u1_index(Direction.IN, [(0, 1)], group)

    tensor = Tensor.random([idx_a, idx_b, idx_c, idx_d], seed=30, itags=["a", "b", "c", "d"])
    
    pt = partial_trace(tensor, axes=[0, 1])
    assert pt.indices == (idx_c, idx_d)


def test_partial_trace_string_axes():
    """Test partial_trace with string axes."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 2)], group)
    idx_c = make_u1_index(Direction.OUT, [(0, 1), (1, 1)], group)
    idx_d = make_u1_index(Direction.IN, [(0, 1)], group)

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
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 2), (1, 1)], group)
    idx_c = make_u1_index(Direction.OUT, [(0, 1), (-1, 1)], group)
    idx_d = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)

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


def test_partial_trace_odd_axes_raises():
    """Test that partial_trace with odd number of axes raises error."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
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
    extra = Index(Direction.OUT, group, sectors=(Sector((0, 0), 1),))
    
    A = Tensor.random([left, extra], seed=10, itags=["x", "a"])
    B = Tensor.random([right, extra.flip()], seed=11, itags=["x", "b"])  # Same itag for contraction
    
    # Contract using manual pairs
    C = contract(A, B, pairs=[(0, 0)])
    
    assert len(C.indices) == 2  # extra and extra.flip() remain
    assert C.itags == ("a", "b")
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
    extra = Index(Direction.OUT, group, sectors=(Sector((0, 0), 1),))
    extra2 = Index(Direction.IN, group, sectors=(Sector((0, 0), 1),))
    
    T = Tensor.random([left, right, extra, extra2], seed=99, itags=["i", "j", "k", "l"])
    
    # Trace over first two indices
    result = trace(T, pairs=[(0, 1)])
    
    assert len(result.indices) == 2  # extra and extra2 remain
    assert result.itags == ("k", "l")
    assert_charge_neutral(result)

